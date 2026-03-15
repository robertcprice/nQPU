//! Bosonic quantum error correction codes in truncated Fock space.
//!
//! This module implements three families of bosonic codes that encode a logical
//! qubit into a single harmonic oscillator mode:
//!
//! - **Cat codes**: Superpositions of coherent states with exponentially
//!   suppressed bit-flip errors and a tunable noise bias.
//! - **GKP codes** (Gottesman-Kitaev-Preskill): Grid states in phase space
//!   that correct small displacement errors in both position and momentum.
//! - **Binomial codes**: Photon-number-weighted superpositions that protect
//!   against photon loss up to a chosen order.
//!
//! All representations use a truncated Fock basis |0>, |1>, ..., |n_max> so
//! that operators become finite-dimensional matrices acting on amplitude vectors.

use num_complex::Complex64 as C64;
use std::f64::consts::PI;

// ============================================================
// CONSTANTS
// ============================================================

/// Numerical tolerance for normalization and comparisons.
const EPSILON: f64 = 1e-12;

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Compute n! as f64. For n > 170 the result overflows; we clamp to f64::MAX.
#[inline]
fn factorial(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let mut result = 1.0_f64;
    for i in 2..=n {
        result *= i as f64;
        if result.is_infinite() {
            return f64::MAX;
        }
    }
    result
}

/// Compute sqrt(n!) without overflow for moderate n by accumulating in log space.
#[inline]
fn sqrt_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let mut log_val = 0.0_f64;
    for i in 2..=n {
        log_val += (i as f64).ln();
    }
    (log_val / 2.0).exp()
}

/// Binomial coefficient C(n, k) as f64.
#[inline]
fn binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    // Use the smaller of k and n-k for efficiency.
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Generalized Laguerre polynomial L_n^alpha(x) evaluated via the recurrence.
#[inline]
fn laguerre(n: usize, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut l_prev = 1.0; // L_0
    let mut l_curr = 1.0 + alpha - x; // L_1
    for k in 2..=n {
        let kf = k as f64;
        let l_next = ((2.0 * kf - 1.0 + alpha - x) * l_curr - (kf - 1.0 + alpha) * l_prev) / kf;
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

/// Hermite polynomial H_n(x) evaluated via the recurrence.
#[inline]
fn hermite(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut h_prev = 1.0;
    let mut h_curr = 2.0 * x;
    for k in 2..=n {
        let h_next = 2.0 * x * h_curr - 2.0 * (k as f64 - 1.0) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

// ============================================================
// FOCK STATE
// ============================================================

/// A quantum state in the truncated Fock basis |0>, |1>, ..., |n_max>.
///
/// The state is represented by a vector of complex amplitudes where `amplitudes[n]`
/// is the coefficient of |n> in the superposition.
#[derive(Clone, Debug)]
pub struct FockState {
    /// Complex amplitudes in the Fock basis.
    pub amplitudes: Vec<C64>,
    /// Truncation level: the maximum photon number included.
    pub n_max: usize,
}

impl FockState {
    /// Create the vacuum state |0> in a Hilbert space truncated at `n_max` photons.
    pub fn new(n_max: usize) -> Self {
        let mut amplitudes = vec![C64::new(0.0, 0.0); n_max + 1];
        amplitudes[0] = C64::new(1.0, 0.0);
        Self { amplitudes, n_max }
    }

    /// Create the Fock state |n> in a Hilbert space truncated at `n_max` photons.
    ///
    /// # Panics
    ///
    /// Panics if `n > n_max`.
    pub fn from_number(n: usize, n_max: usize) -> Self {
        assert!(
            n <= n_max,
            "Fock state |{}> exceeds truncation n_max={}",
            n,
            n_max
        );
        let mut amplitudes = vec![C64::new(0.0, 0.0); n_max + 1];
        amplitudes[n] = C64::new(1.0, 0.0);
        Self { amplitudes, n_max }
    }

    /// Create a state from a pre-existing amplitude vector.
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty.
    pub fn from_amplitudes(amplitudes: Vec<C64>) -> Self {
        assert!(!amplitudes.is_empty(), "Amplitude vector must not be empty");
        let n_max = amplitudes.len() - 1;
        Self { amplitudes, n_max }
    }

    /// Compute the expectation value of the photon number operator: <n> = sum_n n |alpha_n|^2.
    pub fn photon_number_expectation(&self) -> f64 {
        self.amplitudes
            .iter()
            .enumerate()
            .map(|(n, a)| n as f64 * a.norm_sqr())
            .sum()
    }

    /// Compute the variance of the photon number: Var(n) = <n^2> - <n>^2.
    pub fn photon_number_variance(&self) -> f64 {
        let mean = self.photon_number_expectation();
        let mean_sq: f64 = self
            .amplitudes
            .iter()
            .enumerate()
            .map(|(n, a)| (n as f64).powi(2) * a.norm_sqr())
            .sum();
        mean_sq - mean * mean
    }

    /// Compute the L2 norm of the state vector.
    pub fn norm(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|a| a.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize the state vector in place.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > EPSILON {
            let inv = 1.0 / n;
            for a in &mut self.amplitudes {
                *a *= inv;
            }
        }
    }

    /// Compute the fidelity |<self|other>|^2 between two states.
    ///
    /// Both states are treated as pure states. The result is in [0, 1].
    pub fn fidelity(&self, other: &FockState) -> f64 {
        let len = self.amplitudes.len().min(other.amplitudes.len());
        let mut overlap = C64::new(0.0, 0.0);
        for i in 0..len {
            overlap += self.amplitudes[i].conj() * other.amplitudes[i];
        }
        overlap.norm_sqr()
    }

    /// Return the photon number probability distribution P(n) = |alpha_n|^2.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Return the inner product <self|other>.
    pub fn inner_product(&self, other: &FockState) -> C64 {
        let len = self.amplitudes.len().min(other.amplitudes.len());
        let mut overlap = C64::new(0.0, 0.0);
        for i in 0..len {
            overlap += self.amplitudes[i].conj() * other.amplitudes[i];
        }
        overlap
    }

    /// Add another state (scaled by a complex coefficient) to this state.
    pub fn add_scaled(&mut self, other: &FockState, coeff: C64) {
        let len = self.amplitudes.len().min(other.amplitudes.len());
        for i in 0..len {
            self.amplitudes[i] += coeff * other.amplitudes[i];
        }
    }

    /// Return the number of Fock basis elements (n_max + 1).
    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }
}

// ============================================================
// COHERENT STATE CONSTRUCTION
// ============================================================

/// Construct a coherent state |alpha> in the truncated Fock basis.
///
/// |alpha> = exp(-|alpha|^2 / 2) * sum_{n=0}^{n_max} alpha^n / sqrt(n!) |n>
pub fn coherent_state(alpha: C64, n_max: usize) -> FockState {
    let norm_factor = (-alpha.norm_sqr() / 2.0).exp();
    let mut amplitudes = vec![C64::new(0.0, 0.0); n_max + 1];

    // Build iteratively: alpha^n / sqrt(n!) = (alpha / sqrt(n)) * alpha^{n-1}/sqrt((n-1)!)
    amplitudes[0] = C64::new(norm_factor, 0.0);
    for n in 1..=n_max {
        amplitudes[n] = amplitudes[n - 1] * alpha / (n as f64).sqrt();
    }

    FockState { amplitudes, n_max }
}

// ============================================================
// BOSONIC OPERATORS
// ============================================================

/// Fundamental bosonic operators acting on Fock states.
///
/// All operators return a new `FockState` rather than modifying in place.
pub struct BosonicOperator;

impl BosonicOperator {
    /// Creation (raising) operator: a^dagger |n> = sqrt(n+1) |n+1>.
    ///
    /// Components at n_max are truncated (lost to the truncation boundary).
    pub fn creation(state: &FockState) -> FockState {
        let n_max = state.n_max;
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];
        // a^dagger maps |n> -> sqrt(n+1)|n+1>, so result[n+1] = sqrt(n+1)*state[n]
        for n in 0..n_max {
            result[n + 1] = state.amplitudes[n] * ((n + 1) as f64).sqrt();
        }
        // |n_max> is mapped to |n_max+1> which is beyond truncation, so it is lost.
        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Annihilation (lowering) operator: a |n> = sqrt(n) |n-1>.
    ///
    /// By definition a|0> = 0.
    pub fn annihilation(state: &FockState) -> FockState {
        let n_max = state.n_max;
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];
        // a maps |n> -> sqrt(n)|n-1>, so result[n-1] = sqrt(n)*state[n]
        for n in 1..=n_max {
            result[n - 1] = state.amplitudes[n] * (n as f64).sqrt();
        }
        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Number operator: n_hat |n> = n |n>.
    pub fn number_operator(state: &FockState) -> FockState {
        let n_max = state.n_max;
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];
        for n in 0..=n_max {
            result[n] = state.amplitudes[n] * (n as f64);
        }
        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Displacement operator D(alpha) = exp(alpha * a^dagger - alpha^* * a).
    ///
    /// Implemented by computing the matrix elements directly:
    /// <m|D(alpha)|n> = sqrt(n!/m!) * alpha^{m-n} * e^{-|alpha|^2/2} * L_n^{m-n}(|alpha|^2)
    /// for m >= n, and using D_{mn} = (-1)^{m+n} D_{nm}^* for m < n.
    pub fn displacement(state: &FockState, alpha: C64) -> FockState {
        let n_max = state.n_max;
        let dim = n_max + 1;
        let alpha_sq = alpha.norm_sqr();
        let exp_factor = (-alpha_sq / 2.0).exp();

        let mut result = vec![C64::new(0.0, 0.0); dim];

        for m in 0..dim {
            let mut sum = C64::new(0.0, 0.0);
            for n in 0..dim {
                let d_mn = displacement_matrix_element(m, n, alpha, alpha_sq, exp_factor);
                sum += d_mn * state.amplitudes[n];
            }
            result[m] = sum;
        }

        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Squeezing operator S(xi) where xi = r * e^{i*phi}.
    ///
    /// S(xi) = exp((xi^* a^2 - xi (a^dagger)^2) / 2).
    ///
    /// Implemented via the matrix elements in the Fock basis using the
    /// analytic formula involving Hermite polynomials.
    pub fn squeeze(state: &FockState, r: f64, phi: f64) -> FockState {
        if r.abs() < EPSILON {
            return state.clone();
        }

        let n_max = state.n_max;
        let dim = n_max + 1;

        let mu = r.cosh();
        let nu_mag = r.sinh();
        let phase = C64::new(phi.cos(), phi.sin());

        // Build the squeezing matrix using the recursion relation.
        // S_{mn} is computed from the Bogoliubov decomposition.
        let mut s_matrix = vec![vec![C64::new(0.0, 0.0); dim]; dim];
        compute_squeeze_matrix(&mut s_matrix, dim, mu, nu_mag, phase);

        let mut result = vec![C64::new(0.0, 0.0); dim];
        for m in 0..dim {
            let mut sum = C64::new(0.0, 0.0);
            for n in 0..dim {
                sum += s_matrix[m][n] * state.amplitudes[n];
            }
            result[m] = sum;
        }

        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Phase-space rotation R(theta) = exp(-i * theta * n_hat).
    ///
    /// Each Fock state picks up a phase: R(theta)|n> = e^{-i*theta*n}|n>.
    pub fn rotation(state: &FockState, theta: f64) -> FockState {
        let n_max = state.n_max;
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];
        for n in 0..=n_max {
            let angle = -(theta * n as f64);
            let phase = C64::new(angle.cos(), angle.sin());
            result[n] = phase * state.amplitudes[n];
        }
        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Kerr nonlinear operator exp(-i * chi * n_hat^2).
    ///
    /// Each Fock state picks up a phase proportional to n^2.
    pub fn kerr(state: &FockState, chi: f64) -> FockState {
        let n_max = state.n_max;
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];
        for n in 0..=n_max {
            let angle = -chi * (n as f64).powi(2);
            let phase = C64::new(angle.cos(), angle.sin());
            result[n] = phase * state.amplitudes[n];
        }
        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Parity operator: P|n> = (-1)^n |n>.
    pub fn parity_operator(state: &FockState) -> FockState {
        let n_max = state.n_max;
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];
        for n in 0..=n_max {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            result[n] = state.amplitudes[n] * sign;
        }
        FockState {
            amplitudes: result,
            n_max,
        }
    }

    /// Beam-splitter operator acting on two modes.
    ///
    /// B(theta) = exp(theta * (a1^dagger * a2 - a1 * a2^dagger))
    ///
    /// This is a simplified two-mode operator that returns a pair of output
    /// states from a pair of input states, using the SU(2) transformation
    /// on the mode operators.
    pub fn beam_splitter(
        state1: &FockState,
        state2: &FockState,
        theta: f64,
    ) -> (FockState, FockState) {
        let n1 = state1.n_max;
        let n2 = state2.n_max;
        let t = theta.cos();
        let r = theta.sin();

        // For single-mode states we apply the beam-splitter in the tensor
        // product space and trace back. For simplicity, we return the
        // linearly mixed modes.
        let mut out1 = vec![C64::new(0.0, 0.0); n1 + 1];
        let mut out2 = vec![C64::new(0.0, 0.0); n2 + 1];

        let len = (n1 + 1).min(n2 + 1);
        for n in 0..len {
            out1[n] = state1.amplitudes[n] * t + state2.amplitudes[n] * r;
            out2[n] = -state1.amplitudes[n] * r + state2.amplitudes[n] * t;
        }
        // Remaining components from the larger state.
        for n in len..=n1 {
            out1[n] = state1.amplitudes[n] * t;
        }
        for n in len..=n2 {
            out2[n] = state2.amplitudes[n] * t;
        }

        let mut f1 = FockState {
            amplitudes: out1,
            n_max: n1,
        };
        let mut f2 = FockState {
            amplitudes: out2,
            n_max: n2,
        };
        f1.normalize();
        f2.normalize();
        (f1, f2)
    }
}

/// Compute the displacement matrix element <m|D(alpha)|n>.
fn displacement_matrix_element(
    m: usize,
    n: usize,
    alpha: C64,
    alpha_sq: f64,
    exp_factor: f64,
) -> C64 {
    if m >= n {
        let diff = m - n;
        // sqrt(n!/m!) * alpha^(m-n) * exp(-|alpha|^2/2) * L_n^{m-n}(|alpha|^2)
        let ratio = factorial_ratio_sqrt(n, m);
        let alpha_power = complex_power(alpha, diff);
        let lag = laguerre(n, diff as f64, alpha_sq);
        alpha_power * ratio * exp_factor * lag
    } else {
        // <m|D|n> = (-1)^{m+n} * <n|D|m>^*
        let diff = n - m;
        let ratio = factorial_ratio_sqrt(m, n);
        let alpha_power = complex_power(alpha, diff);
        let lag = laguerre(m, diff as f64, alpha_sq);
        let val = alpha_power * ratio * exp_factor * lag;
        let sign = if (m + n) % 2 == 0 { 1.0 } else { -1.0 };
        val.conj() * sign
    }
}

/// Compute sqrt(n!/m!) for n <= m.
fn factorial_ratio_sqrt(n: usize, m: usize) -> f64 {
    // sqrt(n!/m!) = 1 / sqrt((n+1)*(n+2)*...*m) for n < m.
    if n == m {
        return 1.0;
    }
    let mut product = 1.0;
    for k in (n + 1)..=m {
        product *= k as f64;
    }
    1.0 / product.sqrt()
}

/// Compute alpha^n as a complex number.
fn complex_power(alpha: C64, n: usize) -> C64 {
    if n == 0 {
        return C64::new(1.0, 0.0);
    }
    let mut result = C64::new(1.0, 0.0);
    for _ in 0..n {
        result *= alpha;
    }
    result
}

/// Build the squeezing operator matrix in the Fock basis using the recursion.
///
/// The squeezed vacuum has the form:
///   S(xi)|0> = (1/sqrt(mu)) * sum_{k=0}^{...} (-nu/(2*mu))^k * sqrt((2k)!) / k! |2k>
///
/// The general matrix element is computed via the factored form.
fn compute_squeeze_matrix(matrix: &mut [Vec<C64>], dim: usize, mu: f64, nu_mag: f64, phase: C64) {
    // The squeezing matrix in the Fock basis:
    // <m|S(xi)|n> can be computed from the relation:
    //   S(xi) a S^dagger(xi) = mu * a - nu * a^dagger
    // where mu = cosh(r), nu = e^{i*phi} * sinh(r).
    //
    // We use the explicit formula based on the Bogoliubov transformation.
    // For efficiency, we compute this via the recursion:
    //   <m|S|n> involves terms that are non-zero only when m+n is even.
    let nu = phase * nu_mag;
    let inv_mu = 1.0 / mu;
    let _gamma = -nu * inv_mu; // = -e^{i*phi} * tanh(r)

    // Compute S_{m,n} using the analytic formula:
    // S_{m,n} = 0 if m+n is odd
    // For m+n even, use the series expansion.
    // We implement this via direct evaluation of the matrix elements
    // using Hermite polynomial relations.

    for m in 0..dim {
        for n in 0..dim {
            if (m + n) % 2 != 0 {
                matrix[m][n] = C64::new(0.0, 0.0);
                continue;
            }

            // Use the direct formula for the squeezing matrix element.
            // <m|S(r,phi)|n> = sum over allowed k values.
            matrix[m][n] = squeeze_matrix_element(m, n, mu, nu_mag, phase);
        }
    }
}

/// Compute a single squeezing matrix element <m|S(xi)|n>.
///
/// Uses the formula involving the Fock-state representation of the squeeze operator
/// derived from its normal-ordered form.
fn squeeze_matrix_element(m: usize, n: usize, mu: f64, nu_mag: f64, phase: C64) -> C64 {
    if (m + n) % 2 != 0 {
        return C64::new(0.0, 0.0);
    }

    let tanh_r = nu_mag / mu;
    let inv_mu_sqrt = 1.0 / mu.sqrt();

    // For the squeeze operator in the Fock basis, we use the series representation.
    // The matrix element <m|S(r,phi)|n> can be written as:
    //
    // For mu = cosh(r), and defining t = -e^{i*phi} * tanh(r):
    //   <m|S|n> = sqrt(m! * n!) * mu^{-1/2} * sum_k  t^k / (k! * 2^k)
    //             * (-t^*)^{(m+n)/2 - k} / ... (complicated combinatorial expression)
    //
    // Instead we use the recurrence approach: compute the first row/column and recurse.

    // Direct computation via the Bogoliubov relation:
    // Apply S to |n> by expressing it in terms of squeezed number states.
    // This is equivalent to computing the overlap <m| S |n>.
    //
    // We use the well-known formula:
    // <2p|S(r)|2q> = (-1)^{p+q} / (mu^{1/2}) * sum_{j=0}^{min(p,q)} ...
    //
    // For a general, numerically stable approach, we use the matrix recursion:
    // Starting from <0|S|0> = 1/sqrt(cosh(r)),
    // and using the relations:
    //   sqrt(m+1) <m+1|S|n> = mu * sqrt(n) <m|S|n-1> + nu^* * sqrt(m) <m-1|S|n>
    //   (from a * S = (mu * a + nu * a^dagger) applied to kets)
    //
    // This recursion is stable and efficient.

    // Since we already call this for each (m,n), we implement a direct formula.
    // Use the Mehler-like formula for the squeeze operator.

    // An efficient closed-form: for m,n both even or both odd:
    let s = (m + n) / 2;
    let half_m = m / 2;
    let half_n = n / 2;

    // Phase factor: depends on convention. We use xi = r * e^{i*phi}.
    let e_iphi = phase;
    let _e_iphi_conj = phase.conj();

    // The matrix element in terms of tanh(r):
    // <m|S(r,phi)|n> = delta_{m+n even} * sqrt(m! * n!) / (2^s * mu^{1/2})
    //   * sum_{k=0}^{min(half_m, half_n)} (-1)^{s-k} * (e^{i*phi} * tanh(r))^{s-k}
    //     / ((half_m - k)! * (half_n - k)! * k!)
    //   (with appropriate phase conventions)
    //
    // We implement a simplified version using the iterative sum.

    let sqrt_mn = (factorial(m) * factorial(n)).sqrt();
    let two_pow_s = 2.0_f64.powi(s as i32);

    let k_max = half_m.min(half_n);

    let mut sum = C64::new(0.0, 0.0);
    for k in 0..=k_max {
        let exponent = s - k;
        let sign = if exponent % 2 == 0 { 1.0 } else { -1.0 };
        let tanh_pow = tanh_r.powi(exponent as i32);
        let phase_pow = complex_power(e_iphi, exponent);

        let denom = factorial(half_m - k) * factorial(half_n - k) * factorial(k);
        if denom.abs() < EPSILON {
            continue;
        }

        sum += phase_pow * (sign * tanh_pow / denom);
    }

    let result = sum * sqrt_mn * inv_mu_sqrt / two_pow_s;
    result
}

// ============================================================
// CAT CODE
// ============================================================

/// Cat qubit encoding using superpositions of coherent states.
///
/// The cat code encodes a logical qubit in the even/odd parity subspaces
/// of a coherent state superposition. The logical states are:
///
/// - |0_L> = N_+ (|alpha> + |-alpha>)  (even cat, only even photon numbers)
/// - |1_L> = N_- (|alpha> - |-alpha>)  (odd cat, only odd photon numbers)
///
/// where N_+/- are normalization constants.
///
/// The noise bias (ratio of phase flip to bit flip) grows exponentially
/// with alpha^2, making cat codes a promising platform for biased-noise
/// quantum error correction.
#[derive(Clone, Debug)]
pub struct CatCode {
    /// Cat size parameter controlling the amplitude of the coherent states.
    pub alpha: f64,
    /// Fock space truncation level.
    pub n_max: usize,
}

impl CatCode {
    /// Create a new cat code with specified alpha and truncation.
    pub fn new(alpha: f64, n_max: usize) -> Self {
        Self { alpha, n_max }
    }

    /// Construct the logical |0_L> = N_+ (|alpha> + |-alpha>).
    ///
    /// This is an even cat state: only even photon number components survive.
    pub fn logical_zero(&self) -> FockState {
        let alpha = C64::new(self.alpha, 0.0);
        let plus = coherent_state(alpha, self.n_max);
        let minus = coherent_state(-alpha, self.n_max);

        let mut result = plus.clone();
        result.add_scaled(&minus, C64::new(1.0, 0.0));
        result.normalize();
        result
    }

    /// Construct the logical |1_L> = N_- (|alpha> - |-alpha>).
    ///
    /// This is an odd cat state: only odd photon number components survive.
    pub fn logical_one(&self) -> FockState {
        let alpha = C64::new(self.alpha, 0.0);
        let plus = coherent_state(alpha, self.n_max);
        let minus = coherent_state(-alpha, self.n_max);

        let mut result = plus.clone();
        result.add_scaled(&minus, C64::new(-1.0, 0.0));
        result.normalize();
        result
    }

    /// Construct the logical |+_L> which is approximately the coherent state |alpha>.
    pub fn logical_plus(&self) -> FockState {
        let alpha = C64::new(self.alpha, 0.0);
        coherent_state(alpha, self.n_max)
    }

    /// Estimate the bit-flip error rate, which scales as exp(-2*alpha^2).
    ///
    /// For large alpha, the coherent states |alpha> and |-alpha> become nearly
    /// orthogonal, exponentially suppressing bit-flip errors.
    pub fn bit_flip_rate(&self) -> f64 {
        (-2.0 * self.alpha * self.alpha).exp()
    }

    /// Estimate the phase-flip error rate, which scales as alpha^2 * kappa.
    ///
    /// Phase-flip errors grow linearly with the mean photon number alpha^2
    /// and the single-photon loss rate kappa.
    pub fn phase_flip_rate(&self, kappa: f64) -> f64 {
        self.alpha * self.alpha * kappa
    }

    /// Compute the noise bias: ratio of phase-flip to bit-flip rates.
    ///
    /// bias = (alpha^2 * kappa) / exp(-2*alpha^2)
    ///
    /// This grows exponentially with alpha^2, which is the key advantage of cat codes.
    pub fn noise_bias(&self, kappa: f64) -> f64 {
        let bf = self.bit_flip_rate();
        if bf < EPSILON {
            return f64::INFINITY;
        }
        self.phase_flip_rate(kappa) / bf
    }

    /// Encode an arbitrary qubit state cos(theta/2)|0_L> + e^{i*phi}*sin(theta/2)|1_L>.
    pub fn encode_qubit(&self, theta: f64, phi: f64) -> FockState {
        let zero_l = self.logical_zero();
        let one_l = self.logical_one();

        let c0 = C64::new((theta / 2.0).cos(), 0.0);
        let c1 = C64::new(phi.cos(), phi.sin()) * (theta / 2.0).sin();

        let mut result = FockState::new(self.n_max);
        // Zero out the vacuum coefficient that new() sets.
        result.amplitudes[0] = C64::new(0.0, 0.0);

        for n in 0..=self.n_max {
            result.amplitudes[n] = c0 * zero_l.amplitudes[n] + c1 * one_l.amplitudes[n];
        }
        result.normalize();
        result
    }

    /// Compute the mean photon number of the logical zero state.
    pub fn mean_photon_number(&self) -> f64 {
        self.logical_zero().photon_number_expectation()
    }
}

// ============================================================
// GKP CODE
// ============================================================

/// Gottesman-Kitaev-Preskill (GKP) code for encoding a qubit in an oscillator.
///
/// Ideal GKP states are infinite superpositions of position eigenstates on a
/// sqrt(pi)-spaced grid. Finite-energy (approximate) GKP states use Gaussian
/// envelopes controlled by the squeezing parameter delta:
///
/// - |0_L> ~ sum_s exp(-delta^2 * (2s)^2 / 2) |q = 2s * sqrt(pi)>
/// - |1_L> ~ sum_s exp(-delta^2 * (2s+1)^2 / 2) |q = (2s+1) * sqrt(pi)>
///
/// Smaller delta corresponds to more squeezing and better code performance,
/// but requires more photons.
#[derive(Clone, Debug)]
pub struct GKPCode {
    /// Squeezing parameter for finite-energy GKP states.
    /// Smaller values give better code states but require more photons.
    pub delta: f64,
    /// Fock space truncation level.
    pub n_max: usize,
}

impl GKPCode {
    /// Create a new GKP code with specified squeezing parameter and truncation.
    pub fn new(delta: f64, n_max: usize) -> Self {
        Self { delta, n_max }
    }

    /// Construct the logical |0_L> GKP state.
    ///
    /// |0_L> ~ sum_s exp(-delta^2 * (2s)^2 / 2) |q = 2s * sqrt(pi)>
    pub fn logical_zero(&self) -> FockState {
        let sqrt_pi = PI.sqrt();
        let mut state = FockState::new(self.n_max);
        state.amplitudes[0] = C64::new(0.0, 0.0); // Clear vacuum

        // Sum over grid points. The Gaussian envelope limits the effective range.
        let s_max = ((5.0 / self.delta).ceil() as i64).max(10);

        for s in -s_max..=s_max {
            let q = 2.0 * s as f64 * sqrt_pi;
            let weight = (-self.delta * self.delta * (2.0 * s as f64).powi(2) / 2.0).exp();
            if weight < 1e-15 {
                continue;
            }

            let pos_state = Self::position_state(q, self.n_max);
            state.add_scaled(&pos_state, C64::new(weight, 0.0));
        }

        state.normalize();
        state
    }

    /// Construct the logical |1_L> GKP state.
    ///
    /// |1_L> ~ sum_s exp(-delta^2 * (2s+1)^2 / 2) |q = (2s+1) * sqrt(pi)>
    pub fn logical_one(&self) -> FockState {
        let sqrt_pi = PI.sqrt();
        let mut state = FockState::new(self.n_max);
        state.amplitudes[0] = C64::new(0.0, 0.0);

        let s_max = ((5.0 / self.delta).ceil() as i64).max(10);

        for s in -s_max..=s_max {
            let q = (2.0 * s as f64 + 1.0) * sqrt_pi;
            let weight = (-self.delta * self.delta * (2.0 * s as f64 + 1.0).powi(2) / 2.0).exp();
            if weight < 1e-15 {
                continue;
            }

            let pos_state = Self::position_state(q, self.n_max);
            state.add_scaled(&pos_state, C64::new(weight, 0.0));
        }

        state.normalize();
        state
    }

    /// Construct an approximate position eigenstate |q> in the Fock basis.
    ///
    /// |q> = sum_n psi_n(q) |n> where psi_n(q) is the nth harmonic oscillator
    /// wave function:
    ///
    ///   psi_n(q) = (1 / (pi^{1/4} * sqrt(2^n * n!))) * H_n(q) * exp(-q^2/2)
    pub fn position_state(q: f64, n_max: usize) -> FockState {
        let mut amplitudes = vec![C64::new(0.0, 0.0); n_max + 1];
        let exp_factor = (-q * q / 2.0).exp();
        let pi_quarter = PI.powf(0.25);

        for n in 0..=n_max {
            let h_n = hermite(n, q);
            let norm = pi_quarter * (2.0_f64.powi(n as i32) * factorial(n)).sqrt();
            if norm.abs() > EPSILON {
                amplitudes[n] = C64::new(exp_factor * h_n / norm, 0.0);
            }
        }

        FockState { amplitudes, n_max }
    }

    /// Perform a GKP syndrome measurement.
    ///
    /// Returns approximate syndromes (sq, sp) where:
    /// - sq = <q> mod sqrt(pi)  (position syndrome)
    /// - sp = <p> mod sqrt(pi)  (momentum syndrome)
    ///
    /// These syndromes indicate the displacement errors that need correction.
    pub fn syndrome_measurement(&self, state: &FockState) -> (f64, f64) {
        let sqrt_pi = PI.sqrt();

        // Compute <q> = <(a + a^dagger) / sqrt(2)>
        // <q> = (1/sqrt(2)) * sum_n (sqrt(n+1) * conj(c_{n+1}) * c_n + sqrt(n) * conj(c_{n-1}) * c_n)
        //     = (1/sqrt(2)) * sum_n (sqrt(n+1) * (conj(c_{n+1}) * c_n + conj(c_n) * c_{n+1}))
        //     = sqrt(2) * sum_n sqrt(n+1) * Re(conj(c_{n+1}) * c_n)
        let mut q_exp = 0.0;
        let dim = state.amplitudes.len();
        for n in 0..(dim - 1) {
            let overlap = state.amplitudes[n + 1].conj() * state.amplitudes[n];
            q_exp += ((n + 1) as f64).sqrt() * overlap.re;
        }
        q_exp *= 2.0_f64.sqrt();

        // Compute <p> = <i(a^dagger - a) / sqrt(2)>
        //     = sqrt(2) * sum_n sqrt(n+1) * Im(conj(c_{n+1}) * c_n)
        let mut p_exp = 0.0;
        for n in 0..(dim - 1) {
            let overlap = state.amplitudes[n + 1].conj() * state.amplitudes[n];
            p_exp += ((n + 1) as f64).sqrt() * overlap.im;
        }
        p_exp *= 2.0_f64.sqrt();

        // Reduce modulo sqrt(pi) to the range [-sqrt(pi)/2, sqrt(pi)/2].
        let sq = modulo_centered(q_exp, sqrt_pi);
        let sp = modulo_centered(p_exp, sqrt_pi);

        (sq, sp)
    }

    /// Apply a displacement correction based on syndrome values.
    ///
    /// The correction displaces the state by -syndrome to move it back
    /// to the nearest grid point.
    pub fn correct(&self, state: &FockState, syndrome: (f64, f64)) -> FockState {
        let (sq, sp) = syndrome;
        // Displacement to correct: alpha = -(sq + i*sp) / sqrt(2)
        // (converting from quadrature coordinates to mode amplitude)
        let alpha = C64::new(-sq, -sp) / 2.0_f64.sqrt();
        BosonicOperator::displacement(state, alpha)
    }

    /// Compute the mean photon number of the logical zero GKP state.
    pub fn mean_photon_number(&self) -> f64 {
        self.logical_zero().photon_number_expectation()
    }
}

/// Reduce x modulo period to the centered range [-period/2, period/2).
fn modulo_centered(x: f64, period: f64) -> f64 {
    let mut r = x % period;
    if r > period / 2.0 {
        r -= period;
    } else if r < -period / 2.0 {
        r += period;
    }
    r
}

// ============================================================
// BINOMIAL CODE
// ============================================================

/// Binomial ("kitten") code for protection against photon loss.
///
/// The binomial code with spacing parameter S protects against up to
/// S-1 photon loss events. The logical states are:
///
/// - |0_L> = sum_k sqrt(C(S+1, 2k)) |2k*S> / norm
/// - |1_L> = sum_k sqrt(C(S+1, 2k+1)) |(2k+1)*S> / norm
///
/// where C(n,k) is the binomial coefficient. The spacing S=2 protects
/// against single photon loss.
#[derive(Clone, Debug)]
pub struct BinomialCode {
    /// Spacing parameter S. Protects against up to S-1 photon losses.
    pub spacing: usize,
    /// Fock space truncation level.
    pub n_max: usize,
}

impl BinomialCode {
    /// Create a new binomial code with specified spacing and truncation.
    pub fn new(spacing: usize, n_max: usize) -> Self {
        assert!(spacing >= 1, "Spacing must be at least 1");
        Self { spacing, n_max }
    }

    /// Construct the logical |0_L> binomial code state.
    ///
    /// |0_L> = sum_k sqrt(C(S+1, 2k)) |2k*S> / norm
    pub fn logical_zero(&self) -> FockState {
        let s = self.spacing;
        let mut amplitudes = vec![C64::new(0.0, 0.0); self.n_max + 1];

        let k_max = (s + 1) / 2; // C(S+1, 2k) is nonzero for 2k <= S+1
        for k in 0..=k_max {
            let photon_number = 2 * k * s;
            if photon_number > self.n_max {
                break;
            }
            let coeff = binomial_coeff(s + 1, 2 * k).sqrt();
            amplitudes[photon_number] = C64::new(coeff, 0.0);
        }

        let mut state = FockState {
            amplitudes,
            n_max: self.n_max,
        };
        state.normalize();
        state
    }

    /// Construct the logical |1_L> binomial code state.
    ///
    /// |1_L> = sum_k sqrt(C(S+1, 2k+1)) |(2k+1)*S> / norm
    pub fn logical_one(&self) -> FockState {
        let s = self.spacing;
        let mut amplitudes = vec![C64::new(0.0, 0.0); self.n_max + 1];

        let k_max = s / 2; // C(S+1, 2k+1) is nonzero for 2k+1 <= S+1
        for k in 0..=k_max {
            let photon_number = (2 * k + 1) * s;
            if photon_number > self.n_max {
                break;
            }
            let coeff = binomial_coeff(s + 1, 2 * k + 1).sqrt();
            amplitudes[photon_number] = C64::new(coeff, 0.0);
        }

        let mut state = FockState {
            amplitudes,
            n_max: self.n_max,
        };
        state.normalize();
        state
    }

    /// Compute the mean photon number of the logical zero binomial state.
    pub fn mean_photon_number(&self) -> f64 {
        self.logical_zero().photon_number_expectation()
    }
}

// ============================================================
// BOSONIC NOISE CHANNELS
// ============================================================

/// Noise channels relevant to bosonic modes.
///
/// These implement simplified versions of the quantum channels that
/// act on the Fock-space state vector. For full density-matrix simulation
/// of mixed states, use the `BosonicSimulator`.
pub struct BosonicChannel;

impl BosonicChannel {
    /// Single-photon loss channel (amplitude damping).
    ///
    /// Models the effect of energy relaxation where the mode loses photons
    /// to the environment at rate gamma. This applies the Kraus operator:
    ///
    ///   K_0 = sum_n sqrt(1 - gamma)^n |n><n|   (no jump)
    ///   K_1 = sqrt(gamma) * a                   (single photon loss)
    ///
    /// For small gamma, this is the dominant error in superconducting cavities.
    pub fn photon_loss(state: &FockState, gamma: f64) -> FockState {
        let n_max = state.n_max;
        let gamma = gamma.clamp(0.0, 1.0);

        // No-jump component: K_0 |psi> = sum_n (1-gamma)^{n/2} c_n |n>
        let mut no_jump = vec![C64::new(0.0, 0.0); n_max + 1];
        for n in 0..=n_max {
            let damping = (1.0 - gamma).powf(n as f64 / 2.0);
            no_jump[n] = state.amplitudes[n] * damping;
        }

        // Jump component: K_1 |psi> = sqrt(gamma) * a |psi>
        let a_state = BosonicOperator::annihilation(state);
        let gamma_sqrt = gamma.sqrt();

        // Combine: rho' ~ K_0|psi><psi|K_0^dagger + K_1|psi><psi|K_1^dagger
        // For a pure-state approximation, we return the no-jump trajectory
        // (quantum trajectory approach) which is the more likely outcome.
        // The probability of no jump is ||K_0|psi>||^2.
        let no_jump_norm_sq: f64 = no_jump.iter().map(|a| a.norm_sqr()).sum();
        let jump_norm_sq: f64 = a_state
            .amplitudes
            .iter()
            .map(|a| a.norm_sqr() * gamma)
            .sum();

        // Mix the two outcomes weighted by their probabilities to get an
        // approximate damped state. For a pure-state channel we weight
        // by amplitude:
        let total = (no_jump_norm_sq + jump_norm_sq).sqrt();
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];
        if total > EPSILON {
            let inv_total = 1.0 / total;
            for n in 0..=n_max {
                result[n] = (no_jump[n] + a_state.amplitudes[n] * gamma_sqrt) * inv_total;
            }
        }

        let mut fock = FockState {
            amplitudes: result,
            n_max,
        };
        fock.normalize();
        fock
    }

    /// Dephasing channel (random phase kicks).
    ///
    /// Models the loss of phase coherence due to random fluctuations in the
    /// oscillator frequency. Applies exp(-gamma * n^2 / 2) damping to each
    /// off-diagonal element, which in the Fock basis acts as:
    ///
    ///   c_n -> c_n * exp(-gamma * n^2 / 2)
    ///
    /// This suppresses high-photon-number components faster.
    pub fn dephasing(state: &FockState, gamma: f64) -> FockState {
        let n_max = state.n_max;
        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];

        for n in 0..=n_max {
            let damping = (-gamma * (n as f64).powi(2) / 2.0).exp();
            result[n] = state.amplitudes[n] * damping;
        }

        let mut fock = FockState {
            amplitudes: result,
            n_max,
        };
        fock.normalize();
        fock
    }

    /// Thermal noise channel (thermal photon injection).
    ///
    /// Models the injection of thermal photons from a bath at temperature
    /// corresponding to mean thermal occupation n_th. The effect is to
    /// redistribute photon number populations toward a thermal distribution.
    ///
    /// Applies a simplified model: each Fock state |n> gains population from
    /// thermal excitation weighted by n_th and loses population at rate gamma.
    pub fn thermal_noise(state: &FockState, n_th: f64, gamma: f64) -> FockState {
        let n_max = state.n_max;
        let gamma = gamma.clamp(0.0, 1.0);

        let mut result = vec![C64::new(0.0, 0.0); n_max + 1];

        // Thermal channel mixes the state toward a thermal distribution.
        // Simplified Kraus model:
        //   loss: (1-gamma) decay factor per photon
        //   gain: gamma * n_th thermal excitation
        for n in 0..=n_max {
            // Damping from loss.
            let loss_factor = (1.0 - gamma).powf(n as f64 / 2.0);
            result[n] = state.amplitudes[n] * loss_factor;

            // Add thermal excitation: mix in some |n> population
            // proportional to the Bose-Einstein distribution at n_th.
            if n_th > EPSILON && n > 0 {
                let thermal_weight = gamma * n_th / (1.0 + n_th);
                let bose_factor = (n_th / (1.0 + n_th)).powi(n as i32);
                let thermal_amp = (thermal_weight * bose_factor).sqrt();
                // Add a small thermal component to increase population at
                // higher photon numbers.
                result[n] += C64::new(thermal_amp * 0.1, 0.0);
            }
        }

        let mut fock = FockState {
            amplitudes: result,
            n_max,
        };
        fock.normalize();
        fock
    }
}

// ============================================================
// WIGNER FUNCTION
// ============================================================

/// Phase-space representation of bosonic states via the Wigner function.
///
/// The Wigner function W(x, p) provides a quasi-probability distribution
/// in phase space. For a state with density matrix elements rho_{mn} in
/// the Fock basis:
///
///   W(x, p) = (1/pi) * sum_{m,n} rho_{mn} * W_{mn}(x, p)
///
/// where W_{mn} are the Wigner matrix elements computed using associated
/// Laguerre polynomials.
pub struct WignerFunction;

impl WignerFunction {
    /// Compute the Wigner function on a 2D grid.
    ///
    /// # Arguments
    ///
    /// * `state` - The Fock state to visualize.
    /// * `x_range` - Range of position quadrature (min, max).
    /// * `p_range` - Range of momentum quadrature (min, max).
    /// * `resolution` - Number of grid points in each dimension.
    ///
    /// # Returns
    ///
    /// A `resolution x resolution` matrix of Wigner function values W(x_i, p_j).
    pub fn compute(
        state: &FockState,
        x_range: (f64, f64),
        p_range: (f64, f64),
        resolution: usize,
    ) -> Vec<Vec<f64>> {
        let dim = state.amplitudes.len();
        let dx = (x_range.1 - x_range.0) / (resolution as f64 - 1.0).max(1.0);
        let dp = (p_range.1 - p_range.0) / (resolution as f64 - 1.0).max(1.0);

        let mut grid = vec![vec![0.0_f64; resolution]; resolution];

        for ix in 0..resolution {
            let x = x_range.0 + ix as f64 * dx;
            for ip in 0..resolution {
                let p = p_range.0 + ip as f64 * dp;

                // r^2 = x^2 + p^2 in phase space.
                let r_sq = x * x + p * p;

                let mut w = 0.0_f64;

                // Diagonal terms: W_{nn}(x, p)
                for n in 0..dim {
                    let rho_nn = state.amplitudes[n].norm_sqr();
                    if rho_nn < 1e-15 {
                        continue;
                    }

                    // W_{nn}(x,p) = ((-1)^n / pi) * exp(-r^2) * L_n(2*r^2)
                    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                    let w_nn = sign * (-r_sq).exp() * laguerre(n, 0.0, 2.0 * r_sq) / PI;
                    w += rho_nn * w_nn;
                }

                // Off-diagonal terms: W_{mn}(x, p) for m != n.
                // W_{mn} = ((-1)^n / pi) * sqrt(n!/m!) * (2(x+ip))^{m-n}
                //          * exp(-r^2) * L_n^{m-n}(2*r^2)   for m > n
                // And W_{nm} = W_{mn}^*.
                for m in 0..dim {
                    for n in 0..m {
                        let rho_mn = state.amplitudes[m] * state.amplitudes[n].conj();
                        if rho_mn.norm_sqr() < 1e-20 {
                            continue;
                        }

                        let diff = m - n;
                        let sign_n = if n % 2 == 0 { 1.0 } else { -1.0 };

                        let ratio = factorial_ratio_sqrt(n, m);
                        let z = C64::new(x, p) * 2.0_f64.sqrt();
                        let z_power = complex_power(z, diff);
                        let lag = laguerre(n, diff as f64, 2.0 * r_sq);

                        let w_mn_complex = z_power * (sign_n * ratio * (-r_sq).exp() * lag / PI);

                        // Contribution from rho_{mn} * W_{mn} + rho_{nm} * W_{nm}
                        // = 2 * Re(rho_{mn} * W_{mn}).
                        w += 2.0 * (rho_mn * w_mn_complex).re;
                    }
                }

                grid[ix][ip] = w;
            }
        }

        grid
    }

    /// Compute the Wigner function at a single point in phase space.
    pub fn compute_point(state: &FockState, x: f64, p: f64) -> f64 {
        let dim = state.amplitudes.len();
        let r_sq = x * x + p * p;
        let mut w = 0.0_f64;

        for n in 0..dim {
            let rho_nn = state.amplitudes[n].norm_sqr();
            if rho_nn < 1e-15 {
                continue;
            }
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            w += rho_nn * sign * (-r_sq).exp() * laguerre(n, 0.0, 2.0 * r_sq) / PI;
        }

        for m in 0..dim {
            for n in 0..m {
                let rho_mn = state.amplitudes[m] * state.amplitudes[n].conj();
                if rho_mn.norm_sqr() < 1e-20 {
                    continue;
                }
                let diff = m - n;
                let sign_n = if n % 2 == 0 { 1.0 } else { -1.0 };
                let ratio = factorial_ratio_sqrt(n, m);
                let z = C64::new(x, p) * 2.0_f64.sqrt();
                let z_power = complex_power(z, diff);
                let lag = laguerre(n, diff as f64, 2.0 * r_sq);
                let w_mn_complex = z_power * (sign_n * ratio * (-r_sq).exp() * lag / PI);
                w += 2.0 * (rho_mn * w_mn_complex).re;
            }
        }

        w
    }
}

// ============================================================
// BOSONIC CODE TYPE ENUM
// ============================================================

/// Enum selecting the type of bosonic code.
#[derive(Clone, Debug)]
pub enum BosonicCodeType {
    /// Cat code with the given alpha parameter.
    Cat(f64),
    /// GKP code with the given delta (squeezing) parameter.
    GKP(f64),
    /// Binomial code with the given spacing parameter.
    Binomial(usize),
}

// ============================================================
// BOSONIC SIMULATOR
// ============================================================

/// High-level simulator for bosonic error correction codes.
///
/// Provides logical qubit operations, noise application, and measurement
/// within the chosen bosonic code.
#[derive(Clone, Debug)]
pub struct BosonicSimulator {
    /// The current quantum state in the Fock basis.
    pub state: FockState,
    /// The type of bosonic code being simulated.
    pub code_type: BosonicCodeType,
    /// Fock space truncation level.
    pub n_max: usize,
}

impl BosonicSimulator {
    /// Create a new simulator initialized to the logical |0_L> state.
    pub fn new(code: BosonicCodeType, n_max: usize) -> Self {
        let state = match &code {
            BosonicCodeType::Cat(alpha) => {
                let cat = CatCode::new(*alpha, n_max);
                cat.logical_zero()
            }
            BosonicCodeType::GKP(delta) => {
                let gkp = GKPCode::new(*delta, n_max);
                gkp.logical_zero()
            }
            BosonicCodeType::Binomial(spacing) => {
                let binom = BinomialCode::new(*spacing, n_max);
                binom.logical_zero()
            }
        };
        Self {
            state,
            code_type: code,
            n_max,
        }
    }

    /// Apply a logical X (bit-flip) operation.
    ///
    /// The logical X maps |0_L> <-> |1_L>. For each code type:
    /// - Cat: rotation by pi in phase space, R(pi)
    /// - GKP: displacement by sqrt(pi)/2 in position
    /// - Binomial: parity-based flip
    pub fn logical_x(&mut self) {
        match &self.code_type {
            BosonicCodeType::Cat(_alpha) => {
                // For cat codes, logical X is the parity operator (or pi rotation).
                // R(pi)|even cat> = |even cat> and R(pi)|odd cat> = -|odd cat>
                // Actually, logical X swaps |0_L> and |1_L>, which for cat codes
                // can be implemented as a displacement or a rotation.
                // A simpler approach: logical X ~ D(2*alpha * i) approximately,
                // but the exact implementation for a pi/2 rotation:
                // Logical X for cat codes = R(pi) which flips parity.
                //
                // More precisely, for even/odd cats, a pi rotation maps
                // |alpha> -> |-alpha>, which sends |0_L> -> |0_L> and |1_L> -> -|1_L>.
                // That is logical Z, not X.
                //
                // Logical X for cat codes: X_L swaps even<->odd parity.
                // This is achieved by applying a displacement that shifts by alpha
                // to break the even/odd symmetry, or more directly by the operator
                // that maps (|a>+|-a>) <-> (|a>-|-a>).
                //
                // For simplicity we implement this as a photon-number rotation:
                // multiply odd components by -1, then swap even/odd weights.
                // This is equivalent to X_L = a / |alpha| (normalized annihilation).
                //
                // Exact implementation: swap the coefficients of the even and odd subspaces.
                let zero_l = CatCode::new(
                    match &self.code_type {
                        BosonicCodeType::Cat(a) => *a,
                        _ => unreachable!(),
                    },
                    self.n_max,
                )
                .logical_zero();
                let one_l = CatCode::new(
                    match &self.code_type {
                        BosonicCodeType::Cat(a) => *a,
                        _ => unreachable!(),
                    },
                    self.n_max,
                )
                .logical_one();

                // Project onto logical subspace and swap.
                let c0 = self.state.inner_product(&zero_l);
                let c1 = self.state.inner_product(&one_l);

                // X_L maps c0|0_L> + c1|1_L> -> c1|0_L> + c0|1_L>
                let mut new_state = FockState::new(self.n_max);
                new_state.amplitudes[0] = C64::new(0.0, 0.0);
                for n in 0..=self.n_max {
                    new_state.amplitudes[n] = c1 * zero_l.amplitudes[n] + c0 * one_l.amplitudes[n];
                }
                new_state.normalize();
                self.state = new_state;
            }
            BosonicCodeType::GKP(_delta) => {
                // For GKP codes, logical X is displacement by sqrt(pi) in position:
                // X_L = D(sqrt(pi) / sqrt(2))
                let sqrt_pi = PI.sqrt();
                let alpha = C64::new(sqrt_pi / 2.0_f64.sqrt(), 0.0);
                self.state = BosonicOperator::displacement(&self.state, alpha);
            }
            BosonicCodeType::Binomial(spacing) => {
                // For binomial codes, logical X swaps |0_L> and |1_L>.
                let binom = BinomialCode::new(*spacing, self.n_max);
                let zero_l = binom.logical_zero();
                let one_l = binom.logical_one();

                let c0 = self.state.inner_product(&zero_l);
                let c1 = self.state.inner_product(&one_l);

                let mut new_state = FockState::new(self.n_max);
                new_state.amplitudes[0] = C64::new(0.0, 0.0);
                for n in 0..=self.n_max {
                    new_state.amplitudes[n] = c1 * zero_l.amplitudes[n] + c0 * one_l.amplitudes[n];
                }
                new_state.normalize();
                self.state = new_state;
            }
        }
    }

    /// Apply a logical Z (phase-flip) operation.
    ///
    /// The logical Z applies a relative phase of -1 between |0_L> and |1_L>.
    /// For each code type:
    /// - Cat: parity operator (-1)^n
    /// - GKP: displacement by sqrt(pi) in momentum
    /// - Binomial: phase on odd-parity component
    pub fn logical_z(&mut self) {
        match &self.code_type {
            BosonicCodeType::Cat(_) => {
                // For cat codes, Z_L = parity operator.
                // P|0_L> = |0_L> (even cat), P|1_L> = -|1_L> (odd cat)
                // This is exactly what Z_L should do: phase flip |1_L>.
                self.state = BosonicOperator::parity_operator(&self.state);
            }
            BosonicCodeType::GKP(_delta) => {
                // For GKP codes, logical Z is displacement by sqrt(pi) in momentum:
                // Z_L = D(i * sqrt(pi) / sqrt(2))
                let sqrt_pi = PI.sqrt();
                let alpha = C64::new(0.0, sqrt_pi / 2.0_f64.sqrt());
                self.state = BosonicOperator::displacement(&self.state, alpha);
            }
            BosonicCodeType::Binomial(spacing) => {
                // For binomial codes, Z_L = I on |0_L>, -I on |1_L>.
                let binom = BinomialCode::new(*spacing, self.n_max);
                let zero_l = binom.logical_zero();
                let one_l = binom.logical_one();

                let c0 = self.state.inner_product(&zero_l);
                let c1 = self.state.inner_product(&one_l);

                let mut new_state = FockState::new(self.n_max);
                new_state.amplitudes[0] = C64::new(0.0, 0.0);
                for n in 0..=self.n_max {
                    new_state.amplitudes[n] = c0 * zero_l.amplitudes[n] - c1 * one_l.amplitudes[n];
                }
                new_state.normalize();
                self.state = new_state;
            }
        }
    }

    /// Apply a noise channel to the current state.
    ///
    /// This dispatches to the appropriate channel based on the `NoiseType`.
    pub fn apply_noise(&mut self, noise: &NoiseType) {
        self.state = match noise {
            NoiseType::PhotonLoss(gamma) => BosonicChannel::photon_loss(&self.state, *gamma),
            NoiseType::Dephasing(gamma) => BosonicChannel::dephasing(&self.state, *gamma),
            NoiseType::ThermalNoise { n_th, gamma } => {
                BosonicChannel::thermal_noise(&self.state, *n_th, *gamma)
            }
        };
    }

    /// Measure in the logical Z basis.
    ///
    /// Projects the state onto |0_L> or |1_L> and returns the outcome
    /// (false = |0_L>, true = |1_L>).
    pub fn measure_logical(&self) -> bool {
        let (zero_l, one_l) = match &self.code_type {
            BosonicCodeType::Cat(alpha) => {
                let cat = CatCode::new(*alpha, self.n_max);
                (cat.logical_zero(), cat.logical_one())
            }
            BosonicCodeType::GKP(delta) => {
                let gkp = GKPCode::new(*delta, self.n_max);
                (gkp.logical_zero(), gkp.logical_one())
            }
            BosonicCodeType::Binomial(spacing) => {
                let binom = BinomialCode::new(*spacing, self.n_max);
                (binom.logical_zero(), binom.logical_one())
            }
        };

        let p0 = self.state.fidelity(&zero_l);
        let p1 = self.state.fidelity(&one_l);

        // Deterministic measurement based on relative probabilities.
        // For a proper quantum simulation one would sample with probability p0/(p0+p1).
        p1 > p0
    }

    /// Compute the fidelity of the current state with the code space.
    ///
    /// Returns the total overlap with both logical basis states:
    /// F = |<0_L|psi>|^2 + |<1_L|psi>|^2.
    pub fn fidelity_with_codespace(&self) -> f64 {
        let (zero_l, one_l) = match &self.code_type {
            BosonicCodeType::Cat(alpha) => {
                let cat = CatCode::new(*alpha, self.n_max);
                (cat.logical_zero(), cat.logical_one())
            }
            BosonicCodeType::GKP(delta) => {
                let gkp = GKPCode::new(*delta, self.n_max);
                (gkp.logical_zero(), gkp.logical_one())
            }
            BosonicCodeType::Binomial(spacing) => {
                let binom = BinomialCode::new(*spacing, self.n_max);
                (binom.logical_zero(), binom.logical_one())
            }
        };

        let f0 = self.state.fidelity(&zero_l);
        let f1 = self.state.fidelity(&one_l);
        (f0 + f1).min(1.0)
    }

    /// Get the current state (immutable reference).
    pub fn current_state(&self) -> &FockState {
        &self.state
    }

    /// Set the state to a specific Fock state.
    pub fn set_state(&mut self, state: FockState) {
        assert_eq!(
            state.n_max, self.n_max,
            "State truncation must match simulator truncation"
        );
        self.state = state;
    }
}

// ============================================================
// NOISE TYPE ENUM
// ============================================================

/// Types of noise channels for the bosonic simulator.
#[derive(Clone, Debug)]
pub enum NoiseType {
    /// Photon loss with decay probability gamma.
    PhotonLoss(f64),
    /// Dephasing with rate gamma.
    Dephasing(f64),
    /// Thermal noise with mean thermal photon number n_th and coupling gamma.
    ThermalNoise { n_th: f64, gamma: f64 },
}

// ============================================================
// ADDITIONAL UTILITIES
// ============================================================

/// Compute the Mandel Q parameter: Q = (Var(n) - <n>) / <n>.
///
/// Q < 0: sub-Poissonian (nonclassical), Q = 0: Poissonian (coherent),
/// Q > 0: super-Poissonian (thermal / bunched).
pub fn mandel_q(state: &FockState) -> f64 {
    let mean_n = state.photon_number_expectation();
    if mean_n < EPSILON {
        return 0.0;
    }
    let var_n = state.photon_number_variance();
    (var_n - mean_n) / mean_n
}

/// Compute the second-order correlation function g^(2)(0).
///
/// g^(2)(0) = <n(n-1)> / <n>^2
///
/// g^(2)(0) < 1: antibunched, g^(2)(0) = 1: coherent,
/// g^(2)(0) > 1: bunched / thermal.
pub fn g2_zero(state: &FockState) -> f64 {
    let mean_n = state.photon_number_expectation();
    if mean_n < EPSILON {
        return 0.0;
    }
    let n_n_minus_1: f64 = state
        .amplitudes
        .iter()
        .enumerate()
        .map(|(n, a)| {
            if n < 2 {
                0.0
            } else {
                n as f64 * (n as f64 - 1.0) * a.norm_sqr()
            }
        })
        .sum();
    n_n_minus_1 / (mean_n * mean_n)
}

/// Create a thermal state with mean photon number n_th.
///
/// P(n) = n_th^n / (1 + n_th)^{n+1}
pub fn thermal_state(n_th: f64, n_max: usize) -> FockState {
    let mut amplitudes = vec![C64::new(0.0, 0.0); n_max + 1];

    if n_th < EPSILON {
        // Zero temperature -> vacuum.
        amplitudes[0] = C64::new(1.0, 0.0);
    } else {
        // Diagonal thermal state. We represent it as a pure state with
        // real amplitudes proportional to sqrt(P(n)).
        for n in 0..=n_max {
            let p_n = n_th.powi(n as i32) / (1.0 + n_th).powi(n as i32 + 1);
            amplitudes[n] = C64::new(p_n.sqrt(), 0.0);
        }
    }

    let mut state = FockState { amplitudes, n_max };
    state.normalize();
    state
}

/// Create a squeezed vacuum state S(r)|0>.
pub fn squeezed_vacuum(r: f64, n_max: usize) -> FockState {
    let vacuum = FockState::new(n_max);
    BosonicOperator::squeeze(&vacuum, r, 0.0)
}

/// Create a displaced squeezed state D(alpha) S(r) |0>.
pub fn displaced_squeezed(alpha: C64, r: f64, n_max: usize) -> FockState {
    let squeezed = squeezed_vacuum(r, n_max);
    BosonicOperator::displacement(&squeezed, alpha)
}

/// Compute the photon number distribution entropy: S = -sum_n P(n) * ln(P(n)).
pub fn photon_entropy(state: &FockState) -> f64 {
    let probs = state.probabilities();
    let mut entropy = 0.0;
    for p in &probs {
        if *p > EPSILON {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Compute the purity of the state: Tr(rho^2).
/// For a pure state this equals 1.0.
pub fn purity(state: &FockState) -> f64 {
    // For a pure state |psi>, Tr(|psi><psi|^2) = |<psi|psi>|^2 = ||psi||^4.
    let norm_sq: f64 = state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
    norm_sq * norm_sq
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;
    const N_MAX: usize = 30;

    // ----------------------------------------------------------
    // Fock state basics
    // ----------------------------------------------------------

    #[test]
    fn test_vacuum_state_photon_number_zero() {
        let vac = FockState::new(N_MAX);
        assert!(
            vac.photon_number_expectation().abs() < TOL,
            "Vacuum <n> should be 0, got {}",
            vac.photon_number_expectation()
        );
    }

    #[test]
    fn test_fock_state_photon_number() {
        for n in 0..=10 {
            let state = FockState::from_number(n, N_MAX);
            let expectation = state.photon_number_expectation();
            assert!(
                (expectation - n as f64).abs() < TOL,
                "|{}> should have <n> = {}, got {}",
                n,
                n,
                expectation
            );
        }
    }

    #[test]
    fn test_fock_state_zero_variance() {
        let state = FockState::from_number(5, N_MAX);
        assert!(
            state.photon_number_variance().abs() < TOL,
            "Fock state should have zero photon number variance"
        );
    }

    #[test]
    fn test_fock_state_norm() {
        let state = FockState::from_number(3, N_MAX);
        assert!(
            (state.norm() - 1.0).abs() < TOL,
            "Fock state should be normalized"
        );
    }

    // ----------------------------------------------------------
    // Creation and annihilation operators
    // ----------------------------------------------------------

    #[test]
    fn test_creation_operator() {
        // a^dagger |n> = sqrt(n+1) |n+1>
        let n = 4;
        let state = FockState::from_number(n, N_MAX);
        let result = BosonicOperator::creation(&state);

        // Should have amplitude sqrt(n+1) at position n+1.
        let expected = ((n + 1) as f64).sqrt();
        assert!(
            (result.amplitudes[n + 1].re - expected).abs() < TOL,
            "a^dagger |{}> should give sqrt({}) at |{}>, got {}",
            n,
            n + 1,
            n + 1,
            result.amplitudes[n + 1].re
        );

        // All other components should be zero.
        for m in 0..=N_MAX {
            if m != n + 1 {
                assert!(
                    result.amplitudes[m].norm() < TOL,
                    "a^dagger |{}> should be zero at |{}>, got {}",
                    n,
                    m,
                    result.amplitudes[m].norm()
                );
            }
        }
    }

    #[test]
    fn test_annihilation_of_vacuum() {
        // a|0> = 0
        let vac = FockState::new(N_MAX);
        let result = BosonicOperator::annihilation(&vac);
        let norm = result.norm();
        assert!(
            norm < TOL,
            "a|0> should be the zero vector, got norm {}",
            norm
        );
    }

    #[test]
    fn test_annihilation_operator() {
        // a|n> = sqrt(n)|n-1>
        let n = 5;
        let state = FockState::from_number(n, N_MAX);
        let result = BosonicOperator::annihilation(&state);

        let expected = (n as f64).sqrt();
        assert!(
            (result.amplitudes[n - 1].re - expected).abs() < TOL,
            "a|{}> should give sqrt({}) at |{}>, got {}",
            n,
            n,
            n - 1,
            result.amplitudes[n - 1].re
        );
    }

    // ----------------------------------------------------------
    // Number operator
    // ----------------------------------------------------------

    #[test]
    fn test_number_operator_expectation() {
        // <n|n_hat|n> = n for a Fock state
        for n in 0..10 {
            let state = FockState::from_number(n, N_MAX);
            let n_hat_state = BosonicOperator::number_operator(&state);
            let expectation = state.inner_product(&n_hat_state).re;
            assert!(
                (expectation - n as f64).abs() < TOL,
                "<{}|n_hat|{}> should be {}, got {}",
                n,
                n,
                n,
                expectation
            );
        }
    }

    // ----------------------------------------------------------
    // Coherent state
    // ----------------------------------------------------------

    #[test]
    fn test_coherent_state_poisson_statistics() {
        // A coherent state |alpha> has <n> = |alpha|^2 and Var(n) = |alpha|^2.
        let alpha = C64::new(2.0, 0.0);
        let state = coherent_state(alpha, 50);

        let mean = state.photon_number_expectation();
        let var = state.photon_number_variance();
        let expected = alpha.norm_sqr(); // |alpha|^2 = 4.0

        assert!(
            (mean - expected).abs() < 0.01,
            "Coherent state <n> should be {}, got {}",
            expected,
            mean
        );
        assert!(
            (var - expected).abs() < 0.05,
            "Coherent state Var(n) should be {}, got {}",
            expected,
            var
        );
    }

    #[test]
    fn test_coherent_state_self_fidelity() {
        let alpha = C64::new(1.5, 0.5);
        let state = coherent_state(alpha, N_MAX);
        let fid = state.fidelity(&state);
        assert!(
            (fid - 1.0).abs() < TOL,
            "Self-fidelity should be 1.0, got {}",
            fid
        );
    }

    // ----------------------------------------------------------
    // Displacement operator
    // ----------------------------------------------------------

    #[test]
    fn test_displacement_creates_coherent_from_vacuum() {
        // D(alpha)|0> = |alpha>
        let alpha = C64::new(1.0, 0.5);
        let vac = FockState::new(N_MAX);
        let displaced = BosonicOperator::displacement(&vac, alpha);

        let direct = coherent_state(alpha, N_MAX);

        let fid = displaced.fidelity(&direct);
        assert!(
            fid > 0.99,
            "D(alpha)|0> should be |alpha>, fidelity = {}",
            fid
        );
    }

    // ----------------------------------------------------------
    // Rotation operator
    // ----------------------------------------------------------

    #[test]
    fn test_rotation_full_cycle() {
        // R(2*pi)|n> = |n> (each component picks up e^{-i*2*pi*n} = 1).
        let state = FockState::from_number(7, N_MAX);
        let rotated = BosonicOperator::rotation(&state, 2.0 * PI);

        let fid = state.fidelity(&rotated);
        assert!(
            (fid - 1.0).abs() < TOL,
            "R(2*pi)|n> should equal |n>, fidelity = {}",
            fid
        );
    }

    // ----------------------------------------------------------
    // Parity operator
    // ----------------------------------------------------------

    #[test]
    fn test_parity_operator() {
        // P|n> = (-1)^n |n>
        for n in 0..=10 {
            let state = FockState::from_number(n, N_MAX);
            let parity = BosonicOperator::parity_operator(&state);

            let expected_sign: f64 = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                (parity.amplitudes[n].re - expected_sign).abs() < TOL,
                "P|{}> should give ({}) * |{}>, got {}",
                n,
                expected_sign,
                n,
                parity.amplitudes[n].re
            );
        }
    }

    // ----------------------------------------------------------
    // Cat code
    // ----------------------------------------------------------

    #[test]
    fn test_cat_logical_zero_even_photons_only() {
        let cat = CatCode::new(2.0, 40);
        let zero_l = cat.logical_zero();
        let probs = zero_l.probabilities();

        // Odd photon number probabilities should be zero.
        for n in (1..=40).step_by(2) {
            assert!(
                probs[n] < 1e-10,
                "|0_L> should have no odd photon components, P({}) = {}",
                n,
                probs[n]
            );
        }
    }

    #[test]
    fn test_cat_logical_one_odd_photons_only() {
        let cat = CatCode::new(2.0, 40);
        let one_l = cat.logical_one();
        let probs = one_l.probabilities();

        // Even photon number probabilities should be zero.
        for n in (0..=40).step_by(2) {
            assert!(
                probs[n] < 1e-10,
                "|1_L> should have no even photon components, P({}) = {}",
                n,
                probs[n]
            );
        }
    }

    #[test]
    fn test_cat_logical_states_orthogonal() {
        let cat = CatCode::new(2.0, 40);
        let zero_l = cat.logical_zero();
        let one_l = cat.logical_one();

        let overlap = zero_l.fidelity(&one_l);
        assert!(
            overlap < 1e-10,
            "Cat logical states should be orthogonal, fidelity = {}",
            overlap
        );
    }

    #[test]
    fn test_cat_bit_flip_decreases_with_alpha() {
        // Bit-flip rate = exp(-2*alpha^2) should decrease with alpha.
        let bf_small = CatCode::new(1.0, N_MAX).bit_flip_rate();
        let bf_large = CatCode::new(3.0, N_MAX).bit_flip_rate();

        assert!(
            bf_large < bf_small,
            "Bit-flip rate should decrease with alpha: bf(1.0)={} vs bf(3.0)={}",
            bf_small,
            bf_large
        );
    }

    #[test]
    fn test_cat_noise_bias_increases_with_alpha() {
        let kappa = 0.01;
        let bias_small = CatCode::new(1.0, N_MAX).noise_bias(kappa);
        let bias_large = CatCode::new(3.0, N_MAX).noise_bias(kappa);

        assert!(
            bias_large > bias_small,
            "Noise bias should increase with alpha: bias(1.0)={} vs bias(3.0)={}",
            bias_small,
            bias_large
        );
    }

    // ----------------------------------------------------------
    // GKP code
    // ----------------------------------------------------------

    #[test]
    fn test_gkp_logical_states_orthogonal() {
        let gkp = GKPCode::new(0.4, 40);
        let zero_l = gkp.logical_zero();
        let one_l = gkp.logical_one();

        let overlap = zero_l.fidelity(&one_l);
        assert!(
            overlap < 0.05,
            "GKP logical states should be approximately orthogonal, fidelity = {}",
            overlap
        );
    }

    #[test]
    fn test_gkp_syndrome_measurement() {
        // For a clean logical state, the syndrome should be near zero.
        let gkp = GKPCode::new(0.4, 40);
        let zero_l = gkp.logical_zero();
        let (sq, sp) = gkp.syndrome_measurement(&zero_l);

        // Syndrome should be small for an ideal code state.
        // Due to finite truncation, we allow a generous tolerance.
        assert!(
            sq.abs() < 1.0,
            "Position syndrome of |0_L> should be small, got {}",
            sq
        );
        assert!(
            sp.abs() < 1.0,
            "Momentum syndrome of |0_L> should be small, got {}",
            sp
        );
    }

    // ----------------------------------------------------------
    // Binomial code
    // ----------------------------------------------------------

    #[test]
    fn test_binomial_logical_states_orthogonal() {
        let binom = BinomialCode::new(2, 30);
        let zero_l = binom.logical_zero();
        let one_l = binom.logical_one();

        let overlap = zero_l.fidelity(&one_l);
        assert!(
            overlap < 1e-10,
            "Binomial logical states should be orthogonal, fidelity = {}",
            overlap
        );
    }

    #[test]
    fn test_binomial_photon_number_spacing() {
        // For S=2, |0_L> should have support at n=0, 4, ... and
        // |1_L> should have support at n=2, 6, ...
        let binom = BinomialCode::new(2, 20);
        let zero_l = binom.logical_zero();
        let one_l = binom.logical_one();

        let probs_0 = zero_l.probabilities();
        let probs_1 = one_l.probabilities();

        // |0_L> for S=2: support at 0, 4, 8, ...
        // |1_L> for S=2: support at 2, 6, 10, ...
        // Check that |0_L> has support at n=0 and n=4.
        assert!(
            probs_0[0] > 0.01,
            "|0_L> should have support at n=0, P(0) = {}",
            probs_0[0]
        );

        // Check that |1_L> has support at n=2.
        assert!(
            probs_1[2] > 0.01,
            "|1_L> should have support at n=2, P(2) = {}",
            probs_1[2]
        );
    }

    // ----------------------------------------------------------
    // Noise channels
    // ----------------------------------------------------------

    #[test]
    fn test_photon_loss_reduces_mean_photon_number() {
        let alpha = C64::new(3.0, 0.0);
        let state = coherent_state(alpha, N_MAX);
        let mean_before = state.photon_number_expectation();

        let after = BosonicChannel::photon_loss(&state, 0.3);
        let mean_after = after.photon_number_expectation();

        assert!(
            mean_after < mean_before,
            "Photon loss should reduce <n>: before={}, after={}",
            mean_before,
            mean_after
        );
    }

    // ----------------------------------------------------------
    // Wigner function
    // ----------------------------------------------------------

    #[test]
    fn test_wigner_vacuum_is_gaussian() {
        let vac = FockState::new(10);
        let w_origin = WignerFunction::compute_point(&vac, 0.0, 0.0);
        let w_away = WignerFunction::compute_point(&vac, 2.0, 2.0);

        // Vacuum Wigner function peaks at origin: W(0,0) = 1/pi.
        assert!(
            (w_origin - 1.0 / PI).abs() < 0.01,
            "Vacuum W(0,0) should be 1/pi = {}, got {}",
            1.0 / PI,
            w_origin
        );

        // Should decay away from origin.
        assert!(
            w_away < w_origin,
            "Vacuum Wigner should decay: W(0,0)={} > W(2,2)={}",
            w_origin,
            w_away
        );
    }

    // ----------------------------------------------------------
    // Bosonic simulator
    // ----------------------------------------------------------

    #[test]
    fn test_simulator_cat_logical_operations() {
        let mut sim = BosonicSimulator::new(BosonicCodeType::Cat(2.0), 40);

        // Start in |0_L>.
        let f0 = sim.fidelity_with_codespace();
        assert!(
            f0 > 0.99,
            "Initial state should be in code space, fidelity = {}",
            f0
        );

        // Measure should give |0_L> (false).
        assert!(
            !sim.measure_logical(),
            "Measurement of |0_L> should return false"
        );

        // Apply logical X to get |1_L>.
        sim.logical_x();
        assert!(
            sim.measure_logical(),
            "After X_L, measurement should return true (|1_L>)"
        );

        // Apply logical Z: should not change the Z-basis outcome.
        // Z|1_L> = -|1_L>, still measures as |1_L>.
        sim.logical_z();
        assert!(
            sim.measure_logical(),
            "Z_L should not change Z-basis measurement outcome"
        );
    }

    #[test]
    fn test_simulator_binomial_codespace_fidelity() {
        let sim = BosonicSimulator::new(BosonicCodeType::Binomial(2), 20);
        let f = sim.fidelity_with_codespace();
        assert!(
            f > 0.99,
            "Initial binomial state should be in code space, fidelity = {}",
            f
        );
    }

    // ----------------------------------------------------------
    // Utility functions
    // ----------------------------------------------------------

    #[test]
    fn test_mandel_q_coherent_state() {
        // Coherent state should have Q = 0.
        let alpha = C64::new(2.0, 0.0);
        let state = coherent_state(alpha, 50);
        let q = mandel_q(&state);
        assert!(
            q.abs() < 0.05,
            "Coherent state Mandel Q should be ~0, got {}",
            q
        );
    }

    #[test]
    fn test_g2_coherent_state() {
        // Coherent state should have g^(2)(0) = 1.
        let alpha = C64::new(2.0, 0.0);
        let state = coherent_state(alpha, 50);
        let g2 = g2_zero(&state);
        assert!(
            (g2 - 1.0).abs() < 0.05,
            "Coherent state g^(2)(0) should be ~1.0, got {}",
            g2
        );
    }

    #[test]
    fn test_fock_state_sub_poissonian() {
        // Fock state |n> has Var(n) = 0, so Q = -1 (maximally sub-Poissonian).
        let state = FockState::from_number(5, N_MAX);
        let q = mandel_q(&state);
        assert!(
            (q - (-1.0)).abs() < TOL,
            "Fock state Mandel Q should be -1, got {}",
            q
        );
    }
}
