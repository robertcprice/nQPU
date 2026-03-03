//! Real-Time Bayesian Noise Adaptation
//!
//! Update noise model parameters during simulation using measurement outcomes.
//! No other quantum simulator provides real-time Bayesian noise adaptation;
//! this module enables the estimator to learn hardware noise characteristics
//! on-the-fly from mid-circuit measurement results.
//!
//! # Theory
//!
//! We maintain conjugate prior distributions over noise parameters:
//! - **Beta(alpha, beta)** for rate parameters (depolarizing, readout errors, crosstalk)
//!   because Beta is conjugate to the Binomial likelihood.
//! - **LogNormal(mu, sigma)** for time parameters (T1, T2) which are positive and
//!   right-skewed.
//!
//! Each measurement outcome triggers a closed-form Bayesian update of the
//! relevant sufficient statistics, yielding a posterior that remains in the
//! same distributional family. The MAP (maximum a posteriori) estimate is
//! recomputed periodically and fed back into the simulator's noise channels.
//!
//! # Reference
//!
//! Based on: Phys. Rev. A **111**, 062609 (2025).

use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during Bayesian noise estimation.
#[derive(Debug, Clone)]
pub enum BayesianNoiseError {
    /// A noise parameter is outside its valid physical range.
    InvalidParameter(String),
    /// The requested credible interval level is not in (0, 1).
    InvalidCredibleLevel(f64),
    /// The requested parameter name is not recognised.
    UnknownParameter(String),
    /// No observations have been collected yet.
    InsufficientData(String),
}

impl fmt::Display for BayesianNoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BayesianNoiseError::InvalidParameter(msg) => {
                write!(f, "Invalid noise parameter: {}", msg)
            }
            BayesianNoiseError::InvalidCredibleLevel(level) => {
                write!(
                    f,
                    "Credible level {} is not in the open interval (0, 1)",
                    level
                )
            }
            BayesianNoiseError::UnknownParameter(name) => {
                write!(f, "Unknown noise parameter: {}", name)
            }
            BayesianNoiseError::InsufficientData(msg) => {
                write!(f, "Insufficient data for estimation: {}", msg)
            }
        }
    }
}

impl std::error::Error for BayesianNoiseError {}

// ============================================================
// NOISE PARAMETERS
// ============================================================

/// Point estimates for the physical noise parameters of a qubit.
///
/// These are the quantities being estimated by the Bayesian procedure.
#[derive(Clone, Debug)]
pub struct NoiseParameters {
    /// Depolarizing channel strength p_depol in \[0, 1\].
    /// The channel acts as rho -> (1 - p) rho + p I/d.
    pub depolarizing_rate: f64,
    /// Amplitude damping time T1 in microseconds.
    pub t1: f64,
    /// Pure dephasing time T2 in microseconds (T2 <= 2 T1 physically).
    pub t2: f64,
    /// Readout error P(1 | prepared 0).
    pub readout_error_0: f64,
    /// Readout error P(0 | prepared 1).
    pub readout_error_1: f64,
    /// ZZ crosstalk coupling strength in \[0, 1\].
    pub crosstalk: f64,
}

impl Default for NoiseParameters {
    fn default() -> Self {
        Self {
            depolarizing_rate: 1e-3,
            t1: 100.0,
            t2: 80.0,
            readout_error_0: 0.01,
            readout_error_1: 0.02,
            crosstalk: 1e-4,
        }
    }
}

// ============================================================
// PRIOR DISTRIBUTION
// ============================================================

/// Hyper-parameters for the prior distributions over noise parameters.
///
/// Rate parameters (depolarizing, readout errors, crosstalk) use
/// Beta(alpha, beta) priors. Time parameters (T1, T2) use
/// LogNormal(mu, sigma) priors.
#[derive(Clone, Debug)]
pub struct NoisePrior {
    /// Beta distribution alpha for the depolarizing rate.
    pub depol_alpha: f64,
    /// Beta distribution beta for the depolarizing rate.
    pub depol_beta: f64,
    /// Beta alpha for P(1|0) readout error.
    pub readout_0_alpha: f64,
    /// Beta beta for P(1|0) readout error.
    pub readout_0_beta: f64,
    /// Beta alpha for P(0|1) readout error.
    pub readout_1_alpha: f64,
    /// Beta beta for P(0|1) readout error.
    pub readout_1_beta: f64,
    /// LogNormal mu for T1 (log-microseconds).
    pub t1_mu: f64,
    /// LogNormal sigma for T1.
    pub t1_sigma: f64,
    /// LogNormal mu for T2 (log-microseconds).
    pub t2_mu: f64,
    /// LogNormal sigma for T2.
    pub t2_sigma: f64,
    /// Beta alpha for crosstalk strength.
    pub crosstalk_alpha: f64,
    /// Beta beta for crosstalk strength.
    pub crosstalk_beta: f64,
}

impl Default for NoisePrior {
    /// Weakly informative priors for typical superconducting qubit hardware.
    ///
    /// - Depolarizing: Beta(1, 99) => mean ~ 0.01, weakly informative.
    /// - Readout 0: Beta(1, 99) => mean ~ 0.01.
    /// - Readout 1: Beta(2, 98) => mean ~ 0.02.
    /// - T1: LogNormal(ln(100), 0.5) => median ~ 100 us.
    /// - T2: LogNormal(ln(80), 0.5) => median ~ 80 us.
    /// - Crosstalk: Beta(1, 999) => mean ~ 0.001.
    fn default() -> Self {
        Self {
            depol_alpha: 1.0,
            depol_beta: 99.0,
            readout_0_alpha: 1.0,
            readout_0_beta: 99.0,
            readout_1_alpha: 2.0,
            readout_1_beta: 98.0,
            t1_mu: 100.0_f64.ln(),
            t1_sigma: 0.5,
            t2_mu: 80.0_f64.ln(),
            t2_sigma: 0.5,
            crosstalk_alpha: 1.0,
            crosstalk_beta: 999.0,
        }
    }
}

// ============================================================
// BETA DISTRIBUTION HELPERS
// ============================================================

/// Mean of Beta(alpha, beta).
#[inline]
fn beta_mean(alpha: f64, beta: f64) -> f64 {
    alpha / (alpha + beta)
}

/// Variance of Beta(alpha, beta).
#[inline]
fn beta_variance(alpha: f64, beta: f64) -> f64 {
    let s = alpha + beta;
    (alpha * beta) / (s * s * (s + 1.0))
}

/// Conjugate Bayesian update for a Beta prior with a Bernoulli observation.
///
/// If `success` is true, we observed a "1" (increment alpha).
/// If `success` is false, we observed a "0" (increment beta).
#[inline]
fn beta_update(alpha: &mut f64, beta: &mut f64, success: bool) {
    if success {
        *alpha += 1.0;
    } else {
        *beta += 1.0;
    }
}

/// Approximate credible interval for Beta(alpha, beta) using the normal
/// approximation: mode +/- z * sigma.
///
/// Returns (lower, upper) clamped to \[0, 1\].
fn beta_credible_interval(alpha: f64, beta: f64, level: f64) -> (f64, f64) {
    let mean = beta_mean(alpha, beta);
    let var = beta_variance(alpha, beta);
    let sigma = var.sqrt();
    // z-score for symmetric two-tailed interval
    let z = normal_quantile(0.5 + level / 2.0);
    let lower = (mean - z * sigma).max(0.0);
    let upper = (mean + z * sigma).min(1.0);
    (lower, upper)
}

/// Approximate quantile function for the standard normal distribution.
///
/// Uses the rational approximation from Abramowitz and Stegun (26.2.23)
/// which is accurate to about 4.5e-4 in absolute error.
fn normal_quantile(p: f64) -> f64 {
    // Handle boundary cases
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Use symmetry: for p < 0.5 compute -normal_quantile(1-p)
    if p < 0.5 {
        return -normal_quantile(1.0 - p);
    }

    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

/// LogNormal MAP estimate: exp(mu - sigma^2).
/// For mu representing ln(median), this gives a mode < median, which
/// is physically reasonable for right-skewed time distributions.
#[inline]
fn lognormal_map(mu: f64, sigma: f64) -> f64 {
    (mu - sigma * sigma).exp()
}

/// Online Bayesian update of LogNormal sufficient statistics given an
/// observation of survival probability and gate time.
///
/// For T1 estimation: survival_prob = exp(-gate_time / T1), so
/// T1_obs = -gate_time / ln(survival_prob).
/// We incorporate this as a single pseudo-observation that nudges mu
/// toward ln(T1_obs) with a learning rate proportional to the current
/// precision (1/sigma^2).
fn lognormal_update(mu: &mut f64, sigma: &mut f64, observation_ln: f64) {
    // Bayesian update using conjugate normal-normal model in log space.
    // Prior precision = 1/sigma^2, observation precision = 1 (unit noise).
    let prior_precision = 1.0 / (*sigma * *sigma);
    let obs_precision = 1.0; // treat each observation as having unit precision
    let post_precision = prior_precision + obs_precision;
    *mu = (prior_precision * *mu + obs_precision * observation_ln) / post_precision;
    *sigma = (1.0 / post_precision).sqrt();
}

// ============================================================
// POSTERIOR STATE
// ============================================================

/// Posterior state for the Bayesian noise model.
///
/// Maintains sufficient statistics for conjugate updates and caches the
/// current MAP estimate for fast retrieval by the simulator.
#[derive(Clone, Debug)]
pub struct NoisePosterior {
    /// Current hyper-parameters (updated in place from the prior).
    pub prior: NoisePrior,
    /// Total number of observations incorporated.
    pub num_observations: usize,
    /// Cached MAP (maximum a posteriori) point estimate.
    pub map_estimate: NoiseParameters,
    /// Confidence metric in \[0, 1\]: 0 = prior only, 1 = highly confident.
    pub confidence: f64,
}

impl NoisePosterior {
    /// Create a new posterior initialised to the given prior.
    pub fn new(prior: NoisePrior) -> Self {
        let map_estimate = NoiseParameters {
            depolarizing_rate: beta_mean(prior.depol_alpha, prior.depol_beta),
            t1: lognormal_map(prior.t1_mu, prior.t1_sigma),
            t2: lognormal_map(prior.t2_mu, prior.t2_sigma),
            readout_error_0: beta_mean(prior.readout_0_alpha, prior.readout_0_beta),
            readout_error_1: beta_mean(prior.readout_1_alpha, prior.readout_1_beta),
            crosstalk: beta_mean(prior.crosstalk_alpha, prior.crosstalk_beta),
        };
        Self {
            prior,
            num_observations: 0,
            map_estimate,
            confidence: 0.0,
        }
    }

    /// Current MAP point estimate of all noise parameters.
    pub fn map_estimate(&self) -> &NoiseParameters {
        &self.map_estimate
    }

    /// Update the depolarizing rate estimate from a circuit outcome.
    ///
    /// If `outcome != expected` under noiseless simulation, we interpret this
    /// as evidence of a depolarizing error (success for the Beta prior on
    /// the depolarizing rate).
    pub fn update_depolarizing(&mut self, outcome: bool, expected: bool) {
        let error_observed = outcome != expected;
        beta_update(
            &mut self.prior.depol_alpha,
            &mut self.prior.depol_beta,
            error_observed,
        );
        self.num_observations += 1;
        self.refresh_map();
    }

    /// Update readout error estimates.
    ///
    /// `prepared_state`: the state that was prepared (false = |0>, true = |1>).
    /// `measured`: the measurement result obtained.
    pub fn update_readout(&mut self, prepared_state: bool, measured: bool) {
        if !prepared_state {
            // Prepared |0>: error = measured 1
            beta_update(
                &mut self.prior.readout_0_alpha,
                &mut self.prior.readout_0_beta,
                measured, // success (alpha++) if measured=1 (error), beta++ if measured=0 (correct)
            );
        } else {
            // Prepared |1>: error = measured 0
            beta_update(
                &mut self.prior.readout_1_alpha,
                &mut self.prior.readout_1_beta,
                !measured, // success (alpha++) if measured=0 (error), beta++ if measured=1 (correct)
            );
        }
        self.num_observations += 1;
        self.refresh_map();
    }

    /// Update T1 estimate from an amplitude damping observation.
    ///
    /// `survival_probability`: estimated probability that |1> survived
    /// (from repeated measurements or a single shot: 1.0 if survived, 0.0 if decayed).
    /// `gate_time_us`: the idle/gate time in microseconds.
    pub fn update_t1(&mut self, survival_probability: f64, gate_time_us: f64) {
        if gate_time_us <= 0.0 || survival_probability <= 0.0 || survival_probability > 1.0 {
            return; // invalid observation, skip
        }
        // survival_prob = exp(-t / T1) => T1 = -t / ln(survival_prob)
        // Clamp survival_probability away from 0 to avoid ln(0)
        let sp = survival_probability.max(1e-10);
        let t1_obs = -gate_time_us / sp.ln();
        if t1_obs > 0.0 && t1_obs.is_finite() {
            lognormal_update(&mut self.prior.t1_mu, &mut self.prior.t1_sigma, t1_obs.ln());
            self.num_observations += 1;
            self.refresh_map();
        }
    }

    /// Update T2 estimate from a Ramsey/echo coherence measurement.
    ///
    /// `coherence_signal`: the measured coherence (e.g., expectation value of X
    /// after a Ramsey sequence), in \[0, 1\].
    /// `idle_time_us`: the total idle time in microseconds.
    pub fn update_t2(&mut self, coherence_signal: f64, idle_time_us: f64) {
        if idle_time_us <= 0.0 || coherence_signal <= 0.0 || coherence_signal > 1.0 {
            return; // invalid observation, skip
        }
        // coherence = exp(-t / T2) => T2 = -t / ln(coherence)
        let cs = coherence_signal.max(1e-10);
        let t2_obs = -idle_time_us / cs.ln();
        if t2_obs > 0.0 && t2_obs.is_finite() {
            lognormal_update(&mut self.prior.t2_mu, &mut self.prior.t2_sigma, t2_obs.ln());
            self.num_observations += 1;
            self.refresh_map();
        }
    }

    /// Update crosstalk estimate from correlated two-qubit measurements.
    ///
    /// `qubit_a_outcome` and `qubit_b_outcome`: measurement results on two
    /// neighbouring qubits. `expected_correlation`: the expected correlation
    /// under zero crosstalk (typically 0.0 for independent qubits).
    pub fn update_crosstalk(
        &mut self,
        qubit_a_outcome: bool,
        qubit_b_outcome: bool,
        expected_correlation: f64,
    ) {
        // If both qubits agree (both 0 or both 1) when they should be
        // uncorrelated, that is evidence of crosstalk.
        let observed_correlation = if qubit_a_outcome == qubit_b_outcome {
            1.0
        } else {
            0.0
        };
        let anomalous = (observed_correlation - expected_correlation).abs() > 0.5;
        beta_update(
            &mut self.prior.crosstalk_alpha,
            &mut self.prior.crosstalk_beta,
            anomalous,
        );
        self.num_observations += 1;
        self.refresh_map();
    }

    /// Overall confidence in the current estimates.
    ///
    /// Computed as 1 - 1/(1 + n/10) where n is the observation count,
    /// giving a sigmoid-like ramp from 0 to 1.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Approximate credible interval for a named parameter at the given
    /// probability level (e.g., 0.95 for a 95% credible interval).
    ///
    /// Returns `(lower, upper)`.
    pub fn credible_interval(
        &self,
        param: &str,
        level: f64,
    ) -> Result<(f64, f64), BayesianNoiseError> {
        if level <= 0.0 || level >= 1.0 {
            return Err(BayesianNoiseError::InvalidCredibleLevel(level));
        }
        match param {
            "depolarizing_rate" => Ok(beta_credible_interval(
                self.prior.depol_alpha,
                self.prior.depol_beta,
                level,
            )),
            "readout_error_0" => Ok(beta_credible_interval(
                self.prior.readout_0_alpha,
                self.prior.readout_0_beta,
                level,
            )),
            "readout_error_1" => Ok(beta_credible_interval(
                self.prior.readout_1_alpha,
                self.prior.readout_1_beta,
                level,
            )),
            "crosstalk" => Ok(beta_credible_interval(
                self.prior.crosstalk_alpha,
                self.prior.crosstalk_beta,
                level,
            )),
            "t1" => {
                // Normal approximation in log-space, then exponentiate
                let z = normal_quantile(0.5 + level / 2.0);
                let lower = (self.prior.t1_mu - z * self.prior.t1_sigma).exp();
                let upper = (self.prior.t1_mu + z * self.prior.t1_sigma).exp();
                Ok((lower, upper))
            }
            "t2" => {
                let z = normal_quantile(0.5 + level / 2.0);
                let lower = (self.prior.t2_mu - z * self.prior.t2_sigma).exp();
                let upper = (self.prior.t2_mu + z * self.prior.t2_sigma).exp();
                Ok((lower, upper))
            }
            _ => Err(BayesianNoiseError::UnknownParameter(param.to_string())),
        }
    }

    // ----------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------

    /// Recompute the MAP estimate and confidence from current sufficient statistics.
    fn refresh_map(&mut self) {
        self.map_estimate.depolarizing_rate =
            beta_mean(self.prior.depol_alpha, self.prior.depol_beta);
        self.map_estimate.readout_error_0 =
            beta_mean(self.prior.readout_0_alpha, self.prior.readout_0_beta);
        self.map_estimate.readout_error_1 =
            beta_mean(self.prior.readout_1_alpha, self.prior.readout_1_beta);
        self.map_estimate.crosstalk =
            beta_mean(self.prior.crosstalk_alpha, self.prior.crosstalk_beta);
        self.map_estimate.t1 = lognormal_map(self.prior.t1_mu, self.prior.t1_sigma);
        self.map_estimate.t2 = lognormal_map(self.prior.t2_mu, self.prior.t2_sigma);
        // Sigmoid-like confidence: approaches 1 as observations grow
        self.confidence = 1.0 - 1.0 / (1.0 + self.num_observations as f64 / 10.0);
    }
}

// ============================================================
// FLAG GADGETS
// ============================================================

/// The type of mid-circuit characterisation circuit inserted as a "flag".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlagCircuitType {
    /// Prepare |+>, idle, measure in X basis.
    /// Detects depolarizing noise (ideal outcome: 0 in X basis).
    DepolarizingTest,
    /// Prepare |0> or |1>, measure immediately.
    /// Calibrates readout errors.
    ReadoutCalibration,
    /// Prepare |1>, wait for a gate time, measure in Z basis.
    /// Estimates T1 amplitude damping.
    T1Measurement,
    /// Ramsey sequence: H - idle - H - measure.
    /// Estimates T2 dephasing.
    T2Ramsey,
}

/// A flag gadget is a lightweight mid-circuit characterisation sequence.
///
/// During simulation, flag gadgets can be inserted at strategic points to
/// collect evidence about the noise environment. The overhead is controlled
/// by `BayesianNoiseConfig::max_flag_fraction`.
#[derive(Clone, Debug)]
pub struct FlagGadget {
    /// Which characterisation circuit to run.
    pub circuit_type: FlagCircuitType,
    /// The data qubit being characterised.
    pub data_qubit: usize,
    /// An ancilla qubit used by the flag (may equal data_qubit for simple tests).
    pub flag_qubit: usize,
}

impl FlagGadget {
    /// Create a new flag gadget.
    pub fn new(circuit_type: FlagCircuitType, data_qubit: usize, flag_qubit: usize) -> Self {
        Self {
            circuit_type,
            data_qubit,
            flag_qubit,
        }
    }

    /// The measurement outcome expected in the absence of any noise.
    pub fn expected_ideal_outcome(&self) -> bool {
        match self.circuit_type {
            // |+> measured in X basis => 0 (positive eigenvalue)
            FlagCircuitType::DepolarizingTest => false,
            // Readout calibration: depends on prepared state, but the
            // "flag qubit" label encodes the prepared bit. We use a convention
            // that flag_qubit index parity encodes preparation:
            // even => prepare |0> => expect 0
            // odd  => prepare |1> => expect 1
            FlagCircuitType::ReadoutCalibration => self.flag_qubit % 2 == 1,
            // Prepare |1>, no noise => measure 1
            FlagCircuitType::T1Measurement => true,
            // Ramsey: H|0> = |+>, idle (no dephasing), H|+> = |0> => measure 0
            FlagCircuitType::T2Ramsey => false,
        }
    }

    /// Interpret a flag measurement outcome and update the posterior.
    pub fn interpret_outcome(&self, outcome: bool, posterior: &mut NoisePosterior) {
        match self.circuit_type {
            FlagCircuitType::DepolarizingTest => {
                let expected = self.expected_ideal_outcome();
                posterior.update_depolarizing(outcome, expected);
            }
            FlagCircuitType::ReadoutCalibration => {
                let prepared = self.flag_qubit % 2 == 1;
                posterior.update_readout(prepared, outcome);
            }
            FlagCircuitType::T1Measurement => {
                // outcome = true means |1> survived.
                // Single-shot survival probability estimate: 1.0 or 0.0
                let survival = if outcome { 1.0 } else { 0.0 };
                // Assume a standard gate time of 1.0 us for the idle period
                posterior.update_t1(survival, 1.0);
            }
            FlagCircuitType::T2Ramsey => {
                // outcome = false means coherence maintained (|0> measured)
                // outcome = true means dephasing occurred
                let coherence = if !outcome { 1.0 } else { 0.0 };
                // Assume standard Ramsey wait of 1.0 us
                posterior.update_t2(coherence, 1.0);
            }
        }
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the adaptive Bayesian noise estimator.
#[derive(Clone, Debug)]
pub struct BayesianNoiseConfig {
    /// Maximum fraction of total operations that may be flag gadgets.
    /// Default: 0.1 (10%).
    pub max_flag_fraction: f64,
    /// Recompute the MAP estimate every this many observations.
    /// Default: 10.
    pub update_interval: usize,
    /// Stop inserting flags when overall confidence exceeds this threshold.
    /// Default: 0.9.
    pub convergence_threshold: f64,
    /// The prior distribution to start from.
    pub initial_prior: NoisePrior,
    /// Whether to actively estimate T1.
    pub enable_t1_estimation: bool,
    /// Whether to actively estimate T2.
    pub enable_t2_estimation: bool,
    /// Whether to actively estimate crosstalk.
    pub enable_crosstalk_estimation: bool,
}

impl Default for BayesianNoiseConfig {
    fn default() -> Self {
        Self {
            max_flag_fraction: 0.1,
            update_interval: 10,
            convergence_threshold: 0.9,
            initial_prior: NoisePrior::default(),
            enable_t1_estimation: true,
            enable_t2_estimation: true,
            enable_crosstalk_estimation: true,
        }
    }
}

impl BayesianNoiseConfig {
    /// Set the maximum flag fraction (must be in (0, 1]).
    pub fn with_max_flag_fraction(mut self, fraction: f64) -> Self {
        self.max_flag_fraction = fraction.clamp(0.001, 1.0);
        self
    }

    /// Set the MAP update interval.
    pub fn with_update_interval(mut self, interval: usize) -> Self {
        self.update_interval = interval.max(1);
        self
    }

    /// Set the convergence threshold.
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the initial prior.
    pub fn with_initial_prior(mut self, prior: NoisePrior) -> Self {
        self.initial_prior = prior;
        self
    }

    /// Enable or disable T1 estimation.
    pub fn with_t1_estimation(mut self, enable: bool) -> Self {
        self.enable_t1_estimation = enable;
        self
    }

    /// Enable or disable T2 estimation.
    pub fn with_t2_estimation(mut self, enable: bool) -> Self {
        self.enable_t2_estimation = enable;
        self
    }

    /// Enable or disable crosstalk estimation.
    pub fn with_crosstalk_estimation(mut self, enable: bool) -> Self {
        self.enable_crosstalk_estimation = enable;
        self
    }
}

// ============================================================
// CONVERGENCE REPORT
// ============================================================

/// Summary of the estimator's convergence status.
#[derive(Clone, Debug)]
pub struct ConvergenceReport {
    /// Total flag gadgets executed.
    pub total_flags: usize,
    /// Total circuit operations (flags + real operations).
    pub total_operations: usize,
    /// Fraction of operations that were flags.
    pub flag_fraction: f64,
    /// Whether the estimator has converged (confidence > threshold).
    pub converged: bool,
    /// Per-parameter uncertainty: (name, standard_deviation).
    pub parameter_uncertainties: Vec<(String, f64)>,
}

impl fmt::Display for ConvergenceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bayesian Noise Convergence Report")?;
        writeln!(f, "  total flags:      {}", self.total_flags)?;
        writeln!(f, "  total operations: {}", self.total_operations)?;
        writeln!(f, "  flag fraction:    {:.4}", self.flag_fraction)?;
        writeln!(
            f,
            "  converged:        {}",
            if self.converged { "yes" } else { "no" }
        )?;
        writeln!(f, "  parameter uncertainties:")?;
        for (name, unc) in &self.parameter_uncertainties {
            writeln!(f, "    {}: {:.6}", name, unc)?;
        }
        Ok(())
    }
}

// ============================================================
// MAIN ESTIMATOR
// ============================================================

/// Real-time Bayesian noise estimator.
///
/// Sits alongside the quantum simulator and uses flag gadgets to
/// characterise hardware noise during execution. The noise model is
/// updated in closed form after each flag measurement, and the
/// current MAP estimate can be queried at any time.
pub struct BayesianNoiseEstimator {
    config: BayesianNoiseConfig,
    posterior: NoisePosterior,
    flag_count: usize,
    total_operations: usize,
    history: Vec<NoiseParameters>,
}

impl BayesianNoiseEstimator {
    /// Create a new estimator from the given configuration.
    pub fn new(config: BayesianNoiseConfig) -> Self {
        let posterior = NoisePosterior::new(config.initial_prior.clone());
        Self {
            config,
            posterior,
            flag_count: 0,
            total_operations: 0,
            history: Vec::new(),
        }
    }

    /// Whether a flag gadget should be inserted at this point.
    ///
    /// Returns `true` if both:
    /// 1. The flag budget has not been exhausted (flag_count / total < max_flag_fraction).
    /// 2. The estimator has not yet converged (confidence < threshold).
    pub fn should_insert_flag(&self) -> bool {
        // Always allow the first flag
        if self.total_operations == 0 {
            return true;
        }
        let current_fraction = self.flag_count as f64 / self.total_operations.max(1) as f64;
        let within_budget = current_fraction < self.config.max_flag_fraction;
        let not_converged = self.posterior.confidence() < self.config.convergence_threshold;
        within_budget && not_converged
    }

    /// Choose the most informative flag gadget to insert next.
    ///
    /// Selection is based on which parameter currently has the highest
    /// uncertainty (widest credible interval relative to its mean).
    /// Requires at least one available qubit.
    pub fn next_flag(&self, available_qubits: &[usize]) -> Option<FlagGadget> {
        if available_qubits.is_empty() || !self.should_insert_flag() {
            return None;
        }

        // Compute relative uncertainty for each parameter
        let depol_var = beta_variance(self.posterior.prior.depol_alpha, self.posterior.prior.depol_beta);
        let r0_var = beta_variance(
            self.posterior.prior.readout_0_alpha,
            self.posterior.prior.readout_0_beta,
        );
        let r1_var = beta_variance(
            self.posterior.prior.readout_1_alpha,
            self.posterior.prior.readout_1_beta,
        );

        // Collect candidate (variance, circuit_type) pairs
        let mut candidates: Vec<(f64, FlagCircuitType)> = vec![
            (depol_var, FlagCircuitType::DepolarizingTest),
            (r0_var, FlagCircuitType::ReadoutCalibration),
            (r1_var, FlagCircuitType::ReadoutCalibration),
        ];

        if self.config.enable_t1_estimation {
            // Use sigma as the uncertainty proxy for lognormal
            let t1_unc = self.posterior.prior.t1_sigma;
            candidates.push((t1_unc, FlagCircuitType::T1Measurement));
        }

        if self.config.enable_t2_estimation {
            let t2_unc = self.posterior.prior.t2_sigma;
            candidates.push((t2_unc, FlagCircuitType::T2Ramsey));
        }

        // Pick the candidate with the highest uncertainty
        let (_, best_type) = candidates
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))?;

        // Pick a qubit: use the first available for the data qubit,
        // and the second (or first again) for the flag qubit.
        let data_qubit = available_qubits[0];
        let flag_qubit = if available_qubits.len() > 1 {
            available_qubits[1]
        } else {
            available_qubits[0]
        };

        Some(FlagGadget::new(best_type, data_qubit, flag_qubit))
    }

    /// Process the outcome of a flag gadget measurement.
    pub fn process_flag_outcome(&mut self, gadget: &FlagGadget, outcome: bool) {
        gadget.interpret_outcome(outcome, &mut self.posterior);
        self.flag_count += 1;
        self.total_operations += 1;

        // Record history at update intervals
        if self.posterior.num_observations % self.config.update_interval == 0 {
            self.history.push(self.posterior.map_estimate.clone());
        }
    }

    /// Process the outcome of a regular (non-flag) circuit operation.
    ///
    /// `expected`: the noiseless predicted outcome.
    /// `measured`: the actual measurement result.
    pub fn process_circuit_outcome(&mut self, expected: bool, measured: bool) {
        self.posterior.update_depolarizing(measured, expected);
        self.total_operations += 1;

        if self.posterior.num_observations % self.config.update_interval == 0 {
            self.history.push(self.posterior.map_estimate.clone());
        }
    }

    /// Current MAP noise model estimate.
    pub fn current_noise_model(&self) -> &NoiseParameters {
        self.posterior.map_estimate()
    }

    /// Access the full posterior state.
    pub fn posterior(&self) -> &NoisePosterior {
        &self.posterior
    }

    /// Generate a convergence report.
    pub fn convergence_report(&self) -> ConvergenceReport {
        let total_ops = self.total_operations.max(1);
        let flag_fraction = self.flag_count as f64 / total_ops as f64;

        let p = &self.posterior.prior;
        let parameter_uncertainties = vec![
            (
                "depolarizing_rate".to_string(),
                beta_variance(p.depol_alpha, p.depol_beta).sqrt(),
            ),
            (
                "readout_error_0".to_string(),
                beta_variance(p.readout_0_alpha, p.readout_0_beta).sqrt(),
            ),
            (
                "readout_error_1".to_string(),
                beta_variance(p.readout_1_alpha, p.readout_1_beta).sqrt(),
            ),
            ("t1".to_string(), p.t1_sigma),
            ("t2".to_string(), p.t2_sigma),
            (
                "crosstalk".to_string(),
                beta_variance(p.crosstalk_alpha, p.crosstalk_beta).sqrt(),
            ),
        ];

        ConvergenceReport {
            total_flags: self.flag_count,
            total_operations: self.total_operations,
            flag_fraction,
            converged: self.posterior.confidence() >= self.config.convergence_threshold,
            parameter_uncertainties,
        }
    }

    /// Parameter evolution history (recorded at each update interval).
    pub fn history(&self) -> &[NoiseParameters] {
        &self.history
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // ----------------------------------------------------------
    // NoisePrior tests
    // ----------------------------------------------------------

    #[test]
    fn test_default_prior_values_reasonable() {
        let prior = NoisePrior::default();
        // Depolarizing prior mean should be small (~ 0.01)
        let depol_mean = beta_mean(prior.depol_alpha, prior.depol_beta);
        assert!(depol_mean > 0.0 && depol_mean < 0.1, "depol mean = {}", depol_mean);
        // T1 median should be around 100 us
        let t1_median = prior.t1_mu.exp();
        assert!(
            (t1_median - 100.0).abs() < 1.0,
            "T1 median = {}",
            t1_median
        );
        // T2 median should be around 80 us
        let t2_median = prior.t2_mu.exp();
        assert!(
            (t2_median - 80.0).abs() < 1.0,
            "T2 median = {}",
            t2_median
        );
        // Crosstalk prior mean should be very small (~ 0.001)
        let xt_mean = beta_mean(prior.crosstalk_alpha, prior.crosstalk_beta);
        assert!(xt_mean < 0.01, "crosstalk mean = {}", xt_mean);
    }

    #[test]
    fn test_prior_readout_asymmetry() {
        let prior = NoisePrior::default();
        let r0 = beta_mean(prior.readout_0_alpha, prior.readout_0_beta);
        let r1 = beta_mean(prior.readout_1_alpha, prior.readout_1_beta);
        // P(0|1) is typically larger than P(1|0) for superconducting qubits
        assert!(r1 > r0, "readout_1 ({}) should > readout_0 ({})", r1, r0);
    }

    // ----------------------------------------------------------
    // NoisePosterior tests
    // ----------------------------------------------------------

    #[test]
    fn test_posterior_new_from_prior() {
        let prior = NoisePrior::default();
        let posterior = NoisePosterior::new(prior.clone());
        assert_eq!(posterior.num_observations, 0);
        assert!((posterior.confidence() - 0.0).abs() < 1e-10);
        // MAP should match prior mean for beta params
        let expected_depol = beta_mean(prior.depol_alpha, prior.depol_beta);
        assert!(
            (posterior.map_estimate().depolarizing_rate - expected_depol).abs() < 1e-10,
            "depol MAP = {}, expected = {}",
            posterior.map_estimate().depolarizing_rate,
            expected_depol
        );
    }

    #[test]
    fn test_posterior_update_depolarizing_increases_confidence() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        assert!(posterior.confidence() < 0.01);
        for _ in 0..50 {
            posterior.update_depolarizing(true, false); // error observed
        }
        assert!(
            posterior.confidence() > 0.5,
            "confidence = {}",
            posterior.confidence()
        );
    }

    #[test]
    fn test_posterior_update_readout_state_0() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial_r0 = posterior.map_estimate().readout_error_0;
        // Feed many correct outcomes for |0>: measured 0
        for _ in 0..100 {
            posterior.update_readout(false, false);
        }
        // Readout error P(1|0) should decrease
        assert!(
            posterior.map_estimate().readout_error_0 < initial_r0,
            "r0 after correct: {}, initial: {}",
            posterior.map_estimate().readout_error_0,
            initial_r0
        );
    }

    #[test]
    fn test_posterior_update_readout_state_1() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial_r1 = posterior.map_estimate().readout_error_1;
        // Feed many erroneous outcomes for |1>: measured 0 (error)
        for _ in 0..100 {
            posterior.update_readout(true, false); // prepared 1, measured 0 => error
        }
        // Readout error P(0|1) should increase
        assert!(
            posterior.map_estimate().readout_error_1 > initial_r1,
            "r1 after errors: {}, initial: {}",
            posterior.map_estimate().readout_error_1,
            initial_r1
        );
    }

    // ----------------------------------------------------------
    // Bayesian convergence test
    // ----------------------------------------------------------

    #[test]
    fn test_depolarizing_convergence_to_known_rate() {
        // Simulate a device with true depolarizing rate of 0.05
        let true_rate = 0.05;
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let mut rng = rand::thread_rng();

        for _ in 0..500 {
            let error_occurred: bool = rng.gen::<f64>() < true_rate;
            let expected = false;
            let outcome = if error_occurred { !expected } else { expected };
            posterior.update_depolarizing(outcome, expected);
        }

        let estimated = posterior.map_estimate().depolarizing_rate;
        assert!(
            (estimated - true_rate).abs() < 0.03,
            "estimated {}, true {}",
            estimated,
            true_rate
        );
        assert!(
            posterior.confidence() > 0.9,
            "confidence = {}",
            posterior.confidence()
        );
    }

    #[test]
    fn test_readout_convergence_to_known_rate() {
        let true_r0 = 0.03; // P(1|0)
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let mut rng = rand::thread_rng();

        for _ in 0..500 {
            let error: bool = rng.gen::<f64>() < true_r0;
            let measured = error; // prepared 0, measured 1 if error
            posterior.update_readout(false, measured);
        }

        let estimated = posterior.map_estimate().readout_error_0;
        assert!(
            (estimated - true_r0).abs() < 0.02,
            "estimated {}, true {}",
            estimated,
            true_r0
        );
    }

    // ----------------------------------------------------------
    // Beta distribution helpers
    // ----------------------------------------------------------

    #[test]
    fn test_beta_mean_uniform() {
        // Beta(1,1) = Uniform(0,1), mean = 0.5
        assert!((beta_mean(1.0, 1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_beta_mean_skewed() {
        // Beta(2, 8) => mean = 0.2
        assert!((beta_mean(2.0, 8.0) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_beta_variance_formula() {
        let alpha = 3.0;
        let beta_param = 7.0;
        let s: f64 = alpha + beta_param;
        let expected = (alpha * beta_param) / (s * s * (s + 1.0));
        assert!(
            (beta_variance(alpha, beta_param) - expected).abs() < 1e-12,
            "variance = {}, expected = {}",
            beta_variance(alpha, beta_param),
            expected
        );
    }

    #[test]
    fn test_beta_update_increments() {
        let mut a = 1.0;
        let mut b = 1.0;
        beta_update(&mut a, &mut b, true);
        assert!((a - 2.0).abs() < 1e-10);
        assert!((b - 1.0).abs() < 1e-10);
        beta_update(&mut a, &mut b, false);
        assert!((a - 2.0).abs() < 1e-10);
        assert!((b - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_credible_interval_contains_mean() {
        let alpha = 10.0;
        let beta_param = 90.0;
        let (lo, hi) = beta_credible_interval(alpha, beta_param, 0.95);
        let mean = beta_mean(alpha, beta_param);
        assert!(lo < mean && mean < hi, "lo={}, mean={}, hi={}", lo, hi, mean);
        assert!(lo >= 0.0);
        assert!(hi <= 1.0);
    }

    #[test]
    fn test_credible_interval_narrows_with_data() {
        let prior = NoisePrior::default();
        let mut posterior = NoisePosterior::new(prior);

        let (lo_before, hi_before) = posterior
            .credible_interval("depolarizing_rate", 0.95)
            .unwrap();
        let width_before = hi_before - lo_before;

        // Add many observations
        for _ in 0..200 {
            posterior.update_depolarizing(false, false); // no error
        }

        let (lo_after, hi_after) = posterior
            .credible_interval("depolarizing_rate", 0.95)
            .unwrap();
        let width_after = hi_after - lo_after;

        assert!(
            width_after < width_before,
            "after={}, before={}",
            width_after,
            width_before
        );
    }

    #[test]
    fn test_credible_interval_invalid_level() {
        let posterior = NoisePosterior::new(NoisePrior::default());
        assert!(posterior.credible_interval("depolarizing_rate", 0.0).is_err());
        assert!(posterior.credible_interval("depolarizing_rate", 1.0).is_err());
        assert!(posterior.credible_interval("depolarizing_rate", -0.1).is_err());
    }

    #[test]
    fn test_credible_interval_unknown_param() {
        let posterior = NoisePosterior::new(NoisePrior::default());
        assert!(posterior.credible_interval("nonexistent", 0.95).is_err());
    }

    // ----------------------------------------------------------
    // FlagGadget tests
    // ----------------------------------------------------------

    #[test]
    fn test_flag_gadget_depolarizing_ideal_outcome() {
        let gadget = FlagGadget::new(FlagCircuitType::DepolarizingTest, 0, 1);
        // |+> measured in X basis without noise => 0
        assert!(!gadget.expected_ideal_outcome());
    }

    #[test]
    fn test_flag_gadget_readout_calibration_prepared_0() {
        // Even flag_qubit => prepare |0> => expect 0
        let gadget = FlagGadget::new(FlagCircuitType::ReadoutCalibration, 0, 2);
        assert!(!gadget.expected_ideal_outcome());
    }

    #[test]
    fn test_flag_gadget_readout_calibration_prepared_1() {
        // Odd flag_qubit => prepare |1> => expect 1
        let gadget = FlagGadget::new(FlagCircuitType::ReadoutCalibration, 0, 3);
        assert!(gadget.expected_ideal_outcome());
    }

    #[test]
    fn test_flag_gadget_t1_ideal_outcome() {
        let gadget = FlagGadget::new(FlagCircuitType::T1Measurement, 0, 1);
        // |1> survives without noise => expect 1
        assert!(gadget.expected_ideal_outcome());
    }

    #[test]
    fn test_flag_gadget_t2_ramsey_ideal_outcome() {
        let gadget = FlagGadget::new(FlagCircuitType::T2Ramsey, 0, 1);
        // H|0>=|+>, idle (no dephasing), H|+>=|0> => expect 0
        assert!(!gadget.expected_ideal_outcome());
    }

    #[test]
    fn test_flag_gadget_interpret_depolarizing() {
        let gadget = FlagGadget::new(FlagCircuitType::DepolarizingTest, 0, 1);
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial_depol = posterior.map_estimate().depolarizing_rate;

        // Report an error (outcome=true, expected=false)
        gadget.interpret_outcome(true, &mut posterior);
        assert!(
            posterior.map_estimate().depolarizing_rate > initial_depol,
            "depol should increase after error observation"
        );
    }

    #[test]
    fn test_flag_gadget_interpret_readout() {
        let gadget = FlagGadget::new(FlagCircuitType::ReadoutCalibration, 0, 2); // even => prepare |0>
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial_r0 = posterior.map_estimate().readout_error_0;

        // Report correct outcome (prepared 0, measured 0)
        gadget.interpret_outcome(false, &mut posterior);
        // After one correct observation the error rate should slightly decrease
        assert!(
            posterior.map_estimate().readout_error_0 <= initial_r0 + 1e-6,
            "r0 should not increase after correct measurement"
        );
    }

    // ----------------------------------------------------------
    // BayesianNoiseEstimator tests
    // ----------------------------------------------------------

    #[test]
    fn test_estimator_new_defaults() {
        let est = BayesianNoiseEstimator::new(BayesianNoiseConfig::default());
        assert_eq!(est.flag_count, 0);
        assert_eq!(est.total_operations, 0);
        assert!(est.history().is_empty());
    }

    #[test]
    fn test_estimator_should_insert_flag_initially() {
        let est = BayesianNoiseEstimator::new(BayesianNoiseConfig::default());
        assert!(est.should_insert_flag());
    }

    #[test]
    fn test_estimator_respects_flag_budget() {
        let config = BayesianNoiseConfig::default()
            .with_max_flag_fraction(0.1)
            .with_convergence_threshold(0.99); // high so budget is the binding constraint

        let mut est = BayesianNoiseEstimator::new(config);

        // Insert 10 flags
        for _ in 0..10 {
            let gadget = FlagGadget::new(FlagCircuitType::DepolarizingTest, 0, 1);
            est.process_flag_outcome(&gadget, false);
        }
        // flag_count=10, total_operations=10 => fraction=1.0 >> 0.1
        // Should NOT allow more flags
        assert!(
            !est.should_insert_flag(),
            "should not insert flag when budget exceeded"
        );

        // Now add 90 regular operations to bring fraction down to 10/100 = 0.1
        for _ in 0..90 {
            est.process_circuit_outcome(false, false);
        }
        // fraction = 10/100 = 0.1, exactly at budget -- should be under
        // (strictly less than comparison)
        // After 100 total ops, the confidence from 100 observations should be high
        // but we set threshold to 0.99 which takes ~990 observations
        // confidence = 1 - 1/(1 + 100/10) = 1 - 1/11 ≈ 0.909
        // 0.909 < 0.99 so not converged, and fraction = 0.1 which is not < 0.1
        // So should still be false
        assert!(
            !est.should_insert_flag(),
            "fraction is at budget, should not insert"
        );

        // Add 1 more regular operation: fraction = 10/101 < 0.1
        est.process_circuit_outcome(false, false);
        // confidence = 1 - 1/(1 + 101/10) ≈ 0.910, still < 0.99
        assert!(
            est.should_insert_flag(),
            "fraction is now below budget and not converged"
        );
    }

    #[test]
    fn test_estimator_stops_flags_on_convergence() {
        let config = BayesianNoiseConfig::default()
            .with_max_flag_fraction(1.0) // unlimited budget
            .with_convergence_threshold(0.8);

        let mut est = BayesianNoiseEstimator::new(config);

        // Feed enough observations to exceed convergence threshold
        // confidence = 1 - 1/(1+n/10), need > 0.8 => n > 40
        for _ in 0..50 {
            est.process_circuit_outcome(false, false);
        }

        assert!(
            !est.should_insert_flag(),
            "should stop flags after convergence"
        );
    }

    #[test]
    fn test_estimator_next_flag_empty_qubits() {
        let est = BayesianNoiseEstimator::new(BayesianNoiseConfig::default());
        assert!(est.next_flag(&[]).is_none());
    }

    #[test]
    fn test_estimator_next_flag_selects_gadget() {
        let est = BayesianNoiseEstimator::new(BayesianNoiseConfig::default());
        let gadget = est.next_flag(&[0, 1]);
        assert!(gadget.is_some());
        let g = gadget.unwrap();
        assert_eq!(g.data_qubit, 0);
    }

    #[test]
    fn test_estimator_process_flag_and_convergence_report() {
        let config = BayesianNoiseConfig::default().with_update_interval(5);
        let mut est = BayesianNoiseEstimator::new(config);

        for i in 0..20 {
            let gadget = FlagGadget::new(FlagCircuitType::DepolarizingTest, 0, 1);
            est.process_flag_outcome(&gadget, i % 3 == 0);
        }

        let report = est.convergence_report();
        assert_eq!(report.total_flags, 20);
        assert_eq!(report.total_operations, 20);
        assert!((report.flag_fraction - 1.0).abs() < 1e-10);
        assert!(!report.parameter_uncertainties.is_empty());
    }

    #[test]
    fn test_estimator_history_tracking() {
        let config = BayesianNoiseConfig::default().with_update_interval(5);
        let mut est = BayesianNoiseEstimator::new(config);

        for _ in 0..25 {
            est.process_circuit_outcome(false, false);
        }

        // With update_interval=5 and 25 observations, we should get
        // history entries at observations 5, 10, 15, 20, 25
        assert!(
            est.history().len() >= 4,
            "history len = {}",
            est.history().len()
        );
    }

    // ----------------------------------------------------------
    // Crosstalk estimation
    // ----------------------------------------------------------

    #[test]
    fn test_crosstalk_update_increases_on_correlated_outcomes() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial_xt = posterior.map_estimate().crosstalk;

        // Both qubits always agree (correlated) when expected to be independent
        for _ in 0..50 {
            posterior.update_crosstalk(true, true, 0.0);
        }

        assert!(
            posterior.map_estimate().crosstalk > initial_xt,
            "crosstalk should increase: {} vs {}",
            posterior.map_estimate().crosstalk,
            initial_xt
        );
    }

    #[test]
    fn test_crosstalk_decreases_on_independent_outcomes() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial_xt = posterior.map_estimate().crosstalk;

        // Outcomes are anti-correlated (independent behaviour)
        for _ in 0..50 {
            posterior.update_crosstalk(true, false, 0.0);
        }

        assert!(
            posterior.map_estimate().crosstalk < initial_xt,
            "crosstalk should decrease: {} vs {}",
            posterior.map_estimate().crosstalk,
            initial_xt
        );
    }

    // ----------------------------------------------------------
    // Edge cases
    // ----------------------------------------------------------

    #[test]
    fn test_zero_observations_confidence() {
        let posterior = NoisePosterior::new(NoisePrior::default());
        assert!((posterior.confidence() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_same_outcomes_depolarizing() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        // All outcomes match expected => zero observed error rate
        for _ in 0..200 {
            posterior.update_depolarizing(false, false);
        }
        // Depolarizing rate should be very low
        assert!(
            posterior.map_estimate().depolarizing_rate < 0.01,
            "depol = {}",
            posterior.map_estimate().depolarizing_rate
        );
    }

    #[test]
    fn test_all_errors_depolarizing() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        // All outcomes differ from expected
        for _ in 0..200 {
            posterior.update_depolarizing(true, false);
        }
        // Depolarizing rate should be very high
        assert!(
            posterior.map_estimate().depolarizing_rate > 0.5,
            "depol = {}",
            posterior.map_estimate().depolarizing_rate
        );
    }

    #[test]
    fn test_t1_update_with_survival() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial_t1 = posterior.map_estimate().t1;

        // Many survival observations with short gate time => T1 should be estimated large
        for _ in 0..50 {
            posterior.update_t1(0.99, 1.0); // 99% survival in 1 us => T1 ~ 100 us
        }

        // T1 estimate should be reasonable (> 50 us)
        assert!(
            posterior.map_estimate().t1 > 50.0,
            "T1 = {}",
            posterior.map_estimate().t1
        );
    }

    #[test]
    fn test_t1_update_with_invalid_inputs() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());
        let initial = posterior.map_estimate().t1;

        // These should be silently ignored
        posterior.update_t1(0.0, 1.0); // zero survival
        posterior.update_t1(0.5, 0.0); // zero gate time
        posterior.update_t1(1.5, 1.0); // survival > 1
        posterior.update_t1(-0.1, 1.0); // negative survival

        assert!(
            (posterior.map_estimate().t1 - initial).abs() < 1e-10,
            "T1 should not change with invalid inputs"
        );
    }

    #[test]
    fn test_t2_update_with_coherence() {
        let mut posterior = NoisePosterior::new(NoisePrior::default());

        // High coherence signal with short idle time => large T2
        for _ in 0..50 {
            posterior.update_t2(0.95, 1.0); // 95% coherence in 1 us
        }

        assert!(
            posterior.map_estimate().t2 > 10.0,
            "T2 = {}",
            posterior.map_estimate().t2
        );
    }

    // ----------------------------------------------------------
    // Config builder pattern
    // ----------------------------------------------------------

    #[test]
    fn test_config_builder() {
        let config = BayesianNoiseConfig::default()
            .with_max_flag_fraction(0.2)
            .with_update_interval(20)
            .with_convergence_threshold(0.95)
            .with_t1_estimation(false)
            .with_t2_estimation(false)
            .with_crosstalk_estimation(false);

        assert!((config.max_flag_fraction - 0.2).abs() < 1e-10);
        assert_eq!(config.update_interval, 20);
        assert!((config.convergence_threshold - 0.95).abs() < 1e-10);
        assert!(!config.enable_t1_estimation);
        assert!(!config.enable_t2_estimation);
        assert!(!config.enable_crosstalk_estimation);
    }

    #[test]
    fn test_config_builder_clamps() {
        let config = BayesianNoiseConfig::default()
            .with_max_flag_fraction(5.0) // should clamp to 1.0
            .with_convergence_threshold(-0.5); // should clamp to 0.0

        assert!((config.max_flag_fraction - 1.0).abs() < 1e-10);
        assert!((config.convergence_threshold - 0.0).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // Normal quantile helper
    // ----------------------------------------------------------

    #[test]
    fn test_normal_quantile_median() {
        // Phi^{-1}(0.5) = 0.0
        assert!(normal_quantile(0.5).abs() < 1e-4);
    }

    #[test]
    fn test_normal_quantile_975() {
        // Phi^{-1}(0.975) ~ 1.96
        let z = normal_quantile(0.975);
        assert!(
            (z - 1.96).abs() < 0.01,
            "z(0.975) = {}, expected ~1.96",
            z
        );
    }

    #[test]
    fn test_normal_quantile_symmetry() {
        let z_high = normal_quantile(0.95);
        let z_low = normal_quantile(0.05);
        assert!(
            (z_high + z_low).abs() < 1e-4,
            "z(0.95)={}, z(0.05)={}, sum={}",
            z_high,
            z_low,
            z_high + z_low
        );
    }

    // ----------------------------------------------------------
    // ConvergenceReport display
    // ----------------------------------------------------------

    #[test]
    fn test_convergence_report_display() {
        let report = ConvergenceReport {
            total_flags: 10,
            total_operations: 100,
            flag_fraction: 0.1,
            converged: false,
            parameter_uncertainties: vec![
                ("depolarizing_rate".to_string(), 0.005),
            ],
        };
        let text = format!("{}", report);
        assert!(text.contains("total flags:"));
        assert!(text.contains("converged:"));
        assert!(text.contains("depolarizing_rate"));
    }

    // ----------------------------------------------------------
    // Error type coverage
    // ----------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e1 = BayesianNoiseError::InvalidParameter("negative rate".to_string());
        assert!(format!("{}", e1).contains("negative rate"));

        let e2 = BayesianNoiseError::InvalidCredibleLevel(1.5);
        assert!(format!("{}", e2).contains("1.5"));

        let e3 = BayesianNoiseError::UnknownParameter("foo".to_string());
        assert!(format!("{}", e3).contains("foo"));

        let e4 = BayesianNoiseError::InsufficientData("need more".to_string());
        assert!(format!("{}", e4).contains("need more"));

        // Verify Error trait is implemented
        let _: &dyn std::error::Error = &e1;
    }

    // ----------------------------------------------------------
    // NoiseParameters default
    // ----------------------------------------------------------

    #[test]
    fn test_noise_parameters_default() {
        let params = NoiseParameters::default();
        assert!(params.depolarizing_rate > 0.0);
        assert!(params.t1 > 0.0);
        assert!(params.t2 > 0.0);
        assert!(params.t2 <= 2.0 * params.t1); // physical constraint
        assert!(params.readout_error_0 >= 0.0 && params.readout_error_0 <= 1.0);
        assert!(params.readout_error_1 >= 0.0 && params.readout_error_1 <= 1.0);
        assert!(params.crosstalk >= 0.0);
    }
}
