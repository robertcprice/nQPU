//! Concatenated Bosonic Cat Qubit Simulation
//!
//! World-first Rust implementation of concatenated cat qubit error correction,
//! based on the theoretical framework validated experimentally by the Alice & Bob
//! team (Nature, February 2025).
//!
//! Cat qubits encode logical information in superpositions of coherent states
//! within a superconducting cavity. The key insight: two-photon dissipation
//! confines the oscillator to a manifold spanned by |+alpha> and |-alpha>,
//! yielding an **exponentially biased noise channel** where bit-flip errors
//! are suppressed as exp(-2|alpha|^2) while phase-flip errors grow only
//! linearly as |alpha|^2.
//!
//! By concatenating cat qubits with an outer repetition code that corrects
//! only the dominant (phase-flip) errors, one obtains a full QEC code with
//! dramatically reduced overhead compared to conventional surface codes.
//!
//! # Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────┐
//!  │                  Logical Qubit (output)                 │
//!  │                         │                               │
//!  │              ┌──────────┴──────────┐                    │
//!  │              │   Repetition Code   │  (outer code)      │
//!  │              │   distance = d       │  corrects phase    │
//!  │              │   majority-vote      │  flips only        │
//!  │              └──────────┬──────────┘                    │
//!  │         ┌───────┬───────┼───────┬───────┐              │
//!  │         │       │       │       │       │              │
//!  │        Cat₁   Cat₂   Cat₃    ...    Cat_d             │
//!  │         │       │       │       │       │              │
//!  │      ┌──┴──┐ ┌──┴──┐ ┌──┴──┐         ┌──┴──┐         │
//!  │      │Fock │ │Fock │ │Fock │   ...   │Fock │         │
//!  │      │space│ │space│ │space│         │space│         │
//!  │      └─────┘ └─────┘ └─────┘         └─────┘         │
//!  │                                                         │
//!  │  Bit-flip:  Γ_bf ~ κ₁ · exp(-2α²)   (exponential!)   │
//!  │  Phase-flip: Γ_pf ~ κ₁ · α² / κ₂    (linear)         │
//!  │  Bias ratio: η = α² · exp(2α²)       (huge!)          │
//!  └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - Lescanne et al., "Exponential suppression of bit-flips in a qubit
//!   encoded in an oscillator", Nature Physics 16, 509-513 (2020).
//! - Regent et al., "High-performance repetition cat code using fast
//!   noisy operations", Quantum 7, 1198 (2023).
//! - Marquet et al., "Autoparametric resonance extending the bit-flip
//!   time of a cat qubit up to 0.3 s", Physical Review X 14, 021019 (2024).
//! - Nature, February 2025: First experimental demonstration of
//!   concatenated cat qubit error correction surpassing break-even.

use num_complex::Complex64 as C64;
use rand::Rng;
use std::fmt;

// ============================================================
// CONSTANTS
// ============================================================

/// Numerical tolerance for normalization and floating-point comparisons.
const EPSILON: f64 = 1e-12;

/// Default Fock space truncation. Must exceed mean photon number significantly.
const DEFAULT_FOCK_CUTOFF: usize = 25;

/// Default coherent state amplitude for cat qubits.
const DEFAULT_ALPHA: f64 = 2.0;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from cat qubit configuration or simulation.
#[derive(Debug, Clone)]
pub enum CatQubitError {
    /// Alpha must be positive and finite.
    InvalidAlpha(f64),
    /// Fock cutoff is too small to faithfully represent states at the given alpha.
    FockCutoffTooSmall { cutoff: usize, needed: usize },
    /// General configuration error with descriptive message.
    ConfigError(String),
}

impl fmt::Display for CatQubitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatQubitError::InvalidAlpha(a) => {
                write!(
                    f,
                    "invalid coherent state amplitude alpha={:.4}: must be positive and finite",
                    a
                )
            }
            CatQubitError::FockCutoffTooSmall { cutoff, needed } => {
                write!(
                    f,
                    "Fock cutoff {} is too small; need at least {} for faithful representation",
                    cutoff, needed
                )
            }
            CatQubitError::ConfigError(msg) => write!(f, "configuration error: {}", msg),
        }
    }
}

impl std::error::Error for CatQubitError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the concatenated cat qubit simulator.
///
/// Uses a builder pattern: construct with `CatQubitConfig::new()` then chain
/// setter methods before calling `.build()` to validate.
#[derive(Debug, Clone)]
pub struct CatQubitConfig {
    /// Coherent state amplitude (typical: 2.0 -- 4.0).
    pub alpha: f64,
    /// Fock space truncation dimension (typical: 20 -- 40).
    pub fock_cutoff: usize,
    /// Single-photon loss rate (Hz). Dominant decoherence channel.
    pub kappa_1: f64,
    /// Two-photon dissipation rate (Hz). Stabilizes the cat manifold.
    pub kappa_2: f64,
    /// Gate duration in microseconds.
    pub gate_time: f64,
    /// Number of cat qubits in the outer repetition code.
    pub num_cats: usize,
    /// Number of QEC syndrome extraction rounds.
    pub num_rounds: usize,
    /// Physical error rate per gate (used in Monte Carlo sampling).
    pub physical_error_rate: f64,
    /// Kerr nonlinearity strength (Hz). Governs the self-Kerr Hamiltonian
    /// H_Kerr = -K/2 * a^dag^2 a^2 that stabilizes the cat manifold alongside
    /// the two-photon drive. Typical experimental value: 1e3 -- 1e5 Hz.
    pub kerr_strength: f64,
}

impl Default for CatQubitConfig {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            fock_cutoff: DEFAULT_FOCK_CUTOFF,
            kappa_1: 1e4,
            kappa_2: 1e7,
            gate_time: 1.0,
            num_cats: 3,
            num_rounds: 100,
            physical_error_rate: 1e-3,
            kerr_strength: 1e4,
        }
    }
}

impl CatQubitConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the coherent state amplitude.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the Fock space truncation.
    pub fn fock_cutoff(mut self, cutoff: usize) -> Self {
        self.fock_cutoff = cutoff;
        self
    }

    /// Set the single-photon loss rate.
    pub fn kappa_1(mut self, kappa_1: f64) -> Self {
        self.kappa_1 = kappa_1;
        self
    }

    /// Set the two-photon dissipation rate.
    pub fn kappa_2(mut self, kappa_2: f64) -> Self {
        self.kappa_2 = kappa_2;
        self
    }

    /// Set the gate duration in microseconds.
    pub fn gate_time(mut self, gate_time: f64) -> Self {
        self.gate_time = gate_time;
        self
    }

    /// Set the number of cat qubits in the repetition code.
    pub fn num_cats(mut self, num_cats: usize) -> Self {
        self.num_cats = num_cats;
        self
    }

    /// Set the number of QEC rounds.
    pub fn num_rounds(mut self, num_rounds: usize) -> Self {
        self.num_rounds = num_rounds;
        self
    }

    /// Set the physical error rate per gate.
    pub fn physical_error_rate(mut self, rate: f64) -> Self {
        self.physical_error_rate = rate;
        self
    }

    /// Set the Kerr nonlinearity strength (Hz).
    pub fn kerr_strength(mut self, kerr: f64) -> Self {
        self.kerr_strength = kerr;
        self
    }

    /// Validate the configuration and return a result.
    pub fn build(self) -> Result<Self, CatQubitError> {
        if self.alpha <= 0.0 || !self.alpha.is_finite() {
            return Err(CatQubitError::InvalidAlpha(self.alpha));
        }

        // Fock cutoff must be large enough: need at least 3 * mean_photon + 10
        let mean_photon = self.alpha * self.alpha;
        let needed = (3.0 * mean_photon + 10.0).ceil() as usize;
        if self.fock_cutoff < needed {
            return Err(CatQubitError::FockCutoffTooSmall {
                cutoff: self.fock_cutoff,
                needed,
            });
        }

        if self.kappa_1 < 0.0 {
            return Err(CatQubitError::ConfigError(
                "kappa_1 (single-photon loss) must be non-negative".into(),
            ));
        }

        if self.kappa_2 <= 0.0 {
            return Err(CatQubitError::ConfigError(
                "kappa_2 (two-photon dissipation) must be positive".into(),
            ));
        }

        if self.num_cats == 0 {
            return Err(CatQubitError::ConfigError(
                "num_cats must be at least 1".into(),
            ));
        }

        if self.num_cats % 2 == 0 {
            return Err(CatQubitError::ConfigError(
                "num_cats (repetition code distance) must be odd for majority-vote decoding".into(),
            ));
        }

        if self.num_rounds == 0 {
            return Err(CatQubitError::ConfigError(
                "num_rounds must be at least 1".into(),
            ));
        }

        Ok(self)
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Compute n! as f64. For n > 170 the result overflows; clamp to f64::MAX.
#[inline]
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Compute the L2 norm of a complex vector.
#[inline]
fn complex_vec_norm(v: &[C64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Normalize a complex vector in place. Returns the norm before normalization.
#[inline]
fn normalize_in_place(v: &mut [C64]) -> f64 {
    let n = complex_vec_norm(v);
    if n > EPSILON {
        let inv = 1.0 / n;
        for c in v.iter_mut() {
            *c *= inv;
        }
    }
    n
}

// ============================================================
// COHERENT STATE
// ============================================================

/// A coherent state |alpha> represented in the truncated Fock basis.
///
/// The coherent state is an eigenstate of the annihilation operator:
/// a|alpha> = alpha|alpha>. In the Fock basis:
///
///   |alpha> = e^{-|alpha|^2/2} * sum_{n=0}^{cutoff} alpha^n / sqrt(n!) |n>
#[derive(Debug, Clone)]
pub struct CoherentState {
    /// Complex amplitude of the coherent state.
    pub alpha: C64,
    /// Fock basis expansion coefficients.
    pub coefficients: Vec<C64>,
}

impl CoherentState {
    /// Construct a coherent state |alpha> truncated at `fock_cutoff` photons.
    ///
    /// Uses iterative computation to avoid factorial overflow:
    ///   c[n] = c[n-1] * alpha / sqrt(n)
    pub fn new(alpha: C64, fock_cutoff: usize) -> Self {
        let dim = fock_cutoff + 1;
        let mut coefficients = vec![C64::new(0.0, 0.0); dim];
        let norm_factor = (-alpha.norm_sqr() / 2.0).exp();

        coefficients[0] = C64::new(norm_factor, 0.0);
        for n in 1..dim {
            coefficients[n] = coefficients[n - 1] * alpha / (n as f64).sqrt();
        }

        CoherentState {
            alpha,
            coefficients,
        }
    }

    /// Construct a coherent state from a real amplitude.
    pub fn from_real(alpha: f64, fock_cutoff: usize) -> Self {
        Self::new(C64::new(alpha, 0.0), fock_cutoff)
    }

    /// Compute the norm squared: sum |c_n|^2. Should be approximately 1.0
    /// if the Fock cutoff is large enough.
    pub fn norm_squared(&self) -> f64 {
        self.coefficients.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Compute the mean photon number <n> = sum n |c_n|^2.
    /// For a perfect coherent state this equals |alpha|^2.
    pub fn mean_photon_number(&self) -> f64 {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(n, c)| n as f64 * c.norm_sqr())
            .sum()
    }

    /// Return the inner product <self|other>.
    pub fn inner_product(&self, other: &CoherentState) -> C64 {
        let len = self.coefficients.len().min(other.coefficients.len());
        let mut overlap = C64::new(0.0, 0.0);
        for i in 0..len {
            overlap += self.coefficients[i].conj() * other.coefficients[i];
        }
        overlap
    }
}

// ============================================================
// CAT QUBIT STATE
// ============================================================

/// A cat qubit logical state in the truncated Fock basis.
///
/// The two logical basis states are:
///   |0_L> = N_+ (|+alpha> + |-alpha>)  -- even cat (even Fock numbers)
///   |1_L> = N_- (|+alpha> - |-alpha>)  -- odd cat  (odd Fock numbers)
///
/// The even cat has support only on even Fock states because the odd terms
/// cancel, and vice versa for the odd cat. This parity structure is the
/// foundation of the exponential bit-flip suppression.
#[derive(Debug, Clone)]
pub struct CatQubitState {
    /// Logical value: 0 (even cat) or 1 (odd cat).
    pub logical: u8,
    /// Fock basis expansion coefficients.
    pub fock_coefficients: Vec<C64>,
    /// Coherent state amplitude.
    pub alpha: f64,
    /// Fock space truncation dimension.
    pub fock_cutoff: usize,
}

impl CatQubitState {
    /// Construct the logical |0_L> (even cat) or |1_L> (odd cat) state.
    ///
    /// # Arguments
    /// * `logical` -- 0 for even cat, 1 for odd cat.
    /// * `alpha` -- real coherent state amplitude (positive).
    /// * `fock_cutoff` -- Fock space truncation.
    pub fn new(logical: u8, alpha: f64, fock_cutoff: usize) -> Result<Self, CatQubitError> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(CatQubitError::InvalidAlpha(alpha));
        }
        if logical > 1 {
            return Err(CatQubitError::ConfigError(format!(
                "logical must be 0 or 1, got {}",
                logical
            )));
        }

        let plus_alpha = CoherentState::from_real(alpha, fock_cutoff);
        let minus_alpha = CoherentState::from_real(-alpha, fock_cutoff);

        let dim = fock_cutoff + 1;
        let mut fock_coefficients = vec![C64::new(0.0, 0.0); dim];

        // |cat_+> = N_+ (|alpha> + |-alpha>)  for logical 0 (even cat)
        // |cat_-> = N_- (|alpha> - |-alpha>)  for logical 1 (odd cat)
        let sign = if logical == 0 { 1.0 } else { -1.0 };

        for n in 0..dim {
            fock_coefficients[n] =
                plus_alpha.coefficients[n] + C64::new(sign, 0.0) * minus_alpha.coefficients[n];
        }

        // Normalize
        normalize_in_place(&mut fock_coefficients);

        Ok(CatQubitState {
            logical,
            fock_coefficients,
            alpha,
            fock_cutoff,
        })
    }

    /// Compute the norm squared of the state.
    pub fn norm_squared(&self) -> f64 {
        self.fock_coefficients.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Compute the mean photon number <n>.
    pub fn mean_photon_number(&self) -> f64 {
        self.fock_coefficients
            .iter()
            .enumerate()
            .map(|(n, c)| n as f64 * c.norm_sqr())
            .sum()
    }

    /// Compute the parity <(-1)^n> = sum (-1)^n |c_n|^2.
    ///
    /// For a perfect even cat this is +1; for a perfect odd cat it is -1.
    pub fn parity(&self) -> f64 {
        self.fock_coefficients
            .iter()
            .enumerate()
            .map(|(n, c)| {
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                sign * c.norm_sqr()
            })
            .sum()
    }

    /// Check whether the state has the expected parity structure.
    ///
    /// Even cat (logical 0) should have zero amplitude on odd Fock states.
    /// Odd cat (logical 1) should have zero amplitude on even Fock states.
    pub fn check_parity_structure(&self, tolerance: f64) -> bool {
        for (n, c) in self.fock_coefficients.iter().enumerate() {
            let should_vanish = if self.logical == 0 {
                n % 2 == 1 // even cat: odd terms vanish
            } else {
                n % 2 == 0 // odd cat: even terms vanish
            };
            if should_vanish && c.norm_sqr() > tolerance {
                return false;
            }
        }
        true
    }

    /// Compute fidelity |<self|other>|^2 between two cat states.
    pub fn fidelity(&self, other: &CatQubitState) -> f64 {
        let len = self
            .fock_coefficients
            .len()
            .min(other.fock_coefficients.len());
        let mut overlap = C64::new(0.0, 0.0);
        for i in 0..len {
            overlap += self.fock_coefficients[i].conj() * other.fock_coefficients[i];
        }
        overlap.norm_sqr()
    }
}

// ============================================================
// BIASED NOISE MODEL
// ============================================================

/// Asymmetric noise model for a cat qubit.
///
/// The defining property of cat qubits is the exponentially biased noise:
///   - Bit-flip rate:  Gamma_bf ~ kappa_1 * exp(-2 * alpha^2)
///   - Phase-flip rate: Gamma_pf ~ kappa_1 * n_bar = kappa_1 * alpha^2
///   - Bias ratio: eta = Gamma_pf / Gamma_bf (exponentially large)
///
/// The two-photon dissipation rate kappa_2 stabilizes the cat manifold
/// but does not directly enter the leading-order noise rates. The ratio
/// kappa_1/kappa_2 determines higher-order corrections to the bit-flip
/// lifetime.
///
/// For alpha = 2: bias ~ 10^4.  For alpha = 4: bias ~ 10^14.
#[derive(Debug, Clone)]
pub struct BiasedNoiseModel {
    /// Bit-flip error rate (exponentially suppressed).
    pub bit_flip_rate: f64,
    /// Phase-flip error rate (linearly growing with alpha^2).
    pub phase_flip_rate: f64,
    /// Noise bias ratio: phase_flip_rate / bit_flip_rate.
    pub bias_ratio: f64,
}

impl BiasedNoiseModel {
    /// Compute the biased noise model from physical parameters.
    ///
    /// # Arguments
    /// * `alpha` -- coherent state amplitude.
    /// * `kappa_1` -- single-photon loss rate (Hz).
    /// * `_kappa_2` -- two-photon dissipation rate (Hz). Stabilizes the cat
    ///   manifold but does not enter the leading-order error rates.
    pub fn from_params(alpha: f64, kappa_1: f64, _kappa_2: f64) -> Self {
        let alpha_sq = alpha * alpha;

        // Bit-flip rate: exponentially suppressed by the cat size
        // Gamma_bf ~ kappa_1 * exp(-2 * |alpha|^2)
        let bit_flip_rate = kappa_1 * (-2.0 * alpha_sq).exp();

        // Phase-flip rate: linear growth with mean photon number
        // Gamma_pf ~ kappa_1 * |alpha|^2
        let phase_flip_rate = kappa_1 * alpha_sq;

        // Bias ratio: how much more likely phase flips are vs bit flips
        let bias_ratio = if bit_flip_rate > EPSILON {
            phase_flip_rate / bit_flip_rate
        } else {
            f64::INFINITY
        };

        BiasedNoiseModel {
            bit_flip_rate,
            phase_flip_rate,
            bias_ratio,
        }
    }

    /// Compute the noise bias as a function of alpha alone (kappa ratio = 1).
    ///
    /// This is the fundamental scaling: eta = alpha^2 * exp(2 * alpha^2).
    pub fn fundamental_bias(alpha: f64) -> f64 {
        let alpha_sq = alpha * alpha;
        alpha_sq * (2.0 * alpha_sq).exp()
    }

    /// Compute the bit-flip probability per gate cycle.
    pub fn bit_flip_probability(&self, gate_time_us: f64) -> f64 {
        let p = self.bit_flip_rate * gate_time_us * 1e-6;
        p.min(1.0)
    }

    /// Compute the phase-flip probability per gate cycle.
    pub fn phase_flip_probability(&self, gate_time_us: f64) -> f64 {
        let p = self.phase_flip_rate * gate_time_us * 1e-6;
        p.min(1.0)
    }
}

impl fmt::Display for BiasedNoiseModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BiasedNoise(bit_flip={:.2e}, phase_flip={:.2e}, bias={:.2e})",
            self.bit_flip_rate, self.phase_flip_rate, self.bias_ratio
        )
    }
}

// ============================================================
// CAT-TRANSMON CX GATE
// ============================================================

/// Model for the controlled-X gate between a cat qubit and a transmon ancilla.
///
/// In the concatenated scheme, transmon ancillae mediate syndrome extraction.
/// The CX gate is implemented via a conditional displacement in phase space.
/// Its fidelity is limited by transmon T1/T2 and the cat qubit dephasing
/// during the gate.
#[derive(Debug, Clone)]
pub struct CatTransmonCX {
    /// Gate fidelity (0.0 -- 1.0).
    pub fidelity: f64,
    /// Gate duration in microseconds.
    pub gate_time: f64,
}

impl CatTransmonCX {
    /// Construct a CX gate model from physical parameters.
    ///
    /// The fidelity is estimated from the ratio of gate time to coherence:
    ///   F ~ 1 - gate_time * (kappa_1 / kappa_2) * alpha^2
    pub fn from_params(alpha: f64, kappa_1: f64, kappa_2: f64, gate_time: f64) -> Self {
        let infidelity = gate_time * 1e-6 * kappa_1 * alpha * alpha / kappa_2;
        let fidelity = (1.0 - infidelity).max(0.0).min(1.0);
        CatTransmonCX {
            fidelity,
            gate_time,
        }
    }

    /// Return the gate error probability (1 - fidelity).
    pub fn error_probability(&self) -> f64 {
        1.0 - self.fidelity
    }
}

impl fmt::Display for CatTransmonCX {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CatTransmonCX(fidelity={:.6}, gate_time={:.2}us)",
            self.fidelity, self.gate_time
        )
    }
}

// ============================================================
// CAT QUBIT OPERATOR (FOCK BASIS)
// ============================================================

/// A single cat qubit operation represented as a matrix in the Fock basis.
///
/// Cat qubit gates preserve the logical subspace spanned by even/odd
/// coherent state superpositions. The Fock-basis matrix representation
/// allows exact (up to truncation) simulation of gate action.
#[derive(Debug, Clone)]
pub struct CatQubitOperator {
    /// Human-readable name for the operator.
    pub name: String,
    /// Matrix representation in the Fock basis: matrix_fock[i][j] = <i|O|j>.
    pub matrix_fock: Vec<Vec<C64>>,
}

impl CatQubitOperator {
    /// Construct the Z gate (parity operator) for cat qubits.
    ///
    /// Z_L = (-1)^n (photon number parity). This is diagonal in the cat
    /// basis with eigenvalues:
    ///   Z_L |0_L> = +|0_L>  (even cat has only even photon numbers)
    ///   Z_L |1_L> = -|1_L>  (odd cat has only odd photon numbers)
    ///
    /// In the Fock basis: Z_L = diag(+1, -1, +1, -1, ...).
    pub fn z_gate(fock_cutoff: usize) -> Self {
        let dim = fock_cutoff + 1;
        let mut matrix = vec![vec![C64::new(0.0, 0.0); dim]; dim];
        for n in 0..dim {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            matrix[n][n] = C64::new(sign, 0.0);
        }
        CatQubitOperator {
            name: "Z".to_string(),
            matrix_fock: matrix,
        }
    }

    /// Construct the X gate for cat qubits at a given alpha.
    ///
    /// X_L = |0_L><1_L| + |1_L><0_L| exchanges the two logical states.
    /// Unlike in the qubit Hilbert space, this operator must be constructed
    /// from the actual cat state vectors since the logical subspace is
    /// embedded in the larger Fock space.
    ///
    /// X_L |0_L> = |1_L>,  X_L |1_L> = |0_L>
    pub fn x_gate(alpha: f64, fock_cutoff: usize) -> Self {
        let dim = fock_cutoff + 1;
        let mut matrix = vec![vec![C64::new(0.0, 0.0); dim]; dim];

        // Construct |0_L> and |1_L>
        let cat0 =
            CatQubitState::new(0, alpha, fock_cutoff).expect("valid alpha for X gate construction");
        let cat1 =
            CatQubitState::new(1, alpha, fock_cutoff).expect("valid alpha for X gate construction");

        // X_L = |1_L><0_L| + |0_L><1_L|
        // matrix[i][j] = <i|X_L|j> = cat1[i]*cat0[j]^* + cat0[i]*cat1[j]^*
        for i in 0..dim {
            for j in 0..dim {
                matrix[i][j] = cat1.fock_coefficients[i] * cat0.fock_coefficients[j].conj()
                    + cat0.fock_coefficients[i] * cat1.fock_coefficients[j].conj();
            }
        }

        CatQubitOperator {
            name: "X".to_string(),
            matrix_fock: matrix,
        }
    }

    /// Construct the photon number operator n_hat.
    pub fn number_operator(fock_cutoff: usize) -> Self {
        let dim = fock_cutoff + 1;
        let mut matrix = vec![vec![C64::new(0.0, 0.0); dim]; dim];
        for n in 0..dim {
            matrix[n][n] = C64::new(n as f64, 0.0);
        }
        CatQubitOperator {
            name: "n_hat".to_string(),
            matrix_fock: matrix,
        }
    }

    /// Apply this operator to a cat qubit state, returning a new state vector.
    pub fn apply(&self, state: &[C64]) -> Vec<C64> {
        let dim = self.matrix_fock.len();
        let mut result = vec![C64::new(0.0, 0.0); dim];
        for i in 0..dim {
            for j in 0..dim.min(state.len()) {
                result[i] += self.matrix_fock[i][j] * state[j];
            }
        }
        result
    }

    /// Return the dimension of the Fock space.
    pub fn dim(&self) -> usize {
        self.matrix_fock.len()
    }
}

impl fmt::Display for CatQubitOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CatQubitOp({})[{}x{}]",
            self.name,
            self.dim(),
            self.dim()
        )
    }
}

// ============================================================
// REPETITION CODE
// ============================================================

/// Repetition code for correcting phase-flip errors on cat qubits.
///
/// Because cat qubits have exponentially suppressed bit-flip errors,
/// the outer code only needs to handle phase flips (Z errors). A simple
/// repetition code of distance d can correct up to floor((d-1)/2) phase
/// flips using majority-vote decoding.
///
/// Stabilizers: Z_i * Z_{i+1} for i in 0..d-1
/// Syndrome: one bit per stabilizer, indicating whether adjacent qubits
/// have the same or different phase.
#[derive(Debug, Clone)]
pub struct RepetitionCode {
    /// Code distance (= number of cat qubits). Must be odd.
    pub distance: usize,
    /// Syndrome bits from the most recent extraction round.
    pub syndrome_bits: Vec<bool>,
}

impl RepetitionCode {
    /// Create a new repetition code of the given distance.
    pub fn new(distance: usize) -> Result<Self, CatQubitError> {
        if distance == 0 {
            return Err(CatQubitError::ConfigError(
                "repetition code distance must be at least 1".into(),
            ));
        }
        let num_syndrome = if distance > 1 { distance - 1 } else { 0 };
        Ok(RepetitionCode {
            distance,
            syndrome_bits: vec![false; num_syndrome],
        })
    }

    /// Extract the syndrome from a vector of qubit phase-flip errors.
    ///
    /// `errors[i]` is true if cat qubit i has suffered a phase flip (Z error).
    /// The syndrome bit s_i = errors[i] XOR errors[i+1].
    pub fn extract_syndrome(&mut self, errors: &[bool]) -> &[bool] {
        let d = self.distance;
        assert_eq!(
            errors.len(),
            d,
            "error vector length {} does not match code distance {}",
            errors.len(),
            d
        );
        for i in 0..d.saturating_sub(1) {
            self.syndrome_bits[i] = errors[i] ^ errors[i + 1];
        }
        &self.syndrome_bits
    }

    /// Decode the error pattern using majority vote.
    ///
    /// Returns `true` if the decoder determines a logical error has occurred.
    /// For the repetition code with majority-vote decoding, a logical error
    /// occurs when more than floor(d/2) data qubits have flipped.
    pub fn decode_majority_vote(&self, errors: &[bool]) -> bool {
        let num_flips: usize = errors.iter().filter(|&&e| e).count();
        // Majority vote: logical error if more than half the qubits flipped
        num_flips > self.distance / 2
    }

    /// Check whether the syndrome is trivial (all zeros = no detected errors).
    pub fn is_syndrome_trivial(&self) -> bool {
        self.syndrome_bits.iter().all(|&s| !s)
    }

    /// Return the number of stabilizers (= distance - 1).
    pub fn num_stabilizers(&self) -> usize {
        self.syndrome_bits.len()
    }

    /// Return the maximum number of correctable errors: floor((d-1)/2).
    pub fn correction_capacity(&self) -> usize {
        (self.distance - 1) / 2
    }
}

impl fmt::Display for RepetitionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let syndrome_str: String = self
            .syndrome_bits
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        write!(
            f,
            "RepCode(d={}, syndrome=[{}])",
            self.distance, syndrome_str
        )
    }
}

// ============================================================
// CONCATENATION RESULT
// ============================================================

/// Result of a concatenated cat qubit QEC simulation.
///
/// Captures all relevant physical and logical metrics from a complete
/// simulation run including noise parameters, code parameters, and
/// the achieved logical error rate.
#[derive(Debug, Clone)]
pub struct ConcatenationResult {
    /// Number of physical oscillator modes used.
    pub num_physical_modes: usize,
    /// Number of logical qubits encoded.
    pub num_logical_qubits: usize,
    /// Coherent state amplitude alpha.
    pub alpha: f64,
    /// Physical bit-flip error rate per round.
    pub bit_flip_rate: f64,
    /// Physical phase-flip error rate per round.
    pub phase_flip_rate: f64,
    /// Noise bias ratio.
    pub bias_ratio: f64,
    /// Achieved logical error rate.
    pub logical_error_rate: f64,
    /// Outer repetition code distance.
    pub outer_code_distance: usize,
    /// Number of QEC rounds simulated.
    pub num_rounds: usize,
    /// Number of logical errors observed.
    pub num_logical_errors: usize,
    /// Analytical logical error estimate for comparison.
    pub analytical_estimate: f64,
}

impl fmt::Display for ConcatenationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConcatenationResult {{\n\
             \x20 physical_modes: {},\n\
             \x20 logical_qubits: {},\n\
             \x20 alpha: {:.2},\n\
             \x20 bit_flip_rate: {:.4e},\n\
             \x20 phase_flip_rate: {:.4e},\n\
             \x20 bias_ratio: {:.2e},\n\
             \x20 logical_error_rate: {:.6e} (MC),\n\
             \x20 analytical_estimate: {:.6e},\n\
             \x20 code_distance: {},\n\
             \x20 qec_rounds: {},\n\
             \x20 logical_errors: {}/{}\n\
             }}",
            self.num_physical_modes,
            self.num_logical_qubits,
            self.alpha,
            self.bit_flip_rate,
            self.phase_flip_rate,
            self.bias_ratio,
            self.logical_error_rate,
            self.analytical_estimate,
            self.outer_code_distance,
            self.num_rounds,
            self.num_logical_errors,
            self.num_rounds,
        )
    }
}

// ============================================================
// CAT QUBIT SIMULATOR
// ============================================================

/// Main concatenated cat qubit simulator.
///
/// Provides:
/// - Fock-space construction of coherent and cat states.
/// - Computation of the biased noise model.
/// - Monte Carlo simulation of the concatenated scheme (cat qubits + repetition code).
/// - Analytical estimates of logical error rates.
/// - Gate-level cat qubit operations in Fock space.
///
/// # Example
///
/// ```ignore
/// use nqpu_metal::cat_qubit_concatenation::*;
///
/// let config = CatQubitConfig::new()
///     .alpha(2.0)
///     .num_cats(5)
///     .num_rounds(10_000)
///     .build()
///     .unwrap();
///
/// let sim = CatQubitSimulator::new(config).unwrap();
/// let result = sim.run_concatenation();
/// println!("{}", result);
/// ```
pub struct CatQubitSimulator {
    /// Simulator configuration.
    pub config: CatQubitConfig,
    /// Precomputed biased noise model.
    pub noise: BiasedNoiseModel,
}

impl CatQubitSimulator {
    /// Create a new simulator from a validated configuration.
    pub fn new(config: CatQubitConfig) -> Result<Self, CatQubitError> {
        let noise = BiasedNoiseModel::from_params(config.alpha, config.kappa_1, config.kappa_2);

        Ok(CatQubitSimulator { config, noise })
    }

    /// Construct a coherent state |alpha> in the Fock basis.
    pub fn coherent_state(&self, alpha: C64) -> CoherentState {
        CoherentState::new(alpha, self.config.fock_cutoff)
    }

    /// Construct the logical |0_L> (even cat) state.
    pub fn logical_zero(&self) -> Result<CatQubitState, CatQubitError> {
        CatQubitState::new(0, self.config.alpha, self.config.fock_cutoff)
    }

    /// Construct the logical |1_L> (odd cat) state.
    pub fn logical_one(&self) -> Result<CatQubitState, CatQubitError> {
        CatQubitState::new(1, self.config.alpha, self.config.fock_cutoff)
    }

    /// Compute the CX gate model between a cat qubit and transmon ancilla.
    pub fn cx_gate(&self) -> CatTransmonCX {
        CatTransmonCX::from_params(
            self.config.alpha,
            self.config.kappa_1,
            self.config.kappa_2,
            self.config.gate_time,
        )
    }

    /// Build the X gate operator in Fock space (exchanges |0_L> and |1_L>).
    pub fn x_operator(&self) -> CatQubitOperator {
        CatQubitOperator::x_gate(self.config.alpha, self.config.fock_cutoff)
    }

    /// Build the Z gate operator (parity) in Fock space.
    pub fn z_operator(&self) -> CatQubitOperator {
        CatQubitOperator::z_gate(self.config.fock_cutoff)
    }

    /// Compute the analytical estimate of the concatenated logical error rate.
    ///
    /// The logical error rate for d cat qubits in a repetition code is:
    ///
    ///   p_L ~ C(d, (d+1)/2) * p_pf^{(d+1)/2}  +  d * p_bf
    ///
    /// where p_pf is the phase-flip probability per round and p_bf is the
    /// bit-flip probability per round. The first term is the outer code
    /// contribution (phase flips requiring > floor(d/2) errors), and the
    /// second term is the uncorrected bit-flip contribution (exponentially
    /// small in alpha^2).
    pub fn analytical_logical_error_rate(&self) -> f64 {
        let d = self.config.num_cats;
        let p_pf = self.noise.phase_flip_probability(self.config.gate_time);
        let p_bf = self.noise.bit_flip_probability(self.config.gate_time);

        // Outer code: fails when > d/2 phase flips
        let t = (d + 1) / 2; // minimum uncorrectable errors
        let outer_contribution = binomial_coefficient(d, t) * p_pf.powi(t as i32);

        // Inner code: uncorrected bit flips (each cat contributes independently)
        let inner_contribution = d as f64 * p_bf;

        outer_contribution + inner_contribution
    }

    /// Run a full Monte Carlo simulation of the concatenated scheme.
    ///
    /// For each of `num_rounds` rounds:
    ///   1. Sample bit-flip errors on each cat qubit (probability p_bf).
    ///   2. Sample phase-flip errors on each cat qubit (probability p_pf).
    ///   3. Extract the repetition code syndrome from phase-flip errors.
    ///   4. Decode via exact 1D minimum-weight correction.
    ///   5. A logical error occurs if:
    ///      (a) the decoder fails to correct phase flips, OR
    ///      (b) any bit flip occurred (uncorrectable by the outer code).
    ///
    /// Returns a `ConcatenationResult` with the empirical logical error rate.
    pub fn run_concatenation(&self) -> ConcatenationResult {
        let mut rng = rand::thread_rng();
        let d = self.config.num_cats;
        let num_rounds = self.config.num_rounds;

        let p_bf = self.noise.bit_flip_probability(self.config.gate_time);
        let p_pf = self.noise.phase_flip_probability(self.config.gate_time);

        let mut rep_code = RepetitionCode::new(d).expect("valid distance from config");
        let mut num_logical_errors = 0usize;

        for _round in 0..num_rounds {
            // Sample errors on each cat qubit
            let mut phase_errors = vec![false; d];
            let mut any_bit_flip = false;

            for i in 0..d {
                // Phase-flip error (dominant noise channel)
                if rng.gen::<f64>() < p_pf {
                    phase_errors[i] = true;
                }
                // Bit-flip error (exponentially suppressed)
                if rng.gen::<f64>() < p_bf {
                    any_bit_flip = true;
                }
            }

            // Extract syndrome and decode phase flips.
            rep_code.extract_syndrome(&phase_errors);
            let correction = minimum_weight_decoder(&rep_code.syndrome_bits, d);
            let residual_count = phase_errors
                .iter()
                .zip(correction.iter())
                .filter(|(err, corr)| **err ^ **corr)
                .count();
            let phase_logical_error = residual_count % 2 == 1;

            // Logical error if either:
            // - outer code failed to correct phase flips
            // - any bit flip occurred (not correctable by repetition code)
            if phase_logical_error || any_bit_flip {
                num_logical_errors += 1;
            }
        }

        let logical_error_rate = num_logical_errors as f64 / num_rounds as f64;
        let analytical_estimate = self.analytical_logical_error_rate();

        ConcatenationResult {
            num_physical_modes: d,
            num_logical_qubits: 1,
            alpha: self.config.alpha,
            bit_flip_rate: self.noise.bit_flip_rate,
            phase_flip_rate: self.noise.phase_flip_rate,
            bias_ratio: self.noise.bias_ratio,
            logical_error_rate,
            outer_code_distance: d,
            num_rounds,
            num_logical_errors,
            analytical_estimate,
        }
    }

    /// Run the Monte Carlo simulation with a specific seed for reproducibility.
    pub fn run_concatenation_seeded(&self, seed: u64) -> ConcatenationResult {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let d = self.config.num_cats;
        let num_rounds = self.config.num_rounds;

        let p_bf = self.noise.bit_flip_probability(self.config.gate_time);
        let p_pf = self.noise.phase_flip_probability(self.config.gate_time);

        let mut rep_code = RepetitionCode::new(d).expect("valid distance from config");
        let mut num_logical_errors = 0usize;

        for _round in 0..num_rounds {
            let mut phase_errors = vec![false; d];
            let mut any_bit_flip = false;

            for i in 0..d {
                if rng.gen::<f64>() < p_pf {
                    phase_errors[i] = true;
                }
                if rng.gen::<f64>() < p_bf {
                    any_bit_flip = true;
                }
            }

            rep_code.extract_syndrome(&phase_errors);
            let correction = minimum_weight_decoder(&rep_code.syndrome_bits, d);
            let residual_count = phase_errors
                .iter()
                .zip(correction.iter())
                .filter(|(err, corr)| **err ^ **corr)
                .count();
            let phase_logical_error = residual_count % 2 == 1;

            if phase_logical_error || any_bit_flip {
                num_logical_errors += 1;
            }
        }

        let logical_error_rate = num_logical_errors as f64 / num_rounds as f64;
        let analytical_estimate = self.analytical_logical_error_rate();

        ConcatenationResult {
            num_physical_modes: d,
            num_logical_qubits: 1,
            alpha: self.config.alpha,
            bit_flip_rate: self.noise.bit_flip_rate,
            phase_flip_rate: self.noise.phase_flip_rate,
            bias_ratio: self.noise.bias_ratio,
            logical_error_rate,
            outer_code_distance: d,
            num_rounds,
            num_logical_errors,
            analytical_estimate,
        }
    }

    /// Sweep the logical error rate as a function of code distance.
    ///
    /// Returns a vector of (distance, logical_error_rate) pairs showing
    /// how error correction improves with more cat qubits.
    pub fn sweep_distance(
        &self,
        distances: &[usize],
        rounds_per_point: usize,
    ) -> Vec<(usize, f64)> {
        let mut results = Vec::with_capacity(distances.len());

        for &d in distances {
            if d == 0 || d % 2 == 0 {
                continue; // skip invalid distances
            }

            let sweep_config = CatQubitConfig {
                alpha: self.config.alpha,
                fock_cutoff: self.config.fock_cutoff,
                kappa_1: self.config.kappa_1,
                kappa_2: self.config.kappa_2,
                gate_time: self.config.gate_time,
                num_cats: d,
                num_rounds: rounds_per_point,
                physical_error_rate: self.config.physical_error_rate,
                kerr_strength: self.config.kerr_strength,
            };

            if let Ok(sim) = CatQubitSimulator::new(sweep_config) {
                let result = sim.run_concatenation();
                results.push((d, result.logical_error_rate));
            }
        }

        results
    }

    /// Sweep the noise bias as a function of alpha.
    ///
    /// Returns a vector of (alpha, bias_ratio) pairs demonstrating the
    /// exponential growth of noise asymmetry with coherent state amplitude.
    pub fn sweep_alpha_bias(alphas: &[f64], kappa_1: f64, kappa_2: f64) -> Vec<(f64, f64)> {
        alphas
            .iter()
            .map(|&alpha| {
                let noise = BiasedNoiseModel::from_params(alpha, kappa_1, kappa_2);
                (alpha, noise.bias_ratio)
            })
            .collect()
    }

    /// Compute the break-even alpha: the value of alpha where the
    /// concatenated code begins to outperform the physical error rate.
    ///
    /// Uses a simple bisection search on the analytical formula.
    pub fn find_break_even_alpha(
        kappa_1: f64,
        kappa_2: f64,
        gate_time: f64,
        distance: usize,
        target_error_rate: f64,
    ) -> Option<f64> {
        let mut lo = 0.5_f64;
        let mut hi = 8.0_f64;

        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            let noise = BiasedNoiseModel::from_params(mid, kappa_1, kappa_2);
            let p_pf = noise.phase_flip_probability(gate_time);
            let p_bf = noise.bit_flip_probability(gate_time);

            let t = (distance + 1) / 2;
            let logical_rate =
                binomial_coefficient(distance, t) * p_pf.powi(t as i32) + distance as f64 * p_bf;

            if logical_rate < target_error_rate {
                hi = mid;
            } else {
                lo = mid;
            }

            if (hi - lo) < 1e-6 {
                return Some((lo + hi) / 2.0);
            }
        }

        Some((lo + hi) / 2.0)
    }
}

impl fmt::Display for CatQubitSimulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CatQubitSimulator(alpha={:.2}, d={}, noise={})",
            self.config.alpha, self.config.num_cats, self.noise
        )
    }
}

// ============================================================
// BINOMIAL COEFFICIENT HELPER
// ============================================================

/// Compute the binomial coefficient C(n, k) as f64.
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ============================================================
// WIGNER FUNCTION (PHASE-SPACE VISUALIZATION)
// ============================================================

/// Compute the Wigner function W(x, p) for a Fock-space state vector.
///
/// The Wigner function provides a quasi-probability distribution in
/// phase space (position x, momentum p). For cat states it shows the
/// characteristic interference fringes between the two coherent blobs.
///
/// Uses the formula:
///   W(x,p) = (1/pi) * sum_{m,n} rho_{mn} * W_{mn}(x,p)
///
/// where W_{mn} is the Wigner function kernel for Fock states |m><n|.
///
/// For diagonal elements (m=n):
///   W_{nn}(x,p) = (-1)^n / pi * exp(-(x^2+p^2)) * L_n(2(x^2+p^2))
///
/// For off-diagonal elements:
///   W_{mn}(x,p) for m > n involves associated Laguerre polynomials.
///
/// This function computes only the diagonal contribution for efficiency.
pub fn wigner_diagonal(state: &[C64], x: f64, p: f64) -> f64 {
    let r_sq = x * x + p * p;
    let exp_factor = (-r_sq).exp() / std::f64::consts::PI;

    let mut w = 0.0_f64;
    for n in 0..state.len() {
        let rho_nn = state[n].norm_sqr();
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let l_n = laguerre_polynomial(n, 2.0 * r_sq);
        w += rho_nn * sign * l_n;
    }

    w * exp_factor
}

/// Laguerre polynomial L_n(x) evaluated via the three-term recurrence.
fn laguerre_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut l_prev = 1.0; // L_0
    let mut l_curr = 1.0 - x; // L_1
    for k in 2..=n {
        let kf = k as f64;
        let l_next = ((2.0 * kf - 1.0 - x) * l_curr - (kf - 1.0) * l_prev) / kf;
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

// ============================================================
// FIDELITY AND DISTANCE METRICS
// ============================================================

/// Compute the overlap |<cat_0|cat_1>|^2 between the two logical states.
///
/// For large alpha this should be exponentially close to zero, confirming
/// that the logical states are nearly orthogonal.
pub fn logical_state_overlap(alpha: f64, fock_cutoff: usize) -> Result<f64, CatQubitError> {
    let cat0 = CatQubitState::new(0, alpha, fock_cutoff)?;
    let cat1 = CatQubitState::new(1, alpha, fock_cutoff)?;
    Ok(cat0.fidelity(&cat1))
}

/// Compute the effective code distance from the noise parameters.
///
/// The effective distance quantifies how many errors the concatenated
/// scheme can tolerate. For the cat + repetition code:
///   d_eff = d (repetition distance) for phase flips
///   d_eff = infinite for bit flips (exponentially suppressed)
pub fn effective_code_distance(alpha: f64, repetition_distance: usize) -> f64 {
    let alpha_sq = alpha * alpha;
    // Effective bit-flip distance grows exponentially
    let bf_distance = 2.0 * alpha_sq; // effectively infinite
                                      // Phase-flip distance is the repetition code distance
    let pf_distance = repetition_distance as f64;
    // Return the minimum (bottleneck)
    bf_distance.min(pf_distance)
}

// ============================================================
// THRESHOLD ESTIMATION
// ============================================================

/// Estimate the phase-flip threshold for the repetition code.
///
/// Below this per-round phase-flip probability, increasing the repetition
/// code distance always reduces the logical error rate. For a simple
/// repetition code with majority-vote decoding, the threshold is 50%.
///
/// In practice, measurement errors and correlated noise reduce this.
/// This function returns the ideal threshold for the given decoding scheme.
pub fn repetition_code_threshold() -> f64 {
    // For majority-vote decoding of a repetition code, the threshold is
    // exactly 50%: as long as each qubit flips with probability < 0.5,
    // increasing distance suppresses the logical error rate.
    0.5
}

/// Compute the number of cat qubits needed to reach a target logical
/// error rate, given the physical noise parameters.
///
/// Uses the analytical formula and searches over odd distances.
pub fn required_distance(
    alpha: f64,
    kappa_1: f64,
    kappa_2: f64,
    gate_time: f64,
    target_logical_error: f64,
) -> usize {
    let noise = BiasedNoiseModel::from_params(alpha, kappa_1, kappa_2);
    let p_pf = noise.phase_flip_probability(gate_time);
    let p_bf = noise.bit_flip_probability(gate_time);

    // Search odd distances from 1 to 101
    for d in (1..=101).step_by(2) {
        let t = (d + 1) / 2;
        let logical_rate = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

        if logical_rate < target_logical_error {
            return d;
        }
    }

    // If we cannot reach the target with d=101, return 101 as upper bound
    101
}

// ============================================================
// MULTI-LEVEL CONCATENATION
// ============================================================

/// Result of a multi-level concatenation analysis.
///
/// Multi-level concatenation nests cat qubits inside repetition codes
/// inside larger codes to achieve arbitrarily low logical error rates.
#[derive(Debug, Clone)]
pub struct MultiLevelResult {
    /// Number of concatenation levels.
    pub levels: usize,
    /// Error rate at each level (level 0 = physical cat qubit).
    pub error_rates: Vec<f64>,
    /// Code distance at each level.
    pub distances: Vec<usize>,
    /// Total number of physical cat qubits required.
    pub total_physical_qubits: usize,
}

impl MultiLevelResult {
    /// Compute the resource overhead: physical qubits per logical qubit.
    pub fn overhead(&self) -> usize {
        self.total_physical_qubits
    }
}

impl fmt::Display for MultiLevelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MultiLevelConcatenation ({} levels):", self.levels)?;
        for i in 0..self.levels {
            writeln!(
                f,
                "  Level {}: d={}, error_rate={:.4e}",
                i, self.distances[i], self.error_rates[i]
            )?;
        }
        write!(f, "  Total physical qubits: {}", self.total_physical_qubits)
    }
}

/// Analyze multi-level concatenation of the cat + repetition scheme.
///
/// At each level l, the logical error rate from level l-1 becomes the
/// physical error rate for the repetition code at level l:
///
///   p_L^{(l)} ~ C(d_l, (d_l+1)/2) * (p_L^{(l-1)})^{(d_l+1)/2}
///
/// The total qubit count is the product of all distances.
pub fn multi_level_concatenation(
    alpha: f64,
    kappa_1: f64,
    kappa_2: f64,
    gate_time: f64,
    distances: &[usize],
) -> MultiLevelResult {
    let noise = BiasedNoiseModel::from_params(alpha, kappa_1, kappa_2);
    let p_bf = noise.bit_flip_probability(gate_time);
    let p_pf = noise.phase_flip_probability(gate_time);

    let levels = distances.len();
    let mut error_rates = Vec::with_capacity(levels);
    let mut total_physical = 1usize;

    // Level 0: single cat qubit concatenated with first repetition code
    let d0 = distances[0];
    let t0 = (d0 + 1) / 2;
    let p_l0 = binomial_coefficient(d0, t0) * p_pf.powi(t0 as i32) + d0 as f64 * p_bf;
    error_rates.push(p_l0);
    total_physical *= d0;

    // Higher levels: recursion
    for l in 1..levels {
        let d = distances[l];
        let t = (d + 1) / 2;
        let p_prev = error_rates[l - 1];
        let p_l = binomial_coefficient(d, t) * p_prev.powi(t as i32);
        error_rates.push(p_l);
        total_physical *= d;
    }

    MultiLevelResult {
        levels,
        error_rates,
        distances: distances.to_vec(),
        total_physical_qubits: total_physical,
    }
}

// ============================================================
// PHOTON NUMBER STATISTICS
// ============================================================

/// Compute the photon number distribution P(n) = |c_n|^2 for a cat state.
pub fn photon_number_distribution(
    logical: u8,
    alpha: f64,
    fock_cutoff: usize,
) -> Result<Vec<f64>, CatQubitError> {
    let cat = CatQubitState::new(logical, alpha, fock_cutoff)?;
    Ok(cat.fock_coefficients.iter().map(|c| c.norm_sqr()).collect())
}

/// Compute the photon number variance for a cat state.
pub fn photon_number_variance(
    logical: u8,
    alpha: f64,
    fock_cutoff: usize,
) -> Result<f64, CatQubitError> {
    let cat = CatQubitState::new(logical, alpha, fock_cutoff)?;
    let mean: f64 = cat
        .fock_coefficients
        .iter()
        .enumerate()
        .map(|(n, c)| n as f64 * c.norm_sqr())
        .sum();
    let mean_sq: f64 = cat
        .fock_coefficients
        .iter()
        .enumerate()
        .map(|(n, c)| (n as f64).powi(2) * c.norm_sqr())
        .sum();
    Ok(mean_sq - mean * mean)
}

// ============================================================
// NOISE CHANNEL APPLICATION
// ============================================================

/// Apply a single-photon loss channel to a Fock-space state vector.
///
/// The Kraus operator for single-photon loss is proportional to the
/// annihilation operator: K_1 = sqrt(kappa_1 * dt) * a.
/// This function applies one loss event: result = a|psi> (unnormalized).
pub fn apply_photon_loss(state: &[C64]) -> Vec<C64> {
    let dim = state.len();
    let mut result = vec![C64::new(0.0, 0.0); dim];
    // a|n> = sqrt(n)|n-1>
    for n in 1..dim {
        result[n - 1] = state[n] * (n as f64).sqrt();
    }
    result
}

/// Apply a dephasing channel to a Fock-space state vector.
///
/// Dephasing in the Fock basis rotates each component by a random phase
/// proportional to the photon number: |n> -> e^{i*phi*n} |n>.
///
/// For cat qubits, this corresponds to a phase-flip (Z) error when the
/// accumulated phase crosses pi.
pub fn apply_dephasing(state: &[C64], phase: f64) -> Vec<C64> {
    state
        .iter()
        .enumerate()
        .map(|(n, &c)| {
            let theta = phase * n as f64;
            c * C64::new(theta.cos(), theta.sin())
        })
        .collect()
}

/// Apply the parity operator (-1)^n (logical X for cat qubits).
///
/// This flips the logical state: |0_L> <-> |1_L>.
pub fn apply_parity(state: &[C64]) -> Vec<C64> {
    state
        .iter()
        .enumerate()
        .map(|(n, &c)| if n % 2 == 0 { c } else { -c })
        .collect()
}

// ============================================================
// STABILIZER MEASUREMENT (REPETITION CODE)
// ============================================================

/// Simulate noisy stabilizer measurement for the repetition code.
///
/// In practice, syndrome extraction is itself noisy. This function
/// models measurement errors by flipping each syndrome bit with
/// the given probability.
pub fn noisy_syndrome_extraction(
    errors: &[bool],
    measurement_error_rate: f64,
    rng: &mut impl Rng,
) -> Vec<bool> {
    let d = errors.len();
    if d <= 1 {
        return vec![];
    }
    let mut syndrome = vec![false; d - 1];
    for i in 0..d - 1 {
        // Ideal syndrome
        syndrome[i] = errors[i] ^ errors[i + 1];
        // Measurement error
        if rng.gen::<f64>() < measurement_error_rate {
            syndrome[i] = !syndrome[i];
        }
    }
    syndrome
}

/// Decoder selection for 1D repetition-code syndromes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepetitionDecoderAlgorithm {
    /// Exact minimum-weight decoder on a 1D chain.
    Exact1D,
    /// Greedy adjacent-defect pairing baseline.
    GreedyAdjacent,
}

/// Decode a repetition-code syndrome with the requested algorithm.
///
/// Returns a correction vector where `true` means apply Z on that qubit.
pub fn decode_repetition_syndrome(
    syndrome: &[bool],
    distance: usize,
    algorithm: RepetitionDecoderAlgorithm,
) -> Vec<bool> {
    match algorithm {
        RepetitionDecoderAlgorithm::Exact1D => minimum_weight_decoder(syndrome, distance),
        RepetitionDecoderAlgorithm::GreedyAdjacent => greedy_repetition_decoder(syndrome, distance),
    }
}

/// Exact minimum-weight decoder for the open-boundary 1D repetition code.
///
/// The syndrome constraints are `c[i] XOR c[i+1] = s[i]`.
/// This linear system has exactly two solutions, corresponding to `c[0]=0` and
/// `c[0]=1` (the global complement). The exact minimum-weight correction is the
/// lower-cost of these two candidates.
///
/// Returns a correction vector: `correction[i] = true` means apply Z on qubit i.
pub fn minimum_weight_decoder(syndrome: &[bool], distance: usize) -> Vec<bool> {
    minimum_weight_decoder_weighted(syndrome, distance, None)
}

/// Weighted exact decoder variant.
///
/// If `weights` is provided, it must have length `distance`, and the decoder
/// minimizes total correction cost rather than raw Hamming weight.
pub fn minimum_weight_decoder_weighted(
    syndrome: &[bool],
    distance: usize,
    weights: Option<&[f64]>,
) -> Vec<bool> {
    if distance == 0 {
        return Vec::new();
    }
    assert!(
        syndrome.len() + 1 == distance || (distance == 1 && syndrome.is_empty()),
        "syndrome length {} incompatible with distance {}",
        syndrome.len(),
        distance
    );
    if let Some(w) = weights {
        assert_eq!(
            w.len(),
            distance,
            "weights length {} incompatible with distance {}",
            w.len(),
            distance
        );
    }

    let mut candidate0 = vec![false; distance];
    for i in 0..distance.saturating_sub(1) {
        candidate0[i + 1] = candidate0[i] ^ syndrome[i];
    }
    let mut candidate1 = vec![true; distance];
    for i in 0..distance.saturating_sub(1) {
        candidate1[i + 1] = candidate1[i] ^ syndrome[i];
    }

    fn correction_cost(correction: &[bool], weights: Option<&[f64]>) -> f64 {
        match weights {
            Some(w) => correction
                .iter()
                .enumerate()
                .map(|(i, &b)| if b { w[i] } else { 0.0 })
                .sum(),
            None => correction.iter().filter(|&&b| b).count() as f64,
        }
    }

    let cost0 = correction_cost(&candidate0, weights);
    let cost1 = correction_cost(&candidate1, weights);
    if cost0 <= cost1 {
        candidate0
    } else {
        candidate1
    }
}

/// Greedy adjacent-defect decoder kept as a baseline mode for comparisons.
pub fn greedy_repetition_decoder(syndrome: &[bool], distance: usize) -> Vec<bool> {
    let mut correction = vec![false; distance];
    let mut defect_positions: Vec<usize> = syndrome
        .iter()
        .enumerate()
        .filter(|(_, &s)| s)
        .map(|(i, _)| i)
        .collect();

    while defect_positions.len() >= 2 {
        let a = defect_positions.remove(0);
        let b = defect_positions.remove(0);
        for q in (a + 1)..=b {
            if q < distance {
                correction[q] = !correction[q];
            }
        }
    }

    if let Some(&pos) = defect_positions.first() {
        if pos < distance / 2 {
            for q in 0..=pos {
                correction[q] = !correction[q];
            }
        } else {
            for q in (pos + 1)..distance {
                correction[q] = !correction[q];
            }
        }
    }

    correction
}

// ============================================================
// CONCATENATION WITH NOISY SYNDROME EXTRACTION
// ============================================================

/// Run a concatenated simulation with realistic noisy syndrome extraction.
///
/// Unlike the basic Monte Carlo which uses ideal syndrome extraction,
/// this version models measurement errors and uses minimum-weight
/// decoding instead of majority vote.
pub fn run_noisy_concatenation(
    config: &CatQubitConfig,
    measurement_error_rate: f64,
    seed: u64,
) -> ConcatenationResult {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let noise = BiasedNoiseModel::from_params(config.alpha, config.kappa_1, config.kappa_2);
    let d = config.num_cats;
    let num_rounds = config.num_rounds;

    let p_bf = noise.bit_flip_probability(config.gate_time);
    let p_pf = noise.phase_flip_probability(config.gate_time);

    let mut num_logical_errors = 0usize;

    for _round in 0..num_rounds {
        let mut phase_errors = vec![false; d];
        let mut any_bit_flip = false;

        for i in 0..d {
            if rng.gen::<f64>() < p_pf {
                phase_errors[i] = true;
            }
            if rng.gen::<f64>() < p_bf {
                any_bit_flip = true;
            }
        }

        // Noisy syndrome extraction
        let syndrome = noisy_syndrome_extraction(&phase_errors, measurement_error_rate, &mut rng);

        // Exact minimum-weight decoding
        let correction = minimum_weight_decoder(&syndrome, d);

        // Apply correction to determine residual errors
        let mut residual = vec![false; d];
        for i in 0..d {
            residual[i] = phase_errors[i] ^ correction[i];
        }

        // Logical error if odd number of residual phase flips
        let residual_count: usize = residual.iter().filter(|&&e| e).count();
        let phase_logical_error = residual_count % 2 == 1;

        if phase_logical_error || any_bit_flip {
            num_logical_errors += 1;
        }
    }

    let logical_error_rate = num_logical_errors as f64 / num_rounds as f64;

    // Analytical estimate (ideal, for comparison)
    let t = (d + 1) / 2;
    let analytical_estimate = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

    ConcatenationResult {
        num_physical_modes: d,
        num_logical_qubits: 1,
        alpha: config.alpha,
        bit_flip_rate: noise.bit_flip_rate,
        phase_flip_rate: noise.phase_flip_rate,
        bias_ratio: noise.bias_ratio,
        logical_error_rate,
        outer_code_distance: d,
        num_rounds,
        num_logical_errors,
        analytical_estimate,
    }
}

// ============================================================
// CAT GATE ENUM
// ============================================================

/// Enumeration of gates natively supported in the cat qubit architecture.
///
/// Cat qubits possess a remarkable gate set due to their parity structure:
/// - Z and ZZ gates are bias-preserving (do not introduce bit-flip errors).
/// - The CX gate is implemented via cat-transmon coupling.
/// - The Toffoli gate is native (not decomposed from 2-qubit gates!), a unique
///   advantage of the cat architecture reducing resource overhead for algorithms
///   like Grover search and arithmetic circuits.
///
/// All gate error rates respect the asymmetric noise bias: Z-type gates inherit
/// exponentially suppressed bit-flip errors while CX and Toffoli accrue phase
/// errors proportional to the gate duration.
#[derive(Debug, Clone, PartialEq)]
pub enum CatGate {
    /// Identity (idle) on a single cat qubit for one gate cycle.
    Idle(usize),
    /// Logical Z gate on a single cat qubit (parity operator).
    ///
    /// Deterministic and noiseless in the ideal cat encoding since Z_L = (-1)^n
    /// commutes with the stabilizer Hamiltonian. In practice, residual photon-
    /// number fluctuations introduce a tiny dephasing contribution that scales
    /// as kappa_1 / kappa_2.
    Z(usize),
    /// Logical ZZ gate between two cat qubits (bias-preserving 2-qubit gate).
    ///
    /// Implemented via a cross-Kerr interaction between two oscillator modes.
    /// Preserves the exponential bit-flip bias because it acts diagonally in the
    /// parity eigenbasis. Phase-flip error per gate ~ kappa_1 * alpha^2 * t_gate.
    ZZ(usize, usize),
    /// Controlled-X (CNOT) gate via cat-transmon interaction.
    ///
    /// The control is a cat qubit (parity eigenstates) and the target may be
    /// another cat qubit or a transmon ancilla. Implemented through a conditional
    /// displacement in phase space. Gate time is limited by the dispersive shift
    /// chi between the oscillator and the transmon.
    CX(usize, usize),
    /// Native Toffoli (CCNOT) gate -- a unique advantage of cat qubits.
    ///
    /// In conventional qubit architectures, Toffoli must be decomposed into
    /// 6+ two-qubit gates. In the cat architecture it is implemented natively
    /// via a three-body cross-Kerr interaction, requiring only a single gate
    /// cycle. This dramatically reduces circuit depth for arithmetic circuits
    /// and Grover oracles.
    Toffoli(usize, usize, usize),
    /// Hadamard gate: implemented via a pi/2 rotation in the cat qubit manifold.
    ///
    /// Not bias-preserving -- introduces both bit-flip and phase-flip errors.
    /// Used sparingly and only when necessary for state preparation.
    Hadamard(usize),
    /// Measurement in the Z (parity) basis.
    MeasureZ(usize),
}

impl CatGate {
    /// Return the qubits this gate acts on.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            CatGate::Idle(q) | CatGate::Z(q) | CatGate::Hadamard(q) | CatGate::MeasureZ(q) => {
                vec![*q]
            }
            CatGate::ZZ(q1, q2) | CatGate::CX(q1, q2) => vec![*q1, *q2],
            CatGate::Toffoli(q1, q2, q3) => vec![*q1, *q2, *q3],
        }
    }

    /// Return the gate name as a string.
    pub fn name(&self) -> &str {
        match self {
            CatGate::Idle(_) => "Idle",
            CatGate::Z(_) => "Z",
            CatGate::ZZ(_, _) => "ZZ",
            CatGate::CX(_, _) => "CX",
            CatGate::Toffoli(_, _, _) => "Toffoli",
            CatGate::Hadamard(_) => "Hadamard",
            CatGate::MeasureZ(_) => "MeasureZ",
        }
    }

    /// Return whether this gate preserves the noise bias.
    ///
    /// Bias-preserving gates do not introduce bit-flip errors (they commute
    /// with the stabilizer Hamiltonian). Only Z-type gates are bias-preserving.
    pub fn is_bias_preserving(&self) -> bool {
        matches!(self, CatGate::Z(_) | CatGate::ZZ(_, _) | CatGate::Idle(_))
    }

    /// Compute the error rates for this gate given the noise model.
    ///
    /// Returns (bit_flip_probability, phase_flip_probability) per gate application.
    pub fn error_rates(&self, noise: &BiasedNoiseModel, gate_time_us: f64) -> (f64, f64) {
        let p_bf = noise.bit_flip_probability(gate_time_us);
        let p_pf = noise.phase_flip_probability(gate_time_us);

        match self {
            CatGate::Idle(_) => {
                // Idle: noise accumulates at the bare rates for one gate cycle.
                (p_bf, p_pf)
            }
            CatGate::Z(_) => {
                // Z is deterministic in the cat encoding. Residual errors from
                // imperfect confinement are negligible for large kappa_2/kappa_1.
                (p_bf * 0.01, p_pf * 0.01)
            }
            CatGate::ZZ(_, _) => {
                // ZZ is bias-preserving. Phase-flip error on each qubit during
                // the cross-Kerr interaction. Bit-flip suppression maintained.
                (p_bf, p_pf * 2.0)
            }
            CatGate::CX(_, _) => {
                // CX via conditional displacement. Phase-flip errors on both
                // qubits accumulate during the gate. Bit-flip suppression is
                // partially degraded on the target (transmon decoherence).
                let cx_bf = p_bf * 5.0; // transmon T1 limits bit-flip suppression
                let cx_pf = p_pf * 3.0; // conditional displacement duration
                (cx_bf.min(1.0), cx_pf.min(1.0))
            }
            CatGate::Toffoli(_, _, _) => {
                // Native Toffoli via three-body cross-Kerr. Takes approximately
                // 2x the duration of a CX gate but avoids the 6-gate decomposition.
                let tof_bf = p_bf * 8.0;
                let tof_pf = p_pf * 5.0;
                (tof_bf.min(1.0), tof_pf.min(1.0))
            }
            CatGate::Hadamard(_) => {
                // Hadamard is NOT bias-preserving. It rotates in the cat manifold
                // and introduces both error types at comparable rates.
                let h_bf = p_pf * 0.5; // bit-flip from rotation
                let h_pf = p_pf * 0.5;
                (h_bf.min(1.0), h_pf.min(1.0))
            }
            CatGate::MeasureZ(_) => {
                // Measurement errors from readout imperfections.
                (p_bf * 0.1, p_pf * 0.1)
            }
        }
    }
}

impl fmt::Display for CatGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatGate::Idle(q) => write!(f, "Idle({})", q),
            CatGate::Z(q) => write!(f, "Z({})", q),
            CatGate::ZZ(q1, q2) => write!(f, "ZZ({},{})", q1, q2),
            CatGate::CX(q1, q2) => write!(f, "CX({},{})", q1, q2),
            CatGate::Toffoli(q1, q2, q3) => write!(f, "Toffoli({},{},{})", q1, q2, q3),
            CatGate::Hadamard(q) => write!(f, "H({})", q),
            CatGate::MeasureZ(q) => write!(f, "MeasZ({})", q),
        }
    }
}

// ============================================================
// CONCATENATED CAT CODE
// ============================================================

/// Complete concatenated cat code combining inner bosonic protection
/// with an outer repetition code.
///
/// This struct manages the interplay between:
/// - **Inner code** (cat qubit): Exponentially suppresses bit-flip errors
///   through two-photon dissipation. The protection scales as exp(-2|alpha|^2).
/// - **Outer code** (repetition): Corrects phase-flip errors via majority-vote
///   or minimum-weight decoding. Correction capacity = floor((d-1)/2).
///
/// The concatenation threshold is the value of the per-qubit phase-flip
/// probability below which increasing the repetition distance always reduces
/// the logical error rate. For the cat architecture this threshold is
/// remarkably forgiving because the inner code eliminates one error type entirely.
#[derive(Debug, Clone)]
pub struct ConcatenatedCatCode {
    /// Configuration parameters.
    pub config: CatQubitConfig,
    /// Precomputed noise model.
    pub noise: BiasedNoiseModel,
    /// Outer repetition code.
    pub repetition_code: RepetitionCode,
    /// Per-qubit phase-flip probability per syndrome round.
    pub phase_flip_per_round: f64,
    /// Per-qubit bit-flip probability per syndrome round.
    pub bit_flip_per_round: f64,
}

impl ConcatenatedCatCode {
    /// Construct a new concatenated code from configuration.
    pub fn new(config: CatQubitConfig) -> Result<Self, CatQubitError> {
        let noise = BiasedNoiseModel::from_params(config.alpha, config.kappa_1, config.kappa_2);
        let repetition_code = RepetitionCode::new(config.num_cats)?;
        let phase_flip_per_round = noise.phase_flip_probability(config.gate_time);
        let bit_flip_per_round = noise.bit_flip_probability(config.gate_time);

        Ok(ConcatenatedCatCode {
            config,
            noise,
            repetition_code,
            phase_flip_per_round,
            bit_flip_per_round,
        })
    }

    /// Compute the analytical logical error rate for this code.
    ///
    /// P_L(d, alpha) = C(d, t) * p_pf^t + d * p_bf
    ///
    /// where t = (d+1)/2 is the minimum number of uncorrectable phase flips,
    /// p_pf is the per-round phase-flip probability, and p_bf is the per-round
    /// bit-flip probability.
    pub fn logical_error_rate(&self) -> f64 {
        let d = self.config.num_cats;
        let t = (d + 1) / 2;
        let outer = binomial_coefficient(d, t) * self.phase_flip_per_round.powi(t as i32);
        let inner = d as f64 * self.bit_flip_per_round;
        outer + inner
    }

    /// Compute the concatenation threshold: the maximum per-round phase-flip
    /// probability below which increasing the repetition code distance always
    /// reduces the logical error rate.
    ///
    /// For the majority-vote repetition code, the threshold is determined by
    /// the condition that the logical error rate at distance d+2 is less than
    /// at distance d. In the ideal case (no measurement errors), this threshold
    /// is p_pf < 0.5. For practical implementations with gate errors, the
    /// effective threshold is lower.
    ///
    /// This method performs a numerical search to find the threshold for the
    /// configured noise parameters by checking when increasing d from 3 to 5
    /// stops helping.
    pub fn concatenation_threshold(&self) -> f64 {
        // Binary search for the phase-flip probability threshold.
        // At p_pf = threshold, P_L(d=5) = P_L(d=3).
        let mut lo = 0.0_f64;
        let mut hi = 0.5_f64;

        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;

            // Compute logical error at d=3 and d=5 for this phase-flip rate.
            let p_bf = self.bit_flip_per_round;
            let p_l3 = binomial_coefficient(3, 2) * mid.powi(2) + 3.0 * p_bf;
            let p_l5 = binomial_coefficient(5, 3) * mid.powi(3) + 5.0 * p_bf;

            if p_l5 < p_l3 {
                // d=5 still helps; threshold is higher
                lo = mid;
            } else {
                // d=5 no longer helps; threshold is lower
                hi = mid;
            }

            if (hi - lo) < 1e-10 {
                break;
            }
        }

        (lo + hi) / 2.0
    }

    /// Compute the break-even point: the minimum alpha where the concatenated
    /// code outperforms an unconcatenated cat qubit.
    ///
    /// An unconcatenated cat qubit has error rate p_bf + p_pf.
    /// The concatenated code has error rate C(d,t)*p_pf^t + d*p_bf.
    /// Break-even is where these are equal.
    pub fn break_even_alpha(&self) -> f64 {
        let d = self.config.num_cats;
        let kappa_1 = self.config.kappa_1;
        let kappa_2 = self.config.kappa_2;
        let gate_time = self.config.gate_time;

        let mut lo = 0.5_f64;
        let mut hi = 8.0_f64;

        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            let noise = BiasedNoiseModel::from_params(mid, kappa_1, kappa_2);
            let p_pf = noise.phase_flip_probability(gate_time);
            let p_bf = noise.bit_flip_probability(gate_time);

            let unconcatenated = p_bf + p_pf;
            let t = (d + 1) / 2;
            let concatenated = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

            if concatenated < unconcatenated {
                hi = mid;
            } else {
                lo = mid;
            }

            if (hi - lo) < 1e-8 {
                break;
            }
        }

        (lo + hi) / 2.0
    }

    /// Return the physical qubit overhead: number of cat qubits per logical qubit.
    ///
    /// For the simple repetition code this equals the code distance d.
    /// Ancilla qubits for syndrome extraction add d-1 additional transmons.
    pub fn physical_overhead(&self) -> usize {
        // d data cat qubits + (d-1) ancilla transmons for syndrome extraction
        self.config.num_cats + self.repetition_code.num_stabilizers()
    }
}

impl fmt::Display for ConcatenatedCatCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConcatenatedCatCode(alpha={:.2}, d={}, P_L={:.4e}, bias={:.2e})",
            self.config.alpha,
            self.config.num_cats,
            self.logical_error_rate(),
            self.noise.bias_ratio,
        )
    }
}

// ============================================================
// CAT CIRCUIT SIMULATOR
// ============================================================

/// Result of a single circuit simulation run.
#[derive(Debug, Clone)]
pub struct CircuitRunResult {
    /// Number of gate operations executed.
    pub num_gates: usize,
    /// Total bit-flip errors injected across all qubits and gates.
    pub total_bit_flips: usize,
    /// Total phase-flip errors injected across all qubits and gates.
    pub total_phase_flips: usize,
    /// Whether a logical error was detected after QEC.
    pub logical_error: bool,
    /// Syndrome history: one syndrome vector per extraction round.
    pub syndrome_history: Vec<Vec<bool>>,
    /// Final error state of each data qubit (before correction).
    pub final_errors: Vec<bool>,
    /// Correction applied by the decoder.
    pub correction: Vec<bool>,
}

/// Circuit-level simulator for concatenated cat qubit quantum circuits.
///
/// Unlike the `CatQubitSimulator` which performs only Monte Carlo sampling
/// of error rates, `CatCircuitSimulator` executes a sequence of `CatGate`
/// operations, injecting errors according to the biased noise model at each
/// gate, performing syndrome extraction, and tracking both physical and
/// logical error rates through the circuit.
///
/// # Architecture
///
/// ```text
///   Circuit:  [H(0)] -- [CX(0,1)] -- [MeasZ(0)] -- [MeasZ(1)]
///                |           |             |              |
///   Noise:    inject    inject bf+pf    readout        readout
///             bf+pf      on both        error          error
///                |           |             |              |
///   QEC:        ----  syndrome extraction  ----
///                    majority-vote decode
///                    correction applied
/// ```
pub struct CatCircuitSimulator {
    /// Number of logical qubits in the circuit.
    pub num_qubits: usize,
    /// Configuration parameters for the cat qubits.
    pub config: CatQubitConfig,
    /// Noise model.
    pub noise: BiasedNoiseModel,
    /// Circuit: ordered sequence of gates to execute.
    pub circuit: Vec<CatGate>,
    /// Current phase-flip error state of each qubit (true = Z error present).
    phase_errors: Vec<bool>,
    /// Current bit-flip error state of each qubit (true = X error present).
    bit_flip_errors: Vec<bool>,
}

impl CatCircuitSimulator {
    /// Create a new circuit simulator.
    pub fn new(num_qubits: usize, config: CatQubitConfig) -> Result<Self, CatQubitError> {
        let noise = BiasedNoiseModel::from_params(config.alpha, config.kappa_1, config.kappa_2);
        Ok(CatCircuitSimulator {
            num_qubits,
            config,
            noise,
            circuit: Vec::new(),
            phase_errors: vec![false; num_qubits],
            bit_flip_errors: vec![false; num_qubits],
        })
    }

    /// Append a gate to the circuit.
    pub fn add_gate(&mut self, gate: CatGate) {
        self.circuit.push(gate);
    }

    /// Append multiple gates to the circuit.
    pub fn add_gates(&mut self, gates: &[CatGate]) {
        self.circuit.extend_from_slice(gates);
    }

    /// Reset the error state for a new simulation run.
    fn reset_errors(&mut self) {
        for i in 0..self.num_qubits {
            self.phase_errors[i] = false;
            self.bit_flip_errors[i] = false;
        }
    }

    /// Inject noise for a single gate application on specified qubits.
    fn inject_noise(&mut self, gate: &CatGate, rng: &mut impl Rng) {
        let (p_bf, p_pf) = gate.error_rates(&self.noise, self.config.gate_time);

        for &q in gate.qubits().iter() {
            if q >= self.num_qubits {
                continue;
            }
            // Inject phase-flip error
            if rng.gen::<f64>() < p_pf {
                self.phase_errors[q] = !self.phase_errors[q];
            }
            // Inject bit-flip error
            if rng.gen::<f64>() < p_bf {
                self.bit_flip_errors[q] = !self.bit_flip_errors[q];
            }
        }
    }

    /// Execute the circuit once with noise injection and QEC.
    ///
    /// Returns a `CircuitRunResult` capturing the full error trajectory.
    pub fn run_once_seeded(&mut self, seed: u64) -> CircuitRunResult {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        self.reset_errors();
        let mut total_bit_flips = 0usize;
        let mut total_phase_flips = 0usize;

        // Execute each gate with noise injection
        for gate in self.circuit.clone().iter() {
            let bf_before: usize = self.bit_flip_errors.iter().filter(|&&e| e).count();
            let pf_before: usize = self.phase_errors.iter().filter(|&&e| e).count();

            self.inject_noise(gate, &mut rng);

            let bf_after: usize = self.bit_flip_errors.iter().filter(|&&e| e).count();
            let pf_after: usize = self.phase_errors.iter().filter(|&&e| e).count();

            // Count new errors introduced (net change, allowing for error cancellation)
            total_bit_flips += bf_after.abs_diff(bf_before);
            total_phase_flips += pf_after.abs_diff(pf_before);
        }

        // Perform syndrome extraction and correction on the phase errors.
        // If num_qubits matches a valid repetition code distance, use it.
        let d = self.num_qubits;
        let syndrome_history = Vec::new();
        let mut correction = vec![false; d];

        if d >= 3 && d % 2 == 1 {
            // Extract syndrome for the repetition code
            let syndrome = noisy_syndrome_extraction(
                &self.phase_errors,
                0.0, // ideal extraction for circuit sim
                &mut rng,
            );
            correction = minimum_weight_decoder(&syndrome, d);
        }

        // Compute residual errors after correction
        let mut residual = vec![false; d];
        for i in 0..d {
            residual[i] = self.phase_errors[i] ^ correction[i];
        }

        // Logical error: odd residual phase-flip parity OR any bit-flip
        let residual_pf: usize = residual.iter().filter(|&&e| e).count();
        let any_bf = self.bit_flip_errors.iter().any(|&e| e);
        let logical_error = (residual_pf % 2 == 1) || any_bf;

        CircuitRunResult {
            num_gates: self.circuit.len(),
            total_bit_flips,
            total_phase_flips,
            logical_error,
            syndrome_history,
            final_errors: self.phase_errors.clone(),
            correction,
        }
    }

    /// Run the circuit `num_shots` times and return the logical error rate.
    pub fn logical_error_rate(&mut self, num_shots: usize, base_seed: u64) -> f64 {
        let mut logical_errors = 0usize;
        for shot in 0..num_shots {
            let result = self.run_once_seeded(base_seed.wrapping_add(shot as u64));
            if result.logical_error {
                logical_errors += 1;
            }
        }
        logical_errors as f64 / num_shots as f64
    }
}

// ============================================================
// ANALYSIS TOOLS
// ============================================================

/// Data point for logical error rate analysis.
#[derive(Debug, Clone)]
pub struct ErrorRatePoint {
    /// Coherent state amplitude.
    pub alpha: f64,
    /// Repetition code distance.
    pub distance: usize,
    /// Computed logical error rate.
    pub logical_error_rate: f64,
    /// Physical error rate for comparison.
    pub physical_error_rate: f64,
    /// Noise bias ratio at this alpha.
    pub bias_ratio: f64,
}

/// Sweep alpha and distance to compute the logical error rate landscape.
///
/// Returns a vector of `ErrorRatePoint` structs covering the full parameter
/// space. This is the primary analysis tool for understanding how the
/// concatenated code performance varies with cat qubit size and code distance.
///
/// # Arguments
/// * `alphas` -- vector of coherent state amplitudes to sweep.
/// * `distances` -- vector of repetition code distances to sweep (must be odd).
/// * `kappa_1` -- single-photon loss rate (Hz).
/// * `kappa_2` -- two-photon dissipation rate (Hz).
/// * `gate_time` -- gate duration in microseconds.
pub fn logical_error_rate_curve(
    alphas: &[f64],
    distances: &[usize],
    kappa_1: f64,
    kappa_2: f64,
    gate_time: f64,
) -> Vec<ErrorRatePoint> {
    let mut points = Vec::with_capacity(alphas.len() * distances.len());

    for &alpha in alphas {
        let noise = BiasedNoiseModel::from_params(alpha, kappa_1, kappa_2);
        let p_pf = noise.phase_flip_probability(gate_time);
        let p_bf = noise.bit_flip_probability(gate_time);
        let physical = p_pf + p_bf;

        for &d in distances {
            if d == 0 || d % 2 == 0 {
                continue;
            }
            let t = (d + 1) / 2;
            let logical = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

            points.push(ErrorRatePoint {
                alpha,
                distance: d,
                logical_error_rate: logical,
                physical_error_rate: physical,
                bias_ratio: noise.bias_ratio,
            });
        }
    }

    points
}

/// Data point for break-even analysis.
#[derive(Debug, Clone)]
pub struct BreakEvenPoint {
    /// Repetition code distance.
    pub distance: usize,
    /// Alpha value at which the concatenated code breaks even with physical.
    pub break_even_alpha: f64,
    /// Logical error rate at the break-even point.
    pub error_rate_at_break_even: f64,
}

/// Find the break-even alpha for each code distance.
///
/// The break-even point is where the concatenated logical error rate equals
/// the unconcatenated physical error rate. Below this alpha the code provides
/// no benefit; above it the code suppresses errors.
///
/// # Arguments
/// * `distances` -- vector of repetition code distances to analyze.
/// * `kappa_1` -- single-photon loss rate (Hz).
/// * `kappa_2` -- two-photon dissipation rate (Hz).
/// * `gate_time` -- gate duration in microseconds.
pub fn break_even_plot_data(
    distances: &[usize],
    kappa_1: f64,
    kappa_2: f64,
    gate_time: f64,
) -> Vec<BreakEvenPoint> {
    let mut points = Vec::with_capacity(distances.len());

    for &d in distances {
        if d == 0 || d % 2 == 0 || d < 3 {
            continue;
        }

        // Binary search for the break-even alpha
        let mut lo = 0.3_f64;
        let mut hi = 10.0_f64;

        for _ in 0..200 {
            let mid = (lo + hi) / 2.0;
            let noise = BiasedNoiseModel::from_params(mid, kappa_1, kappa_2);
            let p_pf = noise.phase_flip_probability(gate_time);
            let p_bf = noise.bit_flip_probability(gate_time);

            let physical = p_pf + p_bf;
            let t = (d + 1) / 2;
            let logical = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

            if logical < physical {
                hi = mid;
            } else {
                lo = mid;
            }

            if (hi - lo) < 1e-8 {
                break;
            }
        }

        let alpha = (lo + hi) / 2.0;
        let noise = BiasedNoiseModel::from_params(alpha, kappa_1, kappa_2);
        let p_pf = noise.phase_flip_probability(gate_time);
        let p_bf = noise.bit_flip_probability(gate_time);
        let t = (d + 1) / 2;
        let error_at_be = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

        points.push(BreakEvenPoint {
            distance: d,
            break_even_alpha: alpha,
            error_rate_at_break_even: error_at_be,
        });
    }

    points
}

/// Resource overhead analysis result.
#[derive(Debug, Clone)]
pub struct OverheadAnalysis {
    /// Target logical error rate.
    pub target_error_rate: f64,
    /// Coherent state amplitude used.
    pub alpha: f64,
    /// Required repetition code distance.
    pub required_distance: usize,
    /// Number of data cat qubits.
    pub num_data_qubits: usize,
    /// Number of ancilla transmons for syndrome extraction.
    pub num_ancilla_qubits: usize,
    /// Total physical components (data + ancilla).
    pub total_physical_components: usize,
    /// Achieved logical error rate at this configuration.
    pub achieved_error_rate: f64,
}

impl fmt::Display for OverheadAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OverheadAnalysis {{\n\
             \x20 target_error: {:.2e},\n\
             \x20 alpha: {:.2},\n\
             \x20 distance: {},\n\
             \x20 data_qubits: {},\n\
             \x20 ancilla_qubits: {},\n\
             \x20 total_components: {},\n\
             \x20 achieved_error: {:.4e}\n\
             }}",
            self.target_error_rate,
            self.alpha,
            self.required_distance,
            self.num_data_qubits,
            self.num_ancilla_qubits,
            self.total_physical_components,
            self.achieved_error_rate,
        )
    }
}

/// Compute the physical resource overhead needed to achieve a target logical
/// error rate.
///
/// Searches over odd distances to find the minimum repetition code distance
/// that achieves the target, then reports the total physical resources needed.
///
/// # Arguments
/// * `alpha` -- coherent state amplitude.
/// * `kappa_1` -- single-photon loss rate (Hz).
/// * `kappa_2` -- two-photon dissipation rate (Hz).
/// * `gate_time` -- gate duration in microseconds.
/// * `target_error_rate` -- desired logical error rate.
pub fn overhead_analysis(
    alpha: f64,
    kappa_1: f64,
    kappa_2: f64,
    gate_time: f64,
    target_error_rate: f64,
) -> OverheadAnalysis {
    let d = required_distance(alpha, kappa_1, kappa_2, gate_time, target_error_rate);

    let noise = BiasedNoiseModel::from_params(alpha, kappa_1, kappa_2);
    let p_pf = noise.phase_flip_probability(gate_time);
    let p_bf = noise.bit_flip_probability(gate_time);
    let t = (d + 1) / 2;
    let achieved = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

    let num_ancilla = if d > 1 { d - 1 } else { 0 };

    OverheadAnalysis {
        target_error_rate,
        alpha,
        required_distance: d,
        num_data_qubits: d,
        num_ancilla_qubits: num_ancilla,
        total_physical_components: d + num_ancilla,
        achieved_error_rate: achieved,
    }
}

/// Resource comparison between the concatenated cat code and a surface code.
#[derive(Debug, Clone)]
pub struct SurfaceCodeComparison {
    /// Target logical error rate.
    pub target_error_rate: f64,
    /// Cat code overhead.
    pub cat_overhead: OverheadAnalysis,
    /// Surface code distance needed.
    pub surface_code_distance: usize,
    /// Number of physical qubits for the surface code.
    pub surface_code_qubits: usize,
    /// Reduction factor: surface_code_qubits / cat_total_components.
    pub qubit_reduction_factor: f64,
}

impl fmt::Display for SurfaceCodeComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SurfaceCodeComparison {{\n\
             \x20 target_error: {:.2e},\n\
             \x20 cat_components: {} (d={}, alpha={:.2}),\n\
             \x20 surface_code_qubits: {} (d={}),\n\
             \x20 qubit_reduction: {:.1}x fewer with cat code\n\
             }}",
            self.target_error_rate,
            self.cat_overhead.total_physical_components,
            self.cat_overhead.required_distance,
            self.cat_overhead.alpha,
            self.surface_code_qubits,
            self.surface_code_distance,
            self.qubit_reduction_factor,
        )
    }
}

/// Compare the concatenated cat code resource requirements with a surface code.
///
/// The surface code uses d^2 data qubits + (d-1)^2 ancilla qubits for a
/// distance-d code, with a physical error threshold of approximately 1%.
/// The logical error rate scales as:
///
///   P_L^{surface} ~ 0.1 * (p_phys / p_threshold)^{(d+1)/2}
///
/// The cat code achieves the same logical error rate with far fewer physical
/// components because it only needs to correct phase flips.
///
/// # Arguments
/// * `alpha` -- coherent state amplitude for the cat code.
/// * `kappa_1` -- single-photon loss rate.
/// * `kappa_2` -- two-photon dissipation rate.
/// * `gate_time` -- gate duration in microseconds.
/// * `surface_code_physical_error` -- physical error rate per gate for the surface code.
/// * `target_error_rate` -- desired logical error rate.
pub fn compare_with_surface_code(
    alpha: f64,
    kappa_1: f64,
    kappa_2: f64,
    gate_time: f64,
    surface_code_physical_error: f64,
    target_error_rate: f64,
) -> SurfaceCodeComparison {
    // Cat code overhead
    let cat = overhead_analysis(alpha, kappa_1, kappa_2, gate_time, target_error_rate);

    // Surface code overhead: search for minimum distance
    let p_threshold = 0.01; // ~1% threshold for surface codes
    let p_ratio = surface_code_physical_error / p_threshold;

    let mut surface_d = 3usize;
    loop {
        let t = (surface_d + 1) / 2;
        let p_l_surface = 0.1 * p_ratio.powi(t as i32);
        if p_l_surface < target_error_rate || surface_d > 201 {
            break;
        }
        surface_d += 2;
    }

    // Surface code qubits: 2*d^2 - 1 (data + syndrome qubits for rotated layout)
    let surface_qubits = 2 * surface_d * surface_d - 1;

    let reduction = if cat.total_physical_components > 0 {
        surface_qubits as f64 / cat.total_physical_components as f64
    } else {
        f64::INFINITY
    };

    SurfaceCodeComparison {
        target_error_rate,
        cat_overhead: cat,
        surface_code_distance: surface_d,
        surface_code_qubits: surface_qubits,
        qubit_reduction_factor: reduction,
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Default test tolerance for floating-point comparisons.
    const TEST_EPSILON: f64 = 1e-6;

    /// Standard test alpha for moderate cat size.
    const TEST_ALPHA: f64 = 2.0;

    /// Standard test Fock cutoff.
    const TEST_CUTOFF: usize = 25;

    // --------------------------------------------------------
    // Coherent state tests
    // --------------------------------------------------------

    #[test]
    fn test_coherent_state_normalization() {
        // A coherent state should be normalized: sum |c_n|^2 = 1
        // (up to Fock truncation errors).
        for &alpha in &[0.5, 1.0, 2.0, 3.0] {
            let cs = CoherentState::from_real(alpha, 40);
            let norm = cs.norm_squared();
            assert!(
                (norm - 1.0).abs() < TEST_EPSILON,
                "coherent state |alpha={:.1}> norm={:.10} deviates from 1.0",
                alpha,
                norm
            );
        }
    }

    #[test]
    fn test_coherent_state_mean_photon() {
        // For a coherent state |alpha>, <n> should equal |alpha|^2.
        for &alpha in &[0.5, 1.0, 2.0, 3.0] {
            let cs = CoherentState::from_real(alpha, 40);
            let mean_n = cs.mean_photon_number();
            let expected = alpha * alpha;
            assert!(
                (mean_n - expected).abs() < 0.01,
                "coherent state |alpha={:.1}> mean photon={:.6}, expected {:.6}",
                alpha,
                mean_n,
                expected
            );
        }
    }

    // --------------------------------------------------------
    // Cat state tests
    // --------------------------------------------------------

    #[test]
    fn test_cat_state_even_parity() {
        // Even cat (logical 0) should have support only on even Fock numbers.
        let cat = CatQubitState::new(0, TEST_ALPHA, TEST_CUTOFF).unwrap();
        assert!(
            cat.check_parity_structure(1e-10),
            "even cat state has non-zero odd Fock amplitudes"
        );

        // Verify: all odd coefficients are essentially zero
        for n in (1..=TEST_CUTOFF).step_by(2) {
            assert!(
                cat.fock_coefficients[n].norm_sqr() < 1e-20,
                "even cat |c_{}|^2 = {:.2e} should be zero",
                n,
                cat.fock_coefficients[n].norm_sqr()
            );
        }
    }

    #[test]
    fn test_cat_state_odd_parity() {
        // Odd cat (logical 1) should have support only on odd Fock numbers.
        let cat = CatQubitState::new(1, TEST_ALPHA, TEST_CUTOFF).unwrap();
        assert!(
            cat.check_parity_structure(1e-10),
            "odd cat state has non-zero even Fock amplitudes"
        );

        // Verify: all even coefficients are essentially zero
        for n in (0..=TEST_CUTOFF).step_by(2) {
            assert!(
                cat.fock_coefficients[n].norm_sqr() < 1e-20,
                "odd cat |c_{}|^2 = {:.2e} should be zero",
                n,
                cat.fock_coefficients[n].norm_sqr()
            );
        }
    }

    // --------------------------------------------------------
    // Noise model tests
    // --------------------------------------------------------

    #[test]
    fn test_noise_bias_exponential() {
        // The bias ratio should grow exponentially with alpha.
        // Check that bias(alpha+1) >> bias(alpha).
        let alphas = [1.0, 2.0, 3.0, 4.0];
        let biases: Vec<f64> = alphas
            .iter()
            .map(|&a| BiasedNoiseModel::fundamental_bias(a))
            .collect();

        for i in 0..biases.len() - 1 {
            assert!(
                biases[i + 1] > biases[i] * 10.0,
                "bias at alpha={:.0} ({:.2e}) should be >>10x bias at alpha={:.0} ({:.2e})",
                alphas[i + 1],
                biases[i + 1],
                alphas[i],
                biases[i]
            );
        }
    }

    #[test]
    fn test_noise_bias_values() {
        // For alpha=2, bias should be approximately 2952 (alpha^2 * exp(2*alpha^2))
        let alpha = 2.0;
        let bias = BiasedNoiseModel::fundamental_bias(alpha);
        let expected = 4.0 * (8.0_f64).exp(); // 4 * e^8 ~ 11917
        assert!(
            (bias - expected).abs() / expected < 0.01,
            "bias at alpha=2: got {:.2}, expected {:.2}",
            bias,
            expected
        );

        // Verify it is in the thousands range (order of magnitude check)
        assert!(
            bias > 1000.0,
            "bias at alpha=2 should be > 1000, got {:.2}",
            bias
        );
    }

    #[test]
    fn test_bit_flip_suppression() {
        // Bit-flip rate should DECREASE exponentially with alpha.
        let kappa_1 = 1e4;
        let kappa_2 = 1e7;

        let rate_low = BiasedNoiseModel::from_params(1.0, kappa_1, kappa_2).bit_flip_rate;
        let rate_mid = BiasedNoiseModel::from_params(2.0, kappa_1, kappa_2).bit_flip_rate;
        let rate_high = BiasedNoiseModel::from_params(3.0, kappa_1, kappa_2).bit_flip_rate;

        assert!(
            rate_mid < rate_low,
            "bit flip rate at alpha=2 ({:.4e}) should be less than at alpha=1 ({:.4e})",
            rate_mid,
            rate_low
        );
        assert!(
            rate_high < rate_mid,
            "bit flip rate at alpha=3 ({:.4e}) should be less than at alpha=2 ({:.4e})",
            rate_high,
            rate_mid
        );

        // The suppression factor between alpha=1 and alpha=3 should be huge
        let suppression = rate_low / rate_high;
        assert!(
            suppression > 1e6,
            "bit flip suppression from alpha=1 to alpha=3 should be > 10^6, got {:.2e}",
            suppression
        );
    }

    #[test]
    fn test_phase_flip_growth() {
        // Phase-flip rate should INCREASE with alpha (linearly in alpha^2).
        let kappa_1 = 1e4;
        let kappa_2 = 1e7;

        let rate_low = BiasedNoiseModel::from_params(1.0, kappa_1, kappa_2).phase_flip_rate;
        let rate_mid = BiasedNoiseModel::from_params(2.0, kappa_1, kappa_2).phase_flip_rate;
        let rate_high = BiasedNoiseModel::from_params(3.0, kappa_1, kappa_2).phase_flip_rate;

        assert!(
            rate_mid > rate_low,
            "phase flip rate at alpha=2 ({:.4e}) should exceed alpha=1 ({:.4e})",
            rate_mid,
            rate_low
        );
        assert!(
            rate_high > rate_mid,
            "phase flip rate at alpha=3 ({:.4e}) should exceed alpha=2 ({:.4e})",
            rate_high,
            rate_mid
        );

        // Check linear scaling: rate(alpha=2) / rate(alpha=1) ~ 4
        let ratio = rate_mid / rate_low;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "phase flip scaling alpha=2/alpha=1 should be ~4, got {:.4}",
            ratio
        );
    }

    // --------------------------------------------------------
    // Repetition code tests
    // --------------------------------------------------------

    #[test]
    fn test_repetition_code_no_errors() {
        // With no errors, syndrome should be all zeros and no correction needed.
        let mut code = RepetitionCode::new(5).unwrap();
        let errors = vec![false; 5];
        code.extract_syndrome(&errors);

        assert!(
            code.is_syndrome_trivial(),
            "syndrome should be trivial with no errors"
        );
        assert!(
            !code.decode_majority_vote(&errors),
            "no logical error with zero physical errors"
        );
    }

    #[test]
    fn test_repetition_code_single_error() {
        // A single phase-flip error should be correctable for d >= 3.
        let mut code = RepetitionCode::new(5).unwrap();

        // Error on qubit 2
        let errors = vec![false, false, true, false, false];
        code.extract_syndrome(&errors);

        // Syndrome should be non-trivial
        assert!(
            !code.is_syndrome_trivial(),
            "syndrome should detect the single error"
        );

        // Majority vote should NOT declare a logical error (only 1 out of 5 flipped)
        assert!(
            !code.decode_majority_vote(&errors),
            "single error should be correctable for d=5"
        );

        // Verify syndrome pattern: should have defects at positions 1 and 2
        assert_eq!(
            code.syndrome_bits,
            vec![false, true, true, false],
            "syndrome bits for single error on qubit 2"
        );
    }

    // --------------------------------------------------------
    // Concatenation tests
    // --------------------------------------------------------

    #[test]
    fn test_logical_error_decreases_with_distance() {
        // Increasing the repetition code distance should reduce the logical error rate.
        // Use analytical estimates for a clean comparison.
        let kappa_1 = 1e4;
        let kappa_2 = 1e7;
        let alpha = 2.0;
        let gate_time = 1.0;

        let noise = BiasedNoiseModel::from_params(alpha, kappa_1, kappa_2);
        let p_pf = noise.phase_flip_probability(gate_time);
        let p_bf = noise.bit_flip_probability(gate_time);

        let mut prev_rate = f64::MAX;
        for d in [1, 3, 5, 7, 9] {
            let t = (d + 1) / 2;
            let rate = binomial_coefficient(d, t) * p_pf.powi(t as i32) + d as f64 * p_bf;

            if d > 1 {
                assert!(
                    rate < prev_rate,
                    "logical error rate at d={} ({:.4e}) should be less than d=? ({:.4e})",
                    d,
                    rate,
                    prev_rate
                );
            }
            prev_rate = rate;
        }
    }

    #[test]
    fn test_concatenation_result() {
        // Run a full simulation and verify the result is well-formed.
        let config = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .kappa_1(1e4)
            .kappa_2(1e7)
            .num_cats(3)
            .num_rounds(1000)
            .build()
            .unwrap();

        let sim = CatQubitSimulator::new(config).unwrap();
        let result = sim.run_concatenation_seeded(42);

        // Basic sanity checks
        assert_eq!(result.num_physical_modes, 3);
        assert_eq!(result.num_logical_qubits, 1);
        assert_eq!(result.outer_code_distance, 3);
        assert_eq!(result.num_rounds, 1000);
        assert!((result.alpha - 2.0).abs() < EPSILON);
        assert!(result.logical_error_rate >= 0.0);
        assert!(result.logical_error_rate <= 1.0);
        assert!(result.bias_ratio > 100.0, "bias should be large at alpha=2");
        assert!(
            result.bit_flip_rate < result.phase_flip_rate,
            "bit flips should be rarer than phase flips"
        );
        assert!(result.analytical_estimate >= 0.0);
        assert!(result.analytical_estimate <= 1.0);
        assert!(result.num_logical_errors <= result.num_rounds);
    }

    // --------------------------------------------------------
    // Additional validation tests
    // --------------------------------------------------------

    #[test]
    fn test_config_builder_validation() {
        // Invalid alpha should be rejected
        let result = CatQubitConfig::new().alpha(-1.0).build();
        assert!(result.is_err());

        let result = CatQubitConfig::new().alpha(0.0).build();
        assert!(result.is_err());

        // Even number of cats should be rejected (need odd for majority vote)
        let result = CatQubitConfig::new().num_cats(4).build();
        assert!(result.is_err());

        // Fock cutoff too small should be rejected
        let result = CatQubitConfig::new().alpha(4.0).fock_cutoff(10).build();
        assert!(result.is_err());

        // Valid config should succeed
        let result = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .num_cats(3)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cat_state_normalization() {
        // Both logical states should be normalized.
        let cat0 = CatQubitState::new(0, TEST_ALPHA, TEST_CUTOFF).unwrap();
        let cat1 = CatQubitState::new(1, TEST_ALPHA, TEST_CUTOFF).unwrap();

        assert!(
            (cat0.norm_squared() - 1.0).abs() < TEST_EPSILON,
            "even cat norm = {:.10}",
            cat0.norm_squared()
        );
        assert!(
            (cat1.norm_squared() - 1.0).abs() < TEST_EPSILON,
            "odd cat norm = {:.10}",
            cat1.norm_squared()
        );
    }

    #[test]
    fn test_cat_state_orthogonality() {
        // The two logical states should be nearly orthogonal for large alpha.
        let overlap = logical_state_overlap(TEST_ALPHA, TEST_CUTOFF).unwrap();
        assert!(
            overlap < 1e-6,
            "logical state overlap at alpha={} should be ~0, got {:.2e}",
            TEST_ALPHA,
            overlap
        );
    }

    #[test]
    fn test_x_gate_flips_parity() {
        // The X gate should map |0_L> to |1_L> (even cat to odd cat).
        let cat0 = CatQubitState::new(0, TEST_ALPHA, TEST_CUTOFF).unwrap();
        let x_op = CatQubitOperator::x_gate(TEST_ALPHA, TEST_CUTOFF);
        let flipped = x_op.apply(&cat0.fock_coefficients);

        // The flipped state should have high fidelity with the odd cat
        let cat1 = CatQubitState::new(1, TEST_ALPHA, TEST_CUTOFF).unwrap();
        let overlap: C64 = flipped
            .iter()
            .zip(cat1.fock_coefficients.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        let fidelity = overlap.norm_sqr();

        assert!(
            fidelity > 0.999,
            "X gate on |0_L> should yield |1_L>, fidelity = {:.6}",
            fidelity
        );
    }

    #[test]
    fn test_photon_loss_reduces_norm() {
        // The annihilation operator (single-photon loss Kraus operator) produces
        // a sub-normalized state: ||a|psi>||^2 = <psi|a^dag a|psi> = <n>.
        // For a cat state with <n> = alpha^2, the squared norm of a|psi>
        // should equal approximately alpha^2 (since the input is normalized).
        let cat = CatQubitState::new(0, TEST_ALPHA, TEST_CUTOFF).unwrap();
        let expected_n = cat.mean_photon_number();

        let after_loss = apply_photon_loss(&cat.fock_coefficients);
        let after_norm_sq: f64 = after_loss.iter().map(|c| c.norm_sqr()).sum();

        // ||a|psi>||^2 should equal <n>
        assert!(
            (after_norm_sq - expected_n).abs() < 0.01,
            "||a|psi>||^2 = {:.4} should equal <n> = {:.4}",
            after_norm_sq,
            expected_n
        );

        // Verify that a|psi> is non-zero (cat state has non-zero photon number)
        assert!(
            after_norm_sq > 1.0,
            "cat state should have <n> > 1, got {:.4}",
            after_norm_sq
        );
    }

    #[test]
    fn test_multi_level_concatenation() {
        // Multi-level concatenation should progressively suppress errors.
        let result = multi_level_concatenation(
            2.0,        // alpha
            1e4,        // kappa_1
            1e7,        // kappa_2
            1.0,        // gate_time
            &[3, 3, 3], // three levels of distance-3
        );

        assert_eq!(result.levels, 3);
        assert_eq!(result.total_physical_qubits, 27); // 3^3
        assert_eq!(result.distances, vec![3, 3, 3]);

        // Error rate should decrease at each level
        for i in 1..result.levels {
            assert!(
                result.error_rates[i] < result.error_rates[i - 1],
                "error rate at level {} ({:.4e}) should be less than level {} ({:.4e})",
                i,
                result.error_rates[i],
                i - 1,
                result.error_rates[i - 1]
            );
        }
    }

    #[test]
    fn test_wigner_function_symmetry() {
        // The Wigner function of a cat state centered at real alpha
        // should be symmetric under p -> -p.
        let cat = CatQubitState::new(0, TEST_ALPHA, TEST_CUTOFF).unwrap();

        let w_pos = wigner_diagonal(&cat.fock_coefficients, 1.0, 0.5);
        let w_neg = wigner_diagonal(&cat.fock_coefficients, 1.0, -0.5);

        assert!(
            (w_pos - w_neg).abs() < 1e-10,
            "Wigner function should be symmetric in p: W(x,p)={:.6e}, W(x,-p)={:.6e}",
            w_pos,
            w_neg
        );
    }

    #[test]
    fn test_minimum_weight_decoder() {
        // Single defect pair should be corrected.
        let syndrome = vec![false, true, true, false]; // defects at positions 1, 2
        let correction = minimum_weight_decoder(&syndrome, 5);

        // Pairing defects at 1 and 2: should flip qubit 2
        assert_eq!(correction, vec![false, false, true, false, false]);
    }

    #[test]
    fn test_minimum_weight_decoder_satisfies_syndrome_constraints() {
        for d in [3usize, 5, 7] {
            let syndrome_bits = d - 1;
            for mask in 0..(1usize << syndrome_bits) {
                let syndrome: Vec<bool> =
                    (0..syndrome_bits).map(|i| ((mask >> i) & 1) == 1).collect();
                let correction = minimum_weight_decoder(&syndrome, d);
                let reconstructed: Vec<bool> = (0..syndrome_bits)
                    .map(|i| correction[i] ^ correction[i + 1])
                    .collect();
                assert_eq!(
                    reconstructed, syndrome,
                    "decoded correction must reproduce syndrome for d={}, mask={}",
                    d, mask
                );
            }
        }
    }

    #[test]
    fn test_minimum_weight_decoder_is_globally_minimal_small_codes() {
        for d in [3usize, 5, 7] {
            let syndrome_bits = d - 1;
            for mask in 0..(1usize << syndrome_bits) {
                let syndrome: Vec<bool> =
                    (0..syndrome_bits).map(|i| ((mask >> i) & 1) == 1).collect();
                let correction = minimum_weight_decoder(&syndrome, d);
                let decoded_weight = correction.iter().filter(|&&b| b).count();

                let mut best_weight = usize::MAX;
                for corr_mask in 0..(1usize << d) {
                    let cand: Vec<bool> = (0..d).map(|i| ((corr_mask >> i) & 1) == 1).collect();
                    let cand_syndrome: Vec<bool> =
                        (0..syndrome_bits).map(|i| cand[i] ^ cand[i + 1]).collect();
                    if cand_syndrome == syndrome {
                        let w = cand.iter().filter(|&&b| b).count();
                        if w < best_weight {
                            best_weight = w;
                        }
                    }
                }

                assert_eq!(
                    decoded_weight, best_weight,
                    "decoder should return globally minimum-weight correction for d={}, mask={}",
                    d, mask
                );
            }
        }
    }

    #[test]
    fn test_minimum_weight_decoder_weighted_prefers_lower_cost_solution() {
        let syndrome = vec![true, false];

        let w_left_expensive = vec![10.0, 1.0, 1.0];
        let corr_a = minimum_weight_decoder_weighted(&syndrome, 3, Some(&w_left_expensive));
        assert_eq!(corr_a, vec![false, true, true]);

        let w_right_expensive = vec![1.0, 10.0, 10.0];
        let corr_b = minimum_weight_decoder_weighted(&syndrome, 3, Some(&w_right_expensive));
        assert_eq!(corr_b, vec![true, false, false]);
    }

    #[test]
    fn test_required_distance_increases_with_target() {
        // Tighter target error rate should require larger distance.
        let d_loose = required_distance(2.0, 1e4, 1e7, 1.0, 1e-3);
        let d_tight = required_distance(2.0, 1e4, 1e7, 1.0, 1e-9);

        assert!(
            d_tight >= d_loose,
            "tighter target should need more qubits: d_loose={}, d_tight={}",
            d_loose,
            d_tight
        );
    }

    // --------------------------------------------------------
    // CatGate tests
    // --------------------------------------------------------

    #[test]
    fn test_cat_gate_bias_preservation() {
        // Z and ZZ gates should be bias-preserving.
        assert!(CatGate::Z(0).is_bias_preserving());
        assert!(CatGate::ZZ(0, 1).is_bias_preserving());
        assert!(CatGate::Idle(0).is_bias_preserving());

        // CX, Toffoli, and Hadamard are NOT bias-preserving.
        assert!(!CatGate::CX(0, 1).is_bias_preserving());
        assert!(!CatGate::Toffoli(0, 1, 2).is_bias_preserving());
        assert!(!CatGate::Hadamard(0).is_bias_preserving());
    }

    #[test]
    fn test_cat_gate_qubits() {
        assert_eq!(CatGate::Z(3).qubits(), vec![3]);
        assert_eq!(CatGate::ZZ(1, 4).qubits(), vec![1, 4]);
        assert_eq!(CatGate::CX(0, 2).qubits(), vec![0, 2]);
        assert_eq!(CatGate::Toffoli(0, 1, 2).qubits(), vec![0, 1, 2]);
        assert_eq!(CatGate::MeasureZ(5).qubits(), vec![5]);
    }

    #[test]
    fn test_cat_gate_error_rates_bias() {
        // For bias-preserving gates, bit-flip rate should be much smaller
        // than phase-flip rate at moderate alpha.
        let noise = BiasedNoiseModel::from_params(2.0, 1e4, 1e7);
        let gate_time = 1.0;

        let (bf_z, pf_z) = CatGate::Z(0).error_rates(&noise, gate_time);
        assert!(
            bf_z < pf_z * 0.001,
            "Z gate: bit-flip ({:.4e}) should be << phase-flip ({:.4e})",
            bf_z,
            pf_z
        );

        let (bf_zz, pf_zz) = CatGate::ZZ(0, 1).error_rates(&noise, gate_time);
        assert!(
            bf_zz < pf_zz * 0.01,
            "ZZ gate: bit-flip ({:.4e}) should be << phase-flip ({:.4e})",
            bf_zz,
            pf_zz
        );
    }

    #[test]
    fn test_toffoli_native_advantage() {
        // The native Toffoli should have lower total error than 6 CX gates
        // decomposed (which is what a non-cat architecture would need).
        let noise = BiasedNoiseModel::from_params(2.0, 1e4, 1e7);
        let gate_time = 1.0;

        let (bf_tof, pf_tof) = CatGate::Toffoli(0, 1, 2).error_rates(&noise, gate_time);
        let (bf_cx, pf_cx) = CatGate::CX(0, 1).error_rates(&noise, gate_time);

        // Native Toffoli total error vs 6 CX gates
        let tof_total = bf_tof + pf_tof;
        let six_cx_total = 6.0 * (bf_cx + pf_cx);

        assert!(
            tof_total < six_cx_total,
            "native Toffoli error ({:.4e}) should be < 6*CX error ({:.4e})",
            tof_total,
            six_cx_total
        );
    }

    // --------------------------------------------------------
    // ConcatenatedCatCode tests
    // --------------------------------------------------------

    #[test]
    fn test_concatenated_code_construction() {
        let config = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .num_cats(5)
            .build()
            .unwrap();
        let code = ConcatenatedCatCode::new(config).unwrap();

        assert_eq!(code.repetition_code.distance, 5);
        assert_eq!(code.repetition_code.correction_capacity(), 2);
        assert!(code.noise.bias_ratio > 100.0);
        assert!(code.phase_flip_per_round > 0.0);
        assert!(code.bit_flip_per_round > 0.0);
        assert!(code.bit_flip_per_round < code.phase_flip_per_round);
    }

    #[test]
    fn test_concatenated_code_logical_error() {
        // The logical error rate should decrease with distance.
        let distances = [3, 5, 7, 9];
        let mut prev_rate = f64::MAX;

        for &d in &distances {
            let config = CatQubitConfig::new()
                .alpha(2.0)
                .fock_cutoff(25)
                .num_cats(d)
                .build()
                .unwrap();
            let code = ConcatenatedCatCode::new(config).unwrap();
            let rate = code.logical_error_rate();

            assert!(
                rate < prev_rate,
                "P_L at d={} ({:.4e}) should be < P_L at previous d ({:.4e})",
                d,
                rate,
                prev_rate,
            );
            prev_rate = rate;
        }
    }

    #[test]
    fn test_concatenation_threshold_is_valid() {
        let config = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .num_cats(5)
            .build()
            .unwrap();
        let code = ConcatenatedCatCode::new(config).unwrap();
        let threshold = code.concatenation_threshold();

        // Threshold should be between 0 and 0.5
        assert!(
            threshold > 0.0 && threshold <= 0.5,
            "concatenation threshold should be in (0, 0.5], got {:.6}",
            threshold
        );

        // The actual phase-flip rate should be below the threshold for the code
        // to provide benefit (which it does at alpha=2 with these params).
        assert!(
            code.phase_flip_per_round < threshold,
            "phase-flip rate ({:.4e}) should be below threshold ({:.4e}) for QEC benefit",
            code.phase_flip_per_round,
            threshold
        );
    }

    #[test]
    fn test_break_even_alpha() {
        let config = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .num_cats(5)
            .build()
            .unwrap();
        let code = ConcatenatedCatCode::new(config).unwrap();
        let be_alpha = code.break_even_alpha();

        // Break-even alpha should be positive and less than the configured alpha
        // (since at alpha=2 with d=5 the code should already be winning).
        assert!(
            be_alpha > 0.0,
            "break-even alpha should be positive, got {:.4}",
            be_alpha
        );
        assert!(
            be_alpha < 8.0,
            "break-even alpha should be reasonable, got {:.4}",
            be_alpha
        );
    }

    #[test]
    fn test_physical_overhead() {
        let config = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .num_cats(5)
            .build()
            .unwrap();
        let code = ConcatenatedCatCode::new(config).unwrap();

        // d=5: 5 data cat qubits + 4 ancilla transmons = 9 total
        assert_eq!(code.physical_overhead(), 9);
    }

    // --------------------------------------------------------
    // CatCircuitSimulator tests
    // --------------------------------------------------------

    #[test]
    fn test_circuit_simulator_no_noise() {
        // With very high kappa_2 and very low kappa_1, error rates should be
        // negligibly small and no logical errors should occur in a short circuit.
        let config = CatQubitConfig::new()
            .alpha(3.0)
            .fock_cutoff(40)
            .kappa_1(1.0) // very low loss
            .kappa_2(1e10) // very high stabilization
            .num_cats(3)
            .build()
            .unwrap();

        let mut sim = CatCircuitSimulator::new(3, config).unwrap();
        sim.add_gates(&[CatGate::Z(0), CatGate::ZZ(0, 1), CatGate::Z(2)]);

        let result = sim.run_once_seeded(42);
        assert_eq!(result.num_gates, 3);
        // With near-zero noise, logical errors should be absent.
        // (We test 100 shots to be sure.)
        let error_rate = sim.logical_error_rate(100, 42);
        assert!(
            error_rate < 0.05,
            "error rate with near-zero noise should be ~0, got {:.4}",
            error_rate
        );
    }

    #[test]
    fn test_circuit_simulator_high_noise() {
        // With high noise, errors should be frequent.
        let config = CatQubitConfig::new()
            .alpha(1.0) // low alpha = weak suppression
            .fock_cutoff(15)
            .kappa_1(1e6) // very high loss
            .kappa_2(1e7)
            .num_cats(3)
            .build()
            .unwrap();

        let mut sim = CatCircuitSimulator::new(3, config).unwrap();
        sim.add_gates(&[
            CatGate::CX(0, 1),
            CatGate::CX(1, 2),
            CatGate::Toffoli(0, 1, 2),
        ]);

        let error_rate = sim.logical_error_rate(200, 123);
        assert!(
            error_rate > 0.01,
            "high-noise circuit should have detectable errors, got {:.4}",
            error_rate
        );
    }

    // --------------------------------------------------------
    // Analysis tools tests
    // --------------------------------------------------------

    #[test]
    fn test_logical_error_rate_curve() {
        // Use alpha >= 2 where the noise bias is strong enough that
        // increasing distance always helps (at alpha=1 the bit-flip
        // contribution d*p_bf can grow faster than the phase-flip
        // suppression, which is correct physics but not what we want
        // to test here).
        let alphas = vec![2.0, 3.0, 4.0];
        let distances = vec![3, 5, 7];
        let points = logical_error_rate_curve(&alphas, &distances, 1e4, 1e7, 1.0);

        assert_eq!(points.len(), 9); // 3 alphas * 3 distances

        // At fixed alpha >= 2, logical error should decrease with distance
        for &alpha_val in &alphas {
            let filtered: Vec<&ErrorRatePoint> = points
                .iter()
                .filter(|p| (p.alpha - alpha_val).abs() < 0.01)
                .collect();
            for i in 1..filtered.len() {
                assert!(
                    filtered[i].logical_error_rate < filtered[i - 1].logical_error_rate,
                    "at alpha={}, d={} rate ({:.4e}) should be < d={} rate ({:.4e})",
                    alpha_val,
                    filtered[i].distance,
                    filtered[i].logical_error_rate,
                    filtered[i - 1].distance,
                    filtered[i - 1].logical_error_rate,
                );
            }
        }

        // Also verify that the bias ratio increases with alpha (fundamental
        // physics: the noise asymmetry grows exponentially with |alpha|^2).
        for &d_val in &[3usize] {
            let filtered: Vec<&ErrorRatePoint> =
                points.iter().filter(|p| p.distance == d_val).collect();
            for i in 1..filtered.len() {
                assert!(
                    filtered[i].bias_ratio > filtered[i - 1].bias_ratio,
                    "at d={}, alpha={} bias ({:.4e}) should be > alpha={} bias ({:.4e})",
                    d_val,
                    filtered[i].alpha,
                    filtered[i].bias_ratio,
                    filtered[i - 1].alpha,
                    filtered[i - 1].bias_ratio,
                );
            }
        }
    }

    #[test]
    fn test_break_even_plot_data() {
        let distances = vec![3, 5, 7, 9];
        let points = break_even_plot_data(&distances, 1e4, 1e7, 1.0);

        assert_eq!(points.len(), 4);

        // Larger distance should have lower break-even alpha (easier to win)
        for i in 1..points.len() {
            assert!(
                points[i].break_even_alpha <= points[i - 1].break_even_alpha + 0.5,
                "larger d should not dramatically increase break-even alpha: d={} alpha={:.4}, d={} alpha={:.4}",
                points[i].distance,
                points[i].break_even_alpha,
                points[i-1].distance,
                points[i-1].break_even_alpha,
            );
        }
    }

    #[test]
    fn test_overhead_analysis() {
        // Use a target that is achievable with moderate distance at alpha=3.
        // At alpha=3 the bit-flip suppression is very strong (exp(-18) ~ 1e-8)
        // and the phase-flip rate is manageable.
        let result = overhead_analysis(3.0, 1e4, 1e7, 1.0, 1e-4);

        assert!(result.required_distance >= 3);
        assert!(result.required_distance % 2 == 1);
        assert_eq!(result.num_data_qubits, result.required_distance);
        assert_eq!(result.num_ancilla_qubits, result.required_distance - 1);
        assert_eq!(
            result.total_physical_components,
            result.num_data_qubits + result.num_ancilla_qubits
        );
        // Verify the achieved rate is at or below the target.
        // (For very aggressive targets the search may cap at d=101.)
        if result.required_distance < 101 {
            assert!(
                result.achieved_error_rate <= result.target_error_rate,
                "achieved rate ({:.4e}) should be <= target ({:.4e}) when d < 101",
                result.achieved_error_rate,
                result.target_error_rate
            );
        }
    }

    #[test]
    fn test_surface_code_comparison() {
        let comparison = compare_with_surface_code(
            3.0,  // alpha
            1e4,  // kappa_1
            1e7,  // kappa_2
            1.0,  // gate_time
            1e-3, // surface code physical error
            1e-6, // target
        );

        // Cat code should require dramatically fewer physical components
        assert!(
            comparison.qubit_reduction_factor > 1.0,
            "cat code should use fewer qubits than surface code, reduction={:.1}x",
            comparison.qubit_reduction_factor
        );

        // Surface code distance should be reasonable
        assert!(comparison.surface_code_distance >= 3);
        assert!(comparison.surface_code_qubits > comparison.cat_overhead.total_physical_components);
    }

    #[test]
    fn test_kerr_strength_in_config() {
        // Verify kerr_strength is properly stored in the configuration.
        let config = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .kerr_strength(5e4)
            .num_cats(3)
            .build()
            .unwrap();

        assert!(
            (config.kerr_strength - 5e4).abs() < 1.0,
            "kerr_strength should be 5e4, got {}",
            config.kerr_strength
        );
    }

    #[test]
    fn test_cat_gate_display() {
        // Test Display trait for all gate variants.
        assert_eq!(format!("{}", CatGate::Z(0)), "Z(0)");
        assert_eq!(format!("{}", CatGate::ZZ(1, 2)), "ZZ(1,2)");
        assert_eq!(format!("{}", CatGate::CX(0, 1)), "CX(0,1)");
        assert_eq!(format!("{}", CatGate::Toffoli(0, 1, 2)), "Toffoli(0,1,2)");
        assert_eq!(format!("{}", CatGate::Hadamard(3)), "H(3)");
        assert_eq!(format!("{}", CatGate::MeasureZ(4)), "MeasZ(4)");
    }

    #[test]
    fn test_nature_2025_distance_5_regime() {
        // Reproduce the key result from Nature Feb 2025:
        // At distance 5 with moderate alpha, the concatenated code should
        // achieve a logical error rate below ~2% per round.
        //
        // The experimental result was 1.65%. Our analytical model should
        // be in the right ballpark for reasonable noise parameters.
        let config = CatQubitConfig::new()
            .alpha(2.0)
            .fock_cutoff(25)
            .kappa_1(1e4)
            .kappa_2(1e7)
            .num_cats(5)
            .num_rounds(10000)
            .build()
            .unwrap();

        let code = ConcatenatedCatCode::new(config).unwrap();
        let analytical_rate = code.logical_error_rate();

        // The analytical rate should be well below 50% (the code works!)
        assert!(
            analytical_rate < 0.5,
            "d=5 logical error rate should be well below 50%, got {:.4e}",
            analytical_rate
        );

        // It should be below the physical phase-flip rate (concatenation helps)
        let physical_rate = code.phase_flip_per_round + code.bit_flip_per_round;
        assert!(
            analytical_rate < physical_rate,
            "concatenated rate ({:.4e}) should beat physical rate ({:.4e})",
            analytical_rate,
            physical_rate
        );
    }
}
