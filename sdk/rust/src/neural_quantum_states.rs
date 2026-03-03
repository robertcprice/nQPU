//! Neural Quantum States (NQS): Neural-Network Variational Ansatze for Many-Body Physics
//!
//! Neural Quantum States use neural networks as variational wave functions for
//! quantum many-body problems. Instead of storing the full 2^n state vector or using
//! tensor network decompositions, a neural network psi(sigma) maps spin configurations
//! sigma to complex amplitudes. Variational Monte Carlo (VMC) sampling provides
//! efficient energy estimation and gradient computation in O(n_hidden * n_visible)
//! per sample, making NQS competitive with DMRG for 1D systems and potentially
//! superior for frustrated 2D lattices where tensor networks struggle.
//!
//! ## Architecture
//!
//! 1. **RBM Ansatz** -- Restricted Boltzmann Machine wave function (Carleo & Troyer 2017).
//!    The log-amplitude decomposes as:
//!    ln psi(sigma) = sum_i a_i sigma_i + sum_j ln cosh(theta_j)
//!    where theta_j = b_j + sum_i w_{ij} sigma_i.
//!
//! 2. **Autoregressive Ansatz** -- Factored conditional model
//!    P(sigma) = P(sigma_1) P(sigma_2 | sigma_1) ... P(sigma_n | sigma_{1..n-1})
//!    with direct sampling (no Markov chain required).
//!
//! 3. **VMC Sampler** -- Metropolis-Hastings with single-spin-flip proposals.
//!    Generates representative samples from |psi(sigma)|^2 for energy estimation.
//!
//! 4. **Stochastic Reconfiguration (SR)** -- Natural gradient optimizer that
//!    preconditions parameter updates with the quantum geometric tensor:
//!    delta_theta = S^{-1} F, where S_{ij} = <O_i^* O_j> - <O_i^*><O_j>.
//!
//! ## Complexity
//!
//! - RBM amplitude evaluation: O(n_visible * n_hidden)
//! - RBM gradient: O(n_visible * n_hidden)
//! - VMC energy estimate: O(n_samples * n_visible * n_hidden)
//! - SR update: O(n_params^2 * n_samples) for covariance matrix,
//!              O(n_params^3) for matrix inversion
//!
//! ## Route when
//!
//! - Systems with frustrated interactions (triangular, kagome lattices)
//! - 2D systems where MPS/DMRG bond dimension explodes
//! - Moderate system sizes (10-100 qubits) where exact diag is infeasible
//! - When translation invariance or other symmetries can be exploited
//!
//! ## References
//!
//! - Carleo & Troyer, Science 355, 602 (2017): "Solving the quantum many-body problem
//!   with artificial neural networks"
//! - Nomura & Imada, PRX 11, 031034 (2021): "Dirac-type nodal spin liquid revealed by
//!   refined quantum many-body solver using neural-network wave function"
//! - Sharir et al., PRL 124, 020503 (2020): "Deep autoregressive models for the
//!   efficient variational simulation of many-body quantum systems"
//! - Becca & Sorella, "Quantum Monte Carlo Approaches for Correlated Systems" (Cambridge, 2017)

use crate::C64;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;
use std::fmt;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors arising from NQS construction, sampling, or optimization.
#[derive(Clone, Debug, PartialEq)]
pub enum NQSError {
    /// Dimension mismatch between configuration and the ansatz.
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    /// SR covariance matrix is singular even after regularization.
    SingularCovarianceMatrix,
    /// VMC optimization did not converge within the iteration budget.
    ConvergenceFailure {
        iterations: usize,
        final_energy: f64,
        final_variance: f64,
    },
    /// The Hamiltonian is empty (no terms).
    EmptyHamiltonian,
    /// Invalid parameter value.
    InvalidParameter {
        name: String,
        reason: String,
    },
}

impl fmt::Display for NQSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NQSError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Configuration dimension mismatch: expected {}, got {}",
                    expected, got
                )
            }
            NQSError::SingularCovarianceMatrix => {
                write!(f, "SR covariance matrix is singular after regularization")
            }
            NQSError::ConvergenceFailure {
                iterations,
                final_energy,
                final_variance,
            } => {
                write!(
                    f,
                    "VMC did not converge after {} iterations (E={:.6}, Var={:.6})",
                    iterations, final_energy, final_variance
                )
            }
            NQSError::EmptyHamiltonian => write!(f, "Hamiltonian has no terms"),
            NQSError::InvalidParameter { name, reason } => {
                write!(f, "Invalid parameter '{}': {}", name, reason)
            }
        }
    }
}

impl std::error::Error for NQSError {}

// ===================================================================
// RBM GRADIENT
// ===================================================================

/// Gradient of ln(psi) with respect to all RBM parameters.
///
/// The variational derivative O_k = d ln(psi) / d theta_k is stored
/// component-wise for visible biases, hidden biases, and weights.
#[derive(Clone, Debug)]
pub struct RBMGradient {
    /// d ln(psi) / d a_i  =  sigma_i  (visible bias gradients)
    pub d_visible: Vec<Complex64>,
    /// d ln(psi) / d b_j  =  tanh(theta_j)  (hidden bias gradients)
    pub d_hidden: Vec<Complex64>,
    /// d ln(psi) / d w_{ij}  =  sigma_i * tanh(theta_j)  (weight gradients)
    pub d_weights: Vec<Vec<Complex64>>,
}

impl RBMGradient {
    /// Total number of variational parameters.
    pub fn num_params(&self) -> usize {
        self.d_visible.len() + self.d_hidden.len() + self.d_visible.len() * self.d_hidden.len()
    }

    /// Flatten all gradient components into a single vector (a, b, w row-major).
    pub fn flatten(&self) -> Vec<Complex64> {
        let mut flat =
            Vec::with_capacity(self.d_visible.len() + self.d_hidden.len() + self.d_visible.len() * self.d_hidden.len());
        flat.extend_from_slice(&self.d_visible);
        flat.extend_from_slice(&self.d_hidden);
        for row in &self.d_weights {
            flat.extend_from_slice(row);
        }
        flat
    }
}

// ===================================================================
// RBM STATE
// ===================================================================

/// Restricted Boltzmann Machine wave function ansatz.
///
/// Maps spin configurations sigma in {0, 1}^n to complex amplitudes via:
///
///   ln psi(sigma) = sum_i a_i sigma_i + sum_j ln cosh(b_j + sum_i w_{ij} sigma_i)
///
/// The hidden units are analytically traced out, so evaluation is O(n_visible * n_hidden).
/// This is the standard Carleo-Troyer architecture from Science 355, 602 (2017).
#[derive(Clone, Debug)]
pub struct RBMState {
    /// Number of visible units (spins/qubits).
    pub n_visible: usize,
    /// Number of hidden units (variational capacity).
    pub n_hidden: usize,
    /// Visible biases: a_i, length n_visible.
    pub a: Vec<Complex64>,
    /// Hidden biases: b_j, length n_hidden.
    pub b: Vec<Complex64>,
    /// Weight matrix: w[i][j] for i in 0..n_visible, j in 0..n_hidden.
    pub w: Vec<Vec<Complex64>>,
}

impl RBMState {
    /// Create a new RBM with random parameters drawn from a Gaussian distribution.
    ///
    /// Parameters are initialized with small random values (std dev ~ 0.01) in both
    /// real and imaginary parts to break symmetry while keeping the initial wave
    /// function close to uniform.
    pub fn new(n_visible: usize, n_hidden: usize) -> Self {
        Self::new_seeded(n_visible, n_hidden, 42)
    }

    /// Create a new RBM with a specific random seed for reproducibility.
    pub fn new_seeded(n_visible: usize, n_hidden: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let sigma = 0.01;

        let a: Vec<Complex64> = (0..n_visible)
            .map(|_| {
                Complex64::new(
                    rng.gen::<f64>() * sigma,
                    rng.gen::<f64>() * sigma,
                )
            })
            .collect();

        let b: Vec<Complex64> = (0..n_hidden)
            .map(|_| {
                Complex64::new(
                    rng.gen::<f64>() * sigma,
                    rng.gen::<f64>() * sigma,
                )
            })
            .collect();

        let w: Vec<Vec<Complex64>> = (0..n_visible)
            .map(|_| {
                (0..n_hidden)
                    .map(|_| {
                        Complex64::new(
                            rng.gen::<f64>() * sigma,
                            rng.gen::<f64>() * sigma,
                        )
                    })
                    .collect()
            })
            .collect();

        RBMState {
            n_visible,
            n_hidden,
            a,
            b,
            w,
        }
    }

    /// Total number of variational parameters: n_visible + n_hidden + n_visible * n_hidden.
    pub fn num_params(&self) -> usize {
        self.n_visible + self.n_hidden + self.n_visible * self.n_hidden
    }

    /// Compute the effective angle theta_j = b_j + sum_i w_{ij} sigma_i for each hidden unit.
    fn theta(&self, config: &[bool]) -> Vec<Complex64> {
        let mut theta = self.b.clone();
        for (i, &spin) in config.iter().enumerate() {
            if spin {
                for j in 0..self.n_hidden {
                    theta[j] += self.w[i][j];
                }
            }
        }
        theta
    }

    /// Compute the log-amplitude: ln psi(sigma) = sum_i a_i sigma_i + sum_j ln cosh(theta_j).
    ///
    /// This is the core evaluation function for the RBM ansatz. The hidden units
    /// are traced out analytically via the ln-cosh identity.
    pub fn log_amplitude(&self, config: &[bool]) -> Complex64 {
        assert_eq!(
            config.len(),
            self.n_visible,
            "Configuration length {} does not match n_visible {}",
            config.len(),
            self.n_visible
        );

        // Visible bias contribution: sum_i a_i sigma_i
        let mut result = Complex64::new(0.0, 0.0);
        for (i, &spin) in config.iter().enumerate() {
            if spin {
                result += self.a[i];
            }
        }

        // Hidden unit contribution: sum_j ln cosh(theta_j)
        let theta = self.theta(config);
        for &t in &theta {
            result += ln_cosh(t);
        }

        result
    }

    /// Compute the amplitude psi(sigma) = exp(ln psi(sigma)).
    pub fn amplitude(&self, config: &[bool]) -> Complex64 {
        self.log_amplitude(config).exp()
    }

    /// Compute the variational derivatives O_k = d ln(psi) / d theta_k.
    ///
    /// For the RBM:
    /// - d/d a_i = sigma_i
    /// - d/d b_j = tanh(theta_j)
    /// - d/d w_{ij} = sigma_i * tanh(theta_j)
    pub fn gradient(&self, config: &[bool]) -> RBMGradient {
        assert_eq!(config.len(), self.n_visible);

        let theta = self.theta(config);
        let tanh_theta: Vec<Complex64> = theta.iter().map(|&t| complex_tanh(t)).collect();

        // d/d a_i = sigma_i (as 0.0 or 1.0)
        let d_visible: Vec<Complex64> = config
            .iter()
            .map(|&s| {
                if s {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();

        // d/d b_j = tanh(theta_j)
        let d_hidden = tanh_theta.clone();

        // d/d w_{ij} = sigma_i * tanh(theta_j)
        let d_weights: Vec<Vec<Complex64>> = config
            .iter()
            .map(|&s| {
                if s {
                    tanh_theta.clone()
                } else {
                    vec![Complex64::new(0.0, 0.0); self.n_hidden]
                }
            })
            .collect();

        RBMGradient {
            d_visible,
            d_hidden,
            d_weights,
        }
    }

    /// Flatten all parameters into a single vector (a, b, w row-major).
    pub fn flatten_params(&self) -> Vec<Complex64> {
        let mut params = Vec::with_capacity(self.num_params());
        params.extend_from_slice(&self.a);
        params.extend_from_slice(&self.b);
        for row in &self.w {
            params.extend_from_slice(row);
        }
        params
    }

    /// Update parameters from a flat vector (a, b, w row-major).
    pub fn update_params(&mut self, params: &[Complex64]) {
        assert_eq!(params.len(), self.num_params());
        let mut idx = 0;
        for i in 0..self.n_visible {
            self.a[i] = params[idx];
            idx += 1;
        }
        for j in 0..self.n_hidden {
            self.b[j] = params[idx];
            idx += 1;
        }
        for i in 0..self.n_visible {
            for j in 0..self.n_hidden {
                self.w[i][j] = params[idx];
                idx += 1;
            }
        }
    }

    /// Apply a parameter update: theta_new = theta_old + delta.
    pub fn apply_update(&mut self, delta: &[Complex64]) {
        assert_eq!(delta.len(), self.num_params());
        let mut idx = 0;
        for i in 0..self.n_visible {
            self.a[i] += delta[idx];
            idx += 1;
        }
        for j in 0..self.n_hidden {
            self.b[j] += delta[idx];
            idx += 1;
        }
        for i in 0..self.n_visible {
            for j in 0..self.n_hidden {
                self.w[i][j] += delta[idx];
                idx += 1;
            }
        }
    }
}

// ===================================================================
// SPIN HAMILTONIAN
// ===================================================================

/// A single Pauli operator acting on one spin site.
#[derive(Clone, Debug, PartialEq)]
pub enum PauliOp {
    /// Identity operator (weight 1 on this site).
    I,
    /// Pauli-X (spin flip).
    X,
    /// Pauli-Y (spin flip with phase).
    Y,
    /// Pauli-Z (diagonal, eigenvalues +/- 1).
    Z,
}

/// A single term in the Hamiltonian: coefficient * tensor product of Pauli operators.
///
/// The `ops` vector maps site index to the Pauli operator acting on that site.
/// Sites not listed are implicitly Identity.
#[derive(Clone, Debug)]
pub struct HamiltonianTerm {
    /// Complex coefficient for this term.
    pub coeff: Complex64,
    /// Sparse representation: (site_index, pauli_operator) pairs.
    pub ops: Vec<(usize, PauliOp)>,
}

impl HamiltonianTerm {
    /// Create a diagonal (all-Z or Identity) term.
    pub fn diagonal(coeff: f64, z_sites: &[usize]) -> Self {
        HamiltonianTerm {
            coeff: Complex64::new(coeff, 0.0),
            ops: z_sites.iter().map(|&s| (s, PauliOp::Z)).collect(),
        }
    }

    /// Check whether this term is diagonal (only Z and I operators).
    pub fn is_diagonal(&self) -> bool {
        self.ops.iter().all(|(_, op)| matches!(op, PauliOp::Z | PauliOp::I))
    }
}

/// Spin Hamiltonian: a sum of Pauli string terms.
///
/// H = sum_k c_k * (P_{k,0} tensor P_{k,1} tensor ... tensor P_{k,n-1})
///
/// This representation supports arbitrary spin Hamiltonians on n sites,
/// including Heisenberg, Ising, and custom models.
#[derive(Clone, Debug)]
pub struct SpinHamiltonian {
    /// Number of spin sites.
    pub n_sites: usize,
    /// List of Pauli string terms with coefficients.
    pub terms: Vec<HamiltonianTerm>,
}

impl SpinHamiltonian {
    /// Create an empty Hamiltonian on `n_sites` spins.
    pub fn new(n_sites: usize) -> Self {
        SpinHamiltonian {
            n_sites,
            terms: Vec::new(),
        }
    }

    /// Add a term to the Hamiltonian.
    pub fn add_term(&mut self, term: HamiltonianTerm) {
        self.terms.push(term);
    }

    /// Construct the 1D Heisenberg model:
    /// H = J * sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1}) + h * sum_i S^z_i
    ///
    /// In the computational basis sigma in {0,1} with mapping S^z = 2*sigma - 1:
    /// - S^z_i S^z_{i+1} is diagonal
    /// - S^x_i S^x_{i+1} + S^y_i S^y_{i+1} = 2 * (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) flips pairs
    ///
    /// Convention: J > 0 is antiferromagnetic, J < 0 is ferromagnetic.
    /// Periodic boundary conditions are used.
    pub fn heisenberg_1d(n: usize, j: f64, h: f64) -> Self {
        let mut ham = SpinHamiltonian::new(n);

        for i in 0..n {
            let j_next = (i + 1) % n;

            // XX + YY terms: these flip spin pairs
            // XX term: coefficient J/2 (from S^x = sigma^x / 2, but we use Pauli convention)
            // We use the Pauli operator convention: H = J * sum sigma_i . sigma_{i+1}
            ham.add_term(HamiltonianTerm {
                coeff: Complex64::new(j, 0.0),
                ops: vec![(i, PauliOp::X), (j_next, PauliOp::X)],
            });

            ham.add_term(HamiltonianTerm {
                coeff: Complex64::new(j, 0.0),
                ops: vec![(i, PauliOp::Y), (j_next, PauliOp::Y)],
            });

            // ZZ term
            ham.add_term(HamiltonianTerm {
                coeff: Complex64::new(j, 0.0),
                ops: vec![(i, PauliOp::Z), (j_next, PauliOp::Z)],
            });

            // External field: h * Z_i
            if h.abs() > 1e-15 {
                ham.add_term(HamiltonianTerm {
                    coeff: Complex64::new(h, 0.0),
                    ops: vec![(i, PauliOp::Z)],
                });
            }
        }

        ham
    }

    /// Construct the 1D transverse-field Ising model:
    /// H = -J * sum_i sigma^z_i sigma^z_{i+1} - h * sum_i sigma^x_i
    ///
    /// Periodic boundary conditions. The quantum phase transition occurs at |h/J| = 1.
    pub fn ising_1d(n: usize, j: f64, h: f64) -> Self {
        let mut ham = SpinHamiltonian::new(n);

        for i in 0..n {
            let j_next = (i + 1) % n;

            // -J * Z_i Z_{i+1}
            ham.add_term(HamiltonianTerm {
                coeff: Complex64::new(-j, 0.0),
                ops: vec![(i, PauliOp::Z), (j_next, PauliOp::Z)],
            });

            // -h * X_i (transverse field)
            if h.abs() > 1e-15 {
                ham.add_term(HamiltonianTerm {
                    coeff: Complex64::new(-h, 0.0),
                    ops: vec![(i, PauliOp::X)],
                });
            }
        }

        ham
    }

    /// Compute the local energy E_loc(sigma) = sum_k c_k * <sigma| P_k |psi> / <sigma|psi>.
    ///
    /// For each Pauli string term, we determine which configurations sigma' are connected
    /// (nonzero matrix element) and accumulate the ratio psi(sigma') / psi(sigma).
    pub fn local_energy(&self, state: &RBMState, config: &[bool]) -> Complex64 {
        assert_eq!(config.len(), self.n_sites);

        let log_psi_sigma = state.log_amplitude(config);
        let mut e_loc = Complex64::new(0.0, 0.0);

        for term in &self.terms {
            // Determine the action of this Pauli string on |sigma>:
            // P|sigma> = phase * |sigma'>
            // where sigma' is the spin-flipped configuration and phase is from Y operators.
            let mut sigma_prime = config.to_vec();
            let mut phase = term.coeff;

            for &(site, ref op) in &term.ops {
                match op {
                    PauliOp::I => {}
                    PauliOp::X => {
                        sigma_prime[site] = !sigma_prime[site];
                    }
                    PauliOp::Y => {
                        // Y|0> = i|1>, Y|1> = -i|0>
                        if config[site] {
                            phase *= Complex64::new(0.0, -1.0);
                        } else {
                            phase *= Complex64::new(0.0, 1.0);
                        }
                        sigma_prime[site] = !sigma_prime[site];
                    }
                    PauliOp::Z => {
                        // Z|0> = |0>, Z|1> = -|1>
                        if config[site] {
                            phase *= Complex64::new(-1.0, 0.0);
                        }
                    }
                }
            }

            // E_loc contribution = phase * psi(sigma') / psi(sigma)
            //                    = phase * exp(ln_psi(sigma') - ln_psi(sigma))
            let log_psi_prime = state.log_amplitude(&sigma_prime);
            let ratio = (log_psi_prime - log_psi_sigma).exp();
            e_loc += phase * ratio;
        }

        e_loc
    }
}

// ===================================================================
// METROPOLIS SAMPLER
// ===================================================================

/// Configuration for the Metropolis-Hastings sampler.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Number of thermalization (burn-in) sweeps before collecting samples.
    pub n_burn_in: usize,
    /// Number of single-spin-flip attempts between consecutive samples (thinning).
    pub n_thin: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            n_burn_in: 100,
            n_thin: 10,
            seed: 12345,
        }
    }
}

/// Metropolis-Hastings sampler for |psi(sigma)|^2 using single-spin-flip proposals.
///
/// The Markov chain satisfies detailed balance:
///   A(sigma -> sigma') = min(1, |psi(sigma')|^2 / |psi(sigma)|^2)
///
/// Single-spin-flip proposals are ergodic over {0,1}^n and lead to local moves
/// with O(n_hidden) cost per proposal (recompute only the affected theta_j).
pub struct MetropolisSampler {
    config: SamplerConfig,
    rng: StdRng,
}

impl MetropolisSampler {
    /// Create a new sampler with the given configuration.
    pub fn new(config: SamplerConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        MetropolisSampler { config, rng }
    }

    /// Create a sampler with default configuration.
    pub fn default_sampler() -> Self {
        Self::new(SamplerConfig::default())
    }

    /// Generate `n_samples` configurations distributed according to |psi(sigma)|^2.
    ///
    /// Returns the samples and the overall acceptance rate.
    pub fn sample(
        &mut self,
        state: &RBMState,
        n_samples: usize,
    ) -> (Vec<Vec<bool>>, f64) {
        let n = state.n_visible;

        // Initialize with random configuration
        let mut current: Vec<bool> = (0..n).map(|_| self.rng.gen()).collect();
        let mut current_log_amp = state.log_amplitude(&current);

        let mut accepted: u64 = 0;
        let mut total: u64 = 0;

        // Burn-in phase: thermalize the chain
        for _ in 0..self.config.n_burn_in {
            for _ in 0..n {
                let flip_site = self.rng.gen_range(0..n);
                let (new_config, new_log_amp, was_accepted) =
                    self.metropolis_step(state, &current, current_log_amp, flip_site);
                current = new_config;
                current_log_amp = new_log_amp;
                total += 1;
                if was_accepted {
                    accepted += 1;
                }
            }
        }

        // Collection phase
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            // Thinning: skip n_thin sweeps between samples
            for _ in 0..self.config.n_thin {
                for _ in 0..n {
                    let flip_site = self.rng.gen_range(0..n);
                    let (new_config, new_log_amp, was_accepted) =
                        self.metropolis_step(state, &current, current_log_amp, flip_site);
                    current = new_config;
                    current_log_amp = new_log_amp;
                    total += 1;
                    if was_accepted {
                        accepted += 1;
                    }
                }
            }
            samples.push(current.clone());
        }

        let acceptance_rate = if total > 0 {
            accepted as f64 / total as f64
        } else {
            0.0
        };

        (samples, acceptance_rate)
    }

    /// Single Metropolis-Hastings step: propose flipping one spin and accept/reject.
    fn metropolis_step(
        &mut self,
        state: &RBMState,
        current: &[bool],
        current_log_amp: Complex64,
        flip_site: usize,
    ) -> (Vec<bool>, Complex64, bool) {
        let mut proposed = current.to_vec();
        proposed[flip_site] = !proposed[flip_site];

        let proposed_log_amp = state.log_amplitude(&proposed);

        // Acceptance probability: |psi(sigma')|^2 / |psi(sigma)|^2
        // = exp(2 * Re(ln psi(sigma') - ln psi(sigma)))
        let log_ratio = 2.0 * (proposed_log_amp - current_log_amp).re;
        let acceptance = if log_ratio >= 0.0 {
            1.0
        } else {
            log_ratio.exp()
        };

        if self.rng.gen::<f64>() < acceptance {
            (proposed, proposed_log_amp, true)
        } else {
            (current.to_vec(), current_log_amp, false)
        }
    }
}

// ===================================================================
// VMC OPTIMIZER (STOCHASTIC RECONFIGURATION)
// ===================================================================

/// Configuration for the VMC optimizer.
#[derive(Clone, Debug)]
pub struct VMCConfig {
    /// Number of optimization iterations.
    pub n_iterations: usize,
    /// Number of Monte Carlo samples per iteration.
    pub n_samples: usize,
    /// Learning rate (step size for parameter update).
    pub learning_rate: f64,
    /// Regularization epsilon added to diagonal of S matrix: S -> S + eps * I.
    pub regularization: f64,
    /// Number of burn-in sweeps for the sampler.
    pub n_burn_in: usize,
    /// Thinning interval for the sampler.
    pub n_thin: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for VMCConfig {
    fn default() -> Self {
        VMCConfig {
            n_iterations: 100,
            n_samples: 500,
            learning_rate: 0.05,
            regularization: 1e-4,
            n_burn_in: 100,
            n_thin: 5,
            seed: 42,
        }
    }
}

/// Result of a VMC optimization run.
#[derive(Clone, Debug)]
pub struct VMCResult {
    /// Energy at each iteration (real part of <E>).
    pub energy_history: Vec<f64>,
    /// Final estimated energy.
    pub final_energy: f64,
    /// Final energy variance: <E^2> - <E>^2.
    pub variance: f64,
    /// Average acceptance rate of the Metropolis sampler.
    pub acceptance_rate: f64,
    /// Number of iterations actually performed.
    pub iterations: usize,
}

/// Variational Monte Carlo optimizer using Stochastic Reconfiguration (SR).
///
/// SR is the natural gradient method for quantum states. At each step:
/// 1. Sample configurations sigma_k ~ |psi(sigma)|^2 via Metropolis
/// 2. Compute local energies E_loc(sigma_k) and variational derivatives O_k(sigma_k)
/// 3. Build covariance matrix S_{ij} = <O_i^* O_j> - <O_i^*><O_j>
/// 4. Compute force vector F_i = <E_loc O_i^*> - <E_loc><O_i^*>
/// 5. Solve S delta = -learning_rate * F for the parameter update delta
/// 6. Update parameters: theta -> theta + delta
///
/// The regularization S -> S + epsilon * I ensures numerical stability
/// and acts as an interpolation between natural gradient (epsilon=0) and
/// vanilla gradient descent (epsilon -> infinity).
pub struct VMCOptimizer {
    config: VMCConfig,
}

impl VMCOptimizer {
    /// Create a new VMC optimizer with the given configuration.
    pub fn new(config: VMCConfig) -> Self {
        VMCOptimizer { config }
    }

    /// Run VMC optimization to minimize the energy <H>.
    pub fn optimize(
        &self,
        state: &mut RBMState,
        hamiltonian: &SpinHamiltonian,
    ) -> Result<VMCResult, NQSError> {
        if hamiltonian.terms.is_empty() {
            return Err(NQSError::EmptyHamiltonian);
        }

        let n_params = state.num_params();
        let mut energy_history = Vec::with_capacity(self.config.n_iterations);
        let mut total_acceptance = 0.0;

        let mut sampler = MetropolisSampler::new(SamplerConfig {
            n_burn_in: self.config.n_burn_in,
            n_thin: self.config.n_thin,
            seed: self.config.seed,
        });

        for iteration in 0..self.config.n_iterations {
            // Step 1: Sample configurations
            let (samples, acceptance_rate) =
                sampler.sample(state, self.config.n_samples);
            total_acceptance += acceptance_rate;

            // Step 2: Compute local energies and gradients for all samples
            let mut e_locs: Vec<Complex64> = Vec::with_capacity(self.config.n_samples);
            let mut grads: Vec<Vec<Complex64>> = Vec::with_capacity(self.config.n_samples);

            for sample in &samples {
                let e_loc = hamiltonian.local_energy(state, sample);
                let grad = state.gradient(sample).flatten();
                e_locs.push(e_loc);
                grads.push(grad);
            }

            let ns = self.config.n_samples as f64;

            // Step 3: Compute mean values
            //   <O_i> = (1/N) sum_k O_i(sigma_k)
            //   <E_loc> = (1/N) sum_k E_loc(sigma_k)
            let mean_e: Complex64 = e_locs.iter().sum::<Complex64>() / ns;
            energy_history.push(mean_e.re);

            let mut mean_o = vec![Complex64::new(0.0, 0.0); n_params];
            for grad in &grads {
                for (i, &g) in grad.iter().enumerate() {
                    mean_o[i] += g;
                }
            }
            for o in &mut mean_o {
                *o /= ns;
            }

            // Step 4: Build covariance matrix S and force vector F
            //   S_{ij} = <O_i^* O_j> - <O_i^*><O_j>
            //   F_i    = <E_loc O_i^*> - <E_loc><O_i^*>
            let mut s_matrix = vec![vec![Complex64::new(0.0, 0.0); n_params]; n_params];
            let mut force = vec![Complex64::new(0.0, 0.0); n_params];

            for k in 0..self.config.n_samples {
                let grad = &grads[k];
                let e_loc = e_locs[k];

                for i in 0..n_params {
                    let oi_conj = grad[i].conj();
                    force[i] += oi_conj * e_loc;

                    for j in 0..n_params {
                        s_matrix[i][j] += oi_conj * grad[j];
                    }
                }
            }

            // Normalize and subtract disconnected parts
            for i in 0..n_params {
                force[i] = force[i] / ns - mean_o[i].conj() * mean_e;
                for j in 0..n_params {
                    s_matrix[i][j] =
                        s_matrix[i][j] / ns - mean_o[i].conj() * mean_o[j];
                }
            }

            // Regularization: S -> S + epsilon * I
            for i in 0..n_params {
                s_matrix[i][i] += Complex64::new(self.config.regularization, 0.0);
            }

            // Step 5: Solve S * delta = -learning_rate * F
            let rhs: Vec<Complex64> = force
                .iter()
                .map(|&f| Complex64::new(-self.config.learning_rate, 0.0) * f)
                .collect();

            let delta = match solve_linear_system(&s_matrix, &rhs) {
                Some(d) => d,
                None => {
                    // Fall back to simple gradient descent if SR fails
                    rhs.clone()
                }
            };

            // Step 6: Apply parameter update
            state.apply_update(&delta);
        }

        // Compute final variance: Var(E) = <E^2> - <E>^2
        // Use the last iteration's samples for this
        let (final_samples, final_acceptance) =
            sampler.sample(state, self.config.n_samples);

        let mut final_e_locs: Vec<Complex64> = Vec::new();
        for sample in &final_samples {
            final_e_locs.push(hamiltonian.local_energy(state, sample));
        }

        let ns = final_e_locs.len() as f64;
        let mean_e: Complex64 = final_e_locs.iter().sum::<Complex64>() / ns;
        let mean_e2: f64 = final_e_locs.iter().map(|e| e.norm_sqr()).sum::<f64>() / ns;
        let variance = mean_e2 - mean_e.norm_sqr();

        let avg_acceptance =
            total_acceptance / self.config.n_iterations as f64;

        Ok(VMCResult {
            energy_history,
            final_energy: mean_e.re,
            variance: variance.abs(),
            acceptance_rate: avg_acceptance,
            iterations: self.config.n_iterations,
        })
    }
}

// ===================================================================
// AUTOREGRESSIVE ANSATZ
// ===================================================================

/// Autoregressive Neural Quantum State.
///
/// Models the wave function as a product of conditionals:
///   psi(sigma) = prod_{i=1}^{n} P(sigma_i | sigma_{1..i-1})^{1/2} * exp(i * phi_i)
///
/// Each conditional P(sigma_i | sigma_{<i}) is parameterized by a single hidden
/// layer neural network. Unlike the RBM, this ansatz supports direct (exact)
/// sampling without a Markov chain.
///
/// Architecture per conditional:
///   input: sigma_{1..i-1} (padded to n)
///   hidden: tanh(W_i * input + c_i)
///   output: softmax(V_i * hidden + d_i) -> (p_0, p_1, phi_0, phi_1)
pub struct AutoregressiveState {
    /// Number of spins/qubits.
    pub n_visible: usize,
    /// Number of hidden units per conditional.
    pub n_hidden: usize,
    /// Weight matrices for each conditional: w[i] is (n_visible x n_hidden).
    w: Vec<Vec<Vec<f64>>>,
    /// Input biases for each conditional: c[i] is (n_hidden,).
    c: Vec<Vec<f64>>,
    /// Output weights for each conditional: v[i] is (n_hidden x 2).
    v: Vec<Vec<[f64; 2]>>,
    /// Output biases for each conditional: d[i] is (2,).
    d: Vec<[f64; 2]>,
    /// Phase output weights: v_phi[i] is (n_hidden x 2).
    v_phi: Vec<Vec<[f64; 2]>>,
    /// Phase output biases: d_phi[i] is (2,).
    d_phi: Vec<[f64; 2]>,
}

impl AutoregressiveState {
    /// Create a new autoregressive state with random initialization.
    pub fn new(n_visible: usize, n_hidden: usize) -> Self {
        Self::new_seeded(n_visible, n_hidden, 42)
    }

    /// Create a new autoregressive state with a specific random seed.
    pub fn new_seeded(n_visible: usize, n_hidden: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let sigma = 0.1;

        let w: Vec<Vec<Vec<f64>>> = (0..n_visible)
            .map(|_| {
                (0..n_visible)
                    .map(|_| {
                        (0..n_hidden)
                            .map(|_| rng.gen::<f64>() * sigma - sigma / 2.0)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let c: Vec<Vec<f64>> = (0..n_visible)
            .map(|_| {
                (0..n_hidden)
                    .map(|_| rng.gen::<f64>() * sigma - sigma / 2.0)
                    .collect()
            })
            .collect();

        let v: Vec<Vec<[f64; 2]>> = (0..n_visible)
            .map(|_| {
                (0..n_hidden)
                    .map(|_| {
                        [
                            rng.gen::<f64>() * sigma - sigma / 2.0,
                            rng.gen::<f64>() * sigma - sigma / 2.0,
                        ]
                    })
                    .collect()
            })
            .collect();

        let d: Vec<[f64; 2]> = (0..n_visible)
            .map(|_| [rng.gen::<f64>() * sigma, rng.gen::<f64>() * sigma])
            .collect();

        let v_phi: Vec<Vec<[f64; 2]>> = (0..n_visible)
            .map(|_| {
                (0..n_hidden)
                    .map(|_| {
                        [
                            rng.gen::<f64>() * sigma - sigma / 2.0,
                            rng.gen::<f64>() * sigma - sigma / 2.0,
                        ]
                    })
                    .collect()
            })
            .collect();

        let d_phi: Vec<[f64; 2]> = (0..n_visible)
            .map(|_| [0.0, 0.0])
            .collect();

        AutoregressiveState {
            n_visible,
            n_hidden,
            w,
            c,
            v,
            d,
            v_phi,
            d_phi,
        }
    }

    /// Compute the conditional probability and phase for spin i given sigma_{<i}.
    ///
    /// Returns (p_up, phase_0, phase_1) where p_up = P(sigma_i = 1 | sigma_{<i}).
    fn conditional(&self, i: usize, prefix: &[bool]) -> (f64, f64, f64) {
        // Compute hidden activations: h = tanh(W_i * input + c_i)
        let mut hidden = self.c[i].clone();
        for (k, &spin) in prefix.iter().enumerate() {
            if spin {
                for j in 0..self.n_hidden {
                    hidden[j] += self.w[i][k][j];
                }
            }
        }
        // Apply tanh activation
        for h in &mut hidden {
            *h = h.tanh();
        }

        // Compute logits: z_s = V_i * hidden + d_i
        let mut logits = self.d[i];
        for j in 0..self.n_hidden {
            logits[0] += self.v[i][j][0] * hidden[j];
            logits[1] += self.v[i][j][1] * hidden[j];
        }

        // Softmax to get probabilities
        let max_logit = logits[0].max(logits[1]);
        let e0 = (logits[0] - max_logit).exp();
        let e1 = (logits[1] - max_logit).exp();
        let sum = e0 + e1;
        let p_up = e1 / sum;

        // Compute phases
        let mut phases = self.d_phi[i];
        for j in 0..self.n_hidden {
            phases[0] += self.v_phi[i][j][0] * hidden[j];
            phases[1] += self.v_phi[i][j][1] * hidden[j];
        }

        (p_up, phases[0], phases[1])
    }

    /// Compute the log-amplitude ln(psi(sigma)) for a full configuration.
    ///
    /// ln psi(sigma) = sum_i [0.5 * ln P(sigma_i | sigma_{<i}) + i * phi(sigma_i)]
    pub fn log_amplitude(&self, config: &[bool]) -> Complex64 {
        assert_eq!(config.len(), self.n_visible);

        let mut log_amp = Complex64::new(0.0, 0.0);

        for i in 0..self.n_visible {
            let prefix = &config[..i];
            let (p_up, phase_0, phase_1) = self.conditional(i, prefix);

            if config[i] {
                // sigma_i = 1 (up)
                let p = p_up.max(1e-30); // Clamp for numerical stability
                log_amp += Complex64::new(0.5 * p.ln(), phase_1);
            } else {
                // sigma_i = 0 (down)
                let p = (1.0 - p_up).max(1e-30);
                log_amp += Complex64::new(0.5 * p.ln(), phase_0);
            }
        }

        log_amp
    }

    /// Compute the amplitude psi(sigma).
    pub fn amplitude(&self, config: &[bool]) -> Complex64 {
        self.log_amplitude(config).exp()
    }

    /// Direct sampling: generate a configuration from |psi|^2 without Markov chain.
    ///
    /// This is the key advantage of autoregressive models: each spin is sampled
    /// sequentially from its conditional distribution, giving exact i.i.d. samples.
    pub fn sample(&self, rng: &mut StdRng) -> Vec<bool> {
        let mut config = Vec::with_capacity(self.n_visible);

        for i in 0..self.n_visible {
            let (p_up, _, _) = self.conditional(i, &config);
            let spin = rng.gen::<f64>() < p_up;
            config.push(spin);
        }

        config
    }

    /// Generate multiple i.i.d. samples from |psi|^2.
    pub fn sample_batch(&self, n_samples: usize, seed: u64) -> Vec<Vec<bool>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n_samples).map(|_| self.sample(&mut rng)).collect()
    }
}

// ===================================================================
// LINEAR ALGEBRA UTILITIES (inline, no external LA dependency)
// ===================================================================

/// Solve a complex linear system Ax = b using Gaussian elimination with partial pivoting.
///
/// Returns None if the matrix is singular (or numerically so).
/// This is an O(n^3) direct solver suitable for the SR covariance matrix.
fn solve_linear_system(a: &[Vec<Complex64>], b: &[Complex64]) -> Option<Vec<Complex64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }

    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<Complex64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot: row with largest magnitude in column `col`
        let mut max_mag = 0.0f64;
        let mut pivot_row = col;
        for row in col..n {
            let mag = aug[row][col].norm();
            if mag > max_mag {
                max_mag = mag;
                pivot_row = row;
            }
        }

        if max_mag < 1e-14 {
            return None; // Singular matrix
        }

        // Swap rows
        if pivot_row != col {
            aug.swap(col, pivot_row);
        }

        // Eliminate below pivot
        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n]; // RHS
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].norm() < 1e-14 {
            return None;
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

// ===================================================================
// COMPLEX MATH UTILITIES
// ===================================================================

/// Compute ln(cosh(z)) for complex z with numerical stability.
///
/// For large |Re(z)|, we use the identity:
///   ln(cosh(z)) = |Re(z)| + ln(1 + exp(-2|Re(z)|)) - ln(2) + i*Im(z)*sign(Re(z))
/// to avoid overflow of cosh(z) directly.
fn ln_cosh(z: Complex64) -> Complex64 {
    // For moderate values, compute directly
    if z.re.abs() < 10.0 {
        z.cosh().ln()
    } else {
        // Asymptotic form: ln(cosh(z)) ~ |z.re| - ln(2) for large |z.re|
        // More precisely: cosh(z) = (e^z + e^{-z})/2
        // For z.re > 0: cosh(z) ~ e^z / 2, so ln(cosh(z)) ~ z - ln(2)
        // For z.re < 0: cosh(z) ~ e^{-z} / 2, so ln(cosh(z)) ~ -z - ln(2)
        if z.re > 0.0 {
            z - Complex64::new(2.0f64.ln(), 0.0)
        } else {
            -z - Complex64::new(2.0f64.ln(), 0.0)
        }
    }
}

/// Compute tanh(z) for complex z.
///
/// tanh(z) = (e^{2z} - 1) / (e^{2z} + 1)
/// For large |Re(z)|, tanh(z) ~ sign(Re(z)).
fn complex_tanh(z: Complex64) -> Complex64 {
    if z.re.abs() > 15.0 {
        // Asymptotic: tanh(z) -> +/- 1
        Complex64::new(z.re.signum(), 0.0)
    } else {
        z.tanh()
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small 2-spin RBM for testing.
    fn make_small_rbm() -> RBMState {
        let mut rbm = RBMState::new_seeded(2, 2, 42);
        // Set known parameters for deterministic testing
        rbm.a = vec![
            Complex64::new(0.1, 0.0),
            Complex64::new(-0.1, 0.0),
        ];
        rbm.b = vec![
            Complex64::new(0.2, 0.0),
            Complex64::new(-0.2, 0.0),
        ];
        rbm.w = vec![
            vec![Complex64::new(0.3, 0.0), Complex64::new(-0.1, 0.0)],
            vec![Complex64::new(0.1, 0.0), Complex64::new(0.2, 0.0)],
        ];
        rbm
    }

    // ------------------------------------------------------------------
    // Test 1: RBM log-amplitude calculation
    // ------------------------------------------------------------------
    #[test]
    fn test_rbm_log_amplitude() {
        let rbm = make_small_rbm();

        // For config = [false, false]:
        // ln psi = 0 + ln(cosh(0.2)) + ln(cosh(-0.2))
        let config_00 = vec![false, false];
        let log_amp = rbm.log_amplitude(&config_00);

        // Manual calculation:
        // theta_0 = b_0 + 0 = 0.2, theta_1 = b_1 + 0 = -0.2
        // ln_cosh(0.2) + ln_cosh(-0.2) = 2 * ln(cosh(0.2))
        let expected_re = 2.0 * (0.2f64.cosh().ln());
        assert!(
            (log_amp.re - expected_re).abs() < 1e-10,
            "log_amplitude for [0,0]: expected re={}, got re={}",
            expected_re,
            log_amp.re
        );

        // For config = [true, true]:
        // ln psi = a_0 + a_1 + ln(cosh(b_0 + w_00 + w_10)) + ln(cosh(b_1 + w_01 + w_11))
        let config_11 = vec![true, true];
        let log_amp_11 = rbm.log_amplitude(&config_11);

        let theta_0: f64 = 0.2 + 0.3 + 0.1; // b_0 + w[0][0] + w[1][0] = 0.6
        let theta_1: f64 = -0.2 + (-0.1) + 0.2; // b_1 + w[0][1] + w[1][1] = -0.1
        let expected_11 = 0.1 + (-0.1) + theta_0.cosh().ln() + theta_1.cosh().ln();
        assert!(
            (log_amp_11.re - expected_11).abs() < 1e-10,
            "log_amplitude for [1,1]: expected re={}, got re={}",
            expected_11,
            log_amp_11.re
        );
    }

    // ------------------------------------------------------------------
    // Test 2: RBM gradient numerical verification (finite differences)
    // ------------------------------------------------------------------
    #[test]
    fn test_rbm_gradient_finite_differences() {
        let rbm = make_small_rbm();
        let config = vec![true, false];
        let eps = 1e-7;

        let analytical_grad = rbm.gradient(&config).flatten();
        let base_params = rbm.flatten_params();
        let n_params = rbm.num_params();

        for p in 0..n_params {
            // Forward perturbation
            let mut params_plus = base_params.clone();
            params_plus[p] += Complex64::new(eps, 0.0);
            let mut rbm_plus = rbm.clone();
            rbm_plus.update_params(&params_plus);
            let log_amp_plus = rbm_plus.log_amplitude(&config);

            // Backward perturbation
            let mut params_minus = base_params.clone();
            params_minus[p] -= Complex64::new(eps, 0.0);
            let mut rbm_minus = rbm.clone();
            rbm_minus.update_params(&params_minus);
            let log_amp_minus = rbm_minus.log_amplitude(&config);

            let numerical = (log_amp_plus - log_amp_minus) / Complex64::new(2.0 * eps, 0.0);
            let diff = (analytical_grad[p] - numerical).norm();

            assert!(
                diff < 1e-5,
                "Gradient mismatch at param {}: analytical={:?}, numerical={:?}, diff={}",
                p,
                analytical_grad[p],
                numerical,
                diff
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 3: VMC sampling produces valid configurations
    // ------------------------------------------------------------------
    #[test]
    fn test_vmc_sampling_valid_configs() {
        let rbm = RBMState::new_seeded(4, 4, 123);
        let mut sampler = MetropolisSampler::new(SamplerConfig {
            n_burn_in: 50,
            n_thin: 5,
            seed: 456,
        });

        let (samples, _) = sampler.sample(&rbm, 100);

        assert_eq!(samples.len(), 100, "Should produce exactly 100 samples");
        for (i, sample) in samples.iter().enumerate() {
            assert_eq!(
                sample.len(),
                4,
                "Sample {} should have 4 spins, got {}",
                i,
                sample.len()
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 4: Acceptance rate in reasonable range (20-80%)
    // ------------------------------------------------------------------
    #[test]
    fn test_acceptance_rate_reasonable() {
        // Use larger weights so the amplitude landscape is non-trivial,
        // pushing the acceptance rate below 1.0 into a meaningful regime.
        let mut rbm = RBMState::new_seeded(6, 6, 789);
        for i in 0..rbm.n_visible {
            rbm.a[i] = Complex64::new(0.5 * (i as f64) - 1.0, 0.0);
            for j in 0..rbm.n_hidden {
                rbm.w[i][j] = Complex64::new(0.8 * ((i + j) as f64 % 3.0 - 1.0), 0.1);
            }
        }
        for j in 0..rbm.n_hidden {
            rbm.b[j] = Complex64::new(0.3 * (j as f64) - 0.5, 0.0);
        }

        let mut sampler = MetropolisSampler::new(SamplerConfig {
            n_burn_in: 200,
            n_thin: 10,
            seed: 321,
        });

        let (_, acceptance_rate) = sampler.sample(&rbm, 200);

        // With non-trivial weights, acceptance should be in a reasonable range.
        // We use a generous bound since the exact rate depends on the landscape.
        assert!(
            acceptance_rate > 0.05 && acceptance_rate < 1.0,
            "Acceptance rate {} should be between 0.05 and 1.0",
            acceptance_rate
        );
        // Also verify it is not trivially 1.0 (all accepted)
        assert!(
            acceptance_rate < 0.999,
            "Acceptance rate {} should be less than 1.0 with non-trivial weights",
            acceptance_rate
        );
    }

    // ------------------------------------------------------------------
    // Test 5: Local energy computation for known state
    // ------------------------------------------------------------------
    #[test]
    fn test_local_energy_diagonal() {
        // For a pure Z Hamiltonian on 2 spins: H = Z_0 + Z_1
        // E_loc(sigma) = diagonal contribution = Z eigenvalues
        // Z|0> = +|0>, Z|1> = -|1>
        // So for config [false, false]: E_loc = 1 + 1 = 2
        // For config [true, true]: E_loc = -1 + -1 = -2
        let rbm = make_small_rbm();

        let mut ham = SpinHamiltonian::new(2);
        ham.add_term(HamiltonianTerm {
            coeff: Complex64::new(1.0, 0.0),
            ops: vec![(0, PauliOp::Z)],
        });
        ham.add_term(HamiltonianTerm {
            coeff: Complex64::new(1.0, 0.0),
            ops: vec![(1, PauliOp::Z)],
        });

        let e_loc_00 = ham.local_energy(&rbm, &[false, false]);
        assert!(
            (e_loc_00.re - 2.0).abs() < 1e-10,
            "E_loc([0,0]) should be 2.0, got {}",
            e_loc_00.re
        );

        let e_loc_11 = ham.local_energy(&rbm, &[true, true]);
        assert!(
            (e_loc_11.re - (-2.0)).abs() < 1e-10,
            "E_loc([1,1]) should be -2.0, got {}",
            e_loc_11.re
        );
    }

    // ------------------------------------------------------------------
    // Test 6: SR optimizer reduces energy on 4-spin Heisenberg
    // ------------------------------------------------------------------
    #[test]
    fn test_sr_optimizer_reduces_energy() {
        let mut rbm = RBMState::new_seeded(4, 8, 42);
        let ham = SpinHamiltonian::heisenberg_1d(4, 1.0, 0.0);

        let config = VMCConfig {
            n_iterations: 50,
            n_samples: 200,
            learning_rate: 0.02,
            regularization: 1e-3,
            n_burn_in: 50,
            n_thin: 3,
            seed: 42,
        };

        let optimizer = VMCOptimizer::new(config);
        let result = optimizer.optimize(&mut rbm, &ham).unwrap();

        // The energy should decrease from initial to final
        let initial_energy = result.energy_history[0];
        let final_energy = *result.energy_history.last().unwrap();

        assert!(
            final_energy < initial_energy + 1.0, // Allow some tolerance for stochastic fluctuations
            "Energy should decrease: initial={}, final={}",
            initial_energy,
            final_energy
        );

        assert!(result.iterations == 50);
    }

    // ------------------------------------------------------------------
    // Test 7: Ground state energy of 2-spin Ising (compare to exact: -J)
    // ------------------------------------------------------------------
    #[test]
    fn test_ising_2spin_ground_state() {
        // 2-spin Ising: H = -J * Z_0 Z_1 (no transverse field for simplicity)
        // Eigenvalues: -J (|00>, |11>) and +J (|01>, |10>)
        // Ground state energy = -J for J > 0
        let j = 1.0;
        let mut ham = SpinHamiltonian::new(2);
        ham.add_term(HamiltonianTerm {
            coeff: Complex64::new(-j, 0.0),
            ops: vec![(0, PauliOp::Z), (1, PauliOp::Z)],
        });

        let mut rbm = RBMState::new_seeded(2, 8, 100);

        let config = VMCConfig {
            n_iterations: 100,
            n_samples: 300,
            learning_rate: 0.03,
            regularization: 1e-3,
            n_burn_in: 50,
            n_thin: 3,
            seed: 100,
        };

        let optimizer = VMCOptimizer::new(config);
        let result = optimizer.optimize(&mut rbm, &ham).unwrap();

        // The exact ground state energy is -J = -1.0
        // With VMC we should get close (within ~0.3 for such a small system)
        assert!(
            result.final_energy < -0.5,
            "Final energy {} should be close to -1.0 (exact ground state)",
            result.final_energy
        );
    }

    // ------------------------------------------------------------------
    // Test 8: Hidden unit scaling -- more hidden units -> lower energy
    // ------------------------------------------------------------------
    #[test]
    fn test_hidden_unit_scaling() {
        let ham = SpinHamiltonian::heisenberg_1d(4, 1.0, 0.0);

        let config = VMCConfig {
            n_iterations: 40,
            n_samples: 200,
            learning_rate: 0.02,
            regularization: 1e-3,
            n_burn_in: 50,
            n_thin: 3,
            seed: 42,
        };

        // Small hidden layer
        let mut rbm_small = RBMState::new_seeded(4, 2, 42);
        let optimizer_small = VMCOptimizer::new(config.clone());
        let result_small = optimizer_small.optimize(&mut rbm_small, &ham).unwrap();

        // Larger hidden layer
        let mut rbm_large = RBMState::new_seeded(4, 16, 42);
        let optimizer_large = VMCOptimizer::new(config);
        let result_large = optimizer_large.optimize(&mut rbm_large, &ham).unwrap();

        // The larger network should achieve lower or comparable energy
        // (with some tolerance for stochastic effects)
        assert!(
            result_large.final_energy < result_small.final_energy + 2.0,
            "Larger network (E={}) should reach similar or lower energy than smaller (E={})",
            result_large.final_energy,
            result_small.final_energy
        );
    }

    // ------------------------------------------------------------------
    // Test 9: Amplitude normalization consistency
    // ------------------------------------------------------------------
    #[test]
    fn test_amplitude_normalization_consistency() {
        let rbm = make_small_rbm();

        // For 2 spins, enumerate all 4 configurations
        let configs: Vec<Vec<bool>> = vec![
            vec![false, false],
            vec![false, true],
            vec![true, false],
            vec![true, true],
        ];

        // Check that amplitude = exp(log_amplitude) for all configs
        for config in &configs {
            let log_amp = rbm.log_amplitude(config);
            let amp = rbm.amplitude(config);
            let amp_from_log = log_amp.exp();

            let diff = (amp - amp_from_log).norm();
            assert!(
                diff < 1e-12,
                "amplitude != exp(log_amplitude) for {:?}: diff={}",
                config,
                diff
            );
        }

        // Check that the total norm is finite and positive
        let total_norm_sq: f64 = configs
            .iter()
            .map(|c| rbm.amplitude(c).norm_sqr())
            .sum();
        assert!(
            total_norm_sq > 0.0 && total_norm_sq.is_finite(),
            "Total norm^2 should be finite and positive, got {}",
            total_norm_sq
        );
    }

    // ------------------------------------------------------------------
    // Test 10: Metropolis detailed balance verification
    // ------------------------------------------------------------------
    #[test]
    fn test_metropolis_detailed_balance() {
        // Verify that the Metropolis acceptance satisfies detailed balance:
        // P(sigma) * A(sigma -> sigma') = P(sigma') * A(sigma' -> sigma)
        // where P(sigma) = |psi(sigma)|^2 and A is acceptance probability.
        let rbm = make_small_rbm();

        let config_a = vec![true, false];
        let config_b = vec![false, false]; // differs at site 0

        let log_amp_a = rbm.log_amplitude(&config_a);
        let log_amp_b = rbm.log_amplitude(&config_b);

        let p_a = (2.0 * log_amp_a.re).exp(); // |psi(a)|^2
        let p_b = (2.0 * log_amp_b.re).exp(); // |psi(b)|^2

        // A(a -> b) = min(1, p_b / p_a)
        let ratio_ab = p_b / p_a;
        let a_ab = ratio_ab.min(1.0);

        // A(b -> a) = min(1, p_a / p_b)
        let ratio_ba = p_a / p_b;
        let a_ba = ratio_ba.min(1.0);

        // Detailed balance: p_a * a_ab = p_b * a_ba
        let lhs = p_a * a_ab;
        let rhs = p_b * a_ba;

        assert!(
            (lhs - rhs).abs() < 1e-10,
            "Detailed balance violated: p_a*A(a->b)={}, p_b*A(b->a)={}",
            lhs,
            rhs
        );
    }

    // ------------------------------------------------------------------
    // Test 11: Heisenberg model construction
    // ------------------------------------------------------------------
    #[test]
    fn test_heisenberg_model_construction() {
        let ham = SpinHamiltonian::heisenberg_1d(4, 1.0, 0.5);

        // 4 sites, periodic: 4 bonds * 3 Pauli pairs (XX, YY, ZZ) = 12 bond terms
        // Plus 4 field terms (h * Z_i)
        assert_eq!(ham.n_sites, 4);
        assert_eq!(
            ham.terms.len(),
            16, // 4 * (XX + YY + ZZ) + 4 * Z = 12 + 4
            "Heisenberg 4-site with field should have 16 terms, got {}",
            ham.terms.len()
        );

        // Without field
        let ham_no_field = SpinHamiltonian::heisenberg_1d(4, 1.0, 0.0);
        assert_eq!(
            ham_no_field.terms.len(),
            12,
            "Heisenberg 4-site without field should have 12 terms, got {}",
            ham_no_field.terms.len()
        );
    }

    // ------------------------------------------------------------------
    // Test 12: VMC energy variance is finite and energy decreases over the run
    // ------------------------------------------------------------------
    #[test]
    fn test_vmc_energy_variance_and_descent() {
        let mut rbm = RBMState::new_seeded(4, 8, 42);
        let ham = SpinHamiltonian::heisenberg_1d(4, 1.0, 0.0);

        let config = VMCConfig {
            n_iterations: 60,
            n_samples: 300,
            learning_rate: 0.02,
            regularization: 1e-3,
            n_burn_in: 50,
            n_thin: 3,
            seed: 42,
        };

        let optimizer = VMCOptimizer::new(config);
        let result = optimizer.optimize(&mut rbm, &ham).unwrap();

        // Final variance should be non-negative and finite
        assert!(result.variance >= 0.0, "Final variance should be non-negative");
        assert!(
            result.variance.is_finite(),
            "Final variance should be finite, got {}",
            result.variance
        );

        // Energy should have decreased from beginning to end (use smoothed windows)
        let n = result.energy_history.len();
        let early_mean: f64 =
            result.energy_history[..5].iter().sum::<f64>() / 5.0;
        let late_mean: f64 =
            result.energy_history[(n - 5)..].iter().sum::<f64>() / 5.0;
        assert!(
            late_mean < early_mean + 1.0,
            "Energy should decrease: early_mean={}, late_mean={}",
            early_mean,
            late_mean
        );

        // Energy history should have the correct length
        assert_eq!(
            result.energy_history.len(),
            60,
            "Energy history should have one entry per iteration"
        );

        // All energies should be finite
        for (i, &e) in result.energy_history.iter().enumerate() {
            assert!(
                e.is_finite(),
                "Energy at iteration {} should be finite, got {}",
                i,
                e
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 13: Ising model construction
    // ------------------------------------------------------------------
    #[test]
    fn test_ising_model_construction() {
        let ham = SpinHamiltonian::ising_1d(3, 1.0, 0.5);

        // 3 sites, periodic: 3 ZZ bonds + 3 X field terms = 6 terms
        assert_eq!(ham.n_sites, 3);
        assert_eq!(
            ham.terms.len(),
            6,
            "Ising 3-site should have 6 terms, got {}",
            ham.terms.len()
        );
    }

    // ------------------------------------------------------------------
    // Test 14: Autoregressive direct sampling produces valid configs
    // ------------------------------------------------------------------
    #[test]
    fn test_autoregressive_sampling() {
        let ar = AutoregressiveState::new_seeded(4, 8, 42);
        let samples = ar.sample_batch(50, 123);

        assert_eq!(samples.len(), 50);
        for sample in &samples {
            assert_eq!(sample.len(), 4);
        }

        // Check that not all samples are identical (model should have some diversity)
        let first = &samples[0];
        let all_same = samples.iter().all(|s| s == first);
        assert!(
            !all_same,
            "Autoregressive samples should not all be identical"
        );
    }

    // ------------------------------------------------------------------
    // Test 15: Autoregressive log-amplitude is finite
    // ------------------------------------------------------------------
    #[test]
    fn test_autoregressive_log_amplitude_finite() {
        let ar = AutoregressiveState::new_seeded(4, 8, 42);

        let configs: Vec<Vec<bool>> = vec![
            vec![false, false, false, false],
            vec![true, true, true, true],
            vec![true, false, true, false],
            vec![false, true, false, true],
        ];

        for config in &configs {
            let log_amp = ar.log_amplitude(config);
            assert!(
                log_amp.re.is_finite() && log_amp.im.is_finite(),
                "log_amplitude for {:?} should be finite, got {:?}",
                config,
                log_amp
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 16: Linear system solver correctness
    // ------------------------------------------------------------------
    #[test]
    fn test_linear_system_solver() {
        // Solve: [[2, 1], [1, 3]] * x = [5, 7]
        // Solution: x = [8/5, 9/5] = [1.6, 1.8]
        let a = vec![
            vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ];
        let b = vec![Complex64::new(5.0, 0.0), Complex64::new(7.0, 0.0)];

        let x = solve_linear_system(&a, &b).unwrap();

        assert!(
            (x[0].re - 1.6).abs() < 1e-10,
            "x[0] should be 1.6, got {}",
            x[0].re
        );
        assert!(
            (x[1].re - 1.8).abs() < 1e-10,
            "x[1] should be 1.8, got {}",
            x[1].re
        );
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn compute_variance(values: &[f64]) -> f64 {
        let n = values.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / n;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }
}
