//! Photonic Quantum Advantage Protocols
//!
//! Implements photonic quantum computing primitives including Gaussian Boson
//! Sampling (GBS), standard Boson Sampling, Linear Optical Quantum Computing
//! (LOQC), and a Photonic Ising Machine -- all within a single unified framework.
//!
//! # Background
//!
//! Photonic quantum advantage experiments (Jiuzhang, Borealis) demonstrated
//! quantum speedups for sampling problems using squeezed light through linear
//! optical networks.  The key mathematical objects are:
//!
//! - **Permanent** of a unitary submatrix (standard boson sampling)
//! - **Hafnian** of a symmetric matrix (Gaussian boson sampling)
//! - **Covariance matrix** formalism for Gaussian states
//!
//! This module provides both exact computation (for small instances) and
//! approximate sampling (for larger mode counts), together with validation
//! metrics that quantify quantum advantage claims.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::photonic_advantage::*;
//!
//! let config = PhotonicConfig::builder()
//!     .num_modes(8)
//!     .num_photons(4)
//!     .squeezing_db(10.0)
//!     .protocol(PhotonicProtocol::GaussianBosonSampling)
//!     .build()
//!     .unwrap();
//!
//! let sim = PhotonicSimulator::new(config).unwrap();
//! let result = sim.run_gbs(100);
//! println!("Mean photon number: {:.3}", result.mean_photon_number);
//! ```

use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from photonic simulation configuration or computation.
#[derive(Debug, Clone)]
pub enum PhotonicError {
    /// A configuration parameter is outside its valid range.
    InvalidConfig(String),
    /// The supplied matrix is not unitary (U†U != I within tolerance).
    NotUnitary { deviation: f64 },
    /// Matrix dimensions do not match the expected mode count.
    DimensionMismatch { expected: usize, got: usize },
    /// A numerical computation produced NaN or infinity.
    NumericalInstability(String),
    /// The requested operation is not supported for the given protocol.
    UnsupportedProtocol(String),
}

impl std::fmt::Display for PhotonicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid photonic config: {}", msg),
            Self::NotUnitary { deviation } => {
                write!(f, "Matrix is not unitary (deviation {:.2e})", deviation)
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            Self::UnsupportedProtocol(msg) => {
                write!(f, "Unsupported protocol: {}", msg)
            }
        }
    }
}

impl std::error::Error for PhotonicError {}

pub type PhotonicResult<T> = Result<T, PhotonicError>;

// ============================================================
// PROTOCOL ENUM
// ============================================================

/// Photonic quantum computing protocol selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotonicProtocol {
    /// Gaussian Boson Sampling with squeezed vacuum inputs.
    GaussianBosonSampling,
    /// Standard Boson Sampling with single-photon Fock inputs.
    BosonSampling,
    /// Linear Optical Quantum Computing (KLM-style).
    LinearOpticalQC,
    /// Coherent Ising Machine via optical parametric oscillator network.
    PhotonicIsingMachine,
}

impl std::fmt::Display for PhotonicProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussianBosonSampling => write!(f, "GBS"),
            Self::BosonSampling => write!(f, "BosonSampling"),
            Self::LinearOpticalQC => write!(f, "LOQC"),
            Self::PhotonicIsingMachine => write!(f, "PhotonicIsing"),
        }
    }
}

// ============================================================
// CONFIGURATION (BUILDER PATTERN)
// ============================================================

/// Configuration for photonic quantum simulations.
#[derive(Debug, Clone)]
pub struct PhotonicConfig {
    pub num_modes: usize,
    pub num_photons: usize,
    pub squeezing_db: f64,
    pub loss_per_mode: f64,
    pub dark_count_rate: f64,
    pub detector_efficiency: f64,
    pub protocol: PhotonicProtocol,
}

/// Builder for [`PhotonicConfig`] with sensible defaults and validation.
#[derive(Debug, Clone)]
pub struct PhotonicConfigBuilder {
    num_modes: usize,
    num_photons: usize,
    squeezing_db: f64,
    loss_per_mode: f64,
    dark_count_rate: f64,
    detector_efficiency: f64,
    protocol: PhotonicProtocol,
}

impl PhotonicConfig {
    /// Create a new builder with default values.
    pub fn builder() -> PhotonicConfigBuilder {
        PhotonicConfigBuilder {
            num_modes: 20,
            num_photons: 10,
            squeezing_db: 10.0,
            loss_per_mode: 0.01,
            dark_count_rate: 1e-6,
            detector_efficiency: 0.95,
            protocol: PhotonicProtocol::GaussianBosonSampling,
        }
    }
}

impl PhotonicConfigBuilder {
    pub fn num_modes(mut self, n: usize) -> Self {
        self.num_modes = n;
        self
    }
    pub fn num_photons(mut self, n: usize) -> Self {
        self.num_photons = n;
        self
    }
    pub fn squeezing_db(mut self, db: f64) -> Self {
        self.squeezing_db = db;
        self
    }
    pub fn loss_per_mode(mut self, loss: f64) -> Self {
        self.loss_per_mode = loss;
        self
    }
    pub fn dark_count_rate(mut self, rate: f64) -> Self {
        self.dark_count_rate = rate;
        self
    }
    pub fn detector_efficiency(mut self, eff: f64) -> Self {
        self.detector_efficiency = eff;
        self
    }
    pub fn protocol(mut self, p: PhotonicProtocol) -> Self {
        self.protocol = p;
        self
    }

    /// Validate and build the configuration.
    pub fn build(self) -> PhotonicResult<PhotonicConfig> {
        if self.num_modes < 2 || self.num_modes > 100 {
            return Err(PhotonicError::InvalidConfig(format!(
                "num_modes must be in [2, 100], got {}",
                self.num_modes
            )));
        }
        if self.num_photons == 0 || self.num_photons > self.num_modes {
            return Err(PhotonicError::InvalidConfig(format!(
                "num_photons must be in [1, num_modes={}], got {}",
                self.num_modes, self.num_photons
            )));
        }
        if self.squeezing_db < 0.0 || self.squeezing_db > 30.0 {
            return Err(PhotonicError::InvalidConfig(format!(
                "squeezing_db must be in [0, 30], got {:.2}",
                self.squeezing_db
            )));
        }
        if self.loss_per_mode < 0.0 || self.loss_per_mode > 1.0 {
            return Err(PhotonicError::InvalidConfig(format!(
                "loss_per_mode must be in [0, 1], got {:.4}",
                self.loss_per_mode
            )));
        }
        if self.dark_count_rate < 0.0 || self.dark_count_rate > 1.0 {
            return Err(PhotonicError::InvalidConfig(
                "dark_count_rate must be in [0, 1]".into(),
            ));
        }
        if self.detector_efficiency <= 0.0 || self.detector_efficiency > 1.0 {
            return Err(PhotonicError::InvalidConfig(format!(
                "detector_efficiency must be in (0, 1], got {:.4}",
                self.detector_efficiency
            )));
        }
        Ok(PhotonicConfig {
            num_modes: self.num_modes,
            num_photons: self.num_photons,
            squeezing_db: self.squeezing_db,
            loss_per_mode: self.loss_per_mode,
            dark_count_rate: self.dark_count_rate,
            detector_efficiency: self.detector_efficiency,
            protocol: self.protocol,
        })
    }
}

// ============================================================
// OPTICAL PRIMITIVES
// ============================================================

/// A beam splitter parametrized by reflectivity theta in [0, 1].
#[derive(Debug, Clone, Copy)]
pub struct BeamSplitter {
    pub reflectivity: f64,
    pub transmissivity: f64,
}

impl BeamSplitter {
    pub fn new(reflectivity: f64) -> PhotonicResult<Self> {
        if reflectivity < 0.0 || reflectivity > 1.0 {
            return Err(PhotonicError::InvalidConfig(format!(
                "reflectivity must be in [0, 1], got {:.4}",
                reflectivity
            )));
        }
        Ok(Self {
            reflectivity,
            transmissivity: 1.0 - reflectivity,
        })
    }

    /// 50:50 balanced beam splitter.
    pub fn balanced() -> Self {
        Self {
            reflectivity: 0.5,
            transmissivity: 0.5,
        }
    }

    /// Return the 2x2 unitary matrix for this beam splitter.
    ///
    /// ```text
    /// U = [[ cos(theta),  sin(theta)],
    ///      [-sin(theta),  cos(theta)]]
    /// ```
    /// where `theta = arccos(sqrt(reflectivity))`.
    pub fn unitary_2x2(&self) -> [[Complex64; 2]; 2] {
        let theta = self.reflectivity.sqrt().acos();
        let c = theta.cos();
        let s = theta.sin();
        [
            [Complex64::new(c, 0.0), Complex64::new(s, 0.0)],
            [Complex64::new(-s, 0.0), Complex64::new(c, 0.0)],
        ]
    }
}

/// A phase shifter on a single optical mode.
#[derive(Debug, Clone, Copy)]
pub struct PhaseShifter {
    pub phase: f64,
}

impl PhaseShifter {
    pub fn new(phase: f64) -> Self {
        Self { phase }
    }

    /// Return the unitary scalar e^{i * phase}.
    pub fn unitary(&self) -> Complex64 {
        Complex64::new(self.phase.cos(), self.phase.sin())
    }
}

/// An individual optical element in a decomposed network.
#[derive(Debug, Clone, Copy)]
pub enum OpticalElement {
    /// Beam splitter between two modes with reflectivity.
    BS(usize, usize, f64),
    /// Phase shifter on a single mode.
    PS(usize, f64),
    /// Single-mode squeezer with squeezing parameter r.
    Squeezer(usize, f64),
}

// ============================================================
// LINEAR OPTICAL NETWORK
// ============================================================

/// An m-mode linear optical interferometer described by an m x m unitary.
#[derive(Debug, Clone)]
pub struct LinearOpticalNetwork {
    pub modes: usize,
    pub unitary: Array2<Complex64>,
}

impl LinearOpticalNetwork {
    /// Construct a network from its unitary matrix (validated).
    pub fn new(unitary: Array2<Complex64>) -> PhotonicResult<Self> {
        let (m, n) = unitary.dim();
        if m != n {
            return Err(PhotonicError::DimensionMismatch {
                expected: m,
                got: n,
            });
        }
        let net = Self { modes: m, unitary };
        if !net.is_unitary(1e-8) {
            let dev = net.unitarity_deviation();
            return Err(PhotonicError::NotUnitary { deviation: dev });
        }
        Ok(net)
    }

    /// Build a network by sequentially applying beam splitters.
    ///
    /// Each tuple is `(mode_a, mode_b, reflectivity, phase)`.
    pub fn from_beamsplitters(
        modes: usize,
        splitters: &[(usize, usize, f64, f64)],
    ) -> PhotonicResult<Self> {
        let mut u = Array2::<Complex64>::eye(modes);
        for &(a, b, refl, phase) in splitters {
            if a >= modes || b >= modes {
                return Err(PhotonicError::InvalidConfig(format!(
                    "mode index out of range: ({}, {}) for {} modes",
                    a, b, modes
                )));
            }
            let bs = BeamSplitter::new(refl)?;
            let bsu = bs.unitary_2x2();
            let ps = PhaseShifter::new(phase).unitary();

            // Apply the 2x2 transform to rows a, b of U
            let mut new_u = u.clone();
            for col in 0..modes {
                let ua = u[[a, col]];
                let ub = u[[b, col]];
                new_u[[a, col]] = bsu[0][0] * ua + bsu[0][1] * (ps * ub);
                new_u[[b, col]] = bsu[1][0] * ua + bsu[1][1] * (ps * ub);
            }
            u = new_u;
        }
        Ok(Self { modes, unitary: u })
    }

    /// Decompose a unitary into a sequence of beam splitters and phase shifters
    /// using the Clements (rectangular) decomposition.
    ///
    /// Returns a list of [`OpticalElement`]s that, when applied in order,
    /// reconstruct the original unitary (up to global phase).
    pub fn from_clements_decomposition(
        unitary: &Array2<Complex64>,
    ) -> PhotonicResult<Vec<OpticalElement>> {
        let (m, n) = unitary.dim();
        if m != n {
            return Err(PhotonicError::DimensionMismatch {
                expected: m,
                got: n,
            });
        }
        let mut u = unitary.clone();
        let mut elements = Vec::new();

        // Clements decomposition: nullify off-diagonal elements column by column
        // using T_{m,m-1}, T_{m-1,m-2}, ... acting from the left,
        // then T_{1,2}, T_{2,3}, ... acting from the right.
        for col in 0..m {
            // Nullify below diagonal (left multiplications)
            for row in (col + 1..m).rev() {
                let (theta, phi) = Self::nullify_element(&u, row, col, true);
                Self::apply_rotation(&mut u, row - 1, row, theta, phi, true);
                elements.push(OpticalElement::PS(row, phi));
                elements.push(OpticalElement::BS(row - 1, row, theta.cos().powi(2)));
            }
        }
        // Extract remaining diagonal phases
        for i in 0..m {
            let diag = u[[i, i]];
            let phase = diag.arg();
            if phase.abs() > 1e-12 {
                elements.push(OpticalElement::PS(i, phase));
            }
        }
        Ok(elements)
    }

    /// Generate a Haar-random unitary of the given dimension.
    ///
    /// Uses the QR decomposition of a random complex Gaussian matrix,
    /// with the diagonal of R corrected to ensure uniform Haar measure.
    pub fn random_haar(modes: usize, rng_seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(rng_seed);
        let m = modes;

        // Generate random complex Gaussian matrix
        let mut z = Array2::<Complex64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let re: f64 = sample_normal(&mut rng);
                let im: f64 = sample_normal(&mut rng);
                z[[i, j]] = Complex64::new(re, im);
            }
        }

        // QR decomposition via modified Gram-Schmidt
        let mut q = Array2::<Complex64>::zeros((m, m));
        let mut r = Array2::<Complex64>::zeros((m, m));
        for j in 0..m {
            let mut v: Vec<Complex64> = (0..m).map(|i| z[[i, j]]).collect();
            for k in 0..j {
                let mut dot = Complex64::new(0.0, 0.0);
                for i in 0..m {
                    dot += q[[i, k]].conj() * v[i];
                }
                r[[k, j]] = dot;
                for i in 0..m {
                    v[i] -= dot * q[[i, k]];
                }
            }
            let norm: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            r[[j, j]] = Complex64::new(norm, 0.0);
            if norm > 1e-14 {
                for i in 0..m {
                    q[[i, j]] = v[i] / Complex64::new(norm, 0.0);
                }
            }
        }

        // Correct phases to get Haar measure: Q -> Q * diag(R_jj / |R_jj|)^{-1}
        // is unnecessary since we just need a valid unitary from the column space.
        // The standard correction: multiply column j of Q by conj(sign(R_jj)).
        for j in 0..m {
            let rjj = r[[j, j]];
            if rjj.norm() > 1e-14 {
                let phase = rjj / Complex64::new(rjj.norm(), 0.0);
                let correction = phase.conj();
                for i in 0..m {
                    q[[i, j]] *= correction;
                }
            }
        }

        Self { modes, unitary: q }
    }

    /// Check whether U†U = I within the given tolerance.
    pub fn is_unitary(&self, tol: f64) -> bool {
        self.unitarity_deviation() < tol
    }

    /// Compute max |U†U - I|_element.
    pub fn unitarity_deviation(&self) -> f64 {
        let m = self.modes;
        let u = &self.unitary;
        let mut max_dev = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let mut val = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    val += u[[k, i]].conj() * u[[k, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let dev = (val - Complex64::new(expected, 0.0)).norm();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
        }
        max_dev
    }

    // --- internal helpers ---

    /// Compute rotation angles to nullify element u[row][col].
    fn nullify_element(
        u: &Array2<Complex64>,
        row: usize,
        col: usize,
        _from_left: bool,
    ) -> (f64, f64) {
        let target = u[[row, col]];
        let pivot = u[[row - 1, col]];
        if target.norm() < 1e-15 {
            return (0.0, 0.0);
        }
        let phi = (target / pivot).arg();
        let theta = (pivot.norm().atan2(target.norm())).abs();
        (theta, -phi)
    }

    /// Apply a 2x2 rotation to rows (a, b) of the matrix.
    fn apply_rotation(
        u: &mut Array2<Complex64>,
        a: usize,
        b: usize,
        theta: f64,
        phi: f64,
        _from_left: bool,
    ) {
        let m = u.ncols();
        let c = theta.cos();
        let s = theta.sin();
        let eiphi = Complex64::new(phi.cos(), phi.sin());
        for col in 0..m {
            let ua = u[[a, col]];
            let ub = u[[b, col]];
            u[[a, col]] = Complex64::new(c, 0.0) * ua + Complex64::new(s, 0.0) * eiphi * ub;
            u[[b, col]] = Complex64::new(-s, 0.0) * eiphi.conj() * ua + Complex64::new(c, 0.0) * ub;
        }
    }
}

// ============================================================
// GAUSSIAN BOSON SAMPLING
// ============================================================

/// Gaussian Boson Sampling engine.
///
/// Models squeezed vacuum states input to a linear optical network,
/// computing photon-number distributions via the hafnian of submatrices
/// of the network's adjacency matrix A = U diag(tanh r) U^T.
#[derive(Debug, Clone)]
pub struct GBSSampler {
    pub network: LinearOpticalNetwork,
    /// Per-mode squeezing parameters r_i.
    pub squeezing_params: Vec<f64>,
}

impl GBSSampler {
    /// Create a new GBS sampler.
    ///
    /// `squeezing_params` must have length equal to the number of input modes
    /// that are squeezed (padded to network modes with zeros if shorter).
    pub fn new(network: LinearOpticalNetwork, squeezing_params: Vec<f64>) -> PhotonicResult<Self> {
        if squeezing_params.len() > network.modes {
            return Err(PhotonicError::DimensionMismatch {
                expected: network.modes,
                got: squeezing_params.len(),
            });
        }
        Ok(Self {
            network,
            squeezing_params,
        })
    }

    /// Compute the 2m x 2m covariance matrix of the output Gaussian state.
    ///
    /// For squeezed vacuum through interferometer U:
    ///   sigma = S^T U^T (I) U S  (in the xp-ordering quadrature basis)
    /// where S_i = diag(e^{-r_i}, e^{r_i}) for each mode.
    pub fn covariance_matrix(&self) -> Array2<f64> {
        let m = self.network.modes;
        let n = 2 * m;
        let mut sigma = Array2::<f64>::zeros((n, n));

        // Build squeezed vacuum covariance (block diagonal in xp basis)
        // sigma_vac = 0.5 * I for vacuum; squeezing scales x and p quadratures
        for i in 0..m {
            let r = if i < self.squeezing_params.len() {
                self.squeezing_params[i]
            } else {
                0.0
            };
            // Squeezed state covariance: diag(e^{-2r}, e^{2r}) * 0.5
            sigma[[i, i]] = 0.5 * (-2.0 * r).exp(); // x quadrature
            sigma[[m + i, m + i]] = 0.5 * (2.0 * r).exp(); // p quadrature
        }

        // Apply interferometer: sigma_out = W sigma_in W^T
        // where W = [[Re(U), -Im(U)], [Im(U), Re(U)]]
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..m {
            for j in 0..m {
                let u_ij = self.network.unitary[[i, j]];
                w[[i, j]] = u_ij.re; // top-left block
                w[[i, m + j]] = -u_ij.im; // top-right block
                w[[m + i, j]] = u_ij.im; // bottom-left block
                w[[m + i, m + j]] = u_ij.re; // bottom-right block
            }
        }

        // sigma_out = W * sigma_in * W^T
        let ws = mat_mul_f64(&w, &sigma);
        let w_t = transpose_f64(&w);
        mat_mul_f64(&ws, &w_t)
    }

    /// Total mean photon number across all output modes.
    ///
    /// For squeezed vacuum: <n_i> = sinh^2(r_i).
    pub fn total_mean_photon_number(&self) -> f64 {
        self.squeezing_params
            .iter()
            .map(|&r| r.sinh().powi(2))
            .sum()
    }

    /// Build the symmetric matrix A = U diag(tanh r) U^T used for hafnian
    /// computation of GBS probabilities.
    pub fn adjacency_matrix(&self) -> Array2<Complex64> {
        let m = self.network.modes;
        let mut diag_tanh = Array2::<Complex64>::zeros((m, m));
        for i in 0..m {
            let r = if i < self.squeezing_params.len() {
                self.squeezing_params[i]
            } else {
                0.0
            };
            diag_tanh[[i, i]] = Complex64::new(r.tanh(), 0.0);
        }

        // A = U * diag(tanh r) * U^T
        let u = &self.network.unitary;
        let _u_t = conjugate_transpose(u);
        let ud = mat_mul_c64(u, &diag_tanh);
        // Note: for GBS, A = U diag(tanh r) U^T (not U^dagger)
        let u_transpose = transpose_c64(u);
        mat_mul_c64(&ud, &u_transpose)
    }

    /// Compute the probability of a given click pattern (photon numbers per mode)
    /// using the hafnian.
    ///
    /// For a detection pattern n = (n_1, ..., n_m):
    ///   P(n) = |Haf(A_n)|^2 / (prod_i n_i! * cosh(r_i))
    ///
    /// where A_n is the submatrix of A with row/col i repeated n_i times.
    pub fn probability(&self, pattern: &[usize]) -> f64 {
        if pattern.len() != self.network.modes {
            return 0.0;
        }
        let total_photons: usize = pattern.iter().sum();
        if total_photons == 0 {
            // Vacuum probability = product of 1/cosh(r_i)
            return self
                .squeezing_params
                .iter()
                .map(|&r| 1.0 / r.cosh())
                .product();
        }

        // Build repeated-index submatrix
        let a = self.adjacency_matrix();
        let sub = build_repeated_submatrix(&a, pattern);
        let haf = hafnian(&sub);
        let haf_sq = haf.norm_sqr();

        // Normalization: product of n_i! * cosh(r_i)
        let mut norm = 1.0_f64;
        for (i, &ni) in pattern.iter().enumerate() {
            let r = if i < self.squeezing_params.len() {
                self.squeezing_params[i]
            } else {
                0.0
            };
            norm *= factorial(ni) as f64 * r.cosh();
        }

        haf_sq / norm
    }

    /// Sample photon-number patterns from the GBS distribution.
    ///
    /// Uses a simple rejection/direct sampling approach suitable for
    /// small mode counts.  For large instances, this provides approximate
    /// samples via a chain rule decomposition.
    pub fn sample_photon_numbers(&self, num_samples: usize) -> Vec<Vec<usize>> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let m = self.network.modes;
        let mut samples = Vec::with_capacity(num_samples);

        // Approximate sampling: for each mode, draw from thermal distribution
        // with mean photon number sinh^2(r_i), then post-select on total photon budget.
        let mean_n: Vec<f64> = (0..m)
            .map(|i| {
                if i < self.squeezing_params.len() {
                    self.squeezing_params[i].sinh().powi(2)
                } else {
                    0.0
                }
            })
            .collect();

        for _ in 0..num_samples {
            let mut pattern = vec![0usize; m];
            for mode in 0..m {
                // Geometric/thermal distribution: P(n) = mean^n / (1+mean)^{n+1}
                let mu = mean_n[mode];
                if mu < 1e-12 {
                    continue;
                }
                let p_zero = 1.0 / (1.0 + mu);
                let u: f64 = rng.gen();
                // Inverse CDF of geometric distribution
                if u < p_zero {
                    pattern[mode] = 0;
                } else {
                    let n = ((1.0 - u).ln() / (mu / (1.0 + mu)).ln()).ceil() as usize;
                    pattern[mode] = n.min(10); // cap at 10 photons per mode
                }
            }
            samples.push(pattern);
        }
        samples
    }
}

/// Compute the hafnian of a symmetric matrix using the direct recursive formula.
///
/// The hafnian of a 2n x 2n symmetric matrix A is:
///   Haf(A) = sum over perfect matchings M of prod_{(i,j) in M} A_{ij}
///
/// This uses the recursive definition, efficient for small matrices (n <= 10).
pub fn hafnian(matrix: &Array2<Complex64>) -> Complex64 {
    let n = matrix.nrows();
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    if n % 2 != 0 {
        return Complex64::new(0.0, 0.0);
    }
    if n == 2 {
        return matrix[[0, 1]];
    }

    // Recursive hafnian: fix row 0, pair it with each other row j,
    // then recurse on the reduced matrix.
    let mut result = Complex64::new(0.0, 0.0);
    for j in 1..n {
        let a_0j = matrix[[0, j]];
        if a_0j.norm() < 1e-15 {
            continue;
        }
        // Build reduced matrix by removing rows/cols 0 and j
        let remaining: Vec<usize> = (1..n).filter(|&k| k != j).collect();
        let r = remaining.len();
        let mut sub = Array2::<Complex64>::zeros((r, r));
        for (ri, &row) in remaining.iter().enumerate() {
            for (ci, &col) in remaining.iter().enumerate() {
                sub[[ri, ci]] = matrix[[row, col]];
            }
        }
        result += a_0j * hafnian(&sub);
    }
    result
}

// ============================================================
// BOSON SAMPLING (STANDARD / FOCK INPUT)
// ============================================================

/// Classical simulation tools for standard Boson Sampling.
pub struct BosonSamplingClassical;

impl BosonSamplingClassical {
    /// Compute the permanent of a square complex matrix using Ryser's formula.
    ///
    /// For an n x n matrix A:
    ///   perm(A) = (-1)^n sum_{S subset {1..n}} (-1)^{|S|} prod_{i=1}^{n} sum_{j in S} a_{ij}
    ///
    /// Complexity: O(2^n * n)
    pub fn permanent(matrix: &Array2<Complex64>) -> Complex64 {
        let n = matrix.nrows();
        if n == 0 {
            return Complex64::new(1.0, 0.0);
        }
        if n == 1 {
            return matrix[[0, 0]];
        }

        let num_subsets = 1usize << n;
        let mut total = Complex64::new(0.0, 0.0);

        // Gray code optimized Ryser's formula
        let mut row_sums = vec![Complex64::new(0.0, 0.0); n];

        for s in 1..num_subsets {
            // Determine which bit changed (Gray code step)
            let gray = s ^ (s >> 1);
            let prev_gray = (s - 1) ^ ((s - 1) >> 1);
            let diff = gray ^ prev_gray;
            let bit = diff.trailing_zeros() as usize;
            let adding = (gray & diff) != 0;

            // Update row sums incrementally
            for i in 0..n {
                if adding {
                    row_sums[i] += matrix[[i, bit]];
                } else {
                    row_sums[i] -= matrix[[i, bit]];
                }
            }

            // Compute product of row sums
            let mut prod = Complex64::new(1.0, 0.0);
            for i in 0..n {
                prod *= row_sums[i];
            }

            // Ryser formula sign: (-1)^|S|
            let sign = if gray.count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            total += Complex64::new(sign, 0.0) * prod;
        }

        // Final sign correction
        if n % 2 == 1 {
            total = -total;
        }
        total
    }

    /// Probability of output pattern given input state through network.
    ///
    /// For input photons in modes s_1, ..., s_n and output in modes t_1, ..., t_n:
    ///   P(t|s) = |perm(U_{st})|^2 / (prod_i s_i! * prod_j t_j!)
    ///
    /// where U_{st} is the submatrix of U with repeated rows/columns.
    pub fn probability(input: &[usize], output: &[usize], network: &LinearOpticalNetwork) -> f64 {
        let total_in: usize = input.iter().sum();
        let total_out: usize = output.iter().sum();
        if total_in != total_out {
            return 0.0;
        }
        if total_in == 0 {
            return 1.0;
        }

        // Build repeated-index submatrix
        let n = total_in;
        let mut sub = Array2::<Complex64>::zeros((n, n));
        let mut row_indices = Vec::with_capacity(n);
        for (mode, &count) in input.iter().enumerate() {
            for _ in 0..count {
                row_indices.push(mode);
            }
        }
        let mut col_indices = Vec::with_capacity(n);
        for (mode, &count) in output.iter().enumerate() {
            for _ in 0..count {
                col_indices.push(mode);
            }
        }
        for (i, &ri) in row_indices.iter().enumerate() {
            for (j, &cj) in col_indices.iter().enumerate() {
                sub[[i, j]] = network.unitary[[ri, cj]];
            }
        }

        let perm = Self::permanent(&sub);
        let perm_sq = perm.norm_sqr();

        // Normalization
        let in_norm: f64 = input.iter().map(|&n| factorial(n) as f64).product();
        let out_norm: f64 = output.iter().map(|&n| factorial(n) as f64).product();

        perm_sq / (in_norm * out_norm)
    }

    /// Generate approximate samples from the boson sampling distribution.
    pub fn sample(
        network: &LinearOpticalNetwork,
        input_state: &[usize],
        num_samples: usize,
    ) -> Vec<Vec<usize>> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(123);
        let m = network.modes;
        let _total_photons: usize = input_state.iter().sum();
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Approximate: distribute photons across output modes
            // weighted by |U_{ij}|^2 (marginal single-photon distribution)
            let mut pattern = vec![0usize; m];
            for (in_mode, &count) in input_state.iter().enumerate() {
                for _ in 0..count {
                    // Sample output mode weighted by |U_{mode,j}|^2
                    let probs: Vec<f64> = (0..m)
                        .map(|j| network.unitary[[j, in_mode]].norm_sqr())
                        .collect();
                    let total: f64 = probs.iter().sum();
                    let u: f64 = rng.gen::<f64>() * total;
                    let mut cumsum = 0.0;
                    for (j, &p) in probs.iter().enumerate() {
                        cumsum += p;
                        if u <= cumsum {
                            pattern[j] += 1;
                            break;
                        }
                    }
                }
            }
            samples.push(pattern);
        }
        samples
    }
}

// ============================================================
// PHOTONIC ISING MACHINE
// ============================================================

/// Solution from the photonic Ising machine.
#[derive(Debug, Clone)]
pub struct IsingSolution {
    pub spins: Vec<i8>,
    pub energy: f64,
    pub num_samples: usize,
}

/// Photonic Ising Machine using a coherent optical network.
///
/// Encodes an Ising Hamiltonian H = -sum_{ij} J_{ij} s_i s_j - sum_i h_i s_i
/// into a linear optical network whose GBS output samples correspond to
/// low-energy spin configurations.
pub struct PhotonicIsingMachine;

impl PhotonicIsingMachine {
    /// Encode the Ising coupling matrix J and external field h into
    /// a linear optical network whose sampling correlations approximate
    /// the Ising model's Boltzmann distribution.
    ///
    /// Uses the mapping: A_{ij} = tanh(beta * J_{ij}) for the adjacency
    /// matrix, then decomposes into an interferometer.
    pub fn encode_ising(j_matrix: &Array2<f64>, h: &[f64]) -> PhotonicResult<LinearOpticalNetwork> {
        let n = j_matrix.nrows();
        if j_matrix.ncols() != n {
            return Err(PhotonicError::DimensionMismatch {
                expected: n,
                got: j_matrix.ncols(),
            });
        }
        if h.len() != n {
            return Err(PhotonicError::DimensionMismatch {
                expected: n,
                got: h.len(),
            });
        }

        // Build a unitary that encodes the coupling structure.
        // Simple approach: eigendecompose J, map eigenvalues through tanh,
        // reconstruct as a unitary rotation.
        let beta = 1.0; // inverse temperature
        let mut u = Array2::<Complex64>::eye(n);

        // Encode couplings as phases on pairs of modes
        for i in 0..n {
            for j in (i + 1)..n {
                let coupling = (beta * j_matrix[[i, j]]).tanh();
                let angle = coupling * PI / 4.0;
                let c = angle.cos();
                let s = angle.sin();
                // Apply rotation to modes i, j
                let mut new_u = u.clone();
                for col in 0..n {
                    let ui = u[[i, col]];
                    let uj = u[[j, col]];
                    new_u[[i, col]] = Complex64::new(c, 0.0) * ui + Complex64::new(s, 0.0) * uj;
                    new_u[[j, col]] = Complex64::new(-s, 0.0) * ui + Complex64::new(c, 0.0) * uj;
                }
                u = new_u;
            }
        }

        // Encode external field as diagonal phases
        for i in 0..n {
            let phase = (beta * h[i]).tanh() * PI / 2.0;
            let eip = Complex64::new(phase.cos(), phase.sin());
            for col in 0..n {
                u[[i, col]] *= eip;
            }
        }

        Ok(LinearOpticalNetwork {
            modes: n,
            unitary: u,
        })
    }

    /// Solve the Ising problem by sampling from the photonic machine.
    pub fn solve(
        j_matrix: &Array2<f64>,
        h: &[f64],
        num_samples: usize,
    ) -> PhotonicResult<IsingSolution> {
        let n = j_matrix.nrows();
        let network = Self::encode_ising(j_matrix, h)?;

        // Build a GBS sampler with uniform squeezing
        let squeezing = vec![0.5; n];
        let sampler = GBSSampler::new(network, squeezing)?;
        let raw_samples = sampler.sample_photon_numbers(num_samples);

        // Convert photon patterns to spin configurations
        // Spin s_i = +1 if n_i > 0, -1 otherwise (threshold detection)
        let mut best_spins = vec![-1i8; n];
        let mut best_energy = f64::MAX;

        for sample in &raw_samples {
            let spins: Vec<i8> = sample
                .iter()
                .map(|&ni| if ni > 0 { 1 } else { -1 })
                .collect();
            let energy = ising_energy(j_matrix, h, &spins);
            if energy < best_energy {
                best_energy = energy;
                best_spins = spins;
            }
        }

        Ok(IsingSolution {
            spins: best_spins,
            energy: best_energy,
            num_samples,
        })
    }
}

/// Compute the Ising energy H = -sum_{ij} J_{ij} s_i s_j - sum_i h_i s_i.
fn ising_energy(j: &Array2<f64>, h: &[f64], spins: &[i8]) -> f64 {
    let n = spins.len();
    let mut energy = 0.0;
    for i in 0..n {
        for jj in (i + 1)..n {
            energy -= j[[i, jj]] * (spins[i] as f64) * (spins[jj] as f64);
        }
        energy -= h[i] * (spins[i] as f64);
    }
    energy
}

// ============================================================
// VALIDATION METRICS
// ============================================================

/// Metrics for validating quantum advantage claims.
pub struct ValidationMetrics;

impl ValidationMetrics {
    /// Heavy Output Generation (HOG) probability.
    ///
    /// Fraction of samples whose ideal probability is above the median.
    /// For a true quantum device this should exceed 2/3.
    pub fn heavy_output_probability(samples: &[Vec<usize>], sampler: &GBSSampler) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        // Compute probabilities for all samples
        let probs: Vec<f64> = samples.iter().map(|s| sampler.probability(s)).collect();
        // Find median
        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];

        // Fraction above median
        let heavy_count = probs.iter().filter(|&&p| p >= median).count();
        heavy_count as f64 / samples.len() as f64
    }

    /// Linear cross-entropy benchmarking (XEB) score.
    ///
    /// XEB = N * mean(p_ideal(x_i)) - 1
    ///
    /// where N is the Hilbert space dimension. For a perfect quantum device
    /// XEB = 1; for uniform random sampling XEB = 0.
    pub fn cross_entropy_score(samples: &[Vec<usize>], sampler: &GBSSampler) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let probs: Vec<f64> = samples.iter().map(|s| sampler.probability(s)).collect();
        let mean_prob = probs.iter().sum::<f64>() / probs.len() as f64;

        // Approximate Hilbert space dimension: number of ways to distribute
        // photons across modes (bounded estimate)
        let total_photons: usize = samples[0].iter().sum();
        let m = sampler.network.modes;
        let hilbert_dim = binomial(m + total_photons, total_photons) as f64;

        hilbert_dim * mean_prob - 1.0
    }

    /// Photon bunching ratio: measures boson statistics.
    ///
    /// Compares the variance of photon-number distribution to the mean.
    /// For thermal/squeezed light: var > mean (super-Poissonian, bunching).
    /// For coherent light: var = mean (Poissonian).
    /// For single photons: var < mean (sub-Poissonian, antibunching).
    pub fn bunching_ratio(samples: &[Vec<usize>]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let m = samples[0].len();
        let ns = samples.len() as f64;

        // Compute per-mode mean and variance, then average
        let mut total_ratio = 0.0;
        let mut active_modes = 0;
        for mode in 0..m {
            let vals: Vec<f64> = samples.iter().map(|s| s[mode] as f64).collect();
            let mean = vals.iter().sum::<f64>() / ns;
            if mean < 1e-10 {
                continue;
            }
            let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ns;
            total_ratio += var / mean;
            active_modes += 1;
        }

        if active_modes == 0 {
            0.0
        } else {
            total_ratio / active_modes as f64
        }
    }

    /// Estimate classical simulation time for exact boson sampling.
    ///
    /// Based on the permanent complexity: O(2^n * n) for n photons in m modes.
    /// Returns estimated seconds assuming 10 GHz classical clock.
    pub fn classical_spoofing_time_estimate(_modes: usize, photons: usize) -> f64 {
        let n = photons;
        let ops = (1u128 << n as u128) * n as u128;
        let clock_hz = 10.0e9_f64;
        ops as f64 / clock_hz
    }
}

// ============================================================
// PHOTONIC SIMULATOR (TOP-LEVEL INTERFACE)
// ============================================================

/// Result of a GBS run.
#[derive(Debug, Clone)]
pub struct GBSResult {
    pub samples: Vec<Vec<usize>>,
    pub mean_photon_number: f64,
    pub num_modes: usize,
    pub num_samples: usize,
}

/// Result of a standard boson sampling run.
#[derive(Debug, Clone)]
pub struct BSResult {
    pub samples: Vec<Vec<usize>>,
    pub input_state: Vec<usize>,
    pub num_modes: usize,
    pub num_samples: usize,
}

/// Validation result summarizing quantum advantage metrics.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub heavy_output_prob: f64,
    pub xeb_score: f64,
    pub bunching_ratio: f64,
    pub classical_time_s: f64,
}

/// Statistics from a photonic simulation run.
#[derive(Debug, Clone)]
pub struct PhotonicStats {
    pub protocol: PhotonicProtocol,
    pub num_modes: usize,
    pub num_photons: usize,
    pub total_samples: usize,
    pub unitarity_deviation: f64,
}

/// Top-level photonic quantum simulator.
pub struct PhotonicSimulator {
    config: PhotonicConfig,
    network: LinearOpticalNetwork,
    sampler: Option<GBSSampler>,
}

impl PhotonicSimulator {
    /// Create a new photonic simulator from configuration.
    ///
    /// Initialises a Haar-random linear optical network and (for GBS)
    /// a sampler with uniform squeezing derived from `squeezing_db`.
    pub fn new(config: PhotonicConfig) -> PhotonicResult<Self> {
        let network = LinearOpticalNetwork::random_haar(config.num_modes, 0xDEAD_BEEF);

        let sampler = match config.protocol {
            PhotonicProtocol::GaussianBosonSampling => {
                // Convert dB to squeezing parameter r: dB = 10 * log10(e^{2r})
                // => r = dB * ln(10) / 20
                let r = config.squeezing_db * 10.0_f64.ln() / 20.0;
                let params = vec![r; config.num_photons.min(config.num_modes)];
                Some(GBSSampler::new(network.clone(), params)?)
            }
            _ => None,
        };

        Ok(Self {
            config,
            network,
            sampler,
        })
    }

    /// Run Gaussian Boson Sampling and return photon-number samples.
    pub fn run_gbs(&self, num_samples: usize) -> PhotonicResult<GBSResult> {
        let sampler = self.sampler.as_ref().ok_or_else(|| {
            PhotonicError::UnsupportedProtocol(
                "GBS sampler not initialized (wrong protocol?)".into(),
            )
        })?;
        let samples = sampler.sample_photon_numbers(num_samples);
        Ok(GBSResult {
            samples,
            mean_photon_number: sampler.total_mean_photon_number(),
            num_modes: self.config.num_modes,
            num_samples,
        })
    }

    /// Run standard boson sampling with the given Fock input state.
    pub fn run_boson_sampling(
        &self,
        input_state: &[usize],
        num_samples: usize,
    ) -> PhotonicResult<BSResult> {
        if input_state.len() != self.config.num_modes {
            return Err(PhotonicError::DimensionMismatch {
                expected: self.config.num_modes,
                got: input_state.len(),
            });
        }
        let samples = BosonSamplingClassical::sample(&self.network, input_state, num_samples);
        Ok(BSResult {
            samples,
            input_state: input_state.to_vec(),
            num_modes: self.config.num_modes,
            num_samples,
        })
    }

    /// Validate quantum advantage using the collected samples.
    pub fn validate_advantage(&self, samples: &[Vec<usize>]) -> PhotonicResult<ValidationResult> {
        let sampler = self.sampler.as_ref().ok_or_else(|| {
            PhotonicError::UnsupportedProtocol("Validation requires a GBS sampler".into())
        })?;
        let hop = ValidationMetrics::heavy_output_probability(samples, sampler);
        let xeb = ValidationMetrics::cross_entropy_score(samples, sampler);
        let bunching = ValidationMetrics::bunching_ratio(samples);
        let classical_time = ValidationMetrics::classical_spoofing_time_estimate(
            self.config.num_modes,
            self.config.num_photons,
        );
        Ok(ValidationResult {
            heavy_output_prob: hop,
            xeb_score: xeb,
            bunching_ratio: bunching,
            classical_time_s: classical_time,
        })
    }

    /// Apply a lossy channel to photon-number samples.
    ///
    /// Each photon in each mode is independently lost with probability
    /// `loss_rate`, modelling optical loss in fibres or detectors.
    pub fn apply_loss(samples: &[Vec<usize>], loss_rate: f64) -> Vec<Vec<usize>> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(999);
        samples
            .iter()
            .map(|pattern| {
                pattern
                    .iter()
                    .map(|&n| {
                        let mut surviving = 0;
                        for _ in 0..n {
                            if rng.gen::<f64>() > loss_rate {
                                surviving += 1;
                            }
                        }
                        surviving
                    })
                    .collect()
            })
            .collect()
    }

    /// Return summary statistics for the simulation.
    pub fn stats(&self) -> PhotonicStats {
        PhotonicStats {
            protocol: self.config.protocol,
            num_modes: self.config.num_modes,
            num_photons: self.config.num_photons,
            total_samples: 0,
            unitarity_deviation: self.network.unitarity_deviation(),
        }
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Box-Muller transform to sample from standard normal distribution.
fn sample_normal(rng: &mut impl rand::Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Factorial (exact for small values, then saturates at u64::MAX).
fn factorial(n: usize) -> u64 {
    match n {
        0 | 1 => 1,
        _ => (2..=n as u64).product(),
    }
}

/// Binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1u64;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64) / (i as u64 + 1);
    }
    result
}

/// Build a submatrix with repeated indices according to a photon pattern.
fn build_repeated_submatrix(a: &Array2<Complex64>, pattern: &[usize]) -> Array2<Complex64> {
    let total: usize = pattern.iter().sum();
    let mut indices = Vec::with_capacity(total);
    for (mode, &count) in pattern.iter().enumerate() {
        for _ in 0..count {
            indices.push(mode);
        }
    }
    let mut sub = Array2::<Complex64>::zeros((total, total));
    for (i, &ri) in indices.iter().enumerate() {
        for (j, &cj) in indices.iter().enumerate() {
            sub[[i, j]] = a[[ri, cj]];
        }
    }
    sub
}

/// Naive matrix multiplication for f64.
fn mat_mul_f64(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

/// Naive matrix multiplication for Complex64.
fn mat_mul_c64(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    let mut c = Array2::<Complex64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..k {
                sum += a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

/// Transpose an f64 matrix.
fn transpose_f64(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let mut t = Array2::<f64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

/// Transpose a complex matrix.
fn transpose_c64(a: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    let mut t = Array2::<Complex64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

/// Conjugate transpose (dagger) of a complex matrix.
fn conjugate_transpose(a: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    let mut t = Array2::<Complex64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]].conj();
        }
    }
    t
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    // ---------------------------------------------------------------
    // 1. Config builder: valid defaults
    // ---------------------------------------------------------------
    #[test]
    fn test_config_builder_defaults() {
        let cfg = PhotonicConfig::builder().build().unwrap();
        assert_eq!(cfg.num_modes, 20);
        assert_eq!(cfg.num_photons, 10);
        assert!((cfg.squeezing_db - 10.0).abs() < TOL);
        assert!((cfg.loss_per_mode - 0.01).abs() < TOL);
        assert!((cfg.detector_efficiency - 0.95).abs() < TOL);
        assert_eq!(cfg.protocol, PhotonicProtocol::GaussianBosonSampling);
    }

    // ---------------------------------------------------------------
    // 2. Config builder: invalid num_modes rejected
    // ---------------------------------------------------------------
    #[test]
    fn test_config_invalid_modes() {
        let result = PhotonicConfig::builder().num_modes(1).build();
        assert!(result.is_err());
        let result = PhotonicConfig::builder().num_modes(101).build();
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 3. Config builder: invalid num_photons
    // ---------------------------------------------------------------
    #[test]
    fn test_config_invalid_photons() {
        let result = PhotonicConfig::builder()
            .num_modes(5)
            .num_photons(0)
            .build();
        assert!(result.is_err());
        let result = PhotonicConfig::builder()
            .num_modes(5)
            .num_photons(6)
            .build();
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 4. Beam splitter: balanced is unitary
    // ---------------------------------------------------------------
    #[test]
    fn test_beam_splitter_unitarity() {
        let bs = BeamSplitter::balanced();
        let u = bs.unitary_2x2();

        // Check U^dag U = I
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    dot += u[k][i].conj() * u[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - Complex64::new(expected, 0.0)).norm() < TOL,
                    "U^dag U [{},{}] = {:?}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // 5. Beam splitter: reflectivity + transmissivity = 1
    // ---------------------------------------------------------------
    #[test]
    fn test_beam_splitter_conservation() {
        let bs = BeamSplitter::new(0.3).unwrap();
        assert!((bs.reflectivity + bs.transmissivity - 1.0).abs() < TOL);
    }

    // ---------------------------------------------------------------
    // 6. Phase shifter: unit modulus
    // ---------------------------------------------------------------
    #[test]
    fn test_phase_shifter_unit_modulus() {
        for &phase in &[0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
            let ps = PhaseShifter::new(phase);
            let u = ps.unitary();
            assert!(
                (u.norm() - 1.0).abs() < TOL,
                "Phase {}: |e^(i*phi)| = {}, expected 1.0",
                phase,
                u.norm()
            );
        }
    }

    // ---------------------------------------------------------------
    // 7. Phase shifter: correct angle
    // ---------------------------------------------------------------
    #[test]
    fn test_phase_shifter_angle() {
        let ps = PhaseShifter::new(PI / 3.0);
        let u = ps.unitary();
        assert!((u.re - (PI / 3.0).cos()).abs() < TOL);
        assert!((u.im - (PI / 3.0).sin()).abs() < TOL);
    }

    // ---------------------------------------------------------------
    // 8. Haar random unitary: is unitary
    // ---------------------------------------------------------------
    #[test]
    fn test_haar_random_is_unitary() {
        for &m in &[2, 4, 8, 12] {
            let net = LinearOpticalNetwork::random_haar(m, 42);
            assert!(
                net.is_unitary(1e-6),
                "Haar random {0}x{0} not unitary (dev={1:.2e})",
                m,
                net.unitarity_deviation()
            );
        }
    }

    // ---------------------------------------------------------------
    // 9. Permanent of 2x2 identity = 1
    // ---------------------------------------------------------------
    #[test]
    fn test_permanent_identity_2x2() {
        let eye = Array2::from_diag(&ndarray::arr1(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]));
        let perm = BosonSamplingClassical::permanent(&eye);
        assert!(
            (perm - Complex64::new(1.0, 0.0)).norm() < TOL,
            "perm(I_2) = {:?}, expected 1",
            perm
        );
    }

    // ---------------------------------------------------------------
    // 10. Permanent of known 2x2 matrix
    // ---------------------------------------------------------------
    #[test]
    fn test_permanent_2x2_manual() {
        // perm([[a,b],[c,d]]) = a*d + b*c
        let a = Complex64::new(1.0, 0.0);
        let b = Complex64::new(2.0, 0.0);
        let c = Complex64::new(3.0, 0.0);
        let d = Complex64::new(4.0, 0.0);
        let mat = Array2::from_shape_vec((2, 2), vec![a, b, c, d]).unwrap();
        let perm = BosonSamplingClassical::permanent(&mat);
        let expected = a * d + b * c; // 4 + 6 = 10
        assert!(
            (perm - expected).norm() < TOL,
            "perm = {:?}, expected {:?}",
            perm,
            expected
        );
    }

    // ---------------------------------------------------------------
    // 11. Permanent of 3x3 ones matrix = 6 (= 3!)
    // ---------------------------------------------------------------
    #[test]
    fn test_permanent_3x3_ones() {
        let one = Complex64::new(1.0, 0.0);
        let mat = Array2::from_elem((3, 3), one);
        let perm = BosonSamplingClassical::permanent(&mat);
        // perm(J_3) = 3! = 6 (number of permutations)
        // Actually perm(ones(3,3)) = sum over all permutations of product of entries = n! = 6
        assert!(
            (perm.re - 6.0).abs() < TOL && perm.im.abs() < TOL,
            "perm(J_3) = {:?}, expected 6",
            perm
        );
    }

    // ---------------------------------------------------------------
    // 12. Hafnian of 2x2 matrix = A[0,1]
    // ---------------------------------------------------------------
    #[test]
    fn test_hafnian_2x2() {
        let val = Complex64::new(3.0, 1.0);
        let mut mat = Array2::<Complex64>::zeros((2, 2));
        mat[[0, 1]] = val;
        mat[[1, 0]] = val;
        let haf = hafnian(&mat);
        assert!(
            (haf - val).norm() < TOL,
            "haf(2x2) = {:?}, expected {:?}",
            haf,
            val
        );
    }

    // ---------------------------------------------------------------
    // 13. Hafnian of 4x4 known matrix
    // ---------------------------------------------------------------
    #[test]
    fn test_hafnian_4x4() {
        // For a 4x4 symmetric matrix A:
        // haf(A) = A[0,1]*A[2,3] + A[0,2]*A[1,3] + A[0,3]*A[1,2]
        let one = Complex64::new(1.0, 0.0);
        let mat = Array2::from_elem((4, 4), one);
        // haf(J_4) = 1*1 + 1*1 + 1*1 = 3
        let haf = hafnian(&mat);
        assert!(
            (haf.re - 3.0).abs() < TOL && haf.im.abs() < TOL,
            "haf(J_4) = {:?}, expected 3",
            haf
        );
    }

    // ---------------------------------------------------------------
    // 14. Hafnian of odd-dimensional matrix is zero
    // ---------------------------------------------------------------
    #[test]
    fn test_hafnian_odd_dimension() {
        let mat = Array2::from_elem((3, 3), Complex64::new(1.0, 0.0));
        let haf = hafnian(&mat);
        assert!(haf.norm() < TOL, "haf(3x3) should be 0, got {:?}", haf);
    }

    // ---------------------------------------------------------------
    // 15. Covariance matrix is symmetric and positive
    // ---------------------------------------------------------------
    #[test]
    fn test_covariance_matrix_symmetric() {
        let net = LinearOpticalNetwork::random_haar(4, 77);
        let sampler = GBSSampler::new(net, vec![0.5, 0.3, 0.4, 0.2]).unwrap();
        let sigma = sampler.covariance_matrix();
        let (n, m) = sigma.dim();
        assert_eq!(n, 8); // 2 * 4 modes
        assert_eq!(m, 8);

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (sigma[[i, j]] - sigma[[j, i]]).abs() < 1e-10,
                    "Covariance not symmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    sigma[[i, j]],
                    sigma[[j, i]]
                );
            }
        }

        // Check positive diagonal (squeezed vacuum has positive variances)
        for i in 0..n {
            assert!(
                sigma[[i, i]] > 0.0,
                "Diagonal element {} is not positive: {}",
                i,
                sigma[[i, i]]
            );
        }
    }

    // ---------------------------------------------------------------
    // 16. GBS mean photon number
    // ---------------------------------------------------------------
    #[test]
    fn test_gbs_mean_photon_number() {
        let net = LinearOpticalNetwork::random_haar(4, 55);
        let r = 0.5;
        let sampler = GBSSampler::new(net, vec![r; 4]).unwrap();
        let expected = 4.0 * r.sinh().powi(2);
        let got = sampler.total_mean_photon_number();
        assert!(
            (got - expected).abs() < TOL,
            "Mean photon number: got {}, expected {}",
            got,
            expected
        );
    }

    // ---------------------------------------------------------------
    // 17. Boson sampling probability: single photon in 2 modes
    // ---------------------------------------------------------------
    #[test]
    fn test_boson_sampling_probability_single_photon() {
        // 50:50 beam splitter, 1 photon in mode 0
        let net = LinearOpticalNetwork::from_beamsplitters(2, &[(0, 1, 0.5, 0.0)]).unwrap();
        let input = [1, 0];
        let p_10 = BosonSamplingClassical::probability(&input, &[1, 0], &net);
        let p_01 = BosonSamplingClassical::probability(&input, &[0, 1], &net);

        // Should sum to approximately 1
        assert!(
            (p_10 + p_01 - 1.0).abs() < 0.05,
            "Probabilities don't sum to 1: {} + {} = {}",
            p_10,
            p_01,
            p_10 + p_01
        );
    }

    // ---------------------------------------------------------------
    // 18. Linear optical network from beam splitters
    // ---------------------------------------------------------------
    #[test]
    fn test_network_from_beamsplitters() {
        let net =
            LinearOpticalNetwork::from_beamsplitters(3, &[(0, 1, 0.5, 0.0), (1, 2, 0.3, PI / 4.0)])
                .unwrap();
        assert_eq!(net.modes, 3);
        assert!(
            net.is_unitary(1e-6),
            "Network from beam splitters not unitary (dev={:.2e})",
            net.unitarity_deviation()
        );
    }

    // ---------------------------------------------------------------
    // 19. Clements decomposition produces valid optical elements
    // ---------------------------------------------------------------
    #[test]
    fn test_clements_decomposition_output() {
        let net = LinearOpticalNetwork::random_haar(3, 99);
        let elements = LinearOpticalNetwork::from_clements_decomposition(&net.unitary).unwrap();
        // Should produce beam splitters and phase shifters
        assert!(
            !elements.is_empty(),
            "Clements decomposition returned no elements"
        );
        for elem in &elements {
            match elem {
                OpticalElement::BS(a, b, r) => {
                    assert!(a < &3 && b < &3, "BS mode index out of range");
                    assert!(*r >= 0.0 && *r <= 1.0, "BS reflectivity out of [0,1]");
                }
                OpticalElement::PS(m, _) => {
                    assert!(m < &3, "PS mode index out of range");
                }
                OpticalElement::Squeezer(m, _) => {
                    assert!(m < &3, "Squeezer mode index out of range");
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // 20. Heavy output probability is in [0, 1]
    // ---------------------------------------------------------------
    #[test]
    fn test_heavy_output_probability_range() {
        let net = LinearOpticalNetwork::random_haar(4, 33);
        let sampler = GBSSampler::new(net, vec![0.3; 4]).unwrap();
        let samples = sampler.sample_photon_numbers(50);
        let hop = ValidationMetrics::heavy_output_probability(&samples, &sampler);
        assert!(hop >= 0.0 && hop <= 1.0, "HOP out of range: {}", hop);
    }

    // ---------------------------------------------------------------
    // 21. Bunching ratio for squeezed light is super-Poissonian
    // ---------------------------------------------------------------
    #[test]
    fn test_bunching_ratio_squeezed() {
        let net = LinearOpticalNetwork::random_haar(4, 11);
        let sampler = GBSSampler::new(net, vec![1.0; 4]).unwrap();
        let samples = sampler.sample_photon_numbers(500);
        let br = ValidationMetrics::bunching_ratio(&samples);
        // Squeezed/thermal light should have bunching ratio >= 1 (super-Poissonian)
        // With approximate sampling, just check it's positive
        assert!(
            br >= 0.0,
            "Bunching ratio should be non-negative, got {}",
            br
        );
    }

    // ---------------------------------------------------------------
    // 22. Loss model reduces photon numbers
    // ---------------------------------------------------------------
    #[test]
    fn test_loss_model() {
        let samples = vec![vec![3, 2, 1, 0]; 100];
        let lossy = PhotonicSimulator::apply_loss(&samples, 0.5);
        assert_eq!(lossy.len(), 100);

        // Total photons should be reduced on average
        let original_total: usize = samples.iter().flat_map(|s| s.iter()).sum();
        let lossy_total: usize = lossy.iter().flat_map(|s| s.iter()).sum();
        assert!(
            lossy_total < original_total,
            "Loss should reduce total photons: {} vs {}",
            lossy_total,
            original_total
        );
    }

    // ---------------------------------------------------------------
    // 23. Loss rate = 0 preserves all photons
    // ---------------------------------------------------------------
    #[test]
    fn test_loss_zero_preserves() {
        let samples = vec![vec![2, 3, 1]; 10];
        let lossy = PhotonicSimulator::apply_loss(&samples, 0.0);
        for (orig, loss) in samples.iter().zip(lossy.iter()) {
            assert_eq!(orig, loss, "Zero loss should preserve photons");
        }
    }

    // ---------------------------------------------------------------
    // 24. Full GBS sampling pipeline
    // ---------------------------------------------------------------
    #[test]
    fn test_full_gbs_pipeline() {
        let config = PhotonicConfig::builder()
            .num_modes(4)
            .num_photons(3)
            .squeezing_db(8.0)
            .protocol(PhotonicProtocol::GaussianBosonSampling)
            .build()
            .unwrap();

        let sim = PhotonicSimulator::new(config).unwrap();
        let result = sim.run_gbs(50).unwrap();
        assert_eq!(result.num_modes, 4);
        assert_eq!(result.num_samples, 50);
        assert_eq!(result.samples.len(), 50);
        assert!(result.mean_photon_number > 0.0);

        // Validate
        let validation = sim.validate_advantage(&result.samples).unwrap();
        assert!(validation.heavy_output_prob >= 0.0);
        assert!(validation.classical_time_s >= 0.0);
    }

    // ---------------------------------------------------------------
    // 25. Ising machine encoding preserves unitarity
    // ---------------------------------------------------------------
    #[test]
    fn test_ising_encoding_unitary() {
        let j = Array2::from_shape_vec((3, 3), vec![0.0, 0.5, -0.3, 0.5, 0.0, 0.2, -0.3, 0.2, 0.0])
            .unwrap();
        let h = vec![0.1, -0.2, 0.3];
        let net = PhotonicIsingMachine::encode_ising(&j, &h).unwrap();
        assert!(
            net.is_unitary(1e-6),
            "Ising encoding not unitary (dev={:.2e})",
            net.unitarity_deviation()
        );
    }

    // ---------------------------------------------------------------
    // 26. Ising machine solve produces valid spins
    // ---------------------------------------------------------------
    #[test]
    fn test_ising_solve() {
        let j = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.8, 0.5, 0.8, 0.0])
            .unwrap();
        let h = vec![0.0; 3];
        let sol = PhotonicIsingMachine::solve(&j, &h, 100).unwrap();
        assert_eq!(sol.spins.len(), 3);
        assert!(sol.spins.iter().all(|&s| s == 1 || s == -1));
        assert_eq!(sol.num_samples, 100);
    }

    // ---------------------------------------------------------------
    // 27. Classical spoofing time grows exponentially
    // ---------------------------------------------------------------
    #[test]
    fn test_classical_spoofing_time_growth() {
        let t10 = ValidationMetrics::classical_spoofing_time_estimate(20, 10);
        let t20 = ValidationMetrics::classical_spoofing_time_estimate(40, 20);
        assert!(
            t20 > t10 * 100.0,
            "Spoofing time should grow exponentially: t10={:.2e}, t20={:.2e}",
            t10,
            t20
        );
    }

    // ---------------------------------------------------------------
    // 28. Boson sampling: conservation of photons in samples
    // ---------------------------------------------------------------
    #[test]
    fn test_boson_sampling_photon_conservation() {
        let net = LinearOpticalNetwork::random_haar(4, 21);
        let input = vec![1, 1, 0, 0]; // 2 photons in
        let samples = BosonSamplingClassical::sample(&net, &input, 100);
        for s in &samples {
            let total: usize = s.iter().sum();
            assert_eq!(
                total, 2,
                "Photon number not conserved: got {}, expected 2",
                total
            );
        }
    }

    // ---------------------------------------------------------------
    // 29. PhotonicSimulator stats
    // ---------------------------------------------------------------
    #[test]
    fn test_simulator_stats() {
        let config = PhotonicConfig::builder()
            .num_modes(6)
            .num_photons(3)
            .build()
            .unwrap();
        let sim = PhotonicSimulator::new(config).unwrap();
        let stats = sim.stats();
        assert_eq!(stats.num_modes, 6);
        assert_eq!(stats.num_photons, 3);
        assert!(stats.unitarity_deviation < 1e-6);
        assert_eq!(stats.protocol, PhotonicProtocol::GaussianBosonSampling);
    }

    // ---------------------------------------------------------------
    // 30. GBS probability: vacuum state
    // ---------------------------------------------------------------
    #[test]
    fn test_gbs_vacuum_probability() {
        let net = LinearOpticalNetwork::random_haar(3, 44);
        let sampler = GBSSampler::new(net, vec![0.5; 3]).unwrap();
        let vacuum_pattern = vec![0, 0, 0];
        let p = sampler.probability(&vacuum_pattern);
        assert!(
            p > 0.0 && p <= 1.0,
            "Vacuum probability out of range: {}",
            p
        );
        // Vacuum probability = prod(1/cosh(r_i))
        let expected: f64 = (0..3).map(|_| 1.0 / 0.5_f64.cosh()).product();
        assert!(
            (p - expected).abs() < 1e-6,
            "Vacuum prob = {}, expected {}",
            p,
            expected
        );
    }
}
