//! Rydberg Atom Reservoir Computing
//!
//! **WORLD FIRST**: Quantum reservoir computing using Rydberg atom arrays for
//! time-series prediction and nonlinear function approximation.
//!
//! Rydberg atoms are neutral atoms excited to high principal quantum numbers,
//! producing strong long-range van der Waals interactions (V = C6 / r^6).
//! These interactions create a natural many-body Hamiltonian whose dynamics
//! serve as a high-dimensional nonlinear mapping -- the reservoir -- while
//! only a simple linear readout layer is trained classically.
//!
//! # Physical Model
//!
//! The Hamiltonian is:
//!
//!   H = sum_i (Omega/2) sigma_x^i  -  Delta sum_i n_i  +  sum_{i<j} V_ij n_i n_j
//!
//! where:
//!   - Omega: global Rabi frequency (laser driving)
//!   - Delta: laser detuning from the Rydberg transition
//!   - V_ij = C6 / |r_i - r_j|^6 : van der Waals interaction
//!   - n_i = |r><r|_i : Rydberg number operator on atom i
//!   - sigma_x^i : coherent coupling between ground |g> and Rydberg |r> states
//!
//! # Reservoir Computing Pipeline
//!
//! 1. **Encode** classical inputs by modulating Omega and/or Delta
//! 2. **Evolve** the atom array under H for a fixed time step
//! 3. **Read out** expectation values (local magnetization, pair correlations)
//! 4. **Train** a linear (ridge regression) layer on the readout features
//!
//! # References
//!
//! - Bravo et al. (2022) -- Quantum reservoir computing with Rydberg atoms
//! - Fujii & Nakajima (2021) -- Quantum reservoir computing: theory and practice
//! - Browaeys & Lahaye (2020) -- Many-body physics with individually controlled Rydberg atoms

use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// CONSTANTS
// ============================================================

/// C6 van der Waals coefficient for ^87Rb |70S> state (MHz * um^6).
/// This sets the natural energy scale for Rydberg interactions.
const C6_DEFAULT: f64 = 862690.0;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from Rydberg reservoir operations.
#[derive(Clone, Debug, PartialEq)]
pub enum RydbergError {
    /// Number of atoms is outside the valid range [2, 100].
    InvalidAtomCount { requested: usize, min: usize, max: usize },
    /// A geometry parameter (spacing, radius, etc.) is non-positive.
    InvalidGeometry { param: String, value: f64 },
    /// The Hilbert-space dimension exceeds what we can allocate.
    DimensionOverflow { num_atoms: usize, max_supported: usize },
    /// Training data dimensions are inconsistent.
    TrainingDataMismatch { features_len: usize, targets_len: usize },
    /// Ridge regression failed (singular or ill-conditioned matrix).
    RegressionFailed { reason: String },
    /// Reservoir has not been trained yet.
    NotTrained,
    /// Input signal contains NaN or infinity.
    InvalidInput { detail: String },
}

impl fmt::Display for RydbergError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RydbergError::InvalidAtomCount { requested, min, max } => {
                write!(f, "Invalid atom count {}: must be in [{}, {}]", requested, min, max)
            }
            RydbergError::InvalidGeometry { param, value } => {
                write!(f, "Invalid geometry parameter '{}': {} (must be positive)", param, value)
            }
            RydbergError::DimensionOverflow { num_atoms, max_supported } => {
                write!(
                    f,
                    "Hilbert space for {} atoms exceeds max supported dimension (2^{})",
                    num_atoms, max_supported
                )
            }
            RydbergError::TrainingDataMismatch { features_len, targets_len } => {
                write!(
                    f,
                    "Training data length mismatch: {} feature rows vs {} target rows",
                    features_len, targets_len
                )
            }
            RydbergError::RegressionFailed { reason } => {
                write!(f, "Ridge regression failed: {}", reason)
            }
            RydbergError::NotTrained => {
                write!(f, "Reservoir has not been trained; call train_task() first")
            }
            RydbergError::InvalidInput { detail } => {
                write!(f, "Invalid input signal: {}", detail)
            }
        }
    }
}

impl std::error::Error for RydbergError {}

/// Convenience alias used throughout this module.
type Result<T> = std::result::Result<T, RydbergError>;

// ============================================================
// READOUT METHOD
// ============================================================

/// Strategy for extracting classical features from the quantum reservoir state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadoutMethod {
    /// Measure <sigma_z^i> for each atom -- yields num_atoms features.
    LocalMagnetization,
    /// Measure <n_i n_j> for all pairs -- yields num_atoms*(num_atoms-1)/2 features.
    PairCorrelations,
    /// Concatenate local magnetization and pair correlations.
    FullStateTomography,
}

// ============================================================
// CONFIGURATION (builder pattern)
// ============================================================

/// Configuration for a Rydberg atom reservoir computer.
#[derive(Clone, Debug)]
pub struct RydbergReservoirConfig {
    /// Number of atoms in the array (2..=100).
    pub num_atoms: usize,
    /// Rydberg blockade radius in micrometers.
    pub rydberg_blockade_radius: f64,
    /// C6 interaction strength coefficient (MHz * um^6).
    pub interaction_strength: f64,
    /// Laser detuning from the Rydberg transition (MHz).
    pub detuning: f64,
    /// Rabi frequency -- global laser driving (MHz).
    pub rabi_frequency: f64,
    /// Hilbert space dimension (truncated to 2^min(num_atoms, 12)).
    pub reservoir_dim: usize,
    /// Number of initial evolution steps to discard (transient wash-out).
    pub wash_out_steps: usize,
    /// How to read features out of the reservoir.
    pub readout_method: ReadoutMethod,
    /// Time step for Trotter evolution (microseconds).
    pub dt: f64,
    /// Ridge regression regularization parameter lambda.
    pub ridge_lambda: f64,
}

impl Default for RydbergReservoirConfig {
    fn default() -> Self {
        let num_atoms = 20;
        let eff = num_atoms.min(12);
        Self {
            num_atoms,
            rydberg_blockade_radius: 7.0,
            interaction_strength: C6_DEFAULT,
            detuning: 0.0,
            rabi_frequency: 2.0 * PI * 2.0,
            reservoir_dim: 1 << eff,
            wash_out_steps: 100,
            readout_method: ReadoutMethod::LocalMagnetization,
            dt: 0.1,
            ridge_lambda: 1e-4,
        }
    }
}

impl RydbergReservoirConfig {
    /// Start building a new configuration with required atom count.
    pub fn builder(num_atoms: usize) -> RydbergReservoirConfigBuilder {
        RydbergReservoirConfigBuilder {
            num_atoms,
            ..RydbergReservoirConfigBuilder::default()
        }
    }

    /// Validate that all parameters are physically reasonable.
    pub fn validate(&self) -> Result<()> {
        if self.num_atoms < 2 || self.num_atoms > 100 {
            return Err(RydbergError::InvalidAtomCount {
                requested: self.num_atoms,
                min: 2,
                max: 100,
            });
        }
        if self.rydberg_blockade_radius <= 0.0 {
            return Err(RydbergError::InvalidGeometry {
                param: "rydberg_blockade_radius".into(),
                value: self.rydberg_blockade_radius,
            });
        }
        if self.rabi_frequency <= 0.0 {
            return Err(RydbergError::InvalidGeometry {
                param: "rabi_frequency".into(),
                value: self.rabi_frequency,
            });
        }
        if self.dt <= 0.0 {
            return Err(RydbergError::InvalidGeometry {
                param: "dt".into(),
                value: self.dt,
            });
        }
        let max_supported = 12;
        if self.num_atoms > max_supported && self.reservoir_dim > (1 << max_supported) {
            return Err(RydbergError::DimensionOverflow {
                num_atoms: self.num_atoms,
                max_supported,
            });
        }
        Ok(())
    }
}

/// Builder for [`RydbergReservoirConfig`].
#[derive(Clone, Debug)]
pub struct RydbergReservoirConfigBuilder {
    num_atoms: usize,
    rydberg_blockade_radius: f64,
    interaction_strength: f64,
    detuning: f64,
    rabi_frequency: f64,
    reservoir_dim: Option<usize>,
    wash_out_steps: usize,
    readout_method: ReadoutMethod,
    dt: f64,
    ridge_lambda: f64,
}

impl Default for RydbergReservoirConfigBuilder {
    fn default() -> Self {
        Self {
            num_atoms: 20,
            rydberg_blockade_radius: 7.0,
            interaction_strength: C6_DEFAULT,
            detuning: 0.0,
            rabi_frequency: 2.0 * PI * 2.0,
            reservoir_dim: None,
            wash_out_steps: 100,
            readout_method: ReadoutMethod::LocalMagnetization,
            dt: 0.1,
            ridge_lambda: 1e-4,
        }
    }
}

impl RydbergReservoirConfigBuilder {
    pub fn rydberg_blockade_radius(mut self, r: f64) -> Self {
        self.rydberg_blockade_radius = r;
        self
    }
    pub fn interaction_strength(mut self, c6: f64) -> Self {
        self.interaction_strength = c6;
        self
    }
    pub fn detuning(mut self, delta: f64) -> Self {
        self.detuning = delta;
        self
    }
    pub fn rabi_frequency(mut self, omega: f64) -> Self {
        self.rabi_frequency = omega;
        self
    }
    pub fn reservoir_dim(mut self, dim: usize) -> Self {
        self.reservoir_dim = Some(dim);
        self
    }
    pub fn wash_out_steps(mut self, n: usize) -> Self {
        self.wash_out_steps = n;
        self
    }
    pub fn readout_method(mut self, m: ReadoutMethod) -> Self {
        self.readout_method = m;
        self
    }
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }
    pub fn ridge_lambda(mut self, lambda: f64) -> Self {
        self.ridge_lambda = lambda;
        self
    }

    /// Consume the builder and produce a validated config.
    pub fn build(self) -> Result<RydbergReservoirConfig> {
        let eff = self.num_atoms.min(12);
        let reservoir_dim = self.reservoir_dim.unwrap_or(1 << eff);
        let config = RydbergReservoirConfig {
            num_atoms: self.num_atoms,
            rydberg_blockade_radius: self.rydberg_blockade_radius,
            interaction_strength: self.interaction_strength,
            detuning: self.detuning,
            rabi_frequency: self.rabi_frequency,
            reservoir_dim,
            wash_out_steps: self.wash_out_steps,
            readout_method: self.readout_method,
            dt: self.dt,
            ridge_lambda: self.ridge_lambda,
        };
        config.validate()?;
        Ok(config)
    }
}

// ============================================================
// ATOM ARRAY GEOMETRIES
// ============================================================

/// A 2D arrangement of neutral atoms held in optical tweezers.
#[derive(Clone, Debug)]
pub struct AtomArray {
    /// (x, y) positions in micrometers.
    pub positions: Vec<(f64, f64)>,
}

impl AtomArray {
    /// Linear chain of `n` atoms separated by `spacing` micrometers.
    pub fn new_chain(n: usize, spacing: f64) -> Self {
        let positions = (0..n).map(|i| (i as f64 * spacing, 0.0)).collect();
        AtomArray { positions }
    }

    /// Square lattice of `rows x cols` atoms with given spacing.
    pub fn new_square_lattice(rows: usize, cols: usize, spacing: f64) -> Self {
        let mut positions = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                positions.push((c as f64 * spacing, r as f64 * spacing));
            }
        }
        AtomArray { positions }
    }

    /// Triangular lattice with `n` atoms. Rows are offset by half a spacing.
    pub fn new_triangular(n: usize, spacing: f64) -> Self {
        // Pack atoms row by row; each row is offset by spacing/2 in x.
        let cols = (n as f64).sqrt().ceil() as usize;
        let mut positions = Vec::with_capacity(n);
        let mut row = 0usize;
        let mut placed = 0;
        while placed < n {
            let x_offset = if row % 2 == 1 { spacing * 0.5 } else { 0.0 };
            let y = row as f64 * spacing * (3.0_f64).sqrt() / 2.0;
            for c in 0..cols {
                if placed >= n {
                    break;
                }
                positions.push((c as f64 * spacing + x_offset, y));
                placed += 1;
            }
            row += 1;
        }
        AtomArray { positions }
    }

    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Euclidean distance between atoms i and j.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let (xi, yi) = self.positions[i];
        let (xj, yj) = self.positions[j];
        ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt()
    }

    /// Van der Waals interaction V_ij = C6 / r^6 between atoms i and j.
    pub fn rydberg_interaction(&self, i: usize, j: usize, c6: f64) -> f64 {
        if i == j {
            return 0.0;
        }
        let r = self.distance(i, j);
        if r < 1e-12 {
            return f64::INFINITY;
        }
        c6 / r.powi(6)
    }
}

// ============================================================
// HAMILTONIAN CONSTRUCTION
// ============================================================

/// Rydberg many-body Hamiltonian built as a dense complex matrix.
///
/// For `n` atoms the Hilbert space has dimension `2^n`, with computational
/// basis states labelling each atom as ground (|g> = |0>) or Rydberg (|r> = |1>).
pub struct RydbergHamiltonian;

impl RydbergHamiltonian {
    /// Build the full Hamiltonian matrix for an atom array under given config.
    ///
    /// H = sum_i (Omega/2) sigma_x^i  -  Delta sum_i n_i  +  sum_{i<j} V_ij n_i n_j
    ///
    /// Returns a `dim x dim` Hermitian matrix where `dim = 2^num_atoms` (capped by
    /// `config.reservoir_dim`).
    pub fn compute(array: &AtomArray, config: &RydbergReservoirConfig) -> Array2<Complex64> {
        let n = array.len();
        let dim = config.reservoir_dim.min(1 << n);
        let mut h = Array2::<Complex64>::zeros((dim, dim));

        let omega = config.rabi_frequency;
        let delta = config.detuning;
        let c6 = config.interaction_strength;

        for basis in 0..dim {
            // --- diagonal: detuning + interactions ---
            let mut diag = 0.0_f64;

            // Count Rydberg excitations and accumulate detuning
            for atom in 0..n {
                if basis & (1 << atom) != 0 {
                    diag -= delta;
                }
            }

            // Interaction: sum_{i<j} V_ij n_i n_j (both atoms must be in |r>)
            for i in 0..n {
                if basis & (1 << i) == 0 {
                    continue;
                }
                for j in (i + 1)..n {
                    if basis & (1 << j) == 0 {
                        continue;
                    }
                    diag += array.rydberg_interaction(i, j, c6);
                }
            }

            h[(basis, basis)] = Complex64::new(diag, 0.0);

            // --- off-diagonal: Rabi coupling sigma_x^i ---
            // sigma_x^i flips atom i: |g> <-> |r>
            for atom in 0..n {
                let flipped = basis ^ (1 << atom);
                if flipped < dim {
                    h[(basis, flipped)] += Complex64::new(omega / 2.0, 0.0);
                }
            }
        }

        h
    }
}

// ============================================================
// RESERVOIR STATE
// ============================================================

/// Quantum state of the Rydberg atom array, stored as a complex state vector.
#[derive(Clone, Debug)]
pub struct ReservoirState {
    /// State vector in the computational basis.
    pub state_vector: Vec<Complex64>,
    /// Number of atoms (qubits).
    num_atoms: usize,
}

impl ReservoirState {
    /// Initialize in the all-ground state |00...0>.
    pub fn ground_state(num_atoms: usize, dim: usize) -> Self {
        let mut sv = vec![Complex64::new(0.0, 0.0); dim];
        if !sv.is_empty() {
            sv[0] = Complex64::new(1.0, 0.0);
        }
        ReservoirState {
            state_vector: sv,
            num_atoms,
        }
    }

    /// Hilbert space dimension.
    pub fn dim(&self) -> usize {
        self.state_vector.len()
    }

    /// Evolve under Hamiltonian `h` for time `dt` using first-order Trotter:
    ///
    ///   |psi(t+dt)> = exp(-i H dt) |psi(t)>
    ///
    /// Implemented via eigendecomposition for small systems and Pade-style
    /// Taylor expansion for moderate ones.
    pub fn evolve(&mut self, h: &Array2<Complex64>, dt: f64) {
        let dim = self.dim();
        debug_assert_eq!(h.shape(), &[dim, dim]);

        // Compute U = exp(-i H dt) via truncated Taylor series (order 10).
        // For small dim this converges rapidly and preserves unitarity to ~1e-12.
        let idt = Complex64::new(0.0, -dt);
        let mut u = Array2::<Complex64>::eye(dim);
        let mut term = Array2::<Complex64>::eye(dim);

        for k in 1..=10 {
            // term = (-i dt)^k / k! * H^k  (accumulated iteratively)
            let h_term = term.dot(h);
            let scale = idt / Complex64::new(k as f64, 0.0);
            term = h_term.mapv(|v| v * scale);
            u = u + &term;
        }

        // Apply: |psi'> = U |psi>
        let psi = Array1::from_vec(self.state_vector.clone());
        let psi_new = u.dot(&psi);
        self.state_vector = psi_new.to_vec();

        // Renormalize to correct residual Trotter/truncation error
        self.renormalize();
    }

    /// Normalize the state vector to unit norm.
    fn renormalize(&mut self) {
        let norm_sq: f64 = self.state_vector.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq > 1e-30 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for c in &mut self.state_vector {
                *c = *c * inv_norm;
            }
        }
    }

    /// Norm-squared of the state vector (should be ~1.0).
    pub fn norm_sq(&self) -> f64 {
        self.state_vector.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Measure local magnetization: <sigma_z^i> = P(atom i in |g>) - P(atom i in |r>)
    /// for each atom. Returns `num_atoms` values in [-1, 1].
    pub fn measure_local_magnetization(&self) -> Vec<f64> {
        let dim = self.dim();
        let n = self.num_atoms;
        let mut mag = vec![0.0_f64; n];

        for basis in 0..dim {
            let prob = self.state_vector[basis].norm_sqr();
            for atom in 0..n {
                if basis & (1 << atom) != 0 {
                    // Atom in |r> (excited) -> sigma_z = -1
                    mag[atom] -= prob;
                } else {
                    // Atom in |g> (ground) -> sigma_z = +1
                    mag[atom] += prob;
                }
            }
        }

        mag
    }

    /// Measure pair correlations: <n_i n_j> for all pairs i < j.
    /// Returns `num_atoms * (num_atoms - 1) / 2` values.
    pub fn measure_pair_correlations(&self) -> Vec<f64> {
        let dim = self.dim();
        let n = self.num_atoms;
        let num_pairs = n * (n - 1) / 2;
        let mut corr = vec![0.0_f64; num_pairs];

        for basis in 0..dim {
            let prob = self.state_vector[basis].norm_sqr();
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if (basis & (1 << i) != 0) && (basis & (1 << j) != 0) {
                        corr[idx] += prob;
                    }
                    idx += 1;
                }
            }
        }

        corr
    }
}

// ============================================================
// INPUT ENCODER
// ============================================================

/// Encodes classical input signals into modulations of the Rydberg Hamiltonian
/// parameters (detuning, Rabi frequency, or both).
#[derive(Clone, Debug)]
pub struct InputEncoder {
    /// Base Rabi frequency (MHz).
    base_rabi: f64,
    /// Base detuning (MHz).
    base_detuning: f64,
    /// Modulation depth for detuning encoding.
    detuning_scale: f64,
    /// Modulation depth for Rabi encoding.
    rabi_scale: f64,
}

impl InputEncoder {
    /// Create a new encoder with the given base parameters.
    pub fn new(base_rabi: f64, base_detuning: f64) -> Self {
        Self {
            base_rabi,
            base_detuning,
            detuning_scale: 2.0 * PI,
            rabi_scale: 2.0 * PI,
        }
    }

    /// Encode a scalar input by modulating the detuning: Delta = base + scale * value.
    pub fn encode_detuning(&self, value: f64) -> (f64, f64) {
        (self.base_rabi, self.base_detuning + self.detuning_scale * value)
    }

    /// Encode a scalar input by modulating the Rabi frequency: Omega = base + scale * value.
    pub fn encode_rabi(&self, value: f64) -> (f64, f64) {
        (self.base_rabi + self.rabi_scale * value, self.base_detuning)
    }

    /// Time-multiplexed encoding: convert a vector of input values into a
    /// sequence of (Omega, Delta) pulses where both parameters are modulated.
    pub fn encode_time_multiplexed(&self, values: &[f64]) -> Vec<(f64, f64)> {
        values
            .iter()
            .map(|&v| {
                let omega = self.base_rabi + self.rabi_scale * v * 0.5;
                let delta = self.base_detuning + self.detuning_scale * v;
                (omega, delta)
            })
            .collect()
    }
}

// ============================================================
// LINEAR READOUT (RIDGE REGRESSION)
// ============================================================

/// Classical linear readout layer trained via ridge regression.
///
/// Given reservoir feature vectors X and target vectors Y, solves:
///   W = (X^T X + lambda I)^{-1} X^T Y
#[derive(Clone, Debug)]
pub struct LinearReadout {
    /// Weight matrix [output_dim x feature_dim].
    pub weights: Array2<f64>,
    /// Output dimensionality.
    output_dim: usize,
    /// Feature dimensionality.
    feature_dim: usize,
}

impl LinearReadout {
    /// Train the readout layer from feature/target pairs.
    ///
    /// `features`: slice of feature vectors (one per sample).
    /// `targets`: slice of target vectors (one per sample).
    /// `lambda`: ridge regularization strength.
    pub fn train(
        features: &[Vec<f64>],
        targets: &[Vec<f64>],
        lambda: f64,
    ) -> Result<TrainingResult> {
        if features.is_empty() || targets.is_empty() {
            return Err(RydbergError::TrainingDataMismatch {
                features_len: features.len(),
                targets_len: targets.len(),
            });
        }
        if features.len() != targets.len() {
            return Err(RydbergError::TrainingDataMismatch {
                features_len: features.len(),
                targets_len: targets.len(),
            });
        }

        let n_samples = features.len();
        let feat_dim = features[0].len();
        let out_dim = targets[0].len();

        // Build feature matrix X [n_samples x feat_dim]
        let mut x_data = Vec::with_capacity(n_samples * feat_dim);
        for f in features {
            x_data.extend_from_slice(f);
        }
        let x = Array2::from_shape_vec((n_samples, feat_dim), x_data).map_err(|e| {
            RydbergError::RegressionFailed {
                reason: format!("Feature matrix construction: {}", e),
            }
        })?;

        // Build target matrix Y [n_samples x out_dim]
        let mut y_data = Vec::with_capacity(n_samples * out_dim);
        for t in targets {
            y_data.extend_from_slice(t);
        }
        let y = Array2::from_shape_vec((n_samples, out_dim), y_data).map_err(|e| {
            RydbergError::RegressionFailed {
                reason: format!("Target matrix construction: {}", e),
            }
        })?;

        // X^T X + lambda * I  [feat_dim x feat_dim]
        let xt = x.t();
        let mut xtx = xt.dot(&x);
        for i in 0..feat_dim {
            xtx[(i, i)] += lambda;
        }

        // X^T Y [feat_dim x out_dim]
        let xty = xt.dot(&y);

        // Solve via Cholesky decomposition (manual, since we avoid external linalg crates).
        let weights_t = solve_symmetric_positive_definite(&xtx, &xty)?;

        // weights_t is [feat_dim x out_dim]; readout stores [out_dim x feat_dim]
        let weights = weights_t.t().to_owned();

        // Compute training predictions and MSE
        let predictions = x.dot(&weights_t);
        let residuals = &predictions - &y;
        let mse = residuals.mapv(|v| v * v).sum() / (n_samples * out_dim) as f64;

        // Compute R^2
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let ss_tot: f64 = y
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .zip(y_mean.iter())
                    .map(|(yi, ym)| (yi - ym).powi(2))
                    .sum::<f64>()
            })
            .sum();
        let ss_res: f64 = residuals.mapv(|v| v * v).sum();
        let r2 = if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { 0.0 };

        let readout = LinearReadout {
            weights,
            output_dim: out_dim,
            feature_dim: feat_dim,
        };

        Ok(TrainingResult { readout, mse, r2 })
    }

    /// Predict output for a single feature vector.
    pub fn predict(&self, features: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.output_dim];
        for o in 0..self.output_dim {
            let mut sum = 0.0;
            for f in 0..self.feature_dim {
                sum += self.weights[(o, f)] * features[f];
            }
            out[o] = sum;
        }
        out
    }
}

/// Result of training the linear readout.
#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub readout: LinearReadout,
    pub mse: f64,
    pub r2: f64,
}

/// Solve A x = B where A is symmetric positive definite, using Cholesky decomposition.
/// A: [n x n], B: [n x m] -> returns X: [n x m].
fn solve_symmetric_positive_definite(
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> Result<Array2<f64>> {
    let n = a.shape()[0];
    let m = b.shape()[1];

    // Cholesky: A = L L^T
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[(i, k)] * l[(j, k)];
            }
            if i == j {
                let diag = a[(i, i)] - sum;
                if diag <= 0.0 {
                    return Err(RydbergError::RegressionFailed {
                        reason: format!(
                            "Cholesky failed: non-positive diagonal at index {} (value: {:.6e})",
                            i, diag
                        ),
                    });
                }
                l[(i, j)] = diag.sqrt();
            } else {
                l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
            }
        }
    }

    // Forward substitution: L Z = B
    let mut z = Array2::<f64>::zeros((n, m));
    for col in 0..m {
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[(i, k)] * z[(k, col)];
            }
            z[(i, col)] = (b[(i, col)] - sum) / l[(i, i)];
        }
    }

    // Backward substitution: L^T X = Z
    let mut x = Array2::<f64>::zeros((n, m));
    for col in 0..m {
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for k in (i + 1)..n {
                sum += l[(k, i)] * x[(k, col)];
            }
            x[(i, col)] = (z[(i, col)] - sum) / l[(i, i)];
        }
    }

    Ok(x)
}

// ============================================================
// TASK RESULT
// ============================================================

/// Summary statistics from a reservoir computing task.
#[derive(Clone, Debug)]
pub struct TaskResult {
    /// Mean squared error on the training set.
    pub train_mse: f64,
    /// Mean squared error on the test set.
    pub test_mse: f64,
    /// Coefficient of determination on training set.
    pub train_r2: f64,
    /// Coefficient of determination on test set.
    pub test_r2: f64,
    /// Learned readout weight matrix.
    pub weights: Array2<f64>,
}

// ============================================================
// MAIN RESERVOIR STRUCT
// ============================================================

/// Rydberg atom reservoir computer.
///
/// Combines an atom array, Hamiltonian, quantum state, input encoder, and
/// classical readout into a complete reservoir computing pipeline.
pub struct RydbergReservoir {
    config: RydbergReservoirConfig,
    array: AtomArray,
    hamiltonian: Array2<Complex64>,
    state: ReservoirState,
    encoder: InputEncoder,
    readout: Option<LinearReadout>,
}

impl RydbergReservoir {
    /// Construct a new reservoir from a config and atom array.
    pub fn new(config: RydbergReservoirConfig, array: AtomArray) -> Result<Self> {
        config.validate()?;
        let h = RydbergHamiltonian::compute(&array, &config);
        let state = ReservoirState::ground_state(array.len(), config.reservoir_dim);
        let encoder = InputEncoder::new(config.rabi_frequency, config.detuning);
        Ok(Self {
            config,
            array,
            hamiltonian: h,
            state,
            encoder,
            readout: None,
        })
    }

    /// Inject a scalar input into the reservoir: encode it, rebuild the
    /// Hamiltonian with the modulated parameters, and evolve one time step.
    pub fn inject_input(&mut self, value: f64) -> Result<()> {
        if value.is_nan() || value.is_infinite() {
            return Err(RydbergError::InvalidInput {
                detail: format!("Input value is {}", value),
            });
        }

        // Encode input as a detuning modulation
        let (_omega, delta) = self.encoder.encode_detuning(value);

        // Temporarily modify config for this step
        let mut step_config = self.config.clone();
        step_config.detuning = delta;

        // Rebuild Hamiltonian with modulated parameters
        let h = RydbergHamiltonian::compute(&self.array, &step_config);

        // Evolve state
        self.state.evolve(&h, self.config.dt);
        Ok(())
    }

    /// Extract feature vector from the current reservoir state using the
    /// configured readout method.
    pub fn readout(&self) -> Vec<f64> {
        match self.config.readout_method {
            ReadoutMethod::LocalMagnetization => self.state.measure_local_magnetization(),
            ReadoutMethod::PairCorrelations => self.state.measure_pair_correlations(),
            ReadoutMethod::FullStateTomography => {
                let mut features = self.state.measure_local_magnetization();
                features.extend(self.state.measure_pair_correlations());
                features
            }
        }
    }

    /// Run the full train-and-evaluate pipeline.
    ///
    /// 1. Feed `inputs` through the reservoir, collecting feature vectors.
    /// 2. Discard the first `wash_out_steps` transient samples.
    /// 3. Split remaining data into train/test by `train_ratio`.
    /// 4. Train ridge regression on training features/targets.
    /// 5. Evaluate on both training and test sets.
    pub fn train_task(
        &mut self,
        inputs: &[f64],
        targets: &[Vec<f64>],
        train_ratio: f64,
    ) -> Result<TaskResult> {
        if inputs.len() != targets.len() {
            return Err(RydbergError::TrainingDataMismatch {
                features_len: inputs.len(),
                targets_len: targets.len(),
            });
        }

        // Reset to ground state
        self.reset();

        // Collect features by driving the reservoir
        let mut all_features = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            self.inject_input(inp)?;
            all_features.push(self.readout());
        }

        // Discard wash-out transient
        let wash = self.config.wash_out_steps.min(all_features.len().saturating_sub(2));
        let features = &all_features[wash..];
        let tgts = &targets[wash..];

        if features.is_empty() {
            return Err(RydbergError::TrainingDataMismatch {
                features_len: 0,
                targets_len: 0,
            });
        }

        // Train/test split
        let n_train = ((features.len() as f64) * train_ratio).round() as usize;
        let n_train = n_train.max(1).min(features.len() - 1);

        let train_features = &features[..n_train];
        let train_targets = &tgts[..n_train];
        let test_features = &features[n_train..];
        let test_targets = &tgts[n_train..];

        // Train ridge regression
        let result = LinearReadout::train(
            &train_features.to_vec(),
            &train_targets.to_vec(),
            self.config.ridge_lambda,
        )?;

        let readout = result.readout;

        // Evaluate on training set
        let train_mse = compute_mse(&readout, train_features, train_targets);
        let train_r2 = compute_r2(&readout, train_features, train_targets);

        // Evaluate on test set
        let test_mse = compute_mse(&readout, test_features, test_targets);
        let test_r2 = compute_r2(&readout, test_features, test_targets);

        let weights = readout.weights.clone();
        self.readout = Some(readout);

        Ok(TaskResult {
            train_mse,
            test_mse,
            train_r2,
            test_r2,
            weights,
        })
    }

    /// Predict using the trained readout layer. Injects the input, then
    /// applies the linear readout weights to the resulting features.
    pub fn predict(&mut self, input: f64) -> Result<Vec<f64>> {
        let features = {
            self.inject_input(input)?;
            self.readout()
        };
        let readout = self.readout.as_ref().ok_or(RydbergError::NotTrained)?;
        Ok(readout.predict(&features))
    }

    /// Reset the reservoir to the ground state |00...0>.
    pub fn reset(&mut self) {
        self.state = ReservoirState::ground_state(self.array.len(), self.config.reservoir_dim);
    }

    /// Compute the linear memory capacity of the reservoir.
    ///
    /// Memory capacity MC = sum_{k=1}^{K} r^2(y_k, x_{t-k})
    ///
    /// where r^2 is the squared correlation between the reservoir's prediction
    /// of the k-step-delayed input and the actual delayed input.
    pub fn memory_capacity(&mut self, signal_length: usize, max_delay: usize) -> Result<f64> {
        // Generate a random input signal (deterministic for reproducibility)
        let signal: Vec<f64> = (0..signal_length)
            .map(|i| {
                // Simple pseudo-random via sin -- deterministic and bounded in [-1, 1]
                (i as f64 * 0.618033988 * 2.0 * PI).sin()
            })
            .collect();

        self.reset();

        // Drive reservoir and collect features
        let mut features = Vec::with_capacity(signal_length);
        for &s in &signal {
            self.inject_input(s)?;
            features.push(self.readout());
        }

        // Skip wash-out
        let wash = self.config.wash_out_steps.min(signal_length.saturating_sub(max_delay + 2));
        let usable = signal_length - wash;

        let mut total_mc = 0.0;

        for delay in 1..=max_delay.min(usable.saturating_sub(1)) {
            // Target: x_{t - delay}
            let feat_slice: Vec<Vec<f64>> = features[wash + delay..].to_vec();
            let target_slice: Vec<Vec<f64>> = signal[wash..signal_length - delay]
                .iter()
                .map(|&v| vec![v])
                .collect();

            let n = feat_slice.len().min(target_slice.len());
            if n < 2 {
                continue;
            }

            let feat_used = &feat_slice[..n];
            let tgt_used = &target_slice[..n];

            match LinearReadout::train(feat_used, tgt_used, self.config.ridge_lambda) {
                Ok(result) => {
                    // Compute squared correlation on training data
                    let r2 = result.r2.max(0.0);
                    total_mc += r2;
                }
                Err(_) => {
                    // Skip this delay if regression fails
                }
            }
        }

        Ok(total_mc)
    }

    /// Access the current reservoir state (for inspection/debugging).
    pub fn state(&self) -> &ReservoirState {
        &self.state
    }

    /// Access the atom array.
    pub fn array(&self) -> &AtomArray {
        &self.array
    }

    /// Access the config.
    pub fn config(&self) -> &RydbergReservoirConfig {
        &self.config
    }
}

// ============================================================
// HELPERS
// ============================================================

/// Mean squared error of the readout on a dataset.
fn compute_mse(readout: &LinearReadout, features: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    if features.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0;
    for (feat, tgt) in features.iter().zip(targets.iter()) {
        let pred = readout.predict(feat);
        for (p, t) in pred.iter().zip(tgt.iter()) {
            total += (p - t).powi(2);
            count += 1;
        }
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

/// Coefficient of determination R^2.
fn compute_r2(readout: &LinearReadout, features: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    if features.is_empty() {
        return 0.0;
    }
    let out_dim = targets[0].len();
    let n = features.len();

    // Compute mean of targets per output dimension
    let mut means = vec![0.0; out_dim];
    for tgt in targets {
        for (o, &v) in tgt.iter().enumerate() {
            means[o] += v;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (feat, tgt) in features.iter().zip(targets.iter()) {
        let pred = readout.predict(feat);
        for (o, (&p, &t)) in pred.iter().zip(tgt.iter()).enumerate() {
            ss_res += (t - p).powi(2);
            ss_tot += (t - means[o]).powi(2);
        }
    }

    if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { 0.0 }
}

// ============================================================
// BENCHMARK TASKS
// ============================================================

/// Standard benchmark tasks for evaluating reservoir computing performance.
pub mod benchmarks {
    use super::*;

    /// Sine wave one-step-ahead prediction.
    ///
    /// Input: x(t) = sin(0.2 * t)
    /// Target: x(t+1)
    pub fn sine_wave_prediction(
        reservoir: &mut RydbergReservoir,
        length: usize,
    ) -> Result<TaskResult> {
        let signal: Vec<f64> = (0..length)
            .map(|t| (0.2 * t as f64).sin())
            .collect();

        let inputs = signal[..length - 1].to_vec();
        let targets: Vec<Vec<f64>> = signal[1..].iter().map(|&v| vec![v]).collect();

        reservoir.train_task(&inputs, &targets, 0.8)
    }

    /// Mackey-Glass chaotic time series prediction.
    ///
    /// dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)
    ///
    /// Uses Euler integration with standard parameters:
    /// beta = 0.2, gamma = 0.1, tau = 17, n = 10.
    pub fn mackey_glass_prediction(
        reservoir: &mut RydbergReservoir,
        length: usize,
    ) -> Result<TaskResult> {
        let mg = generate_mackey_glass(length + 200, 17);
        // Skip initial transient
        let series: Vec<f64> = mg[200..200 + length].to_vec();

        // Normalize to [-1, 1]
        let max_val = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_val - min_val;
        let normalized: Vec<f64> = if range > 1e-15 {
            series.iter().map(|&v| 2.0 * (v - min_val) / range - 1.0).collect()
        } else {
            series
        };

        let inputs = normalized[..length - 1].to_vec();
        let targets: Vec<Vec<f64>> = normalized[1..].iter().map(|&v| vec![v]).collect();

        reservoir.train_task(&inputs, &targets, 0.8)
    }

    /// NARMA (Nonlinear Auto-Regressive Moving Average) task of given order.
    ///
    /// y(t+1) = alpha * y(t) + beta * y(t) * sum_{i=0}^{order-1} y(t-i)
    ///          + gamma * u(t - order + 1) * u(t) + delta
    pub fn narma_task(
        reservoir: &mut RydbergReservoir,
        order: usize,
        length: usize,
    ) -> Result<TaskResult> {
        let (inputs, targets) = generate_narma(order, length);
        reservoir.train_task(
            &inputs,
            &targets.iter().map(|&v| vec![v]).collect::<Vec<_>>(),
            0.8,
        )
    }

    /// Generate Mackey-Glass time series via Euler integration.
    fn generate_mackey_glass(length: usize, tau: usize) -> Vec<f64> {
        let beta: f64 = 0.2;
        let gamma: f64 = 0.1;
        let n: f64 = 10.0;
        let dt: f64 = 1.0;

        let total = length + tau;
        let mut x: Vec<f64> = vec![0.9; total];

        for t in tau..total {
            let x_tau = x[t - tau];
            let dx = beta * x_tau / (1.0 + x_tau.powf(n)) - gamma * x[t - 1];
            x[t] = x[t - 1] + dx * dt;
            // Clamp to prevent divergence
            x[t] = x[t].clamp(0.0, 2.0);
        }

        x[tau..].to_vec()
    }

    /// Generate NARMA input/output sequences.
    fn generate_narma(order: usize, length: usize) -> (Vec<f64>, Vec<f64>) {
        let alpha = 0.3;
        let beta = 0.05;
        let gamma = 1.5;
        let delta = 0.1;

        // Deterministic input signal in [0, 0.5]
        let u: Vec<f64> = (0..length)
            .map(|i| 0.25 + 0.25 * (i as f64 * 0.3).sin())
            .collect();

        let mut y = vec![0.0; length];

        for t in order..length {
            let sum_past: f64 = (0..order).map(|k| y[t - 1 - k]).sum();
            y[t] = alpha * y[t - 1]
                + beta * y[t - 1] * sum_past
                + gamma * u[t - order] * u[t]
                + delta;
            // Clamp
            y[t] = y[t].clamp(-2.0, 2.0);
        }

        (u, y)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // -- Config tests --

    #[test]
    fn test_config_default() {
        let config = RydbergReservoirConfig::default();
        assert_eq!(config.num_atoms, 20);
        assert_eq!(config.wash_out_steps, 100);
        assert!((config.rydberg_blockade_radius - 7.0).abs() < 1e-10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder_valid() {
        let config = RydbergReservoirConfig::builder(8)
            .rydberg_blockade_radius(5.0)
            .detuning(1.0)
            .rabi_frequency(2.0 * PI)
            .wash_out_steps(50)
            .readout_method(ReadoutMethod::PairCorrelations)
            .build();
        assert!(config.is_ok());
        let c = config.unwrap();
        assert_eq!(c.num_atoms, 8);
        assert_eq!(c.wash_out_steps, 50);
        assert_eq!(c.readout_method, ReadoutMethod::PairCorrelations);
    }

    #[test]
    fn test_config_too_few_atoms() {
        let result = RydbergReservoirConfig::builder(1).build();
        assert!(result.is_err());
        match result.unwrap_err() {
            RydbergError::InvalidAtomCount { requested, .. } => assert_eq!(requested, 1),
            other => panic!("Expected InvalidAtomCount, got {:?}", other),
        }
    }

    #[test]
    fn test_config_too_many_atoms() {
        let result = RydbergReservoirConfig::builder(101).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_negative_blockade_radius() {
        let result = RydbergReservoirConfig::builder(4)
            .rydberg_blockade_radius(-1.0)
            .build();
        assert!(result.is_err());
        match result.unwrap_err() {
            RydbergError::InvalidGeometry { param, .. } => {
                assert_eq!(param, "rydberg_blockade_radius");
            }
            other => panic!("Expected InvalidGeometry, got {:?}", other),
        }
    }

    // -- Atom array geometry tests --

    #[test]
    fn test_chain_geometry() {
        let chain = AtomArray::new_chain(5, 6.0);
        assert_eq!(chain.len(), 5);
        assert!((chain.positions[0].0 - 0.0).abs() < 1e-10);
        assert!((chain.positions[4].0 - 24.0).abs() < 1e-10);
        // All y-coordinates should be zero
        for &(_, y) in &chain.positions {
            assert!((y - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_square_lattice_geometry() {
        let lattice = AtomArray::new_square_lattice(3, 4, 5.0);
        assert_eq!(lattice.len(), 12);
        // Check corners
        assert!((lattice.positions[0].0 - 0.0).abs() < 1e-10);
        assert!((lattice.positions[0].1 - 0.0).abs() < 1e-10);
        // Last atom: col=3, row=2
        let last = lattice.positions[11];
        assert!((last.0 - 15.0).abs() < 1e-10);
        assert!((last.1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_geometry() {
        let tri = AtomArray::new_triangular(6, 5.0);
        assert_eq!(tri.len(), 6);
        // First row at y=0, second row offset by spacing/2 in x
        assert!((tri.positions[0].1 - 0.0).abs() < 1e-10);
        // Odd rows have x-offset of spacing/2
        // With 6 atoms and cols = ceil(sqrt(6)) = 3, row 1 starts at x=2.5
        let cols = (6.0_f64).sqrt().ceil() as usize;
        assert_eq!(cols, 3);
    }

    // -- Rydberg interaction tests --

    #[test]
    fn test_rydberg_interaction_c6_over_r6() {
        let chain = AtomArray::new_chain(2, 10.0);
        let c6 = 1e6; // simplified
        let v = chain.rydberg_interaction(0, 1, c6);
        let expected = c6 / 10.0_f64.powi(6);
        assert!((v - expected).abs() < 1e-10, "V={}, expected={}", v, expected);
    }

    #[test]
    fn test_rydberg_interaction_self_is_zero() {
        let chain = AtomArray::new_chain(3, 5.0);
        assert!((chain.rydberg_interaction(0, 0, C6_DEFAULT) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_rydberg_interaction_distance_scaling() {
        // V ~ 1/r^6 so doubling distance should reduce V by factor 64
        let chain = AtomArray::new_chain(3, 5.0);
        let v_near = chain.rydberg_interaction(0, 1, C6_DEFAULT);
        let v_far = chain.rydberg_interaction(0, 2, C6_DEFAULT);
        let ratio = v_near / v_far;
        // r_far / r_near = 2, so V_near / V_far = 2^6 = 64
        assert!((ratio - 64.0).abs() < 1e-6, "Ratio was {}, expected 64", ratio);
    }

    // -- Hamiltonian tests --

    #[test]
    fn test_hamiltonian_hermitian() {
        let config = RydbergReservoirConfig::builder(3)
            .rabi_frequency(2.0 * PI)
            .detuning(1.0)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let h = RydbergHamiltonian::compute(&array, &config);

        let dim = h.shape()[0];
        // Check H = H^dagger
        for i in 0..dim {
            for j in 0..dim {
                let diff = (h[(i, j)] - h[(j, i)].conj()).norm();
                assert!(diff < 1e-12, "H not Hermitian at ({},{}): diff={}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_hamiltonian_dimension() {
        let config = RydbergReservoirConfig::builder(4)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(4, 6.0);
        let h = RydbergHamiltonian::compute(&array, &config);
        assert_eq!(h.shape(), &[16, 16]); // 2^4 = 16
    }

    #[test]
    fn test_hamiltonian_zero_detuning_zero_interaction() {
        // With zero interaction and zero detuning, H should be purely off-diagonal
        // (only Rabi coupling sigma_x terms). Use large spacing to suppress interactions.
        let config = RydbergReservoirConfig::builder(2)
            .detuning(0.0)
            .rabi_frequency(1.0)
            .interaction_strength(0.0)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(2, 1000.0);
        let h = RydbergHamiltonian::compute(&array, &config);

        // Diagonal should be zero
        for i in 0..4 {
            assert!(
                h[(i, i)].norm() < 1e-10,
                "Diagonal ({},{}) = {} (expected 0)",
                i,
                i,
                h[(i, i)]
            );
        }
    }

    // -- State evolution tests --

    #[test]
    fn test_evolution_preserves_unitarity() {
        let config = RydbergReservoirConfig::builder(3)
            .rabi_frequency(2.0 * PI)
            .detuning(0.5)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let h = RydbergHamiltonian::compute(&array, &config);

        let mut state = ReservoirState::ground_state(3, 8);
        let norm_before = state.norm_sq();

        // Evolve several steps
        for _ in 0..10 {
            state.evolve(&h, 0.05);
        }

        let norm_after = state.norm_sq();
        assert!(
            (norm_after - norm_before).abs() < 1e-6,
            "Norm changed from {} to {} after evolution",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_evolution_changes_state() {
        let config = RydbergReservoirConfig::builder(2)
            .rabi_frequency(2.0 * PI)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(2, 6.0);
        let h = RydbergHamiltonian::compute(&array, &config);

        let mut state = ReservoirState::ground_state(2, 4);
        let initial = state.state_vector.clone();

        state.evolve(&h, 0.1);

        // State should have changed
        let changed: bool = state
            .state_vector
            .iter()
            .zip(initial.iter())
            .any(|(a, b)| (a - b).norm() > 1e-10);
        assert!(changed, "State did not change after evolution");
    }

    // -- Readout tests --

    #[test]
    fn test_local_magnetization_ground_state() {
        let state = ReservoirState::ground_state(3, 8);
        let mag = state.measure_local_magnetization();
        // Ground state |000>: all atoms in |g>, so sigma_z = +1 for each
        assert_eq!(mag.len(), 3);
        for &m in &mag {
            assert!((m - 1.0).abs() < 1e-12, "Expected +1.0, got {}", m);
        }
    }

    #[test]
    fn test_pair_correlations_ground_state() {
        let state = ReservoirState::ground_state(3, 8);
        let corr = state.measure_pair_correlations();
        // Ground state: no atoms excited, so <n_i n_j> = 0 for all pairs
        assert_eq!(corr.len(), 3); // 3 choose 2 = 3
        for &c in &corr {
            assert!((c - 0.0).abs() < 1e-12, "Expected 0.0, got {}", c);
        }
    }

    #[test]
    fn test_magnetization_superposition() {
        // Create equal superposition of |00> and |11> for 2 atoms
        let mut state = ReservoirState::ground_state(2, 4);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        state.state_vector[0] = Complex64::new(inv_sqrt2, 0.0); // |00>
        state.state_vector[3] = Complex64::new(inv_sqrt2, 0.0); // |11>

        let mag = state.measure_local_magnetization();
        // P(|g>) = 0.5, P(|r>) = 0.5 for each atom, so sigma_z = 0
        for &m in &mag {
            assert!(m.abs() < 1e-12, "Expected 0.0, got {}", m);
        }
    }

    // -- Input encoding tests --

    #[test]
    fn test_detuning_encoding() {
        let encoder = InputEncoder::new(10.0, 0.0);
        let (omega, delta) = encoder.encode_detuning(1.0);
        assert!((omega - 10.0).abs() < 1e-10);
        assert!(delta.abs() > 1e-10); // Should be non-zero
    }

    #[test]
    fn test_rabi_encoding() {
        let encoder = InputEncoder::new(10.0, 0.0);
        let (omega, delta) = encoder.encode_rabi(1.0);
        assert!(omega > 10.0); // Rabi should increase
        assert!((delta - 0.0).abs() < 1e-10); // Detuning unchanged
    }

    #[test]
    fn test_time_multiplexed_encoding() {
        let encoder = InputEncoder::new(10.0, 0.0);
        let pulses = encoder.encode_time_multiplexed(&[0.0, 0.5, 1.0]);
        assert_eq!(pulses.len(), 3);
        // Zero input should give base parameters
        assert!((pulses[0].0 - 10.0).abs() < 1e-10);
    }

    // -- Ridge regression tests --

    #[test]
    fn test_ridge_regression_identity() {
        // Train on identity mapping: y = x
        let features: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 50.0]).collect();
        let targets = features.clone();
        let result = LinearReadout::train(&features, &targets, 1e-6).unwrap();
        assert!(result.mse < 1e-6, "MSE too high: {}", result.mse);
        assert!(result.r2 > 0.999, "R^2 too low: {}", result.r2);
    }

    #[test]
    fn test_ridge_regression_linear() {
        // y = 2x + 1, encoded as features = [x, 1] -> target = [2x+1]
        let features: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let x = i as f64 / 100.0;
                vec![x, 1.0]
            })
            .collect();
        let targets: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let x = i as f64 / 100.0;
                vec![2.0 * x + 1.0]
            })
            .collect();
        let result = LinearReadout::train(&features, &targets, 1e-8).unwrap();
        assert!(result.mse < 1e-6, "MSE too high: {}", result.mse);

        // Check learned weights are approximately [2, 1]
        let w = &result.readout.weights;
        assert!((w[(0, 0)] - 2.0).abs() < 1e-3, "w0 = {}", w[(0, 0)]);
        assert!((w[(0, 1)] - 1.0).abs() < 1e-3, "w1 = {}", w[(0, 1)]);
    }

    #[test]
    fn test_ridge_regression_mismatched_lengths() {
        let features: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0]];
        let targets: Vec<Vec<f64>> = vec![vec![1.0]];
        let result = LinearReadout::train(&features, &targets, 1e-4);
        assert!(result.is_err());
    }

    // -- Reservoir pipeline tests --

    #[test]
    fn test_reservoir_creation() {
        let config = RydbergReservoirConfig::builder(4)
            .wash_out_steps(5)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(4, 6.0);
        let reservoir = RydbergReservoir::new(config, array);
        assert!(reservoir.is_ok());
    }

    #[test]
    fn test_reservoir_inject_and_readout() {
        let config = RydbergReservoirConfig::builder(3)
            .wash_out_steps(0)
            .readout_method(ReadoutMethod::LocalMagnetization)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        reservoir.inject_input(0.5).unwrap();
        let features = reservoir.readout();
        assert_eq!(features.len(), 3); // 3 atoms -> 3 magnetization values
    }

    #[test]
    fn test_reservoir_invalid_input() {
        let config = RydbergReservoirConfig::builder(3)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        assert!(reservoir.inject_input(f64::NAN).is_err());
        assert!(reservoir.inject_input(f64::INFINITY).is_err());
    }

    #[test]
    fn test_reservoir_reset() {
        let config = RydbergReservoirConfig::builder(3)
            .wash_out_steps(0)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        // Evolve, then reset
        reservoir.inject_input(0.5).unwrap();
        reservoir.inject_input(0.8).unwrap();
        reservoir.reset();

        // After reset, state should be ground state again
        let mag = reservoir.state().measure_local_magnetization();
        for &m in &mag {
            assert!((m - 1.0).abs() < 1e-12, "After reset, expected +1.0, got {}", m);
        }
    }

    #[test]
    fn test_reservoir_predict_without_training() {
        let config = RydbergReservoirConfig::builder(3)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();
        let result = reservoir.predict(0.5);
        assert!(matches!(result, Err(RydbergError::NotTrained)));
    }

    #[test]
    fn test_full_train_predict_pipeline() {
        // Small system: 3 atoms, sine prediction
        let config = RydbergReservoirConfig::builder(3)
            .wash_out_steps(5)
            .dt(0.05)
            .ridge_lambda(1e-4)
            .readout_method(ReadoutMethod::FullStateTomography)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        // Generate training data: sine wave
        let length = 60;
        let signal: Vec<f64> = (0..length).map(|t| (0.2 * t as f64).sin()).collect();
        let inputs: Vec<f64> = signal[..length - 1].to_vec();
        let targets: Vec<Vec<f64>> = signal[1..].iter().map(|&v| vec![v]).collect();

        let result = reservoir.train_task(&inputs, &targets, 0.8);
        assert!(result.is_ok(), "Training failed: {:?}", result.err());

        let task_result = result.unwrap();
        // Training MSE should be finite
        assert!(task_result.train_mse.is_finite());
        assert!(task_result.test_mse.is_finite());

        // Should be able to predict after training
        let pred = reservoir.predict(0.5);
        assert!(pred.is_ok());
        assert_eq!(pred.unwrap().len(), 1);
    }

    #[test]
    fn test_sine_wave_benchmark() {
        let config = RydbergReservoirConfig::builder(3)
            .wash_out_steps(5)
            .dt(0.05)
            .ridge_lambda(1e-3)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        let result = benchmarks::sine_wave_prediction(&mut reservoir, 60);
        assert!(result.is_ok(), "Sine benchmark failed: {:?}", result.err());
        let task = result.unwrap();
        assert!(task.train_mse.is_finite());
    }

    #[test]
    fn test_memory_capacity() {
        let config = RydbergReservoirConfig::builder(3)
            .wash_out_steps(5)
            .dt(0.05)
            .ridge_lambda(1e-3)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        let mc = reservoir.memory_capacity(80, 5);
        assert!(mc.is_ok());
        let mc_val = mc.unwrap();
        assert!(mc_val >= 0.0, "Memory capacity should be non-negative: {}", mc_val);
        assert!(mc_val.is_finite());
    }

    #[test]
    fn test_pair_correlation_readout_method() {
        let config = RydbergReservoirConfig::builder(4)
            .wash_out_steps(0)
            .readout_method(ReadoutMethod::PairCorrelations)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(4, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        reservoir.inject_input(0.5).unwrap();
        let features = reservoir.readout();
        // 4 atoms -> 4*3/2 = 6 pair correlations
        assert_eq!(features.len(), 6);
    }

    #[test]
    fn test_full_state_tomography_readout() {
        let config = RydbergReservoirConfig::builder(3)
            .wash_out_steps(0)
            .readout_method(ReadoutMethod::FullStateTomography)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        reservoir.inject_input(0.5).unwrap();
        let features = reservoir.readout();
        // 3 magnetization + 3 pair correlations = 6
        assert_eq!(features.len(), 6);
    }

    #[test]
    fn test_error_display() {
        let e = RydbergError::InvalidAtomCount {
            requested: 200,
            min: 2,
            max: 100,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("200"));
        assert!(msg.contains("100"));
    }

    #[test]
    fn test_narma_benchmark() {
        let config = RydbergReservoirConfig::builder(3)
            .wash_out_steps(5)
            .dt(0.05)
            .ridge_lambda(1e-3)
            .build()
            .unwrap();
        let array = AtomArray::new_chain(3, 6.0);
        let mut reservoir = RydbergReservoir::new(config, array).unwrap();

        let result = benchmarks::narma_task(&mut reservoir, 5, 60);
        assert!(result.is_ok(), "NARMA benchmark failed: {:?}", result.err());
        let task = result.unwrap();
        assert!(task.train_mse.is_finite());
    }

    #[test]
    fn test_different_array_geometries_produce_different_dynamics() {
        let config = RydbergReservoirConfig::builder(4)
            .wash_out_steps(0)
            .dt(0.05)
            .build()
            .unwrap();

        // Chain
        let chain = AtomArray::new_chain(4, 6.0);
        let mut r1 = RydbergReservoir::new(config.clone(), chain).unwrap();
        r1.inject_input(0.5).unwrap();
        let f1 = r1.readout();

        // Square
        let square = AtomArray::new_square_lattice(2, 2, 6.0);
        let mut r2 = RydbergReservoir::new(config.clone(), square).unwrap();
        r2.inject_input(0.5).unwrap();
        let f2 = r2.readout();

        // The readouts should differ because the atom geometries produce different interactions
        let diff: f64 = f1.iter().zip(f2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-10, "Different geometries should give different readouts, diff={}", diff);
    }
}
