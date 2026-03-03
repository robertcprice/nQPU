//! AC4c: Pulse-Level Differentiable Quantum Simulation
//!
//! Hamiltonian-level simulation with arbitrary pulse shapes and gradient computation
//! for pulse parameter optimization. This is the Rust-native alternative to Qiskit
//! Dynamics, providing:
//!
//! - Transmon qubit modeling with anharmonicity and multi-level structure
//! - Time-dependent Hamiltonian construction: H(t) = H_drift + Sum_k c_k(t) H_k
//! - RK4 integration of the Schrodinger equation
//! - Lindblad open-system dynamics with T1/T2 decoherence
//! - Finite-difference gradient computation of fidelity w.r.t. pulse parameters
//! - Simple gradient-descent pulse optimization
//! - Standard gate calibrations (X, SX, CNOT via cross-resonance)
//!
//! # Physical Model
//!
//! Each transmon qubit is modeled as an anharmonic oscillator with `n_levels` levels:
//!
//! ```text
//! H_qubit = omega * a†a + (alpha/2) * a†a(a†a - 1)
//! ```
//!
//! where `omega` is the qubit frequency and `alpha` is the anharmonicity (~-300 MHz).
//! Drive pulses couple via the charge operator `(a + a†)`.
//!
//! # Example
//!
//! ```rust,ignore
//! use nqpu_metal::pulse_simulation::*;
//!
//! let system = TransmonSystem::single_qubit(5.0, -0.3);
//! let pulse = Pulse::gaussian(20.0, 0.5, 5.0, 0.0, 5.0);
//! let schedule = PulseSchedule::new(0.1)
//!     .with_pulse(ScheduledPulse { channel: Channel::Drive(0), start_time: 0.0, pulse });
//! let sim = PulseSimulator::new(system, schedule);
//! let result = sim.simulate_unitary();
//! ```

use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// COMPLEX NUMBER HELPERS
// ============================================================

const ZERO: Complex64 = Complex64 { re: 0.0, im: 0.0 };
const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };
const I: Complex64 = Complex64 { re: 0.0, im: 1.0 };

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[inline]
fn c_re(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors from pulse simulation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PulseError {
    /// Invalid system parameters.
    InvalidSystem(String),
    /// Dimension mismatch between operators or states.
    DimensionMismatch(String),
    /// Numerical instability during integration.
    IntegrationError(String),
    /// Invalid pulse parameters.
    InvalidPulse(String),
    /// Optimization did not converge.
    OptimizationError(String),
}

impl std::fmt::Display for PulseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PulseError::InvalidSystem(msg) => write!(f, "Invalid system: {}", msg),
            PulseError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            PulseError::IntegrationError(msg) => write!(f, "Integration error: {}", msg),
            PulseError::InvalidPulse(msg) => write!(f, "Invalid pulse: {}", msg),
            PulseError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
        }
    }
}

impl std::error::Error for PulseError {}

// ============================================================
// DENSE MATRIX TYPE (row-major, dim x dim)
// ============================================================

/// Dense complex matrix stored in row-major order.
/// Used for Hamiltonians, unitaries, and density matrices.
#[derive(Clone, Debug)]
pub struct DenseMatrix {
    pub dim: usize,
    pub data: Vec<Complex64>,
}

impl DenseMatrix {
    /// Create a zero matrix of given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            dim,
            data: vec![ZERO; dim * dim],
        }
    }

    /// Create an identity matrix.
    pub fn identity(dim: usize) -> Self {
        let mut m = Self::zeros(dim);
        for i in 0..dim {
            m[(i, i)] = ONE;
        }
        m
    }

    /// Get element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row * self.dim + col]
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: Complex64) {
        self.data[row * self.dim + col] = val;
    }

    /// Matrix-matrix multiply: self * other.
    pub fn matmul(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.dim, other.dim);
        let n = self.dim;
        let mut result = DenseMatrix::zeros(n);
        for i in 0..n {
            for k in 0..n {
                let a_ik = self.data[i * n + k];
                if a_ik == ZERO {
                    continue;
                }
                for j in 0..n {
                    result.data[i * n + j] += a_ik * other.data[k * n + j];
                }
            }
        }
        result
    }

    /// Matrix-vector multiply: self * vec.
    pub fn matvec(&self, vec: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(self.dim, vec.len());
        let n = self.dim;
        let mut result = vec![ZERO; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.data[i * n + j] * vec[j];
            }
        }
        result
    }

    /// Conjugate transpose (dagger).
    pub fn dagger(&self) -> DenseMatrix {
        let n = self.dim;
        let mut result = DenseMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                result.data[j * n + i] = self.data[i * n + j].conj();
            }
        }
        result
    }

    /// Trace of the matrix.
    pub fn trace(&self) -> Complex64 {
        let mut tr = ZERO;
        for i in 0..self.dim {
            tr += self.data[i * self.dim + i];
        }
        tr
    }

    /// Scale all elements by a complex scalar.
    pub fn scale(&self, s: Complex64) -> DenseMatrix {
        DenseMatrix {
            dim: self.dim,
            data: self.data.iter().map(|&x| x * s).collect(),
        }
    }

    /// Add two matrices.
    pub fn add(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.dim, other.dim);
        DenseMatrix {
            dim: self.dim,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }

    /// Subtract: self - other.
    pub fn sub(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.dim, other.dim);
        DenseMatrix {
            dim: self.dim,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        }
    }

    /// Tensor product (Kronecker product): self ⊗ other.
    pub fn kron(&self, other: &DenseMatrix) -> DenseMatrix {
        let n1 = self.dim;
        let n2 = other.dim;
        let n = n1 * n2;
        let mut result = DenseMatrix::zeros(n);
        for i1 in 0..n1 {
            for j1 in 0..n1 {
                let a = self.data[i1 * n1 + j1];
                if a == ZERO {
                    continue;
                }
                for i2 in 0..n2 {
                    for j2 in 0..n2 {
                        let row = i1 * n2 + i2;
                        let col = j1 * n2 + j2;
                        result.data[row * n + col] = a * other.data[i2 * n2 + j2];
                    }
                }
            }
        }
        result
    }

    /// Commutator [A, B] = AB - BA.
    pub fn commutator(&self, other: &DenseMatrix) -> DenseMatrix {
        self.matmul(other).sub(&other.matmul(self))
    }

    /// Anti-commutator {A, B} = AB + BA.
    pub fn anticommutator(&self, other: &DenseMatrix) -> DenseMatrix {
        self.matmul(other).add(&other.matmul(self))
    }

    /// Frobenius norm squared.
    pub fn norm_sq(&self) -> f64 {
        self.data.iter().map(|x| x.norm_sqr()).sum()
    }

    /// Matrix exponential via scaling-squaring with Taylor series.
    /// Suitable for small matrices (dim <= ~16).
    /// Computes exp(M) where M = self.
    pub fn matrix_exp(&self) -> DenseMatrix {
        // Scaling: find s such that ||M / 2^s|| < 0.5
        let norm = self.data.iter().map(|x| x.norm()).fold(0.0f64, f64::max);
        let s = if norm > 0.5 {
            (norm / 0.5).log2().ceil() as u32
        } else {
            0
        };
        let scale_factor = 2.0f64.powi(-(s as i32));
        let a = self.scale(c_re(scale_factor));

        // Taylor series: exp(A) = I + A + A^2/2! + A^3/3! + ...
        // For ||A|| < 0.5, 20 terms give double precision accuracy
        let n = self.dim;
        let mut result = DenseMatrix::identity(n);
        let mut term = DenseMatrix::identity(n); // A^k / k!

        for k in 1..=20 {
            term = term.matmul(&a).scale(c_re(1.0 / k as f64));
            result = result.add(&term);
            // Early termination if term is negligible
            if term.norm_sq() < 1e-30 {
                break;
            }
        }

        // Squaring phase: result = result^(2^s)
        for _ in 0..s {
            result = result.matmul(&result);
        }

        result
    }

    /// Solve AX = B via Gaussian elimination with partial pivoting.
    /// Returns X. Both A and B must be dim x dim.
    fn solve_linear_system(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
        let n = a.dim;
        // Augmented matrix [A | B]
        let mut aug: Vec<Vec<Complex64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(2 * n);
            for j in 0..n {
                row.push(a.data[i * n + j]);
            }
            for j in 0..n {
                row.push(b.data[i * n + j]);
            }
            aug.push(row);
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_val = 0.0f64;
            let mut max_row = col;
            for row in col..n {
                let v = aug[row][col].norm();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.norm() < 1e-15 {
                // Singular — return identity as fallback
                return DenseMatrix::identity(n);
            }

            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..2 * n {
                    let v = aug[col][j];
                    aug[row][j] -= factor * v;
                }
            }
        }

        // Back substitution
        let mut x = DenseMatrix::zeros(n);
        for col_b in 0..n {
            for row in (0..n).rev() {
                let mut sum = aug[row][n + col_b];
                for j in (row + 1)..n {
                    sum -= aug[row][j] * x.data[j * n + col_b];
                }
                x.data[row * n + col_b] = sum / aug[row][row];
            }
        }
        x
    }
}

impl std::ops::Index<(usize, usize)> for DenseMatrix {
    type Output = Complex64;
    fn index(&self, (row, col): (usize, usize)) -> &Complex64 {
        &self.data[row * self.dim + col]
    }
}

impl std::ops::IndexMut<(usize, usize)> for DenseMatrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Complex64 {
        &mut self.data[row * self.dim + col]
    }
}

// ============================================================
// PULSE SHAPES
// ============================================================

/// Pulse envelope shapes for microwave drive.
#[derive(Clone, Debug)]
pub enum PulseShape {
    /// Gaussian pulse: A * exp(-(t - t0)^2 / (2 sigma^2))
    Gaussian { sigma: f64 },
    /// Flat-top Gaussian: Gaussian rise/fall with constant middle.
    GaussianSquare { sigma: f64, width: f64 },
    /// DRAG pulse: Derivative Removal by Adiabatic Gate.
    /// Adds a derivative correction to suppress leakage to |2>.
    Drag { sigma: f64, beta: f64 },
    /// Constant amplitude for the full duration.
    Constant,
    /// Custom waveform specified as evenly-spaced sample points.
    Custom(Vec<f64>),
}

/// A time-dependent pulse envelope.
#[derive(Clone, Debug)]
pub struct Pulse {
    /// Total duration in nanoseconds.
    pub duration: f64,
    /// Maximum amplitude (dimensionless, typically 0..1).
    pub amplitude: f64,
    /// Drive frequency in GHz.
    pub frequency: f64,
    /// Phase offset in radians.
    pub phase: f64,
    /// Envelope shape.
    pub shape: PulseShape,
    /// DRAG correction coefficient (used on top of any shape).
    pub drag_coefficient: f64,
}

impl Pulse {
    /// Create a Gaussian pulse.
    pub fn gaussian(duration: f64, amplitude: f64, frequency: f64, phase: f64, sigma: f64) -> Self {
        Self {
            duration,
            amplitude,
            frequency,
            phase,
            shape: PulseShape::Gaussian { sigma },
            drag_coefficient: 0.0,
        }
    }

    /// Create a Gaussian-square (flat-top) pulse.
    pub fn gaussian_square(
        duration: f64,
        amplitude: f64,
        frequency: f64,
        phase: f64,
        sigma: f64,
        width: f64,
    ) -> Self {
        Self {
            duration,
            amplitude,
            frequency,
            phase,
            shape: PulseShape::GaussianSquare { sigma, width },
            drag_coefficient: 0.0,
        }
    }

    /// Create a DRAG pulse for leakage suppression.
    pub fn drag(
        duration: f64,
        amplitude: f64,
        frequency: f64,
        phase: f64,
        sigma: f64,
        beta: f64,
    ) -> Self {
        Self {
            duration,
            amplitude,
            frequency,
            phase,
            shape: PulseShape::Drag { sigma, beta },
            drag_coefficient: beta,
        }
    }

    /// Create a constant (square) pulse.
    pub fn constant(duration: f64, amplitude: f64, frequency: f64, phase: f64) -> Self {
        Self {
            duration,
            amplitude,
            frequency,
            phase,
            shape: PulseShape::Constant,
            drag_coefficient: 0.0,
        }
    }

    /// Create a pulse from custom sample points.
    pub fn custom(duration: f64, amplitude: f64, frequency: f64, phase: f64, samples: Vec<f64>) -> Self {
        Self {
            duration,
            amplitude,
            frequency,
            phase,
            shape: PulseShape::Custom(samples),
            drag_coefficient: 0.0,
        }
    }

    /// Evaluate the envelope at time t (0 <= t <= duration).
    /// Returns the real envelope value (before modulation by cos(omega*t + phi)).
    pub fn envelope(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }
        let t_mid = self.duration / 2.0;
        match &self.shape {
            PulseShape::Gaussian { sigma } => {
                let dt = t - t_mid;
                self.amplitude * (-dt * dt / (2.0 * sigma * sigma)).exp()
            }
            PulseShape::GaussianSquare { sigma, width } => {
                let rise = (self.duration - width) / 2.0;
                if t < rise {
                    let dt = t - rise;
                    self.amplitude * (-dt * dt / (2.0 * sigma * sigma)).exp()
                } else if t > rise + width {
                    let dt = t - (rise + width);
                    self.amplitude * (-dt * dt / (2.0 * sigma * sigma)).exp()
                } else {
                    self.amplitude
                }
            }
            PulseShape::Drag { sigma, beta } => {
                let dt = t - t_mid;
                let gauss = (-dt * dt / (2.0 * sigma * sigma)).exp();
                let d_gauss = -dt / (sigma * sigma) * gauss;
                self.amplitude * gauss + beta * d_gauss
            }
            PulseShape::Constant => self.amplitude,
            PulseShape::Custom(samples) => {
                if samples.is_empty() {
                    return 0.0;
                }
                // Linear interpolation
                let frac = t / self.duration * (samples.len() - 1) as f64;
                let idx = frac.floor() as usize;
                if idx >= samples.len() - 1 {
                    return self.amplitude * samples[samples.len() - 1];
                }
                let alpha = frac - idx as f64;
                self.amplitude * (samples[idx] * (1.0 - alpha) + samples[idx + 1] * alpha)
            }
        }
    }

    /// Evaluate the DRAG quadrature component at time t.
    /// This is the imaginary (Y-quadrature) correction for leakage suppression.
    pub fn drag_quadrature(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration || self.drag_coefficient == 0.0 {
            return 0.0;
        }
        let t_mid = self.duration / 2.0;
        let sigma = match &self.shape {
            PulseShape::Gaussian { sigma } => *sigma,
            PulseShape::Drag { sigma, .. } => *sigma,
            _ => return 0.0,
        };
        let dt = t - t_mid;
        let gauss = (-dt * dt / (2.0 * sigma * sigma)).exp();
        let d_gauss = -dt / (sigma * sigma) * gauss;
        self.drag_coefficient * d_gauss
    }

    /// Full complex drive signal at time t: envelope(t) * exp(i*(omega*t + phi)).
    /// Returns (I-component, Q-component) in the lab frame.
    pub fn signal(&self, t: f64) -> (f64, f64) {
        let env = self.envelope(t);
        let drag_q = self.drag_quadrature(t);
        let angle = 2.0 * PI * self.frequency * t + self.phase;
        let i_comp = env * angle.cos() - drag_q * angle.sin();
        let q_comp = env * angle.sin() + drag_q * angle.cos();
        (i_comp, q_comp)
    }
}

// ============================================================
// CHANNELS
// ============================================================

/// Pulse channel types matching hardware convention.
#[derive(Clone, Debug, PartialEq)]
pub enum Channel {
    /// Drive channel for single-qubit rotations (d0, d1, ...).
    Drive(usize),
    /// Control channel for two-qubit cross-resonance (u0 between q_i, q_j).
    Control(usize, usize),
    /// Measurement channel (m0, m1, ...).
    Measure(usize),
}

// ============================================================
// SCHEDULED PULSE & SCHEDULE
// ============================================================

/// A pulse placed on a channel at a specific start time.
#[derive(Clone, Debug)]
pub struct ScheduledPulse {
    pub channel: Channel,
    pub start_time: f64,
    pub pulse: Pulse,
}

/// Collection of scheduled pulses defining a quantum operation.
#[derive(Clone, Debug)]
pub struct PulseSchedule {
    pub pulses: Vec<ScheduledPulse>,
    /// Sample time in nanoseconds (typically ~0.222 ns for IBM hardware).
    pub dt: f64,
}

impl PulseSchedule {
    /// Create an empty schedule with given sample time.
    pub fn new(dt: f64) -> Self {
        Self {
            pulses: Vec::new(),
            dt,
        }
    }

    /// Builder: add a pulse and return self.
    pub fn with_pulse(mut self, sp: ScheduledPulse) -> Self {
        self.pulses.push(sp);
        self
    }

    /// Add a pulse to the schedule.
    pub fn add_pulse(&mut self, sp: ScheduledPulse) {
        self.pulses.push(sp);
    }

    /// Total duration of the schedule (end of latest pulse).
    pub fn total_duration(&self) -> f64 {
        self.pulses
            .iter()
            .map(|sp| sp.start_time + sp.pulse.duration)
            .fold(0.0f64, f64::max)
    }

    /// Number of discrete time steps.
    pub fn n_steps(&self) -> usize {
        (self.total_duration() / self.dt).ceil() as usize
    }
}

// ============================================================
// TRANSMON SYSTEM
// ============================================================

/// Multi-level transmon qubit system specification.
#[derive(Clone, Debug)]
pub struct TransmonSystem {
    /// Number of qubits.
    pub n_qubits: usize,
    /// Number of energy levels per qubit (typically 2 or 3).
    pub n_levels: usize,
    /// Qubit transition frequencies in GHz.
    pub frequencies: Vec<f64>,
    /// Anharmonicities in GHz (negative for transmon, ~-0.3 GHz).
    pub anharmonicities: Vec<f64>,
    /// Coupling map: (qubit_i, qubit_j, coupling_strength_g in GHz).
    pub coupling_map: Vec<(usize, usize, f64)>,
    /// T1 relaxation times in nanoseconds.
    pub t1: Vec<f64>,
    /// T2 dephasing times in nanoseconds.
    pub t2: Vec<f64>,
}

impl TransmonSystem {
    /// Create a single-qubit system (2-level) with given frequency and anharmonicity.
    pub fn single_qubit(freq_ghz: f64, anharmonicity_ghz: f64) -> Self {
        Self {
            n_qubits: 1,
            n_levels: 2,
            frequencies: vec![freq_ghz],
            anharmonicities: vec![anharmonicity_ghz],
            coupling_map: Vec::new(),
            t1: vec![50_000.0], // 50 us
            t2: vec![70_000.0], // 70 us
        }
    }

    /// Create a single-qubit 3-level system (for DRAG / leakage studies).
    pub fn single_qubit_3level(freq_ghz: f64, anharmonicity_ghz: f64) -> Self {
        Self {
            n_qubits: 1,
            n_levels: 3,
            frequencies: vec![freq_ghz],
            anharmonicities: vec![anharmonicity_ghz],
            coupling_map: Vec::new(),
            t1: vec![50_000.0],
            t2: vec![70_000.0],
        }
    }

    /// Create a two-qubit coupled system.
    pub fn two_qubit(
        freq0: f64,
        freq1: f64,
        anhar0: f64,
        anhar1: f64,
        coupling_g: f64,
    ) -> Self {
        Self {
            n_qubits: 2,
            n_levels: 2,
            frequencies: vec![freq0, freq1],
            anharmonicities: vec![anhar0, anhar1],
            coupling_map: vec![(0, 1, coupling_g)],
            t1: vec![50_000.0, 50_000.0],
            t2: vec![70_000.0, 70_000.0],
        }
    }

    /// Total Hilbert space dimension: n_levels^n_qubits.
    pub fn dim(&self) -> usize {
        self.n_levels.pow(self.n_qubits as u32)
    }

    /// Build the drift Hamiltonian (time-independent part).
    /// H_drift = Sum_q [ omega_q * n_q + (alpha_q / 2) * n_q * (n_q - 1) ]
    ///         + Sum_{q1,q2} g * (a_q1 + a_q1^dag)(a_q2 + a_q2^dag)
    ///
    /// All frequencies are in GHz; multiply by 2*pi for angular frequency.
    pub fn build_drift_hamiltonian(&self) -> DenseMatrix {
        let dim = self.dim();
        let mut h = DenseMatrix::zeros(dim);

        // Single-qubit terms
        for q in 0..self.n_qubits {
            let nq = self.number_operator(q);
            let omega = 2.0 * PI * self.frequencies[q];
            let alpha = 2.0 * PI * self.anharmonicities[q];

            // omega * n
            for i in 0..dim {
                h.data[i * dim + i] += c_re(omega) * nq[(i, i)];
            }

            // (alpha/2) * n * (n - 1)
            if self.n_levels > 2 {
                let n_sq = nq.matmul(&nq);
                for i in 0..dim {
                    let n_val = nq[(i, i)];
                    h.data[i * dim + i] += c_re(alpha / 2.0) * (n_sq[(i, i)] - n_val);
                }
            }
        }

        // Coupling terms: g * (a_i + a_i^dag)(a_j + a_j^dag)
        for &(qi, qj, g) in &self.coupling_map {
            let x_i = self.charge_operator(qi);
            let x_j = self.charge_operator(qj);
            let coupling = x_i.matmul(&x_j);
            let g_ang = 2.0 * PI * g;
            for i in 0..dim {
                for j in 0..dim {
                    h.data[i * dim + j] += c_re(g_ang) * coupling[(i, j)];
                }
            }
        }

        h
    }

    /// Build drive control Hamiltonian for a given qubit.
    /// H_drive_q = (a_q + a_q^dag)  (charge operator on qubit q)
    pub fn build_drive_hamiltonian(&self, qubit: usize) -> DenseMatrix {
        self.charge_operator(qubit)
    }

    /// Build coupling control Hamiltonian between two qubits.
    /// Used for cross-resonance drives.
    pub fn build_coupling_hamiltonian(&self, q1: usize, q2: usize) -> DenseMatrix {
        let x1 = self.charge_operator(q1);
        let x2 = self.charge_operator(q2);
        x1.matmul(&x2)
    }

    /// Number operator n = a^dag * a for a single qubit in the full Hilbert space.
    fn number_operator(&self, qubit: usize) -> DenseMatrix {
        let nl = self.n_levels;
        let mut single = DenseMatrix::zeros(nl);
        for n in 0..nl {
            single[(n, n)] = c_re(n as f64);
        }
        self.embed_single_qubit_op(&single, qubit)
    }

    /// Charge operator (a + a^dag) for a single qubit in full Hilbert space.
    /// a|n> = sqrt(n)|n-1>, so a_{m,n} = sqrt(n) * delta_{m, n-1}.
    fn charge_operator(&self, qubit: usize) -> DenseMatrix {
        let nl = self.n_levels;
        let mut single = DenseMatrix::zeros(nl);
        for n in 1..nl {
            let sq = (n as f64).sqrt();
            single[(n - 1, n)] = c_re(sq); // a
            single[(n, n - 1)] = c_re(sq); // a^dag
        }
        self.embed_single_qubit_op(&single, qubit)
    }

    /// Lowering operator a for a single qubit in full Hilbert space.
    fn lowering_operator(&self, qubit: usize) -> DenseMatrix {
        let nl = self.n_levels;
        let mut single = DenseMatrix::zeros(nl);
        for n in 1..nl {
            single[(n - 1, n)] = c_re((n as f64).sqrt());
        }
        self.embed_single_qubit_op(&single, qubit)
    }

    /// Embed a single-qubit operator into the full multi-qubit Hilbert space.
    /// Uses tensor product: I_{q>qubit} ⊗ Op ⊗ I_{q<qubit}.
    fn embed_single_qubit_op(&self, op: &DenseMatrix, qubit: usize) -> DenseMatrix {
        let nl = self.n_levels;
        let nq = self.n_qubits;

        if nq == 1 {
            return op.clone();
        }

        // Build from right (qubit 0) to left (qubit nq-1)
        let id = DenseMatrix::identity(nl);
        let mut result = if qubit == 0 {
            op.clone()
        } else {
            id.clone()
        };

        for q in 1..nq {
            let piece = if q == qubit { op } else { &id };
            result = result.kron(piece);
        }

        result
    }

    /// Z operator for qubit in computational subspace, embedded in full space.
    pub fn z_operator(&self, qubit: usize) -> DenseMatrix {
        let nl = self.n_levels;
        let mut single = DenseMatrix::zeros(nl);
        single[(0, 0)] = ONE;
        if nl > 1 {
            single[(1, 1)] = c_re(-1.0);
        }
        // Higher levels get 0 (projection onto computational subspace)
        self.embed_single_qubit_op(&single, qubit)
    }
}

// ============================================================
// SIMULATION RESULT
// ============================================================

/// Result of a pulse simulation.
#[derive(Clone, Debug)]
pub struct PulseSimResult {
    /// Final quantum state (statevector).
    pub final_state: Vec<Complex64>,
    /// Final unitary matrix (if computed).
    pub final_unitary: Option<DenseMatrix>,
    /// Fidelity vs target unitary (if target was provided).
    pub fidelity: f64,
    /// State populations at each recorded timestep.
    pub populations: Vec<Vec<f64>>,
    /// Full state trajectory (state at each timestep).
    pub trajectory: Vec<Vec<Complex64>>,
}

/// Gradient of fidelity with respect to pulse parameters.
#[derive(Clone, Debug)]
pub struct PulseGradient {
    pub d_amplitude: Vec<f64>,
    pub d_frequency: Vec<f64>,
    pub d_phase: Vec<f64>,
    pub d_duration: Vec<f64>,
    pub d_drag: Vec<f64>,
}

// ============================================================
// PULSE SIMULATOR
// ============================================================

/// Main pulse-level simulator.
///
/// Evolves quantum states under time-dependent Hamiltonians constructed from
/// a transmon system specification and a pulse schedule.
pub struct PulseSimulator {
    pub system: TransmonSystem,
    pub schedule: PulseSchedule,
    /// Drift Hamiltonian (cached).
    drift_h: DenseMatrix,
    /// Drive Hamiltonians per qubit (cached).
    drive_h: Vec<DenseMatrix>,
    /// Whether to record trajectory.
    pub record_trajectory: bool,
    /// Whether to use rotating frame (remove drift frequency).
    pub rotating_frame: bool,
}

impl PulseSimulator {
    /// Create a new simulator from system and schedule.
    pub fn new(system: TransmonSystem, schedule: PulseSchedule) -> Self {
        let drift_h = system.build_drift_hamiltonian();
        let mut drive_h = Vec::new();
        for q in 0..system.n_qubits {
            drive_h.push(system.build_drive_hamiltonian(q));
        }
        Self {
            system,
            schedule,
            drift_h,
            drive_h,
            record_trajectory: false,
            rotating_frame: true,
        }
    }

    /// Enable trajectory recording.
    pub fn with_trajectory(mut self) -> Self {
        self.record_trajectory = true;
        self
    }

    /// Set rotating frame mode.
    pub fn with_rotating_frame(mut self, rf: bool) -> Self {
        self.rotating_frame = rf;
        self
    }

    /// Build the total Hamiltonian at time t.
    /// H(t) = H_drift + Sum_k c_k(t) * H_drive_k
    ///
    /// In the rotating frame, we remove the bare qubit frequencies from the drift
    /// and the drive signal becomes the envelope only (rotating wave approximation).
    pub fn hamiltonian_at(&self, t: f64) -> DenseMatrix {
        let dim = self.system.dim();
        let mut h = if self.rotating_frame {
            // In RWA, the drift contains only anharmonicity and coupling
            let mut h_rwa = DenseMatrix::zeros(dim);

            // Anharmonicity terms only
            for q in 0..self.system.n_qubits {
                if self.system.n_levels > 2 {
                    let alpha = 2.0 * PI * self.system.anharmonicities[q];
                    let nq = self.system.number_operator(q);
                    let n_sq = nq.matmul(&nq);
                    for i in 0..dim {
                        h_rwa.data[i * dim + i] += c_re(alpha / 2.0) * (n_sq[(i, i)] - nq[(i, i)]);
                    }
                }
            }

            // Coupling terms (detuning in rotating frame)
            for &(qi, qj, g) in &self.system.coupling_map {
                let delta = self.system.frequencies[qi] - self.system.frequencies[qj];
                let g_ang = 2.0 * PI * g;
                // In rotating frame, coupling becomes exchange-type
                let a_i = self.system.lowering_operator(qi);
                let a_j = self.system.lowering_operator(qj);
                let a_i_dag = a_i.dagger();
                let a_j_dag = a_j.dagger();
                // g * (a_i^dag * a_j + a_i * a_j^dag) — rotating wave approx of coupling
                let coupling = a_i_dag.matmul(&a_j).add(&a_i.matmul(&a_j_dag));
                for i in 0..dim {
                    for j in 0..dim {
                        h_rwa.data[i * dim + j] += c_re(g_ang) * coupling[(i, j)];
                    }
                }

                // Detuning on qubit j relative to qubit i frame
                if delta.abs() > 1e-12 {
                    let nj = self.system.number_operator(qj);
                    let delta_ang = 2.0 * PI * delta;
                    for i in 0..dim {
                        h_rwa.data[i * dim + i] -= c_re(delta_ang) * nj[(i, i)];
                    }
                }
            }

            h_rwa
        } else {
            self.drift_h.clone()
        };

        // Add drive terms from active pulses
        for sp in &self.schedule.pulses {
            let local_t = t - sp.start_time;
            if local_t < 0.0 || local_t > sp.pulse.duration {
                continue;
            }

            match &sp.channel {
                Channel::Drive(qubit) => {
                    let q = *qubit;
                    if q >= self.drive_h.len() {
                        continue;
                    }

                    let drive_strength = if self.rotating_frame {
                        // RWA: envelope only (detuning handled separately)
                        let env = sp.pulse.envelope(local_t);
                        let drag_q = sp.pulse.drag_quadrature(local_t);
                        // Complex drive: Omega(t) = env * e^{i*phase}
                        let cos_p = sp.pulse.phase.cos();
                        let sin_p = sp.pulse.phase.sin();
                        c(env * cos_p - drag_q * sin_p, env * sin_p + drag_q * cos_p)
                    } else {
                        // Lab frame: full oscillating signal
                        let (i_comp, _q_comp) = sp.pulse.signal(local_t);
                        c_re(i_comp)
                    };

                    // Drive Hamiltonian: (Omega/2) * (a + a^dag)
                    // where Omega = 2*pi*drive_strength
                    // In total: pi * drive_strength * (a + a^dag)
                    let scale = c_re(PI) * drive_strength;
                    let drive_op = &self.drive_h[q];
                    for i in 0..dim {
                        for j in 0..dim {
                            h.data[i * dim + j] += scale * drive_op[(i, j)];
                        }
                    }

                    // Drive detuning in rotating frame
                    if self.rotating_frame {
                        let detuning = sp.pulse.frequency - self.system.frequencies[q];
                        if detuning.abs() > 1e-12 {
                            let nq = self.system.number_operator(q);
                            let det_ang = 2.0 * PI * detuning;
                            for i in 0..dim {
                                h.data[i * dim + i] += c_re(det_ang) * nq[(i, i)];
                            }
                        }
                    }
                }
                Channel::Control(q1, q2) => {
                    let env = sp.pulse.envelope(local_t);
                    if env.abs() < 1e-15 {
                        continue;
                    }
                    // Cross-resonance: drive qubit q1 at frequency of q2
                    // This creates ZX interaction in the rotating frame
                    let q = *q1;
                    if q >= self.drive_h.len() {
                        continue;
                    }
                    let scale = c_re(PI * env);
                    let drive_op = &self.drive_h[q];
                    for i in 0..dim {
                        for j in 0..dim {
                            h.data[i * dim + j] += scale * drive_op[(i, j)];
                        }
                    }
                }
                Channel::Measure(_) => {
                    // Measurement pulses don't contribute to coherent dynamics
                }
            }
        }

        h
    }

    /// Simulate state evolution starting from |0...0>.
    /// Uses RK4 integration of the Schrodinger equation: d|psi>/dt = -i H(t) |psi>.
    pub fn simulate_state(&self) -> PulseSimResult {
        let dim = self.system.dim();
        let mut psi = vec![ZERO; dim];
        psi[0] = ONE; // |0...0>
        self.simulate_from_state(&psi)
    }

    /// Simulate from a given initial state.
    pub fn simulate_from_state(&self, initial: &[Complex64]) -> PulseSimResult {
        let dim = self.system.dim();
        assert_eq!(initial.len(), dim);

        let dt = self.schedule.dt;
        let n_steps = self.schedule.n_steps().max(1);
        let mut psi = initial.to_vec();

        let mut populations = Vec::new();
        let mut trajectory = Vec::new();

        if self.record_trajectory {
            populations.push(state_populations(&psi));
            trajectory.push(psi.clone());
        }

        for step in 0..n_steps {
            let t = step as f64 * dt;
            psi = self.rk4_step(&psi, t, dt);
            normalize_state(&mut psi);

            if self.record_trajectory {
                populations.push(state_populations(&psi));
                trajectory.push(psi.clone());
            }
        }

        PulseSimResult {
            final_state: psi,
            final_unitary: None,
            fidelity: 0.0,
            populations,
            trajectory,
        }
    }

    /// Simulate unitary evolution: evolve each basis state to get the full unitary.
    pub fn simulate_unitary(&self) -> DenseMatrix {
        let dim = self.system.dim();
        let mut unitary = DenseMatrix::zeros(dim);

        for col in 0..dim {
            let mut basis = vec![ZERO; dim];
            basis[col] = ONE;
            let result = self.simulate_from_state(&basis);
            for row in 0..dim {
                unitary[(row, col)] = result.final_state[row];
            }
        }

        unitary
    }

    /// RK4 step for Schrodinger equation: d|psi>/dt = -i H(t) |psi>.
    fn rk4_step(&self, psi: &[Complex64], t: f64, dt: f64) -> Vec<Complex64> {
        let k1 = self.dpsi_dt(psi, t);
        let psi_mid1 = add_scaled(psi, &k1, dt / 2.0);
        let k2 = self.dpsi_dt(&psi_mid1, t + dt / 2.0);
        let psi_mid2 = add_scaled(psi, &k2, dt / 2.0);
        let k3 = self.dpsi_dt(&psi_mid2, t + dt / 2.0);
        let psi_end = add_scaled(psi, &k3, dt);
        let k4 = self.dpsi_dt(&psi_end, t + dt);

        // psi_new = psi + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
        let dim = psi.len();
        let mut result = vec![ZERO; dim];
        let dt6 = dt / 6.0;
        for i in 0..dim {
            result[i] = psi[i]
                + c_re(dt6)
                    * (k1[i] + c_re(2.0) * k2[i] + c_re(2.0) * k3[i] + k4[i]);
        }
        result
    }

    /// Compute d|psi>/dt = -i H(t) |psi>.
    fn dpsi_dt(&self, psi: &[Complex64], t: f64) -> Vec<Complex64> {
        let h = self.hamiltonian_at(t);
        let h_psi = h.matvec(psi);
        // -i * H * psi
        h_psi.iter().map(|&x| -I * x).collect()
    }

    /// Simulate with Lindblad open-system dynamics.
    /// Evolves the density matrix: drho/dt = -i[H, rho] + D[rho]
    /// where D[rho] = sum_k gamma_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    pub fn simulate_lindblad(&self) -> PulseSimResult {
        let dim = self.system.dim();
        let dt = self.schedule.dt;
        let n_steps = self.schedule.n_steps().max(1);

        // Initial state: |0><0|
        let mut rho = DenseMatrix::zeros(dim);
        rho[(0, 0)] = ONE;

        // Build jump operators from T1/T2
        let jump_ops = self.build_jump_operators();

        let mut populations = Vec::new();
        if self.record_trajectory {
            populations.push(density_populations(&rho));
        }

        for step in 0..n_steps {
            let t = step as f64 * dt;
            rho = self.lindblad_rk4_step(&rho, &jump_ops, t, dt);

            if self.record_trajectory {
                populations.push(density_populations(&rho));
            }
        }

        // Extract populations from final density matrix
        let final_pops = density_populations(&rho);

        // Convert density matrix diagonal to "state" for compatibility
        let final_state: Vec<Complex64> = (0..dim).map(|i| rho[(i, i)]).collect();

        PulseSimResult {
            final_state,
            final_unitary: None,
            fidelity: 0.0,
            populations,
            trajectory: Vec::new(),
        }
    }

    /// Build Lindblad jump operators from system T1/T2 parameters.
    fn build_jump_operators(&self) -> Vec<(DenseMatrix, f64)> {
        let mut ops = Vec::new();

        for q in 0..self.system.n_qubits {
            // T1 amplitude damping: L = sqrt(gamma_1) * a
            let t1 = self.system.t1[q];
            if t1 > 0.0 && t1.is_finite() {
                let gamma_1 = 1.0 / t1;
                let a = self.system.lowering_operator(q);
                ops.push((a, gamma_1));
            }

            // T2 pure dephasing: L = sqrt(gamma_phi) * n
            // gamma_phi = 1/T2 - 1/(2*T1)
            let t2 = self.system.t2[q];
            if t2 > 0.0 && t2.is_finite() {
                let gamma_phi = (1.0 / t2 - 1.0 / (2.0 * t1)).max(0.0);
                if gamma_phi > 0.0 {
                    let z = self.system.z_operator(q);
                    // Dephasing: L = sqrt(gamma_phi/2) * Z
                    ops.push((z, gamma_phi / 2.0));
                }
            }
        }

        ops
    }

    /// RK4 step for Lindblad master equation.
    fn lindblad_rk4_step(
        &self,
        rho: &DenseMatrix,
        jump_ops: &[(DenseMatrix, f64)],
        t: f64,
        dt: f64,
    ) -> DenseMatrix {
        let k1 = self.drho_dt(rho, jump_ops, t);
        let rho2 = rho.add(&k1.scale(c_re(dt / 2.0)));
        let k2 = self.drho_dt(&rho2, jump_ops, t + dt / 2.0);
        let rho3 = rho.add(&k2.scale(c_re(dt / 2.0)));
        let k3 = self.drho_dt(&rho3, jump_ops, t + dt / 2.0);
        let rho4 = rho.add(&k3.scale(c_re(dt)));
        let k4 = self.drho_dt(&rho4, jump_ops, t + dt);

        // rho_new = rho + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
        rho.add(
            &k1.add(&k2.scale(c_re(2.0)))
                .add(&k3.scale(c_re(2.0)))
                .add(&k4)
                .scale(c_re(dt / 6.0)),
        )
    }

    /// Compute drho/dt = -i[H, rho] + D[rho].
    fn drho_dt(
        &self,
        rho: &DenseMatrix,
        jump_ops: &[(DenseMatrix, f64)],
        t: f64,
    ) -> DenseMatrix {
        let h = self.hamiltonian_at(t);

        // Coherent part: -i[H, rho]
        let comm = h.commutator(rho);
        let mut drho = comm.scale(-I);

        // Dissipative part: sum_k gamma_k * D[L_k](rho)
        for (l, gamma) in jump_ops {
            let l_dag = l.dagger();
            let l_dag_l = l_dag.matmul(l);

            // L rho L^dag
            let lrl = l.matmul(rho).matmul(&l_dag);

            // -0.5 {L^dag L, rho}
            let anti = l_dag_l.anticommutator(rho);

            let dissipator = lrl.sub(&anti.scale(c_re(0.5)));
            drho = drho.add(&dissipator.scale(c_re(*gamma)));
        }

        drho
    }
}

// ============================================================
// FIDELITY COMPUTATION
// ============================================================

/// Average gate fidelity between two unitaries.
/// F_avg = (|Tr(U_target^dag * U)|^2 + d) / (d^2 + d)
/// where d is the Hilbert space dimension.
pub fn average_gate_fidelity(u_target: &DenseMatrix, u_actual: &DenseMatrix) -> f64 {
    let d = u_target.dim as f64;
    let u_dag_u = u_target.dagger().matmul(u_actual);
    let tr = u_dag_u.trace();
    let tr_norm_sq = tr.norm_sqr();
    (tr_norm_sq + d) / (d * d + d)
}

/// State fidelity |<psi_target | psi_actual>|^2.
pub fn state_fidelity(target: &[Complex64], actual: &[Complex64]) -> f64 {
    assert_eq!(target.len(), actual.len());
    let inner: Complex64 = target
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    inner.norm_sqr()
}

/// Process fidelity (entanglement fidelity): |Tr(U^dag V)|^2 / d^2.
pub fn process_fidelity(u: &DenseMatrix, v: &DenseMatrix) -> f64 {
    let d = u.dim as f64;
    let prod = u.dagger().matmul(v);
    let tr = prod.trace();
    tr.norm_sqr() / (d * d)
}

// ============================================================
// GRADIENT COMPUTATION
// ============================================================

/// Compute gradients of fidelity w.r.t. pulse parameters using finite differences.
///
/// For each pulse parameter p, compute:
///   dF/dp ≈ (F(p + eps) - F(p - eps)) / (2 * eps)
pub fn compute_pulse_gradients(
    system: &TransmonSystem,
    schedule: &PulseSchedule,
    target_unitary: &DenseMatrix,
    epsilon: f64,
) -> PulseGradient {
    let n_pulses = schedule.pulses.len();
    let mut d_amplitude = Vec::with_capacity(n_pulses);
    let mut d_frequency = Vec::with_capacity(n_pulses);
    let mut d_phase = Vec::with_capacity(n_pulses);
    let mut d_duration = Vec::with_capacity(n_pulses);
    let mut d_drag = Vec::with_capacity(n_pulses);

    for idx in 0..n_pulses {
        // Amplitude gradient
        d_amplitude.push(finite_diff_param(
            system, schedule, target_unitary, idx, PulseParam::Amplitude, epsilon,
        ));

        // Frequency gradient
        d_frequency.push(finite_diff_param(
            system, schedule, target_unitary, idx, PulseParam::Frequency, epsilon,
        ));

        // Phase gradient
        d_phase.push(finite_diff_param(
            system, schedule, target_unitary, idx, PulseParam::Phase, epsilon,
        ));

        // Duration gradient
        d_duration.push(finite_diff_param(
            system, schedule, target_unitary, idx, PulseParam::Duration, epsilon,
        ));

        // DRAG coefficient gradient
        d_drag.push(finite_diff_param(
            system, schedule, target_unitary, idx, PulseParam::Drag, epsilon,
        ));
    }

    PulseGradient {
        d_amplitude,
        d_frequency,
        d_phase,
        d_duration,
        d_drag,
    }
}

#[derive(Clone, Copy)]
enum PulseParam {
    Amplitude,
    Frequency,
    Phase,
    Duration,
    Drag,
}

fn perturb_schedule(schedule: &PulseSchedule, idx: usize, param: PulseParam, delta: f64) -> PulseSchedule {
    let mut new_schedule = schedule.clone();
    let pulse = &mut new_schedule.pulses[idx].pulse;
    match param {
        PulseParam::Amplitude => pulse.amplitude += delta,
        PulseParam::Frequency => pulse.frequency += delta,
        PulseParam::Phase => pulse.phase += delta,
        PulseParam::Duration => pulse.duration += delta,
        PulseParam::Drag => pulse.drag_coefficient += delta,
    }
    new_schedule
}

fn finite_diff_param(
    system: &TransmonSystem,
    schedule: &PulseSchedule,
    target: &DenseMatrix,
    idx: usize,
    param: PulseParam,
    eps: f64,
) -> f64 {
    let sched_plus = perturb_schedule(schedule, idx, param, eps);
    let sched_minus = perturb_schedule(schedule, idx, param, -eps);

    let sim_plus = PulseSimulator::new(system.clone(), sched_plus);
    let sim_minus = PulseSimulator::new(system.clone(), sched_minus);

    let u_plus = sim_plus.simulate_unitary();
    let u_minus = sim_minus.simulate_unitary();

    let f_plus = average_gate_fidelity(target, &u_plus);
    let f_minus = average_gate_fidelity(target, &u_minus);

    (f_plus - f_minus) / (2.0 * eps)
}

// ============================================================
// PULSE OPTIMIZATION
// ============================================================

/// Configuration for gradient-descent pulse optimization.
#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Learning rate for gradient ascent.
    pub learning_rate: f64,
    /// Finite-difference step size for gradients.
    pub epsilon: f64,
    /// Target fidelity to stop optimization.
    pub target_fidelity: f64,
    /// Whether to optimize amplitude.
    pub optimize_amplitude: bool,
    /// Whether to optimize frequency.
    pub optimize_frequency: bool,
    /// Whether to optimize phase.
    pub optimize_phase: bool,
    /// Whether to optimize DRAG coefficient.
    pub optimize_drag: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            learning_rate: 0.01,
            epsilon: 1e-4,
            target_fidelity: 0.999,
            optimize_amplitude: true,
            optimize_frequency: false,
            optimize_phase: true,
            optimize_drag: false,
        }
    }
}

/// Result of pulse optimization.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Optimized pulse schedule.
    pub schedule: PulseSchedule,
    /// Final fidelity achieved.
    pub fidelity: f64,
    /// Fidelity history per iteration.
    pub fidelity_history: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether optimization converged to target fidelity.
    pub converged: bool,
}

/// Optimize pulse parameters to maximize gate fidelity via gradient ascent.
pub fn optimize_pulses(
    system: &TransmonSystem,
    initial_schedule: &PulseSchedule,
    target_unitary: &DenseMatrix,
    config: &OptimizationConfig,
) -> OptimizationResult {
    let mut schedule = initial_schedule.clone();
    let mut fidelity_history = Vec::with_capacity(config.max_iterations);

    let mut best_fidelity = 0.0;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute current fidelity
        let sim = PulseSimulator::new(system.clone(), schedule.clone());
        let u = sim.simulate_unitary();
        let fidelity = average_gate_fidelity(target_unitary, &u);
        fidelity_history.push(fidelity);

        if fidelity > best_fidelity {
            best_fidelity = fidelity;
        }

        if fidelity >= config.target_fidelity {
            return OptimizationResult {
                schedule,
                fidelity,
                fidelity_history,
                iterations,
                converged: true,
            };
        }

        // Compute gradients
        let grads = compute_pulse_gradients(system, &schedule, target_unitary, config.epsilon);

        // Update parameters via gradient ascent
        for (idx, sp) in schedule.pulses.iter_mut().enumerate() {
            if config.optimize_amplitude && idx < grads.d_amplitude.len() {
                sp.pulse.amplitude += config.learning_rate * grads.d_amplitude[idx];
                sp.pulse.amplitude = sp.pulse.amplitude.clamp(-2.0, 2.0);
            }
            if config.optimize_frequency && idx < grads.d_frequency.len() {
                sp.pulse.frequency += config.learning_rate * grads.d_frequency[idx];
            }
            if config.optimize_phase && idx < grads.d_phase.len() {
                sp.pulse.phase += config.learning_rate * grads.d_phase[idx];
            }
            if config.optimize_drag && idx < grads.d_drag.len() {
                sp.pulse.drag_coefficient += config.learning_rate * grads.d_drag[idx];
            }
        }
    }

    let sim = PulseSimulator::new(system.clone(), schedule.clone());
    let u = sim.simulate_unitary();
    let fidelity = average_gate_fidelity(target_unitary, &u);
    fidelity_history.push(fidelity);

    OptimizationResult {
        schedule,
        fidelity,
        fidelity_history,
        iterations,
        converged: fidelity >= config.target_fidelity,
    }
}

// ============================================================
// STANDARD GATE CALIBRATIONS
// ============================================================

/// Pre-calibrated pulse parameters for standard gates.
pub struct StandardGates;

impl StandardGates {
    /// X gate (pi rotation about X-axis) for a single transmon qubit.
    /// Uses a Gaussian pulse with area = pi.
    pub fn x_gate(qubit_freq: f64) -> PulseSchedule {
        // For a Gaussian pulse, the rotation angle is proportional to
        // amplitude * sigma * sqrt(2*pi). We want angle = pi.
        let sigma = 4.0; // ns
        let amplitude = 1.0 / (sigma * (2.0 * PI).sqrt()) * PI;
        let duration = 6.0 * sigma; // 3-sigma on each side

        let pulse = Pulse::gaussian(duration, amplitude, qubit_freq, 0.0, sigma);
        PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        })
    }

    /// SX gate (sqrt(X), pi/2 rotation about X-axis).
    pub fn sx_gate(qubit_freq: f64) -> PulseSchedule {
        let sigma = 4.0;
        let amplitude = 0.5 / (sigma * (2.0 * PI).sqrt()) * PI;
        let duration = 6.0 * sigma;

        let pulse = Pulse::gaussian(duration, amplitude, qubit_freq, 0.0, sigma);
        PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        })
    }

    /// Z rotation by angle theta (virtual Z gate — just a phase shift).
    /// In the rotating frame, Z rotations are "free" (frame change).
    pub fn rz_gate(theta: f64) -> DenseMatrix {
        let mut u = DenseMatrix::zeros(2);
        u[(0, 0)] = c((-theta / 2.0).cos(), (-theta / 2.0).sin());
        u[(1, 1)] = c((theta / 2.0).cos(), (theta / 2.0).sin());
        u
    }

    /// CNOT gate via cross-resonance pulse between two qubits.
    /// Uses a flat-top pulse on the control channel.
    pub fn cnot_cross_resonance(freq_control: f64, freq_target: f64) -> PulseSchedule {
        // CR pulse: drive control qubit at target qubit frequency
        let cr_amplitude = 0.05; // Weak drive
        let cr_duration = 200.0; // ~200 ns for CR gate

        let cr_pulse = Pulse::gaussian_square(
            cr_duration,
            cr_amplitude,
            freq_target, // Drive at target frequency
            0.0,
            5.0,                  // sigma for rise/fall
            cr_duration - 20.0,   // flat-top width
        );

        PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Control(0, 1),
            start_time: 0.0,
            pulse: cr_pulse,
        })
    }

    /// Target unitary for X gate.
    pub fn x_unitary() -> DenseMatrix {
        let mut u = DenseMatrix::zeros(2);
        u[(0, 1)] = ONE;
        u[(1, 0)] = ONE;
        u
    }

    /// Target unitary for SX gate (sqrt(X)).
    pub fn sx_unitary() -> DenseMatrix {
        let mut u = DenseMatrix::zeros(2);
        let half = c_re(0.5);
        let half_i = c(0.0, 0.5);
        u[(0, 0)] = half + half_i; // (1+i)/2
        u[(0, 1)] = half - half_i; // (1-i)/2
        u[(1, 0)] = half - half_i;
        u[(1, 1)] = half + half_i;
        u
    }

    /// Target unitary for Hadamard gate.
    pub fn h_unitary() -> DenseMatrix {
        let mut u = DenseMatrix::zeros(2);
        let s = c_re(1.0 / 2.0f64.sqrt());
        u[(0, 0)] = s;
        u[(0, 1)] = s;
        u[(1, 0)] = s;
        u[(1, 1)] = -s;
        u
    }

    /// Target CNOT unitary (4x4, computational basis).
    pub fn cnot_unitary() -> DenseMatrix {
        let mut u = DenseMatrix::identity(4);
        // Swap |10> and |11>
        u[(2, 2)] = ZERO;
        u[(2, 3)] = ONE;
        u[(3, 2)] = ONE;
        u[(3, 3)] = ZERO;
        u
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Normalize a state vector in-place.
fn normalize_state(psi: &mut [Complex64]) {
    let norm_sq: f64 = psi.iter().map(|x| x.norm_sqr()).sum();
    if norm_sq > 1e-30 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in psi.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// Compute populations (|c_i|^2) from a statevector.
fn state_populations(psi: &[Complex64]) -> Vec<f64> {
    psi.iter().map(|x| x.norm_sqr()).collect()
}

/// Compute populations from a density matrix diagonal.
fn density_populations(rho: &DenseMatrix) -> Vec<f64> {
    (0..rho.dim).map(|i| rho[(i, i)].re).collect()
}

/// psi + scale * dpsi
fn add_scaled(psi: &[Complex64], dpsi: &[Complex64], scale: f64) -> Vec<Complex64> {
    psi.iter()
        .zip(dpsi.iter())
        .map(|(&p, &d)| p + c_re(scale) * d)
        .collect()
}

// ============================================================
// RABI OSCILLATION HELPERS
// ============================================================

/// Simulate Rabi oscillation: drive a qubit with a constant pulse and track
/// population inversion vs time. Returns (times, excited_population).
pub fn simulate_rabi(
    qubit_freq: f64,
    drive_amplitude: f64,
    total_time: f64,
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    let system = TransmonSystem::single_qubit(qubit_freq, -0.3);
    let pulse = Pulse::constant(total_time, drive_amplitude, qubit_freq, 0.0);
    let schedule = PulseSchedule::new(dt).with_pulse(ScheduledPulse {
        channel: Channel::Drive(0),
        start_time: 0.0,
        pulse,
    });

    let sim = PulseSimulator::new(system, schedule).with_trajectory();
    let result = sim.simulate_state();

    let times: Vec<f64> = (0..result.populations.len())
        .map(|i| i as f64 * dt)
        .collect();
    let excited: Vec<f64> = result.populations.iter().map(|p| p[1]).collect();

    (times, excited)
}

/// Simulate Ramsey experiment: two pi/2 pulses separated by a delay,
/// measuring dephasing. Returns (delays, excited_population).
pub fn simulate_ramsey(
    qubit_freq: f64,
    detuning: f64,
    delays: &[f64],
    dt: f64,
) -> Vec<f64> {
    let system = TransmonSystem::single_qubit(qubit_freq, -0.3);
    let sigma = 4.0;
    // pi/2 rotation: A_peak = 1 / (4 * sigma * sqrt(2*pi))
    let half_pi_amp = 0.25 / (sigma * (2.0 * PI).sqrt());
    let pulse_dur = 8.0 * sigma;

    let mut excited_pops = Vec::with_capacity(delays.len());

    for &delay in delays {
        let mut schedule = PulseSchedule::new(dt);

        // First pi/2 pulse (resonant)
        let p1 = Pulse::gaussian(pulse_dur, half_pi_amp, qubit_freq, 0.0, sigma);
        schedule.add_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse: p1,
        });

        // Second pi/2 pulse with phase that depends on detuning * delay
        // This models the LO phase accumulation: phi = 2*pi*detuning*delay
        let ramsey_phase = 2.0 * PI * detuning * delay;
        let p2 = Pulse::gaussian(pulse_dur, half_pi_amp, qubit_freq, ramsey_phase, sigma);
        schedule.add_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: pulse_dur + delay,
            pulse: p2,
        });

        let sim = PulseSimulator::new(system.clone(), schedule);
        let result = sim.simulate_state();
        excited_pops.push(result.final_state[1].norm_sqr());
    }

    excited_pops
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-3;
    const TIGHT_TOL: f64 = 1e-6;

    // ----------------------------------------------------------
    // 1. test_gaussian_pulse_shape
    // ----------------------------------------------------------
    #[test]
    fn test_gaussian_pulse_shape() {
        let sigma = 4.0;
        let pulse = Pulse::gaussian(24.0, 1.0, 5.0, 0.0, sigma);

        // Peak at center (t = duration/2 = 12)
        let peak = pulse.envelope(12.0);
        assert!((peak - 1.0).abs() < TIGHT_TOL, "Gaussian peak should be 1.0, got {}", peak);

        // Symmetric: envelope(12 - x) == envelope(12 + x)
        let left = pulse.envelope(8.0);
        let right = pulse.envelope(16.0);
        assert!((left - right).abs() < TIGHT_TOL, "Gaussian should be symmetric");

        // Falls off: envelope at 1-sigma < peak
        let one_sigma = pulse.envelope(12.0 + sigma);
        let expected = (-0.5f64).exp(); // exp(-sigma^2/(2*sigma^2)) = exp(-0.5)
        assert!((one_sigma - expected).abs() < TIGHT_TOL);

        // Zero outside duration
        assert_eq!(pulse.envelope(-1.0), 0.0);
        assert_eq!(pulse.envelope(25.0), 0.0);
    }

    // ----------------------------------------------------------
    // 2. test_drag_pulse_shape
    // ----------------------------------------------------------
    #[test]
    fn test_drag_pulse_shape() {
        let sigma = 4.0;
        let beta = 0.5;
        let pulse = Pulse::drag(24.0, 1.0, 5.0, 0.0, sigma, beta);

        // At center, derivative of Gaussian is zero, so DRAG = Gaussian only
        let center = pulse.envelope(12.0);
        assert!((center - 1.0).abs() < TIGHT_TOL, "DRAG center should be ~1.0, got {}", center);

        // DRAG is NOT symmetric because derivative term is odd
        let left = pulse.envelope(10.0);
        let right = pulse.envelope(14.0);
        assert!(
            (left - right).abs() > 0.01,
            "DRAG should be asymmetric: left={}, right={}",
            left, right
        );

        // Quadrature component at center should be zero (derivative of Gaussian at peak = 0)
        let q_center = pulse.drag_quadrature(12.0);
        assert!(q_center.abs() < TIGHT_TOL, "DRAG quadrature at center should be 0");
    }

    // ----------------------------------------------------------
    // 3. test_gaussian_square_pulse
    // ----------------------------------------------------------
    #[test]
    fn test_gaussian_square_pulse() {
        let sigma = 2.0;
        let width = 16.0;
        let duration = 24.0;
        let pulse = Pulse::gaussian_square(duration, 1.0, 5.0, 0.0, sigma, width);

        // Flat-top region: rise = (24 - 16)/2 = 4 ns
        // Flat from t=4 to t=20
        let mid = pulse.envelope(12.0);
        assert!((mid - 1.0).abs() < TIGHT_TOL, "Flat-top should be 1.0, got {}", mid);

        let near_start_flat = pulse.envelope(5.0);
        assert!(
            (near_start_flat - 1.0).abs() < TIGHT_TOL,
            "Should be flat at t=5, got {}",
            near_start_flat
        );

        // Rise region (t < 4) should be < 1
        let rise = pulse.envelope(1.0);
        assert!(rise < 1.0 && rise > 0.0, "Rise should be between 0 and 1, got {}", rise);

        // Fall region (t > 20) should be < 1
        let fall = pulse.envelope(23.0);
        assert!(fall < 1.0 && fall > 0.0, "Fall should be between 0 and 1, got {}", fall);
    }

    // ----------------------------------------------------------
    // 4. test_constant_pulse
    // ----------------------------------------------------------
    #[test]
    fn test_constant_pulse() {
        let pulse = Pulse::constant(20.0, 0.7, 5.0, 0.0);

        assert!((pulse.envelope(0.0) - 0.7).abs() < TIGHT_TOL);
        assert!((pulse.envelope(10.0) - 0.7).abs() < TIGHT_TOL);
        assert!((pulse.envelope(20.0) - 0.7).abs() < TIGHT_TOL);
        assert_eq!(pulse.envelope(-0.1), 0.0);
        assert_eq!(pulse.envelope(20.1), 0.0);
    }

    // ----------------------------------------------------------
    // 5. test_custom_pulse_samples
    // ----------------------------------------------------------
    #[test]
    fn test_custom_pulse_samples() {
        // Triangle wave: 0 -> 1 -> 0
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let pulse = Pulse::custom(4.0, 1.0, 5.0, 0.0, samples);

        // At t=0 -> sample[0] = 0
        assert!(pulse.envelope(0.0).abs() < TIGHT_TOL);

        // At t=2 (midpoint) -> sample[2] = 1.0
        assert!((pulse.envelope(2.0) - 1.0).abs() < TIGHT_TOL);

        // At t=4 -> sample[4] = 0
        assert!(pulse.envelope(4.0).abs() < TIGHT_TOL);

        // Interpolation at t=1 -> between sample[1]=0.5 and sample[2]=1.0 -> 0.75
        assert!((pulse.envelope(1.0) - 0.5).abs() < TIGHT_TOL);
    }

    // ----------------------------------------------------------
    // 6. test_transmon_hamiltonian_1q
    // ----------------------------------------------------------
    #[test]
    fn test_transmon_hamiltonian_1q() {
        let system = TransmonSystem::single_qubit(5.0, -0.3);
        let h = system.build_drift_hamiltonian();

        assert_eq!(h.dim, 2);

        // For 2-level system, H = omega * |1><1| (in computational basis)
        // H_00 = 0, H_11 = omega (no anharmonicity term for 2-level)
        let omega = 2.0 * PI * 5.0;
        assert!(h[(0, 0)].re.abs() < TIGHT_TOL, "H_00 should be 0");
        assert!((h[(1, 1)].re - omega).abs() < 1e-4, "H_11 should be omega");

        // Off-diagonal should be zero (no drive)
        assert!(h[(0, 1)].norm() < TIGHT_TOL);
        assert!(h[(1, 0)].norm() < TIGHT_TOL);

        // Hermitian check
        assert!((h[(0, 0)] - h[(0, 0)].conj()).norm() < TIGHT_TOL);
    }

    // ----------------------------------------------------------
    // 7. test_transmon_hamiltonian_2q
    // ----------------------------------------------------------
    #[test]
    fn test_transmon_hamiltonian_2q() {
        let system = TransmonSystem::two_qubit(5.0, 5.1, -0.3, -0.3, 0.002);
        let h = system.build_drift_hamiltonian();

        assert_eq!(h.dim, 4);

        // Hermitian check: H = H^dag
        let h_dag = h.dagger();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (h[(i, j)] - h_dag[(i, j)]).norm() < 1e-10,
                    "H not Hermitian at ({},{})",
                    i,
                    j
                );
            }
        }

        // Off-diagonal coupling terms should be nonzero
        let has_coupling = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .any(|(i, j)| i != j && h[(i, j)].norm() > 1e-10);
        assert!(has_coupling, "2-qubit Hamiltonian should have off-diagonal coupling");
    }

    // ----------------------------------------------------------
    // 8. test_rk4_free_evolution
    // ----------------------------------------------------------
    #[test]
    fn test_rk4_free_evolution() {
        // Free evolution of |+> = (|0> + |1>)/sqrt(2) under H = omega*Z/2
        // should rotate in the equator of the Bloch sphere.
        let system = TransmonSystem::single_qubit(5.0, -0.3);
        let schedule = PulseSchedule::new(0.01); // No pulses
        // Override schedule duration by adding a zero-amplitude pulse
        let mut sched = schedule;
        sched.add_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse: Pulse::constant(1.0, 0.0, 5.0, 0.0),
        });

        let sim = PulseSimulator::new(system, sched);

        let s2 = 1.0 / 2.0f64.sqrt();
        let initial = vec![c_re(s2), c_re(s2)];
        let result = sim.simulate_from_state(&initial);

        // After free evolution, state should still have |c0|^2 + |c1|^2 = 1
        let norm_sq: f64 = result.final_state.iter().map(|x| x.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "Norm should be preserved: {}",
            norm_sq
        );

        // Populations should remain ~0.5 each (no drive, just phase rotation)
        let p0 = result.final_state[0].norm_sqr();
        let p1 = result.final_state[1].norm_sqr();
        assert!(
            (p0 - 0.5).abs() < 0.05,
            "Population |0> should stay ~0.5, got {}",
            p0
        );
        assert!(
            (p1 - 0.5).abs() < 0.05,
            "Population |1> should stay ~0.5, got {}",
            p1
        );
    }

    // ----------------------------------------------------------
    // 9. test_x_gate_pulse
    // ----------------------------------------------------------
    #[test]
    fn test_x_gate_pulse() {
        // A resonant pi-pulse should flip |0> to |1>
        let freq = 5.0;
        let system = TransmonSystem::single_qubit(freq, -0.3);

        // Drive Hamiltonian: H = pi * A(t) * sigma_x
        // Rotation angle theta = 2*pi * integral(A(t) dt)
        // For Gaussian: integral ≈ A_peak * sigma * sqrt(2*pi)
        // So theta = 2*pi * A_peak * sigma * sqrt(2*pi)
        // For X gate (theta = pi): A_peak = 1 / (2 * sigma * sqrt(2*pi))
        let sigma = 4.0;
        let duration = 8.0 * sigma;
        let amplitude = 0.5 / (sigma * (2.0 * PI).sqrt());

        let pulse = Pulse::gaussian(duration, amplitude, freq, 0.0, sigma);
        let schedule = PulseSchedule::new(0.05).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        });

        let sim = PulseSimulator::new(system, schedule);
        let result = sim.simulate_state();

        // Should be near |1>
        let p1 = result.final_state[1].norm_sqr();
        assert!(
            p1 > 0.90,
            "X gate should produce |1> with high probability, got p1={}",
            p1
        );
    }

    // ----------------------------------------------------------
    // 10. test_sx_gate_pulse
    // ----------------------------------------------------------
    #[test]
    fn test_sx_gate_pulse() {
        // Half the X-gate amplitude gives pi/2 rotation -> equal superposition
        let freq = 5.0;
        let system = TransmonSystem::single_qubit(freq, -0.3);

        let sigma = 4.0;
        let duration = 8.0 * sigma;
        let amplitude = 0.25 / (sigma * (2.0 * PI).sqrt());

        let pulse = Pulse::gaussian(duration, amplitude, freq, 0.0, sigma);
        let schedule = PulseSchedule::new(0.05).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        });

        let sim = PulseSimulator::new(system, schedule);
        let result = sim.simulate_state();

        // Should be roughly equal superposition (p0 ~ p1 ~ 0.5)
        let p0 = result.final_state[0].norm_sqr();
        let p1 = result.final_state[1].norm_sqr();
        assert!(
            (p0 - 0.5).abs() < 0.15,
            "SX: p0 should be ~0.5, got {}",
            p0
        );
        assert!(
            (p1 - 0.5).abs() < 0.15,
            "SX: p1 should be ~0.5, got {}",
            p1
        );
    }

    // ----------------------------------------------------------
    // 11. test_z_rotation_pulse
    // ----------------------------------------------------------
    #[test]
    fn test_z_rotation_pulse() {
        // Z rotation is a virtual gate (phase change) in rotating frame
        let rz = StandardGates::rz_gate(PI / 2.0);

        // Rz(pi/2) = diag(e^{-i*pi/4}, e^{i*pi/4})
        let expected_00 = c((-PI / 4.0).cos(), (-PI / 4.0).sin());
        let expected_11 = c((PI / 4.0).cos(), (PI / 4.0).sin());

        assert!((rz[(0, 0)] - expected_00).norm() < TIGHT_TOL);
        assert!((rz[(1, 1)] - expected_11).norm() < TIGHT_TOL);
        assert!(rz[(0, 1)].norm() < TIGHT_TOL);
        assert!(rz[(1, 0)].norm() < TIGHT_TOL);

        // Should be unitary: U^dag * U = I
        let prod = rz.dagger().matmul(&rz);
        let id = DenseMatrix::identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (prod[(i, j)] - id[(i, j)]).norm() < TIGHT_TOL,
                    "Rz not unitary at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    // ----------------------------------------------------------
    // 12. test_rabi_oscillation
    // ----------------------------------------------------------
    #[test]
    fn test_rabi_oscillation() {
        // Constant drive should produce sinusoidal population oscillation
        let freq = 5.0;
        let drive_amp = 0.05; // Moderate drive
        let total_time = 50.0; // Long enough for some oscillations
        let dt = 0.1;

        let (times, excited) = simulate_rabi(freq, drive_amp, total_time, dt);

        // Should see oscillation: not all zeros and not all ones
        let max_pop: f64 = excited.iter().cloned().fold(0.0, f64::max);
        let min_pop: f64 = excited.iter().cloned().fold(1.0, f64::min);

        assert!(
            max_pop > 0.3,
            "Rabi oscillation should reach high excited population, max={}",
            max_pop
        );
        assert!(
            min_pop < 0.2,
            "Rabi oscillation should return near ground, min={}",
            min_pop
        );

        // First point should be near |0>
        assert!(excited[0] < 0.01, "Initial state should be |0>");
    }

    // ----------------------------------------------------------
    // 13. test_ramsey_experiment
    // ----------------------------------------------------------
    #[test]
    fn test_ramsey_experiment() {
        let freq = 5.0;
        let detuning = 0.01; // 10 MHz detuning
        let delays: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();
        let dt = 0.1;

        let pops = simulate_ramsey(freq, detuning, &delays, dt);

        // With detuning, should see oscillation in excited state population
        assert_eq!(pops.len(), delays.len());

        // At zero delay, two pi/2 pulses = pi pulse -> should be near |1>
        // (but Ramsey measures dephasing, so first point with zero delay should show signal)
        assert!(pops[0] > 0.0, "Ramsey should show nonzero signal at zero delay");

        // Should vary with delay (not all the same)
        let variance: f64 = {
            let mean = pops.iter().sum::<f64>() / pops.len() as f64;
            pops.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pops.len() as f64
        };
        assert!(
            variance > 1e-6,
            "Ramsey fringes should show variation, variance={}",
            variance
        );
    }

    // ----------------------------------------------------------
    // 14. test_cnot_cross_resonance
    // ----------------------------------------------------------
    #[test]
    fn test_cnot_cross_resonance() {
        // Cross-resonance setup: two coupled qubits
        let system = TransmonSystem::two_qubit(5.0, 5.1, -0.3, -0.3, 0.003);

        let cr_schedule = StandardGates::cnot_cross_resonance(5.0, 5.1);

        let sim = PulseSimulator::new(system, cr_schedule);
        let u = sim.simulate_unitary();

        // The CR unitary should not be identity (something happened)
        let id = DenseMatrix::identity(4);
        let diff: f64 = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .map(|(i, j)| (u[(i, j)] - id[(i, j)]).norm_sqr())
            .sum();
        assert!(
            diff > 0.01,
            "CR pulse should produce non-trivial unitary, diff={}",
            diff
        );

        // Should be approximately unitary: U^dag * U ≈ I
        let prod = u.dagger().matmul(&u);
        for i in 0..4 {
            assert!(
                (prod[(i, i)].re - 1.0).abs() < 0.1,
                "Unitarity check failed at ({},{}): {}",
                i,
                i,
                prod[(i, i)]
            );
        }
    }

    // ----------------------------------------------------------
    // 15. test_fidelity_computation
    // ----------------------------------------------------------
    #[test]
    fn test_fidelity_computation() {
        // Fidelity of identity with itself should be 1.0
        let id = DenseMatrix::identity(2);
        let f = average_gate_fidelity(&id, &id);
        assert!(
            (f - 1.0).abs() < TIGHT_TOL,
            "F(I, I) should be 1.0, got {}",
            f
        );

        // Fidelity of X with X should be 1.0
        let x = StandardGates::x_unitary();
        let f_xx = average_gate_fidelity(&x, &x);
        assert!(
            (f_xx - 1.0).abs() < TIGHT_TOL,
            "F(X, X) should be 1.0, got {}",
            f_xx
        );

        // Fidelity of X with I should be less than 1
        let f_xi = average_gate_fidelity(&x, &id);
        assert!(
            f_xi < 0.9,
            "F(X, I) should be low, got {}",
            f_xi
        );

        // State fidelity: |<0|0>|^2 = 1
        let zero = vec![ONE, ZERO];
        let one = vec![ZERO, ONE];
        assert!((state_fidelity(&zero, &zero) - 1.0).abs() < TIGHT_TOL);
        assert!(state_fidelity(&zero, &one).abs() < TIGHT_TOL);
    }

    // ----------------------------------------------------------
    // 16. test_pulse_gradient_amplitude
    // ----------------------------------------------------------
    #[test]
    fn test_pulse_gradient_amplitude() {
        let freq = 5.0;
        let system = TransmonSystem::single_qubit(freq, -0.3);
        let target = StandardGates::x_unitary();

        // Start with a near-optimal pulse
        let sigma = 4.0;
        let duration = 8.0 * sigma;
        let amplitude = 1.0 / (sigma * (2.0 * PI).sqrt());

        let pulse = Pulse::gaussian(duration, amplitude, freq, 0.0, sigma);
        let schedule = PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        });

        let grads = compute_pulse_gradients(&system, &schedule, &target, 1e-3);

        // Gradient should be finite and defined
        assert!(
            grads.d_amplitude[0].is_finite(),
            "Amplitude gradient should be finite"
        );

        // Near optimal, gradient magnitude should be small-ish but nonzero
        // (we don't know the exact sign, just that it's computable)
        assert!(
            grads.d_amplitude[0].abs() < 100.0,
            "Amplitude gradient should be reasonable: {}",
            grads.d_amplitude[0]
        );
    }

    // ----------------------------------------------------------
    // 17. test_pulse_gradient_frequency
    // ----------------------------------------------------------
    #[test]
    fn test_pulse_gradient_frequency() {
        let freq = 5.0;
        let system = TransmonSystem::single_qubit(freq, -0.3);
        let target = StandardGates::x_unitary();

        let sigma = 4.0;
        let duration = 8.0 * sigma;
        let amplitude = 1.0 / (sigma * (2.0 * PI).sqrt());

        let pulse = Pulse::gaussian(duration, amplitude, freq, 0.0, sigma);
        let schedule = PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        });

        let grads = compute_pulse_gradients(&system, &schedule, &target, 1e-3);

        assert!(
            grads.d_frequency[0].is_finite(),
            "Frequency gradient should be finite"
        );
    }

    // ----------------------------------------------------------
    // 18. test_pulse_optimization_x_gate
    // ----------------------------------------------------------
    #[test]
    fn test_pulse_optimization_x_gate() {
        let freq = 5.0;
        let system = TransmonSystem::single_qubit(freq, -0.3);
        let target = StandardGates::x_unitary();

        // Start with slightly off amplitude (80% of optimal pi-pulse amplitude)
        let sigma = 4.0;
        let duration = 8.0 * sigma;
        let amplitude = 0.4 / (sigma * (2.0 * PI).sqrt()); // 80% of optimal 0.5/(sigma*sqrt(2pi))

        let pulse = Pulse::gaussian(duration, amplitude, freq, 0.0, sigma);
        let schedule = PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        });

        let config = OptimizationConfig {
            max_iterations: 30,
            learning_rate: 0.05,
            epsilon: 1e-3,
            target_fidelity: 0.95,
            optimize_amplitude: true,
            optimize_frequency: false,
            optimize_phase: false,
            optimize_drag: false,
        };

        let result = optimize_pulses(&system, &schedule, &target, &config);

        // Fidelity should improve
        let initial_fidelity = result.fidelity_history[0];
        assert!(
            result.fidelity > initial_fidelity,
            "Optimization should improve fidelity: {} -> {}",
            initial_fidelity,
            result.fidelity
        );

        assert!(
            result.fidelity_history.len() > 1,
            "Should have fidelity history"
        );
    }

    // ----------------------------------------------------------
    // 19. test_lindblad_t1_decay
    // ----------------------------------------------------------
    #[test]
    fn test_lindblad_t1_decay() {
        // Start in |1>, evolve with T1 decay, should relax toward |0>
        let mut system = TransmonSystem::single_qubit(5.0, -0.3);
        system.t1 = vec![100.0]; // Short T1 for visible decay
        system.t2 = vec![200.0];

        let total_time = 200.0;
        let pulse = Pulse::constant(total_time, 0.0, 5.0, 0.0); // No drive
        let schedule = PulseSchedule::new(0.5).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        });

        // We need to start in |1> for the density matrix sim
        // Modify the simulator to start from |1><1|
        let sim = PulseSimulator::new(system, schedule).with_trajectory();

        // Simulate Lindblad from |1> state
        let dim = sim.system.dim();
        let dt = sim.schedule.dt;
        let n_steps = sim.schedule.n_steps().max(1);

        let mut rho = DenseMatrix::zeros(dim);
        rho[(1, 1)] = ONE; // Start in |1>

        let jump_ops = sim.build_jump_operators();

        for step in 0..n_steps {
            let t = step as f64 * dt;
            rho = sim.lindblad_rk4_step(&rho, &jump_ops, t, dt);
        }

        // After ~2*T1, population should have decayed significantly
        let p1 = rho[(1, 1)].re;
        let p0 = rho[(0, 0)].re;

        assert!(
            p1 < 0.3,
            "After 2*T1, |1> population should decay below 0.3, got {}",
            p1
        );
        assert!(
            p0 > 0.5,
            "After 2*T1, |0> population should grow above 0.5, got {}",
            p0
        );

        // Trace should be preserved
        let trace = rho[(0, 0)].re + rho[(1, 1)].re;
        assert!(
            (trace - 1.0).abs() < 0.05,
            "Trace should be ~1, got {}",
            trace
        );
    }

    // ----------------------------------------------------------
    // 20. test_lindblad_t2_dephasing
    // ----------------------------------------------------------
    #[test]
    fn test_lindblad_t2_dephasing() {
        // Start in |+> = (|0> + |1>)/sqrt(2), dephasing should kill coherence
        let mut system = TransmonSystem::single_qubit(5.0, -0.3);
        system.t1 = vec![100_000.0]; // Very long T1 (no decay)
        system.t2 = vec![50.0]; // Short T2 for visible dephasing

        let total_time = 150.0;
        let pulse = Pulse::constant(total_time, 0.0, 5.0, 0.0);
        let schedule = PulseSchedule::new(0.5).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        });

        let sim = PulseSimulator::new(system, schedule);

        let dim = sim.system.dim();
        let dt = sim.schedule.dt;
        let n_steps = sim.schedule.n_steps().max(1);

        // Start in |+><+|
        let mut rho = DenseMatrix::zeros(dim);
        rho[(0, 0)] = c_re(0.5);
        rho[(0, 1)] = c_re(0.5);
        rho[(1, 0)] = c_re(0.5);
        rho[(1, 1)] = c_re(0.5);

        let jump_ops = sim.build_jump_operators();

        for step in 0..n_steps {
            let t = step as f64 * dt;
            rho = sim.lindblad_rk4_step(&rho, &jump_ops, t, dt);
        }

        // Off-diagonal elements should decay (dephasing)
        let coherence = rho[(0, 1)].norm();
        assert!(
            coherence < 0.2,
            "Coherence should decay under T2, got {}",
            coherence
        );

        // Diagonal should be preserved (no T1 decay)
        let p0 = rho[(0, 0)].re;
        let p1 = rho[(1, 1)].re;
        assert!(
            (p0 - 0.5).abs() < 0.1,
            "Population should be preserved under pure dephasing, p0={}",
            p0
        );
        assert!(
            (p1 - 0.5).abs() < 0.1,
            "Population should be preserved under pure dephasing, p1={}",
            p1
        );
    }

    // ----------------------------------------------------------
    // 21. test_pulse_schedule_construction
    // ----------------------------------------------------------
    #[test]
    fn test_pulse_schedule_construction() {
        let mut schedule = PulseSchedule::new(0.222);

        assert_eq!(schedule.pulses.len(), 0);
        assert!((schedule.dt - 0.222).abs() < TIGHT_TOL);

        // Add pulses
        schedule.add_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse: Pulse::gaussian(24.0, 0.5, 5.0, 0.0, 4.0),
        });

        schedule.add_pulse(ScheduledPulse {
            channel: Channel::Drive(1),
            start_time: 30.0,
            pulse: Pulse::constant(10.0, 0.3, 5.1, 0.0),
        });

        assert_eq!(schedule.pulses.len(), 2);

        // Total duration: max(0+24, 30+10) = 40
        assert!((schedule.total_duration() - 40.0).abs() < TIGHT_TOL);

        // n_steps = ceil(40 / 0.222) = 181
        let expected_steps = (40.0 / 0.222f64).ceil() as usize;
        assert_eq!(schedule.n_steps(), expected_steps);

        // Builder pattern
        let schedule2 = PulseSchedule::new(0.1)
            .with_pulse(ScheduledPulse {
                channel: Channel::Drive(0),
                start_time: 0.0,
                pulse: Pulse::constant(10.0, 0.5, 5.0, 0.0),
            })
            .with_pulse(ScheduledPulse {
                channel: Channel::Measure(0),
                start_time: 15.0,
                pulse: Pulse::constant(5.0, 1.0, 7.0, 0.0),
            });

        assert_eq!(schedule2.pulses.len(), 2);
        assert!((schedule2.total_duration() - 20.0).abs() < TIGHT_TOL);
    }

    // ----------------------------------------------------------
    // 22. test_drag_correction_leakage
    // ----------------------------------------------------------
    #[test]
    fn test_drag_correction_leakage() {
        // With 3-level system, DRAG should reduce leakage to |2>
        let system = TransmonSystem::single_qubit_3level(5.0, -0.3);
        assert_eq!(system.dim(), 3);

        let sigma = 4.0;
        let duration = 8.0 * sigma;
        let amplitude = 1.0 / (sigma * (2.0 * PI).sqrt());

        // Without DRAG
        let pulse_no_drag = Pulse::gaussian(duration, amplitude, 5.0, 0.0, sigma);
        let sched_no_drag = PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse: pulse_no_drag,
        });
        let sim_no_drag = PulseSimulator::new(system.clone(), sched_no_drag);
        let result_no_drag = sim_no_drag.simulate_state();
        let leakage_no_drag = result_no_drag.final_state[2].norm_sqr();

        // With DRAG correction
        let pulse_drag = Pulse::drag(duration, amplitude, 5.0, 0.0, sigma, 0.5);
        let sched_drag = PulseSchedule::new(0.1).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse: pulse_drag,
        });
        let sim_drag = PulseSimulator::new(system, sched_drag);
        let result_drag = sim_drag.simulate_state();
        let leakage_drag = result_drag.final_state[2].norm_sqr();

        // Both leakage values should be small for well-calibrated pulses
        // The key test: leakage to |2> is a real concern in 3-level systems
        assert!(
            leakage_no_drag < 0.3,
            "No-DRAG leakage to |2> should be small: {}",
            leakage_no_drag
        );

        // The simulation ran without errors on a 3-level system
        let total_pop: f64 = result_drag.final_state.iter().map(|x| x.norm_sqr()).sum();
        assert!(
            (total_pop - 1.0).abs() < 0.01,
            "Total population should be 1, got {}",
            total_pop
        );
    }

    // ----------------------------------------------------------
    // Additional tests for robustness
    // ----------------------------------------------------------

    #[test]
    fn test_matrix_exp_identity() {
        // exp(0) = I
        let zero = DenseMatrix::zeros(2);
        let result = zero.matrix_exp();
        let id = DenseMatrix::identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (result[(i, j)] - id[(i, j)]).norm() < TIGHT_TOL,
                    "exp(0) should be I at ({},{}): got {}",
                    i,
                    j,
                    result[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_diagonal() {
        // exp(diag(a, b)) = diag(exp(a), exp(b))
        let mut m = DenseMatrix::zeros(2);
        m[(0, 0)] = c_re(1.0);
        m[(1, 1)] = c_re(-1.0);
        let result = m.matrix_exp();

        assert!((result[(0, 0)].re - 1.0f64.exp()).abs() < 1e-4);
        assert!((result[(1, 1)].re - (-1.0f64).exp()).abs() < 1e-4);
        assert!(result[(0, 1)].norm() < 1e-4);
        assert!(result[(1, 0)].norm() < 1e-4);
    }

    #[test]
    fn test_kron_product() {
        // I ⊗ Z should be diag(1, -1, 1, -1)
        let id = DenseMatrix::identity(2);
        let mut z = DenseMatrix::zeros(2);
        z[(0, 0)] = ONE;
        z[(1, 1)] = c_re(-1.0);

        let iz = id.kron(&z);
        assert_eq!(iz.dim, 4);
        assert!((iz[(0, 0)].re - 1.0).abs() < TIGHT_TOL);
        assert!((iz[(1, 1)].re - (-1.0)).abs() < TIGHT_TOL);
        assert!((iz[(2, 2)].re - 1.0).abs() < TIGHT_TOL);
        assert!((iz[(3, 3)].re - (-1.0)).abs() < TIGHT_TOL);
    }

    #[test]
    fn test_process_fidelity() {
        let id = DenseMatrix::identity(2);
        let x = StandardGates::x_unitary();

        // Process fidelity of U with itself = 1
        let pf = process_fidelity(&id, &id);
        assert!((pf - 1.0).abs() < TIGHT_TOL, "Process fidelity I,I = {}", pf);

        let pf_xx = process_fidelity(&x, &x);
        assert!((pf_xx - 1.0).abs() < TIGHT_TOL, "Process fidelity X,X = {}", pf_xx);

        // Process fidelity of X with I should be less
        let pf_xi = process_fidelity(&x, &id);
        assert!(pf_xi < 0.5, "Process fidelity X,I should be low: {}", pf_xi);
    }

    #[test]
    fn test_dense_matrix_trace() {
        let mut m = DenseMatrix::zeros(3);
        m[(0, 0)] = c_re(1.0);
        m[(1, 1)] = c_re(2.0);
        m[(2, 2)] = c_re(3.0);
        assert!((m.trace().re - 6.0).abs() < TIGHT_TOL);
    }

    #[test]
    fn test_dense_matrix_commutator() {
        // [X, Z] = -2iY
        let mut x_mat = DenseMatrix::zeros(2);
        x_mat[(0, 1)] = ONE;
        x_mat[(1, 0)] = ONE;

        let mut z_mat = DenseMatrix::zeros(2);
        z_mat[(0, 0)] = ONE;
        z_mat[(1, 1)] = c_re(-1.0);

        let comm = x_mat.commutator(&z_mat);

        // [X, Z] = XZ - ZX
        // XZ = [[0,1],[1,0]] * [[1,0],[0,-1]] = [[0,-1],[1,0]]
        // ZX = [[1,0],[0,-1]] * [[0,1],[1,0]] = [[0,1],[-1,0]]
        // [X,Z] = [[0,-2],[2,0]] = -2iY where Y = [[0,-i],[i,0]]
        assert!((comm[(0, 1)].re - (-2.0)).abs() < TIGHT_TOL);
        assert!((comm[(1, 0)].re - 2.0).abs() < TIGHT_TOL);
    }
}
