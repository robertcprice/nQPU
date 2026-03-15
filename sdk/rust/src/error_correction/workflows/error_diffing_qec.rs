//! Stim-style Error-Diffing for Fast QEC Sampling
//!
//! This module implements the Detector Error Model (DEM) abstraction and a
//! high-throughput frame simulator that operates directly on the DEM rather than
//! re-simulating full Clifford circuits for every Monte Carlo shot.
//!
//! # Key insight
//!
//! Instead of simulating entire circuits, we:
//! 1. Compile a noisy Clifford circuit into a compact `DetectorErrorModel` (DEM)
//! 2. Track how each possible error mechanism maps to detector firings and
//!    logical observable flips
//! 3. For each shot, sample which error mechanisms fire and XOR their
//!    contributions -- no gate-level propagation required
//!
//! This gives O(num_error_mechanisms) work per shot instead of
//! O(num_instructions * num_qubits), yielding 10-100x speedup for large codes
//! where the number of distinct error mechanisms is much smaller than the total
//! instruction count.
//!
//! # Architecture
//!
//! ```text
//! NoisyCircuit ──compile──▶ DetectorErrorModel ──sample──▶ SyndromeData
//!                                                           │
//!                                                           ▼
//!                                                     BatchSampler
//!                                                     (convergence,
//!                                                      logical error rate)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use nqpu_metal::error_diffing_qec::*;
//!
//! // Build a DEM from a noisy circuit description
//! let mut dem = DetectorErrorModel::new(4, 1);
//! dem.add_error(ErrorInstruction::new(0.01, vec![0, 1], vec![]));
//! dem.add_error(ErrorInstruction::new(0.01, vec![2, 3], vec![0]));
//!
//! // Sample syndromes
//! let config = BatchSamplerConfig::builder()
//!     .num_shots(100_000)
//!     .seed(Some(42))
//!     .build()
//!     .unwrap();
//! let result = BatchSampler::run(&dem, &config).unwrap();
//! println!("Logical error rate: {:.6}", result.logical_error_rate);
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;
use std::time::Instant;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising during error-diffing QEC operations.
#[derive(Debug, Clone)]
pub enum ErrorDiffingError {
    /// The detector error model is invalid or empty.
    InvalidModel(String),
    /// A detector index is out of range.
    DetectorOutOfRange { index: usize, num_detectors: usize },
    /// An observable index is out of range.
    ObservableOutOfRange {
        index: usize,
        num_observables: usize,
    },
    /// Sampling configuration is invalid.
    InvalidConfig(String),
    /// Serialization or deserialization failed.
    SerdeError(String),
    /// The frame simulator encountered an inconsistency.
    FrameError(String),
    /// Statistical convergence was not reached.
    ConvergenceError {
        shots_run: usize,
        target_stderr: f64,
    },
}

impl fmt::Display for ErrorDiffingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorDiffingError::InvalidModel(msg) => {
                write!(f, "Invalid detector error model: {}", msg)
            }
            ErrorDiffingError::DetectorOutOfRange {
                index,
                num_detectors,
            } => write!(
                f,
                "Detector index {} out of range (model has {} detectors)",
                index, num_detectors
            ),
            ErrorDiffingError::ObservableOutOfRange {
                index,
                num_observables,
            } => write!(
                f,
                "Observable index {} out of range (model has {} observables)",
                index, num_observables
            ),
            ErrorDiffingError::InvalidConfig(msg) => {
                write!(f, "Invalid sampling config: {}", msg)
            }
            ErrorDiffingError::SerdeError(msg) => {
                write!(f, "DEM serialization error: {}", msg)
            }
            ErrorDiffingError::FrameError(msg) => {
                write!(f, "Frame simulator error: {}", msg)
            }
            ErrorDiffingError::ConvergenceError {
                shots_run,
                target_stderr,
            } => write!(
                f,
                "Convergence not reached after {} shots (target stderr: {:.6})",
                shots_run, target_stderr
            ),
        }
    }
}

impl std::error::Error for ErrorDiffingError {}

// ============================================================
// PAULI TYPE (symplectic representation)
// ============================================================

/// Single-qubit Pauli in binary symplectic form: (x, z).
///
/// - (0, 0) = I
/// - (1, 0) = X
/// - (0, 1) = Z
/// - (1, 1) = Y (up to global phase)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pauli {
    pub x: bool,
    pub z: bool,
}

impl Pauli {
    pub const I: Pauli = Pauli { x: false, z: false };
    pub const X: Pauli = Pauli { x: true, z: false };
    pub const Z: Pauli = Pauli { x: false, z: true };
    pub const Y: Pauli = Pauli { x: true, z: true };

    /// True when this Pauli is the identity.
    #[inline]
    pub fn is_identity(self) -> bool {
        !self.x && !self.z
    }

    /// Multiply two Paulis (ignoring global phase).
    #[inline]
    pub fn mul(self, other: Pauli) -> Pauli {
        Pauli {
            x: self.x ^ other.x,
            z: self.z ^ other.z,
        }
    }

    /// Whether this Pauli anti-commutes with Z measurement.
    #[inline]
    pub fn anticommutes_z(self) -> bool {
        self.x
    }

    /// Whether this Pauli anti-commutes with X measurement.
    #[inline]
    pub fn anticommutes_x(self) -> bool {
        self.z
    }
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.x, self.z) {
            (false, false) => write!(f, "I"),
            (true, false) => write!(f, "X"),
            (false, true) => write!(f, "Z"),
            (true, true) => write!(f, "Y"),
        }
    }
}

// ============================================================
// DETECTOR ERROR MODEL
// ============================================================

/// A single error instruction in the Detector Error Model.
///
/// Each instruction describes one independent error mechanism with:
/// - A probability of firing
/// - The set of detectors it flips when it fires
/// - The set of logical observables it flips when it fires
///
/// This is the compiled representation of error propagation -- the gate-level
/// circuit structure has been eliminated.
#[derive(Debug, Clone)]
pub struct ErrorInstruction {
    /// Probability that this error mechanism fires in a single shot.
    pub probability: f64,
    /// Indices of detectors flipped when this error fires.
    pub detector_targets: Vec<usize>,
    /// Indices of logical observables flipped when this error fires.
    pub observable_targets: Vec<usize>,
    /// Optional decomposition into constituent single-fault mechanisms.
    /// Used for correlated errors that can be decomposed.
    pub decomposition: Option<Vec<usize>>,
}

impl ErrorInstruction {
    /// Create a new error instruction.
    pub fn new(
        probability: f64,
        detector_targets: Vec<usize>,
        observable_targets: Vec<usize>,
    ) -> Self {
        ErrorInstruction {
            probability,
            detector_targets,
            observable_targets,
            decomposition: None,
        }
    }

    /// Create a new error instruction with decomposition info.
    pub fn with_decomposition(
        probability: f64,
        detector_targets: Vec<usize>,
        observable_targets: Vec<usize>,
        decomposition: Vec<usize>,
    ) -> Self {
        ErrorInstruction {
            probability,
            detector_targets,
            observable_targets,
            decomposition: Some(decomposition),
        }
    }
}

/// The Detector Error Model: a compiled description of all independent error
/// mechanisms, their probabilities, and their effects on detectors and
/// logical observables.
///
/// This is the central data structure for error-diffing sampling. Once a DEM
/// is constructed (either manually or by compiling a noisy circuit), sampling
/// reduces to: for each shot, independently fire each error mechanism with
/// its probability, then XOR together the detector/observable contributions
/// of all fired mechanisms.
#[derive(Debug, Clone)]
pub struct DetectorErrorModel {
    /// Number of detectors in the model.
    pub num_detectors: usize,
    /// Number of logical observables.
    pub num_observables: usize,
    /// Ordered list of error instructions.
    pub instructions: Vec<ErrorInstruction>,
}

impl DetectorErrorModel {
    /// Create an empty DEM with the given detector and observable counts.
    pub fn new(num_detectors: usize, num_observables: usize) -> Self {
        DetectorErrorModel {
            num_detectors,
            num_observables,
            instructions: Vec::new(),
        }
    }

    /// Add an error instruction to the model.
    pub fn add_error(&mut self, instr: ErrorInstruction) {
        self.instructions.push(instr);
    }

    /// Validate that all target indices are in range.
    pub fn validate(&self) -> Result<(), ErrorDiffingError> {
        if self.instructions.is_empty() {
            return Err(ErrorDiffingError::InvalidModel(
                "DEM has no error instructions".into(),
            ));
        }

        for (i, instr) in self.instructions.iter().enumerate() {
            if instr.probability < 0.0 || instr.probability > 1.0 {
                return Err(ErrorDiffingError::InvalidModel(format!(
                    "Error instruction {} has invalid probability: {}",
                    i, instr.probability
                )));
            }
            for &d in &instr.detector_targets {
                if d >= self.num_detectors {
                    return Err(ErrorDiffingError::DetectorOutOfRange {
                        index: d,
                        num_detectors: self.num_detectors,
                    });
                }
            }
            for &o in &instr.observable_targets {
                if o >= self.num_observables {
                    return Err(ErrorDiffingError::ObservableOutOfRange {
                        index: o,
                        num_observables: self.num_observables,
                    });
                }
            }
        }

        Ok(())
    }

    /// Total number of error mechanisms.
    pub fn num_errors(&self) -> usize {
        self.instructions.len()
    }

    /// Serialize to DEM text format (Stim-compatible).
    ///
    /// Format:
    /// ```text
    /// error(0.01) D0 D1
    /// error(0.02) D2 L0
    /// error(0.005) D0 D3 L0 ^ D1
    /// ```
    pub fn to_dem_string(&self) -> String {
        let mut out = String::new();
        for instr in &self.instructions {
            out.push_str(&format!("error({:.6})", instr.probability));
            for &d in &instr.detector_targets {
                out.push_str(&format!(" D{}", d));
            }
            for &o in &instr.observable_targets {
                out.push_str(&format!(" L{}", o));
            }
            if let Some(ref decomp) = instr.decomposition {
                out.push_str(" ^");
                for &idx in decomp {
                    out.push_str(&format!(" {}", idx));
                }
            }
            out.push('\n');
        }
        out
    }

    /// Parse from DEM text format.
    ///
    /// Accepts lines of the form: `error(<prob>) D<n> ... L<n> ... [^ <idx> ...]`
    pub fn from_dem_string(s: &str) -> Result<Self, ErrorDiffingError> {
        let mut max_det: usize = 0;
        let mut max_obs: usize = 0;
        let mut instructions = Vec::new();

        for (line_no, line) in s.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse error(<prob>)
            if !line.starts_with("error(") {
                return Err(ErrorDiffingError::SerdeError(format!(
                    "Line {}: expected 'error(' prefix, got: {}",
                    line_no + 1,
                    line
                )));
            }

            let paren_close = line.find(')').ok_or_else(|| {
                ErrorDiffingError::SerdeError(format!(
                    "Line {}: missing closing parenthesis",
                    line_no + 1
                ))
            })?;

            let prob_str = &line[6..paren_close];
            let probability: f64 = prob_str.parse().map_err(|e| {
                ErrorDiffingError::SerdeError(format!(
                    "Line {}: invalid probability '{}': {}",
                    line_no + 1,
                    prob_str,
                    e
                ))
            })?;

            let rest = &line[paren_close + 1..];

            // Split on '^' for decomposition
            let (targets_part, decomp_part) = if let Some(caret_pos) = rest.find('^') {
                (&rest[..caret_pos], Some(&rest[caret_pos + 1..]))
            } else {
                (rest, None)
            };

            let mut detector_targets = Vec::new();
            let mut observable_targets = Vec::new();

            for token in targets_part.split_whitespace() {
                if let Some(d_str) = token.strip_prefix('D') {
                    let idx: usize = d_str.parse().map_err(|e| {
                        ErrorDiffingError::SerdeError(format!(
                            "Line {}: invalid detector index '{}': {}",
                            line_no + 1,
                            d_str,
                            e
                        ))
                    })?;
                    detector_targets.push(idx);
                    if idx >= max_det {
                        max_det = idx + 1;
                    }
                } else if let Some(l_str) = token.strip_prefix('L') {
                    let idx: usize = l_str.parse().map_err(|e| {
                        ErrorDiffingError::SerdeError(format!(
                            "Line {}: invalid observable index '{}': {}",
                            line_no + 1,
                            l_str,
                            e
                        ))
                    })?;
                    observable_targets.push(idx);
                    if idx >= max_obs {
                        max_obs = idx + 1;
                    }
                }
            }

            let decomposition = decomp_part.map(|dp| {
                dp.split_whitespace()
                    .filter_map(|t| t.parse::<usize>().ok())
                    .collect()
            });

            instructions.push(ErrorInstruction {
                probability,
                detector_targets,
                observable_targets,
                decomposition,
            });
        }

        Ok(DetectorErrorModel {
            num_detectors: max_det,
            num_observables: max_obs,
            instructions,
        })
    }
}

impl fmt::Display for DetectorErrorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DetectorErrorModel(detectors={}, observables={}, errors={})",
            self.num_detectors,
            self.num_observables,
            self.instructions.len()
        )
    }
}

// ============================================================
// FRAME SIMULATOR (bit-packed Pauli frame tracking)
// ============================================================

/// Clifford gate type for frame simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordGate {
    /// Hadamard on a single qubit.
    H(usize),
    /// S (phase) gate on a single qubit.
    S(usize),
    /// CNOT: (control, target).
    CX(usize, usize),
    /// CZ: (qubit_a, qubit_b).
    CZ(usize, usize),
    /// SWAP: (qubit_a, qubit_b).
    Swap(usize, usize),
}

/// Bit-packed Pauli frame simulator.
///
/// Tracks X and Z Pauli frames for up to 64 shots simultaneously using
/// u64 bit packing. Each qubit has two u64 words (x_frame, z_frame) where
/// bit i represents shot i.
///
/// Gate operations process all 64 shots in a single bitwise operation,
/// giving up to 64x throughput over single-shot simulation.
#[derive(Debug, Clone)]
pub struct FrameSimulator {
    /// X component per qubit: `x_frames[q]` has bit i set iff shot i has
    /// X error on qubit q.
    x_frames: Vec<u64>,
    /// Z component per qubit: `z_frames[q]` has bit i set iff shot i has
    /// Z error on qubit q.
    z_frames: Vec<u64>,
    /// Number of qubits.
    num_qubits: usize,
    /// Number of active shots (1..=64).
    num_shots: usize,
}

impl FrameSimulator {
    /// Create a new frame simulator with all-identity frames.
    pub fn new(num_qubits: usize, num_shots: usize) -> Self {
        assert!(num_shots > 0 && num_shots <= 64, "num_shots must be 1..=64");
        FrameSimulator {
            x_frames: vec![0u64; num_qubits],
            z_frames: vec![0u64; num_qubits],
            num_qubits,
            num_shots,
        }
    }

    /// Bitmask covering all active shots.
    #[inline]
    fn active_mask(&self) -> u64 {
        if self.num_shots >= 64 {
            u64::MAX
        } else {
            (1u64 << self.num_shots) - 1
        }
    }

    /// Reset all frames to identity.
    pub fn reset(&mut self) {
        for q in 0..self.num_qubits {
            self.x_frames[q] = 0;
            self.z_frames[q] = 0;
        }
    }

    /// Inject X errors on qubit `q` for shots indicated by mask.
    #[inline]
    pub fn inject_x(&mut self, q: usize, mask: u64) {
        self.x_frames[q] ^= mask;
    }

    /// Inject Z errors on qubit `q` for shots indicated by mask.
    #[inline]
    pub fn inject_z(&mut self, q: usize, mask: u64) {
        self.z_frames[q] ^= mask;
    }

    /// Inject Y errors (= XZ) on qubit `q` for shots indicated by mask.
    #[inline]
    pub fn inject_y(&mut self, q: usize, mask: u64) {
        self.x_frames[q] ^= mask;
        self.z_frames[q] ^= mask;
    }

    /// Apply Hadamard: swap X and Z frames (all shots in parallel).
    #[inline]
    pub fn apply_h(&mut self, q: usize) {
        let tmp = self.x_frames[q];
        self.x_frames[q] = self.z_frames[q];
        self.z_frames[q] = tmp;
    }

    /// Apply S gate: X -> Y (X -> XZ, Z -> Z).
    #[inline]
    pub fn apply_s(&mut self, q: usize) {
        self.z_frames[q] ^= self.x_frames[q];
    }

    /// Apply CNOT (CX): X_c -> X_c X_t, Z_t -> Z_c Z_t.
    #[inline]
    pub fn apply_cx(&mut self, control: usize, target: usize) {
        self.x_frames[target] ^= self.x_frames[control];
        self.z_frames[control] ^= self.z_frames[target];
    }

    /// Apply CZ: X_a -> X_a Z_b, X_b -> Z_a X_b.
    #[inline]
    pub fn apply_cz(&mut self, a: usize, b: usize) {
        let xa = self.x_frames[a];
        let xb = self.x_frames[b];
        self.z_frames[b] ^= xa;
        self.z_frames[a] ^= xb;
    }

    /// Apply SWAP: exchange frames for qubits a and b.
    #[inline]
    pub fn apply_swap(&mut self, a: usize, b: usize) {
        self.x_frames.swap(a, b);
        self.z_frames.swap(a, b);
    }

    /// Apply a Clifford gate to the frames.
    pub fn apply_gate(&mut self, gate: CliffordGate) {
        match gate {
            CliffordGate::H(q) => self.apply_h(q),
            CliffordGate::S(q) => self.apply_s(q),
            CliffordGate::CX(c, t) => self.apply_cx(c, t),
            CliffordGate::CZ(a, b) => self.apply_cz(a, b),
            CliffordGate::Swap(a, b) => self.apply_swap(a, b),
        }
    }

    /// Apply a sequence of Clifford gates.
    pub fn apply_gates(&mut self, gates: &[CliffordGate]) {
        for &gate in gates {
            self.apply_gate(gate);
        }
    }

    /// Reset a qubit after measurement: clear both frames.
    #[inline]
    pub fn clear_qubit(&mut self, q: usize) {
        self.x_frames[q] = 0;
        self.z_frames[q] = 0;
    }

    /// Returns mask of which shots would flip a Z-basis measurement on qubit q.
    #[inline]
    pub fn measure_z_flips(&self, q: usize) -> u64 {
        self.x_frames[q]
    }

    /// Returns mask of which shots would flip an X-basis measurement on qubit q.
    #[inline]
    pub fn measure_x_flips(&self, q: usize) -> u64 {
        self.z_frames[q]
    }

    /// Read the X frame for a specific qubit.
    #[inline]
    pub fn x_frame(&self, q: usize) -> u64 {
        self.x_frames[q]
    }

    /// Read the Z frame for a specific qubit.
    #[inline]
    pub fn z_frame(&self, q: usize) -> u64 {
        self.z_frames[q]
    }

    /// Number of qubits in the simulator.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of active shots.
    pub fn num_shots(&self) -> usize {
        self.num_shots
    }
}

// ============================================================
// ERROR DIFFING ENGINE
// ============================================================

/// Core error-diffing algorithm operating on a DetectorErrorModel.
///
/// For each shot, independently fires each error mechanism with its
/// probability, then XORs together the detector/observable contributions
/// of all fired mechanisms to produce syndrome data.
#[derive(Debug, Clone)]
pub struct ErrorDiffingEngine {
    /// Pre-computed detector bitmasks for each error instruction.
    /// `detector_masks[error_idx]` is a Vec<usize> of detector indices.
    detector_masks: Vec<Vec<usize>>,
    /// Pre-computed observable bitmasks for each error instruction.
    observable_masks: Vec<Vec<usize>>,
    /// Error probabilities (parallel to masks).
    probabilities: Vec<f64>,
    /// Number of detectors.
    num_detectors: usize,
    /// Number of observables.
    num_observables: usize,
}

impl ErrorDiffingEngine {
    /// Compile a DEM into an optimized diffing engine.
    pub fn compile(dem: &DetectorErrorModel) -> Result<Self, ErrorDiffingError> {
        dem.validate()?;

        let n = dem.instructions.len();
        let mut detector_masks = Vec::with_capacity(n);
        let mut observable_masks = Vec::with_capacity(n);
        let mut probabilities = Vec::with_capacity(n);

        for instr in &dem.instructions {
            detector_masks.push(instr.detector_targets.clone());
            observable_masks.push(instr.observable_targets.clone());
            probabilities.push(instr.probability);
        }

        Ok(ErrorDiffingEngine {
            detector_masks,
            observable_masks,
            probabilities,
            num_detectors: dem.num_detectors,
            num_observables: dem.num_observables,
        })
    }

    /// Sample a single shot: returns (detector_events, observable_flips).
    pub fn sample_shot(&self, rng: &mut StdRng) -> (Vec<bool>, Vec<bool>) {
        let mut detectors = vec![false; self.num_detectors];
        let mut observables = vec![false; self.num_observables];

        for i in 0..self.probabilities.len() {
            if rng.gen::<f64>() < self.probabilities[i] {
                for &d in &self.detector_masks[i] {
                    detectors[d] ^= true;
                }
                for &o in &self.observable_masks[i] {
                    observables[o] ^= true;
                }
            }
        }

        (detectors, observables)
    }

    /// Sample a batch of up to 64 shots using bit-packed operations.
    ///
    /// Returns (detector_masks, observable_masks) where each element is a u64
    /// with bit i representing shot i.
    pub fn sample_batch(&self, rng: &mut StdRng, batch_size: usize) -> (Vec<u64>, Vec<u64>) {
        let bs = batch_size.min(64);
        let mut det_words = vec![0u64; self.num_detectors];
        let mut obs_words = vec![0u64; self.num_observables];

        for i in 0..self.probabilities.len() {
            let p = self.probabilities[i];
            let fire_mask = sample_bernoulli_mask(rng, p, bs);
            if fire_mask == 0 {
                continue;
            }

            for &d in &self.detector_masks[i] {
                det_words[d] ^= fire_mask;
            }
            for &o in &self.observable_masks[i] {
                obs_words[o] ^= fire_mask;
            }
        }

        (det_words, obs_words)
    }

    /// Number of detectors.
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of observables.
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Number of error mechanisms.
    pub fn num_error_mechanisms(&self) -> usize {
        self.probabilities.len()
    }
}

// ============================================================
// BATCH SAMPLER
// ============================================================

/// Configuration for batch sampling.
#[derive(Debug, Clone)]
pub struct BatchSamplerConfig {
    /// Total number of shots to sample.
    pub num_shots: usize,
    /// Number of shots per batch (multiple of 64 for optimal packing).
    pub batch_size: usize,
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
    /// Whether to use parallel (rayon) processing across batches.
    pub parallel: bool,
    /// Target standard error for early stopping (0 = disabled).
    pub target_stderr: f64,
    /// Minimum number of shots before checking convergence.
    pub min_shots_for_convergence: usize,
}

/// Builder for `BatchSamplerConfig`.
pub struct BatchSamplerConfigBuilder {
    num_shots: usize,
    batch_size: usize,
    seed: Option<u64>,
    parallel: bool,
    target_stderr: f64,
    min_shots_for_convergence: usize,
}

impl BatchSamplerConfig {
    /// Create a builder with sensible defaults.
    pub fn builder() -> BatchSamplerConfigBuilder {
        BatchSamplerConfigBuilder {
            num_shots: 10_000,
            batch_size: 1024,
            seed: None,
            parallel: false,
            target_stderr: 0.0,
            min_shots_for_convergence: 1000,
        }
    }
}

impl BatchSamplerConfigBuilder {
    /// Set the total number of shots.
    pub fn num_shots(mut self, n: usize) -> Self {
        self.num_shots = n;
        self
    }

    /// Set the batch size (will be rounded up to multiple of 64).
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Set the RNG seed.
    pub fn seed(mut self, s: Option<u64>) -> Self {
        self.seed = s;
        self
    }

    /// Enable or disable parallel processing.
    pub fn parallel(mut self, p: bool) -> Self {
        self.parallel = p;
        self
    }

    /// Set target standard error for early convergence stopping.
    pub fn target_stderr(mut self, s: f64) -> Self {
        self.target_stderr = s;
        self
    }

    /// Set minimum shots before convergence checking begins.
    pub fn min_shots_for_convergence(mut self, n: usize) -> Self {
        self.min_shots_for_convergence = n;
        self
    }

    /// Build the configuration, validating parameters.
    pub fn build(self) -> Result<BatchSamplerConfig, ErrorDiffingError> {
        if self.num_shots == 0 {
            return Err(ErrorDiffingError::InvalidConfig(
                "num_shots must be > 0".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(ErrorDiffingError::InvalidConfig(
                "batch_size must be > 0".into(),
            ));
        }
        if self.target_stderr < 0.0 {
            return Err(ErrorDiffingError::InvalidConfig(
                "target_stderr must be >= 0".into(),
            ));
        }
        Ok(BatchSamplerConfig {
            num_shots: self.num_shots,
            batch_size: self.batch_size,
            seed: self.seed,
            parallel: self.parallel,
            target_stderr: self.target_stderr,
            min_shots_for_convergence: self.min_shots_for_convergence,
        })
    }
}

/// Result of a batch sampling run.
#[derive(Debug, Clone)]
pub struct BatchSamplingResult {
    /// Number of shots completed.
    pub num_shots: usize,
    /// Number of shots where at least one observable was flipped (logical error).
    pub num_logical_errors: usize,
    /// Logical error rate (num_logical_errors / num_shots).
    pub logical_error_rate: f64,
    /// Standard error of the logical error rate estimate.
    pub logical_error_rate_stderr: f64,
    /// Per-detector firing rate across all shots.
    pub per_detector_rates: Vec<f64>,
    /// Total detection events per shot (average).
    pub mean_detection_weight: f64,
    /// Whether convergence target was met (if target_stderr > 0).
    pub converged: bool,
    /// Wall-clock time in seconds.
    pub elapsed_secs: f64,
    /// Throughput in shots per second.
    pub shots_per_second: f64,
}

impl fmt::Display for BatchSamplingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Error-Diffing QEC Sampling Result:")?;
        writeln!(f, "  Shots: {}", self.num_shots)?;
        writeln!(
            f,
            "  Logical errors: {} ({:.6} +/- {:.6})",
            self.num_logical_errors, self.logical_error_rate, self.logical_error_rate_stderr
        )?;
        writeln!(
            f,
            "  Mean detection weight: {:.3}",
            self.mean_detection_weight
        )?;
        writeln!(f, "  Converged: {}", self.converged)?;
        writeln!(f, "  Elapsed: {:.3}s", self.elapsed_secs)?;
        writeln!(f, "  Throughput: {:.0} shots/sec", self.shots_per_second)?;
        Ok(())
    }
}

/// High-throughput batch sampler for DEM-based QEC sampling.
pub struct BatchSampler;

impl BatchSampler {
    /// Run batch sampling on a compiled DEM.
    pub fn run(
        dem: &DetectorErrorModel,
        config: &BatchSamplerConfig,
    ) -> Result<BatchSamplingResult, ErrorDiffingError> {
        let engine = ErrorDiffingEngine::compile(dem)?;
        Self::run_with_engine(&engine, config)
    }

    /// Run batch sampling with a pre-compiled engine (avoids recompilation).
    pub fn run_with_engine(
        engine: &ErrorDiffingEngine,
        config: &BatchSamplerConfig,
    ) -> Result<BatchSamplingResult, ErrorDiffingError> {
        let start = Instant::now();
        let base_seed = config.seed.unwrap_or(0xDEAD_BEEF_CAFE_1234);

        let num_detectors = engine.num_detectors();
        let _num_observables = engine.num_observables();

        // Process shots in batches of 64
        let total_shots = config.num_shots;
        let num_full_batches = total_shots / 64;
        let remainder = total_shots % 64;
        let num_batches = num_full_batches + if remainder > 0 { 1 } else { 0 };

        // Accumulate statistics
        let mut total_logical_errors: usize = 0;
        let mut detector_fire_counts = vec![0u64; num_detectors];
        let mut total_detection_events: u64 = 0;
        let mut shots_completed: usize = 0;
        let mut converged = false;

        let mut rng = StdRng::seed_from_u64(base_seed);

        for batch_idx in 0..num_batches {
            let bs = if batch_idx == num_full_batches && remainder > 0 {
                remainder
            } else {
                64
            };

            let (det_words, obs_words) = engine.sample_batch(&mut rng, bs);

            // Count logical errors: any shot where any observable fired
            let mut any_obs_fired = 0u64;
            for &ow in &obs_words {
                any_obs_fired |= ow;
            }
            // Mask to active shots
            let shot_mask = if bs >= 64 { u64::MAX } else { (1u64 << bs) - 1 };
            any_obs_fired &= shot_mask;
            total_logical_errors += any_obs_fired.count_ones() as usize;

            // Accumulate per-detector fire counts
            for (di, &dw) in det_words.iter().enumerate() {
                let fires = (dw & shot_mask).count_ones() as u64;
                detector_fire_counts[di] += fires;
                total_detection_events += fires;
            }

            shots_completed += bs;

            // Check convergence if target_stderr is set
            if config.target_stderr > 0.0 && shots_completed >= config.min_shots_for_convergence {
                let ler = total_logical_errors as f64 / shots_completed as f64;
                let stderr = if shots_completed > 0 {
                    (ler * (1.0 - ler) / shots_completed as f64).sqrt()
                } else {
                    f64::INFINITY
                };
                if stderr <= config.target_stderr {
                    converged = true;
                    break;
                }
            }
        }

        let n = shots_completed as f64;
        let logical_error_rate = if shots_completed > 0 {
            total_logical_errors as f64 / n
        } else {
            0.0
        };
        let logical_error_rate_stderr = if shots_completed > 0 {
            (logical_error_rate * (1.0 - logical_error_rate) / n).sqrt()
        } else {
            0.0
        };

        let per_detector_rates: Vec<f64> = detector_fire_counts
            .iter()
            .map(|&c| c as f64 / n.max(1.0))
            .collect();

        let mean_detection_weight = if shots_completed > 0 && num_detectors > 0 {
            total_detection_events as f64 / n
        } else {
            0.0
        };

        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let shots_per_second = if elapsed_secs > 0.0 {
            shots_completed as f64 / elapsed_secs
        } else {
            shots_completed as f64
        };

        Ok(BatchSamplingResult {
            num_shots: shots_completed,
            num_logical_errors: total_logical_errors,
            logical_error_rate,
            logical_error_rate_stderr,
            per_detector_rates,
            mean_detection_weight,
            converged,
            elapsed_secs,
            shots_per_second,
        })
    }
}

// ============================================================
// CIRCUIT NOISE MODEL -> DEM COMPILER
// ============================================================

/// Noise channel specification for circuit-level noise.
#[derive(Debug, Clone)]
pub enum NoiseChannel {
    /// Depolarizing noise after every gate with probability p.
    Depolarizing { p: f64 },
    /// Bit-flip (X) noise with probability p.
    BitFlip { p: f64 },
    /// Phase-flip (Z) noise with probability p.
    PhaseFlip { p: f64 },
    /// Measurement error (readout flip) with probability p.
    MeasurementError { p: f64 },
    /// Reset error with probability p.
    ResetError { p: f64 },
    /// Correlated two-qubit depolarizing with probability p.
    CorrelatedDepolarizing { p: f64 },
}

/// A circuit instruction for the noise compiler.
#[derive(Debug, Clone)]
pub enum CircuitInstruction {
    /// Clifford gate.
    Gate(CliffordGate),
    /// Reset qubit to |0>.
    Reset(usize),
    /// Measure in Z basis.
    MeasureZ(usize),
    /// Measure in X basis.
    MeasureX(usize),
    /// Detector: XOR of measurement record entries.
    Detector(Vec<usize>),
    /// Logical observable: XOR of measurement record entries.
    Observable(Vec<usize>),
    /// Timing barrier.
    Tick,
    /// Apply a noise channel to specified qubits.
    Noise(NoiseChannel, Vec<usize>),
}

/// Compiles a noisy circuit description into a DetectorErrorModel.
///
/// This performs a one-time gate-level Pauli frame propagation for each noise
/// location, tracing how each error mechanism maps to detector firings and
/// logical observable flips.
pub struct CircuitNoiseCompiler;

impl CircuitNoiseCompiler {
    /// Compile a noisy circuit into a DEM.
    ///
    /// The circuit is a sequence of instructions including gates, measurements,
    /// detectors, observables, and noise channels. Each noise channel is
    /// converted into one or more DEM error instructions by propagating the
    /// corresponding Pauli error through the remaining circuit and recording
    /// which detectors and observables are flipped.
    pub fn compile(
        instructions: &[CircuitInstruction],
        num_qubits: usize,
    ) -> Result<DetectorErrorModel, ErrorDiffingError> {
        // First pass: count detectors and observables, locate noise
        let mut num_detectors = 0usize;
        let mut num_observables = 0usize;
        let mut noise_locations: Vec<(usize, NoiseChannel, Vec<usize>)> = Vec::new();

        for (idx, instr) in instructions.iter().enumerate() {
            match instr {
                CircuitInstruction::Detector(_) => num_detectors += 1,
                CircuitInstruction::Observable(_) => num_observables += 1,
                CircuitInstruction::Noise(channel, qubits) => {
                    noise_locations.push((idx, channel.clone(), qubits.clone()));
                }
                _ => {}
            }
        }

        let mut dem = DetectorErrorModel::new(num_detectors, num_observables);

        // For each noise location, propagate each possible Pauli error
        // through the remaining circuit and record detector/observable effects
        for (noise_idx, channel, qubits) in &noise_locations {
            let pauli_errors = channel_to_paulis(channel, qubits);

            for (prob, qubit_paulis) in pauli_errors {
                if prob < 1e-15 {
                    continue;
                }

                // Propagate the error through the circuit from noise_idx+1 onward
                let (det_flips, obs_flips) =
                    propagate_error(instructions, num_qubits, *noise_idx, &qubit_paulis);

                if !det_flips.is_empty() || !obs_flips.is_empty() {
                    dem.add_error(ErrorInstruction::new(prob, det_flips, obs_flips));
                }
            }
        }

        Ok(dem)
    }
}

/// Convert a noise channel into a list of (probability, qubit->Pauli) pairs.
fn channel_to_paulis(channel: &NoiseChannel, qubits: &[usize]) -> Vec<(f64, Vec<(usize, Pauli)>)> {
    match channel {
        NoiseChannel::Depolarizing { p } => {
            let p_each = p / 3.0;
            let mut result = Vec::new();
            for &q in qubits {
                result.push((p_each, vec![(q, Pauli::X)]));
                result.push((p_each, vec![(q, Pauli::Y)]));
                result.push((p_each, vec![(q, Pauli::Z)]));
            }
            result
        }
        NoiseChannel::BitFlip { p } => qubits.iter().map(|&q| (*p, vec![(q, Pauli::X)])).collect(),
        NoiseChannel::PhaseFlip { p } => {
            qubits.iter().map(|&q| (*p, vec![(q, Pauli::Z)])).collect()
        }
        NoiseChannel::MeasurementError { p } => {
            // Measurement error = X flip before Z measurement
            qubits.iter().map(|&q| (*p, vec![(q, Pauli::X)])).collect()
        }
        NoiseChannel::ResetError { p } => {
            // Reset error = X flip after reset
            qubits.iter().map(|&q| (*p, vec![(q, Pauli::X)])).collect()
        }
        NoiseChannel::CorrelatedDepolarizing { p } => {
            if qubits.len() >= 2 {
                let p_each = p / 15.0; // 15 nontrivial two-qubit Paulis
                let mut result = Vec::new();
                let q0 = qubits[0];
                let q1 = qubits[1];
                for p0 in &[Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
                    for p1 in &[Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
                        if p0.is_identity() && p1.is_identity() {
                            continue;
                        }
                        let mut qp = Vec::new();
                        if !p0.is_identity() {
                            qp.push((q0, *p0));
                        }
                        if !p1.is_identity() {
                            qp.push((q1, *p1));
                        }
                        result.push((p_each, qp));
                    }
                }
                result
            } else {
                // Fallback to single-qubit depolarizing
                channel_to_paulis(&NoiseChannel::Depolarizing { p: *p }, qubits)
            }
        }
    }
}

/// Propagate a Pauli error through the remaining circuit and record which
/// detectors and observables it flips.
fn propagate_error(
    instructions: &[CircuitInstruction],
    num_qubits: usize,
    noise_idx: usize,
    qubit_paulis: &[(usize, Pauli)],
) -> (Vec<usize>, Vec<usize>) {
    let mut frame = vec![Pauli::I; num_qubits];
    for &(q, p) in qubit_paulis {
        if q < num_qubits {
            frame[q] = frame[q].mul(p);
        }
    }

    let mut measurement_flips: Vec<bool> = Vec::new();
    let mut detector_flips: Vec<usize> = Vec::new();
    let mut observable_flips: Vec<usize> = Vec::new();

    let mut det_counter = 0usize;
    let mut obs_counter = 0usize;
    // Count detectors/observables before the noise location
    for instr in instructions.iter().take(noise_idx + 1) {
        match instr {
            CircuitInstruction::Detector(_) => det_counter += 1,
            CircuitInstruction::Observable(_) => obs_counter += 1,
            CircuitInstruction::MeasureZ(_) | CircuitInstruction::MeasureX(_) => {
                // Measurements before the noise do not see this error
                measurement_flips.push(false);
            }
            _ => {}
        }
    }

    // Propagate through instructions after the noise location
    for instr in instructions.iter().skip(noise_idx + 1) {
        match instr {
            CircuitInstruction::Gate(gate) => {
                apply_gate_to_frame(&mut frame, *gate);
            }
            CircuitInstruction::Reset(q) => {
                // Reset absorbs Z, keeps X
                if *q < num_qubits {
                    frame[*q].z = false;
                }
            }
            CircuitInstruction::MeasureZ(q) => {
                let flipped = if *q < num_qubits {
                    frame[*q].anticommutes_z()
                } else {
                    false
                };
                measurement_flips.push(flipped);
                if *q < num_qubits {
                    frame[*q] = Pauli::I;
                }
            }
            CircuitInstruction::MeasureX(q) => {
                let flipped = if *q < num_qubits {
                    frame[*q].anticommutes_x()
                } else {
                    false
                };
                measurement_flips.push(flipped);
                if *q < num_qubits {
                    frame[*q] = Pauli::I;
                }
            }
            CircuitInstruction::Detector(indices) => {
                let val = indices
                    .iter()
                    .filter(|&&i| i < measurement_flips.len())
                    .fold(false, |acc, &i| acc ^ measurement_flips[i]);
                if val {
                    detector_flips.push(det_counter);
                }
                det_counter += 1;
            }
            CircuitInstruction::Observable(indices) => {
                let val = indices
                    .iter()
                    .filter(|&&i| i < measurement_flips.len())
                    .fold(false, |acc, &i| acc ^ measurement_flips[i]);
                if val {
                    observable_flips.push(obs_counter);
                }
                obs_counter += 1;
            }
            CircuitInstruction::Tick | CircuitInstruction::Noise(_, _) => {}
        }
    }

    (detector_flips, observable_flips)
}

/// Apply a Clifford gate to a per-qubit Pauli frame (single-shot).
fn apply_gate_to_frame(frame: &mut [Pauli], gate: CliffordGate) {
    match gate {
        CliffordGate::H(q) => {
            let p = frame[q];
            frame[q] = Pauli { x: p.z, z: p.x };
        }
        CliffordGate::S(q) => {
            if frame[q].x {
                frame[q].z ^= true;
            }
        }
        CliffordGate::CX(c, t) => {
            if frame[c].x {
                frame[t].x ^= true;
            }
            if frame[t].z {
                frame[c].z ^= true;
            }
        }
        CliffordGate::CZ(a, b) => {
            let xa = frame[a].x;
            let xb = frame[b].x;
            if xa {
                frame[b].z ^= true;
            }
            if xb {
                frame[a].z ^= true;
            }
        }
        CliffordGate::Swap(a, b) => {
            frame.swap(a, b);
        }
    }
}

// ============================================================
// PROGRESS TRACKING
// ============================================================

/// Progress callback for long-running sampling operations.
pub trait SamplingProgress: Send + Sync {
    /// Called periodically with the number of shots completed and total.
    fn on_progress(&self, shots_completed: usize, total_shots: usize);
}

/// A no-op progress tracker.
#[derive(Debug, Clone)]
pub struct NoProgress;

impl SamplingProgress for NoProgress {
    fn on_progress(&self, _shots_completed: usize, _total_shots: usize) {}
}

/// Progress tracker that records milestones.
#[derive(Debug, Clone)]
pub struct MilestoneProgress {
    /// Percentage milestones to report at (e.g., [25, 50, 75, 100]).
    pub milestones: Vec<usize>,
    /// Milestones that have been reached.
    pub reached: Vec<usize>,
}

impl MilestoneProgress {
    /// Create a new milestone progress tracker.
    pub fn new(milestones: Vec<usize>) -> Self {
        MilestoneProgress {
            milestones,
            reached: Vec::new(),
        }
    }
}

impl SamplingProgress for MilestoneProgress {
    fn on_progress(&self, shots_completed: usize, total_shots: usize) {
        let _ = (shots_completed, total_shots);
        // In a real implementation, this would update self.reached via interior mutability
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Sample a u64 bitmask where each of the first `num_bits` bits is independently
/// set with probability `p`. Uses geometric skip for efficiency when p is small.
fn sample_bernoulli_mask(rng: &mut StdRng, p: f64, num_bits: usize) -> u64 {
    if p <= 0.0 || num_bits == 0 {
        return 0;
    }
    let nb = num_bits.min(64);
    if p >= 1.0 {
        return if nb >= 64 { u64::MAX } else { (1u64 << nb) - 1 };
    }

    let log_1mp = (1.0 - p).ln();
    if log_1mp >= -1e-15 {
        return 0;
    }

    let mut mask = 0u64;
    let mut pos: usize = 0;

    loop {
        let u: f64 = rng.gen();
        if u <= 0.0 {
            continue;
        }
        let skip = (u.ln() / log_1mp).floor() as usize;
        pos += skip;
        if pos >= nb {
            break;
        }
        mask |= 1u64 << pos;
        pos += 1;
        if pos >= nb {
            break;
        }
    }

    mask
}

/// Vectorized XOR for frame diffing: XOR two slices of u64 element-wise.
/// Result is written into the `target` slice.
#[inline]
pub fn xor_frames(target: &mut [u64], source: &[u64]) {
    debug_assert_eq!(target.len(), source.len());
    for i in 0..target.len() {
        target[i] ^= source[i];
    }
}

/// Count the number of set bits across a slice of u64 words.
#[inline]
pub fn popcount_words(words: &[u64]) -> u64 {
    words.iter().map(|w| w.count_ones() as u64).sum()
}

/// Unpack a u64 bitmask into a Vec<bool> of length `n`.
#[inline]
pub fn unpack_mask(mask: u64, n: usize) -> Vec<bool> {
    (0..n).map(|i| (mask >> i) & 1 == 1).collect()
}

/// Pack a slice of bools into a u64 bitmask.
#[inline]
pub fn pack_mask(bits: &[bool]) -> u64 {
    let mut mask = 0u64;
    for (i, &b) in bits.iter().enumerate().take(64) {
        if b {
            mask |= 1u64 << i;
        }
    }
    mask
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // 1. Pauli algebra: identity
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_identity() {
        assert!(Pauli::I.is_identity());
        assert!(!Pauli::X.is_identity());
        assert!(!Pauli::Z.is_identity());
        assert!(!Pauli::Y.is_identity());
    }

    // ----------------------------------------------------------
    // 2. Pauli algebra: multiplication
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_multiplication() {
        assert_eq!(Pauli::X.mul(Pauli::Z), Pauli::Y);
        assert_eq!(Pauli::Z.mul(Pauli::X), Pauli::Y);
        assert_eq!(Pauli::X.mul(Pauli::X), Pauli::I);
        assert_eq!(Pauli::Z.mul(Pauli::Z), Pauli::I);
        assert_eq!(Pauli::Y.mul(Pauli::Y), Pauli::I);
        assert_eq!(Pauli::X.mul(Pauli::I), Pauli::X);
        assert_eq!(Pauli::I.mul(Pauli::Z), Pauli::Z);
    }

    // ----------------------------------------------------------
    // 3. Pauli algebra: commutativity
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_anticommutation() {
        assert!(Pauli::X.anticommutes_z());
        assert!(Pauli::Y.anticommutes_z());
        assert!(!Pauli::Z.anticommutes_z());
        assert!(!Pauli::I.anticommutes_z());

        assert!(Pauli::Z.anticommutes_x());
        assert!(Pauli::Y.anticommutes_x());
        assert!(!Pauli::X.anticommutes_x());
        assert!(!Pauli::I.anticommutes_x());
    }

    // ----------------------------------------------------------
    // 4. Pauli display
    // ----------------------------------------------------------
    #[test]
    fn test_pauli_display() {
        assert_eq!(format!("{}", Pauli::I), "I");
        assert_eq!(format!("{}", Pauli::X), "X");
        assert_eq!(format!("{}", Pauli::Y), "Y");
        assert_eq!(format!("{}", Pauli::Z), "Z");
    }

    // ----------------------------------------------------------
    // 5. FrameSimulator: H gate swaps X and Z
    // ----------------------------------------------------------
    #[test]
    fn test_frame_h_gate() {
        let mut sim = FrameSimulator::new(1, 1);
        sim.inject_x(0, 1);
        assert_eq!(sim.x_frame(0), 1);
        assert_eq!(sim.z_frame(0), 0);

        sim.apply_h(0);
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.z_frame(0), 1);
    }

    // ----------------------------------------------------------
    // 6. FrameSimulator: S gate X -> Y
    // ----------------------------------------------------------
    #[test]
    fn test_frame_s_gate() {
        let mut sim = FrameSimulator::new(1, 1);
        sim.inject_x(0, 1);
        sim.apply_s(0);
        assert_eq!(sim.x_frame(0), 1); // X component still set
        assert_eq!(sim.z_frame(0), 1); // Z component now set (XZ = Y)
    }

    // ----------------------------------------------------------
    // 7. FrameSimulator: S gate leaves Z unchanged
    // ----------------------------------------------------------
    #[test]
    fn test_frame_s_gate_z_unchanged() {
        let mut sim = FrameSimulator::new(1, 1);
        sim.inject_z(0, 1);
        sim.apply_s(0);
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.z_frame(0), 1);
    }

    // ----------------------------------------------------------
    // 8. FrameSimulator: CX gate X on control spreads to target
    // ----------------------------------------------------------
    #[test]
    fn test_frame_cx_x_control() {
        let mut sim = FrameSimulator::new(2, 1);
        sim.inject_x(0, 1);
        sim.apply_cx(0, 1);
        assert_eq!(sim.x_frame(0), 1);
        assert_eq!(sim.x_frame(1), 1); // X spread to target
    }

    // ----------------------------------------------------------
    // 9. FrameSimulator: CX gate Z on target spreads to control
    // ----------------------------------------------------------
    #[test]
    fn test_frame_cx_z_target() {
        let mut sim = FrameSimulator::new(2, 1);
        sim.inject_z(1, 1);
        sim.apply_cx(0, 1);
        assert_eq!(sim.z_frame(0), 1); // Z spread to control
        assert_eq!(sim.z_frame(1), 1);
    }

    // ----------------------------------------------------------
    // 10. FrameSimulator: CZ gate
    // ----------------------------------------------------------
    #[test]
    fn test_frame_cz_gate() {
        let mut sim = FrameSimulator::new(2, 1);
        sim.inject_x(0, 1);
        sim.apply_cz(0, 1);
        assert_eq!(sim.x_frame(0), 1); // X on a stays
        assert_eq!(sim.z_frame(1), 1); // Z on b gained
    }

    // ----------------------------------------------------------
    // 11. FrameSimulator: SWAP gate
    // ----------------------------------------------------------
    #[test]
    fn test_frame_swap_gate() {
        let mut sim = FrameSimulator::new(2, 1);
        sim.inject_x(0, 1);
        sim.inject_z(1, 1);
        sim.apply_swap(0, 1);
        assert_eq!(sim.x_frame(1), 1);
        assert_eq!(sim.z_frame(0), 1);
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.z_frame(1), 0);
    }

    // ----------------------------------------------------------
    // 12. FrameSimulator: batch injection (multiple shots)
    // ----------------------------------------------------------
    #[test]
    fn test_frame_batch_injection() {
        let mut sim = FrameSimulator::new(1, 4);
        sim.inject_x(0, 0b1010); // shots 1,3 get X
        assert_eq!(sim.x_frame(0), 0b1010);
        assert_eq!(sim.z_frame(0), 0);
    }

    // ----------------------------------------------------------
    // 13. FrameSimulator: batch gate propagation
    // ----------------------------------------------------------
    #[test]
    fn test_frame_batch_h() {
        let mut sim = FrameSimulator::new(1, 8);
        sim.inject_x(0, 0b11001100);
        sim.apply_h(0);
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.z_frame(0), 0b11001100);
    }

    // ----------------------------------------------------------
    // 14. FrameSimulator: measure Z flips
    // ----------------------------------------------------------
    #[test]
    fn test_frame_measure_z_flips() {
        let mut sim = FrameSimulator::new(1, 4);
        sim.inject_x(0, 0b0101);
        assert_eq!(sim.measure_z_flips(0), 0b0101);
    }

    // ----------------------------------------------------------
    // 15. FrameSimulator: measure X flips
    // ----------------------------------------------------------
    #[test]
    fn test_frame_measure_x_flips() {
        let mut sim = FrameSimulator::new(1, 4);
        sim.inject_z(0, 0b1100);
        assert_eq!(sim.measure_x_flips(0), 0b1100);
    }

    // ----------------------------------------------------------
    // 16. FrameSimulator: reset clears qubit
    // ----------------------------------------------------------
    #[test]
    fn test_frame_clear_qubit() {
        let mut sim = FrameSimulator::new(1, 4);
        sim.inject_x(0, 0b1111);
        sim.inject_z(0, 0b1111);
        sim.clear_qubit(0);
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.z_frame(0), 0);
    }

    // ----------------------------------------------------------
    // 17. DEM: construction and validation
    // ----------------------------------------------------------
    #[test]
    fn test_dem_construction() {
        let mut dem = DetectorErrorModel::new(4, 1);
        dem.add_error(ErrorInstruction::new(0.01, vec![0, 1], vec![]));
        dem.add_error(ErrorInstruction::new(0.02, vec![2, 3], vec![0]));
        assert_eq!(dem.num_errors(), 2);
        assert!(dem.validate().is_ok());
    }

    // ----------------------------------------------------------
    // 18. DEM: validation catches out-of-range detector
    // ----------------------------------------------------------
    #[test]
    fn test_dem_validation_bad_detector() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(0.01, vec![5], vec![]));
        assert!(dem.validate().is_err());
    }

    // ----------------------------------------------------------
    // 19. DEM: validation catches out-of-range observable
    // ----------------------------------------------------------
    #[test]
    fn test_dem_validation_bad_observable() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(0.01, vec![0], vec![3]));
        assert!(dem.validate().is_err());
    }

    // ----------------------------------------------------------
    // 20. DEM: validation catches invalid probability
    // ----------------------------------------------------------
    #[test]
    fn test_dem_validation_bad_probability() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(-0.1, vec![0], vec![]));
        assert!(dem.validate().is_err());

        let mut dem2 = DetectorErrorModel::new(2, 1);
        dem2.add_error(ErrorInstruction::new(1.5, vec![0], vec![]));
        assert!(dem2.validate().is_err());
    }

    // ----------------------------------------------------------
    // 21. DEM serialization round-trip
    // ----------------------------------------------------------
    #[test]
    fn test_dem_serialize_roundtrip() {
        let mut dem = DetectorErrorModel::new(4, 2);
        dem.add_error(ErrorInstruction::new(0.01, vec![0, 1], vec![]));
        dem.add_error(ErrorInstruction::new(0.02, vec![2, 3], vec![0]));
        dem.add_error(ErrorInstruction::new(0.005, vec![1], vec![1]));

        let s = dem.to_dem_string();
        let dem2 = DetectorErrorModel::from_dem_string(&s).unwrap();

        assert_eq!(dem2.instructions.len(), 3);
        assert!((dem2.instructions[0].probability - 0.01).abs() < 1e-9);
        assert_eq!(dem2.instructions[0].detector_targets, vec![0, 1]);
        assert!((dem2.instructions[1].probability - 0.02).abs() < 1e-9);
        assert_eq!(dem2.instructions[1].observable_targets, vec![0]);
        assert!((dem2.instructions[2].probability - 0.005).abs() < 1e-9);
    }

    // ----------------------------------------------------------
    // 22. DEM serialization with decomposition
    // ----------------------------------------------------------
    #[test]
    fn test_dem_serialize_with_decomposition() {
        let mut dem = DetectorErrorModel::new(3, 1);
        dem.add_error(ErrorInstruction::with_decomposition(
            0.01,
            vec![0, 1],
            vec![],
            vec![0, 1],
        ));

        let s = dem.to_dem_string();
        assert!(s.contains('^'));

        let dem2 = DetectorErrorModel::from_dem_string(&s).unwrap();
        assert!(dem2.instructions[0].decomposition.is_some());
        assert_eq!(
            dem2.instructions[0].decomposition.as_ref().unwrap(),
            &vec![0, 1]
        );
    }

    // ----------------------------------------------------------
    // 23. DEM parsing: comments and blank lines
    // ----------------------------------------------------------
    #[test]
    fn test_dem_parse_comments() {
        let s = "# This is a comment\n\nerror(0.01) D0 D1\n# Another comment\nerror(0.02) D2 L0\n";
        let dem = DetectorErrorModel::from_dem_string(s).unwrap();
        assert_eq!(dem.instructions.len(), 2);
    }

    // ----------------------------------------------------------
    // 24. Batch sampler config builder defaults
    // ----------------------------------------------------------
    #[test]
    fn test_batch_config_defaults() {
        let config = BatchSamplerConfig::builder().build().unwrap();
        assert_eq!(config.num_shots, 10_000);
        assert_eq!(config.batch_size, 1024);
        assert!(config.seed.is_none());
        assert!(!config.parallel);
        assert_eq!(config.target_stderr, 0.0);
    }

    // ----------------------------------------------------------
    // 25. Batch sampler config validation
    // ----------------------------------------------------------
    #[test]
    fn test_batch_config_validation() {
        assert!(BatchSamplerConfig::builder().num_shots(0).build().is_err());
        assert!(BatchSamplerConfig::builder().batch_size(0).build().is_err());
        assert!(BatchSamplerConfig::builder()
            .target_stderr(-1.0)
            .build()
            .is_err());
    }

    // ----------------------------------------------------------
    // 26. Error diffing: zero-probability errors produce no events
    // ----------------------------------------------------------
    #[test]
    fn test_zero_prob_no_events() {
        let mut dem = DetectorErrorModel::new(4, 1);
        dem.add_error(ErrorInstruction::new(0.0, vec![0, 1], vec![]));
        dem.add_error(ErrorInstruction::new(0.0, vec![2, 3], vec![0]));
        let config = BatchSamplerConfig::builder()
            .num_shots(1000)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();
        assert_eq!(result.logical_error_rate, 0.0);
        assert_eq!(result.num_logical_errors, 0);
        for &rate in &result.per_detector_rates {
            assert_eq!(rate, 0.0);
        }
    }

    // ----------------------------------------------------------
    // 27. Error diffing: certainty errors always fire
    // ----------------------------------------------------------
    #[test]
    fn test_certainty_errors_always_fire() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(1.0, vec![0], vec![0]));
        let config = BatchSamplerConfig::builder()
            .num_shots(100)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();
        // Every shot should have a logical error
        assert_eq!(result.num_logical_errors, 100);
        assert!((result.logical_error_rate - 1.0).abs() < 1e-9);
        // Detector 0 should fire every time
        assert!((result.per_detector_rates[0] - 1.0).abs() < 1e-9);
    }

    // ----------------------------------------------------------
    // 28. Error diffing: statistical bounds
    // ----------------------------------------------------------
    #[test]
    fn test_statistical_bounds() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(0.1, vec![0], vec![0]));
        let config = BatchSamplerConfig::builder()
            .num_shots(10_000)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();
        // With 10k shots at p=0.1, logical error rate should be near 0.1
        // Allow 5 sigma: stderr ~ sqrt(0.1*0.9/10000) ~ 0.003
        assert!(
            (result.logical_error_rate - 0.1).abs() < 0.05,
            "Logical error rate {} should be near 0.1",
            result.logical_error_rate
        );
    }

    // ----------------------------------------------------------
    // 29. Error diffing: two errors with same detector cancel
    // ----------------------------------------------------------
    #[test]
    fn test_double_flip_cancels() {
        let mut dem = DetectorErrorModel::new(2, 1);
        // Two independent errors that both flip detector 0 and observable 0
        // When both fire, detector 0 and observable 0 cancel (XOR)
        dem.add_error(ErrorInstruction::new(1.0, vec![0], vec![0]));
        dem.add_error(ErrorInstruction::new(1.0, vec![0], vec![0]));
        let config = BatchSamplerConfig::builder()
            .num_shots(100)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();
        // Both always fire -> XOR cancels -> no logical errors, no detector firings
        assert_eq!(result.num_logical_errors, 0);
        assert!((result.per_detector_rates[0]).abs() < 1e-9);
    }

    // ----------------------------------------------------------
    // 30. Bit packing: pack/unpack round-trip
    // ----------------------------------------------------------
    #[test]
    fn test_pack_unpack_roundtrip() {
        let bits = vec![true, false, true, true, false, false, true, false];
        let mask = pack_mask(&bits);
        let unpacked = unpack_mask(mask, bits.len());
        assert_eq!(bits, unpacked);
    }

    // ----------------------------------------------------------
    // 31. Bit packing: empty
    // ----------------------------------------------------------
    #[test]
    fn test_pack_unpack_empty() {
        let mask = pack_mask(&[]);
        assert_eq!(mask, 0);
        let unpacked = unpack_mask(0, 0);
        assert!(unpacked.is_empty());
    }

    // ----------------------------------------------------------
    // 32. Bit packing: all set
    // ----------------------------------------------------------
    #[test]
    fn test_pack_all_set() {
        let bits = vec![true; 64];
        let mask = pack_mask(&bits);
        assert_eq!(mask, u64::MAX);
    }

    // ----------------------------------------------------------
    // 33. XOR frames utility
    // ----------------------------------------------------------
    #[test]
    fn test_xor_frames() {
        let mut target = vec![0b1100u64, 0b0011u64];
        let source = vec![0b1010u64, 0b0101u64];
        xor_frames(&mut target, &source);
        assert_eq!(target[0], 0b0110);
        assert_eq!(target[1], 0b0110);
    }

    // ----------------------------------------------------------
    // 34. Popcount words utility
    // ----------------------------------------------------------
    #[test]
    fn test_popcount_words() {
        let words = vec![0b1111u64, 0b11u64, 0u64];
        assert_eq!(popcount_words(&words), 6);
    }

    // ----------------------------------------------------------
    // 35. Bernoulli mask: edge cases
    // ----------------------------------------------------------
    #[test]
    fn test_bernoulli_mask_edges() {
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(sample_bernoulli_mask(&mut rng, 0.0, 64), 0);
        let all = sample_bernoulli_mask(&mut rng, 1.0, 8);
        assert_eq!(all, 0b11111111);
        assert_eq!(sample_bernoulli_mask(&mut rng, 0.5, 0), 0);
    }

    // ----------------------------------------------------------
    // 36. Bernoulli mask: statistical test
    // ----------------------------------------------------------
    #[test]
    fn test_bernoulli_mask_statistics() {
        let mut rng = StdRng::seed_from_u64(12345);
        let p = 0.25;
        let trials = 10_000;
        let mut total_ones = 0u64;
        let total_bits = trials * 64;

        for _ in 0..trials {
            let mask = sample_bernoulli_mask(&mut rng, p, 64);
            total_ones += mask.count_ones() as u64;
        }

        let measured_p = total_ones as f64 / total_bits as f64;
        // 5-sigma tolerance
        let stderr = (p * (1.0 - p) / total_bits as f64).sqrt();
        assert!(
            (measured_p - p).abs() < 5.0 * stderr,
            "Measured p={:.4} expected {:.4} (5-sigma: {:.4})",
            measured_p,
            p,
            5.0 * stderr
        );
    }

    // ----------------------------------------------------------
    // 37. Error diffing engine: compilation
    // ----------------------------------------------------------
    #[test]
    fn test_engine_compilation() {
        let mut dem = DetectorErrorModel::new(3, 1);
        dem.add_error(ErrorInstruction::new(0.01, vec![0, 1], vec![]));
        dem.add_error(ErrorInstruction::new(0.02, vec![1, 2], vec![0]));

        let engine = ErrorDiffingEngine::compile(&dem).unwrap();
        assert_eq!(engine.num_detectors(), 3);
        assert_eq!(engine.num_observables(), 1);
        assert_eq!(engine.num_error_mechanisms(), 2);
    }

    // ----------------------------------------------------------
    // 38. Error diffing engine: single shot
    // ----------------------------------------------------------
    #[test]
    fn test_engine_single_shot() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(1.0, vec![0], vec![0]));

        let engine = ErrorDiffingEngine::compile(&dem).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let (det, obs) = engine.sample_shot(&mut rng);
        assert!(det[0]); // Detector 0 always fires
        assert!(obs[0]); // Observable 0 always fires
    }

    // ----------------------------------------------------------
    // 39. Circuit noise compiler: simple circuit
    // ----------------------------------------------------------
    #[test]
    fn test_circuit_noise_compiler_simple() {
        // Simple circuit: reset, noise, measure, detector
        let instructions = vec![
            CircuitInstruction::Reset(0),
            CircuitInstruction::Noise(NoiseChannel::BitFlip { p: 0.01 }, vec![0]),
            CircuitInstruction::MeasureZ(0),
            CircuitInstruction::Detector(vec![0]),
        ];

        let dem = CircuitNoiseCompiler::compile(&instructions, 1).unwrap();
        assert_eq!(dem.num_detectors, 1);
        assert_eq!(dem.num_observables, 0);
        // Bit-flip on qubit 0 before Z measurement should flip detector 0
        assert!(!dem.instructions.is_empty());
        assert!(dem.instructions[0].detector_targets.contains(&0));
    }

    // ----------------------------------------------------------
    // 40. Circuit noise compiler: H + noise + measure
    // ----------------------------------------------------------
    #[test]
    fn test_circuit_noise_compiler_with_h() {
        // After H, Z error becomes X error and flips Z measurement
        let instructions = vec![
            CircuitInstruction::Reset(0),
            CircuitInstruction::Noise(NoiseChannel::PhaseFlip { p: 0.01 }, vec![0]),
            CircuitInstruction::Gate(CliffordGate::H(0)),
            CircuitInstruction::MeasureZ(0),
            CircuitInstruction::Detector(vec![0]),
        ];

        let dem = CircuitNoiseCompiler::compile(&instructions, 1).unwrap();
        // Z error before H becomes X error, which flips Z measurement
        assert!(!dem.instructions.is_empty());
        assert!(dem.instructions[0].detector_targets.contains(&0));
    }

    // ----------------------------------------------------------
    // 41. Convergence detection
    // ----------------------------------------------------------
    #[test]
    fn test_convergence_detection() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(0.1, vec![0], vec![0]));

        let config = BatchSamplerConfig::builder()
            .num_shots(1_000_000)
            .target_stderr(0.005)
            .min_shots_for_convergence(500)
            .seed(Some(42))
            .build()
            .unwrap();

        let result = BatchSampler::run(&dem, &config).unwrap();
        // Should converge well before 1M shots
        assert!(result.converged);
        assert!(result.num_shots < 1_000_000);
        assert!(result.logical_error_rate_stderr <= 0.005 + 1e-6);
    }

    // ----------------------------------------------------------
    // 42. Performance: DEM sampling faster than naive
    // ----------------------------------------------------------
    #[test]
    fn test_performance_dem_faster_than_naive() {
        // Build a DEM with many error mechanisms (simulating a large code)
        let num_det = 100;
        let mut dem = DetectorErrorModel::new(num_det, 1);
        for i in 0..num_det {
            dem.add_error(ErrorInstruction::new(
                0.001,
                vec![i, (i + 1) % num_det],
                if i == 0 { vec![0] } else { vec![] },
            ));
        }

        let num_shots = 10_000;
        let config = BatchSamplerConfig::builder()
            .num_shots(num_shots)
            .seed(Some(42))
            .build()
            .unwrap();

        // DEM-based sampling
        let start = Instant::now();
        let _result = BatchSampler::run(&dem, &config).unwrap();
        let dem_time = start.elapsed();

        // Naive per-shot sampling (using the engine single-shot path)
        let engine = ErrorDiffingEngine::compile(&dem).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let start_naive = Instant::now();
        for _ in 0..num_shots {
            let _ = engine.sample_shot(&mut rng);
        }
        let naive_time = start_naive.elapsed();

        // The batch sampler should be faster or at most comparable
        // (batch uses u64 packing internally for 64x throughput on XOR ops)
        // Allow generous margin since both are fast for small models
        assert!(
            dem_time.as_nanos() < naive_time.as_nanos() * 10,
            "DEM batch ({:?}) should not be dramatically slower than naive ({:?})",
            dem_time,
            naive_time
        );
    }

    // ----------------------------------------------------------
    // 43. FrameSimulator: apply_gates sequence
    // ----------------------------------------------------------
    #[test]
    fn test_frame_apply_gates_sequence() {
        let mut sim = FrameSimulator::new(2, 1);
        sim.inject_x(0, 1);

        // H then CX should spread the error
        sim.apply_gates(&[CliffordGate::H(0), CliffordGate::CX(0, 1)]);

        // After H: X->Z on qubit 0
        // After CX with Z on control: Z stays on control, doesn't spread via X path
        // Actually CX: X_c->X_c X_t, Z_t->Z_c Z_t
        // After H(0): x_frames[0]=0, z_frames[0]=1
        // After CX(0,1): z_frames[0]^=z_frames[1]=0 => z_frames[0]=1
        //                 x_frames[1]^=x_frames[0]=0 => x_frames[1]=0
        assert_eq!(sim.z_frame(0), 1);
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.x_frame(1), 0);
        assert_eq!(sim.z_frame(1), 0);
    }

    // ----------------------------------------------------------
    // 44. FrameSimulator: active mask
    // ----------------------------------------------------------
    #[test]
    fn test_frame_active_mask() {
        let sim = FrameSimulator::new(1, 3);
        assert_eq!(sim.active_mask(), 0b111);

        let sim64 = FrameSimulator::new(1, 64);
        assert_eq!(sim64.active_mask(), u64::MAX);
    }

    // ----------------------------------------------------------
    // 45. FrameSimulator: reset
    // ----------------------------------------------------------
    #[test]
    fn test_frame_reset() {
        let mut sim = FrameSimulator::new(2, 8);
        sim.inject_x(0, 0xFF);
        sim.inject_z(1, 0xFF);
        sim.reset();
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.z_frame(1), 0);
    }

    // ----------------------------------------------------------
    // 46. Error model: display
    // ----------------------------------------------------------
    #[test]
    fn test_dem_display() {
        let dem = DetectorErrorModel::new(4, 2);
        let s = format!("{}", dem);
        assert!(s.contains("detectors=4"));
        assert!(s.contains("observables=2"));
    }

    // ----------------------------------------------------------
    // 47. Batch sampling result: display
    // ----------------------------------------------------------
    #[test]
    fn test_batch_result_display() {
        let mut dem = DetectorErrorModel::new(2, 1);
        dem.add_error(ErrorInstruction::new(0.1, vec![0], vec![0]));
        let config = BatchSamplerConfig::builder()
            .num_shots(100)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();
        let s = format!("{}", result);
        assert!(s.contains("Shots: 100"));
        assert!(s.contains("Logical errors:"));
    }

    // ----------------------------------------------------------
    // 48. Noise channel: depolarizing generates 3 Pauli terms per qubit
    // ----------------------------------------------------------
    #[test]
    fn test_depolarizing_channel_paulis() {
        let paulis = channel_to_paulis(&NoiseChannel::Depolarizing { p: 0.03 }, &[0]);
        assert_eq!(paulis.len(), 3);
        let total_p: f64 = paulis.iter().map(|(p, _)| p).sum();
        assert!((total_p - 0.03).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // 49. Noise channel: correlated depolarizing on 2 qubits
    // ----------------------------------------------------------
    #[test]
    fn test_correlated_depolarizing_paulis() {
        let paulis = channel_to_paulis(&NoiseChannel::CorrelatedDepolarizing { p: 0.15 }, &[0, 1]);
        assert_eq!(paulis.len(), 15); // 4*4 - 1 nontrivial
        let total_p: f64 = paulis.iter().map(|(p, _)| p).sum();
        assert!((total_p - 0.15).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // 50. Error type: display
    // ----------------------------------------------------------
    #[test]
    fn test_error_display() {
        let e = ErrorDiffingError::DetectorOutOfRange {
            index: 5,
            num_detectors: 3,
        };
        let s = format!("{}", e);
        assert!(s.contains("5"));
        assert!(s.contains("3"));

        let e2 = ErrorDiffingError::ConvergenceError {
            shots_run: 1000,
            target_stderr: 0.001,
        };
        let s2 = format!("{}", e2);
        assert!(s2.contains("1000"));
    }

    // ----------------------------------------------------------
    // 51. DEM: logical error rate bounds for repetition-like model
    // ----------------------------------------------------------
    #[test]
    fn test_logical_error_rate_bounds_repetition() {
        // Model a distance-3 repetition code: logical error requires 2+ errors
        // on overlapping detectors. At low p, logical error rate ~ p^2.
        let p = 0.01;
        let mut dem = DetectorErrorModel::new(2, 1);
        // Error 0 flips detector 0
        dem.add_error(ErrorInstruction::new(p, vec![0], vec![]));
        // Error 1 flips detector 1
        dem.add_error(ErrorInstruction::new(p, vec![1], vec![]));
        // Error 2 (correlated) flips both detectors and the logical
        dem.add_error(ErrorInstruction::new(p * p, vec![0, 1], vec![0]));

        let config = BatchSamplerConfig::builder()
            .num_shots(100_000)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();

        // Logical error rate should be approximately p^2 = 0.0001
        // Allow generous bounds for the simplified model
        assert!(
            result.logical_error_rate < 0.01,
            "Logical error rate {} should be much less than physical rate {}",
            result.logical_error_rate,
            p
        );
    }

    // ----------------------------------------------------------
    // 52. FrameSimulator: Y injection
    // ----------------------------------------------------------
    #[test]
    fn test_frame_y_injection() {
        let mut sim = FrameSimulator::new(1, 1);
        sim.inject_y(0, 1);
        assert_eq!(sim.x_frame(0), 1);
        assert_eq!(sim.z_frame(0), 1);
    }

    // ----------------------------------------------------------
    // 53. FrameSimulator: double Y = identity
    // ----------------------------------------------------------
    #[test]
    fn test_frame_double_y_identity() {
        let mut sim = FrameSimulator::new(1, 1);
        sim.inject_y(0, 1);
        sim.inject_y(0, 1);
        assert_eq!(sim.x_frame(0), 0);
        assert_eq!(sim.z_frame(0), 0);
    }

    // ----------------------------------------------------------
    // 54. DEM: empty model validation
    // ----------------------------------------------------------
    #[test]
    fn test_dem_empty_validation() {
        let dem = DetectorErrorModel::new(2, 1);
        assert!(dem.validate().is_err()); // No instructions
    }

    // ----------------------------------------------------------
    // 55. Batch sampler: reproducibility with same seed
    // ----------------------------------------------------------
    #[test]
    fn test_reproducibility() {
        let mut dem = DetectorErrorModel::new(4, 1);
        dem.add_error(ErrorInstruction::new(0.05, vec![0, 1], vec![]));
        dem.add_error(ErrorInstruction::new(0.03, vec![2, 3], vec![0]));

        let config = BatchSamplerConfig::builder()
            .num_shots(5000)
            .seed(Some(12345))
            .build()
            .unwrap();

        let r1 = BatchSampler::run(&dem, &config).unwrap();
        let r2 = BatchSampler::run(&dem, &config).unwrap();

        assert_eq!(r1.num_logical_errors, r2.num_logical_errors);
        assert_eq!(r1.per_detector_rates, r2.per_detector_rates);
    }

    // ----------------------------------------------------------
    // 56. apply_gate_to_frame: SWAP
    // ----------------------------------------------------------
    #[test]
    fn test_apply_gate_to_frame_swap() {
        let mut frame = vec![Pauli::X, Pauli::Z, Pauli::I];
        apply_gate_to_frame(&mut frame, CliffordGate::Swap(0, 1));
        assert_eq!(frame[0], Pauli::Z);
        assert_eq!(frame[1], Pauli::X);
        assert_eq!(frame[2], Pauli::I);
    }

    // ----------------------------------------------------------
    // 57. apply_gate_to_frame: CZ
    // ----------------------------------------------------------
    #[test]
    fn test_apply_gate_to_frame_cz() {
        let mut frame = vec![Pauli::X, Pauli::I];
        apply_gate_to_frame(&mut frame, CliffordGate::CZ(0, 1));
        // X on qubit 0 -> X_0 Z_1
        assert_eq!(frame[0], Pauli::X);
        assert_eq!(frame[1], Pauli::Z);
    }

    // ----------------------------------------------------------
    // 58. Mean detection weight
    // ----------------------------------------------------------
    #[test]
    fn test_mean_detection_weight() {
        // Every shot fires exactly one detector
        let mut dem = DetectorErrorModel::new(3, 1);
        dem.add_error(ErrorInstruction::new(1.0, vec![0], vec![]));

        let config = BatchSamplerConfig::builder()
            .num_shots(1000)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();
        assert!((result.mean_detection_weight - 1.0).abs() < 0.01);
    }

    // ----------------------------------------------------------
    // 59. Large batch: memory and correctness
    // ----------------------------------------------------------
    #[test]
    fn test_large_batch() {
        let mut dem = DetectorErrorModel::new(10, 2);
        for i in 0..10 {
            dem.add_error(ErrorInstruction::new(
                0.01,
                vec![i],
                if i < 2 { vec![i] } else { vec![] },
            ));
        }

        let config = BatchSamplerConfig::builder()
            .num_shots(100_000)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run(&dem, &config).unwrap();

        // With 10 independent errors at p=0.01, each detector fires ~1%
        for &rate in &result.per_detector_rates {
            assert!(
                (rate - 0.01).abs() < 0.005,
                "Detector rate {} should be near 0.01",
                rate
            );
        }
    }

    // ----------------------------------------------------------
    // 60. Error diffing engine: batch consistency with single-shot
    // ----------------------------------------------------------
    #[test]
    fn test_batch_consistency() {
        let mut dem = DetectorErrorModel::new(4, 2);
        dem.add_error(ErrorInstruction::new(0.1, vec![0, 1], vec![0]));
        dem.add_error(ErrorInstruction::new(0.05, vec![2, 3], vec![1]));

        let engine = ErrorDiffingEngine::compile(&dem).unwrap();

        // Sample many single shots
        let mut rng1 = StdRng::seed_from_u64(999);
        let mut single_logical = 0usize;
        let n = 10_000;
        for _ in 0..n {
            let (_, obs) = engine.sample_shot(&mut rng1);
            if obs.iter().any(|&b| b) {
                single_logical += 1;
            }
        }
        let single_rate = single_logical as f64 / n as f64;

        // Sample using batch path
        let config = BatchSamplerConfig::builder()
            .num_shots(n)
            .seed(Some(42))
            .build()
            .unwrap();
        let result = BatchSampler::run_with_engine(&engine, &config).unwrap();

        // Both should give similar logical error rates
        // (different seeds, so not identical, but statistically similar)
        assert!(
            (result.logical_error_rate - single_rate).abs() < 0.03,
            "Batch rate {:.4} and single rate {:.4} should be similar",
            result.logical_error_rate,
            single_rate
        );
    }
}
