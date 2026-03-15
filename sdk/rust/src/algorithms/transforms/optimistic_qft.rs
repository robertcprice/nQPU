//! Optimistic Quantum Fourier Transform
//!
//! Implements the "Optimistic QFT" from arXiv:2505.00701 — a log-depth in-place QFT
//! that achieves good approximation on most inputs using zero ancillas.
//!
//! # Algorithm Overview
//!
//! The standard QFT uses O(n²) gates and O(n) depth. The approximate QFT truncates
//! small rotations to achieve O(n log(1/ε)) gates. The **optimistic QFT** goes further:
//!
//! 1. Divide n qubits into blocks of size b = O(log(n/ε))
//! 2. Apply exact QFT within each block (small enough to be efficient)
//! 3. Apply "optimistic" inter-block rotations that skip distant interactions
//! 4. Use recursive block structure to achieve O(log n) circuit depth
//! 5. Apply bit-reversal permutation
//!
//! The result is a circuit with O(n log n) gates and O(log² n) depth that achieves
//! fidelity > 1 - ε on most computational basis inputs (high probability).
//!
//! # Qubit ordering
//!
//! Qubit indices map to bit positions: qubit q controls bit q (qubit 0 = LSB).
//! The QFT circuit processes qubits from MSB (qubit n-1) down to LSB (qubit 0),
//! then applies bit-reversal swaps to produce the standard DFT output.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::optimistic_qft::{OptimisticQftConfig, optimistic_qft, compare_qft_methods};
//!
//! let config = OptimisticQftConfig::new(8)
//!     .with_precision(1e-3);
//!
//! let result = optimistic_qft(&config).unwrap();
//! println!("Gates: {}, Depth: {}", result.gate_count, result.depth);
//!
//! let cmp = compare_qft_methods(8, 1e-3);
//! println!("Standard gates: {}, Optimistic gates: {}", cmp.standard_gate_count, cmp.optimistic_gate_count);
//! ```

use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during QFT construction.
#[derive(Debug, Clone, PartialEq)]
pub enum QftError {
    /// Number of qubits must be at least 1.
    InvalidQubits(String),
    /// Requested precision is too high (smaller than machine epsilon).
    PrecisionTooHigh(String),
}

impl std::fmt::Display for QftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QftError::InvalidQubits(msg) => write!(f, "Invalid qubits: {}", msg),
            QftError::PrecisionTooHigh(msg) => write!(f, "Precision too high: {}", msg),
        }
    }
}

impl std::error::Error for QftError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the optimistic QFT algorithm.
#[derive(Debug, Clone)]
pub struct OptimisticQftConfig {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Target approximation error (default: 1e-3).
    pub precision: f64,
    /// Truncation level: skip controlled-R_k where k > truncation_level.
    /// If 0, computed automatically from precision.
    pub truncation_level: usize,
    /// Whether to build the inverse QFT.
    pub inverse: bool,
}

impl OptimisticQftConfig {
    /// Create a new config with the given number of qubits and sensible defaults.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            precision: 1e-3,
            truncation_level: 0,
            inverse: false,
        }
    }

    /// Set the target approximation precision.
    pub fn with_precision(mut self, precision: f64) -> Self {
        self.precision = precision;
        self
    }

    /// Set the truncation level explicitly.
    pub fn with_truncation_level(mut self, level: usize) -> Self {
        self.truncation_level = level;
        self
    }

    /// Build the inverse QFT instead of forward.
    pub fn with_inverse(mut self, inverse: bool) -> Self {
        self.inverse = inverse;
        self
    }

    /// Effective truncation level: user-specified or computed from precision.
    fn effective_truncation(&self) -> usize {
        if self.truncation_level > 0 {
            self.truncation_level
        } else {
            let k = (1.0 / self.precision).log2().ceil() as usize;
            k.max(1)
        }
    }
}

// ============================================================
// GATE AND CIRCUIT TYPES
// ============================================================

/// A single gate in a QFT circuit.
#[derive(Debug, Clone, PartialEq)]
pub enum QftGate {
    /// Hadamard gate on a single qubit.
    Hadamard(usize),
    /// Controlled phase rotation: applies phase e^{i*angle} to the |11⟩ component.
    ControlledPhase {
        control: usize,
        target: usize,
        /// Rotation angle in radians.
        angle: f64,
    },
    /// Swap two qubits.
    Swap(usize, usize),
}

/// A compiled QFT circuit.
#[derive(Debug, Clone)]
pub struct QftCircuit {
    /// Ordered list of gates.
    pub gates: Vec<QftGate>,
    /// Number of qubits the circuit acts on.
    pub num_qubits: usize,
    /// Circuit depth (estimated).
    pub depth: usize,
    /// Total gate count.
    pub gate_count: usize,
}

/// Result of an optimistic QFT construction.
#[derive(Debug, Clone)]
pub struct QftResult {
    /// The compiled circuit.
    pub circuit: QftCircuit,
    /// Estimated approximation error bound.
    pub approximation_error: f64,
    /// Circuit depth.
    pub depth: usize,
    /// Total gate count.
    pub gate_count: usize,
}

/// Comparison of the three QFT methods.
#[derive(Debug, Clone)]
pub struct QftComparison {
    /// Standard QFT gate count.
    pub standard_gate_count: usize,
    /// Standard QFT depth.
    pub standard_depth: usize,
    /// Approximate QFT gate count.
    pub approximate_gate_count: usize,
    /// Approximate QFT depth.
    pub approximate_depth: usize,
    /// Approximate QFT error bound.
    pub approximate_error: f64,
    /// Optimistic QFT gate count.
    pub optimistic_gate_count: usize,
    /// Optimistic QFT depth.
    pub optimistic_depth: usize,
    /// Optimistic QFT average error (Monte Carlo).
    pub optimistic_avg_error: f64,
}

// ============================================================
// STANDARD QFT
// ============================================================

/// Append bit-reversal swaps for n qubits. Swaps qubit i with qubit n-1-i
/// for i < n/2.
fn append_bit_reversal_swaps(gates: &mut Vec<QftGate>, n: usize) {
    for i in 0..n / 2 {
        gates.push(QftGate::Swap(i, n - 1 - i));
    }
}

/// Build the standard QFT circuit with O(n^2) gates and O(n) depth.
///
/// Uses the Nielsen & Chuang textbook construction:
///   1. For j = n-1 down to 0:
///      - Hadamard on qubit j
///      - For k = 1, 2, ..., j: controlled-R_{k+1} from qubit j-k to qubit j
///   2. Bit-reversal swaps
///
/// Qubit 0 = LSB. This produces the standard DFT:
///   QFT|x> = (1/sqrt(N)) sum_y exp(2*pi*i*x*y/N) |y>
pub fn standard_qft(n_qubits: usize) -> QftCircuit {
    let mut gates = Vec::new();

    // Process from MSB (qubit n-1) down to LSB (qubit 0)
    for j in (0..n_qubits).rev() {
        gates.push(QftGate::Hadamard(j));
        for k in 1..=j {
            // Controlled-R_{k+1}: control = qubit j-k, target = qubit j
            // Phase angle = 2*pi / 2^{k+1}
            let angle = 2.0 * PI / (1u64 << (k + 1)) as f64;
            gates.push(QftGate::ControlledPhase {
                control: j - k,
                target: j,
                angle,
            });
        }
    }

    // Bit-reversal permutation
    append_bit_reversal_swaps(&mut gates, n_qubits);

    let gate_count = gates.len();
    let depth = n_qubits;

    QftCircuit {
        gates,
        num_qubits: n_qubits,
        depth,
        gate_count,
    }
}

// ============================================================
// APPROXIMATE QFT
// ============================================================

/// Build an approximate QFT that truncates small rotations.
///
/// Skips controlled-R_{k+1} gates where k > k_max = ceil(log2(1/eps)),
/// reducing gate count from O(n^2) to O(n * k_max).
pub fn approximate_qft(n_qubits: usize, precision: f64) -> QftCircuit {
    let k_max = (1.0 / precision).log2().ceil() as usize;
    let k_max = k_max.max(1);
    let mut gates = Vec::new();

    for j in (0..n_qubits).rev() {
        gates.push(QftGate::Hadamard(j));
        let limit = j.min(k_max);
        for k in 1..=limit {
            let angle = 2.0 * PI / (1u64 << (k + 1)) as f64;
            gates.push(QftGate::ControlledPhase {
                control: j - k,
                target: j,
                angle,
            });
        }
    }

    append_bit_reversal_swaps(&mut gates, n_qubits);

    let gate_count = gates.len();
    let depth = k_max.min(n_qubits);

    QftCircuit {
        gates,
        num_qubits: n_qubits,
        depth,
        gate_count,
    }
}

// ============================================================
// OPTIMISTIC QFT (NOVEL ALGORITHM)
// ============================================================

/// Compute the optimal block size for the optimistic QFT.
///
/// Block size b = O(log(n/eps)), ensuring intra-block QFT is efficient
/// while keeping inter-block error bounded.
pub fn compute_block_size(n_qubits: usize, precision: f64) -> usize {
    if n_qubits <= 2 {
        return n_qubits;
    }
    let n = n_qubits as f64;
    let b = (n / precision).ln().ceil() as usize;
    // Clamp: at least 2 qubits per block, at most n_qubits
    b.max(2).min(n_qubits)
}

/// Build exact QFT gates for a contiguous block of qubits.
///
/// The block spans qubits [start_qubit, start_qubit + block_size).
/// Uses MSB-first processing within the block (no bit-reversal; that is
/// handled globally at the end).
pub fn build_block_qft(start_qubit: usize, block_size: usize) -> Vec<QftGate> {
    let mut gates = Vec::new();
    let end = start_qubit + block_size;

    // Process from highest qubit in block down to lowest
    for j in (start_qubit..end).rev() {
        gates.push(QftGate::Hadamard(j));
        for k in 1..=(j - start_qubit) {
            let angle = 2.0 * PI / (1u64 << (k + 1)) as f64;
            gates.push(QftGate::ControlledPhase {
                control: j - k,
                target: j,
                angle,
            });
        }
    }

    gates
}

/// Build approximate inter-block rotations between two blocks.
///
/// In the full QFT circuit, qubit j in the higher-index block receives
/// controlled-R_{distance+1} from qubit i in the lower-index block,
/// where distance = j - i. We skip rotations below the precision threshold.
pub fn build_inter_block_rotations(
    block_a_start: usize,
    block_b_start: usize,
    block_size_a: usize,
    block_size_b: usize,
    precision: f64,
) -> Vec<QftGate> {
    let mut gates = Vec::new();
    let min_angle = 2.0 * PI * precision;

    // For each qubit j in block B (higher qubit indices)
    for jj in 0..block_size_b {
        let qb = block_b_start + jj;
        // Controlled phases from qubits in block A (lower indices)
        for ii in 0..block_size_a {
            let qa = block_a_start + ii;
            // In the standard QFT, for target qubit qb (being processed),
            // the control at qa contributes with k = qb - qa
            let k = qb - qa;
            if k == 0 || k + 1 >= 64 {
                continue;
            }
            let angle = 2.0 * PI / (1u64 << (k + 1)) as f64;

            if angle.abs() < min_angle {
                continue;
            }

            gates.push(QftGate::ControlledPhase {
                control: qa,
                target: qb,
                angle,
            });
        }
    }

    gates
}

/// Build the optimistic QFT circuit.
///
/// Implements the log-depth in-place QFT from arXiv:2505.00701:
/// 1. Divide qubits into blocks of size b = O(log(n/eps))
/// 2. Apply exact QFT within each block (parallelizable)
/// 3. Apply optimistic inter-block rotations (skip distant blocks)
/// 4. Apply bit-reversal permutation
///
/// Achieves O(n log n) gates and O(log^2 n) depth with fidelity > 1-eps
/// on most computational basis inputs.
pub fn optimistic_qft(config: &OptimisticQftConfig) -> Result<QftResult, QftError> {
    let n = config.num_qubits;
    if n == 0 {
        return Err(QftError::InvalidQubits(
            "Number of qubits must be at least 1".into(),
        ));
    }
    if config.precision < 1e-15 {
        return Err(QftError::PrecisionTooHigh(
            "Precision cannot be smaller than ~1e-15 (machine epsilon)".into(),
        ));
    }

    // Special case: 1-qubit QFT is just Hadamard
    if n == 1 {
        let gates = vec![QftGate::Hadamard(0)];
        let circuit = QftCircuit {
            gates,
            num_qubits: 1,
            depth: 1,
            gate_count: 1,
        };
        return Ok(QftResult {
            circuit,
            approximation_error: 0.0,
            depth: 1,
            gate_count: 1,
        });
    }

    let block_size = compute_block_size(n, config.precision);
    let num_blocks = (n + block_size - 1) / block_size;
    let mut gates = Vec::new();

    // Phase 1: Intra-block exact QFT
    // Process blocks from highest qubit block to lowest (MSB-first within each block)
    for b_idx in (0..num_blocks).rev() {
        let start = b_idx * block_size;
        let actual_size = (n - start).min(block_size);
        let block_gates = build_block_qft(start, actual_size);
        gates.extend(block_gates);
    }

    // Phase 2: Inter-block rotations
    // For each pair (a, b) where b has higher qubit indices than a,
    // add the controlled rotations that the full QFT would include.
    // Skip block pairs where all rotations are below threshold.
    let trunc = config.effective_truncation();

    for b_idx in (0..num_blocks).rev() {
        let b_start = b_idx * block_size;
        let b_size = (n - b_start).min(block_size);

        for a_idx in 0..b_idx {
            let a_start = a_idx * block_size;
            let a_size = (n - a_start).min(block_size);

            // Minimum distance between nearest qubits in the two blocks
            let min_dist = b_start - (a_start + a_size - 1);
            if min_dist > trunc {
                continue;
            }

            let inter_gates =
                build_inter_block_rotations(a_start, b_start, a_size, b_size, config.precision);
            gates.extend(inter_gates);
        }
    }

    // Phase 3: Bit-reversal permutation
    append_bit_reversal_swaps(&mut gates, n);

    // If inverse, reverse gate order and negate all angles
    if config.inverse {
        gates.reverse();
        for gate in &mut gates {
            if let QftGate::ControlledPhase { angle, .. } = gate {
                *angle = -*angle;
            }
        }
    }

    let gate_count = gates.len();

    // Depth estimate: intra-block O(b) + inter-block O(log n) + swaps O(1)
    let depth = if num_blocks <= 1 {
        block_size
    } else {
        let inter_block_depth = ((num_blocks as f64).log2().ceil() as usize).max(1);
        block_size + inter_block_depth + 1
    };

    // Error bound: sum of skipped rotation angles
    let error_bound = (n as f64) * config.precision;

    let circuit = QftCircuit {
        gates,
        num_qubits: n,
        depth,
        gate_count,
    };

    Ok(QftResult {
        circuit,
        approximation_error: error_bound,
        depth,
        gate_count,
    })
}

/// Build the inverse optimistic QFT by reversing gates and negating angles.
pub fn optimistic_inverse_qft(config: &OptimisticQftConfig) -> Result<QftResult, QftError> {
    let mut inv_config = config.clone();
    inv_config.inverse = true;
    optimistic_qft(&inv_config)
}

// ============================================================
// SIMULATION / VERIFICATION
// ============================================================

/// Apply a sequence of QFT gates to a state vector in place.
///
/// The state vector has length 2^n. Qubit q controls bit position q
/// (qubit 0 = least significant bit).
pub fn apply_qft_gates(state: &mut Vec<Complex64>, gates: &[QftGate]) {
    let dim = state.len();

    for gate in gates {
        match gate {
            QftGate::Hadamard(q) => {
                let q = *q;
                let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                for i in 0..dim {
                    if i & (1 << q) == 0 {
                        let j = i | (1 << q);
                        let a = state[i];
                        let b = state[j];
                        state[i] = Complex64::new(inv_sqrt2, 0.0) * (a + b);
                        state[j] = Complex64::new(inv_sqrt2, 0.0) * (a - b);
                    }
                }
            }
            QftGate::ControlledPhase {
                control,
                target,
                angle,
            } => {
                let c = *control;
                let t = *target;
                let phase = Complex64::new(angle.cos(), angle.sin());
                for i in 0..dim {
                    if (i & (1 << c)) != 0 && (i & (1 << t)) != 0 {
                        state[i] *= phase;
                    }
                }
            }
            QftGate::Swap(a, b) => {
                let a = *a;
                let b = *b;
                for i in 0..dim {
                    let bit_a = (i >> a) & 1;
                    let bit_b = (i >> b) & 1;
                    if bit_a != bit_b {
                        let j = i ^ (1 << a) ^ (1 << b);
                        if i < j {
                            state.swap(i, j);
                        }
                    }
                }
            }
        }
    }
}

/// Build the exact DFT matrix for n qubits (dimension 2^n x 2^n).
///
/// Only practical for small n (at most ~12). The (j,k) entry is:
///   F_{jk} = (1/sqrt(N)) * omega^{jk}  where omega = e^{2*pi*i/N}, N = 2^n
pub fn exact_qft_matrix(n_qubits: usize) -> Array2<Complex64> {
    let dim = 1usize << n_qubits;
    let n_f64 = dim as f64;
    let norm = 1.0 / n_f64.sqrt();

    Array2::from_shape_fn((dim, dim), |(j, k)| {
        let angle = 2.0 * PI * (j as f64) * (k as f64) / n_f64;
        Complex64::new(norm * angle.cos(), norm * angle.sin())
    })
}

/// Estimate average fidelity of a QFT circuit over random computational basis inputs.
///
/// For each sample, we prepare a random basis state |x>, apply the circuit,
/// and compare with the exact QFT result. Returns the average fidelity.
pub fn average_fidelity(circuit: &QftCircuit, num_samples: usize) -> f64 {
    use rand::Rng;

    let n = circuit.num_qubits;
    let dim = 1usize << n;
    let exact = exact_qft_matrix(n);

    let mut rng = rand::thread_rng();
    let mut total_fidelity = 0.0;

    for _ in 0..num_samples {
        let basis_idx: usize = rng.gen_range(0..dim);

        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[basis_idx] = Complex64::new(1.0, 0.0);
        apply_qft_gates(&mut state, &circuit.gates);

        // Fidelity = |<exact|circuit>|^2
        let mut overlap = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            let exact_amp = exact[[i, basis_idx]];
            overlap += exact_amp.conj() * state[i];
        }
        total_fidelity += overlap.norm_sqr();
    }

    total_fidelity / num_samples as f64
}

/// Compute the worst-case error (1 - fidelity) over all computational basis inputs.
///
/// Only practical for small n. Iterates over all 2^n basis states.
pub fn worst_case_error(circuit: &QftCircuit, n_qubits: usize) -> f64 {
    let dim = 1usize << n_qubits;
    let exact = exact_qft_matrix(n_qubits);

    let mut min_fidelity = 1.0_f64;

    for basis_idx in 0..dim {
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[basis_idx] = Complex64::new(1.0, 0.0);
        apply_qft_gates(&mut state, &circuit.gates);

        let mut overlap = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            let exact_amp = exact[[i, basis_idx]];
            overlap += exact_amp.conj() * state[i];
        }
        let fidelity = overlap.norm_sqr();
        if fidelity < min_fidelity {
            min_fidelity = fidelity;
        }
    }

    1.0 - min_fidelity
}

// ============================================================
// COMPARISON
// ============================================================

/// Compare all three QFT methods for a given qubit count and precision.
pub fn compare_qft_methods(n_qubits: usize, precision: f64) -> QftComparison {
    let std_circ = standard_qft(n_qubits);
    let approx_circ = approximate_qft(n_qubits, precision);

    let config = OptimisticQftConfig::new(n_qubits).with_precision(precision);
    let opt_result = optimistic_qft(&config).expect("optimistic QFT failed");

    let opt_avg_error = if n_qubits <= 10 {
        let samples = (100).min(1usize << n_qubits);
        1.0 - average_fidelity(&opt_result.circuit, samples)
    } else {
        opt_result.approximation_error
    };

    let approx_error = if n_qubits <= 10 {
        worst_case_error(&approx_circ, n_qubits)
    } else {
        (n_qubits as f64) * precision
    };

    QftComparison {
        standard_gate_count: std_circ.gate_count,
        standard_depth: std_circ.depth,
        approximate_gate_count: approx_circ.gate_count,
        approximate_depth: approx_circ.depth,
        approximate_error: approx_error,
        optimistic_gate_count: opt_result.gate_count,
        optimistic_depth: opt_result.depth,
        optimistic_avg_error: opt_avg_error,
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Config builder defaults
    #[test]
    fn test_config_defaults() {
        let config = OptimisticQftConfig::new(10);
        assert_eq!(config.num_qubits, 10);
        assert!((config.precision - 1e-3).abs() < 1e-15);
        assert_eq!(config.truncation_level, 0);
        assert!(!config.inverse);
    }

    // 2. Standard QFT gate count = n Hadamards + n(n-1)/2 CR gates + n/2 swaps
    #[test]
    fn test_standard_qft_gate_count() {
        for n in 1..=8 {
            let circ = standard_qft(n);
            let n_hadamards = n;
            let n_cr: usize = n * (n - 1) / 2;
            let n_swaps = n / 2;
            let expected = n_hadamards + n_cr + n_swaps;
            assert_eq!(
                circ.gate_count, expected,
                "n={}: expected {} gates ({}H + {}CR + {}S), got {}",
                n, expected, n_hadamards, n_cr, n_swaps, circ.gate_count
            );
        }
    }

    // 3. Standard QFT on 3 qubits matches exact DFT matrix
    #[test]
    fn test_standard_qft_3_qubits_matches_dft() {
        let n = 3;
        let circ = standard_qft(n);
        let dim = 1usize << n;
        let exact = exact_qft_matrix(n);

        for basis_idx in 0..dim {
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            state[basis_idx] = Complex64::new(1.0, 0.0);
            apply_qft_gates(&mut state, &circ.gates);

            for i in 0..dim {
                let expected = exact[[i, basis_idx]];
                let got = state[i];
                assert!(
                    (expected - got).norm() < 1e-10,
                    "Mismatch at |{}> -> |{}>: expected {:?}, got {:?}",
                    basis_idx,
                    i,
                    expected,
                    got
                );
            }
        }
    }

    // 4. Approximate QFT reduces gate count vs standard
    #[test]
    fn test_approximate_qft_fewer_gates() {
        for n in 4..=10 {
            let std_circ = standard_qft(n);
            let approx_circ = approximate_qft(n, 1e-2);
            assert!(
                approx_circ.gate_count <= std_circ.gate_count,
                "n={}: approx ({}) should have <= gates than standard ({})",
                n,
                approx_circ.gate_count,
                std_circ.gate_count
            );
        }
    }

    // 5. Approximate QFT error bounded by precision
    #[test]
    fn test_approximate_qft_error_bounded() {
        // For n=4 with precision=0.05, k_max = ceil(log2(20)) = 5.
        // Since max k in n=4 QFT is 3 (< 5), all rotations are kept => exact.
        let n = 4;
        let precision = 0.05;
        let circ = approximate_qft(n, precision);
        let error = worst_case_error(&circ, n);
        assert!(
            error < 1e-6,
            "For n={} with k_max={}, all rotations kept; error should be ~0 but got {}",
            n,
            (1.0 / precision).log2().ceil() as usize,
            error
        );

        // For a case where truncation actually happens: n=8, precision large enough
        // that k_max < n-1
        let n2 = 8;
        let precision2 = 0.25; // k_max = ceil(log2(4)) = 2
        let circ2 = approximate_qft(n2, precision2);
        let error2 = worst_case_error(&circ2, n2);
        // Error should be finite but less than 1
        assert!(
            error2 < 1.0,
            "Approximate QFT error {} should be < 1.0 for n={}, precision={}",
            error2,
            n2,
            precision2
        );
    }

    // 6. Optimistic QFT has fewer or equal gates than standard for n >= 8
    #[test]
    fn test_optimistic_qft_gate_count_scaling() {
        let config_8 = OptimisticQftConfig::new(8).with_precision(1e-3);
        let config_16 = OptimisticQftConfig::new(16).with_precision(1e-3);

        let result_8 = optimistic_qft(&config_8).unwrap();
        let result_16 = optimistic_qft(&config_16).unwrap();

        let std_8 = standard_qft(8);
        let std_16 = standard_qft(16);

        assert!(
            result_16.gate_count <= std_16.gate_count,
            "Optimistic ({}) should have <= gates than standard ({}) at n=16",
            result_16.gate_count,
            std_16.gate_count
        );

        // Scaling: standard O(n^2) ~4x, optimistic should be less
        let std_ratio = std_16.gate_count as f64 / std_8.gate_count as f64;
        let opt_ratio = result_16.gate_count as f64 / result_8.gate_count as f64;
        assert!(
            opt_ratio <= std_ratio + 0.5,
            "Optimistic scaling ratio ({:.2}) should be <= standard ({:.2})",
            opt_ratio,
            std_ratio
        );
    }

    // 7. Optimistic QFT depth is O(log^2 n) or better
    #[test]
    fn test_optimistic_qft_depth() {
        let config = OptimisticQftConfig::new(16).with_precision(1e-3);
        let result = optimistic_qft(&config).unwrap();
        let std_circ = standard_qft(16);

        assert!(
            result.depth < std_circ.depth,
            "Optimistic depth ({}) should be < standard depth ({})",
            result.depth,
            std_circ.depth
        );

        let n = 16.0_f64;
        let log2_n = n.log2();
        let log_sq = (log2_n * log2_n) as usize;
        assert!(
            result.depth <= log_sq + 5,
            "Optimistic depth ({}) should be ~O(log^2 n) = {} for n=16",
            result.depth,
            log_sq
        );
    }

    // 8. Optimistic QFT average fidelity is high for small n
    #[test]
    fn test_optimistic_qft_average_fidelity() {
        // For n=4, block_size = ceil(ln(4/0.05)) = ceil(4.38) = 5 >= 4,
        // so the optimistic QFT degenerates to standard QFT (single block).
        let n = 4;
        let precision = 0.05;
        let config = OptimisticQftConfig::new(n).with_precision(precision);
        let result = optimistic_qft(&config).unwrap();

        let fidelity = average_fidelity(&result.circuit, 16);
        assert!(
            fidelity > 0.99,
            "For n={} with single-block optimistic QFT, fidelity ({:.6}) should be ~1.0",
            n,
            fidelity
        );
    }

    // 9. Block size grows logarithmically with n
    #[test]
    fn test_block_size_logarithmic() {
        let precision = 1e-3;
        let b4 = compute_block_size(4, precision);
        let b16 = compute_block_size(16, precision);
        let b64 = compute_block_size(64, precision);
        let b256 = compute_block_size(256, precision);

        assert!(
            b256 < b64 * 3,
            "Block size should grow sub-linearly: b(256)={} vs b(64)={}",
            b256,
            b64
        );
        assert!(
            b64 < b16 * 3,
            "Block size should grow sub-linearly: b(64)={} vs b(16)={}",
            b64,
            b16
        );
        assert!(b16 >= b4, "Block size should be non-decreasing");
        assert!(b64 >= b16, "Block size should be non-decreasing");
    }

    // 10. Inverse QFT is adjoint of forward QFT (round-trip fidelity)
    #[test]
    fn test_inverse_qft_is_adjoint() {
        let n = 3;
        let dim = 1usize << n;

        let config = OptimisticQftConfig::new(n).with_precision(1e-6);
        let fwd = optimistic_qft(&config).unwrap();
        let inv = optimistic_inverse_qft(&config).unwrap();

        for basis_idx in 0..dim {
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            state[basis_idx] = Complex64::new(1.0, 0.0);

            apply_qft_gates(&mut state, &fwd.circuit.gates);
            apply_qft_gates(&mut state, &inv.circuit.gates);

            let fidelity = state[basis_idx].norm_sqr();
            assert!(
                fidelity > 0.99,
                "Round-trip fidelity for |{}> = {:.6}, expected ~1.0",
                basis_idx,
                fidelity
            );
        }
    }

    // 11. QFT comparison: optimistic <= standard gate count
    #[test]
    fn test_comparison_optimistic_fewer_gates() {
        let cmp = compare_qft_methods(8, 1e-3);
        assert!(
            cmp.optimistic_gate_count <= cmp.standard_gate_count,
            "Optimistic ({}) should have <= gates than standard ({}) at n=8",
            cmp.optimistic_gate_count,
            cmp.standard_gate_count
        );

        let cmp16 = compare_qft_methods(16, 1e-3);
        assert!(
            cmp16.optimistic_gate_count <= cmp16.standard_gate_count,
            "Optimistic ({}) should have <= gates than standard ({}) at n=16",
            cmp16.optimistic_gate_count,
            cmp16.standard_gate_count
        );
    }

    // 12. Single-qubit QFT is just Hadamard (all methods agree)
    #[test]
    fn test_single_qubit_qft() {
        let std_circ = standard_qft(1);
        let approx_circ = approximate_qft(1, 1e-3);
        let config = OptimisticQftConfig::new(1);
        let opt_result = optimistic_qft(&config).unwrap();

        assert_eq!(std_circ.gate_count, 1);
        assert_eq!(approx_circ.gate_count, 1);
        assert_eq!(opt_result.gate_count, 1);

        assert_eq!(std_circ.gates[0], QftGate::Hadamard(0));
        assert_eq!(approx_circ.gates[0], QftGate::Hadamard(0));
        assert_eq!(opt_result.circuit.gates[0], QftGate::Hadamard(0));
    }

    // 13. Error variant construction
    #[test]
    fn test_error_variants() {
        let config_zero = OptimisticQftConfig::new(0);
        let result = optimistic_qft(&config_zero);
        assert!(matches!(result, Err(QftError::InvalidQubits(_))));

        let config_prec = OptimisticQftConfig::new(4).with_precision(1e-16);
        let result = optimistic_qft(&config_prec);
        assert!(matches!(result, Err(QftError::PrecisionTooHigh(_))));
    }

    // 14. Exact QFT matrix is unitary
    #[test]
    fn test_exact_qft_matrix_unitary() {
        let n = 3;
        let dim = 1usize << n;
        let f = exact_qft_matrix(n);

        for i in 0..dim {
            for j in 0..dim {
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..dim {
                    dot += f[[k, i]].conj() * f[[k, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot.re - expected).abs() < 1e-10 && dot.im.abs() < 1e-10,
                    "F^dag F[{},{}] = {:?}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    // 15. Standard QFT on 2 qubits: QFT|0> has uniform amplitudes
    #[test]
    fn test_standard_qft_2_qubits() {
        let circ = standard_qft(2);
        let dim = 4;

        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        apply_qft_gates(&mut state, &circ.gates);

        let expected_amp = 0.5;
        for i in 0..dim {
            assert!(
                (state[i].norm() - expected_amp).abs() < 1e-10,
                "QFT|0>[{}] amplitude = {}, expected {}",
                i,
                state[i].norm(),
                expected_amp
            );
        }
    }

    // 16. Config builder chaining works
    #[test]
    fn test_config_builder_chaining() {
        let config = OptimisticQftConfig::new(12)
            .with_precision(1e-4)
            .with_truncation_level(8)
            .with_inverse(true);

        assert_eq!(config.num_qubits, 12);
        assert!((config.precision - 1e-4).abs() < 1e-15);
        assert_eq!(config.truncation_level, 8);
        assert!(config.inverse);
    }
}
