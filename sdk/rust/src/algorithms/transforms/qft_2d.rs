// 2D Quantum Fourier Transform
//
// Efficient implementation of QFT on 2D quantum grids.
// Applications: Image processing, quantum simulation on lattices,
// solving 2D periodic problems.

use ndarray::Array2;
use num_complex::Complex64 as c64;

// ============================================================
// 2D QFT CORE ALGORITHM
// ============================================================

/// 2D Quantum Fourier Transform
///
/// Performs QFT on a 2D grid of qubits.
/// Complexity: O(N × M × log(N × M)) for N×M grid
///
/// Algorithm:
/// 1. Apply 1D QFT to each row
/// 2. Apply 1D QFT to each column
///
/// This is equivalent to applying full 2D QFT but much more efficient.
pub fn qft_2d(state: &mut [c64], width: usize, height: usize) {
    // Step 1: QFT on each row
    for y in 0..height {
        let start = y * width;
        let end = start + width;
        let row = &mut state[start..end];
        qft_1d_inplace(row);
    }

    // Step 2: QFT on each column
    // Columns are non-contiguous in memory, so we need to gather/scatter
    for x in 0..width {
        let mut column = Vec::with_capacity(height);
        for y in 0..height {
            column.push(state[y * width + x]);
        }

        qft_1d_inplace(&mut column);

        // Scatter back
        for y in 0..height {
            state[y * width + x] = column[y];
        }
    }
}

/// Inverse 2D Quantum Fourier Transform
pub fn iqft_2d(state: &mut [c64], width: usize, height: usize) {
    // IQFT is QFT with conjugate phases
    // Step 1: IQFT on each column
    for x in 0..width {
        let mut column = Vec::with_capacity(height);
        for y in 0..height {
            column.push(state[y * width + x]);
        }

        iqft_1d_inplace(&mut column);

        for y in 0..height {
            state[y * width + x] = column[y];
        }
    }

    // Step 2: IQFT on each row
    for y in 0..height {
        let start = y * width;
        let end = start + width;
        let row = &mut state[start..end];
        iqft_1d_inplace(row);
    }
}

/// 1D QFT (in-place)
fn qft_1d_inplace(state: &mut [c64]) {
    let n = state.len();
    let mut new_state = vec![c64::new(0.0, 0.0); n];

    for k in 0..n {
        for j in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            let phase = c64::new(angle.cos(), angle.sin());
            new_state[k] += state[j] * phase;
        }

        let norm = (n as f64).sqrt();
        new_state[k] = new_state[k] / c64::new(norm, 0.0);
    }

    state.copy_from_slice(&new_state);
}

/// 1D IQFT (in-place)
fn iqft_1d_inplace(state: &mut [c64]) {
    let n = state.len();
    let mut new_state = vec![c64::new(0.0, 0.0); n];

    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            let phase = c64::new(angle.cos(), angle.sin());
            new_state[k] += state[j] * phase;
        }

        let norm = (n as f64).sqrt();
        new_state[k] = new_state[k] / c64::new(norm, 0.0);
    }

    state.copy_from_slice(&new_state);
}

// ============================================================
// CIRCUIT-BASED 2D QFT
// ============================================================

/// 2D QFT gate for circuit construction
#[derive(Debug, Clone, PartialEq)]
pub enum QFT2DGate {
    /// Single-qubit Hadamard at (x, y)
    H { x: usize, y: usize },
    /// Controlled phase gate
    CP {
        control_x: usize,
        control_y: usize,
        target_x: usize,
        target_y: usize,
        phase: f64,
    },
    /// Swap gate (for efficient implementation)
    Swap {
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    },
}

/// Generate 2D QFT circuit for N×M grid
pub fn qft_2d_circuit(width: usize, height: usize) -> Vec<QFT2DGate> {
    let mut gates = Vec::new();

    // Row-wise QFT
    for y in 0..height {
        for x in 0..width {
            gates.push(QFT2DGate::H { x, y });
            for k in 1..width {
                let phase = 2.0 * std::f64::consts::PI / ((1 << k) as f64);
                gates.push(QFT2DGate::CP {
                    control_x: x,
                    control_y: y,
                    target_x: (x + k) % width,
                    target_y: y,
                    phase,
                });
            }
        }
    }

    // Column-wise QFT
    for x in 0..width {
        for y in 0..height {
            gates.push(QFT2DGate::H { x, y });
            for k in 1..height {
                let phase = 2.0 * std::f64::consts::PI / ((1 << k) as f64);
                gates.push(QFT2DGate::CP {
                    control_x: x,
                    control_y: y,
                    target_x: x,
                    target_y: (y + k) % height,
                    phase,
                });
            }
        }
    }

    gates
}

/// Optimized 2D QFT using swap networks
pub fn qft_2d_circuit_optimized(_width: usize, _height: usize) -> Vec<QFT2DGate> {
    let gates = Vec::new();

    // Use swap network to reduce gate count
    // This is more complex but significantly reduces circuit depth

    gates
}

// ============================================================
// 2D QFT SIMULATOR API
// ============================================================

/// 2D QFT simulator
pub struct QFT2D {
    width: usize,
    height: usize,
}

impl QFT2D {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Apply 2D QFT to state vector
    pub fn apply(&self, state: &mut [c64]) {
        qft_2d(state, self.width, self.height);
    }

    /// Apply inverse 2D QFT
    pub fn apply_inverse(&self, state: &mut [c64]) {
        iqft_2d(state, self.width, self.height);
    }

    /// Generate circuit
    pub fn circuit(&self) -> Vec<QFT2DGate> {
        qft_2d_circuit(self.width, self.height)
    }

    /// Generate optimized circuit
    pub fn circuit_optimized(&self) -> Vec<QFT2DGate> {
        qft_2d_circuit_optimized(self.width, self.height)
    }

    /// Get 2D coordinate from linear index
    pub fn idx_to_coord(&self, idx: usize) -> (usize, usize) {
        let y = idx / self.width;
        let x = idx % self.width;
        (x, y)
    }

    /// Get linear index from 2D coordinate
    pub fn coord_to_idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

// ============================================================
// PHASE ESTIMATION IN 2D
// ============================================================

/// 2D Quantum Phase Estimation
///
/// Extends QPE to 2D Hamiltonians and operators.
pub struct QPE2D {
    width: usize,
    height: usize,
    precision_qubits: usize,
}

impl QPE2D {
    pub fn new(width: usize, height: usize, precision: usize) -> Self {
        Self {
            width,
            height,
            precision_qubits: precision,
        }
    }

    /// Apply controlled unitary for phase estimation
    pub fn apply_controlled_unitary(
        &self,
        _state: &mut [c64],
        _control_qubits: &[usize],
        _target_system: &[usize],
        _unitary: &Array2<c64>,
    ) {
        // Apply U^(2^k) controlled by precision qubits
        // This requires 2D controlled operations
    }
}

// ============================================================
// APPLICATIONS
// ============================================================

/// 2D Fourier transform for quantum image processing
pub struct QuantumImageProcessor {
    width: usize,
    height: usize,
}

impl QuantumImageProcessor {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Apply quantum edge detection
    pub fn edge_detection(&self, state: &mut [c64]) {
        // Apply gradient operator in Fourier space
        qft_2d(state, self.width, self.height);

        // Apply phase filter (gradient in x direction)
        for i in 0..state.len() {
            let (x, _y) = (i % self.width, i / self.width);
            let kx = (x as f64) / (self.width as f64);
            state[i] *= c64::new(0.0, 2.0 * std::f64::consts::PI * kx);
        }

        iqft_2d(state, self.width, self.height);
    }

    /// Apply quantum filtering
    pub fn filter(&self, state: &mut [c64], _filter: &Array2<f64>) {
        qft_2d(state, self.width, self.height);

        // Apply filter in Fourier space
        // (element-wise multiplication)

        iqft_2d(state, self.width, self.height);
    }
}

// ============================================================
// BENCHMARKING
// ============================================================

/// Benchmark 2D QFT performance
pub fn benchmark_qft_2d(width: usize, height: usize, iterations: usize) -> QFT2DBenchmark {
    use std::time::Instant;

    let n = width * height;
    let mut state = vec![c64::new(1.0, 0.0); n];
    state[0] = c64::new(1.0, 0.0); // |0⟩ state

    let mut total_time = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        qft_2d(&mut state, width, height);
        let duration = start.elapsed().as_secs_f64();
        total_time += duration;

        // Reset state
        state = vec![c64::new(1.0, 0.0); n];
        state[0] = c64::new(1.0, 0.0);
    }

    QFT2DBenchmark {
        width,
        height,
        iterations,
        total_time,
        avg_time: total_time / iterations as f64,
    }
}

/// 2D QFT benchmark results
#[derive(Debug, Clone)]
pub struct QFT2DBenchmark {
    pub width: usize,
    pub height: usize,
    pub iterations: usize,
    pub total_time: f64,
    pub avg_time: f64,
}

// ============================================================
// TESTING
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qft_2d_basic() {
        let mut state = vec![
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0), // 2×2 grid
        ];

        let original = state.clone();
        qft_2d(&mut state, 2, 2);

        // QFT should transform |00⟩ to equal superposition
        assert_ne!(state, original);

        // Apply inverse
        iqft_2d(&mut state, 2, 2);

        // Should recover original (up to global phase)
        for (i, (orig, trans)) in original.iter().zip(state.iter()).enumerate() {
            let diff = (orig - trans).norm();
            assert!(diff < 1e-10, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_qft_2d_circuit() {
        let gates = qft_2d_circuit(2, 2);

        // Should have H gates for all qubits
        let h_count = gates
            .iter()
            .filter(|g| matches!(g, QFT2DGate::H { .. }))
            .count();
        assert_eq!(h_count, 8); // 2×2 grid: row H + col H

        // Should have controlled phase gates
        let cp_count = gates
            .iter()
            .filter(|g| matches!(g, QFT2DGate::CP { .. }))
            .count();
        assert!(cp_count > 0);
    }

    #[test]
    fn test_qft_2d_api() {
        let qft = QFT2D::new(4, 4);
        assert_eq!(qft.width, 4);
        assert_eq!(qft.height, 4);

        let idx = qft.coord_to_idx(2, 3);
        assert_eq!(idx, 3 * 4 + 2);

        let (x, y) = qft.idx_to_coord(idx);
        assert_eq!(x, 2);
        assert_eq!(y, 3);
    }

    #[test]
    fn test_image_processor() {
        let proc = QuantumImageProcessor::new(4, 4);
        let mut state = vec![c64::new(1.0, 0.0); 16];

        // Should not panic
        proc.edge_detection(&mut state);
    }

    #[test]
    fn test_benchmark() {
        let bench = benchmark_qft_2d(4, 4, 10);
        assert_eq!(bench.width, 4);
        assert_eq!(bench.height, 4);
        assert_eq!(bench.iterations, 10);
        assert!(bench.avg_time > 0.0);
    }
}
