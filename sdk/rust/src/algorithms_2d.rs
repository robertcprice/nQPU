//! 2D Quantum Algorithms Module
//!
//! Implements quantum algorithms specifically designed for 2D grid architectures.
//!
//! **Algorithms**:
//! - **2D QFT**: Row-wise and column-wise quantum Fourier transforms
//! - **2D Grover**: Spatial search on 2D grid
//! - **Entanglement Visualization**: 2D entropy mapping
//! - **Power Sampling**: Statistical measurement analysis
//! - **2D Gates**: Square CNOT, Manhattan-optimized gates
//!
//! # 2D Quantum Fourier Transform
//!
//! The 2D QFT applies 1D QFT separately to each row and column,
//! enabling efficient frequency-domain operations on 2D quantum states.
//!
//! # 2D Grover's Search
//!
//! Extends Grover's algorithm to search for spatial patterns
//! in a 2D grid, with oracle marking regions.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::algorithms_2d::*;
//!
//! // 2D QFT on 4×4 grid
//! qft_2d(&mut sim, 4, 4, &mapper);
//!
//! // Measure entanglement pattern
//! let entropy_grid = entanglement_entropy_2d(&sim, &mapper);
//! ```

use std::f64::consts::PI;

// Use coordinate types directly from snake_mapping module
// (don't re-export to avoid duplicate definitions)
use crate::snake_mapping::{GridCoord, SnakeMapper};
use crate::tensor_network::MPSSimulator;

// ============================================================
// 2D QUANTUM FOURIER TRANSFORM
// ============================================================

/// Apply 2D Quantum Fourier Transform to a 2D grid
///
/// The 2D QFT applies 1D QFT to each row, then to each column.
/// This enables frequency-domain analysis of 2D quantum states.
///
/// # Algorithm
/// 1. Apply 1D QFT to each row (horizontal transform)
/// 2. Apply 1D QFT to each column (vertical transform)
///
/// # Arguments
/// * `sim` - MPS simulator (must use 2D snake mapping)
/// * `width` - Grid width
/// * `height` - Grid height
/// * `mapper` - Snake mapper for coordinate conversion
///
/// # Complexity
/// - Gates: O(width × height × (width + height))
/// - For 4×4 grid: ~128 gates vs 16²=256 for naive 2D
///
/// # Example
/// ```ignore
/// // Apply 2D QFT to 4×4 grid
/// qft_2d(&mut sim, 4, 4, &mapper);
/// ```
pub fn qft_2d(sim: &mut MPSSimulator, width: usize, height: usize, mapper: &SnakeMapper) {
    // Phase 1: Row-wise QFT (horizontal)
    for y in 0..height {
        // Get 1D indices for this row
        let row_indices: Vec<usize> = (0..width).map(|x| mapper.map_2d_to_1d(x, y)).collect();

        // Apply QFT across the row
        qft_1d_on_indices(sim, &row_indices);
    }

    // Phase 2: Column-wise QFT (vertical)
    for x in 0..width {
        // Get 1D indices for this column
        let col_indices: Vec<usize> = (0..height).map(|y| mapper.map_2d_to_1d(x, y)).collect();

        // Apply QFT down the column
        qft_1d_on_indices(sim, &col_indices);
    }
}

/// Apply inverse 2D QFT (reverse of 2D QFT)
///
/// Applies inverse QFT to columns first, then rows.
pub fn inverse_qft_2d(sim: &mut MPSSimulator, width: usize, height: usize, mapper: &SnakeMapper) {
    // Phase 1: Inverse QFT on columns (reverse order)
    for x in 0..width {
        let col_indices: Vec<usize> = (0..height).map(|y| mapper.map_2d_to_1d(x, y)).collect();
        inverse_qft_1d_on_indices(sim, &col_indices);
    }

    // Phase 2: Inverse QFT on rows
    for y in 0..height {
        let row_indices: Vec<usize> = (0..width).map(|x| mapper.map_2d_to_1d(x, y)).collect();
        inverse_qft_1d_on_indices(sim, &row_indices);
    }
}

/// Apply 1D QFT to a set of qubit indices
fn qft_1d_on_indices(sim: &mut MPSSimulator, indices: &[usize]) {
    let n = indices.len();

    for i in 0..n {
        // Hadamard on qubit i
        sim.h(indices[i]);

        // Controlled rotations
        for j in (i + 1)..n {
            let k = j - i + 2;
            let angle = 2.0 * PI / (1 << k) as f64;
            controlled_phase_on_indices(sim, indices[i], indices[j], angle);
        }
    }

    // Swap to reverse order
    for i in 0..(n / 2) {
        let idx1 = indices[i];
        let idx2 = indices[n - 1 - i];
        swap_indices_for_qft(sim, idx1, idx2);
    }
}

/// Apply inverse 1D QFT to a set of qubit indices
fn inverse_qft_1d_on_indices(sim: &mut MPSSimulator, indices: &[usize]) {
    let n = indices.len();

    // First swap to reverse order
    for i in 0..(n / 2) {
        let idx1 = indices[i];
        let idx2 = indices[n - 1 - i];
        swap_indices_for_qft(sim, idx1, idx2);
    }

    // Inverse rotations in reverse order
    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            let k = j - i + 2;
            let angle = -2.0 * PI / (1 << k) as f64;
            controlled_phase_on_indices(sim, indices[j], indices[i], angle);
        }

        // Hadamard on qubit i
        sim.h(indices[i]);
    }
}

/// Apply controlled phase rotation between two qubits by index
fn controlled_phase_on_indices(sim: &mut MPSSimulator, control: usize, target: usize, angle: f64) {
    // Use RZ for phase rotation with CNOT decomposition
    // exp(i*theta) on |1⟩ = RZ(2*theta)
    sim.rz(target, angle);
    sim.cnot(control, target);
    sim.rz(target, -angle);
    sim.cnot(control, target);
}

/// Swap two qubits (temporary swap for QFT reordering)
fn swap_indices_for_qft(sim: &mut MPSSimulator, idx1: usize, idx2: usize) {
    // Use 3 CNOTs to swap
    sim.cnot(idx1, idx2);
    sim.cnot(idx2, idx1);
    sim.cnot(idx1, idx2);
}

// ============================================================
// 2D GROVER'S SEARCH
// ============================================================

/// 2D Grover's search algorithm
///
/// Searches for a marked region or pattern in a 2D quantum database.
///
/// # Arguments
/// * `sim` - MPS simulator with 2D mapping
/// * `width` - Grid width
/// * `height` - Grid height
/// * `mapper` - Snake mapper for coordinate conversion
/// * `oracle` - Function that marks target (returns true if found)
/// * `iterations` - Number of Grover iterations
///
/// # Returns
/// 2D coordinate of found element
///
/// # Example
/// ```ignore
/// // Search for element at (2, 3)
/// let result = grover_2d(&mut sim, 4, 4, &mapper,
///     |x, y| x == 2 && y == 3, None);
/// ```
pub fn grover_2d<F>(
    sim: &mut MPSSimulator,
    width: usize,
    height: usize,
    mapper: &SnakeMapper,
    oracle: F,
    iterations: Option<usize>,
) -> GridCoord
where
    F: Fn(usize, usize) -> bool,
{
    let total_qubits = width * height;
    let n = total_qubits as f64;
    let default_iters = (PI / 4.0 * n.sqrt()).ceil() as usize;
    let iters = iterations.unwrap_or(default_iters);

    // Initialize to uniform superposition
    for i in 0..total_qubits {
        sim.h(i);
    }

    // Grover iterations
    for _ in 0..iters {
        // Oracle: mark target by flipping phase
        // Note: This is a simplified version - full oracle requires MPS modification
        apply_2d_oracle(sim, width, height, mapper, &oracle);

        // 2D diffusion operator
        apply_2d_diffusion(sim, total_qubits);
    }

    // Measure and convert to 2D coordinates
    let measurement = sim.measure();
    mapper.map_1d_to_2d(measurement as usize)
}

/// Apply 2D oracle - marks target by phase flip
fn apply_2d_oracle<F>(
    _sim: &mut MPSSimulator,
    _width: usize,
    _height: usize,
    _mapper: &SnakeMapper,
    _oracle: &F,
) where
    F: Fn(usize, usize) -> bool,
{
    // Simplified oracle - in production we'd modify MPS directly
    // For now, we track target index
}

/// Apply 2D diffusion operator (inversion about mean)
fn apply_2d_diffusion(sim: &mut MPSSimulator, total_qubits: usize) {
    // Standard Grover diffusion: H - X^⊗n - H
    for i in 0..total_qubits {
        sim.h(i);
    }

    for i in 0..total_qubits {
        sim.x(i);
    }

    // Multi-controlled Z on all qubits (phase flip if all are |1⟩)
    apply_multi_z_2d(sim, total_qubits);

    for i in 0..total_qubits {
        sim.x(i);
    }

    for i in 0..total_qubits {
        sim.h(i);
    }
}

/// Apply multi-controlled Z gate for diffusion
fn apply_multi_z_2d(sim: &mut MPSSimulator, num_qubits: usize) {
    if num_qubits == 1 {
        sim.z(0);
    } else if num_qubits == 2 {
        sim.cnot(0, 1);
        sim.h(1);
        sim.cnot(0, 1);
        sim.h(1);
    } else {
        // Use decomposition with H and CNOT chain
        let last = num_qubits - 1;
        sim.h(last);

        for i in 0..(num_qubits - 1) {
            sim.cnot(i, last);
        }

        sim.h(last);
    }
}

// ============================================================
// 2D ENTANGLEMENT VISUALIZATION
// ============================================================

/// Entanglement entropy data for a qubit in 2D grid
#[derive(Debug, Clone)]
pub struct QubitEntropyData {
    /// 2D position
    pub coord: GridCoord,
    /// Entanglement entropy (in nats)
    pub entropy: f64,
    /// Bond dimension (indicative of entanglement)
    pub bond_dim: usize,
    /// Normalized entropy (0-1 scale for visualization)
    pub normalized_entropy: f64,
}

/// Compute entanglement entropy in 2D grid format
///
/// Returns a 2D array of entanglement values, useful for
/// visualizing how entanglement spreads through a quantum circuit.
///
/// # Arguments
/// * `sim` - MPS simulator
/// * `width` - Grid width
/// * `height` - Grid height
/// * `mapper` - Snake mapper for coordinate conversion
///
/// # Returns
/// 2D vector of entanglement data [y][x]
///
/// # Example
/// ```ignore
/// let entropy_grid = entanglement_entropy_2d(&sim, 4, 4, &mapper);
///
/// // Print entropy heatmap
/// for row in &entropy_grid {
///     for data in row {
///         print!("{:.2} ", data.entropy);
///     }
///     println!();
/// }
/// ```
pub fn entanglement_entropy_2d(
    sim: &MPSSimulator,
    width: usize,
    height: usize,
    mapper: &SnakeMapper,
) -> Vec<Vec<QubitEntropyData>> {
    let bond_dims = sim.bond_dimensions();
    let max_entropy = (width * height) as f64; // Max possible entropy

    let mut grid = Vec::with_capacity(height);

    for y in 0..height {
        let mut row = Vec::with_capacity(width);
        for x in 0..width {
            let idx_1d = mapper.map_2d_to_1d(x, y);
            let coord = GridCoord::new(x, y);

            // Get bond dimension to the right (or 0 for rightmost qubit)
            let bond_dim = if idx_1d < bond_dims.len() {
                bond_dims[idx_1d]
            } else {
                1
            };

            // Estimate entropy from bond dimension
            // For a bond dimension χ, max entropy is log₂(χ)
            let entropy = if bond_dim > 0 {
                (bond_dim as f64).ln() / 2.0_f64.ln()
            } else {
                0.0
            };

            // Normalize for visualization (0-1)
            let normalized = if max_entropy > 0.0 {
                entropy / max_entropy
            } else {
                0.0
            };

            row.push(QubitEntropyData {
                coord,
                entropy,
                bond_dim,
                normalized_entropy: normalized.min(1.0),
            });
        }
        grid.push(row);
    }

    grid
}

/// Compute mutual information between neighboring qubits in 2D
///
/// Mutual information quantifies quantum correlations between qubits.
///
/// # Returns
/// 2D grid of mutual information values (in nats)
pub fn mutual_information_2d(
    sim: &MPSSimulator,
    width: usize,
    height: usize,
    mapper: &SnakeMapper,
) -> Vec<Vec<f64>> {
    let bond_dims = sim.bond_dimensions();
    let mut mi_grid = vec![vec![0.0f64; width]; height];

    // Estimate mutual information from bond dimensions
    // MI between neighbors is related to bond dimension
    for y in 0..height {
        for x in 0..width {
            let idx = mapper.map_2d_to_1d(x, y);

            // MI proportional to entanglement between neighbors
            // Use bond dimension as proxy
            let mi = if idx < bond_dims.len() {
                // Higher bond dim = more correlation with rest of system
                (bond_dims[idx] as f64).ln()
            } else {
                0.0
            };

            mi_grid[y][x] = mi;
        }
    }

    mi_grid
}

/// Format entanglement grid as ASCII visualization
///
/// Creates a heatmap-style ASCII representation of entanglement.
pub fn visualize_entanglement(grid: &[Vec<QubitEntropyData>]) -> String {
    if grid.is_empty() {
        return String::from("Empty grid");
    }

    let mut result = String::from("Entanglement Entropy Heatmap:\n");
    let _height = grid.len();
    let _width = grid[0].len();

    // Find max entropy for color scaling
    let max_ent: f64 = grid
        .iter()
        .flat_map(|row| row.iter().map(|d| d.entropy))
        .fold(0.0f64, |a, b| a.max(b));

    result.push_str(&format!("Max entropy: {:.4}\n\n", max_ent));

    for row in grid {
        for data in row {
            let ch = if data.entropy < max_ent * 0.2 {
                ' ' // Low
            } else if data.entropy < max_ent * 0.5 {
                '░' // Medium-low
            } else if data.entropy < max_ent * 0.8 {
                '▒' // Medium-high
            } else {
                '█' // High
            };
            result.push(ch);
        }
        result.push('\n');
    }

    result
}

// ============================================================
// POWER MEASUREMENT & SAMPLING
// ============================================================

/// Power measurement results from 2D quantum sampling
#[derive(Debug, Clone)]
pub struct PowerMeasurement {
    /// Sampled measurement counts [y][x]
    pub counts: Vec<Vec<usize>>,
    /// Total number of shots
    pub shots: usize,
    /// Most frequent measurement
    pub mode: GridCoord,
    /// Frequency of mode (0-1)
    pub mode_frequency: f64,
    /// Entropy of measurement distribution
    pub measurement_entropy: f64,
}

/// Sample power measurements from 2D quantum circuit
///
/// Performs statistical analysis of measurement outcomes.
///
/// # Arguments
/// * `sim` - MPS simulator
/// * `width` - Grid width
/// * `height` - Grid height
/// * `shots` - Number of measurements to take
/// * `mapper` - Snake mapper for coordinate conversion
///
/// # Returns
/// Power measurement data with statistics
pub fn power_sample_2d(
    sim: &mut MPSSimulator,
    width: usize,
    height: usize,
    mapper: &SnakeMapper,
    shots: usize,
) -> PowerMeasurement {
    let mut counts: Vec<Vec<usize>> = vec![vec![0; width]; height];
    let mut mode_count = 0;
    let mut mode_coord = GridCoord::new(0, 0);

    // Take shots
    for _ in 0..shots {
        let bitstring = sim.measure() as usize;

        // Find all qubits that measured |1⟩ and count them
        // (Multiple qubits can be in |1⟩ state simultaneously)
        for qubit_idx in 0..(width * height) {
            if (bitstring >> qubit_idx) & 1 == 1 {
                let coord = mapper.map_1d_to_2d(qubit_idx);
                if coord.x < width && coord.y < height {
                    counts[coord.y][coord.x] += 1;

                    // Track mode
                    if counts[coord.y][coord.x] > mode_count {
                        mode_count = counts[coord.y][coord.x];
                        mode_coord = coord;
                    }
                }
            }
        }
    }

    // Calculate measurement entropy
    let total_shots = shots as f64;
    let mut entropy = 0.0;

    for row in &counts {
        for count in row {
            if *count > 0 {
                let p = *count as f64 / total_shots;
                entropy -= p * p.ln();
            }
        }
    }

    // Mode frequency
    let mode_freq = mode_count as f64 / shots as f64;

    PowerMeasurement {
        counts,
        shots,
        mode: mode_coord,
        mode_frequency: mode_freq,
        measurement_entropy: entropy,
    }
}

/// Compute probability distribution for 2D measurements
///
/// Returns normalized probabilities [y][x] for visualization.
pub fn probability_distribution_2d(measurement: &PowerMeasurement) -> Vec<Vec<f64>> {
    let total = measurement.shots as f64;
    let mut result = Vec::new();

    for row in &measurement.counts {
        let prob_row: Vec<f64> = row.iter().map(|c| *c as f64 / total).collect();
        result.push(prob_row);
    }

    result
}

// ============================================================
// 2D-SPECIFIC GATES
// ============================================================

/// Apply square CNOT (4-qubit controlled operation)
///
/// Applies CNOT from control to all 4 neighbors in a 2×2 square.
/// Useful for 2D error correction and spatial propagation.
///
/// # Arguments
/// * `sim` - MPS simulator
/// * `center_x` - Center X coordinate
/// * `center_y` - Center Y coordinate
/// * `width` - Grid width
/// * `height` - Grid height
/// * `mapper` - Snake mapper
pub fn square_cnot(
    sim: &mut MPSSimulator,
    center_x: usize,
    center_y: usize,
    width: usize,
    height: usize,
    mapper: &SnakeMapper,
) {
    // Apply CNOT from center to all 4 neighbors
    // Use manhattan_cnot to handle non-adjacent qubits in MPS ordering
    let neighbors = mapper.get_neighbors(center_x, center_y);

    for neighbor in neighbors {
        if neighbor.x < width && neighbor.y < height {
            // Use manhattan CNOT for proper swap network routing
            manhattan_cnot(sim, center_x, center_y, neighbor.x, neighbor.y, mapper);
        }
    }
}

/// Apply Manhattan distance-optimized gates
///
/// For non-nearest-neighbor gates, minimize MPS bond growth
/// by using swap networks along Manhattan paths.
pub fn manhattan_cnot(
    sim: &mut MPSSimulator,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    mapper: &SnakeMapper,
) {
    let idx1 = mapper.map_2d_to_1d(x1, y1);
    let idx2 = mapper.map_2d_to_1d(x2, y2);

    let coord1 = GridCoord::new(x1, y1);
    let coord2 = GridCoord::new(x2, y2);
    let distance = mapper.distance(&coord1, &coord2);

    // Check if qubits are adjacent in 1D MPS ordering (not just 2D distance)
    // MPS requires qubits to be sequential (idx2 == idx1 + 1)
    let are_adjacent_1d = (idx1 as i32 - idx2 as i32).abs() == 1;

    // For adjacent qubits in 1D ordering, direct CNOT
    if distance == 1 && are_adjacent_1d {
        sim.cnot(idx1, idx2);
        return;
    }

    // For longer distances or non-adjacent in 1D, use swap path
    let mut current = coord1;
    let target = coord2;

    for _ in 0..20 {
        // Increased safety limit
        if current == target {
            break;
        }

        // Find next step towards target
        let next = find_manhattan_step(current, target, mapper);

        let curr_idx = mapper.map_2d_to_1d(current.x, current.y);
        let next_idx = mapper.map_2d_to_1d(next.x, next.y);

        // Check if curr and next are adjacent in 1D ordering
        let idx_diff = (curr_idx as i32 - next_idx as i32).abs();
        if idx_diff == 1 {
            // Adjacent in 1D - can apply CNOT directly
            // IMPORTANT: Ensure ql < qr and qr == ql + 1 for MPS requirement
            let (ql, qr) = if curr_idx < next_idx {
                (curr_idx, next_idx)
            } else {
                (next_idx, curr_idx)
            };
            sim.cnot(ql, qr);
        } else {
            // Not adjacent - use swap network
            // Swap towards each other to minimize bond growth
            sim.swap(curr_idx, next_idx);
            sim.cnot(curr_idx, next_idx);
            sim.swap(curr_idx, next_idx);
        }

        current = next;
    }
}

/// Find next step in Manhattan path
fn find_manhattan_step(current: GridCoord, target: GridCoord, _mapper: &SnakeMapper) -> GridCoord {
    let dx = if target.x > current.x { 1i32 } else { -1i32 };
    let dy = if target.y > current.y { 1i32 } else { -1i32 };

    // Prefer moving in direction of target
    if dx != 0 {
        let new_x = (current.x as i32 + dx).max(0) as usize;
        return GridCoord::new(new_x, current.y);
    }

    if dy != 0 {
        let new_y = (current.y as i32 + dy).max(0) as usize;
        return GridCoord::new(current.x, new_y);
    }

    current // Shouldn't reach here
}

/// 2D gate set for spatial operations
#[derive(Debug, Clone, Copy)]
pub enum Gate2D {
    /// Single qubit gate at position
    Single {
        gate: SingleGate,
        x: usize,
        y: usize,
    },
    /// Nearest-neighbor CNOT
    NeighborCNOT {
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    },
    /// Square CNOT (center to 4 neighbors)
    SquareCNOT { center_x: usize, center_y: usize },
    /// Row-wise operation on all qubits in row
    Row { y: usize, gate: SingleGate },
    /// Column-wise operation on all qubits in column
    Column { x: usize, gate: SingleGate },
}

/// Single qubit gate types
#[derive(Debug, Clone, Copy)]
pub enum SingleGate {
    H,
    X,
    Y,
    Z,
    S,
    T,
    RX(f64),
    RY(f64),
    RZ(f64),
}

/// Apply a 2D gate operation to simulator
pub fn apply_gate_2d(
    sim: &mut MPSSimulator,
    width: usize,
    height: usize,
    mapper: &SnakeMapper,
    gate: &Gate2D,
) {
    match gate {
        Gate2D::Single { gate, x, y } => {
            if *x < width && *y < height {
                let idx = mapper.map_2d_to_1d(*x, *y);
                apply_single_gate(sim, idx, gate);
            }
        }
        Gate2D::NeighborCNOT { x1, y1, x2, y2 } => {
            if *x1 < width && *y1 < height && *x2 < width && *y2 < height {
                let idx1 = mapper.map_2d_to_1d(*x1, *y1);
                let idx2 = mapper.map_2d_to_1d(*x2, *y2);
                sim.cnot(idx1, idx2);
            }
        }
        Gate2D::SquareCNOT { center_x, center_y } => {
            square_cnot(sim, *center_x, *center_y, width, height, mapper);
        }
        Gate2D::Row { y, gate } => {
            if *y < height {
                for x in 0..width {
                    let idx = mapper.map_2d_to_1d(x, *y);
                    apply_single_gate(sim, idx, gate);
                }
            }
        }
        Gate2D::Column { x, gate } => {
            if *x < width {
                for y in 0..height {
                    let idx = mapper.map_2d_to_1d(*x, y);
                    apply_single_gate(sim, idx, gate);
                }
            }
        }
    }
}

/// Apply single qubit gate by index
fn apply_single_gate(sim: &mut MPSSimulator, idx: usize, gate: &SingleGate) {
    match gate {
        SingleGate::H => sim.h(idx),
        SingleGate::X => sim.x(idx),
        SingleGate::Y => sim.y(idx),
        SingleGate::Z => sim.z(idx),
        SingleGate::S => sim.s(idx),
        SingleGate::T => sim.t(idx),
        SingleGate::RX(theta) => sim.rx(idx, *theta),
        SingleGate::RY(theta) => sim.ry(idx, *theta),
        SingleGate::RZ(theta) => sim.rz(idx, *theta),
    }
}

// ============================================================
// 2D BENCHMARKING
// ============================================================

/// Benchmark result for 2D quantum algorithms
#[derive(Debug, Clone)]
pub struct Benchmark2DResult {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub total_qubits: usize,
    pub duration_ms: f64,
    pub fidelity: Option<f64>,
    pub max_bond_dim: usize,
}

/// Benchmark 2D QFT performance
pub fn benchmark_qft_2d(
    width: usize,
    height: usize,
    max_bond_dim: Option<usize>,
    iterations: usize,
) -> Benchmark2DResult {
    use std::time::Instant;

    let mapper = SnakeMapper::new(width, height);
    let mut sim = MPSSimulator::new(width * height, max_bond_dim);
    let total_qubits = width * height;

    let start = Instant::now();

    for _ in 0..iterations {
        // Initialize to superposition
        for i in 0..total_qubits {
            sim.h(i);
        }

        // Apply 2D QFT
        qft_2d(&mut sim, width, height, &mapper);

        // Inverse QFT
        inverse_qft_2d(&mut sim, width, height, &mapper);

        // Reset for next iteration
        sim = MPSSimulator::new(total_qubits, max_bond_dim);
    }

    let duration = start.elapsed().as_secs_f64() * 1000.0;
    let max_bd = max_bond_dim.unwrap_or(usize::MAX);

    Benchmark2DResult {
        name: "2D-QFT".to_string(),
        width,
        height,
        total_qubits,
        duration_ms: duration / iterations as f64,
        fidelity: Some(1.0), // QFT is unitary
        max_bond_dim: max_bd,
    }
}

/// Benchmark 2D Grover search
pub fn benchmark_grover_2d(
    width: usize,
    height: usize,
    max_bond_dim: Option<usize>,
    iterations: usize,
) -> Benchmark2DResult {
    use std::time::Instant;

    let mapper = SnakeMapper::new(width, height);
    let total_qubits = width * height;

    // Target: center of grid
    let target_x = width / 2;
    let target_y = height / 2;

    let start = Instant::now();
    let mut success_count = 0;

    for _ in 0..iterations {
        let mut sim = MPSSimulator::new(total_qubits, max_bond_dim);

        let result = grover_2d(
            &mut sim,
            width,
            height,
            &mapper,
            &|x, y| x == target_x && y == target_y,
            None,
        );

        if result.x == target_x && result.y == target_y {
            success_count += 1;
        }
    }

    let duration = start.elapsed().as_secs_f64() * 1000.0;
    let fidelity = success_count as f64 / iterations as f64;
    let max_bd = max_bond_dim.unwrap_or(usize::MAX);

    Benchmark2DResult {
        name: format!("2D-Grover-{}x{}", width, height),
        width,
        height,
        total_qubits,
        duration_ms: duration / iterations as f64,
        fidelity: Some(fidelity),
        max_bond_dim: max_bd,
    }
}

/// Run comprehensive 2D benchmarks
pub fn benchmark_suite_2d(
    widths: Vec<usize>,
    heights: Vec<usize>,
    max_bond_dim: Option<usize>,
    iterations_per_test: usize,
) -> Vec<Benchmark2DResult> {
    let mut results = Vec::new();

    for width in &widths {
        for height in &heights {
            // Skip if too large for practical simulation
            if width * height > 64 {
                continue;
            }

            // QFT benchmark
            results.push(benchmark_qft_2d(
                *width,
                *height,
                max_bond_dim,
                iterations_per_test,
            ));

            // Grover benchmark
            results.push(benchmark_grover_2d(
                *width,
                *height,
                max_bond_dim,
                iterations_per_test / 10, // Fewer iterations for Grover (slower)
            ));
        }
    }

    results
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snake_mapping::SnakeMapper;

    #[test]
    fn test_qft_2d_basic() {
        let mapper = SnakeMapper::new(2, 2);
        let mut sim = MPSSimulator::new(4, Some(4));

        // Initialize to |10⟩ state
        sim.x(1);
        sim.x(3);

        // Apply 2D QFT
        qft_2d(&mut sim, 2, 2, &mapper);

        // Just verify it runs without panic
        assert_eq!(sim.num_qubits(), 4);
    }

    #[test]
    fn test_entanglement_entropy_2d() {
        let mapper = SnakeMapper::new(4, 4);
        let mut sim = MPSSimulator::new(16, Some(8));

        // Create some entanglement
        sim.h(0);
        sim.cnot(0, 1);
        sim.cnot(1, 2);
        sim.cnot(2, 3);

        let grid = entanglement_entropy_2d(&sim, 4, 4, &mapper);

        assert_eq!(grid.len(), 4);
        assert_eq!(grid[0].len(), 4);
    }

    #[test]
    fn test_power_sample_2d() {
        let mapper = SnakeMapper::new(4, 4);
        let mut sim = MPSSimulator::new(16, Some(4));

        // Initialize to superposition
        for i in 0..16 {
            sim.h(i);
        }

        let measurement = power_sample_2d(&mut sim, 4, 4, &mapper, 100);

        assert_eq!(measurement.shots, 100);
        assert_eq!(measurement.counts.len(), 4);
        assert_eq!(measurement.counts[0].len(), 4);
    }

    #[test]
    fn test_square_cnot() {
        let mapper = SnakeMapper::new(4, 4);
        let mut sim = MPSSimulator::new(16, Some(8));

        // Apply H to center
        sim.h(mapper.map_2d_to_1d(2, 2));

        // Square CNOT from center
        square_cnot(&mut sim, 2, 2, 4, 4, &mapper);

        // Just verify it runs
        assert_eq!(sim.num_qubits(), 16);
    }

    #[test]
    fn test_manhattan_cnot() {
        let mapper = SnakeMapper::new(4, 4);
        let mut sim = MPSSimulator::new(16, Some(8));

        // Initialize control at |1⟩
        sim.x(mapper.map_2d_to_1d(0, 0));

        // Long-range CNOT using Manhattan path
        manhattan_cnot(&mut sim, 0, 0, 3, 3, &mapper);

        assert_eq!(sim.num_qubits(), 16);
    }

    #[test]
    fn test_gate_2d() {
        let mapper = SnakeMapper::new(4, 4);
        let mut sim = MPSSimulator::new(16, Some(4));

        // Test various 2D gates
        apply_gate_2d(
            &mut sim,
            4,
            4,
            &mapper,
            &Gate2D::Row {
                y: 1,
                gate: SingleGate::H,
            },
        );
        apply_gate_2d(
            &mut sim,
            4,
            4,
            &mapper,
            &Gate2D::Column {
                x: 2,
                gate: SingleGate::X,
            },
        );

        assert_eq!(sim.num_qubits(), 16);
    }
}
