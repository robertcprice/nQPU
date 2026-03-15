//! Lattice-aware MPS simulators for 2D/3D grids.
//!
//! These map higher-dimensional lattices to a 1D MPS chain using
//! snake/serpentine orderings to keep neighbor distances small.
//!
//! # Features
//! - Checkpointing: Save/restore MPS state at any point
//! - Entropy tracking: Monitor entanglement growth during evolution
//! - Entanglement heatmaps: 2D/3D visualization of bond entanglement

use num_complex::Complex64;
use std::collections::HashMap;

use crate::entanglement_scheduler::{
    apply_gate_with_swap_network, cz_gate, greedy_layers, order_edges_by_entanglement,
    Edge as GateEdge,
};
use crate::snake_mapping::{GridCoord, SnakeMapper};
use crate::tensor_network::MPSSimulator;

/// MPS checkpoint for state persistence.
#[derive(Clone, Debug)]
pub struct MPSCheckpoint {
    /// Timestamp/step when checkpoint was created.
    pub step: usize,
    /// Entanglement profile at checkpoint time.
    pub entanglement: Vec<f64>,
    /// Bond dimensions at checkpoint time.
    pub bond_dims: Vec<usize>,
    /// Tensor shapes (left, phys, right) for each tensor.
    pub tensor_shapes: Vec<(usize, usize, usize)>,
    /// Serialized tensor data (simplified representation).
    pub tensors: Vec<Vec<Complex64>>,
}

/// Entropy tracking record for analysis.
#[derive(Clone, Debug, Default)]
pub struct EntropyRecord {
    /// Total entanglement entropy at this step.
    pub total_entropy: f64,
    /// Maximum bond entanglement.
    pub max_entropy: f64,
    /// Maximum bond dimension.
    pub max_bond_dim: usize,
    /// Per-bond entropy values.
    pub bond_entropies: Vec<f64>,
}

/// Entanglement heatmap for 2D lattices.
#[derive(Clone, Debug)]
pub struct EntanglementHeatmap2D {
    pub width: usize,
    pub height: usize,
    /// Horizontal bonds (width-1 x height).
    pub horizontal_bonds: Vec<Vec<f64>>,
    /// Vertical bonds (width x height-1).
    pub vertical_bonds: Vec<Vec<f64>>,
}

/// Entanglement heatmap for 3D lattices.
#[derive(Clone, Debug)]
pub struct EntanglementHeatmap3D {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    /// X-direction bonds.
    pub x_bonds: Vec<Vec<Vec<f64>>>,
    /// Y-direction bonds.
    pub y_bonds: Vec<Vec<Vec<f64>>>,
    /// Z-direction bonds.
    pub z_bonds: Vec<Vec<Vec<f64>>>,
}

/// 3D grid coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridCoord3D {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl GridCoord3D {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }
}

/// Snake mapper for 3D grids (serpentine 2D layers, alternating layer order).
#[derive(Debug, Clone)]
pub struct SnakeMapper3D {
    width: usize,
    height: usize,
    depth: usize,
    layer: SnakeMapper,
}

impl SnakeMapper3D {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let layer = SnakeMapper::new(width, height);
        Self {
            width,
            height,
            depth,
            layer,
        }
    }

    pub fn size(&self) -> usize {
        self.width * self.height * self.depth
    }

    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    pub fn map_3d_to_1d(&self, x: usize, y: usize, z: usize) -> usize {
        assert!(x < self.width, "x out of bounds");
        assert!(y < self.height, "y out of bounds");
        assert!(z < self.depth, "z out of bounds");
        let layer_size = self.width * self.height;
        let local = self.layer.map_2d_to_1d(x, y);
        let local_maybe_reversed = if z % 2 == 0 {
            local
        } else {
            layer_size - 1 - local
        };
        z * layer_size + local_maybe_reversed
    }

    pub fn map_1d_to_3d(&self, index: usize) -> GridCoord3D {
        let layer_size = self.width * self.height;
        assert!(index < layer_size * self.depth, "index out of bounds");
        let z = index / layer_size;
        let in_layer = index % layer_size;
        let local = if z % 2 == 0 {
            in_layer
        } else {
            layer_size - 1 - in_layer
        };
        let coord2d = self.layer.map_1d_to_2d(local);
        GridCoord3D::new(coord2d.x, coord2d.y, z)
    }

    /// Get 3D nearest neighbors (6-connectivity).
    pub fn neighbors(&self, x: usize, y: usize, z: usize) -> Vec<GridCoord3D> {
        let mut out = Vec::new();
        if x > 0 {
            out.push(GridCoord3D::new(x - 1, y, z));
        }
        if x + 1 < self.width {
            out.push(GridCoord3D::new(x + 1, y, z));
        }
        if y > 0 {
            out.push(GridCoord3D::new(x, y - 1, z));
        }
        if y + 1 < self.height {
            out.push(GridCoord3D::new(x, y + 1, z));
        }
        if z > 0 {
            out.push(GridCoord3D::new(x, y, z - 1));
        }
        if z + 1 < self.depth {
            out.push(GridCoord3D::new(x, y, z + 1));
        }
        out
    }
}

fn zz_gate(theta: f64) -> [[Complex64; 4]; 4] {
    let c = theta.cos();
    let s = theta.sin();
    let e_minus = Complex64::new(c, -s);
    let e_plus = Complex64::new(c, s);
    [
        [
            e_minus,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            e_plus,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            e_plus,
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            e_minus,
        ],
    ]
}

fn xy_gate(theta: f64) -> [[Complex64; 4]; 4] {
    let c = theta.cos();
    let s = theta.sin();
    let minus_i_s = Complex64::new(0.0, -s);
    [
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(c, 0.0),
            minus_i_s,
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            minus_i_s,
            Complex64::new(c, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    ]
}

/// 2D lattice MPS simulator using snake mapping.
pub struct LatticeMPS2D {
    width: usize,
    height: usize,
    mapper: SnakeMapper,
    sim: MPSSimulator,
    /// Checkpoint storage.
    checkpoints: HashMap<String, MPSCheckpoint>,
    /// Entropy history tracking.
    entropy_history: Vec<EntropyRecord>,
    /// Current step counter.
    step: usize,
}

impl LatticeMPS2D {
    pub fn new(width: usize, height: usize, max_bond_dim: Option<usize>) -> Self {
        let mapper = SnakeMapper::new(width, height);
        let sim = MPSSimulator::new(width * height, max_bond_dim);
        Self {
            width,
            height,
            mapper,
            sim,
            checkpoints: HashMap::new(),
            entropy_history: Vec::new(),
            step: 0,
        }
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        self.mapper.map_2d_to_1d(x, y)
    }

    pub fn h(&mut self, x: usize, y: usize) {
        let q = self.idx(x, y);
        self.sim.h(q);
    }

    pub fn prepare_plus_state(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.h(x, y);
            }
        }
    }

    pub fn cnot(&mut self, control: GridCoord, target: GridCoord) {
        let qc = self.idx(control.x, control.y);
        let qt = self.idx(target.x, target.y);
        self.sim.cnot(qc, qt);
    }

    pub fn cz(&mut self, control: GridCoord, target: GridCoord) {
        let qc = self.idx(control.x, control.y);
        let qt = self.idx(target.x, target.y);
        self.sim.cz(qc, qt);
    }

    /// Prepare a 2D cluster state on the lattice (|+> with CZ on edges).
    pub fn apply_cluster_state(&mut self) {
        self.prepare_plus_state();
        let gate = cz_gate();
        for y in 0..self.height {
            for x in 0..self.width {
                let q = self.idx(x, y);
                if x + 1 < self.width {
                    let q2 = self.idx(x + 1, y);
                    apply_gate_with_swap_network(&mut self.sim, q, q2, &gate);
                }
                if y + 1 < self.height {
                    let q2 = self.idx(x, y + 1);
                    apply_gate_with_swap_network(&mut self.sim, q, q2, &gate);
                }
            }
        }
    }

    /// Apply a checkerboard ZZ layer for 2D nearest neighbors.
    pub fn apply_ising_layer(&mut self, theta: f64) {
        let gate = zz_gate(theta);
        for pass in 0..2 {
            for y in 0..self.height {
                for x in 0..self.width {
                    if (x + y) % 2 != pass {
                        continue;
                    }
                    if x + 1 < self.width {
                        let q1 = self.idx(x, y);
                        let q2 = self.idx(x + 1, y);
                        apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                    }
                    if y + 1 < self.height {
                        let q1 = self.idx(x, y);
                        let q2 = self.idx(x, y + 1);
                        apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                    }
                }
            }
        }
    }

    /// Apply a checkerboard XY layer for 2D nearest neighbors.
    pub fn apply_xy_layer(&mut self, theta: f64) {
        let gate = xy_gate(theta);
        for pass in 0..2 {
            for y in 0..self.height {
                for x in 0..self.width {
                    if (x + y) % 2 != pass {
                        continue;
                    }
                    if x + 1 < self.width {
                        let q1 = self.idx(x, y);
                        let q2 = self.idx(x + 1, y);
                        apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                    }
                    if y + 1 < self.height {
                        let q1 = self.idx(x, y);
                        let q2 = self.idx(x, y + 1);
                        apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                    }
                }
            }
        }
    }

    /// Apply mixer layer (Rx) across all qubits.
    pub fn apply_mixer_layer(&mut self, beta: f64) {
        let angle = 2.0 * beta;
        for y in 0..self.height {
            for x in 0..self.width {
                let q = self.idx(x, y);
                self.sim.rx(q, angle);
            }
        }
    }

    /// Run a simple QAOA Ising routine on the 2D lattice.
    pub fn run_qaoa_ising(&mut self, p: usize, gamma: f64, beta: f64) {
        self.prepare_plus_state();
        for _ in 0..p {
            self.apply_ising_layer(gamma);
            self.apply_mixer_layer(beta);
        }
    }

    pub fn set_truncation_threshold(&mut self, threshold: f64) {
        self.sim.set_truncation_threshold(threshold);
    }

    pub fn set_relative_truncation(&mut self, threshold: f64) {
        self.sim.set_relative_truncation(threshold);
    }

    pub fn enable_entanglement_tracking(&mut self, enabled: bool) {
        self.sim.enable_entanglement_tracking(enabled);
    }

    pub fn bond_entanglement_entropy(&self, bond: usize) -> Option<f64> {
        self.sim.bond_entanglement_entropy(bond)
    }

    pub fn max_bond_dim(&self) -> usize {
        self.sim.max_bond_dim()
    }

    pub fn measure(&mut self) -> usize {
        self.sim.measure()
    }

    // ================== Checkpointing ==================

    /// Save current state as a named checkpoint.
    pub fn save_checkpoint(&mut self, name: &str) {
        let entanglement: Vec<f64> = (0..self.width * self.height - 1)
            .filter_map(|b| self.sim.bond_entanglement_entropy(b))
            .collect();
        let bond_dims = self.sim.bond_dimensions();
        let tensor_shapes = self.sim.tensor_shapes();
        let tensors = self.sim.tensor_data();

        self.checkpoints.insert(
            name.to_string(),
            MPSCheckpoint {
                step: self.step,
                entanglement,
                bond_dims,
                tensor_shapes,
                tensors,
            },
        );
    }

    /// Restore state from a named checkpoint.
    pub fn restore_checkpoint(&mut self, name: &str) -> Result<(), String> {
        if let Some(cp) = self.checkpoints.get(name).cloned() {
            self.step = cp.step;
            self.sim
                .restore_tensors_with_shapes(&cp.tensors, &cp.tensor_shapes)?;
            Ok(())
        } else {
            Err(format!("Checkpoint '{}' not found", name))
        }
    }

    /// List available checkpoints.
    pub fn list_checkpoints(&self) -> Vec<&String> {
        self.checkpoints.keys().collect()
    }

    /// Delete a checkpoint.
    pub fn delete_checkpoint(&mut self, name: &str) -> bool {
        self.checkpoints.remove(name).is_some()
    }

    // ================== Entropy Tracking ==================

    /// Record current entropy state.
    pub fn record_entropy(&mut self) {
        let mut bond_entropies = Vec::new();
        let mut total_entropy = 0.0;
        let mut max_entropy = 0.0;

        for b in 0..(self.width * self.height - 1) {
            if let Some(e) = self.sim.bond_entanglement_entropy(b) {
                bond_entropies.push(e);
                total_entropy += e;
                if e > max_entropy {
                    max_entropy = e;
                }
            }
        }

        self.entropy_history.push(EntropyRecord {
            total_entropy,
            max_entropy,
            max_bond_dim: self.sim.max_bond_dim(),
            bond_entropies,
        });
        self.step += 1;
    }

    /// Get entropy history.
    pub fn entropy_history(&self) -> &[EntropyRecord] {
        &self.entropy_history
    }

    /// Clear entropy history.
    pub fn clear_entropy_history(&mut self) {
        self.entropy_history.clear();
    }

    // ================== Entanglement Heatmap ==================

    /// Extract 2D entanglement heatmap from bond entropies.
    pub fn entanglement_heatmap(&self) -> EntanglementHeatmap2D {
        let mut horizontal = vec![vec![0.0; self.height]; self.width - 1];
        let mut vertical = vec![vec![0.0; self.height - 1]; self.width];

        // Map 1D bond entropies to 2D lattice edges
        for y in 0..self.height {
            for x in 0..self.width {
                let q1 = self.idx(x, y);

                // Horizontal bond to right neighbor
                if x + 1 < self.width {
                    let q2 = self.idx(x + 1, y);
                    let bond = q1.min(q2);
                    if let Some(e) = self.sim.bond_entanglement_entropy(bond) {
                        horizontal[x][y] = e;
                    }
                }

                // Vertical bond to bottom neighbor
                if y + 1 < self.height {
                    let q2 = self.idx(x, y + 1);
                    let bond = q1.min(q2);
                    if let Some(e) = self.sim.bond_entanglement_entropy(bond) {
                        vertical[x][y] = e;
                    }
                }
            }
        }

        EntanglementHeatmap2D {
            width: self.width,
            height: self.height,
            horizontal_bonds: horizontal,
            vertical_bonds: vertical,
        }
    }

    /// Apply entanglement-aware scheduled CZ gates.
    pub fn apply_scheduled_cz(&mut self, edges: &[(usize, usize, usize, usize)]) {
        let gate = cz_gate();
        let gate_edges: Vec<GateEdge> = edges
            .iter()
            .map(|(x1, y1, x2, y2)| {
                let q1 = self.idx(*x1, *y1);
                let q2 = self.idx(*x2, *y2);
                GateEdge::new(q1, q2, 1.0)
            })
            .collect();

        let ordered = order_edges_by_entanglement(&self.sim, &gate_edges);
        let layers = greedy_layers(&ordered);

        for layer in layers {
            for edge in layer {
                apply_gate_with_swap_network(&mut self.sim, edge.q1, edge.q2, &gate);
            }
            self.record_entropy();
        }
    }
}

/// 3D lattice MPS simulator using snake mapping.
pub struct LatticeMPS3D {
    width: usize,
    height: usize,
    depth: usize,
    mapper: SnakeMapper3D,
    sim: MPSSimulator,
    /// Checkpoint storage.
    checkpoints: HashMap<String, MPSCheckpoint>,
    /// Entropy history tracking.
    entropy_history: Vec<EntropyRecord>,
    /// Current step counter.
    step: usize,
}

impl LatticeMPS3D {
    pub fn new(width: usize, height: usize, depth: usize, max_bond_dim: Option<usize>) -> Self {
        let mapper = SnakeMapper3D::new(width, height, depth);
        let sim = MPSSimulator::new(width * height * depth, max_bond_dim);
        Self {
            width,
            height,
            depth,
            mapper,
            sim,
            checkpoints: HashMap::new(),
            entropy_history: Vec::new(),
            step: 0,
        }
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        self.mapper.map_3d_to_1d(x, y, z)
    }

    pub fn h(&mut self, x: usize, y: usize, z: usize) {
        let q = self.idx(x, y, z);
        self.sim.h(q);
    }

    pub fn prepare_plus_state(&mut self) {
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    self.h(x, y, z);
                }
            }
        }
    }

    /// Prepare a 3D cluster state on the lattice (|+> with CZ on edges).
    pub fn apply_cluster_state(&mut self) {
        self.prepare_plus_state();
        let gate = cz_gate();
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let q = self.idx(x, y, z);
                    if x + 1 < self.width {
                        let q2 = self.idx(x + 1, y, z);
                        apply_gate_with_swap_network(&mut self.sim, q, q2, &gate);
                    }
                    if y + 1 < self.height {
                        let q2 = self.idx(x, y + 1, z);
                        apply_gate_with_swap_network(&mut self.sim, q, q2, &gate);
                    }
                    if z + 1 < self.depth {
                        let q2 = self.idx(x, y, z + 1);
                        apply_gate_with_swap_network(&mut self.sim, q, q2, &gate);
                    }
                }
            }
        }
    }

    /// Apply a checkerboard ZZ layer for 3D nearest neighbors.
    pub fn apply_ising_layer(&mut self, theta: f64) {
        let gate = zz_gate(theta);
        for pass in 0..2 {
            for z in 0..self.depth {
                for y in 0..self.height {
                    for x in 0..self.width {
                        if (x + y + z) % 2 != pass {
                            continue;
                        }
                        if x + 1 < self.width {
                            let q1 = self.idx(x, y, z);
                            let q2 = self.idx(x + 1, y, z);
                            apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                        }
                        if y + 1 < self.height {
                            let q1 = self.idx(x, y, z);
                            let q2 = self.idx(x, y + 1, z);
                            apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                        }
                        if z + 1 < self.depth {
                            let q1 = self.idx(x, y, z);
                            let q2 = self.idx(x, y, z + 1);
                            apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                        }
                    }
                }
            }
        }
    }

    /// Apply a checkerboard XY layer for 3D nearest neighbors.
    pub fn apply_xy_layer(&mut self, theta: f64) {
        let gate = xy_gate(theta);
        for pass in 0..2 {
            for z in 0..self.depth {
                for y in 0..self.height {
                    for x in 0..self.width {
                        if (x + y + z) % 2 != pass {
                            continue;
                        }
                        if x + 1 < self.width {
                            let q1 = self.idx(x, y, z);
                            let q2 = self.idx(x + 1, y, z);
                            apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                        }
                        if y + 1 < self.height {
                            let q1 = self.idx(x, y, z);
                            let q2 = self.idx(x, y + 1, z);
                            apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                        }
                        if z + 1 < self.depth {
                            let q1 = self.idx(x, y, z);
                            let q2 = self.idx(x, y, z + 1);
                            apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                        }
                    }
                }
            }
        }
    }

    /// Apply mixer layer (Rx) across all qubits.
    pub fn apply_mixer_layer(&mut self, beta: f64) {
        let angle = 2.0 * beta;
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let q = self.idx(x, y, z);
                    self.sim.rx(q, angle);
                }
            }
        }
    }

    /// Run a simple QAOA Ising routine on the 3D lattice.
    pub fn run_qaoa_ising(&mut self, p: usize, gamma: f64, beta: f64) {
        self.prepare_plus_state();
        for _ in 0..p {
            self.apply_ising_layer(gamma);
            self.apply_mixer_layer(beta);
        }
    }

    pub fn set_truncation_threshold(&mut self, threshold: f64) {
        self.sim.set_truncation_threshold(threshold);
    }

    pub fn set_relative_truncation(&mut self, threshold: f64) {
        self.sim.set_relative_truncation(threshold);
    }

    pub fn enable_entanglement_tracking(&mut self, enabled: bool) {
        self.sim.enable_entanglement_tracking(enabled);
    }

    pub fn bond_entanglement_entropy(&self, bond: usize) -> Option<f64> {
        self.sim.bond_entanglement_entropy(bond)
    }

    pub fn max_bond_dim(&self) -> usize {
        self.sim.max_bond_dim()
    }

    pub fn measure(&mut self) -> usize {
        self.sim.measure()
    }

    // ================== Checkpointing ==================

    /// Save current state as a named checkpoint.
    pub fn save_checkpoint(&mut self, name: &str) {
        let n_qubits = self.width * self.height * self.depth;
        let entanglement: Vec<f64> = (0..n_qubits - 1)
            .filter_map(|b| self.sim.bond_entanglement_entropy(b))
            .collect();
        let bond_dims = self.sim.bond_dimensions();
        let tensor_shapes = self.sim.tensor_shapes();
        let tensors = self.sim.tensor_data();

        self.checkpoints.insert(
            name.to_string(),
            MPSCheckpoint {
                step: self.step,
                entanglement,
                bond_dims,
                tensor_shapes,
                tensors,
            },
        );
    }

    /// Restore state from a named checkpoint.
    pub fn restore_checkpoint(&mut self, name: &str) -> Result<(), String> {
        if let Some(cp) = self.checkpoints.get(name).cloned() {
            self.step = cp.step;
            self.sim
                .restore_tensors_with_shapes(&cp.tensors, &cp.tensor_shapes)?;
            Ok(())
        } else {
            Err(format!("Checkpoint '{}' not found", name))
        }
    }

    /// List available checkpoints.
    pub fn list_checkpoints(&self) -> Vec<&String> {
        self.checkpoints.keys().collect()
    }

    // ================== Entropy Tracking ==================

    /// Record current entropy state.
    pub fn record_entropy(&mut self) {
        let n_qubits = self.width * self.height * self.depth;
        let mut bond_entropies = Vec::new();
        let mut total_entropy = 0.0;
        let mut max_entropy = 0.0;

        for b in 0..(n_qubits - 1) {
            if let Some(e) = self.sim.bond_entanglement_entropy(b) {
                bond_entropies.push(e);
                total_entropy += e;
                if e > max_entropy {
                    max_entropy = e;
                }
            }
        }

        self.entropy_history.push(EntropyRecord {
            total_entropy,
            max_entropy,
            max_bond_dim: self.sim.max_bond_dim(),
            bond_entropies,
        });
        self.step += 1;
    }

    /// Get entropy history.
    pub fn entropy_history(&self) -> &[EntropyRecord] {
        &self.entropy_history
    }

    // ================== Entanglement Heatmap ==================

    /// Extract 3D entanglement heatmap from bond entropies.
    pub fn entanglement_heatmap(&self) -> EntanglementHeatmap3D {
        let mut x_bonds = vec![vec![vec![0.0; self.depth]; self.height]; self.width - 1];
        let mut y_bonds = vec![vec![vec![0.0; self.depth]; self.height - 1]; self.width];
        let mut z_bonds = vec![vec![vec![0.0; self.depth - 1]; self.height]; self.width];

        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let q1 = self.idx(x, y, z);

                    // X-direction bond
                    if x + 1 < self.width {
                        let q2 = self.idx(x + 1, y, z);
                        let bond = q1.min(q2);
                        if let Some(e) = self.sim.bond_entanglement_entropy(bond) {
                            x_bonds[x][y][z] = e;
                        }
                    }

                    // Y-direction bond
                    if y + 1 < self.height {
                        let q2 = self.idx(x, y + 1, z);
                        let bond = q1.min(q2);
                        if let Some(e) = self.sim.bond_entanglement_entropy(bond) {
                            y_bonds[x][y][z] = e;
                        }
                    }

                    // Z-direction bond
                    if z + 1 < self.depth {
                        let q2 = self.idx(x, y, z + 1);
                        let bond = q1.min(q2);
                        if let Some(e) = self.sim.bond_entanglement_entropy(bond) {
                            z_bonds[x][y][z] = e;
                        }
                    }
                }
            }
        }

        EntanglementHeatmap3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            x_bonds,
            y_bonds,
            z_bonds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_mapper_3d_roundtrip() {
        let mapper = SnakeMapper3D::new(3, 2, 2);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..3 {
                    let idx = mapper.map_3d_to_1d(x, y, z);
                    let coord = mapper.map_1d_to_3d(idx);
                    assert_eq!(coord, GridCoord3D::new(x, y, z));
                }
            }
        }
    }

    #[test]
    fn test_lattice_mps_2d_ising() {
        let mut sim = LatticeMPS2D::new(2, 2, Some(4));
        sim.h(0, 0);
        sim.apply_ising_layer(0.2);
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_mps_2d_xy() {
        let mut sim = LatticeMPS2D::new(2, 2, Some(4));
        sim.h(0, 0);
        sim.apply_xy_layer(0.2);
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_mps_2d_cluster() {
        let mut sim = LatticeMPS2D::new(2, 2, Some(4));
        sim.apply_cluster_state();
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_mps_3d_ising() {
        let mut sim = LatticeMPS3D::new(2, 2, 2, Some(4));
        sim.h(0, 0, 0);
        sim.apply_ising_layer(0.2);
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_mps_3d_xy() {
        let mut sim = LatticeMPS3D::new(2, 2, 2, Some(4));
        sim.h(0, 0, 0);
        sim.apply_xy_layer(0.2);
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_mps_3d_cluster() {
        let mut sim = LatticeMPS3D::new(2, 2, 2, Some(4));
        sim.apply_cluster_state();
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_2d_checkpointing() {
        let mut sim = LatticeMPS2D::new(2, 2, Some(4));
        sim.h(0, 0);
        sim.h(1, 0);
        sim.save_checkpoint("initial");

        sim.apply_ising_layer(0.5);
        sim.save_checkpoint("after_ising");

        // Verify checkpoints exist
        assert!(sim.list_checkpoints().contains(&&"initial".to_string()));
        assert!(sim.list_checkpoints().contains(&&"after_ising".to_string()));

        // Restore and verify
        sim.restore_checkpoint("initial").unwrap();
        assert!(sim.delete_checkpoint("after_ising"));
        assert!(!sim.list_checkpoints().contains(&&"after_ising".to_string()));
    }

    #[test]
    fn test_lattice_2d_entropy_tracking() {
        let mut sim = LatticeMPS2D::new(2, 2, Some(8));
        sim.enable_entanglement_tracking(true);

        sim.prepare_plus_state();
        sim.record_entropy();

        sim.apply_ising_layer(0.3);
        sim.record_entropy();

        let history = sim.entropy_history();
        assert_eq!(history.len(), 2);
        // Entropy should increase after entangling operations
        assert!(history[1].total_entropy >= history[0].total_entropy);
    }

    #[test]
    fn test_lattice_2d_heatmap() {
        let mut sim = LatticeMPS2D::new(3, 3, Some(8));
        sim.prepare_plus_state();
        sim.apply_ising_layer(0.3);

        let heatmap = sim.entanglement_heatmap();
        assert_eq!(heatmap.horizontal_bonds.len(), 2); // width - 1
        assert_eq!(heatmap.vertical_bonds.len(), 3); // width
    }

    #[test]
    fn test_lattice_3d_checkpointing() {
        let mut sim = LatticeMPS3D::new(2, 2, 2, Some(4));
        sim.h(0, 0, 0);
        sim.save_checkpoint("start");

        sim.apply_cluster_state();
        sim.save_checkpoint("cluster");

        // Restore and verify
        sim.restore_checkpoint("start").unwrap();
        assert!(sim.list_checkpoints().contains(&&"start".to_string()));
    }

    #[test]
    fn test_lattice_3d_heatmap() {
        let mut sim = LatticeMPS3D::new(2, 2, 2, Some(8));
        sim.prepare_plus_state();
        sim.apply_ising_layer(0.3);

        let heatmap = sim.entanglement_heatmap();
        assert_eq!(heatmap.x_bonds.len(), 1); // width - 1
        assert_eq!(heatmap.y_bonds.len(), 2); // width
        assert_eq!(heatmap.z_bonds.len(), 2); // width
    }

    #[test]
    fn test_scheduled_cz_gates() {
        let mut sim = LatticeMPS2D::new(3, 3, Some(8));
        sim.prepare_plus_state();

        // Apply CZ to some edges
        let edges = vec![
            (0, 0, 1, 0), // horizontal edge
            (0, 1, 0, 2), // vertical edge
            (1, 1, 2, 1), // another horizontal
        ];
        sim.apply_scheduled_cz(&edges);

        // Should have recorded entropy for each layer
        assert!(!sim.entropy_history().is_empty());
    }
}
