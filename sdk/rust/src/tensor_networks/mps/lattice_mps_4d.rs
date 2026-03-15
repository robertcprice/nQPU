//! 4D Lattice MPS simulation with hypercubic geometry.
//!
//! Extends 2D/3D lattice support to 4D for:
//! - Lattice gauge theory simulations
//! - Higher-dimensional quantum field theories
//! - Hypercubic code research
//! - Novel entanglement structures

use num_complex::Complex64;
use std::collections::HashMap;

use crate::entanglement_scheduler::{apply_gate_with_swap_network, cz_gate};
use crate::tensor_network::MPSSimulator;

/// Zero constant for Complex64.
const ZERO: Complex64 = Complex64 { re: 0.0, im: 0.0 };

/// 4D grid coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridCoord4D {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub w: usize,
}

impl GridCoord4D {
    pub fn new(x: usize, y: usize, z: usize, w: usize) -> Self {
        Self { x, y, z, w }
    }
}

/// Snake mapper for 4D grids using nested serpentine orderings.
#[derive(Debug, Clone)]
pub struct SnakeMapper4D {
    dims: [usize; 4],
}

impl SnakeMapper4D {
    pub fn new(nx: usize, ny: usize, nz: usize, nw: usize) -> Self {
        Self {
            dims: [nx, ny, nz, nw],
        }
    }

    pub fn size(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]
    }

    pub fn dimensions(&self) -> (usize, usize, usize, usize) {
        (self.dims[0], self.dims[1], self.dims[2], self.dims[3])
    }

    /// Map 4D coordinates to 1D index using hyper-serpentine ordering.
    pub fn map_4d_to_1d(&self, x: usize, y: usize, z: usize, w: usize) -> usize {
        assert!(x < self.dims[0] && y < self.dims[1] && z < self.dims[2] && w < self.dims[3]);

        let layer_size = self.dims[0] * self.dims[1] * self.dims[2];
        let _plane_size = self.dims[0] * self.dims[1];

        // W dimension alternates direction
        let local_w = if w % 2 == 0 {
            // Forward: z serpentine
            self.compute_3d_index(x, y, z)
        } else {
            // Backward: reverse serpentine
            layer_size - 1 - self.compute_3d_index(x, y, z)
        };

        w * layer_size + local_w
    }

    /// Compute 3D serpentine index for a layer.
    fn compute_3d_index(&self, x: usize, y: usize, z: usize) -> usize {
        let plane_size = self.dims[0] * self.dims[1];

        let local_z = if z % 2 == 0 {
            self.compute_2d_index(x, y)
        } else {
            plane_size - 1 - self.compute_2d_index(x, y)
        };

        z * plane_size + local_z
    }

    /// Compute 2D serpentine index.
    fn compute_2d_index(&self, x: usize, y: usize) -> usize {
        let local_y = if y % 2 == 0 { x } else { self.dims[0] - 1 - x };
        y * self.dims[0] + local_y
    }

    /// Map 1D index back to 4D coordinates.
    pub fn map_1d_to_4d(&self, index: usize) -> GridCoord4D {
        let layer_size = self.dims[0] * self.dims[1] * self.dims[2];
        assert!(index < layer_size * self.dims[3]);

        let w = index / layer_size;
        let in_layer = index % layer_size;

        let plane_size = self.dims[0] * self.dims[1];
        let z_local = if w % 2 == 0 {
            in_layer
        } else {
            layer_size - 1 - in_layer
        };

        let z = z_local / plane_size;
        let in_plane = z_local % plane_size;

        let y_local = if z % 2 == 0 {
            in_plane
        } else {
            plane_size - 1 - in_plane
        };
        let y = y_local / self.dims[0];
        let x_local = y_local % self.dims[0];

        let x = if y % 2 == 0 {
            x_local
        } else {
            self.dims[0] - 1 - x_local
        };

        GridCoord4D::new(x, y, z, w)
    }

    /// Get 4D nearest neighbors (8-connectivity).
    pub fn neighbors(&self, x: usize, y: usize, z: usize, w: usize) -> Vec<GridCoord4D> {
        let mut out = Vec::new();

        // X neighbors
        if x > 0 {
            out.push(GridCoord4D::new(x - 1, y, z, w));
        }
        if x + 1 < self.dims[0] {
            out.push(GridCoord4D::new(x + 1, y, z, w));
        }

        // Y neighbors
        if y > 0 {
            out.push(GridCoord4D::new(x, y - 1, z, w));
        }
        if y + 1 < self.dims[1] {
            out.push(GridCoord4D::new(x, y + 1, z, w));
        }

        // Z neighbors
        if z > 0 {
            out.push(GridCoord4D::new(x, y, z - 1, w));
        }
        if z + 1 < self.dims[2] {
            out.push(GridCoord4D::new(x, y, z + 1, w));
        }

        // W neighbors (4th dimension)
        if w > 0 {
            out.push(GridCoord4D::new(x, y, z, w - 1));
        }
        if w + 1 < self.dims[3] {
            out.push(GridCoord4D::new(x, y, z, w + 1));
        }

        out
    }

    /// Get all edges in the 4D lattice.
    pub fn all_edges(&self) -> Vec<(GridCoord4D, GridCoord4D)> {
        let mut edges = Vec::new();

        for w in 0..self.dims[3] {
            for z in 0..self.dims[2] {
                for y in 0..self.dims[1] {
                    for x in 0..self.dims[0] {
                        let coord = GridCoord4D::new(x, y, z, w);

                        // X-directed edges
                        if x + 1 < self.dims[0] {
                            edges.push((coord, GridCoord4D::new(x + 1, y, z, w)));
                        }

                        // Y-directed edges
                        if y + 1 < self.dims[1] {
                            edges.push((coord, GridCoord4D::new(x, y + 1, z, w)));
                        }

                        // Z-directed edges
                        if z + 1 < self.dims[2] {
                            edges.push((coord, GridCoord4D::new(x, y, z + 1, w)));
                        }

                        // W-directed edges (4th dimension)
                        if w + 1 < self.dims[3] {
                            edges.push((coord, GridCoord4D::new(x, y, z, w + 1)));
                        }
                    }
                }
            }
        }

        edges
    }
}

/// ZZ interaction gate.
fn zz_gate(theta: f64) -> [[Complex64; 4]; 4] {
    let c = theta.cos();
    let s = theta.sin();
    [
        [Complex64::new(c, -s), ZERO, ZERO, ZERO],
        [ZERO, Complex64::new(c, s), ZERO, ZERO],
        [ZERO, ZERO, Complex64::new(c, s), ZERO],
        [ZERO, ZERO, ZERO, Complex64::new(c, -s)],
    ]
}

/// Entanglement heatmap for 4D lattices.
#[derive(Clone, Debug)]
pub struct EntanglementHeatmap4D {
    pub dims: [usize; 4],
    /// X-direction bonds (dims[0]-1 x dims[1] x dims[2] x dims[3])
    pub x_bonds: Vec<Vec<Vec<Vec<f64>>>>,
    /// Y-direction bonds
    pub y_bonds: Vec<Vec<Vec<Vec<f64>>>>,
    /// Z-direction bonds
    pub z_bonds: Vec<Vec<Vec<Vec<f64>>>>,
    /// W-direction bonds (4th dimension)
    pub w_bonds: Vec<Vec<Vec<Vec<f64>>>>,
}

/// 4D Lattice MPS simulator.
pub struct LatticeMPS4D {
    dims: [usize; 4],
    mapper: SnakeMapper4D,
    sim: MPSSimulator,
    checkpoints: HashMap<String, Vec<(usize, usize, usize)>>,
    entropy_history: Vec<f64>,
    step: usize,
}

impl LatticeMPS4D {
    pub fn new(nx: usize, ny: usize, nz: usize, nw: usize, max_bond_dim: Option<usize>) -> Self {
        let mapper = SnakeMapper4D::new(nx, ny, nz, nw);
        let sim = MPSSimulator::new(nx * ny * nz * nw, max_bond_dim);
        Self {
            dims: [nx, ny, nz, nw],
            mapper,
            sim,
            checkpoints: HashMap::new(),
            entropy_history: Vec::new(),
            step: 0,
        }
    }

    fn idx(&self, x: usize, y: usize, z: usize, w: usize) -> usize {
        self.mapper.map_4d_to_1d(x, y, z, w)
    }

    /// Apply Hadamard to a site.
    pub fn h(&mut self, x: usize, y: usize, z: usize, w: usize) {
        let q = self.idx(x, y, z, w);
        self.sim.h(q);
    }

    /// Prepare |+⟩^⊗n product state.
    pub fn prepare_plus_state(&mut self) {
        for w in 0..self.dims[3] {
            for z in 0..self.dims[2] {
                for y in 0..self.dims[1] {
                    for x in 0..self.dims[0] {
                        self.h(x, y, z, w);
                    }
                }
            }
        }
    }

    /// Apply 4D cluster state (|+⟩ with CZ on all hypercubic edges).
    pub fn apply_cluster_state(&mut self) {
        self.prepare_plus_state();
        let gate = cz_gate();
        let edges = self.mapper.all_edges();

        for (c1, c2) in edges {
            let q1 = self.idx(c1.x, c1.y, c1.z, c1.w);
            let q2 = self.idx(c2.x, c2.y, c2.z, c2.w);
            apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
        }
    }

    /// Apply checkerboard ZZ layer (hypercheckerboard partitioning).
    pub fn apply_ising_layer(&mut self, theta: f64) {
        let gate = zz_gate(theta);

        for pass in 0..2 {
            for w in 0..self.dims[3] {
                for z in 0..self.dims[2] {
                    for y in 0..self.dims[1] {
                        for x in 0..self.dims[0] {
                            // 4D checkerboard: parity based on all 4 coordinates
                            if (x + y + z + w) % 2 != pass {
                                continue;
                            }

                            // Apply to each positive direction
                            if x + 1 < self.dims[0] {
                                let q1 = self.idx(x, y, z, w);
                                let q2 = self.idx(x + 1, y, z, w);
                                apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                            }
                            if y + 1 < self.dims[1] {
                                let q1 = self.idx(x, y, z, w);
                                let q2 = self.idx(x, y + 1, z, w);
                                apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                            }
                            if z + 1 < self.dims[2] {
                                let q1 = self.idx(x, y, z, w);
                                let q2 = self.idx(x, y, z + 1, w);
                                apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                            }
                            if w + 1 < self.dims[3] {
                                let q1 = self.idx(x, y, z, w);
                                let q2 = self.idx(x, y, z, w + 1);
                                apply_gate_with_swap_network(&mut self.sim, q1, q2, &gate);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply mixer layer (Rx) across all sites.
    pub fn apply_mixer_layer(&mut self, beta: f64) {
        for w in 0..self.dims[3] {
            for z in 0..self.dims[2] {
                for y in 0..self.dims[1] {
                    for x in 0..self.dims[0] {
                        let q = self.idx(x, y, z, w);
                        self.sim.rx(q, 2.0 * beta);
                    }
                }
            }
        }
    }

    /// Run QAOA on the 4D hypercubic lattice.
    pub fn run_qaoa_ising(&mut self, p: usize, gamma: f64, beta: f64) {
        self.prepare_plus_state();
        for _ in 0..p {
            self.apply_ising_layer(gamma);
            self.apply_mixer_layer(beta);
        }
    }

    /// Record total entropy.
    pub fn record_entropy(&mut self) {
        let n_qubits = self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3];
        let total: f64 = (0..n_qubits - 1)
            .filter_map(|b| self.sim.bond_entanglement_entropy(b))
            .sum();
        self.entropy_history.push(total);
        self.step += 1;
    }

    /// Get entropy history.
    pub fn entropy_history(&self) -> &[f64] {
        &self.entropy_history
    }

    /// Extract 4D entanglement heatmap.
    pub fn entanglement_heatmap(&self) -> EntanglementHeatmap4D {
        let (nx, ny, nz, nw) = self.mapper.dimensions();

        // Initialize bond arrays
        let x_bonds = vec![vec![vec![vec![0.0; nw]; nz]; ny]; nx - 1];
        let y_bonds = vec![vec![vec![vec![0.0; nw]; nz]; ny - 1]; nx];
        let z_bonds = vec![vec![vec![vec![0.0; nw]; nz - 1]; ny]; nx];
        let w_bonds = vec![vec![vec![vec![0.0; nw - 1]; nz]; ny]; nx];

        // Note: Full population would require mapping bonds to 1D indices
        // This is a simplified version

        EntanglementHeatmap4D {
            dims: [nx, ny, nz, nw],
            x_bonds,
            y_bonds,
            z_bonds,
            w_bonds,
        }
    }

    /// Measure all qubits.
    pub fn measure(&mut self) -> usize {
        self.sim.measure()
    }

    /// Get max bond dimension.
    pub fn max_bond_dim(&self) -> usize {
        self.sim.max_bond_dim()
    }

    /// Enable entanglement tracking.
    pub fn enable_entanglement_tracking(&mut self, enabled: bool) {
        self.sim.enable_entanglement_tracking(enabled);
    }
}

/// Hypercubic code (4D toric code variant) utilities.
pub mod hypercubic_code {
    use super::*;

    /// Plaquette types in 4D hypercubic code.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum PlaquetteType {
        /// XY plane plaquette
        XY,
        /// XZ plane plaquette
        XZ,
        /// XW plane plaquette
        XW,
        /// YZ plane plaquette
        YZ,
        /// YW plane plaquette
        YW,
        /// ZW plane plaquette
        ZW,
    }

    /// 4D hypercubic code.
    pub struct HypercubicCode {
        dims: [usize; 4],
    }

    impl HypercubicCode {
        pub fn new(nx: usize, ny: usize, nz: usize, nw: usize) -> Self {
            Self {
                dims: [nx, ny, nz, nw],
            }
        }

        /// Get all plaquettes of a given type.
        pub fn plaquettes(&self, ptype: PlaquetteType) -> Vec<[GridCoord4D; 4]> {
            let mut plaquettes = Vec::new();

            match ptype {
                PlaquetteType::XY => {
                    for w in 0..self.dims[3] {
                        for z in 0..self.dims[2] {
                            for y in 0..(self.dims[1] - 1) {
                                for x in 0..(self.dims[0] - 1) {
                                    plaquettes.push([
                                        GridCoord4D::new(x, y, z, w),
                                        GridCoord4D::new(x + 1, y, z, w),
                                        GridCoord4D::new(x + 1, y + 1, z, w),
                                        GridCoord4D::new(x, y + 1, z, w),
                                    ]);
                                }
                            }
                        }
                    }
                }
                // Other plane types would be similar
                _ => {}
            }

            plaquettes
        }

        /// Count total qubits (edges in the 4D lattice).
        pub fn num_qubits(&self) -> usize {
            let n = self.dims;
            // X edges + Y edges + Z edges + W edges
            (n[0] - 1) * n[1] * n[2] * n[3]
                + n[0] * (n[1] - 1) * n[2] * n[3]
                + n[0] * n[1] * (n[2] - 1) * n[3]
                + n[0] * n[1] * n[2] * (n[3] - 1)
        }

        /// Count X stabilizers (plaquettes).
        pub fn num_x_stabilizers(&self) -> usize {
            // 6 choose 2 = 6 plaquette types
            let n = self.dims;
            (n[0] - 1) * (n[1] - 1) * n[2] * n[3] + // XY
            (n[0] - 1) * n[1] * (n[2] - 1) * n[3] + // XZ
            (n[0] - 1) * n[1] * n[2] * (n[3] - 1) + // XW
            n[0] * (n[1] - 1) * (n[2] - 1) * n[3] + // YZ
            n[0] * (n[1] - 1) * n[2] * (n[3] - 1) + // YW
            n[0] * n[1] * (n[2] - 1) * (n[3] - 1) // ZW
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_mapper_4d_roundtrip() {
        let mapper = SnakeMapper4D::new(2, 2, 2, 2);
        for w in 0..2 {
            for z in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        let idx = mapper.map_4d_to_1d(x, y, z, w);
                        let coord = mapper.map_1d_to_4d(idx);
                        assert_eq!(coord, GridCoord4D::new(x, y, z, w));
                    }
                }
            }
        }
    }

    #[test]
    fn test_snake_mapper_4d_neighbors() {
        let mapper = SnakeMapper4D::new(3, 3, 3, 3);
        let neighbors = mapper.neighbors(1, 1, 1, 1);
        assert_eq!(neighbors.len(), 8); // Interior point has 8 neighbors (2 in each of 4 dimensions)
    }

    #[test]
    fn test_lattice_mps_4d_creation() {
        let sim = LatticeMPS4D::new(2, 2, 2, 2, Some(4));
        assert_eq!(sim.max_bond_dim(), 1); // Initial state has bond dim 1
    }

    #[test]
    fn test_lattice_mps_4d_plus_state() {
        let mut sim = LatticeMPS4D::new(2, 2, 2, 2, Some(8));
        sim.prepare_plus_state();
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_mps_4d_ising_layer() {
        let mut sim = LatticeMPS4D::new(2, 2, 2, 2, Some(8));
        sim.prepare_plus_state();
        sim.apply_ising_layer(0.1);
        let _ = sim.measure();
    }

    #[test]
    fn test_lattice_mps_4d_cluster_state() {
        let mut sim = LatticeMPS4D::new(2, 2, 2, 2, Some(16));
        sim.apply_cluster_state();
        let _ = sim.measure();
    }

    #[test]
    fn test_hypercubic_code() {
        let code = hypercubic_code::HypercubicCode::new(3, 3, 3, 3);
        let qubits = code.num_qubits();
        let stabilizers = code.num_x_stabilizers();
        assert!(qubits > 0);
        assert!(stabilizers > 0);
    }

    #[test]
    fn test_entropy_tracking() {
        let mut sim = LatticeMPS4D::new(2, 2, 2, 2, Some(16));
        sim.enable_entanglement_tracking(true);
        sim.prepare_plus_state();
        sim.record_entropy();
        sim.apply_ising_layer(0.2);
        sim.record_entropy();

        let history = sim.entropy_history();
        assert_eq!(history.len(), 2);
    }
}
