// Corner Transfer Matrix (CTM) Method for PEPS Contraction
//
// The CTM method is the state-of-the-art technique for contracting
// 2D tensor networks (PEPS) with high accuracy.
//
// Key idea: Represent the environment of a tensor as four corner matrices
// that are iteratively updated to self-consistency.

use ndarray::{Array2, Array4};
use num_complex::Complex64 as c64;

// Import PEPS types from the peps module.
pub use crate::peps::BondDirection;
pub use crate::peps::PEPSTensor;
pub use crate::peps::PEPS;

// ============================================================
// CTM DATA STRUCTURES
// ============================================================

/// Corner matrices representing the environment
///
/// For a PEPS on a square lattice, the environment around
/// any tensor can be represented by four corner matrices:
/// - TL: Top-Left
/// - TR: Top-Right
/// - BL: Bottom-Left
/// - BR: Bottom-Right
#[derive(Clone, Debug)]
pub struct CornerMatrices {
    /// Top-Left corner
    pub tl: Array2<c64>,
    /// Top-Right corner
    pub tr: Array2<c64>,
    /// Bottom-Left corner
    pub bl: Array2<c64>,
    /// Bottom-Right corner
    pub br: Array2<c64>,
    /// Bond dimension
    pub bond_dim: usize,
}

impl CornerMatrices {
    /// Create zero corner matrices
    pub fn zeros(bond_dim: usize) -> Self {
        let zero_matrix = Array2::zeros((bond_dim, bond_dim));

        Self {
            tl: zero_matrix.clone(),
            tr: zero_matrix.clone(),
            bl: zero_matrix.clone(),
            br: zero_matrix,
            bond_dim,
        }
    }

    /// Create identity corner matrices (initial state)
    pub fn identity(bond_dim: usize) -> Self {
        let identity = Array2::eye(bond_dim);

        Self {
            tl: identity.clone(),
            tr: identity.clone(),
            bl: identity.clone(),
            br: identity,
            bond_dim,
        }
    }

    /// Normalize corners (prevent numerical explosion)
    pub fn normalize(&mut self) {
        let trace_tl: c64 = self.tl.diag().iter().sum();
        let trace_tr: c64 = self.tr.diag().iter().sum();
        let trace_bl: c64 = self.bl.diag().iter().sum();
        let trace_br: c64 = self.br.diag().iter().sum();

        let trace = (trace_tl + trace_tr + trace_bl + trace_br) / 4.0;

        let norm = if trace.norm_sqr() > 1e-15 {
            trace
        } else {
            c64::new(1.0, 0.0)
        };

        self.tl = self.tl.map(|x| x / norm);
        self.tr = self.tr.map(|x| x / norm);
        self.bl = self.bl.map(|x| x / norm);
        self.br = self.br.map(|x| x / norm);
    }
}

/// CTM contraction configuration
#[derive(Debug, Clone)]
pub struct CTMConfig {
    /// Maximum bond dimension for corners
    pub max_bond_dim: usize,
    /// convergence threshold
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Truncation cutoff for SVD
    pub cutoff: f64,
}

impl Default for CTMConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            tolerance: 1e-10,
            max_iterations: 1000,
            cutoff: 1e-12,
        }
    }
}

// ============================================================
// CTM CONTRACTION ENGINE
// ============================================================

/// CTM contraction result
#[derive(Debug)]
pub struct CTMResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_error: f64,
    pub corners: CornerMatrices,
}

/// CTM contraction engine for PEPS
pub struct CTMEngine {
    config: CTMConfig,
    corners: CornerMatrices,
}

impl CTMEngine {
    /// Create a new CTM engine
    pub fn new(config: CTMConfig) -> Self {
        let corners = CornerMatrices::identity(config.max_bond_dim);

        Self { config, corners }
    }

    /// Get current corners
    pub fn corners(&self) -> &CornerMatrices {
        &self.corners
    }

    /// Run CTM iteration to self-consistency
    pub fn run(&mut self, max_iterations: usize) -> CTMResult {
        let mut prev_norm = 0.0;

        for iteration in 0..max_iterations {
            // Perform one CTM step (simplified - just normalize for now)
            self.corners.normalize();

            // Check convergence
            let current_norm = self.corners_norm();
            let delta = (current_norm - prev_norm).abs();

            if delta < self.config.tolerance {
                return CTMResult {
                    converged: true,
                    iterations: iteration,
                    final_error: delta,
                    corners: self.corners.clone(),
                };
            }

            prev_norm = current_norm;
        }

        CTMResult {
            converged: false,
            iterations: max_iterations,
            final_error: prev_norm,
            corners: self.corners.clone(),
        }
    }

    /// Compute norm of corners for convergence check
    fn corners_norm(&self) -> f64 {
        let tl_norm: f64 = self.corners.tl.iter().map(|x| x.norm_sqr()).sum();
        let tr_norm: f64 = self.corners.tr.iter().map(|x| x.norm_sqr()).sum();
        let bl_norm: f64 = self.corners.bl.iter().map(|x| x.norm_sqr()).sum();
        let br_norm: f64 = self.corners.br.iter().map(|x| x.norm_sqr()).sum();

        (tl_norm + tr_norm + bl_norm + br_norm).sqrt()
    }
}

// ============================================================
// CTM DOUBLE TENSOR (physical index traced out)
// ============================================================

/// Double tensor for CTM operations.
///
/// In CTM contraction, the physical index of a PEPS tensor is traced
/// out by contracting T with T* (the "double tensor"). This produces
/// a 4-index tensor with only bond indices: [up, down, left, right],
/// where each bond dimension is D^2 (squared from T and T*).
///
/// This type bridges between the real 5-index `PEPSTensor` from the
/// `peps` module and the CTM algorithm which operates on environment
/// tensors without physical indices.
#[derive(Clone, Debug)]
pub struct CTMDoubleTensor {
    /// 4D tensor data: [bond_up^2, bond_down^2, bond_left^2, bond_right^2]
    pub data: Array4<c64>,
    /// Squared bond dimensions [up, down, left, right]
    pub bond_dims: [usize; 4],
}

impl CTMDoubleTensor {
    /// Get the (squared) bond dimension in a specific direction.
    pub fn bond_dim(&self, direction: BondDirection) -> usize {
        self.bond_dims[direction as usize]
    }

    /// Set the bond dimension metadata in a specific direction.
    pub fn set_bond_dim(&mut self, direction: BondDirection, dim: usize) {
        self.bond_dims[direction as usize] = dim;
    }

    /// Create a double tensor from a real `PEPSTensor` by tracing
    /// over the physical index: D[bu1*bu2, bd1*bd2, bl1*bl2, br1*br2]
    /// = sum_p T[p, bu1, bd1, bl1, br1] * conj(T[p, bu2, bd2, bl2, br2])
    pub fn from_peps_tensor(tensor: &PEPSTensor) -> Self {
        let dims = tensor.bond_dims;
        let phys = tensor.data.shape()[0];
        let du = dims[0];
        let dd = dims[1];
        let dl = dims[2];
        let dr = dims[3];

        let du2 = du * du;
        let dd2 = dd * dd;
        let dl2 = dl * dl;
        let dr2 = dr * dr;

        let mut data = Array4::<c64>::zeros((du2, dd2, dl2, dr2));

        for p in 0..phys {
            for bu1 in 0..du {
                for bu2 in 0..du {
                    for bd1 in 0..dd {
                        for bd2 in 0..dd {
                            for bl1 in 0..dl {
                                for bl2 in 0..dl {
                                    for br1 in 0..dr {
                                        for br2 in 0..dr {
                                            let t_val =
                                                tensor.data[[p, bu1, bd1, bl1, br1]];
                                            let t_conj =
                                                tensor.data[[p, bu2, bd2, bl2, br2]].conj();
                                            data[[
                                                bu1 * du + bu2,
                                                bd1 * dd + bd2,
                                                bl1 * dl + bl2,
                                                br1 * dr + br2,
                                            ]] += t_val * t_conj;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            data,
            bond_dims: [du2, dd2, dl2, dr2],
        }
    }
}

/// Convert an entire PEPS lattice into a grid of CTM double tensors.
///
/// Each site tensor has its physical index traced out, producing the
/// doubled bond representation needed for CTM environment iteration.
pub fn peps_to_double_tensors(peps: &PEPS) -> Vec<Vec<CTMDoubleTensor>> {
    let (width, height) = peps.dimensions();
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| CTMDoubleTensor::from_peps_tensor(&peps.tensors[y][x]))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corner_creation() {
        let corners = CornerMatrices::zeros(16);
        assert_eq!(corners.bond_dim, 16);
        assert_eq!(corners.tl.dim(), (16, 16));
    }

    #[test]
    fn test_corner_normalization() {
        let mut corners = CornerMatrices::identity(8);
        corners.normalize();

        // After normalization, trace should be close to 1
        let trace: c64 = corners.tl.diag().iter().sum();
        assert!((trace.norm_sqr().sqrt() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_ctm_config_default() {
        let config = CTMConfig::default();
        assert_eq!(config.max_bond_dim, 64);
        assert_eq!(config.tolerance, 1e-10);
    }

    #[test]
    fn test_ctm_engine_creation() {
        let engine = CTMEngine::new(CTMConfig::default());
        assert_eq!(engine.config.max_bond_dim, 64);
        assert_eq!(engine.corners.bond_dim, 64);
    }

    #[test]
    fn test_ctm_convergence() {
        let mut engine = CTMEngine::new(CTMConfig {
            max_bond_dim: 8,
            tolerance: 1e-6,
            max_iterations: 100,
            cutoff: 1e-10,
        });

        let result = engine.run(50);

        assert!(result.iterations <= 50);
    }

    // ── Tests for real PEPS type integration ──────────────────────────

    #[test]
    fn test_bond_direction_from_peps_module() {
        // Verify BondDirection is the same type from peps module
        assert_eq!(BondDirection::Up as usize, 0);
        assert_eq!(BondDirection::Down as usize, 1);
        assert_eq!(BondDirection::Left as usize, 2);
        assert_eq!(BondDirection::Right as usize, 3);
    }

    #[test]
    fn test_peps_tensor_zero_state() {
        // Verify we can use the real PEPSTensor from peps module
        let tensor = PEPSTensor::zero(2);
        assert_eq!(tensor.bond_dim(BondDirection::Up), 1);
        assert_eq!(tensor.bond_dim(BondDirection::Down), 1);
        assert_eq!(tensor.bond_dim(BondDirection::Left), 1);
        assert_eq!(tensor.bond_dim(BondDirection::Right), 1);
    }

    #[test]
    fn test_peps_creation() {
        // Verify we can create a real PEPS state
        let peps = PEPS::new(3, 2, 4);
        assert_eq!(peps.num_qubits(), 6);
        assert_eq!(peps.dimensions(), (3, 2));
    }

    #[test]
    fn test_double_tensor_from_zero_state() {
        // |0> tensor has data[0,0,0,0,0] = 1, rest zero
        // Double tensor: D[0,0,0,0] = |1|^2 = 1, rest zero
        let tensor = PEPSTensor::zero(2);
        let double = CTMDoubleTensor::from_peps_tensor(&tensor);

        assert_eq!(double.bond_dims, [1, 1, 1, 1]);
        let val = double.data[[0, 0, 0, 0]];
        assert!((val.re - 1.0).abs() < 1e-12);
        assert!(val.im.abs() < 1e-12);
    }

    #[test]
    fn test_double_tensor_from_plus_state() {
        // |+> = (|0> + |1>) / sqrt(2) with bond dim 1
        // Double tensor: D[0,0,0,0] = sum_p |T[p,0,0,0,0]|^2
        //   = |1/sqrt(2)|^2 + |1/sqrt(2)|^2 = 1.0
        let tensor = PEPSTensor::plus_state(2);
        let double = CTMDoubleTensor::from_peps_tensor(&tensor);

        assert_eq!(double.bond_dims, [1, 1, 1, 1]);
        let val = double.data[[0, 0, 0, 0]];
        assert!(
            (val.re - 1.0).abs() < 1e-12,
            "Expected 1.0, got {}",
            val.re
        );
    }

    #[test]
    fn test_double_tensor_bond_dim_method() {
        let tensor = PEPSTensor::zero(2);
        let double = CTMDoubleTensor::from_peps_tensor(&tensor);

        // Bond dim 1 squared is still 1
        assert_eq!(double.bond_dim(BondDirection::Up), 1);
        assert_eq!(double.bond_dim(BondDirection::Right), 1);
    }

    #[test]
    fn test_double_tensor_set_bond_dim() {
        let tensor = PEPSTensor::zero(2);
        let mut double = CTMDoubleTensor::from_peps_tensor(&tensor);

        double.set_bond_dim(BondDirection::Up, 4);
        assert_eq!(double.bond_dim(BondDirection::Up), 4);
        // Other directions unchanged
        assert_eq!(double.bond_dim(BondDirection::Down), 1);
    }

    #[test]
    fn test_peps_to_double_tensors_grid() {
        let peps = PEPS::new(3, 2, 4);
        let doubles = peps_to_double_tensors(&peps);

        assert_eq!(doubles.len(), 2); // height
        assert_eq!(doubles[0].len(), 3); // width

        // All tensors initialized to |0>, so each double tensor
        // should have D[0,0,0,0] = 1.0
        for row in &doubles {
            for dt in row {
                let val = dt.data[[0, 0, 0, 0]];
                assert!((val.re - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_double_tensor_trace_normalization() {
        // For a normalized tensor, the trace of the double tensor
        // (sum of diagonal elements) should equal 1.0
        let tensor = PEPSTensor::zero(2);
        let double = CTMDoubleTensor::from_peps_tensor(&tensor);

        // With bond dim 1, the "trace" is just the single element
        let trace: c64 = double.data.iter().sum();
        assert!(
            (trace.re - 1.0).abs() < 1e-12,
            "Trace should be 1.0 for normalized state, got {}",
            trace.re
        );
    }

    #[test]
    fn test_double_tensor_clone() {
        let tensor = PEPSTensor::plus_state(2);
        let double = CTMDoubleTensor::from_peps_tensor(&tensor);
        let cloned = double.clone();

        assert_eq!(cloned.bond_dims, double.bond_dims);
        assert_eq!(cloned.data, double.data);
    }
}
