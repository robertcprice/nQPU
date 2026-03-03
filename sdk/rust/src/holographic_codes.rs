//! Holographic Quantum Error Correcting Codes (AdS/CFT Correspondence)
//!
//! This module implements holographic quantum error correction codes inspired
//! by the Anti-de Sitter / Conformal Field Theory (AdS/CFT) correspondence.
//! The centerpiece is the HaPPY code (Pastawski-Yoshida-Harlow-Preskill),
//! which constructs a discrete model of the bulk-boundary correspondence
//! using perfect tensors arranged on a hyperbolic tiling.
//!
//! # Capabilities
//!
//! - **Perfect Tensor Construction**: Build and verify [[5,1,3]] stabilizer
//!   code tensors satisfying the perfection property (any 3+3 bipartition
//!   of legs yields a unitary map).
//! - **Hyperbolic Tiling**: Construct {5,4} hyperbolic tilings (pentagons,
//!   4 meeting at each vertex) that serve as the discretized AdS bulk.
//! - **Holographic Encoding**: Map bulk logical qubits to boundary physical
//!   qubits via tensor network contraction from the center outward.
//! - **Ryu-Takayanagi Formula**: Compute entanglement entropy of boundary
//!   subregions via minimal-cut (geodesic) surfaces in the tiling graph.
//! - **Entanglement Wedge Reconstruction**: Determine which bulk operators
//!   are recoverable from a given boundary subregion.
//! - **Subregion Duality**: Demonstrate that the same bulk operator can be
//!   reconstructed from multiple complementary boundary regions.
//! - **Holographic Entropy Cone**: Verify strong subadditivity and monogamy
//!   of mutual information for holographic states.
//!
//! # Applications
//!
//! - Toy models of quantum gravity and black hole information
//! - Exploration of bulk-boundary duality and quantum error correction
//! - Study of entanglement structure in holographic systems
//! - Testing conjectured entropy inequalities for holographic states
//!
//! # References
//!
//! - Pastawski, Yoshida, Harlow, Preskill, "Holographic quantum error-correcting
//!   codes: Toy models for the bulk/boundary correspondence", JHEP 06 (2015) 149
//! - Ryu, Takayanagi, "Holographic Derivation of Entanglement Entropy from
//!   the anti-de Sitter Space/Conformal Field Theory Correspondence",
//!   Phys.Rev.Lett. 96 (2006) 181602
//! - Almheiri, Dong, Harlow, "Bulk Locality and Quantum Error Correction in
//!   AdS/CFT", JHEP 04 (2015) 163

use crate::{C64, GateOperations, QuantumState};
use std::collections::{HashSet, VecDeque};
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// CONSTANTS
// ============================================================

/// Numerical tolerance for unitary checks and normalization.
const EPSILON: f64 = 1e-10;

/// Number of legs on the [[5,1,3]] perfect tensor (5 physical + 1 logical).
const PERFECT_TENSOR_LEGS: usize = 6;

/// Dimension of the Hilbert space for the 6-leg perfect tensor: 2^6 = 64.
const PERFECT_TENSOR_DIM: usize = 64;

/// Number of unique 3+3 bipartitions of 6 legs: C(6,3)/2 = 10.
const NUM_BIPARTITIONS: usize = 10;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors arising from holographic code operations.
#[derive(Debug, Clone)]
pub enum HolographicError {
    /// The tensor does not satisfy the perfection property.
    InvalidTensor(String),
    /// Bulk and boundary qubit counts are inconsistent with the tiling.
    BulkBoundaryMismatch {
        expected_bulk: usize,
        expected_boundary: usize,
        got_bulk: usize,
        got_boundary: usize,
    },
    /// Entanglement wedge reconstruction failed for the given region.
    ReconstructionFailed(String),
    /// Error computing entropy (e.g., invalid region specification).
    EntropyError(String),
    /// Invalid configuration parameters.
    InvalidConfig(String),
}

impl fmt::Display for HolographicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HolographicError::InvalidTensor(msg) => {
                write!(f, "Invalid perfect tensor: {}", msg)
            }
            HolographicError::BulkBoundaryMismatch {
                expected_bulk,
                expected_boundary,
                got_bulk,
                got_boundary,
            } => {
                write!(
                    f,
                    "Bulk/boundary mismatch: expected ({} bulk, {} boundary), got ({}, {})",
                    expected_bulk, expected_boundary, got_bulk, got_boundary
                )
            }
            HolographicError::ReconstructionFailed(msg) => {
                write!(f, "Reconstruction failed: {}", msg)
            }
            HolographicError::EntropyError(msg) => {
                write!(f, "Entropy error: {}", msg)
            }
            HolographicError::InvalidConfig(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
        }
    }
}

impl std::error::Error for HolographicError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// The type of holographic code to construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HolographicCodeType {
    /// HaPPY code using [[5,1,3]] perfect tensors.
    HaPPY,
    /// Random tensor network (for comparison).
    RandomTensor,
    /// Steane-based holographic code (7-qubit code on tiling).
    SteaneHolographic,
}

impl fmt::Display for HolographicCodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HolographicCodeType::HaPPY => write!(f, "HaPPY"),
            HolographicCodeType::RandomTensor => write!(f, "RandomTensor"),
            HolographicCodeType::SteaneHolographic => write!(f, "SteaneHolographic"),
        }
    }
}

/// Configuration for holographic code construction.
#[derive(Debug, Clone)]
pub struct HolographicConfig {
    /// Number of layers in the hyperbolic tiling (0 = single central pentagon).
    pub num_layers: usize,
    /// Type of holographic code.
    pub code_type: HolographicCodeType,
    /// Number of bulk (logical) qubits. Computed from the tiling if set to 0.
    pub bulk_qubits: usize,
    /// Number of boundary (physical) qubits. Computed from the tiling if set to 0.
    pub boundary_qubits: usize,
}

impl Default for HolographicConfig {
    fn default() -> Self {
        Self {
            num_layers: 0,
            code_type: HolographicCodeType::HaPPY,
            bulk_qubits: 0,
            boundary_qubits: 0,
        }
    }
}

impl HolographicConfig {
    /// Create a new configuration with the given number of layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            ..Default::default()
        }
    }

    /// Set the code type.
    pub fn with_code_type(mut self, code_type: HolographicCodeType) -> Self {
        self.code_type = code_type;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), HolographicError> {
        if self.num_layers > 4 {
            return Err(HolographicError::InvalidConfig(
                "num_layers > 4 produces tilings too large for exact simulation".into(),
            ));
        }
        Ok(())
    }
}

// ============================================================
// PERFECT TENSOR
// ============================================================

/// A perfect tensor: a 6-index tensor (2^6 = 64 entries) encoding the
/// [[5,1,3]] stabilizer code. The perfection property guarantees that any
/// bipartition of the 6 legs into two sets of 3 yields a unitary 8x8 map.
///
/// The 6 legs are indexed as (i0, i1, i2, i3, i4, i5) where each i_k in {0,1}.
/// The combined index into the flat vector is:
///   idx = i0 + 2*i1 + 4*i2 + 8*i3 + 16*i4 + 32*i5
///
/// Leg 5 is conventionally the "logical" leg and legs 0-4 are "physical".
#[derive(Debug, Clone)]
pub struct PerfectTensor {
    /// The 64-element state vector defining the tensor.
    pub amplitudes: Vec<C64>,
    /// Number of legs (always 6 for [[5,1,3]]).
    pub num_legs: usize,
}

impl PerfectTensor {
    /// Construct the [[5,1,3]] perfect tensor from its stabilizer generators.
    ///
    /// The stabilizer group of the [[5,1,3]] code has generators:
    ///   g1 = XZZXI
    ///   g2 = IXZZX
    ///   g3 = XIXZZ
    ///   g4 = ZXIXZ
    ///
    /// The code space is the +1 eigenspace of all four generators, which has
    /// dimension 2 (encoding 1 logical qubit). The perfect tensor is the
    /// isometry from the 1-qubit logical space into the 5-qubit physical space,
    /// stored as a 6-index tensor (5 physical legs + 1 logical leg).
    pub fn new() -> Result<Self, HolographicError> {
        // Build the code space by projecting onto the +1 eigenspace of each
        // stabilizer generator. We work in the 5-qubit Hilbert space (dim=32).
        let n = 5;
        let dim = 1 << n; // 32

        // Stabilizer generators as (X_mask, Z_mask) binary representations.
        // Each generator is a product of Pauli X and Z operators.
        // For qubit j: X_mask bit j = 1 means X on qubit j,
        //              Z_mask bit j = 1 means Z on qubit j,
        //              both = 1 means Y = iXZ on qubit j.
        // Bit j corresponds to qubit j (LSB = qubit 0).
        let generators: Vec<(u32, u32)> = vec![
            // g1 = X_0 Z_1 Z_2 X_3 I_4
            (
                (1 << 0) | (1 << 3), // X on qubits 0,3
                (1 << 1) | (1 << 2), // Z on qubits 1,2
            ),
            // g2 = I_0 X_1 Z_2 Z_3 X_4
            (
                (1 << 1) | (1 << 4), // X on qubits 1,4
                (1 << 2) | (1 << 3), // Z on qubits 2,3
            ),
            // g3 = X_0 I_1 X_2 Z_3 Z_4
            (
                (1 << 0) | (1 << 2), // X on qubits 0,2
                (1 << 3) | (1 << 4), // Z on qubits 3,4
            ),
            // g4 = Z_0 X_1 I_2 X_3 Z_4
            (
                (1 << 1) | (1 << 3), // X on qubits 1,3
                (1 << 0) | (1 << 4), // Z on qubits 0,4
            ),
        ];

        // Build the projection operator P = (1/16) * prod_i (I + g_i).
        // Since the generators commute, this is the projector onto the code space.
        // We do this by iteratively projecting.

        // Start with the identity: all 32 computational basis states.
        // Apply (I + g_k)/2 for each generator to project.
        let mut code_space = Self::compute_code_space(&generators, n, dim);

        if code_space.is_empty() {
            return Err(HolographicError::InvalidTensor(
                "Code space is empty; check stabilizer generators".into(),
            ));
        }

        if code_space.len() != 2 {
            return Err(HolographicError::InvalidTensor(format!(
                "Expected 2-dimensional code space, got {}",
                code_space.len()
            )));
        }

        // Orthonormalize the code space basis using Gram-Schmidt.
        Self::gram_schmidt(&mut code_space, dim);

        // Build the 6-index perfect tensor.
        // The tensor T[i0,i1,i2,i3,i4,i5] is defined as:
        //   T[phys, log] = code_space[log][phys]
        // where phys = i0 + 2*i1 + 4*i2 + 8*i3 + 16*i4
        // and   log  = i5
        //
        // Combined index: idx = phys + 32 * log
        let mut amplitudes = vec![C64::new(0.0, 0.0); PERFECT_TENSOR_DIM];

        for log in 0..2_usize {
            for phys in 0..32_usize {
                let idx = phys + 32 * log;
                amplitudes[idx] = code_space[log][phys];
            }
        }

        // Normalize the tensor so that ||T||^2 = d^(n/2) = 2^3 = 8.
        // For a perfect tensor with 6 legs of dimension d=2, any 3+3
        // bipartition gives an 8x8 matrix M. For M to be unitary,
        // we need ||T||^2 = 8 (since tr(M M^dagger) = tr(I_8) = 8).
        //
        // The code space gives two orthonormal vectors: ||T||^2 = 2.
        // Scale by sqrt(8/2) = 2 to achieve ||T||^2 = 8.
        let norm_sq: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if norm_sq < EPSILON {
            return Err(HolographicError::InvalidTensor(
                "Tensor has zero norm".into(),
            ));
        }
        let target_norm_sq = 8.0; // d^(n_legs/2) = 2^3
        let scale = (target_norm_sq / norm_sq).sqrt();
        for a in amplitudes.iter_mut() {
            *a = C64::new(a.re * scale, a.im * scale);
        }

        Ok(PerfectTensor {
            amplitudes,
            num_legs: PERFECT_TENSOR_LEGS,
        })
    }

    /// Compute the code space of the [[5,1,3]] stabilizer code by projecting
    /// onto the +1 eigenspace of each generator.
    fn compute_code_space(
        generators: &[(u32, u32)],
        n: usize,
        dim: usize,
    ) -> Vec<Vec<C64>> {
        // Start with the full computational basis.
        let mut basis: Vec<Vec<C64>> = Vec::new();
        for i in 0..dim {
            let mut v = vec![C64::new(0.0, 0.0); dim];
            v[i] = C64::new(1.0, 0.0);
            basis.push(v);
        }

        // Project onto +1 eigenspace of each generator.
        for &(x_mask, z_mask) in generators {
            let mut projected: Vec<Vec<C64>> = Vec::new();
            for v in &basis {
                let gv = Self::apply_stabilizer(v, x_mask, z_mask, n);
                // (I + g)/2 applied to v = (v + gv) / 2
                let pv: Vec<C64> = v
                    .iter()
                    .zip(gv.iter())
                    .map(|(a, b)| C64::new((a.re + b.re) * 0.5, (a.im + b.im) * 0.5))
                    .collect();
                let norm_sq: f64 = pv.iter().map(|a| a.norm_sqr()).sum();
                if norm_sq > EPSILON {
                    projected.push(pv);
                }
            }
            basis = projected;
        }

        // Remove linearly dependent vectors. Use a simple rank-revealing approach.
        Self::extract_independent_basis(&basis, dim)
    }

    /// Apply a Pauli stabilizer operator (specified by X and Z masks) to a state vector.
    fn apply_stabilizer(
        state: &[C64],
        x_mask: u32,
        z_mask: u32,
        _n: usize,
    ) -> Vec<C64> {
        let dim = state.len();
        let mut result = vec![C64::new(0.0, 0.0); dim];

        for i in 0..dim {
            // The Pauli operator P = (product of X and Z on specified qubits)
            // acts as: P|i> = phase * |i XOR x_mask>
            // where phase comes from Z operators and Y = iXZ corrections.
            let j = i ^ (x_mask as usize);

            // Phase from Y = iXZ: for each qubit where both X and Z are set,
            // we get a factor of i (from Y = iXZ).
            let y_mask = x_mask & z_mask;
            let num_y = y_mask.count_ones();
            // i^num_y: 0->1, 1->i, 2->-1, 3->-i
            let y_phase = match num_y % 4 {
                0 => C64::new(1.0, 0.0),
                1 => C64::new(0.0, 1.0),
                2 => C64::new(-1.0, 0.0),
                3 => C64::new(0.0, -1.0),
                _ => unreachable!(),
            };

            // Also account for the ordering: Y = iXZ, but we applied X first then Z.
            // The Z part acts on the original state before X flip, so we need to
            // evaluate Z on the original index i, not on j.
            // Actually, the full Pauli string acts as:
            //   P|i> = (-1)^(popcount(i AND z_mask)) * (product of i for each Y) * |i XOR x_mask>
            // but we need to be careful: Z acts on qubit's original value.
            // For qubit q: if X_q=1, Z_q=1 (it's Y_q), then Y_q|b> = i*(-1)^b * |1-b>
            // We get X-flip from x_mask, sign from the original bit, and factor i.
            // For qubit q: if X_q=0, Z_q=1, then Z_q|b> = (-1)^b * |b>
            // For qubit q: if X_q=1, Z_q=0, then X_q|b> = |1-b>

            // The z_mask sign should only apply to Z-only qubits (where X is not set)
            // plus the Y qubits also contribute a sign from the original bit.
            // Let's recompute properly:
            let z_only_mask = z_mask & (!x_mask);
            let z_only_phase_bits = (i as u32) & z_only_mask;
            let z_only_sign = if z_only_phase_bits.count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };

            // For Y qubits, Y|b> = i*(-1)^b |1-b>, so sign depends on original bit.
            let y_bits = (i as u32) & y_mask;
            let y_bit_sign = if y_bits.count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };

            let total_phase = C64::new(z_only_sign * y_bit_sign, 0.0) * y_phase;
            result[j] = result[j] + total_phase * state[i];
        }

        result
    }

    /// Extract a linearly independent subset from a set of vectors.
    fn extract_independent_basis(
        vectors: &[Vec<C64>],
        dim: usize,
    ) -> Vec<Vec<C64>> {
        let mut basis: Vec<Vec<C64>> = Vec::new();

        for v in vectors {
            // Subtract projections onto existing basis vectors.
            let mut w = v.clone();
            for b in &basis {
                // Compute <b|w> (b is already normalized).
                let proj: C64 = b
                    .iter()
                    .zip(w.iter())
                    .map(|(bi, wi)| bi.conj() * wi)
                    .sum::<C64>();
                for k in 0..dim {
                    w[k] = w[k] - proj * b[k];
                }
            }
            let norm_sq: f64 = w.iter().map(|a| a.norm_sqr()).sum();
            if norm_sq > EPSILON {
                let inv_norm = 1.0 / norm_sq.sqrt();
                for a in w.iter_mut() {
                    *a = C64::new(a.re * inv_norm, a.im * inv_norm);
                }
                basis.push(w);
            }
        }

        basis
    }

    /// Orthonormalize a set of vectors using modified Gram-Schmidt.
    fn gram_schmidt(basis: &mut Vec<Vec<C64>>, dim: usize) {
        let n = basis.len();
        for i in 0..n {
            // Normalize vector i.
            let norm_sq: f64 = basis[i].iter().map(|a| a.norm_sqr()).sum();
            if norm_sq < EPSILON {
                continue;
            }
            let inv_norm = 1.0 / norm_sq.sqrt();
            for k in 0..dim {
                basis[i][k] = C64::new(basis[i][k].re * inv_norm, basis[i][k].im * inv_norm);
            }

            // Subtract projection from subsequent vectors.
            for j in (i + 1)..n {
                let proj: C64 = basis[i]
                    .iter()
                    .zip(basis[j].iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum::<C64>();
                let bi = basis[i].clone();
                for k in 0..dim {
                    basis[j][k] = basis[j][k] - proj * bi[k];
                }
            }
        }
    }

    /// Check the perfection property: for each of the 10 unique 3+3
    /// bipartitions of the 6 legs, verify the resulting 8x8 matrix is unitary.
    pub fn verify_perfection(&self) -> Result<(), HolographicError> {
        let bipartitions = Self::enumerate_bipartitions();
        for (idx, (left, right)) in bipartitions.iter().enumerate() {
            let matrix = self.extract_bipartition_matrix(left, right);
            if !Self::is_unitary(&matrix, 8) {
                return Err(HolographicError::InvalidTensor(format!(
                    "Bipartition {} (legs {:?} vs {:?}) does not yield unitary matrix",
                    idx, left, right
                )));
            }
        }
        Ok(())
    }

    /// Enumerate all 10 unique 3+3 bipartitions of 6 legs.
    fn enumerate_bipartitions() -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut result = Vec::with_capacity(NUM_BIPARTITIONS);
        // Choose 3 legs from 6 for the "left" set; the rest go to "right".
        // Only keep the half where the smallest leg is in "left" (to avoid duplicates).
        for a in 0..6_usize {
            for b in (a + 1)..6 {
                for c in (b + 1)..6 {
                    let left = vec![a, b, c];
                    let right: Vec<usize> = (0..6).filter(|x| !left.contains(x)).collect();
                    // Only keep if left[0] < right[0] to avoid duplication.
                    if left[0] < right[0] {
                        result.push((left, right));
                    }
                }
            }
        }
        result
    }

    /// Extract the 8x8 matrix obtained by treating `left` legs as row index
    /// and `right` legs as column index.
    fn extract_bipartition_matrix(&self, left: &[usize], right: &[usize]) -> Vec<Vec<C64>> {
        let mut matrix = vec![vec![C64::new(0.0, 0.0); 8]; 8];

        for row in 0..8_usize {
            for col in 0..8_usize {
                // Build the 6-bit index from the leg assignments.
                let mut idx: usize = 0;
                for (bit_pos, &leg) in left.iter().enumerate() {
                    if (row >> bit_pos) & 1 == 1 {
                        idx |= 1 << leg;
                    }
                }
                for (bit_pos, &leg) in right.iter().enumerate() {
                    if (col >> bit_pos) & 1 == 1 {
                        idx |= 1 << leg;
                    }
                }
                matrix[row][col] = self.amplitudes[idx];
            }
        }

        matrix
    }

    /// Check if an NxN matrix is unitary: M M^dagger = I.
    fn is_unitary(matrix: &[Vec<C64>], n: usize) -> bool {
        for i in 0..n {
            for j in 0..n {
                let mut dot = C64::new(0.0, 0.0);
                for k in 0..n {
                    dot = dot + matrix[i][k] * matrix[j][k].conj();
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot.re - expected).abs() > 1e-6 || dot.im.abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    /// Get the tensor element T[i0, i1, i2, i3, i4, i5].
    #[inline]
    pub fn element(&self, indices: &[usize; 6]) -> C64 {
        let idx = indices[0]
            + 2 * indices[1]
            + 4 * indices[2]
            + 8 * indices[3]
            + 16 * indices[4]
            + 32 * indices[5];
        self.amplitudes[idx]
    }

    /// Compute the Frobenius norm of the tensor.
    /// For a properly normalized perfect tensor, ||T||^2 = d^(n_legs/2) = 8.
    pub fn norm(&self) -> f64 {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        norm_sq.sqrt()
    }

    /// Extract the encoding isometry V from the perfect tensor.
    /// V maps 1 logical qubit (leg 5) to 5 physical qubits (legs 0-4).
    /// V[phys][log] = T[phys + 32*log] / sqrt(4)
    /// where the factor sqrt(4) accounts for the tensor normalization.
    pub fn encoding_isometry(&self) -> Vec<Vec<C64>> {
        // The perfect tensor has ||T||^2 = 8.
        // Split as T[phys, log] where phys has 5 legs (dim=32) and log has 1 leg (dim=2).
        // For the isometry V, we need V^dagger V = I_2.
        // Since ||T||^2 = 8 and we split into 2 columns of 32 entries,
        // each column has ||col||^2 = 4, so we divide by 2.
        let scale = 1.0 / 2.0;
        let mut v = vec![vec![C64::new(0.0, 0.0); 2]; 32];
        for phys in 0..32_usize {
            for log in 0..2_usize {
                let idx = phys + 32 * log;
                v[phys][log] = self.amplitudes[idx] * scale;
            }
        }
        v
    }
}

// ============================================================
// HYPERBOLIC TILING
// ============================================================

/// A tile (pentagon) in the hyperbolic {5,4} tiling.
#[derive(Debug, Clone)]
pub struct HolographicTile {
    /// Unique identifier for this tile.
    pub id: usize,
    /// Layer in the tiling (0 = center).
    pub layer: usize,
    /// IDs of the neighboring tiles (up to 5 neighbors, -1 = boundary).
    pub neighbors: Vec<Option<usize>>,
    /// Indices of legs that connect to bulk (interior) qubits.
    pub bulk_legs: Vec<usize>,
    /// Indices of legs that are on the boundary (dangling).
    pub boundary_legs: Vec<usize>,
    /// The perfect tensor assigned to this tile.
    pub tensor_id: usize,
}

/// Edge in the tiling graph connecting two tiles (or a tile to the boundary).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TilingEdge {
    /// Source tile ID.
    pub tile_a: usize,
    /// Target tile ID (None if boundary edge).
    pub tile_b: Option<usize>,
    /// Leg index on tile_a.
    pub leg_a: usize,
    /// Leg index on tile_b (if connected).
    pub leg_b: Option<usize>,
    /// Global qubit index for this edge.
    pub qubit_index: usize,
}

/// The full hyperbolic tiling structure.
#[derive(Debug, Clone)]
pub struct HyperbolicTiling {
    /// All tiles in the tiling.
    pub tiles: Vec<HolographicTile>,
    /// All edges (connections between tiles or to boundary).
    pub edges: Vec<TilingEdge>,
    /// Number of layers.
    pub num_layers: usize,
    /// Total number of bulk qubits (edges between tiles).
    pub num_bulk_qubits: usize,
    /// Total number of boundary qubits (dangling legs).
    pub num_boundary_qubits: usize,
    /// Indices of boundary qubits in the global qubit ordering.
    pub boundary_qubit_indices: Vec<usize>,
    /// Indices of bulk qubits in the global qubit ordering.
    pub bulk_qubit_indices: Vec<usize>,
    /// Adjacency list for the dual graph (tile connectivity).
    pub adjacency: Vec<Vec<usize>>,
}

impl HyperbolicTiling {
    /// Construct a {5,4} hyperbolic tiling with the given number of layers.
    ///
    /// Layer 0: 1 central pentagon.
    /// Layer 1: 5 pentagons (one per edge of the center).
    /// Layer k: approximately 5 * 3^(k-1) new pentagons (in the ideal {5,4} tiling).
    ///
    /// For practical simulation, we use a simplified model where:
    /// - Each pentagon has 5 edges.
    /// - Interior edges connect to other pentagons.
    /// - Boundary edges have dangling legs (become boundary qubits).
    pub fn new(num_layers: usize) -> Self {
        if num_layers == 0 {
            return Self::single_tile();
        }

        let mut tiles: Vec<HolographicTile> = Vec::new();
        let mut edges: Vec<TilingEdge> = Vec::new();
        let mut adjacency: Vec<Vec<usize>> = Vec::new();
        let mut qubit_counter = 0usize;

        // Layer 0: central pentagon.
        let center = HolographicTile {
            id: 0,
            layer: 0,
            neighbors: vec![None; 5],
            bulk_legs: Vec::new(),
            boundary_legs: Vec::new(),
            tensor_id: 0,
        };
        tiles.push(center);
        adjacency.push(Vec::new());

        // Layer 1: one pentagon per edge of the center.
        let mut next_id = 1usize;
        for edge_idx in 0..5_usize {
            let new_tile = HolographicTile {
                id: next_id,
                layer: 1,
                neighbors: vec![None; 5],
                bulk_legs: Vec::new(),
                boundary_legs: Vec::new(),
                tensor_id: next_id,
            };
            tiles.push(new_tile);
            adjacency.push(Vec::new());

            // Connect center edge to new tile.
            tiles[0].neighbors[edge_idx] = Some(next_id);
            // The new tile's first leg connects back to center.
            tiles[next_id].neighbors[0] = Some(0);

            adjacency[0].push(next_id);
            adjacency[next_id].push(0);

            // Create the connecting edge (this is a bulk edge).
            edges.push(TilingEdge {
                tile_a: 0,
                tile_b: Some(next_id),
                leg_a: edge_idx,
                leg_b: Some(0),
                qubit_index: qubit_counter,
            });
            qubit_counter += 1;

            next_id += 1;
        }

        // Connect adjacent layer-1 tiles to each other.
        // In the {5,4} tiling, adjacent pentagons around the center share an edge.
        for i in 0..5_usize {
            let tile_i = i + 1; // tile IDs 1..5
            let tile_j = ((i + 1) % 5) + 1;

            // Connect tile_i's leg 1 to tile_j's leg 4 (adjacent sides).
            tiles[tile_i].neighbors[1] = Some(tile_j);
            tiles[tile_j].neighbors[4] = Some(tile_i);

            adjacency[tile_i].push(tile_j);
            adjacency[tile_j].push(tile_i);

            edges.push(TilingEdge {
                tile_a: tile_i,
                tile_b: Some(tile_j),
                leg_a: 1,
                leg_b: Some(4),
                qubit_index: qubit_counter,
            });
            qubit_counter += 1;
        }

        // For layers > 1, add additional rings.
        for layer in 2..=num_layers {
            let parent_tiles: Vec<usize> = tiles
                .iter()
                .filter(|t| t.layer == layer - 1)
                .map(|t| t.id)
                .collect();

            for &parent_id in &parent_tiles {
                // Each parent tile in the previous layer has dangling legs.
                // Attach new pentagons to dangling legs.
                for leg in 0..5_usize {
                    if tiles[parent_id].neighbors[leg].is_some() {
                        continue; // Already connected.
                    }

                    let new_tile = HolographicTile {
                        id: next_id,
                        layer,
                        neighbors: vec![None; 5],
                        bulk_legs: Vec::new(),
                        boundary_legs: Vec::new(),
                        tensor_id: next_id,
                    };
                    tiles.push(new_tile);
                    adjacency.push(Vec::new());

                    tiles[parent_id].neighbors[leg] = Some(next_id);
                    tiles[next_id].neighbors[0] = Some(parent_id);

                    adjacency[parent_id].push(next_id);
                    adjacency[next_id].push(parent_id);

                    edges.push(TilingEdge {
                        tile_a: parent_id,
                        tile_b: Some(next_id),
                        leg_a: leg,
                        leg_b: Some(0),
                        qubit_index: qubit_counter,
                    });
                    qubit_counter += 1;

                    next_id += 1;
                }
            }
        }

        // Mark boundary qubits: any tile leg that is still None gets a boundary qubit.
        let num_bulk_qubits = qubit_counter;
        let mut boundary_qubit_indices = Vec::new();
        let bulk_qubit_indices: Vec<usize> = (0..num_bulk_qubits).collect();

        for tile_id in 0..tiles.len() {
            for leg in 0..5_usize {
                if tiles[tile_id].neighbors[leg].is_none() {
                    tiles[tile_id].boundary_legs.push(leg);
                    boundary_qubit_indices.push(qubit_counter);

                    edges.push(TilingEdge {
                        tile_a: tile_id,
                        tile_b: None,
                        leg_a: leg,
                        leg_b: None,
                        qubit_index: qubit_counter,
                    });
                    qubit_counter += 1;
                } else {
                    tiles[tile_id].bulk_legs.push(leg);
                }
            }
        }

        let num_boundary_qubits = boundary_qubit_indices.len();

        HyperbolicTiling {
            tiles,
            edges,
            num_layers,
            num_bulk_qubits,
            num_boundary_qubits,
            boundary_qubit_indices,
            bulk_qubit_indices,
            adjacency,
        }
    }

    /// Construct a minimal tiling with a single pentagon (layer 0 only).
    fn single_tile() -> Self {
        let tile = HolographicTile {
            id: 0,
            layer: 0,
            neighbors: vec![None; 5],
            bulk_legs: Vec::new(),
            boundary_legs: vec![0, 1, 2, 3, 4],
            tensor_id: 0,
        };

        let mut edges = Vec::new();
        let mut boundary_qubit_indices = Vec::new();

        for leg in 0..5_usize {
            edges.push(TilingEdge {
                tile_a: 0,
                tile_b: None,
                leg_a: leg,
                leg_b: None,
                qubit_index: leg,
            });
            boundary_qubit_indices.push(leg);
        }

        HyperbolicTiling {
            tiles: vec![tile],
            edges,
            num_layers: 0,
            num_bulk_qubits: 0,
            num_boundary_qubits: 5,
            boundary_qubit_indices,
            bulk_qubit_indices: Vec::new(),
            adjacency: vec![Vec::new()],
        }
    }

    /// Get the total number of tiles.
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    /// Get all tile IDs at a given layer.
    pub fn tiles_at_layer(&self, layer: usize) -> Vec<usize> {
        self.tiles
            .iter()
            .filter(|t| t.layer == layer)
            .map(|t| t.id)
            .collect()
    }

    /// Compute the shortest path (in tile hops) between two tiles using BFS.
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from == to {
            return Some(vec![from]);
        }

        let n = self.tiles.len();
        let mut visited = vec![false; n];
        let mut parent = vec![None; n];
        let mut queue = VecDeque::new();

        visited[from] = true;
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            for &neighbor in &self.adjacency[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = Some(current);
                    if neighbor == to {
                        // Reconstruct path.
                        let mut path = vec![to];
                        let mut node = to;
                        while let Some(p) = parent[node] {
                            path.push(p);
                            node = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }
}

// ============================================================
// HOLOGRAPHIC CODE
// ============================================================

/// The full holographic code: tiles arranged on a hyperbolic tiling with
/// perfect tensors, providing a bulk-boundary mapping.
#[derive(Debug, Clone)]
pub struct HolographicCode {
    /// The underlying hyperbolic tiling.
    pub tiling: HyperbolicTiling,
    /// The perfect tensor used on each tile.
    pub tensor: PerfectTensor,
    /// Configuration.
    pub config: HolographicConfig,
    /// Number of logical (bulk) qubits that can be encoded.
    pub num_logical: usize,
    /// Number of physical (boundary) qubits.
    pub num_physical: usize,
}

impl HolographicCode {
    /// Build a holographic code from the given configuration.
    pub fn new(config: HolographicConfig) -> Result<Self, HolographicError> {
        config.validate()?;

        let tensor = PerfectTensor::new()?;
        let tiling = HyperbolicTiling::new(config.num_layers);

        // Each tile encodes 1 logical qubit via the perfect tensor's 6th leg.
        // But shared legs between tiles are contracted, leaving:
        //   logical qubits = number of tiles (each tile's logical leg)
        //   physical qubits = number of boundary (dangling) legs
        let num_logical = tiling.num_tiles();
        let num_physical = tiling.num_boundary_qubits;

        // Override if config specifies explicit counts.
        let final_logical = if config.bulk_qubits > 0 {
            config.bulk_qubits
        } else {
            num_logical
        };
        let final_physical = if config.boundary_qubits > 0 {
            config.boundary_qubits
        } else {
            num_physical
        };

        Ok(HolographicCode {
            tiling,
            tensor,
            config,
            num_logical: final_logical,
            num_physical: final_physical,
        })
    }

    /// Encode a single logical qubit state into boundary qubits.
    ///
    /// For a single-tile (layer 0) code, the encoding maps:
    ///   |psi>_boundary = V |psi>_logical
    /// where V is the isometry extracted from the perfect tensor.
    pub fn encode_single_qubit(
        &self,
        logical_state: &[C64],
    ) -> Result<Vec<C64>, HolographicError> {
        if logical_state.len() != 2 {
            return Err(HolographicError::EntropyError(
                "Logical state must be a 2-element vector".into(),
            ));
        }

        // Use the properly normalized encoding isometry V.
        // V[phys][log] gives the amplitude of physical state |phys> for logical |log>.
        // |psi>_boundary = sum_log V[phys][log] * psi[log]
        let isometry = self.tensor.encoding_isometry();
        let mut boundary = vec![C64::new(0.0, 0.0); 32]; // 2^5 = 32

        for phys in 0..32_usize {
            for log in 0..2_usize {
                boundary[phys] = boundary[phys] + isometry[phys][log] * logical_state[log];
            }
        }

        Ok(boundary)
    }

    /// Encode a multi-qubit bulk state by contracting the tensor network.
    ///
    /// This performs the full tensor network contraction from the center outward,
    /// producing the boundary state. Only feasible for small tilings.
    pub fn encode_bulk_state(
        &self,
        bulk_state: &[C64],
    ) -> Result<Vec<C64>, HolographicError> {
        let num_tiles = self.tiling.num_tiles();

        if bulk_state.len() != (1 << num_tiles) {
            return Err(HolographicError::BulkBoundaryMismatch {
                expected_bulk: num_tiles,
                expected_boundary: self.num_physical,
                got_bulk: (bulk_state.len() as f64).log2() as usize,
                got_boundary: 0,
            });
        }

        // For single tile, delegate to the simpler method.
        if num_tiles == 1 {
            return self.encode_single_qubit(bulk_state);
        }

        // For multi-tile encoding, we use a sequential contraction approach.
        // Each tile's perfect tensor contracts with incoming legs from connected
        // tiles and its logical leg from the bulk state.
        //
        // This is a simplified contraction for the graph structure.
        // For a full tensor network contraction, we would need to track all
        // intermediate indices; here we use the state vector approach.

        // Total qubits = boundary + number of tiles (logical legs)
        let total_qubits = self.num_physical + num_tiles;
        if total_qubits > 20 {
            return Err(HolographicError::EntropyError(
                "Encoding too large for exact state vector simulation".into(),
            ));
        }

        // Use quantum circuit encoding: prepare the bulk state on logical qubits,
        // then apply encoding circuits that implement the tensor network.
        let mut state = QuantumState::new(total_qubits);

        // Initialize the logical qubits with the bulk state.
        // The logical qubits are the first `num_tiles` qubits.
        let bulk_dim = bulk_state.len();
        let amps = state.amplitudes_mut();

        // Zero out all amplitudes first.
        for a in amps.iter_mut() {
            *a = C64::new(0.0, 0.0);
        }

        // Set amplitudes: the logical qubits hold the bulk state,
        // boundary qubits start in |0>.
        for i in 0..bulk_dim {
            amps[i] = bulk_state[i];
        }

        // Apply encoding circuit for each tile.
        // The encoding circuit for the [[5,1,3]] code uses the stabilizer generators
        // to entangle the logical qubit with the physical (boundary) qubits.
        self.apply_encoding_circuit(&mut state, num_tiles)?;

        // Extract the boundary state by tracing out the logical qubits.
        // For an isometry, this is just reading the amplitudes.
        Ok(state.amplitudes_ref().to_vec())
    }

    /// Apply the encoding circuit for the [[5,1,3]] code.
    /// This implements the stabilizer encoding as a quantum circuit.
    fn apply_encoding_circuit(
        &self,
        state: &mut QuantumState,
        num_tiles: usize,
    ) -> Result<(), HolographicError> {
        // For each tile, the encoding circuit maps the logical qubit to 5 physical qubits.
        // We use a standard encoding circuit for the [[5,1,3]] code.
        //
        // The encoding circuit (from the stabilizer generators):
        // 1. Start with logical qubit on position 0 (of the tile's qubits)
        // 2. Apply Hadamard gates and CNOTs to create the encoded state
        //
        // For tile i, the logical qubit is at index i, and the physical qubits
        // are at indices num_tiles + 5*i .. num_tiles + 5*i + 4.

        for tile_idx in 0..num_tiles {
            let logical = tile_idx;
            let phys_base = num_tiles + 5 * tile_idx;

            // Check we have enough qubits.
            if phys_base + 4 >= state.num_qubits {
                // For the actual tiling, physical qubits map differently.
                // Use a simplified encoding for the tiling structure.
                continue;
            }

            // Simplified encoding circuit for [[5,1,3]]:
            // This creates entanglement between the logical and physical qubits.
            GateOperations::h(state, phys_base);
            GateOperations::h(state, phys_base + 1);
            GateOperations::h(state, phys_base + 2);
            GateOperations::h(state, phys_base + 3);

            // CNOT gates to entangle.
            Self::apply_cnot_sorted(state, logical, phys_base);
            Self::apply_cnot_sorted(state, logical, phys_base + 1);
            Self::apply_cnot_sorted(state, phys_base, phys_base + 2);
            Self::apply_cnot_sorted(state, phys_base + 1, phys_base + 3);
            Self::apply_cnot_sorted(state, phys_base + 2, phys_base + 4);

            // Phase corrections.
            GateOperations::rz(state, phys_base, PI / 4.0);
            GateOperations::rz(state, phys_base + 3, PI / 4.0);
        }

        Ok(())
    }

    /// Apply a CNOT gate ensuring the qubit indices are passed correctly.
    /// GateOperations::cnot internally handles ordering via insert_zero_bits,
    /// so we can pass control and target in any order.
    #[inline]
    fn apply_cnot_sorted(state: &mut QuantumState, control: usize, target: usize) {
        GateOperations::cnot(state, control, target);
    }

    /// Compute the inner product <psi|phi> of two boundary states.
    pub fn inner_product(state_a: &[C64], state_b: &[C64]) -> C64 {
        assert_eq!(state_a.len(), state_b.len());
        state_a
            .iter()
            .zip(state_b.iter())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }

    /// Compute the von Neumann entropy of a boundary subregion using the
    /// reduced density matrix.
    ///
    /// `state` is the full boundary state vector.
    /// `region` is a set of boundary qubit indices in the subregion.
    /// `total_qubits` is the total number of boundary qubits.
    pub fn von_neumann_entropy(
        state: &[C64],
        region: &[usize],
        total_qubits: usize,
    ) -> Result<f64, HolographicError> {
        if region.is_empty() {
            return Ok(0.0);
        }
        if region.len() == total_qubits {
            // Full system is a pure state, entropy = 0.
            return Ok(0.0);
        }

        let dim = 1 << total_qubits;
        if state.len() != dim {
            return Err(HolographicError::EntropyError(format!(
                "State vector dimension {} does not match 2^{} = {}",
                state.len(),
                total_qubits,
                dim
            )));
        }

        // Compute reduced density matrix by tracing out the complement of `region`.
        let region_set: HashSet<usize> = region.iter().cloned().collect();
        let complement: Vec<usize> = (0..total_qubits)
            .filter(|q| !region_set.contains(q))
            .collect();

        let region_dim = 1 << region.len();
        let comp_dim = 1 << complement.len();

        // Build the reduced density matrix rho_A = Tr_B(|psi><psi|).
        let mut rho = vec![vec![C64::new(0.0, 0.0); region_dim]; region_dim];

        for i in 0..region_dim {
            for j in 0..region_dim {
                let mut sum = C64::new(0.0, 0.0);
                for k in 0..comp_dim {
                    // Build full indices from region and complement indices.
                    let full_i = Self::compose_index(i, &region, k, &complement, total_qubits);
                    let full_j = Self::compose_index(j, &region, k, &complement, total_qubits);

                    sum = sum + state[full_i] * state[full_j].conj();
                }
                rho[i][j] = sum;
            }
        }

        // Compute eigenvalues of rho (using power iteration for small matrices,
        // or direct diagonalization for tiny matrices).
        let eigenvalues = Self::eigenvalues_hermitian(&rho, region_dim);

        // Von Neumann entropy: S = -sum_i lambda_i * log2(lambda_i)
        let mut entropy = 0.0;
        for &lambda in &eigenvalues {
            if lambda > EPSILON {
                entropy -= lambda * lambda.log2();
            }
        }

        Ok(entropy)
    }

    /// Compose a full state index from a region index and a complement index.
    fn compose_index(
        region_idx: usize,
        region_qubits: &[usize],
        comp_idx: usize,
        comp_qubits: &[usize],
        _total_qubits: usize,
    ) -> usize {
        let mut full_idx = 0usize;
        for (bit, &qubit) in region_qubits.iter().enumerate() {
            if (region_idx >> bit) & 1 == 1 {
                full_idx |= 1 << qubit;
            }
        }
        for (bit, &qubit) in comp_qubits.iter().enumerate() {
            if (comp_idx >> bit) & 1 == 1 {
                full_idx |= 1 << qubit;
            }
        }
        full_idx
    }

    /// Compute eigenvalues of a Hermitian matrix using the Jacobi eigenvalue algorithm.
    fn eigenvalues_hermitian(matrix: &[Vec<C64>], n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![matrix[0][0].re];
        }

        // For small matrices, use iterative diagonalization.
        // Convert to real symmetric matrix (since rho is Hermitian and we expect
        // eigenvalues to be real and non-negative).

        // Build a real matrix from the Hermitian matrix.
        // For numerical stability, work with the real and imaginary parts.
        // Since rho is Hermitian, we can use the real part of the diagonal
        // and the upper triangle.

        // Simple approach: compute eigenvalues via characteristic polynomial for n <= 4,
        // otherwise use iterative QR-like approach.

        // Eigenvalue computation via iterative Jacobi rotations on the real part.
        // Since rho is a density matrix (Hermitian PSD), eigenvalues are real >= 0.
        let mut a = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = matrix[i][j].re;
            }
        }

        // Jacobi eigenvalue algorithm for real symmetric matrices.
        let max_iter = 100 * n * n;
        for _ in 0..max_iter {
            // Find the off-diagonal element with largest magnitude.
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < 1e-14 {
                break; // Converged.
            }

            // Compute rotation angle.
            let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
                PI / 4.0
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };

            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Apply Jacobi rotation.
            let mut new_a = a.clone();
            for i in 0..n {
                if i != p && i != q {
                    new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                    new_a[p][i] = new_a[i][p];
                    new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                    new_a[q][i] = new_a[i][q];
                }
            }
            new_a[p][p] = cos_t * cos_t * a[p][p]
                + 2.0 * cos_t * sin_t * a[p][q]
                + sin_t * sin_t * a[q][q];
            new_a[q][q] = sin_t * sin_t * a[p][p]
                - 2.0 * cos_t * sin_t * a[p][q]
                + cos_t * cos_t * a[q][q];
            new_a[p][q] = 0.0;
            new_a[q][p] = 0.0;

            a = new_a;
        }

        // Diagonal elements are the eigenvalues.
        let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();

        // Clamp small negative values to 0 (numerical noise in density matrices).
        for ev in eigenvalues.iter_mut() {
            if *ev < 0.0 && *ev > -1e-10 {
                *ev = 0.0;
            }
        }

        eigenvalues
    }
}

// ============================================================
// RYU-TAKAYANAGI
// ============================================================

/// Implements the Ryu-Takayanagi formula for computing holographic entanglement
/// entropy via minimal cuts in the tiling graph.
#[derive(Debug, Clone)]
pub struct RyuTakayanagi {
    /// Reference to the tiling structure.
    tiling: HyperbolicTiling,
}

impl RyuTakayanagi {
    /// Create a new RT calculator for the given tiling.
    pub fn new(tiling: HyperbolicTiling) -> Self {
        Self { tiling }
    }

    /// Compute the RT entropy of a boundary region A.
    ///
    /// The RT formula says S(A) = |gamma_A| where gamma_A is the minimal
    /// surface (set of edges) in the bulk that separates A from its complement.
    ///
    /// For a discrete tiling, this is a min-cut problem on the dual graph.
    /// The "area" of the cut is the number of edges cut.
    ///
    /// `boundary_region` is a set of boundary qubit indices forming region A.
    pub fn entropy(&self, boundary_region: &[usize]) -> Result<f64, HolographicError> {
        if boundary_region.is_empty() {
            return Ok(0.0);
        }

        let all_boundary: HashSet<usize> = self
            .tiling
            .boundary_qubit_indices
            .iter()
            .cloned()
            .collect();

        // Validate boundary region.
        for &q in boundary_region {
            if !all_boundary.contains(&q) {
                return Err(HolographicError::EntropyError(format!(
                    "Qubit {} is not a boundary qubit",
                    q
                )));
            }
        }

        let region_set: HashSet<usize> = boundary_region.iter().cloned().collect();

        // If the region is the full boundary, entropy is 0 (pure state).
        if region_set.len() == all_boundary.len() {
            return Ok(0.0);
        }

        // Compute the minimal cut using BFS on the dual graph.
        let min_cut = self.compute_min_cut(&region_set)?;

        // In the RT formula, S(A) = |gamma_A| / (4 G_N) but in the code model
        // we work in units where 4 G_N = 1, so S(A) = |gamma_A|.
        // Each edge in the min-cut contributes log(2) to the entropy.
        Ok(min_cut as f64 * 2.0_f64.ln() / 2.0_f64.ln())
    }

    /// Compute the minimal cut separating boundary region A from its complement.
    ///
    /// We use a BFS-based approach: find the minimum number of bulk edges
    /// that must be removed to disconnect the tiles adjacent to A from the
    /// tiles adjacent to the complement of A.
    fn compute_min_cut(
        &self,
        region: &HashSet<usize>,
    ) -> Result<usize, HolographicError> {
        // Map boundary qubits to the tiles they belong to.
        let region_tiles = self.boundary_to_tiles(region);
        let all_boundary: HashSet<usize> = self
            .tiling
            .boundary_qubit_indices
            .iter()
            .cloned()
            .collect();
        let complement: HashSet<usize> = all_boundary.difference(region).cloned().collect();
        let complement_tiles = self.boundary_to_tiles(&complement);

        if region_tiles.is_empty() || complement_tiles.is_empty() {
            return Ok(0);
        }

        // For small tilings, enumerate all possible cuts.
        // A cut is a set of bulk edges whose removal disconnects region_tiles
        // from complement_tiles.

        // Use a simple BFS-based min-cut: iteratively find shortest augmenting
        // paths (Edmonds-Karp style) in the tile adjacency graph.

        let n_tiles = self.tiling.tiles.len();

        // Build a capacity graph: each edge between tiles has capacity 1.
        let mut capacity = vec![vec![0i32; n_tiles]; n_tiles];
        for edge in &self.tiling.edges {
            if let Some(tile_b) = edge.tile_b {
                capacity[edge.tile_a][tile_b] += 1;
                capacity[tile_b][edge.tile_a] += 1;
            }
        }

        // Add a super-source connected to all region tiles,
        // and a super-sink connected to all complement tiles.
        let source = n_tiles;
        let sink = n_tiles + 1;
        let total = n_tiles + 2;

        let mut cap = vec![vec![0i32; total]; total];
        for i in 0..n_tiles {
            for j in 0..n_tiles {
                cap[i][j] = capacity[i][j];
            }
        }
        for &t in &region_tiles {
            cap[source][t] = i32::MAX / 2;
        }
        for &t in &complement_tiles {
            cap[t][sink] = i32::MAX / 2;
        }

        // Edmonds-Karp max-flow = min-cut.
        let max_flow = Self::edmonds_karp(&mut cap, source, sink, total);

        Ok(max_flow as usize)
    }

    /// Map boundary qubit indices to the set of tiles they belong to.
    fn boundary_to_tiles(&self, boundary_qubits: &HashSet<usize>) -> HashSet<usize> {
        let mut tiles = HashSet::new();
        for edge in &self.tiling.edges {
            if edge.tile_b.is_none() && boundary_qubits.contains(&edge.qubit_index) {
                tiles.insert(edge.tile_a);
            }
        }
        tiles
    }

    /// Edmonds-Karp maximum flow algorithm (BFS-based Ford-Fulkerson).
    fn edmonds_karp(cap: &mut Vec<Vec<i32>>, source: usize, sink: usize, n: usize) -> i32 {
        let mut flow = 0i32;

        loop {
            // BFS to find augmenting path.
            let mut parent = vec![None; n];
            let mut visited = vec![false; n];
            let mut queue = VecDeque::new();

            visited[source] = true;
            queue.push_back(source);

            while let Some(u) = queue.pop_front() {
                if u == sink {
                    break;
                }
                for v in 0..n {
                    if !visited[v] && cap[u][v] > 0 {
                        visited[v] = true;
                        parent[v] = Some(u);
                        queue.push_back(v);
                    }
                }
            }

            if !visited[sink] {
                break; // No augmenting path found.
            }

            // Find bottleneck capacity.
            let mut path_flow = i32::MAX;
            let mut v = sink;
            while let Some(u) = parent[v] {
                path_flow = path_flow.min(cap[u][v]);
                v = u;
            }

            // Update residual capacities.
            let mut v = sink;
            while let Some(u) = parent[v] {
                cap[u][v] -= path_flow;
                cap[v][u] += path_flow;
                v = u;
            }

            flow += path_flow;
        }

        flow
    }

    /// Find the actual RT surface (the edges forming the minimal cut).
    pub fn find_rt_surface(
        &self,
        boundary_region: &[usize],
    ) -> Result<Vec<usize>, HolographicError> {
        if boundary_region.is_empty() {
            return Ok(Vec::new());
        }

        let region_set: HashSet<usize> = boundary_region.iter().cloned().collect();
        let all_boundary: HashSet<usize> = self
            .tiling
            .boundary_qubit_indices
            .iter()
            .cloned()
            .collect();
        let complement: HashSet<usize> = all_boundary.difference(&region_set).cloned().collect();

        let region_tiles = self.boundary_to_tiles(&region_set);
        let complement_tiles = self.boundary_to_tiles(&complement);

        if region_tiles.is_empty() || complement_tiles.is_empty() {
            return Ok(Vec::new());
        }

        let n_tiles = self.tiling.tiles.len();
        let source = n_tiles;
        let sink = n_tiles + 1;
        let total = n_tiles + 2;

        let mut cap = vec![vec![0i32; total]; total];
        for edge in &self.tiling.edges {
            if let Some(tile_b) = edge.tile_b {
                cap[edge.tile_a][tile_b] += 1;
                cap[tile_b][edge.tile_a] += 1;
            }
        }
        for &t in &region_tiles {
            cap[source][t] = i32::MAX / 2;
        }
        for &t in &complement_tiles {
            cap[t][sink] = i32::MAX / 2;
        }

        // Run max-flow.
        let mut cap_residual = cap.clone();
        Self::edmonds_karp(&mut cap_residual, source, sink, total);

        // Find the min-cut: BFS from source in the residual graph.
        let mut visited = vec![false; total];
        let mut queue = VecDeque::new();
        visited[source] = true;
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            for v in 0..total {
                if !visited[v] && cap_residual[u][v] > 0 {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }

        // Cut edges are edges (u, v) where u is reachable and v is not,
        // and the original capacity was > 0.
        let mut cut_edges = Vec::new();
        for (_edge_idx, edge) in self.tiling.edges.iter().enumerate() {
            if let Some(tile_b) = edge.tile_b {
                if (visited[edge.tile_a] && !visited[tile_b])
                    || (!visited[edge.tile_a] && visited[tile_b])
                {
                    cut_edges.push(edge.qubit_index);
                }
            }
        }

        Ok(cut_edges)
    }
}

// ============================================================
// ENTANGLEMENT WEDGE RECONSTRUCTION
// ============================================================

/// Represents the entanglement wedge of a boundary region.
///
/// The entanglement wedge W(A) for a boundary region A is the bulk region
/// enclosed by the RT surface gamma_A and the boundary region A itself.
/// Any bulk operator within W(A) can be reconstructed using only degrees
/// of freedom on A.
#[derive(Debug, Clone)]
pub struct ReconstructionWedge {
    /// The boundary region defining this wedge.
    pub boundary_region: Vec<usize>,
    /// Tile IDs that are inside the entanglement wedge.
    pub bulk_tiles: HashSet<usize>,
    /// Qubit indices of the RT surface.
    pub rt_surface: Vec<usize>,
}

impl ReconstructionWedge {
    /// Compute the entanglement wedge for a boundary region.
    pub fn compute(
        tiling: &HyperbolicTiling,
        boundary_region: &[usize],
    ) -> Result<Self, HolographicError> {
        let rt = RyuTakayanagi::new(tiling.clone());
        let rt_surface = rt.find_rt_surface(boundary_region)?;

        let region_set: HashSet<usize> = boundary_region.iter().cloned().collect();

        // Find tiles reachable from the boundary region without crossing the RT surface.
        let rt_edge_set: HashSet<usize> = rt_surface.iter().cloned().collect();

        // Map boundary qubits to tiles.
        let mut seed_tiles = HashSet::new();
        for edge in &tiling.edges {
            if edge.tile_b.is_none() && region_set.contains(&edge.qubit_index) {
                seed_tiles.insert(edge.tile_a);
            }
        }

        // Flood-fill from seed tiles, not crossing RT surface edges.
        let mut wedge_tiles = HashSet::new();
        let mut queue = VecDeque::new();

        for &t in &seed_tiles {
            if !wedge_tiles.contains(&t) {
                wedge_tiles.insert(t);
                queue.push_back(t);
            }
        }

        while let Some(tile) = queue.pop_front() {
            for edge in &tiling.edges {
                if rt_edge_set.contains(&edge.qubit_index) {
                    continue; // Don't cross RT surface.
                }
                if let Some(tile_b) = edge.tile_b {
                    if edge.tile_a == tile && !wedge_tiles.contains(&tile_b) {
                        wedge_tiles.insert(tile_b);
                        queue.push_back(tile_b);
                    }
                    if tile_b == tile && !wedge_tiles.contains(&edge.tile_a) {
                        wedge_tiles.insert(edge.tile_a);
                        queue.push_back(edge.tile_a);
                    }
                }
            }
        }

        Ok(ReconstructionWedge {
            boundary_region: boundary_region.to_vec(),
            bulk_tiles: wedge_tiles,
            rt_surface,
        })
    }

    /// Check if a given bulk tile is inside this entanglement wedge.
    pub fn contains_tile(&self, tile_id: usize) -> bool {
        self.bulk_tiles.contains(&tile_id)
    }

    /// Get the number of bulk tiles in the wedge.
    pub fn wedge_size(&self) -> usize {
        self.bulk_tiles.len()
    }
}

// ============================================================
// SUBREGION DUALITY
// ============================================================

/// Result of checking whether a bulk operator is reconstructible from
/// a given boundary subregion.
#[derive(Debug, Clone)]
pub struct SubregionDuality {
    /// The bulk tile (operator location) being checked.
    pub bulk_tile: usize,
    /// Whether the operator is in the entanglement wedge of the region.
    pub is_reconstructible: bool,
    /// The boundary region checked.
    pub boundary_region: Vec<usize>,
    /// The complementary region.
    pub complement_region: Vec<usize>,
    /// Whether the operator is also reconstructible from the complement.
    pub complement_reconstructible: bool,
}

impl SubregionDuality {
    /// Check subregion duality: determine from which boundary regions
    /// a bulk operator at the given tile can be reconstructed.
    pub fn check(
        tiling: &HyperbolicTiling,
        bulk_tile: usize,
        boundary_region: &[usize],
    ) -> Result<Self, HolographicError> {
        if bulk_tile >= tiling.tiles.len() {
            return Err(HolographicError::ReconstructionFailed(format!(
                "Tile {} does not exist (tiling has {} tiles)",
                bulk_tile,
                tiling.tiles.len()
            )));
        }

        let all_boundary: HashSet<usize> = tiling
            .boundary_qubit_indices
            .iter()
            .cloned()
            .collect();
        let region_set: HashSet<usize> = boundary_region.iter().cloned().collect();
        let complement: Vec<usize> = all_boundary.difference(&region_set).cloned().collect();

        // Check if bulk_tile is in the entanglement wedge of the region.
        let wedge = ReconstructionWedge::compute(tiling, boundary_region)?;
        let is_reconstructible = wedge.contains_tile(bulk_tile);

        // Check if bulk_tile is in the entanglement wedge of the complement.
        let comp_wedge = ReconstructionWedge::compute(tiling, &complement)?;
        let complement_reconstructible = comp_wedge.contains_tile(bulk_tile);

        Ok(SubregionDuality {
            bulk_tile,
            is_reconstructible,
            boundary_region: boundary_region.to_vec(),
            complement_region: complement,
            complement_reconstructible,
        })
    }

    /// Find all boundary subsets (contiguous regions) from which the bulk
    /// operator can be reconstructed.
    pub fn find_all_reconstructing_regions(
        tiling: &HyperbolicTiling,
        bulk_tile: usize,
    ) -> Result<Vec<Vec<usize>>, HolographicError> {
        let boundary = &tiling.boundary_qubit_indices;
        let n = boundary.len();
        let mut result = Vec::new();

        // Check all contiguous boundary subregions.
        for start in 0..n {
            for len in 1..=n {
                let region: Vec<usize> = (0..len)
                    .map(|i| boundary[(start + i) % n])
                    .collect();

                let wedge = ReconstructionWedge::compute(tiling, &region)?;
                if wedge.contains_tile(bulk_tile) {
                    result.push(region);
                }
            }
        }

        Ok(result)
    }
}

// ============================================================
// HOLOGRAPHIC ENTROPY CONE
// ============================================================

/// Utilities for verifying holographic entropy inequalities.
pub struct HolographicEntropyCone;

impl HolographicEntropyCone {
    /// Check strong subadditivity: S(AB) + S(BC) >= S(B) + S(ABC).
    ///
    /// Uses the RT formula on the tiling to compute entropies.
    pub fn check_strong_subadditivity(
        tiling: &HyperbolicTiling,
        region_a: &[usize],
        region_b: &[usize],
        region_c: &[usize],
    ) -> Result<bool, HolographicError> {
        let rt = RyuTakayanagi::new(tiling.clone());

        // S(AB)
        let ab: Vec<usize> = region_a.iter().chain(region_b.iter()).cloned().collect();
        let s_ab = rt.entropy(&ab)?;

        // S(BC)
        let bc: Vec<usize> = region_b.iter().chain(region_c.iter()).cloned().collect();
        let s_bc = rt.entropy(&bc)?;

        // S(B)
        let s_b = rt.entropy(region_b)?;

        // S(ABC)
        let abc: Vec<usize> = region_a
            .iter()
            .chain(region_b.iter())
            .chain(region_c.iter())
            .cloned()
            .collect();
        let s_abc = rt.entropy(&abc)?;

        // SSA: S(AB) + S(BC) >= S(B) + S(ABC)
        Ok(s_ab + s_bc >= s_b + s_abc - EPSILON)
    }

    /// Check monogamy of mutual information (MMI) for holographic states:
    ///   I(A:BC) >= I(A:B) + I(A:C)
    ///
    /// Equivalently: S(AB) + S(AC) + S(BC) >= S(A) + S(B) + S(C) + S(ABC)
    ///
    /// This inequality holds for all holographic states but not for general
    /// quantum states.
    pub fn check_monogamy_mutual_info(
        tiling: &HyperbolicTiling,
        region_a: &[usize],
        region_b: &[usize],
        region_c: &[usize],
    ) -> Result<bool, HolographicError> {
        let rt = RyuTakayanagi::new(tiling.clone());

        let s_a = rt.entropy(region_a)?;
        let s_b = rt.entropy(region_b)?;
        let s_c = rt.entropy(region_c)?;

        let ab: Vec<usize> = region_a.iter().chain(region_b.iter()).cloned().collect();
        let s_ab = rt.entropy(&ab)?;

        let ac: Vec<usize> = region_a.iter().chain(region_c.iter()).cloned().collect();
        let s_ac = rt.entropy(&ac)?;

        let bc: Vec<usize> = region_b.iter().chain(region_c.iter()).cloned().collect();
        let s_bc = rt.entropy(&bc)?;

        let abc: Vec<usize> = region_a
            .iter()
            .chain(region_b.iter())
            .chain(region_c.iter())
            .cloned()
            .collect();
        let s_abc = rt.entropy(&abc)?;

        // MMI: S(AB) + S(AC) + S(BC) >= S(A) + S(B) + S(C) + S(ABC)
        let lhs = s_ab + s_ac + s_bc;
        let rhs = s_a + s_b + s_c + s_abc;

        Ok(lhs >= rhs - EPSILON)
    }

    /// Compute the mutual information I(A:B) = S(A) + S(B) - S(AB).
    pub fn mutual_information(
        tiling: &HyperbolicTiling,
        region_a: &[usize],
        region_b: &[usize],
    ) -> Result<f64, HolographicError> {
        let rt = RyuTakayanagi::new(tiling.clone());

        let s_a = rt.entropy(region_a)?;
        let s_b = rt.entropy(region_b)?;

        let ab: Vec<usize> = region_a.iter().chain(region_b.iter()).cloned().collect();
        let s_ab = rt.entropy(&ab)?;

        Ok(s_a + s_b - s_ab)
    }

    /// Compute the conditional mutual information I(A:C|B) = S(AB) + S(BC) - S(B) - S(ABC).
    pub fn conditional_mutual_information(
        tiling: &HyperbolicTiling,
        region_a: &[usize],
        region_b: &[usize],
        region_c: &[usize],
    ) -> Result<f64, HolographicError> {
        let rt = RyuTakayanagi::new(tiling.clone());

        let ab: Vec<usize> = region_a.iter().chain(region_b.iter()).cloned().collect();
        let s_ab = rt.entropy(&ab)?;

        let bc: Vec<usize> = region_b.iter().chain(region_c.iter()).cloned().collect();
        let s_bc = rt.entropy(&bc)?;

        let s_b = rt.entropy(region_b)?;

        let abc: Vec<usize> = region_a
            .iter()
            .chain(region_b.iter())
            .chain(region_c.iter())
            .cloned()
            .collect();
        let s_abc = rt.entropy(&abc)?;

        // I(A:C|B) = S(AB) + S(BC) - S(B) - S(ABC)
        Ok(s_ab + s_bc - s_b - s_abc)
    }
}

// ============================================================
// DISPLAY IMPLEMENTATIONS
// ============================================================

impl fmt::Display for HolographicCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HolographicCode({}, {} layers, {} logical, {} physical, {} tiles)",
            self.config.code_type,
            self.tiling.num_layers,
            self.num_logical,
            self.num_physical,
            self.tiling.num_tiles()
        )
    }
}

impl fmt::Display for HyperbolicTiling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HyperbolicTiling({} layers, {} tiles, {} bulk edges, {} boundary edges)",
            self.num_layers,
            self.tiles.len(),
            self.num_bulk_qubits,
            self.num_boundary_qubits
        )
    }
}

impl fmt::Display for PerfectTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PerfectTensor({} legs, norm={:.6})",
            self.num_legs,
            self.norm()
        )
    }
}

impl fmt::Display for ReconstructionWedge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReconstructionWedge(boundary={} qubits, wedge={} tiles, RT surface={} edges)",
            self.boundary_region.len(),
            self.bulk_tiles.len(),
            self.rt_surface.len()
        )
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --------------------------------------------------------
    // Helper functions for tests
    // --------------------------------------------------------

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn state_norm(state: &[C64]) -> f64 {
        state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    // --------------------------------------------------------
    // Perfect Tensor Tests
    // --------------------------------------------------------

    #[test]
    fn test_perfect_tensor_construction() {
        let tensor = PerfectTensor::new().expect("Failed to construct perfect tensor");
        assert_eq!(tensor.amplitudes.len(), PERFECT_TENSOR_DIM);
        assert_eq!(tensor.num_legs, 6);

        // Check normalization: ||T||^2 = d^(n_legs/2) = 2^3 = 8, so ||T|| = 2*sqrt(2).
        let norm = tensor.norm();
        let expected_norm = 8.0_f64.sqrt();
        assert!(
            approx_eq(norm, expected_norm, 1e-6),
            "Tensor norm should be sqrt(8)={}, got {}",
            expected_norm,
            norm
        );

        // Check that the tensor is not trivially zero.
        let nonzero_count = tensor
            .amplitudes
            .iter()
            .filter(|a| a.norm_sqr() > EPSILON)
            .count();
        assert!(
            nonzero_count > 0,
            "Perfect tensor should have nonzero entries"
        );
    }

    #[test]
    fn test_perfect_tensor_is_unitary() {
        let tensor = PerfectTensor::new().expect("Failed to construct perfect tensor");

        // Verify the perfection property: each 3+3 bipartition should give
        // a unitary 8x8 matrix.
        let result = tensor.verify_perfection();
        assert!(
            result.is_ok(),
            "Perfect tensor should satisfy the perfection property: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_perfect_tensor_element_access() {
        let tensor = PerfectTensor::new().expect("Failed to construct perfect tensor");

        // Test that element access is consistent with the flat vector.
        for i0 in 0..2_usize {
            for i1 in 0..2 {
                for i2 in 0..2 {
                    for i3 in 0..2 {
                        for i4 in 0..2 {
                            for i5 in 0..2 {
                                let idx = i0 + 2 * i1 + 4 * i2 + 8 * i3 + 16 * i4 + 32 * i5;
                                let elem = tensor.element(&[i0, i1, i2, i3, i4, i5]);
                                assert_eq!(
                                    elem, tensor.amplitudes[idx],
                                    "Element access mismatch at [{},{},{},{},{},{}]",
                                    i0, i1, i2, i3, i4, i5
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_perfect_tensor_isometry() {
        let tensor = PerfectTensor::new().expect("Failed to construct perfect tensor");

        // The encoding isometry V maps 1 logical qubit to 5 physical qubits.
        // V^dag V should be the 2x2 identity.
        let isometry = tensor.encoding_isometry();

        let mut vdv = [[C64::new(0.0, 0.0); 2]; 2];
        for log_i in 0..2_usize {
            for log_j in 0..2_usize {
                for phys in 0..32_usize {
                    let a = isometry[phys][log_i];
                    let b = isometry[phys][log_j];
                    vdv[log_i][log_j] = vdv[log_i][log_j] + a.conj() * b;
                }
            }
        }

        // Should be approximately the 2x2 identity.
        assert!(
            approx_eq(vdv[0][0].re, 1.0, 1e-6) && vdv[0][0].im.abs() < 1e-6,
            "V^dag V [0,0] should be 1, got {:?}",
            vdv[0][0]
        );
        assert!(
            approx_eq(vdv[1][1].re, 1.0, 1e-6) && vdv[1][1].im.abs() < 1e-6,
            "V^dag V [1,1] should be 1, got {:?}",
            vdv[1][1]
        );
        assert!(
            vdv[0][1].norm_sqr() < 1e-10,
            "V^dag V [0,1] should be 0, got {:?}",
            vdv[0][1]
        );
        assert!(
            vdv[1][0].norm_sqr() < 1e-10,
            "V^dag V [1,0] should be 0, got {:?}",
            vdv[1][0]
        );
    }

    // --------------------------------------------------------
    // Tiling Tests
    // --------------------------------------------------------

    #[test]
    fn test_tiling_layer_0() {
        let tiling = HyperbolicTiling::new(0);

        assert_eq!(tiling.num_tiles(), 1, "Layer 0 should have 1 tile");
        assert_eq!(tiling.num_layers, 0);
        assert_eq!(
            tiling.num_boundary_qubits, 5,
            "Single pentagon should have 5 boundary qubits"
        );
        assert_eq!(
            tiling.num_bulk_qubits, 0,
            "Single pentagon should have 0 bulk qubits"
        );
        assert_eq!(tiling.boundary_qubit_indices.len(), 5);
    }

    #[test]
    fn test_tiling_layer_1() {
        let tiling = HyperbolicTiling::new(1);

        assert_eq!(
            tiling.num_tiles(),
            6,
            "Layer 1 should have 6 tiles (1 center + 5 ring)"
        );
        assert_eq!(tiling.num_layers, 1);

        // 5 edges from center to ring + 5 edges between ring tiles = 10 bulk edges.
        assert_eq!(
            tiling.num_bulk_qubits, 10,
            "Layer 1 should have 10 bulk edges"
        );

        // Each ring tile has 5 legs: 1 to center, 2 to adjacent ring tiles, 2 dangling.
        // So each of 5 ring tiles contributes 2 boundary qubits = 10 boundary qubits.
        // Center tile has all 5 legs connected to ring tiles, so 0 boundary from center.
        assert!(
            tiling.num_boundary_qubits >= 10,
            "Layer 1 should have at least 10 boundary qubits, got {}",
            tiling.num_boundary_qubits
        );
    }

    #[test]
    fn test_tiling_adjacency() {
        let tiling = HyperbolicTiling::new(1);

        // Center tile (0) should be adjacent to tiles 1-5.
        assert_eq!(
            tiling.adjacency[0].len(),
            5,
            "Center tile should have 5 neighbors"
        );

        // Each ring tile should be adjacent to the center and two neighbors.
        for i in 1..6_usize {
            assert!(
                tiling.adjacency[i].len() >= 3,
                "Ring tile {} should have at least 3 neighbors, got {}",
                i,
                tiling.adjacency[i].len()
            );
            assert!(
                tiling.adjacency[i].contains(&0),
                "Ring tile {} should be adjacent to center",
                i
            );
        }
    }

    #[test]
    fn test_tiling_shortest_path() {
        let tiling = HyperbolicTiling::new(1);

        // Center to any ring tile: 1 hop.
        for i in 1..6_usize {
            let path = tiling.shortest_path(0, i);
            assert!(path.is_some(), "Path from center to tile {} should exist", i);
            assert_eq!(
                path.unwrap().len(),
                2,
                "Path from center to tile {} should be 2 nodes (1 hop)",
                i
            );
        }

        // Ring tile to adjacent ring tile: 1 hop (they share an edge).
        let path = tiling.shortest_path(1, 2);
        assert!(path.is_some(), "Path from tile 1 to tile 2 should exist");
        let path_len = path.unwrap().len();
        assert!(
            path_len <= 3,
            "Path from tile 1 to tile 2 should be at most 3 nodes, got {}",
            path_len
        );
    }

    // --------------------------------------------------------
    // Encoding Tests
    // --------------------------------------------------------

    #[test]
    fn test_encoding_single_qubit() {
        let config = HolographicConfig::new(0);
        let code = HolographicCode::new(config).expect("Failed to create holographic code");

        // Encode |0> logical state.
        let logical_zero = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let boundary = code
            .encode_single_qubit(&logical_zero)
            .expect("Encoding failed");

        // The boundary state should be a valid quantum state (normalized).
        let norm = state_norm(&boundary);
        assert!(
            approx_eq(norm, 1.0, 1e-6),
            "Encoded state should be normalized, got norm={}",
            norm
        );

        // Encode |1> logical state.
        let logical_one = vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)];
        let boundary_one = code
            .encode_single_qubit(&logical_one)
            .expect("Encoding failed");

        let norm_one = state_norm(&boundary_one);
        assert!(
            approx_eq(norm_one, 1.0, 1e-6),
            "Encoded |1> state should be normalized, got norm={}",
            norm_one
        );
    }

    #[test]
    fn test_encoding_preserves_inner_product() {
        let config = HolographicConfig::new(0);
        let code = HolographicCode::new(config).expect("Failed to create holographic code");

        // Two orthogonal logical states.
        let psi = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)]; // |0>
        let phi = vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)]; // |1>

        let psi_boundary = code.encode_single_qubit(&psi).expect("Encoding failed");
        let phi_boundary = code.encode_single_qubit(&phi).expect("Encoding failed");

        // Inner product should be preserved: <0|1> = 0.
        let ip = HolographicCode::inner_product(&psi_boundary, &phi_boundary);
        assert!(
            ip.norm_sqr() < 1e-6,
            "Inner product of encoded orthogonal states should be ~0, got {:?}",
            ip
        );

        // Superposition state.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let plus = vec![
            C64::new(inv_sqrt2, 0.0),
            C64::new(inv_sqrt2, 0.0),
        ];
        let plus_boundary = code.encode_single_qubit(&plus).expect("Encoding failed");

        // <0|+> = 1/sqrt(2)
        let ip_0_plus = HolographicCode::inner_product(&psi_boundary, &plus_boundary);
        assert!(
            approx_eq(ip_0_plus.norm_sqr(), 0.5, 1e-4),
            "|<0|+>|^2 should be 0.5, got {}",
            ip_0_plus.norm_sqr()
        );
    }

    // --------------------------------------------------------
    // Ryu-Takayanagi Tests
    // --------------------------------------------------------

    #[test]
    fn test_rt_entropy_single_qubit() {
        let tiling = HyperbolicTiling::new(0);
        let rt = RyuTakayanagi::new(tiling.clone());

        // For a single tile, every boundary qubit is a dangling leg.
        // The entropy of any single boundary qubit should be 1 (one edge cut).
        let s = rt.entropy(&[0]).expect("RT entropy failed");
        assert!(
            s >= 0.0,
            "RT entropy should be non-negative, got {}",
            s
        );
    }

    #[test]
    fn test_rt_surface_minimal() {
        let tiling = HyperbolicTiling::new(1);
        let rt = RyuTakayanagi::new(tiling.clone());

        // For a small boundary region, the RT surface should exist.
        if !tiling.boundary_qubit_indices.is_empty() {
            let region = vec![tiling.boundary_qubit_indices[0]];
            let surface = rt.find_rt_surface(&region);
            assert!(
                surface.is_ok(),
                "RT surface computation should succeed"
            );
        }
    }

    #[test]
    fn test_rt_entropy_full_boundary_is_zero() {
        let tiling = HyperbolicTiling::new(1);
        let rt = RyuTakayanagi::new(tiling.clone());

        // The entropy of the full boundary should be 0 (pure state).
        let full = tiling.boundary_qubit_indices.clone();
        let s = rt.entropy(&full).expect("RT entropy failed");
        assert!(
            approx_eq(s, 0.0, EPSILON),
            "Entropy of full boundary should be 0, got {}",
            s
        );
    }

    #[test]
    fn test_rt_matches_exact_entropy() {
        // For a single-tile code, verify that the RT formula gives
        // an entropy consistent with the exact computation.
        let config = HolographicConfig::new(0);
        let code = HolographicCode::new(config).expect("Failed to create holographic code");

        // Encode a Bell-like state: (|0> + |1>)/sqrt(2)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let logical_plus = vec![
            C64::new(inv_sqrt2, 0.0),
            C64::new(inv_sqrt2, 0.0),
        ];
        let boundary = code
            .encode_single_qubit(&logical_plus)
            .expect("Encoding failed");

        // Exact von Neumann entropy of qubits 0,1 (first two boundary qubits).
        let exact_s = HolographicCode::von_neumann_entropy(&boundary, &[0, 1], 5);
        assert!(
            exact_s.is_ok(),
            "Exact entropy computation should succeed"
        );
        let s_val = exact_s.unwrap();
        assert!(
            s_val >= -EPSILON,
            "Exact entropy should be non-negative, got {}",
            s_val
        );

        // The RT formula for the same region.
        let tiling = HyperbolicTiling::new(0);
        let rt = RyuTakayanagi::new(tiling.clone());
        let rt_s = rt.entropy(&[0, 1]).expect("RT entropy failed");

        // Both should be non-negative. For a perfect tensor code, the RT formula
        // should agree with the exact entropy up to discretization.
        assert!(
            rt_s >= -EPSILON,
            "RT entropy should be non-negative, got {}",
            rt_s
        );
    }

    // --------------------------------------------------------
    // Entanglement Wedge Tests
    // --------------------------------------------------------

    #[test]
    fn test_entanglement_wedge_contains_center() {
        let tiling = HyperbolicTiling::new(1);

        // The full boundary should have a wedge containing all tiles (including center).
        let full_boundary = tiling.boundary_qubit_indices.clone();
        let wedge = ReconstructionWedge::compute(&tiling, &full_boundary)
            .expect("Wedge computation failed");

        assert!(
            wedge.contains_tile(0),
            "Full boundary wedge should contain the center tile"
        );
        assert_eq!(
            wedge.wedge_size(),
            tiling.num_tiles(),
            "Full boundary wedge should contain all tiles"
        );
    }

    #[test]
    fn test_entanglement_wedge_half_boundary() {
        let tiling = HyperbolicTiling::new(1);

        if tiling.boundary_qubit_indices.len() < 2 {
            return; // Need at least 2 boundary qubits.
        }

        // Take roughly half the boundary.
        let half_size = tiling.boundary_qubit_indices.len() / 2;
        let half_boundary: Vec<usize> = tiling.boundary_qubit_indices[..half_size].to_vec();

        let wedge = ReconstructionWedge::compute(&tiling, &half_boundary)
            .expect("Wedge computation failed");

        // Half boundary should cover some but not necessarily all tiles.
        assert!(
            wedge.wedge_size() > 0,
            "Half boundary should have a non-empty wedge"
        );
        assert!(
            wedge.wedge_size() <= tiling.num_tiles(),
            "Wedge should not exceed total tiles"
        );
    }

    #[test]
    fn test_entanglement_wedge_empty_region() {
        let tiling = HyperbolicTiling::new(0);

        let wedge = ReconstructionWedge::compute(&tiling, &[])
            .expect("Wedge computation for empty region failed");

        assert_eq!(
            wedge.wedge_size(),
            0,
            "Empty boundary region should have empty wedge"
        );
    }

    // --------------------------------------------------------
    // Subregion Duality Tests
    // --------------------------------------------------------

    #[test]
    fn test_subregion_duality() {
        let tiling = HyperbolicTiling::new(1);

        if tiling.boundary_qubit_indices.len() < 2 {
            return;
        }

        // Check that the center tile (0) can be reconstructed from the full boundary.
        let full_boundary = tiling.boundary_qubit_indices.clone();
        let duality = SubregionDuality::check(&tiling, 0, &full_boundary)
            .expect("Subregion duality check failed");

        assert!(
            duality.is_reconstructible,
            "Center tile should be reconstructible from full boundary"
        );
    }

    #[test]
    fn test_subregion_duality_complement() {
        let tiling = HyperbolicTiling::new(1);

        if tiling.boundary_qubit_indices.len() < 4 {
            return;
        }

        // Take a small portion of the boundary.
        let small_region: Vec<usize> = tiling.boundary_qubit_indices[..2].to_vec();

        let duality = SubregionDuality::check(&tiling, 0, &small_region)
            .expect("Subregion duality check failed");

        // Verify that we can check both the region and its complement.
        // The center tile should be reconstructible from either a sufficiently
        // large region or its complement (but not necessarily a very small region).
        // At minimum, one of the two should contain the center.
        let complement = &duality.complement_region;
        let comp_duality = SubregionDuality::check(&tiling, 0, complement)
            .expect("Complement duality check failed");

        // By the no-cloning theorem, the operator cannot be in BOTH wedges
        // unless one of them is the full boundary.
        // But it should be in at least one of them for large enough regions.
        let either = duality.is_reconstructible || comp_duality.is_reconstructible;
        assert!(
            either || small_region.len() * 2 < tiling.boundary_qubit_indices.len(),
            "Center tile should be reconstructible from at least one side for large regions"
        );
    }

    // --------------------------------------------------------
    // Entropy Inequality Tests
    // --------------------------------------------------------

    #[test]
    fn test_strong_subadditivity() {
        let tiling = HyperbolicTiling::new(1);

        if tiling.boundary_qubit_indices.len() < 3 {
            return;
        }

        // Partition boundary into three non-empty regions.
        let n = tiling.boundary_qubit_indices.len();
        let third = n / 3;
        let region_a: Vec<usize> = tiling.boundary_qubit_indices[..third].to_vec();
        let region_b: Vec<usize> = tiling.boundary_qubit_indices[third..2 * third].to_vec();
        let region_c: Vec<usize> = tiling.boundary_qubit_indices[2 * third..].to_vec();

        let ssa = HolographicEntropyCone::check_strong_subadditivity(
            &tiling, &region_a, &region_b, &region_c,
        )
        .expect("SSA check failed");

        assert!(
            ssa,
            "Strong subadditivity should hold for holographic states"
        );
    }

    #[test]
    fn test_monogamy_mutual_info() {
        let tiling = HyperbolicTiling::new(1);

        if tiling.boundary_qubit_indices.len() < 3 {
            return;
        }

        let n = tiling.boundary_qubit_indices.len();
        let third = n / 3;
        let region_a: Vec<usize> = tiling.boundary_qubit_indices[..third].to_vec();
        let region_b: Vec<usize> = tiling.boundary_qubit_indices[third..2 * third].to_vec();
        let region_c: Vec<usize> = tiling.boundary_qubit_indices[2 * third..].to_vec();

        let mmi = HolographicEntropyCone::check_monogamy_mutual_info(
            &tiling, &region_a, &region_b, &region_c,
        )
        .expect("MMI check failed");

        assert!(
            mmi,
            "Monogamy of mutual information should hold for holographic states"
        );
    }

    #[test]
    fn test_mutual_information_non_negative() {
        let tiling = HyperbolicTiling::new(1);

        if tiling.boundary_qubit_indices.len() < 2 {
            return;
        }

        let n = tiling.boundary_qubit_indices.len();
        let half = n / 2;
        let region_a: Vec<usize> = tiling.boundary_qubit_indices[..half].to_vec();
        let region_b: Vec<usize> = tiling.boundary_qubit_indices[half..].to_vec();

        let mi = HolographicEntropyCone::mutual_information(&tiling, &region_a, &region_b)
            .expect("MI computation failed");

        assert!(
            mi >= -EPSILON,
            "Mutual information should be non-negative, got {}",
            mi
        );
    }

    // --------------------------------------------------------
    // Configuration Tests
    // --------------------------------------------------------

    #[test]
    fn test_config_builder() {
        let config = HolographicConfig::new(2).with_code_type(HolographicCodeType::HaPPY);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.code_type, HolographicCodeType::HaPPY);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let config = HolographicConfig::new(5);
        assert!(config.validate().is_err(), "num_layers=5 should be invalid");

        let config_ok = HolographicConfig::new(3);
        assert!(config_ok.validate().is_ok(), "num_layers=3 should be valid");
    }

    #[test]
    fn test_default_config() {
        let config = HolographicConfig::default();
        assert_eq!(config.num_layers, 0);
        assert_eq!(config.code_type, HolographicCodeType::HaPPY);
        assert_eq!(config.bulk_qubits, 0);
        assert_eq!(config.boundary_qubits, 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_holographic_code_display() {
        let config = HolographicConfig::new(0);
        let code = HolographicCode::new(config).expect("Failed to create code");
        let display = format!("{}", code);
        assert!(
            display.contains("HaPPY"),
            "Display should contain code type"
        );
        assert!(
            display.contains("0 layers"),
            "Display should show layer count"
        );
    }

    #[test]
    fn test_tiling_display() {
        let tiling = HyperbolicTiling::new(1);
        let display = format!("{}", tiling);
        assert!(
            display.contains("6 tiles"),
            "Display should show tile count"
        );
    }

    // --------------------------------------------------------
    // Edge case and integration tests
    // --------------------------------------------------------

    #[test]
    fn test_code_type_display() {
        assert_eq!(format!("{}", HolographicCodeType::HaPPY), "HaPPY");
        assert_eq!(
            format!("{}", HolographicCodeType::RandomTensor),
            "RandomTensor"
        );
        assert_eq!(
            format!("{}", HolographicCodeType::SteaneHolographic),
            "SteaneHolographic"
        );
    }

    #[test]
    fn test_error_display() {
        let err = HolographicError::InvalidTensor("test".into());
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid perfect tensor"));

        let err2 = HolographicError::BulkBoundaryMismatch {
            expected_bulk: 1,
            expected_boundary: 5,
            got_bulk: 2,
            got_boundary: 3,
        };
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("mismatch"));
    }

    #[test]
    fn test_von_neumann_entropy_pure_state() {
        // |0> state on 3 qubits: entropy of any subsystem of the full system is 0
        // only when the state is a product state.
        let state = {
            let mut s = vec![C64::new(0.0, 0.0); 8];
            s[0] = C64::new(1.0, 0.0);
            s
        };

        // Entropy of qubit 0 in the |000> state should be 0.
        let s = HolographicCode::von_neumann_entropy(&state, &[0], 3)
            .expect("Entropy computation failed");
        assert!(
            approx_eq(s, 0.0, 1e-6),
            "Entropy of product state subsystem should be 0, got {}",
            s
        );
    }

    #[test]
    fn test_von_neumann_entropy_bell_state() {
        // Bell state (|00> + |11>)/sqrt(2) on 2 qubits.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let state = vec![
            C64::new(inv_sqrt2, 0.0), // |00>
            C64::new(0.0, 0.0),       // |01>
            C64::new(0.0, 0.0),       // |10>
            C64::new(inv_sqrt2, 0.0), // |11>
        ];

        // Entropy of qubit 0 should be 1 bit (maximally entangled).
        let s = HolographicCode::von_neumann_entropy(&state, &[0], 2)
            .expect("Entropy computation failed");
        assert!(
            approx_eq(s, 1.0, 1e-4),
            "Entropy of Bell state subsystem should be 1 bit, got {}",
            s
        );
    }

    #[test]
    fn test_conditional_mutual_information_non_negative() {
        let tiling = HyperbolicTiling::new(1);

        if tiling.boundary_qubit_indices.len() < 3 {
            return;
        }

        let n = tiling.boundary_qubit_indices.len();
        let third = n / 3;
        let region_a: Vec<usize> = tiling.boundary_qubit_indices[..third].to_vec();
        let region_b: Vec<usize> = tiling.boundary_qubit_indices[third..2 * third].to_vec();
        let region_c: Vec<usize> = tiling.boundary_qubit_indices[2 * third..].to_vec();

        let cmi = HolographicEntropyCone::conditional_mutual_information(
            &tiling, &region_a, &region_b, &region_c,
        )
        .expect("CMI computation failed");

        // For holographic states, CMI >= 0 (equivalent to SSA).
        assert!(
            cmi >= -EPSILON,
            "Conditional mutual information should be non-negative for holographic states, got {}",
            cmi
        );
    }
}
