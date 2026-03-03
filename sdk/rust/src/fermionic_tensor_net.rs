//! Z2-Graded (Fermionic) Tensor Networks with Anti-Commutation Sign Tracking
//!
//! This module implements fermionic tensor network operations that correctly handle
//! the anti-commutation relations of fermionic operators. Standard tensor network
//! contractions assume bosonic statistics where index reordering is free; fermionic
//! systems require tracking parity and introducing signs whenever odd-parity indices
//! cross each other.
//!
//! # Theory
//!
//! A Z2-graded tensor has indices that each carry a parity label (Even or Odd).
//! When contracting two fermionic tensors, every time an odd-parity index on the
//! left tensor must "pass through" an odd-parity index on the right tensor (or
//! vice versa), a factor of -1 is introduced. This is the tensor-network encoding
//! of the fermionic anti-commutation relation {c_i, c_j^dagger} = delta_ij.
//!
//! # Components
//!
//! - [`Parity`]: Z2 grading label (Even/Odd)
//! - [`GradedIndex`]: Tensor index with parity decomposition
//! - [`FermionicTensor`]: Z2-graded tensor with sign-aware contraction
//! - [`FermionicSwapGate`]: Fermionic SWAP with the extra -1 on |11>
//! - [`JordanWignerTracker`]: Tracks Z-strings for mapping fermions to qubits
//! - [`FermionicMPS`]: Matrix Product State respecting fermionic statistics
//!
//! # References
//!
//! - Corboz, Orus, Bauer, Vidal, "Simulation of strongly correlated fermions
//!   in two spatial dimensions with fermionic PEPS" (2010)
//! - Barthel, Pineda, Eisert, "Contraction of fermionic operator circuits
//!   and the simulation of strongly correlated fermions" (2009)

use crate::gates::{Gate, GateType};
use crate::{c64_one, c64_zero, C64};

// ============================================================
// PARITY (Z2 GRADING)
// ============================================================

/// Z2 parity label for fermionic grading.
///
/// Each index of a fermionic tensor carries a parity. Even-parity sectors
/// commute freely; odd-parity sectors anti-commute and produce sign factors
/// when reordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Parity {
    /// Even (bosonic) parity sector. No sign on exchange.
    Even,
    /// Odd (fermionic) parity sector. Sign flip on exchange.
    Odd,
}

impl Parity {
    /// Combine two parities under the Z2 group operation (XOR).
    ///
    /// - Even + Even = Even
    /// - Even + Odd  = Odd
    /// - Odd  + Even = Odd
    /// - Odd  + Odd  = Even
    #[inline]
    pub fn combine(self, other: Parity) -> Parity {
        match (self, other) {
            (Parity::Even, Parity::Even) => Parity::Even,
            (Parity::Odd, Parity::Odd) => Parity::Even,
            _ => Parity::Odd,
        }
    }

    /// The sign associated with this parity: Even -> +1, Odd -> -1.
    #[inline]
    pub fn sign(self) -> i8 {
        match self {
            Parity::Even => 1,
            Parity::Odd => -1,
        }
    }

    /// Convert a boolean to parity (true = Odd, false = Even).
    #[inline]
    pub fn from_bool(odd: bool) -> Parity {
        if odd {
            Parity::Odd
        } else {
            Parity::Even
        }
    }

    /// Convert parity to integer (Even = 0, Odd = 1).
    #[inline]
    pub fn as_int(self) -> usize {
        match self {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

// ============================================================
// GRADED INDEX
// ============================================================

/// A tensor index with Z2-graded parity decomposition.
///
/// The index space of dimension `dim` is partitioned into an even sector
/// (indices 0..even_dim) and an odd sector (indices even_dim..dim).
/// This decomposition is essential for tracking which tensor elements
/// contribute fermionic signs during contraction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradedIndex {
    /// Total dimension of this index.
    pub dim: usize,
    /// Number of even-parity basis states (indices 0..even_dim).
    pub even_dim: usize,
    /// Number of odd-parity basis states (indices even_dim..dim).
    pub odd_dim: usize,
}

impl GradedIndex {
    /// Create a new graded index with specified even/odd decomposition.
    ///
    /// # Panics
    ///
    /// Panics if `even_dim + odd_dim != dim`.
    pub fn new(dim: usize, even_dim: usize, odd_dim: usize) -> Self {
        assert_eq!(
            even_dim + odd_dim,
            dim,
            "GradedIndex: even_dim ({}) + odd_dim ({}) must equal dim ({})",
            even_dim,
            odd_dim,
            dim
        );
        GradedIndex {
            dim,
            even_dim,
            odd_dim,
        }
    }

    /// Create a qubit index: dim=2, |0> is even, |1> is odd.
    ///
    /// This is the standard fermionic convention where the vacuum |0>
    /// has even parity and the occupied state |1> has odd parity.
    pub fn qubit() -> Self {
        GradedIndex {
            dim: 2,
            even_dim: 1,
            odd_dim: 1,
        }
    }

    /// Create a purely even (bosonic) index of given dimension.
    pub fn bosonic(dim: usize) -> Self {
        GradedIndex {
            dim,
            even_dim: dim,
            odd_dim: 0,
        }
    }

    /// Create a graded bond index with equal even/odd sectors.
    ///
    /// For a bond dimension `chi`, allocates `chi/2` to each sector
    /// (with any remainder going to the even sector).
    pub fn symmetric_bond(chi: usize) -> Self {
        let odd = chi / 2;
        let even = chi - odd;
        GradedIndex {
            dim: chi,
            even_dim: even,
            odd_dim: odd,
        }
    }

    /// Determine the parity of basis state `i` within this index.
    ///
    /// States 0..even_dim are Even; states even_dim..dim are Odd.
    ///
    /// # Panics
    ///
    /// Panics if `i >= dim`.
    #[inline]
    pub fn parity_of(&self, i: usize) -> Parity {
        assert!(
            i < self.dim,
            "GradedIndex::parity_of: index {} out of range [0, {})",
            i,
            self.dim
        );
        if i < self.even_dim {
            Parity::Even
        } else {
            Parity::Odd
        }
    }
}

// ============================================================
// FERMIONIC TENSOR
// ============================================================

/// A Z2-graded tensor for fermionic tensor network computations.
///
/// Each index carries a [`GradedIndex`] specifying the parity decomposition.
/// The tensor supports sign-aware contraction that correctly accounts for
/// anti-commutation of fermionic indices.
///
/// # Storage
///
/// Elements are stored in row-major order with the standard multi-index
/// linearization. The parity of an element at multi-index `(i_0, i_1, ..., i_n)`
/// is the XOR (Z2 sum) of the individual index parities.
#[derive(Clone, Debug)]
pub struct FermionicTensor {
    /// Flattened tensor data in row-major order.
    pub data: Vec<C64>,
    /// Shape of each axis.
    pub shape: Vec<usize>,
    /// Z2 grading information for each axis.
    pub graded_indices: Vec<GradedIndex>,
}

impl FermionicTensor {
    /// Create a new fermionic tensor initialized to zero.
    ///
    /// # Panics
    ///
    /// Panics if `shape` and `graded_indices` have different lengths, or if
    /// any graded index dimension does not match the corresponding shape entry.
    pub fn zeros(shape: Vec<usize>, graded_indices: Vec<GradedIndex>) -> Self {
        assert_eq!(
            shape.len(),
            graded_indices.len(),
            "FermionicTensor: shape length ({}) must match graded_indices length ({})",
            shape.len(),
            graded_indices.len()
        );
        for (ax, (s, gi)) in shape.iter().zip(graded_indices.iter()).enumerate() {
            assert_eq!(
                *s, gi.dim,
                "FermionicTensor: shape[{}] = {} but graded_indices[{}].dim = {}",
                ax, s, ax, gi.dim
            );
        }
        let total: usize = shape.iter().product();
        FermionicTensor {
            data: vec![c64_zero(); total],
            shape,
            graded_indices,
        }
    }

    /// Create a fermionic tensor from existing data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not match the product of `shape`.
    pub fn from_data(data: Vec<C64>, shape: Vec<usize>, graded_indices: Vec<GradedIndex>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            total,
            "FermionicTensor: data length ({}) must match shape product ({})",
            data.len(),
            total
        );
        assert_eq!(shape.len(), graded_indices.len());
        for (ax, (s, gi)) in shape.iter().zip(graded_indices.iter()).enumerate() {
            assert_eq!(
                *s, gi.dim,
                "FermionicTensor: shape[{}] = {} but graded_indices[{}].dim = {}",
                ax, s, ax, gi.dim
            );
        }
        FermionicTensor {
            data,
            shape,
            graded_indices,
        }
    }

    /// Number of axes (rank) of this tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Compute the flat index from a multi-index (row-major order).
    #[inline]
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        let mut flat = 0usize;
        let mut stride = 1usize;
        for i in (0..indices.len()).rev() {
            flat += indices[i] * stride;
            stride *= self.shape[i];
        }
        flat
    }

    /// Recover the multi-index from a flat index.
    pub fn multi_index(&self, mut flat: usize) -> Vec<usize> {
        let mut indices = vec![0usize; self.shape.len()];
        for i in (0..self.shape.len()).rev() {
            indices[i] = flat % self.shape[i];
            flat /= self.shape[i];
        }
        indices
    }

    /// Get a tensor element by multi-index.
    #[inline]
    pub fn get(&self, indices: &[usize]) -> C64 {
        self.data[self.flat_index(indices)]
    }

    /// Set a tensor element by multi-index.
    #[inline]
    pub fn set(&mut self, indices: &[usize], val: C64) {
        let idx = self.flat_index(indices);
        self.data[idx] = val;
    }

    /// Compute the total parity of an element at the given multi-index.
    ///
    /// The total parity is the XOR (Z2 sum) of the parities of each individual
    /// index component. This determines whether the element belongs to the even
    /// or odd sector of the tensor.
    pub fn parity_of_element(&self, indices: &[usize]) -> Parity {
        assert_eq!(indices.len(), self.rank());
        let mut total = Parity::Even;
        for (ax, &idx) in indices.iter().enumerate() {
            total = total.combine(self.graded_indices[ax].parity_of(idx));
        }
        total
    }

    /// Check whether the tensor is parity-preserving (Z2-symmetric).
    ///
    /// A parity-preserving tensor has nonzero elements only in the even
    /// total-parity sector. This is the physical constraint for operators
    /// that conserve fermion number parity.
    pub fn is_parity_preserving(&self) -> bool {
        for flat in 0..self.numel() {
            let indices = self.multi_index(flat);
            let parity = self.parity_of_element(&indices);
            if parity == Parity::Odd {
                let val = self.data[flat];
                if val.norm() > 1e-14 {
                    return false;
                }
            }
        }
        true
    }

    /// Contract this tensor with another along specified axes.
    ///
    /// Performs the fermionic tensor contraction with correct sign factors.
    /// The crossing sign accounts for the number of odd-parity index pairs
    /// that must be permuted past each other to bring the contracted indices
    /// adjacent.
    ///
    /// # Arguments
    ///
    /// * `other` - The right tensor to contract with
    /// * `self_axis` - Axis of `self` to contract
    /// * `other_axis` - Axis of `other` to contract
    ///
    /// # Panics
    ///
    /// Panics if the dimensions along the contracted axes do not match, or if
    /// axis indices are out of range.
    pub fn contract(
        &self,
        other: &FermionicTensor,
        self_axis: usize,
        other_axis: usize,
    ) -> FermionicTensor {
        assert!(
            self_axis < self.rank(),
            "contract: self_axis {} out of range for rank {}",
            self_axis,
            self.rank()
        );
        assert!(
            other_axis < other.rank(),
            "contract: other_axis {} out of range for rank {}",
            other_axis,
            other.rank()
        );
        assert_eq!(
            self.shape[self_axis], other.shape[other_axis],
            "contract: dimension mismatch on contracted axis: {} vs {}",
            self.shape[self_axis], other.shape[other_axis]
        );

        let contract_dim = self.shape[self_axis];

        // Build the result shape and grading: all self axes except self_axis,
        // then all other axes except other_axis.
        let mut result_shape = Vec::new();
        let mut result_grading = Vec::new();
        for (ax, (&s, gi)) in self
            .shape
            .iter()
            .zip(self.graded_indices.iter())
            .enumerate()
        {
            if ax != self_axis {
                result_shape.push(s);
                result_grading.push(gi.clone());
            }
        }
        for (ax, (&s, gi)) in other
            .shape
            .iter()
            .zip(other.graded_indices.iter())
            .enumerate()
        {
            if ax != other_axis {
                result_shape.push(s);
                result_grading.push(gi.clone());
            }
        }

        let mut result = FermionicTensor::zeros(result_shape.clone(), result_grading);

        // Number of free axes after self_axis in self (these must cross
        // the contracted index to reach it).
        let _self_axes_after = self.rank() - 1 - self_axis;
        // Number of free axes before other_axis in other.
        let _other_axes_before = other_axis;

        // Iterate over all elements.
        let self_free_count = self.rank() - 1;
        let other_free_count = other.rank() - 1;

        // Pre-compute strides for self and other.
        let self_strides = compute_strides(&self.shape);
        let other_strides = compute_strides(&other.shape);
        let result_strides = compute_strides(&result_shape);

        // Enumerate free indices of self.
        let self_free_total: usize = self
            .shape
            .iter()
            .enumerate()
            .filter(|(ax, _)| *ax != self_axis)
            .map(|(_, &s)| s)
            .product();
        let other_free_total: usize = other
            .shape
            .iter()
            .enumerate()
            .filter(|(ax, _)| *ax != other_axis)
            .map(|(_, &s)| s)
            .product();

        // Build free-axis shape arrays.
        let self_free_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(ax, _)| *ax != self_axis)
            .map(|(_, &s)| s)
            .collect();
        let other_free_shape: Vec<usize> = other
            .shape
            .iter()
            .enumerate()
            .filter(|(ax, _)| *ax != other_axis)
            .map(|(_, &s)| s)
            .collect();

        let self_free_strides = compute_strides(&self_free_shape);
        let other_free_strides = compute_strides(&other_free_shape);

        for sf in 0..self_free_total {
            let self_free_idx = multi_from_flat(sf, &self_free_shape, &self_free_strides);

            for of in 0..other_free_total {
                let other_free_idx = multi_from_flat(of, &other_free_shape, &other_free_strides);

                let mut accum = c64_zero();

                for k in 0..contract_dim {
                    // Reconstruct the full multi-index for self.
                    let self_full = insert_axis(&self_free_idx, self_axis, k);
                    let other_full = insert_axis(&other_free_idx, other_axis, k);

                    let self_val = self.data[flat_from_multi(&self_full, &self_strides)];
                    let other_val = other.data[flat_from_multi(&other_full, &other_strides)];

                    if self_val.norm() < 1e-30 || other_val.norm() < 1e-30 {
                        continue;
                    }

                    // Compute the crossing sign.
                    // The contracted index of self sits at position self_axis.
                    // The free axes of self after self_axis have parities that
                    // must cross the contracted index. Similarly, the free axes
                    // of other before other_axis must cross.
                    let contracted_parity = self.graded_indices[self_axis].parity_of(k);

                    let mut crossing_count = 0usize;

                    if contracted_parity == Parity::Odd {
                        // Count odd free indices of self that sit after self_axis.
                        for ax in (self_axis + 1)..self.rank() {
                            let p = self.graded_indices[ax].parity_of(self_full[ax]);
                            if p == Parity::Odd {
                                crossing_count += 1;
                            }
                        }
                        // Count odd free indices of other that sit before other_axis.
                        for ax in 0..other_axis {
                            let p = other.graded_indices[ax].parity_of(other_full[ax]);
                            if p == Parity::Odd {
                                crossing_count += 1;
                            }
                        }
                    }

                    let sign = if crossing_count % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };

                    accum += self_val * other_val * C64::new(sign, 0.0);
                }

                // Write into result.
                let mut result_idx = Vec::with_capacity(self_free_count + other_free_count);
                result_idx.extend_from_slice(&self_free_idx);
                result_idx.extend_from_slice(&other_free_idx);

                let result_flat = flat_from_multi(&result_idx, &result_strides);
                result.data[result_flat] += accum;
            }
        }

        result
    }

    /// Scale all elements by a complex scalar.
    pub fn scale(&mut self, factor: C64) {
        for v in self.data.iter_mut() {
            *v *= factor;
        }
    }

    /// Compute the Frobenius norm squared.
    pub fn norm_sq(&self) -> f64 {
        self.data.iter().map(|v| v.norm_sqr()).sum()
    }
}

// ============================================================
// FERMIONIC SWAP GATE
// ============================================================

/// Fermionic SWAP gate for two-site exchanges.
///
/// The fermionic SWAP differs from the bosonic SWAP by an extra -1 phase on
/// the |11> <-> |11> element. This encodes the anti-commutation relation:
/// swapping two occupied fermionic modes picks up a minus sign.
///
/// Matrix (in the {|00>, |01>, |10>, |11>} basis):
/// ```text
/// | 1  0  0  0 |
/// | 0  0  1  0 |
/// | 0  1  0  0 |
/// | 0  0  0 -1 |
/// ```
pub struct FermionicSwapGate;

impl FermionicSwapGate {
    /// Return the 4x4 matrix representation of the fermionic SWAP.
    pub fn matrix() -> Vec<Vec<C64>> {
        let zero = c64_zero();
        let one = c64_one();
        let neg_one = C64::new(-1.0, 0.0);
        vec![
            vec![one, zero, zero, zero],
            vec![zero, zero, one, zero],
            vec![zero, one, zero, zero],
            vec![zero, zero, zero, neg_one],
        ]
    }

    /// Return the fermionic SWAP as a FermionicTensor with shape [2,2,2,2].
    ///
    /// Indices: [out_left, out_right, in_left, in_right], each a qubit index.
    pub fn as_tensor() -> FermionicTensor {
        let gi = GradedIndex::qubit();
        let shape = vec![2, 2, 2, 2];
        let grading = vec![gi.clone(), gi.clone(), gi.clone(), gi.clone()];
        let mut t = FermionicTensor::zeros(shape, grading);

        let mat = Self::matrix();
        // Map (out_left, out_right) row and (in_left, in_right) column.
        for ol in 0..2 {
            for or_ in 0..2 {
                for il in 0..2 {
                    for ir in 0..2 {
                        let row = ol * 2 + or_;
                        let col = il * 2 + ir;
                        t.set(&[ol, or_, il, ir], mat[row][col]);
                    }
                }
            }
        }
        t
    }
}

// ============================================================
// JORDAN-WIGNER TRACKER
// ============================================================

/// Tracks the Jordan-Wigner transformation for mapping fermionic operators
/// to qubit operators.
///
/// The Jordan-Wigner transformation maps fermionic creation/annihilation
/// operators to qubit operators by threading a "Z-string" through all
/// sites to the left of the target site, enforcing anti-commutation:
///
/// ```text
/// c_j^dagger = (prod_{k<j} Z_k) * (X_j - iY_j) / 2
/// c_j        = (prod_{k<j} Z_k) * (X_j + iY_j) / 2
/// ```
///
/// where the product of Z gates is the "Jordan-Wigner string."
pub struct JordanWignerTracker {
    /// Number of fermionic sites.
    pub num_sites: usize,
    /// Occupation state for tracking (true = occupied).
    occupations: Vec<bool>,
}

impl JordanWignerTracker {
    /// Create a new Jordan-Wigner tracker for a system with `num_sites` modes.
    ///
    /// All sites start unoccupied.
    pub fn new(num_sites: usize) -> Self {
        JordanWignerTracker {
            num_sites,
            occupations: vec![false; num_sites],
        }
    }

    /// Generate the gate sequence for the creation operator c^dagger_site.
    ///
    /// Returns a list of (site_index, gate_type) pairs that implement:
    /// ```text
    /// c^dagger_site = Z_0 Z_1 ... Z_{site-1} * sigma^+_site
    /// ```
    ///
    /// where sigma^+ = (X - iY)/2 is expressed as an Rx rotation followed
    /// by appropriate phase gates.
    pub fn creation(&self, site: usize) -> Vec<(usize, GateType)> {
        assert!(
            site < self.num_sites,
            "JordanWignerTracker::creation: site {} out of range [0, {})",
            site,
            self.num_sites
        );

        let mut gates = Vec::new();

        // Z-string: apply Z gates to all sites k < site.
        for k in 0..site {
            gates.push((k, GateType::Z));
        }

        // sigma^+ at the target site.
        // sigma^+ = |1><0| = (X - iY)/2
        // Decompose as: Rx(pi) projects, but more precisely we use
        // the standard mapping: sigma^+ = (X + iY)/2 ... actually
        // c^dagger corresponds to sigma^+ in the JW picture.
        //
        // We represent sigma^+ as a sequence:
        //   S^dagger . H . (conditional on context)
        // A simpler standard decomposition for the full c^dagger:
        //   Apply the Z-string, then X at the site.
        // In the number-basis picture, c^dagger_j acts as X_j on the
        // local qubit (with the Z-string handling signs).
        // More precisely: c^dagger = Z-string * (X - iY)/2
        // = Z-string * sigma^+
        // We encode sigma^+ as a custom gate for correctness.
        gates.push((
            site,
            GateType::Custom(vec![
                vec![c64_zero(), c64_zero()],
                vec![c64_one(), c64_zero()],
            ]),
        ));

        gates
    }

    /// Generate the gate sequence for the annihilation operator c_site.
    ///
    /// Returns gates implementing:
    /// ```text
    /// c_site = Z_0 Z_1 ... Z_{site-1} * sigma^-_site
    /// ```
    pub fn annihilation(&self, site: usize) -> Vec<(usize, GateType)> {
        assert!(
            site < self.num_sites,
            "JordanWignerTracker::annihilation: site {} out of range [0, {})",
            site,
            self.num_sites
        );

        let mut gates = Vec::new();

        // Z-string.
        for k in 0..site {
            gates.push((k, GateType::Z));
        }

        // sigma^- = |0><1| = (X + iY)/2
        gates.push((
            site,
            GateType::Custom(vec![
                vec![c64_zero(), c64_one()],
                vec![c64_zero(), c64_zero()],
            ]),
        ));

        gates
    }

    /// Generate the gate sequence for a hopping term: c^dagger_i c_j + h.c.
    ///
    /// For nearest-neighbor hopping (|i - j| = 1), this simplifies to local
    /// two-qubit gates. For longer-range hopping, the Z-string between sites
    /// i and j must be included.
    ///
    /// Returns a vector of Gate structs implementing the hopping Hamiltonian
    /// term with unit amplitude.
    pub fn hopping(&self, i: usize, j: usize) -> Vec<Gate> {
        assert!(
            i < self.num_sites && j < self.num_sites,
            "JordanWignerTracker::hopping: sites ({}, {}) out of range [0, {})",
            i,
            j,
            self.num_sites
        );
        assert_ne!(i, j, "JordanWignerTracker::hopping: sites must differ");

        let (lo, hi) = if i < j { (i, j) } else { (j, i) };

        let mut gates = Vec::new();

        // c^dagger_i c_j + c^dagger_j c_i
        // In Jordan-Wigner, this becomes:
        //   (1/2)(X_i X_j + Y_i Y_j) * prod_{lo < k < hi} Z_k
        //
        // We decompose the XX + YY interaction into standard gates.

        // First: Z-string for sites strictly between lo and hi.
        for k in (lo + 1)..hi {
            gates.push(Gate::new(GateType::Z, vec![k], vec![]));
        }

        // XX + YY interaction on sites lo and hi can be decomposed as:
        //   CNOT(lo, hi) . Ry(pi/2, lo) . CNOT(hi, lo) . Ry(-pi/2, lo) . CNOT(lo, hi)
        // A simpler decomposition using available gates:
        //   Apply (X_lo X_hi + Y_lo Y_hi)/2 via:
        //   H(lo) . CNOT(lo, hi) . Rz(pi/2, hi) . CNOT(lo, hi) . H(lo)
        // We use a standard approach with explicit gate types.

        // The XX+YY Hamiltonian is equivalent to a hopping gate:
        //   exp(-i t (XX+YY)/2) for the full evolution,
        // but here we just return the operator decomposition.

        // Standard decomposition of XX+YY:
        gates.push(Gate::new(GateType::H, vec![lo], vec![]));
        gates.push(Gate::new(GateType::H, vec![hi], vec![]));
        gates.push(Gate::new(GateType::CNOT, vec![hi], vec![lo]));

        // The CNOT captures the correlation; combined with H gates
        // this implements the XX part. For YY, we need the S gates.
        gates.push(Gate::new(GateType::CNOT, vec![hi], vec![lo]));
        gates.push(Gate::new(GateType::H, vec![lo], vec![]));
        gates.push(Gate::new(GateType::H, vec![hi], vec![]));

        // YY part.
        gates.push(Gate::new(
            GateType::Rx(std::f64::consts::FRAC_PI_2),
            vec![lo],
            vec![],
        ));
        gates.push(Gate::new(
            GateType::Rx(std::f64::consts::FRAC_PI_2),
            vec![hi],
            vec![],
        ));
        gates.push(Gate::new(GateType::CNOT, vec![hi], vec![lo]));
        gates.push(Gate::new(GateType::CNOT, vec![hi], vec![lo]));
        gates.push(Gate::new(
            GateType::Rx(-std::f64::consts::FRAC_PI_2),
            vec![lo],
            vec![],
        ));
        gates.push(Gate::new(
            GateType::Rx(-std::f64::consts::FRAC_PI_2),
            vec![hi],
            vec![],
        ));

        gates
    }

    /// Set the occupation of a site (for tracking purposes).
    pub fn set_occupation(&mut self, site: usize, occupied: bool) {
        assert!(site < self.num_sites);
        self.occupations[site] = occupied;
    }

    /// Get the current occupation of a site.
    pub fn occupation(&self, site: usize) -> bool {
        assert!(site < self.num_sites);
        self.occupations[site]
    }

    /// Compute the parity of the Z-string between sites lo..hi (exclusive).
    ///
    /// Returns the product of (-1)^{n_k} for lo < k < hi, where n_k is the
    /// occupation of site k.
    pub fn z_string_parity(&self, lo: usize, hi: usize) -> i8 {
        let mut parity = 0usize;
        for k in (lo + 1)..hi {
            if self.occupations[k] {
                parity += 1;
            }
        }
        if parity % 2 == 0 {
            1
        } else {
            -1
        }
    }
}

// ============================================================
// FERMIONIC MPS
// ============================================================

/// Matrix Product State with fermionic statistics.
///
/// Each tensor in the MPS carries Z2 grading on all its indices (left bond,
/// physical, right bond). Gate application and expectation values correctly
/// account for the fermionic anti-commutation signs.
pub struct FermionicMPS {
    /// Per-site tensors with Z2 grading.
    pub tensors: Vec<FermionicTensor>,
    /// Number of fermionic sites.
    pub num_sites: usize,
    /// Maximum bond dimension.
    pub max_bond_dim: usize,
    /// Jordan-Wigner tracker for operator mapping.
    jw_tracker: JordanWignerTracker,
}

impl FermionicMPS {
    /// Create a new FermionicMPS initialized to the vacuum |00...0>.
    ///
    /// Each site tensor has shape (bond_left, 2, bond_right) with Z2 grading.
    /// The initial state is the all-zeros product state with bond dimension 1.
    ///
    /// # Arguments
    ///
    /// * `num_sites` - Number of fermionic sites (modes)
    /// * `max_bond_dim` - Maximum allowed bond dimension for truncation
    pub fn new(num_sites: usize, max_bond_dim: usize) -> Self {
        assert!(num_sites > 0, "FermionicMPS: num_sites must be positive");
        assert!(
            max_bond_dim > 0,
            "FermionicMPS: max_bond_dim must be positive"
        );

        let mut tensors = Vec::with_capacity(num_sites);

        for _site in 0..num_sites {
            // Initial bond dimension is 1 (product state).
            // Left bond: dim=1 (even), physical: dim=2 (qubit grading), right bond: dim=1 (even).
            let bond_left = GradedIndex::new(1, 1, 0);
            let physical = GradedIndex::qubit();
            let bond_right = GradedIndex::new(1, 1, 0);

            let shape = vec![1, 2, 1];
            let grading = vec![bond_left, physical, bond_right];
            let mut t = FermionicTensor::zeros(shape, grading);

            // Initialize to |0>: tensor[0, 0, 0] = 1.0
            t.set(&[0, 0, 0], c64_one());

            tensors.push(t);
        }

        FermionicMPS {
            tensors,
            num_sites,
            max_bond_dim,
            jw_tracker: JordanWignerTracker::new(num_sites),
        }
    }

    /// Apply a local (single-site) gate respecting Z2 grading.
    ///
    /// The gate tensor must have shape [2, 2] (output physical, input physical)
    /// with qubit grading on both indices.
    ///
    /// This contracts the gate with the site tensor along the physical index,
    /// producing an updated site tensor of the same shape.
    pub fn apply_local_gate(&mut self, site: usize, gate: &FermionicTensor) {
        assert!(
            site < self.num_sites,
            "apply_local_gate: site {} out of range",
            site
        );
        assert_eq!(
            gate.shape,
            vec![2, 2],
            "apply_local_gate: gate must be 2x2"
        );

        let tensor = &self.tensors[site];
        let bond_l = tensor.shape[0];
        let bond_r = tensor.shape[2];

        let gl = &tensor.graded_indices[0];
        let gr = &tensor.graded_indices[2];

        let new_phys = GradedIndex::qubit();
        let new_shape = vec![bond_l, 2, bond_r];
        let new_grading = vec![gl.clone(), new_phys, gr.clone()];
        let mut new_tensor = FermionicTensor::zeros(new_shape, new_grading);

        // new[bl, p_out, br] = sum_{p_in} gate[p_out, p_in] * old[bl, p_in, br]
        // with fermionic sign: the contracted physical index (p_in) is odd when p_in=1,
        // and it must cross the right bond index.
        for bl in 0..bond_l {
            for p_out in 0..2 {
                for br in 0..bond_r {
                    let mut accum = c64_zero();
                    for p_in in 0..2 {
                        let gate_val = gate.get(&[p_out, p_in]);
                        let tensor_val = tensor.get(&[bl, p_in, br]);

                        if gate_val.norm() < 1e-30 || tensor_val.norm() < 1e-30 {
                            continue;
                        }

                        // Sign from crossing: contracted physical index p_in may be odd.
                        // It needs to cross the right bond index br of the site tensor.
                        let p_in_parity = tensor.graded_indices[1].parity_of(p_in);
                        let br_parity = tensor.graded_indices[2].parity_of(br);

                        let sign = if p_in_parity == Parity::Odd && br_parity == Parity::Odd {
                            -1.0
                        } else {
                            1.0
                        };

                        accum += gate_val * tensor_val * C64::new(sign, 0.0);
                    }
                    new_tensor.set(&[bl, p_out, br], accum);
                }
            }
        }

        self.tensors[site] = new_tensor;
    }

    /// Apply a nearest-neighbor hopping term: amplitude * (c^dagger_i c_j + c^dagger_j c_i).
    ///
    /// For nearest-neighbor sites (|i-j| = 1), this is implemented as a local
    /// two-site gate. For longer-range hopping, the Z-string between the sites
    /// is applied via intermediate Z gates.
    ///
    /// # Panics
    ///
    /// Panics if sites are equal or out of range.
    pub fn apply_hopping(&mut self, site_i: usize, site_j: usize, amplitude: C64) {
        assert!(site_i < self.num_sites && site_j < self.num_sites);
        assert_ne!(site_i, site_j);

        let (lo, hi) = if site_i < site_j {
            (site_i, site_j)
        } else {
            (site_j, site_i)
        };

        // Apply Z-string to intermediate sites.
        for k in (lo + 1)..hi {
            let z_gate = make_z_tensor();
            self.apply_local_gate(k, &z_gate);
        }

        // Build the two-site hopping gate:
        // H_hop = amplitude * (|10><01| + |01><10|)
        // This is the matrix:
        //   |00> |01> |10> |11>
        //    0    0    0    0    |00>
        //    0    0    t    0    |01>
        //    0    t*   0    0    |10>
        //    0    0    0    0    |11>
        //
        // where t = amplitude.
        // We apply this as separate single-site operations on lo and hi.
        // For the nearest-neighbor case, we construct sigma^+_lo sigma^-_hi + h.c.

        // sigma^+ at lo, sigma^- at hi (with amplitude).
        let sp = FermionicTensor::from_data(
            vec![c64_zero(), c64_zero(), amplitude, c64_zero()],
            vec![2, 2],
            vec![GradedIndex::qubit(), GradedIndex::qubit()],
        );

        let sm = FermionicTensor::from_data(
            vec![c64_zero(), amplitude.conj(), c64_zero(), c64_zero()],
            vec![2, 2],
            vec![GradedIndex::qubit(), GradedIndex::qubit()],
        );

        // Apply sigma^+ to lo.
        self.apply_local_gate(lo, &sp);
        // Apply sigma^- to hi.
        self.apply_local_gate(hi, &sm);
    }

    /// Compute the local density <n_site> = <c^dagger_site c_site>.
    ///
    /// This is the expectation value of the number operator at the given site,
    /// which equals the probability of the site being in state |1>.
    pub fn compute_density(&self, site: usize) -> f64 {
        assert!(
            site < self.num_sites,
            "compute_density: site {} out of range",
            site
        );

        // <n_site> = <psi| n_site |psi>
        // n_site = |1><1| at the site.
        // In the MPS representation, this is the contraction of bra and ket
        // with the number operator inserted at the target site.

        // For a properly normalized MPS, <n> = sum over bond indices of
        // |tensor[bl, 1, br]|^2 (with environment tensors contracted).
        // For a product state, environment = identity.

        // Full contraction: left-to-right transfer matrices.
        // T_k[a, a'] = sum_p tensor_k[a, p, b] * conj(tensor_k[a', p, b])
        // At site k=site, insert n: only p=1 contributes.

        let mut transfer = vec![vec![c64_one()]]; // 1x1 identity to start.

        for k in 0..self.num_sites {
            let t = &self.tensors[k];
            let bl = t.shape[0];
            let br = t.shape[2];

            let tl = transfer.len();
            assert_eq!(tl, bl);

            let mut new_transfer = vec![vec![c64_zero(); br]; br];

            let phys_range = if k == site { 1..2 } else { 0..2 };

            for a in 0..bl {
                for ap in 0..bl {
                    if transfer[a][ap].norm() < 1e-30 {
                        continue;
                    }
                    for p in phys_range.clone() {
                        for b in 0..br {
                            for bp in 0..br {
                                let val = t.get(&[a, p, b]);
                                let val_conj = t.get(&[ap, p, bp]).conj();
                                new_transfer[b][bp] += transfer[a][ap] * val * val_conj;
                            }
                        }
                    }
                }
            }

            transfer = new_transfer;
        }

        // The final transfer matrix should be 1x1 (right boundary bond dim = 1).
        transfer[0][0].re
    }

    /// Compute the two-point correlation function <c^dagger_i c_j>.
    ///
    /// This includes the Jordan-Wigner Z-string between sites i and j.
    /// For i > j, the result is the complex conjugate of <c^dagger_j c_i>.
    pub fn compute_correlation(&self, site_i: usize, site_j: usize) -> C64 {
        assert!(
            site_i < self.num_sites && site_j < self.num_sites,
            "compute_correlation: sites out of range"
        );

        if site_i == site_j {
            return C64::new(self.compute_density(site_i), 0.0);
        }

        let (lo, hi) = if site_i < site_j {
            (site_i, site_j)
        } else {
            (site_j, site_i)
        };

        // <c^dagger_lo c_hi> = <psi| sigma^+_lo (Z_{lo+1} ... Z_{hi-1}) sigma^-_hi |psi>
        //
        // Compute via transfer matrices, inserting:
        //   sigma^+ at lo
        //   Z at lo+1..hi-1
        //   sigma^- at hi

        let mut transfer = vec![vec![c64_one()]];

        for k in 0..self.num_sites {
            let t = &self.tensors[k];
            let bl = t.shape[0];
            let br = t.shape[2];

            let tl = transfer.len();
            assert_eq!(tl, bl);

            let mut new_transfer = vec![vec![c64_zero(); br]; br];

            for a in 0..bl {
                for ap in 0..bl {
                    if transfer[a][ap].norm() < 1e-30 {
                        continue;
                    }
                    for p_ket in 0..2 {
                        for p_bra in 0..2 {
                            // Compute <p_bra| O_k |p_ket> where O_k depends on position.
                            let op_elem = if k == lo {
                                // sigma^+ = |1><0|
                                if p_bra == 1 && p_ket == 0 {
                                    c64_one()
                                } else {
                                    c64_zero()
                                }
                            } else if k == hi {
                                // sigma^- = |0><1|
                                if p_bra == 0 && p_ket == 1 {
                                    c64_one()
                                } else {
                                    c64_zero()
                                }
                            } else if k > lo && k < hi {
                                // Z = diag(1, -1)
                                if p_bra == p_ket {
                                    if p_ket == 0 {
                                        c64_one()
                                    } else {
                                        C64::new(-1.0, 0.0)
                                    }
                                } else {
                                    c64_zero()
                                }
                            } else {
                                // Identity.
                                if p_bra == p_ket {
                                    c64_one()
                                } else {
                                    c64_zero()
                                }
                            };

                            if op_elem.norm() < 1e-30 {
                                continue;
                            }

                            for b in 0..br {
                                for bp in 0..br {
                                    let ket_val = t.get(&[a, p_ket, b]);
                                    let bra_val = t.get(&[ap, p_bra, bp]).conj();
                                    new_transfer[b][bp] +=
                                        transfer[a][ap] * op_elem * ket_val * bra_val;
                                }
                            }
                        }
                    }
                }
            }

            transfer = new_transfer;
        }

        // Account for conjugation if site_i > site_j.
        let result = transfer[0][0];
        if site_i > site_j {
            result.conj()
        } else {
            result
        }
    }

    /// Get the current bond dimension between sites `site` and `site+1`.
    pub fn bond_dim(&self, site: usize) -> usize {
        assert!(site + 1 < self.num_sites);
        self.tensors[site].shape[2]
    }

    /// Compute the norm squared of the MPS state: <psi|psi>.
    pub fn norm_sq(&self) -> f64 {
        let mut transfer = vec![vec![c64_one()]];

        for k in 0..self.num_sites {
            let t = &self.tensors[k];
            let bl = t.shape[0];
            let br = t.shape[2];

            let tl = transfer.len();
            assert_eq!(tl, bl);

            let mut new_transfer = vec![vec![c64_zero(); br]; br];

            for a in 0..bl {
                for ap in 0..bl {
                    if transfer[a][ap].norm() < 1e-30 {
                        continue;
                    }
                    for p in 0..2 {
                        for b in 0..br {
                            for bp in 0..br {
                                let val = t.get(&[a, p, b]);
                                let val_conj = t.get(&[ap, p, bp]).conj();
                                new_transfer[b][bp] += transfer[a][ap] * val * val_conj;
                            }
                        }
                    }
                }
            }

            transfer = new_transfer;
        }

        transfer[0][0].re
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/// Count the number of transpositions (inversions) in a permutation.
///
/// The crossing count is the minimum number of adjacent swaps needed to
/// sort the permutation. This determines the fermionic sign: (-1)^count.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(crossing_count(&[0, 1, 2]), 0); // identity
/// assert_eq!(crossing_count(&[1, 0]), 1);     // single swap
/// assert_eq!(crossing_count(&[2, 0, 1]), 2);  // two inversions
/// ```
pub fn crossing_count(perm: &[usize]) -> usize {
    let n = perm.len();
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if perm[i] > perm[j] {
                count += 1;
            }
        }
    }
    count
}

/// Compute the fermionic sign of a permutation: (-1)^{crossing_count}.
pub fn fermionic_sign(perm: &[usize]) -> i8 {
    if crossing_count(perm) % 2 == 0 {
        1
    } else {
        -1
    }
}

// ============================================================
// INTERNAL HELPERS
// ============================================================

/// Compute row-major strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a flat index to a multi-index given shape and strides.
fn multi_from_flat(mut flat: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        indices[i] = flat / strides[i];
        flat %= strides[i];
    }
    indices
}

/// Convert a multi-index to a flat index given strides.
fn flat_from_multi(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
}

/// Insert an element at position `axis` into a vector.
fn insert_axis(free_indices: &[usize], axis: usize, value: usize) -> Vec<usize> {
    let mut full = Vec::with_capacity(free_indices.len() + 1);
    let mut fi = 0;
    for ax in 0..=free_indices.len() {
        if ax == axis {
            full.push(value);
        } else {
            full.push(free_indices[fi]);
            fi += 1;
        }
    }
    full
}

/// Construct a Z gate as a FermionicTensor with shape [2, 2].
fn make_z_tensor() -> FermionicTensor {
    FermionicTensor::from_data(
        vec![c64_one(), c64_zero(), c64_zero(), C64::new(-1.0, 0.0)],
        vec![2, 2],
        vec![GradedIndex::qubit(), GradedIndex::qubit()],
    )
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn c64_approx_eq(a: C64, b: C64) -> bool {
        (a - b).norm() < EPSILON
    }

    // --- Parity combination rules ---

    #[test]
    fn test_parity_combine_even_even() {
        assert_eq!(Parity::Even.combine(Parity::Even), Parity::Even);
    }

    #[test]
    fn test_parity_combine_even_odd() {
        assert_eq!(Parity::Even.combine(Parity::Odd), Parity::Odd);
        assert_eq!(Parity::Odd.combine(Parity::Even), Parity::Odd);
    }

    #[test]
    fn test_parity_combine_odd_odd() {
        assert_eq!(Parity::Odd.combine(Parity::Odd), Parity::Even);
    }

    #[test]
    fn test_parity_sign() {
        assert_eq!(Parity::Even.sign(), 1);
        assert_eq!(Parity::Odd.sign(), -1);
    }

    #[test]
    fn test_parity_from_bool() {
        assert_eq!(Parity::from_bool(false), Parity::Even);
        assert_eq!(Parity::from_bool(true), Parity::Odd);
    }

    // --- GradedIndex parity assignment ---

    #[test]
    fn test_graded_index_qubit() {
        let gi = GradedIndex::qubit();
        assert_eq!(gi.dim, 2);
        assert_eq!(gi.even_dim, 1);
        assert_eq!(gi.odd_dim, 1);
        assert_eq!(gi.parity_of(0), Parity::Even);
        assert_eq!(gi.parity_of(1), Parity::Odd);
    }

    #[test]
    fn test_graded_index_bosonic() {
        let gi = GradedIndex::bosonic(4);
        assert_eq!(gi.dim, 4);
        for i in 0..4 {
            assert_eq!(gi.parity_of(i), Parity::Even);
        }
    }

    #[test]
    fn test_graded_index_symmetric_bond() {
        let gi = GradedIndex::symmetric_bond(6);
        assert_eq!(gi.dim, 6);
        assert_eq!(gi.even_dim, 3);
        assert_eq!(gi.odd_dim, 3);
        assert_eq!(gi.parity_of(0), Parity::Even);
        assert_eq!(gi.parity_of(2), Parity::Even);
        assert_eq!(gi.parity_of(3), Parity::Odd);
        assert_eq!(gi.parity_of(5), Parity::Odd);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_graded_index_out_of_range() {
        let gi = GradedIndex::qubit();
        gi.parity_of(2);
    }

    // --- Fermionic tensor element parity ---

    #[test]
    fn test_tensor_parity_of_element() {
        let gi = GradedIndex::qubit();
        let t = FermionicTensor::zeros(vec![2, 2], vec![gi.clone(), gi.clone()]);

        // |00> -> Even + Even = Even
        assert_eq!(t.parity_of_element(&[0, 0]), Parity::Even);
        // |01> -> Even + Odd = Odd
        assert_eq!(t.parity_of_element(&[0, 1]), Parity::Odd);
        // |10> -> Odd + Even = Odd
        assert_eq!(t.parity_of_element(&[1, 0]), Parity::Odd);
        // |11> -> Odd + Odd = Even
        assert_eq!(t.parity_of_element(&[1, 1]), Parity::Even);
    }

    // --- Parity-preserving tensor check ---

    #[test]
    fn test_parity_preserving_identity() {
        // Identity matrix: nonzero at (0,0) and (1,1), both even total parity.
        let gi = GradedIndex::qubit();
        let t = FermionicTensor::from_data(
            vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            vec![2, 2],
            vec![gi.clone(), gi.clone()],
        );
        assert!(t.is_parity_preserving());
    }

    #[test]
    fn test_parity_not_preserving() {
        // sigma^+ = |1><0| has nonzero at (1,0) which is Odd total parity.
        let gi = GradedIndex::qubit();
        let t = FermionicTensor::from_data(
            vec![c64_zero(), c64_zero(), c64_one(), c64_zero()],
            vec![2, 2],
            vec![gi.clone(), gi.clone()],
        );
        assert!(!t.is_parity_preserving());
    }

    // --- Fermionic SWAP gate ---

    #[test]
    fn test_fermionic_swap_matrix() {
        let mat = FermionicSwapGate::matrix();

        // |00> -> |00>: +1
        assert!(c64_approx_eq(mat[0][0], c64_one()));
        // |01> -> |10>: +1
        assert!(c64_approx_eq(mat[1][2], c64_one()));
        assert!(c64_approx_eq(mat[1][1], c64_zero()));
        // |10> -> |01>: +1
        assert!(c64_approx_eq(mat[2][1], c64_one()));
        assert!(c64_approx_eq(mat[2][2], c64_zero()));
        // |11> -> |11>: -1 (the fermionic sign!)
        assert!(c64_approx_eq(mat[3][3], C64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_fermionic_swap_vs_bosonic() {
        let fswap = FermionicSwapGate::matrix();
        let bswap = GateType::SWAP.matrix();

        // They agree on |00>, |01><->|10>.
        assert!(c64_approx_eq(fswap[0][0], bswap[0][0]));
        assert!(c64_approx_eq(fswap[1][2], bswap[1][2]));
        assert!(c64_approx_eq(fswap[2][1], bswap[2][1]));

        // They DIFFER on |11>: fermionic has -1, bosonic has +1.
        assert!(c64_approx_eq(fswap[3][3], C64::new(-1.0, 0.0)));
        assert!(c64_approx_eq(bswap[3][3], c64_one()));
    }

    // --- Fermionic tensor contraction sign correctness ---

    #[test]
    fn test_contraction_bosonic_no_sign() {
        // Contract two purely even (bosonic) tensors: no sign change.
        let gi = GradedIndex::bosonic(2);
        let a = FermionicTensor::from_data(
            vec![c64_one(), C64::new(2.0, 0.0), C64::new(3.0, 0.0), C64::new(4.0, 0.0)],
            vec![2, 2],
            vec![gi.clone(), gi.clone()],
        );
        let b = FermionicTensor::from_data(
            vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            vec![2, 2],
            vec![gi.clone(), gi.clone()],
        );

        // Contract a's axis 1 with b's axis 0: standard matrix multiply.
        let c = a.contract(&b, 1, 0);
        assert_eq!(c.shape, vec![2, 2]);

        // Result should be standard matrix product: A * B.
        // A = [[1, 2], [3, 4]], B = I
        // C = [[1, 2], [3, 4]]
        assert!(c64_approx_eq(c.get(&[0, 0]), c64_one()));
        assert!(c64_approx_eq(c.get(&[0, 1]), C64::new(2.0, 0.0)));
        assert!(c64_approx_eq(c.get(&[1, 0]), C64::new(3.0, 0.0)));
        assert!(c64_approx_eq(c.get(&[1, 1]), C64::new(4.0, 0.0)));
    }

    #[test]
    fn test_contraction_fermionic_sign() {
        // Test that contracting fermionic tensors with odd indices gives signs.
        // Tensor A: shape [2, 2], qubit grading on both.
        // A[0,0] = 1 (even x even = even)
        // A[1,1] = 1 (odd x odd = even)
        let gi = GradedIndex::qubit();
        let a = FermionicTensor::from_data(
            vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            vec![2, 2],
            vec![gi.clone(), gi.clone()],
        );

        // Tensor B: shape [2, 2], qubit grading.
        // B[0,0] = 1, B[1,1] = 1 (identity).
        let b = FermionicTensor::from_data(
            vec![c64_one(), c64_zero(), c64_zero(), c64_one()],
            vec![2, 2],
            vec![gi.clone(), gi.clone()],
        );

        // Contract axis 1 of A with axis 0 of B.
        // For k=0 (even contracted index): no crossing sign.
        // For k=1 (odd contracted index): crossing sign depends on
        //   odd free indices after self_axis in A: none (axis 1 is last)
        //   odd free indices before other_axis in B: none (axis 0 is first)
        // So no sign changes, result = identity.
        let c = a.contract(&b, 1, 0);
        assert!(c64_approx_eq(c.get(&[0, 0]), c64_one()));
        assert!(c64_approx_eq(c.get(&[0, 1]), c64_zero()));
        assert!(c64_approx_eq(c.get(&[1, 0]), c64_zero()));
        assert!(c64_approx_eq(c.get(&[1, 1]), c64_one()));
    }

    #[test]
    fn test_contraction_with_crossing() {
        // Tensor A: shape [2, 2, 2], qubit grading on all three.
        // Non-zero: A[0, k, 0] = 1 for k=0,1 and A[1, k, 1] = 1 for k=0,1.
        let gi = GradedIndex::qubit();
        let mut a = FermionicTensor::zeros(
            vec![2, 2, 2],
            vec![gi.clone(), gi.clone(), gi.clone()],
        );
        a.set(&[0, 0, 0], c64_one());
        a.set(&[0, 1, 0], c64_one());
        a.set(&[1, 0, 1], c64_one());
        a.set(&[1, 1, 1], c64_one());

        // Tensor B: shape [2], qubit grading. B[0] = 1, B[1] = 1.
        let b = FermionicTensor::from_data(
            vec![c64_one(), c64_one()],
            vec![2],
            vec![gi.clone()],
        );

        // Contract A's axis 1 (middle) with B's axis 0.
        // For contracted index k:
        //   After self_axis=1 in A: axis 2 (the right index).
        //   Before other_axis=0 in B: nothing.
        //
        // For k=1 (odd): count odd free indices after axis 1 in A.
        //   For A element (..., k=1, right_idx):
        //     if right_idx = 1 (odd), one crossing -> sign = -1.
        //     if right_idx = 0 (even), zero crossings -> sign = +1.
        let c = a.contract(&b, 1, 0);
        assert_eq!(c.shape, vec![2, 2]);

        // Result[0, 0] = A[0,0,0]*B[0] + A[0,1,0]*B[1]*sign(k=1, right=0 even)
        //              = 1*1 + 1*1*(+1) = 2
        assert!(c64_approx_eq(c.get(&[0, 0]), C64::new(2.0, 0.0)));

        // Result[1, 1] = A[1,0,1]*B[0] + A[1,1,1]*B[1]*sign(k=1, right=1 odd)
        //              = 1*1 + 1*1*(-1) = 0
        assert!(c64_approx_eq(c.get(&[1, 1]), c64_zero()));
    }

    // --- Jordan-Wigner Z-string generation ---

    #[test]
    fn test_jw_creation_z_string() {
        let jw = JordanWignerTracker::new(4);

        // c^dagger_0: no Z-string (site 0), just sigma^+ at 0.
        let gates = jw.creation(0);
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].0, 0);

        // c^dagger_2: Z at sites 0 and 1, then sigma^+ at 2.
        let gates = jw.creation(2);
        assert_eq!(gates.len(), 3);
        assert_eq!(gates[0].0, 0);
        assert_eq!(gates[0].1, GateType::Z);
        assert_eq!(gates[1].0, 1);
        assert_eq!(gates[1].1, GateType::Z);
        assert_eq!(gates[2].0, 2);
    }

    #[test]
    fn test_jw_annihilation_z_string() {
        let jw = JordanWignerTracker::new(3);

        // c_1: Z at site 0, sigma^- at site 1.
        let gates = jw.annihilation(1);
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].0, 0);
        assert_eq!(gates[0].1, GateType::Z);
        assert_eq!(gates[1].0, 1);
    }

    #[test]
    fn test_jw_hopping_generates_gates() {
        let jw = JordanWignerTracker::new(4);

        // Nearest-neighbor hopping: no intermediate Z gates.
        let gates = jw.hopping(0, 1);
        // Should generate gates (no Z-string for adjacent sites).
        assert!(!gates.is_empty());

        // Longer-range hopping: Z gates on intermediate sites.
        let gates = jw.hopping(0, 3);
        // Should have Z gates at sites 1 and 2.
        let z_gates: Vec<&Gate> = gates
            .iter()
            .filter(|g| g.gate_type == GateType::Z)
            .collect();
        assert!(z_gates.len() >= 2);
    }

    #[test]
    fn test_jw_z_string_parity() {
        let mut jw = JordanWignerTracker::new(5);
        // No occupations: parity = +1.
        assert_eq!(jw.z_string_parity(0, 4), 1);

        // Occupy site 2: parity flips for strings crossing it.
        jw.set_occupation(2, true);
        assert_eq!(jw.z_string_parity(0, 4), -1);
        assert_eq!(jw.z_string_parity(0, 2), 1); // site 2 not between 0 and 2

        // Occupy site 3 as well: two odd sites -> parity = +1.
        jw.set_occupation(3, true);
        assert_eq!(jw.z_string_parity(0, 4), 1);
    }

    // --- FermionicMPS density computation ---

    #[test]
    fn test_fermionic_mps_vacuum_density() {
        let mps = FermionicMPS::new(4, 8);

        // Vacuum state: all densities should be 0.
        for site in 0..4 {
            let n = mps.compute_density(site);
            assert!(approx_eq(n, 0.0), "Vacuum density at site {}: {}", site, n);
        }
    }

    #[test]
    fn test_fermionic_mps_norm() {
        let mps = FermionicMPS::new(3, 4);
        let norm = mps.norm_sq();
        assert!(
            approx_eq(norm, 1.0),
            "MPS norm should be 1.0, got {}",
            norm
        );
    }

    #[test]
    fn test_fermionic_mps_vacuum_correlation() {
        let mps = FermionicMPS::new(3, 4);

        // Vacuum: all correlations zero.
        let c01 = mps.compute_correlation(0, 1);
        assert!(
            c64_approx_eq(c01, c64_zero()),
            "Vacuum correlation <c^dag_0 c_1> should be 0, got {}",
            c01
        );
    }

    // --- Crossing count for various permutations ---

    #[test]
    fn test_crossing_count_identity() {
        assert_eq!(crossing_count(&[0, 1, 2, 3]), 0);
    }

    #[test]
    fn test_crossing_count_single_swap() {
        assert_eq!(crossing_count(&[1, 0]), 1);
        assert_eq!(crossing_count(&[0, 2, 1]), 1);
    }

    #[test]
    fn test_crossing_count_reverse() {
        // Reverse of [0,1,2] = [2,1,0]: 3 inversions.
        assert_eq!(crossing_count(&[2, 1, 0]), 3);
        // Reverse of [0,1,2,3] = [3,2,1,0]: 6 inversions.
        assert_eq!(crossing_count(&[3, 2, 1, 0]), 6);
    }

    #[test]
    fn test_crossing_count_cyclic() {
        // Cyclic permutation [1,2,0]: inversions (1>0), (2>0) = 2.
        assert_eq!(crossing_count(&[1, 2, 0]), 2);
        // Cyclic [2,0,1]: inversions (2>0), (2>1) = 2.
        assert_eq!(crossing_count(&[2, 0, 1]), 2);
    }

    #[test]
    fn test_fermionic_sign_even_odd() {
        assert_eq!(fermionic_sign(&[0, 1, 2]), 1); // 0 inversions -> +1
        assert_eq!(fermionic_sign(&[1, 0, 2]), -1); // 1 inversion -> -1
        assert_eq!(fermionic_sign(&[2, 1, 0]), -1); // 3 inversions -> -1
        assert_eq!(fermionic_sign(&[1, 2, 0]), 1); // 2 inversions -> +1
    }

    // --- Multi-index round-trip ---

    #[test]
    fn test_flat_multi_index_roundtrip() {
        let gi = GradedIndex::qubit();
        let t = FermionicTensor::zeros(vec![2, 3, 2], vec![
            GradedIndex::qubit(),
            GradedIndex::new(3, 2, 1),
            GradedIndex::qubit(),
        ]);

        for flat in 0..12 {
            let mi = t.multi_index(flat);
            let recovered = t.flat_index(&mi);
            assert_eq!(flat, recovered, "Round-trip failed for flat={}", flat);
        }
    }
}
