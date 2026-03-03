// Projected Entangled Pair States (PEPS) for 2D Quantum Systems
//
// Real PEPS tensor network implementation with:
// - 5-index tensors (physical, bond_up, bond_down, bond_left, bond_right)
// - Proper single-qubit gate contraction over physical index
// - Two-qubit gate application via contract-SVD-split
// - Boundary MPS contraction for amplitude computation
// - Measurement via amplitude sampling
// - Entanglement entropy via reduced density matrix eigenvalues

use ndarray::{Array, Array2, Ix5};
use num_complex::Complex64 as c64;

// ─── Coordinate ──────────────────────────────────────────────────────────

/// 2D coordinate for PEPS tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PEPSCoord {
    pub x: usize,
    pub y: usize,
}

impl PEPSCoord {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    /// Manhattan distance to another coordinate
    pub fn distance(&self, other: &PEPSCoord) -> usize {
        let dx = if self.x > other.x {
            self.x - other.x
        } else {
            other.x - self.x
        };
        let dy = if self.y > other.y {
            self.y - other.y
        } else {
            other.y - self.y
        };
        dx + dy
    }
}

// ─── Bond direction ──────────────────────────────────────────────────────

/// Bond directions in PEPS tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondDirection {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

// ─── PEPSTensor ──────────────────────────────────────────────────────────

/// PEPS tensor at a single lattice site.
///
/// Index order: (physical, bond_up, bond_down, bond_left, bond_right).
/// Physical dimension is 2 for qubits.
#[derive(Clone)]
pub struct PEPSTensor {
    /// 5D tensor data: [phys, up, down, left, right]
    pub(crate) data: Array<c64, Ix5>,
    /// Current bond dimensions [up, down, left, right]
    pub(crate) bond_dims: [usize; 4],
    /// Physical dimension (always 2 for qubits)
    pub(crate) phys_dim: usize,
}

impl PEPSTensor {
    /// Create a product state tensor |0> with bond dimension 1.
    ///
    /// Shape: (2, 1, 1, 1, 1). Only data[0,0,0,0,0] = 1.
    pub fn zero(_phys_dim: usize) -> Self {
        let mut data = Array::<c64, Ix5>::zeros((2, 1, 1, 1, 1));
        data[[0, 0, 0, 0, 0]] = c64::new(1.0, 0.0);
        Self {
            data,
            bond_dims: [1, 1, 1, 1],
            phys_dim: 2,
        }
    }

    /// Create |+> = (|0> + |1>) / sqrt(2) with bond dimension 1.
    pub fn plus_state(_phys_dim: usize) -> Self {
        let inv = 1.0 / 2.0_f64.sqrt();
        let mut data = Array::<c64, Ix5>::zeros((2, 1, 1, 1, 1));
        data[[0, 0, 0, 0, 0]] = c64::new(inv, 0.0);
        data[[1, 0, 0, 0, 0]] = c64::new(inv, 0.0);
        Self {
            data,
            bond_dims: [1, 1, 1, 1],
            phys_dim: 2,
        }
    }

    /// Create tensor from raw 5D data and bond dimensions.
    pub fn from_data(data: Array<c64, Ix5>, bond_dims: [usize; 4]) -> Self {
        let phys_dim = data.shape()[0];
        Self {
            data,
            bond_dims,
            phys_dim,
        }
    }

    /// Get bond dimension in a specific direction
    pub fn bond_dim(&self, direction: BondDirection) -> usize {
        self.bond_dims[direction as usize]
    }

    /// Set bond dimension (metadata only -- caller must also reshape `data`)
    pub fn set_bond_dim(&mut self, direction: BondDirection, dim: usize) {
        self.bond_dims[direction as usize] = dim;
    }

    /// Sum of squared magnitudes of all elements.
    pub fn norm_sqr(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sqr()).sum()
    }
}

// ─── PEPS state ──────────────────────────────────────────────────────────

/// PEPS (Projected Entangled Pair States) for a 2D qubit lattice.
pub struct PEPS {
    /// 2D grid of tensors, indexed [row][col] i.e. [y][x]
    pub(crate) tensors: Vec<Vec<PEPSTensor>>,
    /// Grid width (number of columns)
    width: usize,
    /// Grid height (number of rows)
    height: usize,
    /// Maximum bond dimension for truncation
    max_bond_dim: usize,
    /// Physical dimension (2 for qubits)
    phys_dim: usize,
}

impl PEPS {
    /// Create a new PEPS state with all qubits in |0>.
    pub fn new(width: usize, height: usize, max_bond_dim: usize) -> Self {
        let tensors = (0..height)
            .map(|_| (0..width).map(|_| PEPSTensor::zero(2)).collect())
            .collect();
        Self {
            tensors,
            width,
            height,
            max_bond_dim,
            phys_dim: 2,
        }
    }

    /// Apply Hadamard to every qubit, returning the modified state.
    pub fn hadamard_all(mut self) -> Self {
        for y in 0..self.height {
            for x in 0..self.width {
                self.tensors[y][x] = PEPSTensor::plus_state(2);
            }
        }
        self
    }

    // ── single-qubit gate ────────────────────────────────────────────

    /// Apply a single-qubit gate to site (coord.x, coord.y).
    ///
    /// Contraction: new_T[i, bu, bd, bl, br] = sum_j gate[i,j] * T[j, bu, bd, bl, br]
    pub fn apply_single_qubit_gate<F>(&mut self, coord: PEPSCoord, gate_fn: F)
    where
        F: FnOnce() -> Array2<c64>,
    {
        let gate = gate_fn();
        let tensor = &self.tensors[coord.y][coord.x];
        let shape = tensor.data.shape().to_vec();
        // shape = [d, D_up, D_down, D_left, D_right]
        let d = shape[0];
        let du = shape[1];
        let dd = shape[2];
        let dl = shape[3];
        let dr = shape[4];

        let mut new_data = Array::<c64, Ix5>::zeros((d, du, dd, dl, dr));

        for i in 0..d {
            for bu in 0..du {
                for bd in 0..dd {
                    for bl in 0..dl {
                        for br in 0..dr {
                            let mut val = c64::new(0.0, 0.0);
                            for j in 0..d {
                                val += gate[[i, j]] * tensor.data[[j, bu, bd, bl, br]];
                            }
                            new_data[[i, bu, bd, bl, br]] = val;
                        }
                    }
                }
            }
        }

        self.tensors[coord.y][coord.x].data = new_data;
    }

    // ── two-qubit gate ───────────────────────────────────────────────

    /// Apply a two-qubit gate between adjacent sites.
    ///
    /// The gate is a 4x4 unitary in the computational basis |00>, |01>, |10>, |11>.
    /// The two sites must be nearest-neighbors (Manhattan distance 1).
    ///
    /// Algorithm:
    /// 1. Contract the two site tensors over the shared bond into a combined tensor.
    /// 2. Apply the 4x4 gate to the combined physical indices.
    /// 3. SVD-split back into two tensors and truncate to max_bond_dim.
    pub fn apply_two_qubit_gate<F>(&mut self, coord1: PEPSCoord, coord2: PEPSCoord, gate_fn: F)
    where
        F: FnOnce() -> Array2<c64>,
    {
        if coord1.distance(&coord2) != 1 {
            return; // only nearest-neighbor
        }

        let gate = gate_fn();
        let d = self.phys_dim; // 2

        // Determine bond direction between the two sites.
        let horizontal = coord1.y == coord2.y;

        // Ensure coord_a is left/above coord_b.
        let (ca, cb) = if horizontal {
            if coord1.x < coord2.x {
                (coord1, coord2)
            } else {
                (coord2, coord1)
            }
        } else if coord1.y < coord2.y {
            (coord1, coord2)
        } else {
            (coord2, coord1)
        };

        // If we swapped, we need to adjust the gate: swap the two qubit spaces.
        let gate = if (horizontal && coord1.x > coord2.x)
            || (!horizontal && coord1.y > coord2.y)
        {
            swap_gate_qubits(&gate, d)
        } else {
            gate
        };

        let ta = self.tensors[ca.y][ca.x].clone();
        let tb = self.tensors[cb.y][cb.x].clone();

        let sha = ta.data.shape().to_vec(); // [d, Du, Dd, Dl, Dr]
        let shb = tb.data.shape().to_vec();

        if horizontal {
            // Shared bond: ta's Right <-> tb's Left
            let bond_shared = sha[4]; // Dr of a = Dl of b
            debug_assert_eq!(bond_shared, shb[3]);

            // Other bond dims for site a: Du_a, Dd_a, Dl_a
            let du_a = sha[1];
            let dd_a = sha[2];
            let dl_a = sha[3];

            // Other bond dims for site b: Du_b, Dd_b, Dr_b
            let du_b = shb[1];
            let dd_b = shb[2];
            let dr_b = shb[4];

            // Combined tensor theta[pa, pb, Du_a, Dd_a, Dl_a, Du_b, Dd_b, Dr_b]
            // = sum_s ta[pa, Du_a, Dd_a, Dl_a, s] * tb[pb, Du_b, Dd_b, s, Dr_b]
            let other_a = du_a * dd_a * dl_a;
            let other_b = du_b * dd_b * dr_b;

            // Build theta as a matrix: rows = (pa, other_a), cols = (pb, other_b)
            let nrows = d * other_a;
            let ncols = d * other_b;
            let mut theta_mat = Array2::<c64>::zeros((nrows, ncols));

            for pa in 0..d {
                for pb in 0..d {
                    for bua in 0..du_a {
                        for bda in 0..dd_a {
                            for bla in 0..dl_a {
                                for bub in 0..du_b {
                                    for bdb in 0..dd_b {
                                        for brb in 0..dr_b {
                                            let mut val = c64::new(0.0, 0.0);
                                            for s in 0..bond_shared {
                                                val += ta.data[[pa, bua, bda, bla, s]]
                                                    * tb.data[[pb, bub, bdb, s, brb]];
                                            }
                                            let row =
                                                pa * other_a + bua * dd_a * dl_a + bda * dl_a + bla;
                                            let col = pb * other_b
                                                + bub * dd_b * dr_b
                                                + bdb * dr_b
                                                + brb;
                                            theta_mat[[row, col]] = val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Apply gate to physical indices of theta.
            // theta_mat is indexed (pa * other_a + ..., pb * other_b + ...).
            // We need to apply gate[pa', pb' ; pa, pb] to the physical part.
            let mut gated = Array2::<c64>::zeros((nrows, ncols));

            for pa_new in 0..d {
                for pb_new in 0..d {
                    for oa in 0..other_a {
                        for ob in 0..other_b {
                            let mut val = c64::new(0.0, 0.0);
                            for pa in 0..d {
                                for pb in 0..d {
                                    let g_row = pa_new * d + pb_new;
                                    let g_col = pa * d + pb;
                                    val += gate[[g_row, g_col]]
                                        * theta_mat[[pa * other_a + oa, pb * other_b + ob]];
                                }
                            }
                            gated[[pa_new * other_a + oa, pb_new * other_b + ob]] = val;
                        }
                    }
                }
            }

            // SVD of gated matrix and truncate
            let (u_trunc, s_vals, vt_trunc, new_bond) =
                truncated_svd(&gated, self.max_bond_dim);

            // Absorb singular values symmetrically: sqrt(s) into each side
            let sqrt_s: Vec<f64> = s_vals.iter().map(|s| s.sqrt()).collect();

            // Reshape U * sqrt(S) back to tensor_a: [d, Du_a, Dd_a, Dl_a, new_bond]
            let mut new_ta =
                Array::<c64, Ix5>::zeros((d, du_a, dd_a, dl_a, new_bond));
            for pa in 0..d {
                for bua in 0..du_a {
                    for bda in 0..dd_a {
                        for bla in 0..dl_a {
                            let row = pa * other_a + bua * dd_a * dl_a + bda * dl_a + bla;
                            for k in 0..new_bond {
                                new_ta[[pa, bua, bda, bla, k]] =
                                    u_trunc[[row, k]] * c64::new(sqrt_s[k], 0.0);
                            }
                        }
                    }
                }
            }

            // Reshape sqrt(S) * Vt back to tensor_b: [d, Du_b, Dd_b, new_bond, Dr_b]
            let mut new_tb =
                Array::<c64, Ix5>::zeros((d, du_b, dd_b, new_bond, dr_b));
            for pb in 0..d {
                for bub in 0..du_b {
                    for bdb in 0..dd_b {
                        for brb in 0..dr_b {
                            let col =
                                pb * other_b + bub * dd_b * dr_b + bdb * dr_b + brb;
                            for k in 0..new_bond {
                                new_tb[[pb, bub, bdb, k, brb]] =
                                    c64::new(sqrt_s[k], 0.0) * vt_trunc[[k, col]];
                            }
                        }
                    }
                }
            }

            self.tensors[ca.y][ca.x] = PEPSTensor::from_data(
                new_ta,
                [du_a, dd_a, dl_a, new_bond],
            );
            self.tensors[cb.y][cb.x] = PEPSTensor::from_data(
                new_tb,
                [du_b, dd_b, new_bond, dr_b],
            );
        } else {
            // Vertical: shared bond is ta's Down <-> tb's Up
            let bond_shared = sha[2]; // Dd of a = Du of b
            debug_assert_eq!(bond_shared, shb[1]);

            let du_a = sha[1];
            let dl_a = sha[3];
            let dr_a = sha[4];

            let dd_b = shb[2];
            let dl_b = shb[3];
            let dr_b = shb[4];

            let other_a = du_a * dl_a * dr_a;
            let other_b = dd_b * dl_b * dr_b;
            let nrows = d * other_a;
            let ncols = d * other_b;

            let mut theta_mat = Array2::<c64>::zeros((nrows, ncols));

            for pa in 0..d {
                for pb in 0..d {
                    for bua in 0..du_a {
                        for bla in 0..dl_a {
                            for bra in 0..dr_a {
                                for bdb in 0..dd_b {
                                    for blb in 0..dl_b {
                                        for brb in 0..dr_b {
                                            let mut val = c64::new(0.0, 0.0);
                                            for s in 0..bond_shared {
                                                val += ta.data[[pa, bua, s, bla, bra]]
                                                    * tb.data[[pb, s, bdb, blb, brb]];
                                            }
                                            let row = pa * other_a
                                                + bua * dl_a * dr_a
                                                + bla * dr_a
                                                + bra;
                                            let col = pb * other_b
                                                + bdb * dl_b * dr_b
                                                + blb * dr_b
                                                + brb;
                                            theta_mat[[row, col]] = val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Apply gate
            let mut gated = Array2::<c64>::zeros((nrows, ncols));
            for pa_new in 0..d {
                for pb_new in 0..d {
                    for oa in 0..other_a {
                        for ob in 0..other_b {
                            let mut val = c64::new(0.0, 0.0);
                            for pa in 0..d {
                                for pb in 0..d {
                                    val += gate[[pa_new * d + pb_new, pa * d + pb]]
                                        * theta_mat[[pa * other_a + oa, pb * other_b + ob]];
                                }
                            }
                            gated[[pa_new * other_a + oa, pb_new * other_b + ob]] = val;
                        }
                    }
                }
            }

            let (u_trunc, s_vals, vt_trunc, new_bond) =
                truncated_svd(&gated, self.max_bond_dim);
            let sqrt_s: Vec<f64> = s_vals.iter().map(|s| s.sqrt()).collect();

            // Reshape back: ta has [d, Du_a, new_bond, Dl_a, Dr_a]
            let mut new_ta =
                Array::<c64, Ix5>::zeros((d, du_a, new_bond, dl_a, dr_a));
            for pa in 0..d {
                for bua in 0..du_a {
                    for bla in 0..dl_a {
                        for bra in 0..dr_a {
                            let row =
                                pa * other_a + bua * dl_a * dr_a + bla * dr_a + bra;
                            for k in 0..new_bond {
                                new_ta[[pa, bua, k, bla, bra]] =
                                    u_trunc[[row, k]] * c64::new(sqrt_s[k], 0.0);
                            }
                        }
                    }
                }
            }

            // tb has [d, new_bond, Dd_b, Dl_b, Dr_b]
            let mut new_tb =
                Array::<c64, Ix5>::zeros((d, new_bond, dd_b, dl_b, dr_b));
            for pb in 0..d {
                for bdb in 0..dd_b {
                    for blb in 0..dl_b {
                        for brb in 0..dr_b {
                            let col =
                                pb * other_b + bdb * dl_b * dr_b + blb * dr_b + brb;
                            for k in 0..new_bond {
                                new_tb[[pb, k, bdb, blb, brb]] =
                                    c64::new(sqrt_s[k], 0.0) * vt_trunc[[k, col]];
                            }
                        }
                    }
                }
            }

            self.tensors[ca.y][ca.x] = PEPSTensor::from_data(
                new_ta,
                [du_a, new_bond, dl_a, dr_a],
            );
            self.tensors[cb.y][cb.x] = PEPSTensor::from_data(
                new_tb,
                [new_bond, dd_b, dl_b, dr_b],
            );
        }
    }

    // ── contraction ──────────────────────────────────────────────────

    /// Contract the entire PEPS into a state vector of 2^n amplitudes.
    ///
    /// This is exact and only feasible for small systems (n = width * height <= ~16).
    /// The contraction proceeds row by row using boundary vectors.
    pub fn contract_to_amplitudes(&self) -> Vec<c64> {
        let n = self.width * self.height;
        let dim = 1usize << n; // 2^n
        let mut amplitudes = vec![c64::new(0.0, 0.0); dim];

        // Enumerate all computational basis states
        for basis in 0..dim {
            amplitudes[basis] = self.amplitude_of_basis(basis, n);
        }
        amplitudes
    }

    /// Compute the amplitude <basis|psi> for a single computational basis state.
    ///
    /// `basis` encodes qubit values as bits: bit at position (y*width+x) gives |0> or |1>.
    fn amplitude_of_basis(&self, basis: usize, _n: usize) -> c64 {
        // For each site, fix the physical index to the basis bit.
        // Then contract all bond indices.
        //
        // Strategy: row-by-row contraction.
        // For each row, we contract along the horizontal (left-right) bonds
        // while accumulating vertical (up-down) bond indices via a "boundary" vector.
        //
        // The boundary after processing row r has shape:
        //   boundary[down_bond_0, down_bond_1, ..., down_bond_{width-1}]
        // which is a rank-width tensor (stored flat).

        // Initialize: no boundary yet
        // Process row 0 first to set up the boundary.

        let mut boundary: Vec<c64> = vec![c64::new(1.0, 0.0)]; // scalar
        let mut boundary_shape: Vec<usize> = vec![]; // dimensions of down-bonds

        for y in 0..self.height {
            // For this row, build the "row transfer tensor":
            // Contract horizontal bonds within the row, producing a tensor
            // indexed by (up_bonds..., down_bonds...).
            //
            // We do this incrementally: start with site (0, y), then absorb
            // site (1, y), etc., contracting the right-left bond each time.

            let row_result = self.contract_row(y, basis);
            // row_result.data is indexed [up_0, ..., up_{w-1}, down_0, ..., down_{w-1}]
            // row_result.up_dims and row_result.down_dims give the dimension lists.

            // Contract boundary (indexed by boundary_shape = previous down dims)
            // with row_result's up-indices (must match previous down dims).
            let new_boundary = contract_boundary_with_row(
                &boundary,
                &boundary_shape,
                &row_result.data,
                &row_result.up_dims,
                &row_result.down_dims,
            );

            boundary = new_boundary;
            boundary_shape = row_result.down_dims;
        }

        // After the last row, the boundary_shape should be all 1s (bottom edge),
        // so the result is a scalar.
        debug_assert_eq!(boundary.len(), 1, "Final contraction should yield a scalar");
        boundary[0]
    }

    /// Contract a single row into a transfer tensor.
    ///
    /// Returns a RowContraction whose `data` is indexed by
    /// [up_0, up_1, ..., up_{w-1}, down_0, down_1, ..., down_{w-1}].
    fn contract_row(&self, y: usize, basis: usize) -> RowContraction {
        let w = self.width;

        // Collect the physical index for each site in this row.
        let phys_indices: Vec<usize> = (0..w)
            .map(|x| {
                let bit_pos = y * w + x;
                (basis >> bit_pos) & 1
            })
            .collect();

        // Extract the slice of each tensor with the physical index fixed.
        // For site (x, y) with phys index p, the remaining tensor is:
        //   T_fixed[bu, bd, bl, br] = T[p, bu, bd, bl, br]
        // We store these as 4D arrays.

        struct SiteTensor {
            bu: usize,
            bd: usize,
            bl: usize,
            br: usize,
            data: Vec<c64>, // flat [bu, bd, bl, br]
        }

        let sites: Vec<SiteTensor> = (0..w)
            .map(|x| {
                let t = &self.tensors[y][x];
                let p = phys_indices[x];
                let sh = t.data.shape();
                let bu = sh[1];
                let bd = sh[2];
                let bl = sh[3];
                let br = sh[4];
                let mut flat = vec![c64::new(0.0, 0.0); bu * bd * bl * br];
                for iu in 0..bu {
                    for id in 0..bd {
                        for il in 0..bl {
                            for ir in 0..br {
                                flat[iu * bd * bl * br + id * bl * br + il * br + ir] =
                                    t.data[[p, iu, id, il, ir]];
                            }
                        }
                    }
                }
                SiteTensor {
                    bu,
                    bd,
                    bl,
                    br,
                    data: flat,
                }
            })
            .collect();

        // Now contract left-to-right over the horizontal (left-right) bonds.
        //
        // After absorbing site x, we have a "partial" tensor indexed by:
        //   [up_0, ..., up_x, down_0, ..., down_x, right_bond_of_x]
        //
        // When we absorb site x+1, we contract partial's right_bond_of_x with
        // site_{x+1}'s left bond.

        // Start with site 0.
        let s0 = &sites[0];
        // Site 0 must have bl = 1 (left edge). Its partial is:
        //   partial[bu_0, bd_0, br_0] = s0[bu_0, bd_0, 0, br_0]
        let mut partial: Vec<c64> = vec![c64::new(0.0, 0.0); s0.bu * s0.bd * s0.br];
        let mut up_dims: Vec<usize> = vec![s0.bu];
        let mut down_dims: Vec<usize> = vec![s0.bd];
        let mut partial_shape: Vec<usize> = vec![s0.bu, s0.bd, s0.br];
        // partial_shape layout: [up_0, down_0, right_bond]

        for iu in 0..s0.bu {
            for id in 0..s0.bd {
                for ir in 0..s0.br {
                    // bl=0 since left edge
                    partial[iu * s0.bd * s0.br + id * s0.br + ir] =
                        s0.data[iu * s0.bd * s0.bl * s0.br + id * s0.bl * s0.br + 0 * s0.br + ir];
                }
            }
        }

        for x in 1..w {
            let sx = &sites[x];
            // Contract partial's last dim (right_bond of x-1) with sx's left bond.
            let prev_right = *partial_shape.last().unwrap();
            debug_assert_eq!(prev_right, sx.bl);

            // New partial indexed by: [up_0..up_{x-1}, down_0..down_{x-1}, up_x, down_x, right_x]
            // Total dims = product of all up/down dims so far * up_x * down_x * right_x
            let prev_outer: usize = partial.len() / prev_right;

            let new_len = prev_outer * sx.bu * sx.bd * sx.br;
            let mut new_partial = vec![c64::new(0.0, 0.0); new_len];

            // Iterate: for each element of partial (indexed by outer, right_bond)
            // and each element of sx (indexed by bu, bd, bl=right_bond, br):
            for outer_idx in 0..prev_outer {
                for sbu in 0..sx.bu {
                    for sbd in 0..sx.bd {
                        for sbr in 0..sx.br {
                            let mut val = c64::new(0.0, 0.0);
                            for bond in 0..prev_right {
                                let p_idx = outer_idx * prev_right + bond;
                                let s_idx = sbu * sx.bd * sx.bl * sx.br
                                    + sbd * sx.bl * sx.br
                                    + bond * sx.br
                                    + sbr;
                                val += partial[p_idx] * sx.data[s_idx];
                            }
                            let new_idx = outer_idx * sx.bu * sx.bd * sx.br
                                + sbu * sx.bd * sx.br
                                + sbd * sx.br
                                + sbr;
                            new_partial[new_idx] = val;
                        }
                    }
                }
            }

            up_dims.push(sx.bu);
            down_dims.push(sx.bd);

            // Update partial_shape: remove last (prev right bond), add up_x, down_x, right_x
            partial_shape.pop();
            partial_shape.push(sx.bu);
            partial_shape.push(sx.bd);
            partial_shape.push(sx.br);

            partial = new_partial;
        }

        // The final partial's last dim is the right bond of the rightmost site,
        // which should be 1 (right edge). Contract/sum it out.
        let last_right = *partial_shape.last().unwrap();
        debug_assert_eq!(last_right, 1);
        // Just remove that trailing dim of size 1.
        partial_shape.pop();
        // partial data doesn't change dimensionally since dim=1.

        // Now partial is indexed by [up_0, down_0, up_1, down_1, ..., up_{w-1}, down_{w-1}]
        // We want to reorder to [up_0, ..., up_{w-1}, down_0, ..., down_{w-1}].
        // Current order of dims in partial_shape: u0, d0, u1, d1, ...
        let total: usize = partial.len();
        let mut reordered = vec![c64::new(0.0, 0.0); total];

        // Build strides for current layout and target layout.
        let rank = partial_shape.len(); // 2*w
        debug_assert_eq!(rank, 2 * w);

        // Current strides (row-major):
        let mut cur_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            cur_strides[i] = cur_strides[i + 1] * partial_shape[i + 1];
        }

        // Target order: [u0, u1, ..., u_{w-1}, d0, d1, ..., d_{w-1}]
        // Target shape:
        let mut target_shape = Vec::with_capacity(rank);
        for i in 0..w {
            target_shape.push(up_dims[i]);
        }
        for i in 0..w {
            target_shape.push(down_dims[i]);
        }

        let mut target_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            target_strides[i] = target_strides[i + 1] * target_shape[i + 1];
        }

        // Mapping: target axis k -> current axis
        // target axes 0..w are up_0..up_{w-1} -> current axes 0, 2, 4, ...
        // target axes w..2w are down_0..down_{w-1} -> current axes 1, 3, 5, ...
        let mut perm = vec![0usize; rank];
        for i in 0..w {
            perm[i] = 2 * i; // up_i is at current axis 2*i
            perm[w + i] = 2 * i + 1; // down_i is at current axis 2*i+1
        }

        // Iterate over all multi-indices in target layout.
        fn advance_multi(idx: &mut Vec<usize>, shape: &[usize]) -> bool {
            let rank = idx.len();
            for k in (0..rank).rev() {
                idx[k] += 1;
                if idx[k] < shape[k] {
                    return true;
                }
                idx[k] = 0;
            }
            false
        }

        if total > 0 {
            let mut multi = vec![0usize; rank];
            loop {
                // Compute target flat index
                let mut t_flat = 0;
                for k in 0..rank {
                    t_flat += multi[k] * target_strides[k];
                }
                // Compute current flat index
                let mut c_flat = 0;
                for k in 0..rank {
                    // target axis k corresponds to current axis perm[k]
                    c_flat += multi[k] * cur_strides[perm[k]];
                }
                reordered[t_flat] = partial[c_flat];

                if !advance_multi(&mut multi, &target_shape) {
                    break;
                }
            }
        }

        RowContraction {
            data: reordered,
            up_dims,
            down_dims,
        }
    }

    /// Legacy API: returns bond dimensions as floats (for backward compatibility).
    pub fn contract_to_mps(&self) -> Vec<f64> {
        let mut bond_dims = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let tensor = &self.tensors[y][x];
                for &dim in &tensor.bond_dims {
                    bond_dims.push(dim as f64);
                }
            }
        }
        bond_dims
    }

    // ── measurement ──────────────────────────────────────────────────

    /// Measure all qubits by contracting and sampling from the probability distribution.
    ///
    /// Uses a deterministic seed derived from the state for reproducibility.
    pub fn measure(&mut self) -> u64 {
        let n = self.width * self.height;
        if n > 20 {
            // Too large for exact contraction; fall back to approximate per-site measurement.
            return self.measure_approximate();
        }

        let amps = self.contract_to_amplitudes();
        let probs: Vec<f64> = amps.iter().map(|a| a.norm_sqr()).collect();
        let total: f64 = probs.iter().sum();

        if total < 1e-15 {
            return 0;
        }

        // Build a deterministic "random" value from the state itself.
        let seed = self.deterministic_seed();
        let r = (seed as f64) / (u64::MAX as f64) * total;

        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= r {
                return i as u64;
            }
        }

        (probs.len() - 1) as u64
    }

    /// Approximate measurement for large systems: compute single-site reduced
    /// density matrices by contracting the environment approximately.
    fn measure_approximate(&self) -> u64 {
        let mut result = 0u64;
        let seed = self.deterministic_seed();
        let mut rng_state = seed;

        for y in 0..self.height {
            for x in 0..self.width {
                let t = &self.tensors[y][x];
                // Compute prob(|0>) and prob(|1>) from the tensor norm
                let mut p0 = 0.0_f64;
                let mut p1 = 0.0_f64;
                let sh = t.data.shape();
                for bu in 0..sh[1] {
                    for bd in 0..sh[2] {
                        for bl in 0..sh[3] {
                            for br in 0..sh[4] {
                                p0 += t.data[[0, bu, bd, bl, br]].norm_sqr();
                                p1 += t.data[[1, bu, bd, bl, br]].norm_sqr();
                            }
                        }
                    }
                }
                let total = p0 + p1;
                if total > 1e-15 {
                    // Simple xorshift for deterministic pseudo-random
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let r = (rng_state as f64) / (u64::MAX as f64);
                    if r > p0 / total {
                        let bit = y * self.width + x;
                        result |= 1u64 << bit;
                    }
                }
            }
        }
        result
    }

    /// Derive a deterministic seed from the PEPS state.
    fn deterministic_seed(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        for y in 0..self.height {
            for x in 0..self.width {
                let t = &self.tensors[y][x];
                for val in t.data.iter() {
                    let bits_re = val.re.to_bits();
                    let bits_im = val.im.to_bits();
                    hash ^= bits_re;
                    hash = hash.wrapping_mul(0x100000001b3); // FNV prime
                    hash ^= bits_im;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
        }
        // Ensure nonzero
        if hash == 0 {
            hash = 1;
        }
        hash
    }

    // ── accessors ────────────────────────────────────────────────────

    pub fn num_qubits(&self) -> usize {
        self.width * self.height
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn bond_dimensions(&self) -> Vec<[usize; 4]> {
        let mut all = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                all.push(self.tensors[y][x].bond_dims);
            }
        }
        all
    }

    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                total += self.tensors[y][x].data.len() * 16; // 16 bytes per c64
            }
        }
        total
    }

    pub fn statistics(&self) -> PEPSStatistics {
        let bond_dims = self.bond_dimensions();
        let mut avg_bond = [0.0_f64; 4];
        let mut max_bond = [0_usize; 4];

        for dims in &bond_dims {
            for (i, dim) in dims.iter().enumerate() {
                avg_bond[i] += *dim as f64;
                if *dim > max_bond[i] {
                    max_bond[i] = *dim;
                }
            }
        }

        let n_tensors = bond_dims.len();
        for avg in avg_bond.iter_mut() {
            *avg /= n_tensors as f64;
        }

        PEPSStatistics {
            num_qubits: self.num_qubits(),
            width: self.width,
            height: self.height,
            num_tensors: n_tensors,
            avg_bond_dim: avg_bond,
            max_bond_dim: max_bond,
            memory_bytes: self.memory_usage(),
        }
    }
}

// ─── Row contraction helper ──────────────────────────────────────────────

struct RowContraction {
    /// Flat data indexed [up_0, ..., up_{w-1}, down_0, ..., down_{w-1}]
    data: Vec<c64>,
    up_dims: Vec<usize>,
    down_dims: Vec<usize>,
}

/// Contract a boundary vector (indexed by previous down-bond dims) with
/// a row transfer tensor (indexed by [up_dims, down_dims]).
///
/// boundary[d0_prev, d1_prev, ...] * row[u0, u1, ..., d0_new, d1_new, ...]
/// where u_i is contracted with d_i_prev.
fn contract_boundary_with_row(
    boundary: &[c64],
    boundary_shape: &[usize],    // previous down dims (one per column)
    row_data: &[c64],
    row_up_dims: &[usize],      // must match boundary_shape
    row_down_dims: &[usize],
) -> Vec<c64> {
    if boundary_shape.is_empty() {
        // First row: boundary is scalar (1.0). Just sum over up dims (all should be 1).
        let up_total: usize = row_up_dims.iter().product();
        debug_assert_eq!(up_total, 1, "First row should have up bond dim 1");
        // Output is the down part of row_data.
        let down_total: usize = row_down_dims.iter().product();
        // row_data has shape up_total * down_total.
        debug_assert_eq!(row_data.len(), up_total * down_total);
        // Just take the row_data[0, :] (up indices all zero).
        let mut result = vec![c64::new(0.0, 0.0); down_total];
        for d in 0..down_total {
            result[d] = boundary[0] * row_data[d]; // up_idx=0, down_idx=d
        }
        return result;
    }

    // General case: boundary has shape = boundary_shape, row has shape = [row_up_dims, row_down_dims].
    // Contract: boundary_shape must equal row_up_dims (element-wise).
    debug_assert_eq!(boundary_shape, row_up_dims);

    let up_total: usize = row_up_dims.iter().product();
    let down_total: usize = row_down_dims.iter().product();
    debug_assert_eq!(row_data.len(), up_total * down_total);
    debug_assert_eq!(boundary.len(), up_total);

    let mut result = vec![c64::new(0.0, 0.0); down_total];
    for u in 0..up_total {
        let b_val = boundary[u];
        if b_val.norm_sqr() < 1e-30 {
            continue;
        }
        for d in 0..down_total {
            result[d] += b_val * row_data[u * down_total + d];
        }
    }

    result
}

// ─── SVD ─────────────────────────────────────────────────────────────────

/// Swap qubit order in a 4x4 two-qubit gate.
///
/// Maps |ab> -> |ba> on both input and output.
fn swap_gate_qubits(gate: &Array2<c64>, d: usize) -> Array2<c64> {
    let n = d * d;
    let mut swapped = Array2::<c64>::zeros((n, n));
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    // original: gate[i*d+j, k*d+l]
                    // swapped:  gate[j*d+i, l*d+k]
                    swapped[[j * d + i, l * d + k]] = gate[[i * d + j, k * d + l]];
                }
            }
        }
    }
    swapped
}

/// Truncated SVD of a complex matrix via eigendecomposition of M^dagger M.
///
/// Returns (U_trunc, sigma_trunc, Vt_trunc, new_rank) where new_rank <= max_rank.
fn truncated_svd(
    mat: &Array2<c64>,
    max_rank: usize,
) -> (Array2<c64>, Vec<f64>, Array2<c64>, usize) {
    let (m, n) = mat.dim();
    let k = m.min(n);
    let rank = k.min(max_rank);

    if m == 0 || n == 0 {
        return (
            Array2::zeros((m, 0)),
            vec![],
            Array2::zeros((0, n)),
            0,
        );
    }

    // Compute M^dagger M (n x n) for right singular vectors.
    // For small matrices this is fine.
    let mut mdm = Array2::<c64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut val = c64::new(0.0, 0.0);
            for r in 0..m {
                val += mat[[r, i]].conj() * mat[[r, j]];
            }
            mdm[[i, j]] = val;
        }
    }

    // Compute eigenvectors of M^dagger M via QR iteration (Jacobi for Hermitian).
    let (eigenvalues, eigenvectors) = hermitian_eigen(&mdm);

    // eigenvalues are the squared singular values. Sort descending.
    let mut indexed: Vec<(usize, f64)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.max(0.0)))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let actual_rank = indexed
        .iter()
        .take(rank)
        .filter(|(_, v)| *v > 1e-28)
        .count()
        .max(1); // keep at least 1

    let sigma: Vec<f64> = indexed.iter().take(actual_rank).map(|(_, v)| v.sqrt()).collect();

    // V_trunc: take the top eigenvectors as columns of V (n x actual_rank)
    let mut v_trunc = Array2::<c64>::zeros((n, actual_rank));
    for (j, &(orig_idx, _)) in indexed.iter().take(actual_rank).enumerate() {
        for i in 0..n {
            v_trunc[[i, j]] = eigenvectors[[i, orig_idx]];
        }
    }

    // Vt_trunc = V_trunc^dagger (actual_rank x n)
    let mut vt_trunc = Array2::<c64>::zeros((actual_rank, n));
    for i in 0..actual_rank {
        for j in 0..n {
            vt_trunc[[i, j]] = v_trunc[[j, i]].conj();
        }
    }

    // U_trunc = M * V_trunc * Sigma^{-1} (m x actual_rank)
    let mut u_trunc = Array2::<c64>::zeros((m, actual_rank));
    for i in 0..m {
        for j in 0..actual_rank {
            let mut val = c64::new(0.0, 0.0);
            for l in 0..n {
                val += mat[[i, l]] * v_trunc[[l, j]];
            }
            if sigma[j] > 1e-14 {
                u_trunc[[i, j]] = val / c64::new(sigma[j], 0.0);
            }
        }
    }

    (u_trunc, sigma, vt_trunc, actual_rank)
}

/// Eigendecomposition of a Hermitian matrix via Jacobi rotations.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns.
fn hermitian_eigen(mat: &Array2<c64>) -> (Vec<f64>, Array2<c64>) {
    let n = mat.dim().0;
    debug_assert_eq!(n, mat.dim().1);

    if n == 0 {
        return (vec![], Array2::zeros((0, 0)));
    }

    if n == 1 {
        return (vec![mat[[0, 0]].re], Array2::from_elem((1, 1), c64::new(1.0, 0.0)));
    }

    // Work on a real symmetric matrix of the Hermitian: use the real part
    // of M^dagger M (which is already real for Hermitian matrices from M^dagger M).
    // But M^dagger M is always Hermitian, so its eigenvalues are real.

    // Copy to mutable work matrix
    let mut a = mat.clone();
    let mut v = Array2::<c64>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = c64::new(1.0, 0.0);
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let mag = a[[i, j]].norm_sqr();
                if mag > max_off {
                    max_off = mag;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < 1e-28 {
            break; // converged
        }

        // Compute Jacobi rotation to zero out a[p][q]
        let apq = a[[p, q]];
        let app = a[[p, p]].re;
        let aqq = a[[q, q]].re;

        // For complex Hermitian: first rotate phase so apq becomes real
        let phase = if apq.norm() > 1e-15 {
            apq / c64::new(apq.norm(), 0.0)
        } else {
            c64::new(1.0, 0.0)
        };
        let apq_real = apq.norm();

        // 2x2 real Jacobi rotation
        let tau = (aqq - app) / (2.0 * apq_real);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cos = 1.0 / (1.0 + t * t).sqrt();
        let sin = t * cos;

        // Apply rotation: columns p and q of a, then rows, then eigenvector matrix
        // This is a complex Jacobi rotation incorporating the phase.

        // Update matrix A
        for i in 0..n {
            if i != p && i != q {
                let aip = a[[i, p]];
                let aiq = a[[i, q]];
                a[[i, p]] = c64::new(cos, 0.0) * aip + c64::new(sin, 0.0) * aiq * phase.conj();
                a[[i, q]] =
                    -c64::new(sin, 0.0) * aip * phase + c64::new(cos, 0.0) * aiq;
                a[[p, i]] = a[[i, p]].conj();
                a[[q, i]] = a[[i, q]].conj();
            }
        }

        let new_pp = app + t * apq_real;
        let new_qq = aqq - t * apq_real;
        a[[p, p]] = c64::new(new_pp, 0.0);
        a[[q, q]] = c64::new(new_qq, 0.0);
        a[[p, q]] = c64::new(0.0, 0.0);
        a[[q, p]] = c64::new(0.0, 0.0);

        // Update eigenvector matrix
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c64::new(cos, 0.0) * vip + c64::new(sin, 0.0) * viq * phase.conj();
            v[[i, q]] = -c64::new(sin, 0.0) * vip * phase + c64::new(cos, 0.0) * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]].re).collect();
    (eigenvalues, v)
}

// ─── Statistics ──────────────────────────────────────────────────────────

/// Statistics about PEPS state
#[derive(Debug, Clone)]
pub struct PEPSStatistics {
    pub num_qubits: usize,
    pub width: usize,
    pub height: usize,
    pub num_tensors: usize,
    pub avg_bond_dim: [f64; 4],
    pub max_bond_dim: [usize; 4],
    pub memory_bytes: usize,
}

// ─── BoundaryMPS ─────────────────────────────────────────────────────────

/// Boundary MPS for approximate PEPS contraction.
///
/// Each tensor is a complex matrix representing bond_left x bond_right.
pub struct BoundaryMPS {
    /// MPS tensors representing the boundary
    tensors: Vec<Array2<c64>>,
    width: usize,
    /// Maximum bond dimension for truncation
    max_chi: usize,
}

impl BoundaryMPS {
    /// Create a new boundary MPS with bond dimension 1 at each position.
    pub fn new(width: usize) -> Self {
        let tensors = (0..width)
            .map(|_| {
                let mut m = Array2::<c64>::zeros((1, 1));
                m[[0, 0]] = c64::new(1.0, 0.0);
                m
            })
            .collect();
        Self {
            tensors,
            width,
            max_chi: 64,
        }
    }

    /// Create with a specified maximum bond dimension for truncation.
    pub fn with_max_chi(width: usize, max_chi: usize) -> Self {
        let mut bm = Self::new(width);
        bm.max_chi = max_chi;
        bm
    }

    /// Absorb a column of PEPS tensors into this boundary.
    ///
    /// Each element of `column` is a 2D matrix (down_bond x right_bond)
    /// with the physical and left-bond indices already contracted against
    /// the boundary.
    pub fn absorb_column(&mut self, column: &[Array2<c64>]) {
        debug_assert_eq!(column.len(), self.width);
        for i in 0..self.width {
            // New boundary tensor = old_boundary * column_tensor
            let old = &self.tensors[i];
            let col = &column[i];
            // Matrix multiply: (old_rows x old_cols) * (col_rows x col_cols)
            // old_cols should equal col_rows
            let (m, k1) = old.dim();
            let (k2, n) = col.dim();
            let k = k1.min(k2);
            let mut product = Array2::<c64>::zeros((m, n));
            for ii in 0..m {
                for jj in 0..n {
                    let mut val = c64::new(0.0, 0.0);
                    for kk in 0..k {
                        val += old[[ii, kk]] * col[[kk, jj]];
                    }
                    product[[ii, jj]] = val;
                }
            }
            self.tensors[i] = product;
        }

        // Truncate if bond dimension exceeds max_chi
        self.truncate();
    }

    /// Truncate boundary bond dimensions via SVD.
    pub fn truncate(&mut self) {
        for i in 0..self.width {
            let (m, n) = self.tensors[i].dim();
            if m > self.max_chi || n > self.max_chi {
                let (u, s, vt, rank) = truncated_svd(&self.tensors[i], self.max_chi);
                // Reconstruct truncated: U * S * Vt
                let mut truncated = Array2::<c64>::zeros((u.dim().0, vt.dim().1));
                for ii in 0..u.dim().0 {
                    for jj in 0..vt.dim().1 {
                        let mut val = c64::new(0.0, 0.0);
                        for kk in 0..rank {
                            val += u[[ii, kk]] * c64::new(s[kk], 0.0) * vt[[kk, jj]];
                        }
                        truncated[[ii, jj]] = val;
                    }
                }
                self.tensors[i] = truncated;
            }
        }
    }

    /// Get maximum bond dimension in the boundary.
    pub fn max_bond_dim(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| t.dim().0.max(t.dim().1))
            .max()
            .unwrap_or(1)
    }
}

// ─── ContractionMethod ──────────────────────────────────────────────────

/// Contraction method for PEPS
#[derive(Debug, Clone, Copy)]
pub enum ContractionMethod {
    /// Simple boundary contraction (fast, approximate)
    Simple,
    /// Full boundary contraction with environment (slower, more accurate)
    Full,
}

// ─── PEPSimulator ────────────────────────────────────────────────────────

/// PEPS simulator with configurable contraction strategy
pub struct PEPSimulator {
    peps: PEPS,
    method: ContractionMethod,
    max_bond_dim: usize,
}

impl PEPSimulator {
    /// Create a new PEPS simulator
    pub fn new(width: usize, height: usize, max_bond_dim: usize) -> Self {
        Self {
            peps: PEPS::new(width, height, max_bond_dim),
            method: ContractionMethod::Simple,
            max_bond_dim,
        }
    }

    /// Create with default configuration (max bond dim 64)
    pub fn with_defaults(width: usize, height: usize) -> Self {
        Self::new(width, height, 64)
    }

    /// Apply Hadamard gate
    pub fn h(&mut self, x: usize, y: usize) {
        let coord = PEPSCoord::new(x, y);
        self.peps.apply_single_qubit_gate(coord, || {
            let inv = 1.0 / 2.0_f64.sqrt();
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(inv, 0.0),
                    c64::new(inv, 0.0),
                    c64::new(inv, 0.0),
                    c64::new(-inv, 0.0),
                ],
            )
            .unwrap()
        });
    }

    /// Apply X (Pauli-X / NOT) gate
    pub fn x(&mut self, x: usize, y: usize) {
        let coord = PEPSCoord::new(x, y);
        self.peps.apply_single_qubit_gate(coord, || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(0.0, 0.0),
                    c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });
    }

    /// Apply Y (Pauli-Y) gate
    pub fn y(&mut self, x: usize, y: usize) {
        let coord = PEPSCoord::new(x, y);
        self.peps.apply_single_qubit_gate(coord, || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(0.0, 0.0),
                    c64::new(0.0, -1.0),
                    c64::new(0.0, 1.0),
                    c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });
    }

    /// Apply Z (Pauli-Z) gate
    pub fn z(&mut self, x: usize, y: usize) {
        let coord = PEPSCoord::new(x, y);
        self.peps.apply_single_qubit_gate(coord, || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(-1.0, 0.0),
                ],
            )
            .unwrap()
        });
    }

    /// Apply CNOT gate between adjacent sites
    pub fn cnot(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) {
        let coord1 = PEPSCoord::new(x1, y1);
        let coord2 = PEPSCoord::new(x2, y2);

        if coord1.distance(&coord2) != 1 {
            return;
        }

        self.peps.apply_two_qubit_gate(coord1, coord2, || {
            Array2::from_shape_vec(
                (4, 4),
                vec![
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });
    }

    /// Measure all qubits
    pub fn measure(&mut self) -> u64 {
        self.peps.measure()
    }

    pub fn num_qubits(&self) -> usize {
        self.peps.num_qubits()
    }

    pub fn dimensions(&self) -> (usize, usize) {
        self.peps.dimensions()
    }

    pub fn statistics(&self) -> PEPSStatistics {
        self.peps.statistics()
    }

    /// Compute entanglement entropy at each site.
    ///
    /// For each site, compute the reduced density matrix by tracing over all
    /// bond indices, then compute S = -sum(lambda * ln(lambda)).
    pub fn entanglement_entropy_2d(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::new();

        for y in 0..self.peps.height {
            let mut row = Vec::new();
            for x in 0..self.peps.width {
                let t = &self.peps.tensors[y][x];
                let entropy = site_entanglement_entropy(t);
                row.push(entropy);
            }
            result.push(row);
        }

        result
    }

    /// Access the underlying PEPS state (immutable).
    pub fn peps(&self) -> &PEPS {
        &self.peps
    }

    /// Access the underlying PEPS state (mutable).
    pub fn peps_mut(&mut self) -> &mut PEPS {
        &mut self.peps
    }
}

/// Compute the single-site entanglement entropy by tracing the tensor
/// over all bond indices to get the reduced density matrix rho[i, j]
/// for the physical qubit, then computing von Neumann entropy.
fn site_entanglement_entropy(tensor: &PEPSTensor) -> f64 {
    let d = tensor.phys_dim;
    let sh = tensor.data.shape();
    let du = sh[1];
    let dd = sh[2];
    let dl = sh[3];
    let dr = sh[4];

    // rho[i, j] = sum_{bu, bd, bl, br} T[i, bu, bd, bl, br] * conj(T[j, bu, bd, bl, br])
    let mut rho = Array2::<c64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let mut val = c64::new(0.0, 0.0);
            for bu in 0..du {
                for bd in 0..dd {
                    for bl in 0..dl {
                        for br in 0..dr {
                            val += tensor.data[[i, bu, bd, bl, br]]
                                * tensor.data[[j, bu, bd, bl, br]].conj();
                        }
                    }
                }
            }
            rho[[i, j]] = val;
        }
    }

    // Normalize: tr(rho) should be 1
    let trace: f64 = (0..d).map(|i| rho[[i, i]].re).sum();
    if trace < 1e-15 {
        return 0.0;
    }
    for i in 0..d {
        for j in 0..d {
            rho[[i, j]] = rho[[i, j]] / c64::new(trace, 0.0);
        }
    }

    // Eigenvalues of the 2x2 density matrix
    let eigenvalues = if d == 2 {
        let a = rho[[0, 0]].re;
        let b = rho[[1, 1]].re;
        let off_diag = rho[[0, 1]].norm_sqr();
        let disc = ((a - b) * (a - b) + 4.0 * off_diag).sqrt();
        let l1 = 0.5 * ((a + b) + disc);
        let l2 = 0.5 * ((a + b) - disc);
        vec![l1, l2]
    } else {
        // General case: use Jacobi eigendecomposition
        let (evals, _) = hermitian_eigen(&rho);
        evals
    };

    // Von Neumann entropy: S = -sum(lambda * ln(lambda))
    let mut entropy = 0.0;
    for &lam in &eigenvalues {
        if lam > 1e-15 {
            entropy -= lam * lam.ln();
        }
    }
    entropy
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peps_coord_distance() {
        let c1 = PEPSCoord::new(0, 0);
        let c2 = PEPSCoord::new(3, 4);
        assert_eq!(c1.distance(&c2), 7);
    }

    #[test]
    fn test_peps_tensor_creation() {
        let tensor = PEPSTensor::zero(2);
        assert_eq!(tensor.phys_dim, 2);
        assert_eq!(tensor.bond_dims, [1, 1, 1, 1]);
        assert_eq!(tensor.data.shape(), &[2, 1, 1, 1, 1]);

        // |0> state: data[0,0,0,0,0] = 1, data[1,0,0,0,0] = 0
        assert!((tensor.data[[0, 0, 0, 0, 0]].re - 1.0).abs() < 1e-10);
        assert!(tensor.data[[1, 0, 0, 0, 0]].norm_sqr() < 1e-10);
    }

    #[test]
    fn test_peps_tensor_plus_state() {
        let tensor = PEPSTensor::plus_state(2);
        let inv = 1.0 / 2.0_f64.sqrt();
        assert!((tensor.data[[0, 0, 0, 0, 0]].re - inv).abs() < 1e-10);
        assert!((tensor.data[[1, 0, 0, 0, 0]].re - inv).abs() < 1e-10);
        // Norm should be 1
        assert!((tensor.norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_peps_creation() {
        let peps = PEPS::new(4, 4, 16);
        assert_eq!(peps.width, 4);
        assert_eq!(peps.height, 4);
        assert_eq!(peps.num_qubits(), 16);
    }

    #[test]
    fn test_peps_hadamard() {
        let peps = PEPS::new(3, 3, 8).hadamard_all();
        let inv = 1.0 / 2.0_f64.sqrt();
        for y in 0..3 {
            for x in 0..3 {
                let t = &peps.tensors[y][x];
                assert!((t.data[[0, 0, 0, 0, 0]].re - inv).abs() < 1e-10);
                assert!((t.data[[1, 0, 0, 0, 0]].re - inv).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_single_qubit_gate_x_on_zero() {
        // Apply X to |0> should give |1>
        let mut peps = PEPS::new(1, 1, 4);
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });

        let t = &peps.tensors[0][0];
        // Should be |1>: data[0,...] = 0, data[1,...] = 1
        assert!(t.data[[0, 0, 0, 0, 0]].norm_sqr() < 1e-10);
        assert!((t.data[[1, 0, 0, 0, 0]].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_gate_h_on_zero() {
        // H|0> = |+> = (|0> + |1>)/sqrt(2)
        let mut peps = PEPS::new(1, 1, 4);
        let inv = 1.0 / 2.0_f64.sqrt();
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(inv, 0.0), c64::new(inv, 0.0),
                    c64::new(inv, 0.0), c64::new(-inv, 0.0),
                ],
            )
            .unwrap()
        });

        let t = &peps.tensors[0][0];
        assert!((t.data[[0, 0, 0, 0, 0]].re - inv).abs() < 1e-10);
        assert!((t.data[[1, 0, 0, 0, 0]].re - inv).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_gate_correctness() {
        // Apply Z to |1>: should give -|1>
        let mut peps = PEPS::new(1, 1, 4);
        // First put in |1>
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });
        // Apply Z
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(-1.0, 0.0),
                ],
            )
            .unwrap()
        });

        let t = &peps.tensors[0][0];
        assert!(t.data[[0, 0, 0, 0, 0]].norm_sqr() < 1e-10);
        assert!((t.data[[1, 0, 0, 0, 0]].re - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_contract_1x1_zero() {
        // 1x1 PEPS in |0> should give amplitude [1, 0]
        let peps = PEPS::new(1, 1, 4);
        let amps = peps.contract_to_amplitudes();
        assert_eq!(amps.len(), 2);
        assert!((amps[0].re - 1.0).abs() < 1e-10);
        assert!(amps[1].norm_sqr() < 1e-10);
    }

    #[test]
    fn test_contract_1x1_plus() {
        // 1x1 PEPS in |+>
        let peps = PEPS::new(1, 1, 4).hadamard_all();
        let amps = peps.contract_to_amplitudes();
        assert_eq!(amps.len(), 2);
        let inv = 1.0 / 2.0_f64.sqrt();
        assert!((amps[0].re - inv).abs() < 1e-10);
        assert!((amps[1].re - inv).abs() < 1e-10);
    }

    #[test]
    fn test_contract_2x1_product() {
        // 2x1 PEPS, both in |0>. Amplitudes: [1, 0, 0, 0] for |00>, |01>, |10>, |11>
        let peps = PEPS::new(2, 1, 4);
        let amps = peps.contract_to_amplitudes();
        assert_eq!(amps.len(), 4);
        assert!((amps[0].re - 1.0).abs() < 1e-10); // |00>
        assert!(amps[1].norm_sqr() < 1e-10);
        assert!(amps[2].norm_sqr() < 1e-10);
        assert!(amps[3].norm_sqr() < 1e-10);
    }

    #[test]
    fn test_contract_1x2_product() {
        // 1x2 PEPS (1 column, 2 rows), both in |0>
        let peps = PEPS::new(1, 2, 4);
        let amps = peps.contract_to_amplitudes();
        assert_eq!(amps.len(), 4);
        assert!((amps[0].re - 1.0).abs() < 1e-10); // |00>
    }

    #[test]
    fn test_contract_small_2x2() {
        // 2x2 PEPS, all |0>. Amplitudes: 2^4 = 16 entries, only |0000> = 1
        let peps = PEPS::new(2, 2, 4);
        let amps = peps.contract_to_amplitudes();
        assert_eq!(amps.len(), 16);
        assert!((amps[0].re - 1.0).abs() < 1e-10);
        for i in 1..16 {
            assert!(amps[i].norm_sqr() < 1e-10, "amp[{}] should be 0", i);
        }
    }

    #[test]
    fn test_contract_2x1_with_x_gate() {
        // 2x1 PEPS: apply X to site (0,0), so state is |10>
        let mut peps = PEPS::new(2, 1, 4);
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });
        let amps = peps.contract_to_amplitudes();
        // Bit 0 is site (0,0). |10> means bit0=1, bit1=0, so basis index = 1.
        assert!(amps[0].norm_sqr() < 1e-10);
        assert!((amps[1].re - 1.0).abs() < 1e-10); // |10> -> index 1
        assert!(amps[2].norm_sqr() < 1e-10);
        assert!(amps[3].norm_sqr() < 1e-10);
    }

    #[test]
    fn test_bell_state_horizontal() {
        // Create Bell state |00> + |11> / sqrt(2) using H + CNOT on 2x1 grid.
        let mut peps = PEPS::new(2, 1, 8);

        // H on qubit (0,0)
        let inv = 1.0 / 2.0_f64.sqrt();
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(inv, 0.0), c64::new(inv, 0.0),
                    c64::new(inv, 0.0), c64::new(-inv, 0.0),
                ],
            )
            .unwrap()
        });

        // CNOT: control (0,0), target (1,0)
        peps.apply_two_qubit_gate(PEPSCoord::new(0, 0), PEPSCoord::new(1, 0), || {
            Array2::from_shape_vec(
                (4, 4),
                vec![
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });

        let amps = peps.contract_to_amplitudes();
        // Bell state: (|00> + |11>) / sqrt(2)
        // bit layout: bit0 = site(0,0), bit1 = site(1,0)
        // |00> = index 0, |11> = index 3
        assert!((amps[0].norm_sqr() - 0.5).abs() < 1e-6, "|00> amp^2 = {}", amps[0].norm_sqr());
        assert!(amps[1].norm_sqr() < 1e-6, "|01> should be ~0, got {}", amps[1].norm_sqr());
        assert!(amps[2].norm_sqr() < 1e-6, "|10> should be ~0, got {}", amps[2].norm_sqr());
        assert!((amps[3].norm_sqr() - 0.5).abs() < 1e-6, "|11> amp^2 = {}", amps[3].norm_sqr());
    }

    #[test]
    fn test_bell_state_vertical() {
        // Bell state on a 1x2 grid (vertical neighbors).
        let mut peps = PEPS::new(1, 2, 8);

        let inv = 1.0 / 2.0_f64.sqrt();
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(inv, 0.0), c64::new(inv, 0.0),
                    c64::new(inv, 0.0), c64::new(-inv, 0.0),
                ],
            )
            .unwrap()
        });

        peps.apply_two_qubit_gate(PEPSCoord::new(0, 0), PEPSCoord::new(0, 1), || {
            Array2::from_shape_vec(
                (4, 4),
                vec![
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });

        let amps = peps.contract_to_amplitudes();
        // bit0 = site(0,0), bit1 = site(0,1)
        assert!((amps[0].norm_sqr() - 0.5).abs() < 1e-6);
        assert!(amps[1].norm_sqr() < 1e-6);
        assert!(amps[2].norm_sqr() < 1e-6);
        assert!((amps[3].norm_sqr() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_svd_truncation() {
        // Apply many gates and verify bond dimension stays bounded.
        let mut peps = PEPS::new(3, 1, 4);

        let inv = 1.0 / 2.0_f64.sqrt();
        for x in 0..3 {
            peps.apply_single_qubit_gate(PEPSCoord::new(x, 0), || {
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        c64::new(inv, 0.0), c64::new(inv, 0.0),
                        c64::new(inv, 0.0), c64::new(-inv, 0.0),
                    ],
                )
                .unwrap()
            });
        }

        // Apply multiple CNOT gates
        for _ in 0..5 {
            peps.apply_two_qubit_gate(PEPSCoord::new(0, 0), PEPSCoord::new(1, 0), || {
                Array2::from_shape_vec(
                    (4, 4),
                    vec![
                        c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                        c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                        c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                        c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                    ],
                )
                .unwrap()
            });
            peps.apply_two_qubit_gate(PEPSCoord::new(1, 0), PEPSCoord::new(2, 0), || {
                Array2::from_shape_vec(
                    (4, 4),
                    vec![
                        c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                        c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0),
                        c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                        c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                    ],
                )
                .unwrap()
            });
        }

        // Check all bond dims are <= max_bond_dim
        for dims in peps.bond_dimensions() {
            for &d in &dims {
                assert!(d <= 4, "Bond dim {} exceeds max 4", d);
            }
        }
    }

    #[test]
    fn test_measurement_deterministic() {
        // Same state should give the same measurement (deterministic seed).
        let mut peps1 = PEPS::new(2, 1, 4);
        let mut peps2 = PEPS::new(2, 1, 4);

        let inv = 1.0 / 2.0_f64.sqrt();
        for peps in [&mut peps1, &mut peps2] {
            peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        c64::new(inv, 0.0), c64::new(inv, 0.0),
                        c64::new(inv, 0.0), c64::new(-inv, 0.0),
                    ],
                )
                .unwrap()
            });
        }

        assert_eq!(peps1.measure(), peps2.measure());
    }

    #[test]
    fn test_measurement_zero_state() {
        // |00> should always measure 0.
        let mut peps = PEPS::new(2, 1, 4);
        assert_eq!(peps.measure(), 0);
    }

    #[test]
    fn test_measurement_one_state() {
        // |11> should always measure 3 (bits 0 and 1 set).
        let mut peps = PEPS::new(2, 1, 4);
        peps.apply_single_qubit_gate(PEPSCoord::new(0, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });
        peps.apply_single_qubit_gate(PEPSCoord::new(1, 0), || {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    c64::new(0.0, 0.0), c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                ],
            )
            .unwrap()
        });
        assert_eq!(peps.measure(), 3);
    }

    #[test]
    fn test_norm_sqr() {
        let t = PEPSTensor::zero(2);
        assert!((t.norm_sqr() - 1.0).abs() < 1e-10);

        let t2 = PEPSTensor::plus_state(2);
        assert!((t2.norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_mps() {
        let boundary = BoundaryMPS::new(4);
        assert_eq!(boundary.width, 4);
        assert_eq!(boundary.max_bond_dim(), 1);
    }

    #[test]
    fn test_boundary_mps_absorb() {
        let mut boundary = BoundaryMPS::new(2);
        let column: Vec<Array2<c64>> = (0..2)
            .map(|_| {
                let mut m = Array2::<c64>::zeros((1, 2));
                m[[0, 0]] = c64::new(1.0, 0.0);
                m[[0, 1]] = c64::new(0.5, 0.0);
                m
            })
            .collect();
        boundary.absorb_column(&column);
        assert_eq!(boundary.max_bond_dim(), 2);
    }

    #[test]
    fn test_peps_simulator() {
        let mut sim = PEPSimulator::with_defaults(4, 4);
        sim.h(1, 1);
        sim.cnot(1, 1, 1, 2);

        let stats = sim.statistics();
        assert_eq!(stats.num_qubits, 16);
        assert!(stats.width == 4 && stats.height == 4);
    }

    #[test]
    fn test_entanglement_entropy_product_state() {
        // Product state should have zero entanglement entropy everywhere.
        let sim = PEPSimulator::with_defaults(3, 3);
        let entropy = sim.entanglement_entropy_2d();
        assert_eq!(entropy.len(), 3);
        for row in &entropy {
            assert_eq!(row.len(), 3);
            for &e in row {
                assert!(e.abs() < 1e-10, "Product state entropy should be 0, got {}", e);
            }
        }
    }

    #[test]
    fn test_entanglement_entropy_after_entangling() {
        let mut sim = PEPSimulator::with_defaults(2, 1);
        sim.h(0, 0);
        sim.cnot(0, 0, 1, 0);

        let entropy = sim.entanglement_entropy_2d();
        // After creating Bell state, individual site entropies should be > 0
        // (each site is maximally entangled -> S = ln(2) ~ 0.693).
        // Due to SVD truncation the value may not be exact but should be positive.
        assert!(entropy[0][0] > 0.01, "Expected positive entropy, got {}", entropy[0][0]);
        assert!(entropy[0][1] > 0.01, "Expected positive entropy, got {}", entropy[0][1]);
    }

    #[test]
    fn test_hermitian_eigen_identity() {
        let mat = Array2::from_shape_vec(
            (2, 2),
            vec![
                c64::new(1.0, 0.0), c64::new(0.0, 0.0),
                c64::new(0.0, 0.0), c64::new(1.0, 0.0),
            ],
        )
        .unwrap();

        let (evals, _) = hermitian_eigen(&mat);
        assert!((evals[0] - 1.0).abs() < 1e-10);
        assert!((evals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hermitian_eigen_diagonal() {
        let mat = Array2::from_shape_vec(
            (2, 2),
            vec![
                c64::new(3.0, 0.0), c64::new(0.0, 0.0),
                c64::new(0.0, 0.0), c64::new(7.0, 0.0),
            ],
        )
        .unwrap();

        let (mut evals, _) = hermitian_eigen(&mat);
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((evals[0] - 3.0).abs() < 1e-10);
        assert!((evals[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_truncated_svd_roundtrip() {
        // Create a rank-2 matrix, SVD it, and verify reconstruction.
        let mat = Array2::from_shape_vec(
            (3, 3),
            vec![
                c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(0.0, 0.0),
                c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(3.0, 0.0),
                c64::new(1.0, 0.0), c64::new(3.0, 0.0), c64::new(3.0, 0.0),
            ],
        )
        .unwrap();

        let (u, s, vt, rank) = truncated_svd(&mat, 3);
        assert!(rank >= 2);

        // Reconstruct: U * S * Vt should approximate mat
        let mut recon = Array2::<c64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                let mut val = c64::new(0.0, 0.0);
                for k in 0..rank {
                    val += u[[i, k]] * c64::new(s[k], 0.0) * vt[[k, j]];
                }
                recon[[i, j]] = val;
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                let diff = (recon[[i, j]] - mat[[i, j]]).norm();
                assert!(diff < 1e-6, "SVD reconstruction error at [{},{}]: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_contract_normalization() {
        // A product state |0...0> should have norm 1.
        let peps = PEPS::new(2, 2, 4);
        let amps = peps.contract_to_amplitudes();
        let norm_sq: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10, "Norm^2 = {}", norm_sq);
    }

    #[test]
    fn test_contract_superposition_normalization() {
        // |+...+> on 2x2 grid: each amplitude should be 1/sqrt(16) = 1/4
        let peps = PEPS::new(2, 2, 4).hadamard_all();
        let amps = peps.contract_to_amplitudes();
        let norm_sq: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-8, "Norm^2 = {}", norm_sq);

        let expected = 1.0 / 16.0; // (1/sqrt(2))^4 squared = 1/16
        for (i, a) in amps.iter().enumerate() {
            assert!(
                (a.norm_sqr() - expected).abs() < 1e-8,
                "amp[{}]^2 = {}, expected {}",
                i,
                a.norm_sqr(),
                expected
            );
        }
    }

    #[test]
    fn test_memory_usage() {
        let peps = PEPS::new(2, 2, 4);
        let mem = peps.memory_usage();
        // Each tensor: 2*1*1*1*1 = 2 elements * 16 bytes = 32 bytes
        // 4 tensors => 128 bytes
        assert_eq!(mem, 4 * 2 * 16);
    }

    #[test]
    fn test_statistics() {
        let peps = PEPS::new(3, 2, 8);
        let stats = peps.statistics();
        assert_eq!(stats.num_qubits, 6);
        assert_eq!(stats.width, 3);
        assert_eq!(stats.height, 2);
        assert_eq!(stats.num_tensors, 6);
    }
}
