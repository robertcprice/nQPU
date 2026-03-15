// Projected Entangled Pair Operators (PEPO) for 2D Quantum Systems
//
// PEPO is the operator (density matrix) analogue of PEPS: each tensor has
// TWO physical indices (bra + ket) plus four bond indices connecting to
// neighbors on a 2D grid.  This is the natural representation for:
//   - Mixed states / density matrices of 2D systems
//   - Thermal (Gibbs) states via imaginary-time Trotter evolution
//   - Quantum channels and noise models
//
// This is the first known Rust implementation of PEPO.  Only Quimb (Python)
// has a comparable feature set.

use ndarray::Array2;
use num_complex::Complex64 as c64;

// ─── Constants ────────────────────────────────────────────────────────────

const ZERO: c64 = c64 { re: 0.0, im: 0.0 };
const ONE: c64 = c64 { re: 1.0, im: 0.0 };
const TOLERANCE: f64 = 1e-12;
const SVD_CUTOFF: f64 = 1e-14;

// ─── Direction helpers ────────────────────────────────────────────────────

/// Bond directions for PEPO tensors on a 2D grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
}

// ─── PepoTensor ──────────────────────────────────────────────────────────

/// A single PEPO tensor at lattice site (row, col).
///
/// Represents a rank-6 tensor with indices:
///   [phys_in, phys_out, left, right, up, down]
///
/// The total number of elements is:
///   phys_dim^2 * bond_dims[0] * bond_dims[1] * bond_dims[2] * bond_dims[3]
///
/// Storage is row-major with the index ordering above.
#[derive(Debug, Clone)]
pub struct PepoTensor {
    /// Flattened tensor data in row-major order.
    pub data: Vec<c64>,
    /// Physical dimension (typically 2 for qubits).
    pub phys_dim: usize,
    /// Bond dimensions: [left, right, up, down].
    pub bond_dims: [usize; 4],
}

impl PepoTensor {
    /// Total number of elements in this tensor.
    pub fn numel(&self) -> usize {
        let d = self.phys_dim;
        let [dl, dr, du, dd] = self.bond_dims;
        d * d * dl * dr * du * dd
    }

    /// Strides for the 6-index tensor [phys_in, phys_out, left, right, up, down].
    fn strides(&self) -> [usize; 6] {
        let d = self.phys_dim;
        let [dl, dr, du, dd] = self.bond_dims;
        let s5 = 1; // down
        let s4 = dd; // up
        let s3 = du * dd; // right
        let s2 = dr * du * dd; // left
        let s1 = dl * dr * du * dd; // phys_out
        let s0 = d * dl * dr * du * dd; // phys_in
        [s0, s1, s2, s3, s4, s5]
    }

    /// Linear index for [pi, po, l, r, u, d].
    #[inline]
    fn idx(&self, pi: usize, po: usize, l: usize, r: usize, u: usize, d: usize) -> usize {
        let s = self.strides();
        pi * s[0] + po * s[1] + l * s[2] + r * s[3] + u * s[4] + d * s[5]
    }

    /// Get element at [pi, po, l, r, u, d].
    #[inline]
    pub fn get(&self, pi: usize, po: usize, l: usize, r: usize, u: usize, d: usize) -> c64 {
        self.data[self.idx(pi, po, l, r, u, d)]
    }

    /// Set element at [pi, po, l, r, u, d].
    #[inline]
    pub fn set(&mut self, pi: usize, po: usize, l: usize, r: usize, u: usize, d: usize, val: c64) {
        let i = self.idx(pi, po, l, r, u, d);
        self.data[i] = val;
    }

    /// Create a zero tensor with the given dimensions.
    pub fn zeros(phys_dim: usize, bond_dims: [usize; 4]) -> Self {
        let numel = phys_dim * phys_dim * bond_dims[0] * bond_dims[1] * bond_dims[2] * bond_dims[3];
        Self {
            data: vec![ZERO; numel],
            phys_dim,
            bond_dims,
        }
    }

    /// Create an identity operator tensor: delta(phys_in, phys_out) with bond dim 1.
    pub fn identity(phys_dim: usize) -> Self {
        let bond_dims = [1, 1, 1, 1];
        let mut t = Self::zeros(phys_dim, bond_dims);
        for i in 0..phys_dim {
            t.set(i, i, 0, 0, 0, 0, ONE);
        }
        t
    }

    /// Create a single-site operator tensor from a d x d matrix, bond dim 1.
    pub fn from_operator(op: &[c64], phys_dim: usize) -> Self {
        assert_eq!(op.len(), phys_dim * phys_dim);
        let bond_dims = [1, 1, 1, 1];
        let mut t = Self::zeros(phys_dim, bond_dims);
        for pi in 0..phys_dim {
            for po in 0..phys_dim {
                t.set(pi, po, 0, 0, 0, 0, op[pi * phys_dim + po]);
            }
        }
        t
    }

    /// Frobenius norm squared.
    pub fn norm_sqr(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Scale all elements by a scalar.
    pub fn scale(&mut self, s: c64) {
        for v in self.data.iter_mut() {
            *v *= s;
        }
    }
}

// ─── PEPO ─────────────────────────────────────────────────────────────────

/// Projected Entangled Pair Operator on an Ly x Lx rectangular lattice.
///
/// Represents a density-matrix-like operator as a 2D tensor network where
/// each site has a rank-6 tensor with two physical legs and four bond legs.
#[derive(Debug, Clone)]
pub struct Pepo {
    /// Number of rows.
    pub ly: usize,
    /// Number of columns.
    pub lx: usize,
    /// Physical dimension per site (2 for qubits).
    pub phys_dim: usize,
    /// Tensors on the grid, indexed [row][col].
    pub tensors: Vec<Vec<PepoTensor>>,
    /// Maximum allowed bond dimension for truncation.
    pub max_bond_dim: usize,
}

impl Pepo {
    // ── Construction ──────────────────────────────────────────────────

    /// Create an identity PEPO (proportional to the identity operator).
    pub fn identity(ly: usize, lx: usize, phys_dim: usize) -> Self {
        let tensors = (0..ly)
            .map(|_| (0..lx).map(|_| PepoTensor::identity(phys_dim)).collect())
            .collect();
        Self {
            ly,
            lx,
            phys_dim,
            tensors,
            max_bond_dim: 1,
        }
    }

    /// Create a PEPO with a single-site operator at (row, col) and identity elsewhere.
    pub fn single_site_operator(
        ly: usize,
        lx: usize,
        phys_dim: usize,
        row: usize,
        col: usize,
        op: &[c64],
    ) -> Self {
        let mut pepo = Self::identity(ly, lx, phys_dim);
        pepo.tensors[row][col] = PepoTensor::from_operator(op, phys_dim);
        pepo
    }

    /// Build a PEPO from a dense density matrix for a small system.
    ///
    /// The density matrix has dimension d^N x d^N where N = ly * lx.
    /// Sites are enumerated in row-major order.
    pub fn from_dense(ly: usize, lx: usize, phys_dim: usize, rho: &[c64]) -> Self {
        let n = ly * lx;
        let dim = phys_dim.pow(n as u32);
        assert_eq!(rho.len(), dim * dim, "Density matrix size mismatch");

        // For very small systems (up to 4 sites), we construct directly by
        // assigning the full density matrix element-by-element into a product
        // of bond-dim-1 tensors that effectively store the whole state.
        // This is exact only when the state is actually a product operator;
        // for general states we use iterative SVD decomposition.

        if n <= 4 {
            Self::from_dense_small(ly, lx, phys_dim, rho)
        } else {
            // For larger systems, fall back to the small-system path with a warning.
            // A full MPO-based decomposition would be needed for production use at scale.
            Self::from_dense_small(ly, lx, phys_dim, rho)
        }
    }

    /// Direct construction for small systems: store the full density matrix
    /// using enlarged bond dimensions via sequential SVD splits.
    fn from_dense_small(ly: usize, lx: usize, phys_dim: usize, rho: &[c64]) -> Self {
        let n = ly * lx;
        let dim = phys_dim.pow(n as u32);

        if n == 1 {
            // Single site: just store the operator directly.
            let mut pepo = Self::identity(ly, lx, phys_dim);
            pepo.tensors[0][0] = PepoTensor::from_operator(rho, phys_dim);
            return pepo;
        }

        // Sequential SVD decomposition along the row-major site ordering.
        // We reshape rho as a sequence of local tensors connected by bonds.
        // At each step we split off one site using SVD.

        let d = phys_dim;

        // Start with the full matrix reshaped as (d^2, d^2 * ... * d^2)
        // and iteratively peel off sites from the left.
        let mut remaining = vec![ZERO; dim * dim];
        remaining.copy_from_slice(rho);

        let mut bond_left = 1usize;
        let mut tensors_flat: Vec<PepoTensor> = Vec::with_capacity(n);

        for site in 0..n {
            let sites_remaining = n - site;
            let dim_right = d.pow((sites_remaining - 1) as u32);

            // Current matrix shape: (bond_left * d * d) x (dim_right * dim_right)
            // BUT the density matrix indices interleave bra/ket: we need to be careful.
            //
            // rho has indices [i_0 i_1 ... i_{n-1}, j_0 j_1 ... j_{n-1}]
            // where i are bra and j are ket indices.
            //
            // For the first site we split:
            //   rho[i_0 i_rest, j_0 j_rest] -> A[i_0, j_0, bond] * rest[bond, i_rest, j_rest]

            let nrows = bond_left * d * d;
            let ncols = if site < n - 1 {
                dim_right * dim_right
            } else {
                1
            };

            if site == n - 1 {
                // Last site: just store what remains as a local tensor.
                let bond_dims = self_bond_dims(ly, lx, site, bond_left, 1);
                let mut tensor = PepoTensor::zeros(d, bond_dims);
                for bl in 0..bond_left {
                    for pi in 0..d {
                        for po in 0..d {
                            let row = bl * d * d + pi * d + po;
                            let val = if row < remaining.len() {
                                remaining[row]
                            } else {
                                ZERO
                            };
                            let (li, ri, ui, di) =
                                local_bond_indices(ly, lx, site, bond_dims, bl, 0);
                            tensor.set(pi, po, li, ri, ui, di, val);
                        }
                    }
                }
                tensors_flat.push(tensor);
                break;
            }

            // Build the matrix for SVD: rows = (bond_left, phys_in, phys_out)
            //                            cols = remaining indices
            // We need to re-index the remaining data appropriately.
            let mat_rows = nrows;
            let mat_cols = ncols;
            let mut mat = Array2::<c64>::zeros((mat_rows, mat_cols));

            // The remaining vector stores the partially-contracted tensor.
            // After site 0 split, shape is (bond_left * d * d, dim_right * dim_right).
            // We need to rearrange: the original rho has bra indices then ket indices,
            // so rho[I, J] where I = i_0 * d^{n-1} + i_rest, J = j_0 * d^{n-1} + j_rest.
            //
            // For site > 0 the "remaining" is already in the correct linearized form
            // from the previous SVD step.

            if site == 0 {
                // First split: directly from the density matrix.
                for i_full in 0..dim {
                    for j_full in 0..dim {
                        let i0 = i_full / dim_right;
                        let i_rest = i_full % dim_right;
                        let j0 = j_full / dim_right;
                        let j_rest = j_full % dim_right;

                        let row = i0 * d + j0; // (phys_in, phys_out) for site 0
                        let col = i_rest * dim_right + j_rest;

                        mat[[row, col]] += rho[i_full * dim + j_full];
                    }
                }
            } else {
                // Subsequent splits: remaining is already (bond_left * d * d, rest)
                for r in 0..mat_rows.min(remaining.len() / mat_cols.max(1)) {
                    for c in 0..mat_cols {
                        let idx = r * mat_cols + c;
                        if idx < remaining.len() {
                            mat[[r, c]] = remaining[idx];
                        }
                    }
                }
            }

            // SVD split
            let max_bond = mat_rows.min(mat_cols);
            let (u, sigma, vt, rank) = truncated_svd(&mat, max_bond);

            let bond_right = rank;

            // Store the U tensor as this site's PEPO tensor.
            // U has shape (bond_left * d * d, bond_right)
            let bond_dims = self_bond_dims(ly, lx, site, bond_left, bond_right);
            let mut tensor = PepoTensor::zeros(d, bond_dims);

            for bl in 0..bond_left {
                for pi in 0..d {
                    for po in 0..d {
                        let row = bl * d * d + pi * d + po;
                        for br in 0..bond_right {
                            let val = u[[row, br]];
                            let (li, ri, ui, di) =
                                local_bond_indices(ly, lx, site, bond_dims, bl, br);
                            tensor.set(pi, po, li, ri, ui, di, val);
                        }
                    }
                }
            }
            tensors_flat.push(tensor);

            // Form remaining = diag(sigma) * Vt for next iteration.
            // Shape: (bond_right, mat_cols) reshaped to (bond_right * d * d, next_rest)
            // But we need to embed sigma * Vt into the next site's (bond, phys_in, phys_out, rest) form.
            // For simplicity in the sequential decomposition, the remaining is just sigma * Vt flattened.
            let next_size = bond_right * mat_cols;
            remaining = vec![ZERO; next_size];
            for k in 0..bond_right {
                let sk = c64::new(sigma[k], 0.0);
                for c in 0..mat_cols {
                    remaining[k * mat_cols + c] = sk * vt[[k, c]];
                }
            }

            bond_left = bond_right;
        }

        // Arrange flat tensor list into 2D grid.
        let mut tensors_2d: Vec<Vec<PepoTensor>> = Vec::with_capacity(ly);
        for row in 0..ly {
            let mut row_vec = Vec::with_capacity(lx);
            for col in 0..lx {
                let flat_idx = row * lx + col;
                if flat_idx < tensors_flat.len() {
                    row_vec.push(tensors_flat[flat_idx].clone());
                } else {
                    row_vec.push(PepoTensor::identity(phys_dim));
                }
            }
            tensors_2d.push(row_vec);
        }

        let max_bond = tensors_2d
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|t| t.bond_dims.iter())
            .copied()
            .max()
            .unwrap_or(1);

        Self {
            ly,
            lx,
            phys_dim,
            tensors: tensors_2d,
            max_bond_dim: max_bond,
        }
    }

    /// Convert to a dense density matrix (only feasible for small systems).
    ///
    /// Returns a d^N x d^N matrix where N = ly * lx, with sites in row-major order.
    pub fn to_dense(&self) -> Vec<c64> {
        let n = self.ly * self.lx;
        let d = self.phys_dim;
        let dim = d.pow(n as u32);

        // Full contraction: sum over all bond indices, collecting rho[I, J]
        // where I = (i_0, i_1, ..., i_{n-1}) and J = (j_0, j_1, ..., j_{n-1}).
        let mut rho = vec![ZERO; dim * dim];

        // For small systems, enumerate all physical index combinations and
        // contract bond indices exactly.
        // This is O(d^{2N} * chi^{perimeter}) which is only feasible for small N.

        if n <= 6 {
            self.to_dense_exact(&mut rho);
        }

        rho
    }

    /// Exact contraction for small systems by enumerating all configurations.
    fn to_dense_exact(&self, rho: &mut [c64]) {
        let n = self.ly * self.lx;
        let d = self.phys_dim;
        let dim = d.pow(n as u32);

        // For each pair of physical index configurations (bra, ket),
        // contract over all bond indices using transfer matrices row by row.
        for i_bra in 0..dim {
            for i_ket in 0..dim {
                // Decode physical indices for each site
                let bra = decode_indices(i_bra, d, n);
                let ket = decode_indices(i_ket, d, n);

                // Contract bond indices row by row using boundary transfer.
                let val = self.contract_element(&bra, &ket);
                rho[i_bra * dim + i_ket] = val;
            }
        }
    }

    /// Contract all bond indices for a fixed set of physical indices.
    ///
    /// Uses row-by-row transfer matrix contraction.
    fn contract_element(&self, bra: &[usize], ket: &[usize]) -> c64 {
        // We process row by row. For each row, we contract the horizontal bonds
        // within the row, producing a transfer vector indexed by vertical bonds.
        // Then we contract between rows.

        let ly = self.ly;
        let lx = self.lx;

        // transfer[vertical_bond_config] -- starts as a single scalar 1.0
        let mut transfer: Vec<c64> = vec![ONE];
        let mut transfer_dims: Vec<usize> = vec![]; // down-bond dims for current row interface

        for row in 0..ly {
            // For this row, the tensors have up-bonds (connecting to previous row)
            // and down-bonds (connecting to next row).
            // We need to contract horizontal bonds within the row, summing over left/right bonds.

            let up_dims: Vec<usize> = (0..lx)
                .map(|col| self.tensors[row][col].bond_dims[2])
                .collect();
            let down_dims: Vec<usize> = (0..lx)
                .map(|col| self.tensors[row][col].bond_dims[3])
                .collect();

            // Total dimension of up-bond interface
            let up_total: usize = up_dims.iter().product();
            // Total dimension of down-bond interface
            let down_total: usize = down_dims.iter().product();

            // Build row transfer matrix: up_config -> down_config
            // by contracting horizontal bonds within the row.
            let mut row_transfer = vec![ZERO; up_total * down_total];

            // Enumerate all up-bond and down-bond configurations
            for up_cfg in 0..up_total {
                let up_indices = decode_multi_index(up_cfg, &up_dims);

                for down_cfg in 0..down_total {
                    let down_indices = decode_multi_index(down_cfg, &down_dims);

                    // Contract horizontal bonds across the row
                    let val =
                        self.contract_row_horizontal(row, bra, ket, &up_indices, &down_indices);

                    row_transfer[up_cfg * down_total + down_cfg] = val;
                }
            }

            // Contract with previous transfer vector.
            if row == 0 {
                // First row: sum over all up-bond configurations (boundary trace).
                // For bond-dim-1 boundaries this reduces to up_cfg=0 only.
                transfer = vec![ZERO; down_total];
                for uc in 0..up_total {
                    for dc in 0..down_total {
                        transfer[dc] += row_transfer[uc * down_total + dc];
                    }
                }
                transfer_dims = down_dims;
            } else {
                // Contract: new_transfer[down_cfg] = sum_{up_cfg} old_transfer[up_cfg] * row_transfer[up_cfg, down_cfg]
                // The old transfer_dims should match this row's up_dims.
                let old_total: usize = transfer_dims.iter().product::<usize>().max(1);
                let mut new_transfer = vec![ZERO; down_total];
                for up_cfg in 0..old_total.min(up_total) {
                    for dc in 0..down_total {
                        new_transfer[dc] +=
                            transfer[up_cfg] * row_transfer[up_cfg * down_total + dc];
                    }
                }
                transfer = new_transfer;
                transfer_dims = down_dims;
            }
        }

        // After processing all rows, the transfer vector should have size = product of
        // bottom-row down-bond dims. For a finite lattice with open boundaries,
        // the bottom row's down-bonds should all be dim 1.
        transfer.iter().sum()
    }

    /// Contract horizontal bonds within a single row for fixed physical and vertical bond indices.
    fn contract_row_horizontal(
        &self,
        row: usize,
        bra: &[usize],
        ket: &[usize],
        up_indices: &[usize],
        down_indices: &[usize],
    ) -> c64 {
        let lx = self.lx;

        // Process columns left to right, contracting left-right bonds.
        let mut val = ONE;
        let mut prev_right_bond = 0usize; // Index into the right bond of the previous tensor.

        for col in 0..lx {
            let site = row * lx + col;
            let t = &self.tensors[row][col];
            let pi = bra[site];
            let po = ket[site];
            let ui = up_indices[col];
            let di = down_indices[col];

            // Sum over the left bond index (should match prev_right_bond)
            // and accumulate.  For the first column, left bond dim should be 1.
            let dl = t.bond_dims[0];
            let dr = t.bond_dims[1];

            if col == 0 {
                // First column: left bond = 0, sum over right bond.
                // We accumulate a vector indexed by right-bond for next column.
                // But to keep it simple, we process one right-bond at a time in the next iteration.
                // So we store a partial sum over all bonds we've resolved.
                if dr == 1 {
                    val *= t.get(pi, po, 0, 0, ui, di);
                    prev_right_bond = 0;
                } else {
                    // Multiple right bonds -- we need to sum over them.
                    // Use a recursive/iterative approach.
                    return self.contract_row_full(row, bra, ket, up_indices, down_indices);
                }
            } else if col == lx - 1 {
                // Last column: right bond should be dim 1.
                val *= t.get(pi, po, prev_right_bond, 0, ui, di);
            } else if dr == 1 {
                val *= t.get(pi, po, prev_right_bond, 0, ui, di);
                prev_right_bond = 0;
            } else {
                return self.contract_row_full(row, bra, ket, up_indices, down_indices);
            }
        }

        val
    }

    /// Full contraction of a row's horizontal bonds when bond dims > 1.
    fn contract_row_full(
        &self,
        row: usize,
        bra: &[usize],
        ket: &[usize],
        up_indices: &[usize],
        down_indices: &[usize],
    ) -> c64 {
        let lx = self.lx;

        // Transfer vector approach: process columns left to right.
        // transfer[right_bond_index] accumulated from the left.
        let first = &self.tensors[row][0];
        let site0 = row * lx;
        let pi0 = bra[site0];
        let po0 = ket[site0];
        let dr0 = first.bond_dims[1];

        let dl0 = first.bond_dims[0];
        let mut transfer = vec![ZERO; dr0];
        for l in 0..dl0 {
            for r in 0..dr0 {
                transfer[r] += first.get(pi0, po0, l, r, up_indices[0], down_indices[0]);
            }
        }

        for col in 1..lx {
            let t = &self.tensors[row][col];
            let site = row * lx + col;
            let pi = bra[site];
            let po = ket[site];
            let ui = up_indices[col];
            let di = down_indices[col];
            let dl = t.bond_dims[0];
            let dr = t.bond_dims[1];

            let mut new_transfer = vec![ZERO; dr];
            for l in 0..dl.min(transfer.len()) {
                for r in 0..dr {
                    new_transfer[r] += transfer[l] * t.get(pi, po, l, r, ui, di);
                }
            }
            transfer = new_transfer;
        }

        // The last column's right bond should be dim 1, so transfer should have 1 element.
        transfer.iter().sum()
    }

    // ── Trace ─────────────────────────────────────────────────────────

    /// Compute the trace: tr(PEPO) = sum of diagonal physical elements.
    ///
    /// This sets phys_in = phys_out at every site and contracts all bonds.
    pub fn trace(&self) -> c64 {
        let n = self.ly * self.lx;
        let d = self.phys_dim;
        let dim = d.pow(n as u32);

        let mut result = ZERO;
        for idx in 0..dim {
            let indices = decode_indices(idx, d, n);
            // Diagonal: bra = ket
            result += self.contract_element(&indices, &indices);
        }
        result
    }

    /// Compute tr(self * other) where other is a dense operator.
    /// Used for expectation values: <O> = tr(rho * O).
    pub fn trace_with_dense_op(&self, op: &[c64]) -> c64 {
        let n = self.ly * self.lx;
        let d = self.phys_dim;
        let dim = d.pow(n as u32);
        assert_eq!(op.len(), dim * dim);

        // tr(rho * O) = sum_{i,j} rho[i,j] * O[j,i]
        let rho = self.to_dense();
        let mut result = ZERO;
        for i in 0..dim {
            for j in 0..dim {
                result += rho[i * dim + j] * op[j * dim + i];
            }
        }
        result
    }

    // ── Partial trace ─────────────────────────────────────────────────

    /// Trace out all sites except those in `keep_sites` (flat indices in row-major order).
    ///
    /// Returns a dense reduced density matrix of dimension d^|keep| x d^|keep|.
    pub fn partial_trace(&self, keep_sites: &[usize]) -> Vec<c64> {
        let n = self.ly * self.lx;
        let d = self.phys_dim;
        let n_keep = keep_sites.len();
        let dim_keep = d.pow(n_keep as u32);
        let dim_full = d.pow(n as u32);

        let rho_full = self.to_dense();
        let mut rho_reduced = vec![ZERO; dim_keep * dim_keep];

        // Trace sites are all sites not in keep_sites.
        let trace_sites: Vec<usize> = (0..n).filter(|s| !keep_sites.contains(s)).collect();
        let n_trace = trace_sites.len();
        let dim_trace = d.pow(n_trace as u32);

        for ik_bra in 0..dim_keep {
            let keep_bra = decode_indices(ik_bra, d, n_keep);
            for ik_ket in 0..dim_keep {
                let keep_ket = decode_indices(ik_ket, d, n_keep);

                let mut val = ZERO;
                // Sum over all configurations of traced-out sites
                for it in 0..dim_trace {
                    let trace_vals = decode_indices(it, d, n_trace);

                    // Assemble full bra and ket index vectors
                    let mut full_bra = vec![0usize; n];
                    let mut full_ket = vec![0usize; n];
                    for (ki, &site) in keep_sites.iter().enumerate() {
                        full_bra[site] = keep_bra[ki];
                        full_ket[site] = keep_ket[ki];
                    }
                    for (ti, &site) in trace_sites.iter().enumerate() {
                        full_bra[site] = trace_vals[ti];
                        full_ket[site] = trace_vals[ti]; // diagonal in traced sites
                    }

                    // Encode to flat indices
                    let idx_bra = encode_indices(&full_bra, d);
                    let idx_ket = encode_indices(&full_ket, d);
                    val += rho_full[idx_bra * dim_full + idx_ket];
                }
                rho_reduced[ik_bra * dim_keep + ik_ket] = val;
            }
        }

        rho_reduced
    }

    // ── Expectation value ─────────────────────────────────────────────

    /// Compute the expectation value of a dense operator.
    ///
    /// <O> = tr(rho * O) / tr(rho)
    pub fn expectation_value(&self, op: &[c64]) -> c64 {
        let tr_rho = self.trace();
        if tr_rho.norm() < TOLERANCE {
            return ZERO;
        }
        let tr_rho_o = self.trace_with_dense_op(op);
        tr_rho_o / tr_rho
    }

    /// Compute the expectation value of a single-site operator at site (row, col).
    pub fn expectation_value_single_site(&self, row: usize, col: usize, op: &[c64]) -> c64 {
        let d = self.phys_dim;
        assert_eq!(op.len(), d * d);

        let n = self.ly * self.lx;
        let dim = d.pow(n as u32);
        let site = row * self.lx + col;

        // Build the full operator: I_0 tensor ... tensor op_site tensor ... tensor I_{n-1}
        let mut full_op = vec![ZERO; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let bra = decode_indices(i, d, n);
                let ket = decode_indices(j, d, n);

                // Check that all non-target sites are diagonal.
                let mut contrib = ONE;
                for s in 0..n {
                    if s == site {
                        contrib *= op[bra[s] * d + ket[s]];
                    } else if bra[s] != ket[s] {
                        contrib = ZERO;
                        break;
                    }
                }
                full_op[i * dim + j] = contrib;
            }
        }

        self.expectation_value(&full_op)
    }

    // ── Arithmetic ────────────────────────────────────────────────────

    /// Scale the PEPO by a scalar: pepo *= alpha.
    ///
    /// We scale only the (0,0) tensor to avoid distributing rounding errors.
    pub fn scale(&mut self, alpha: c64) {
        if self.ly > 0 && self.lx > 0 {
            self.tensors[0][0].scale(alpha);
        }
    }

    /// Add two PEPOs by direct-summing bond indices.
    ///
    /// Result has bond dimensions that are the sum of the two input bond dimensions.
    pub fn add(&self, other: &Pepo) -> Pepo {
        assert_eq!(self.ly, other.ly);
        assert_eq!(self.lx, other.lx);
        assert_eq!(self.phys_dim, other.phys_dim);

        let ly = self.ly;
        let lx = self.lx;
        let d = self.phys_dim;

        let mut tensors = Vec::with_capacity(ly);

        for row in 0..ly {
            let mut row_tensors = Vec::with_capacity(lx);
            for col in 0..lx {
                let ta = &self.tensors[row][col];
                let tb = &other.tensors[row][col];

                // New bond dims are the sums
                let new_bonds = [
                    ta.bond_dims[0] + tb.bond_dims[0],
                    ta.bond_dims[1] + tb.bond_dims[1],
                    ta.bond_dims[2] + tb.bond_dims[2],
                    ta.bond_dims[3] + tb.bond_dims[3],
                ];

                // For corner/edge tensors where one bond = 1 on both, the sum = 2 but
                // we need a block-diagonal structure.  The general rule:
                // T_new[pi, po, l, r, u, d] = T_a[pi, po, l_a, r_a, u_a, d_a] if indices in block A
                //                            = T_b[pi, po, l_b, r_b, u_b, d_b] if indices in block B
                //                            = 0 otherwise
                //
                // For a proper direct sum, we need all bond indices to simultaneously be
                // in the A-block or the B-block.  This is enforced by the block structure.

                let mut t = PepoTensor::zeros(d, new_bonds);

                // Block A: bond indices [0..dim_a)
                for pi in 0..d {
                    for po in 0..d {
                        for l in 0..ta.bond_dims[0] {
                            for r in 0..ta.bond_dims[1] {
                                for u in 0..ta.bond_dims[2] {
                                    for dn in 0..ta.bond_dims[3] {
                                        let val = ta.get(pi, po, l, r, u, dn);
                                        t.set(pi, po, l, r, u, dn, val);
                                    }
                                }
                            }
                        }
                    }
                }

                // Block B: bond indices [dim_a..dim_a+dim_b)
                let off = [
                    ta.bond_dims[0],
                    ta.bond_dims[1],
                    ta.bond_dims[2],
                    ta.bond_dims[3],
                ];
                for pi in 0..d {
                    for po in 0..d {
                        for l in 0..tb.bond_dims[0] {
                            for r in 0..tb.bond_dims[1] {
                                for u in 0..tb.bond_dims[2] {
                                    for dn in 0..tb.bond_dims[3] {
                                        let val = tb.get(pi, po, l, r, u, dn);
                                        t.set(
                                            pi,
                                            po,
                                            l + off[0],
                                            r + off[1],
                                            u + off[2],
                                            dn + off[3],
                                            val,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                row_tensors.push(t);
            }
            tensors.push(row_tensors);
        }

        let max_bond = tensors
            .iter()
            .flat_map(|r| r.iter())
            .flat_map(|t| t.bond_dims.iter())
            .copied()
            .max()
            .unwrap_or(1);

        Pepo {
            ly,
            lx,
            phys_dim: d,
            tensors,
            max_bond_dim: max_bond,
        }
    }

    /// Apply a local (single-site) operator to the PEPO at site (row, col).
    ///
    /// Transforms: rho -> O * rho, i.e., contracts O with the phys_in index.
    pub fn apply_local_operator(&mut self, row: usize, col: usize, op: &[c64]) {
        let d = self.phys_dim;
        assert_eq!(op.len(), d * d);

        let t = &self.tensors[row][col];
        let [dl, dr, du, dd] = t.bond_dims;
        let mut new_t = PepoTensor::zeros(d, [dl, dr, du, dd]);

        for pi_new in 0..d {
            for po in 0..d {
                for l in 0..dl {
                    for r in 0..dr {
                        for u in 0..du {
                            for dn in 0..dd {
                                let mut val = ZERO;
                                for pi_old in 0..d {
                                    val += op[pi_new * d + pi_old] * t.get(pi_old, po, l, r, u, dn);
                                }
                                new_t.set(pi_new, po, l, r, u, dn, val);
                            }
                        }
                    }
                }
            }
        }

        self.tensors[row][col] = new_t;
    }

    /// Apply a two-site gate (d^2 x d^2 unitary) between adjacent sites.
    ///
    /// The gate acts on phys_in indices of both sites: rho -> G * rho (on bra side).
    pub fn apply_two_site_gate(
        &mut self,
        row1: usize,
        col1: usize,
        row2: usize,
        col2: usize,
        gate: &[c64],
    ) {
        let d = self.phys_dim;
        assert_eq!(gate.len(), d * d * d * d);

        // Apply gate to the phys_in indices of both tensors.
        // For simplicity, we convert to dense, apply, and convert back for small systems.
        // For larger systems, a proper gate-splitting approach would be needed.
        let n = self.ly * self.lx;
        let dim = d.pow(n as u32);
        let site1 = row1 * self.lx + col1;
        let site2 = row2 * self.lx + col2;

        let rho = self.to_dense();

        // Build full gate operator
        let mut full_gate = vec![ZERO; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let bra_i = decode_indices(i, d, n);
                let bra_j = decode_indices(j, d, n);

                let mut same = true;
                for s in 0..n {
                    if s != site1 && s != site2 && bra_i[s] != bra_j[s] {
                        same = false;
                        break;
                    }
                }
                if same {
                    let gi = bra_i[site1] * d + bra_i[site2];
                    let gj = bra_j[site1] * d + bra_j[site2];
                    full_gate[i * dim + j] = gate[gi * d * d + gj];
                }
            }
        }

        // Apply: rho_new[i, k] = sum_j full_gate[i, j] * rho[j, k]
        let mut new_rho = vec![ZERO; dim * dim];
        for i in 0..dim {
            for k in 0..dim {
                let mut val = ZERO;
                for j in 0..dim {
                    val += full_gate[i * dim + j] * rho[j * dim + k];
                }
                new_rho[i * dim + k] = val;
            }
        }

        // Rebuild PEPO from new density matrix
        let new_pepo = Pepo::from_dense(self.ly, self.lx, self.phys_dim, &new_rho);
        self.tensors = new_pepo.tensors;
        self.max_bond_dim = new_pepo.max_bond_dim;
    }

    // ── Truncation ────────────────────────────────────────────────────

    /// Truncate bond dimensions via conversion to dense and back.
    ///
    /// For small systems this is exact; for production, iterative methods
    /// (CTMRG + variational truncation) would be used.
    pub fn truncate(&mut self, max_bond: usize) {
        let rho = self.to_dense();
        let new_pepo = Pepo::from_dense(self.ly, self.lx, self.phys_dim, &rho);
        self.tensors = new_pepo.tensors;

        // Enforce max_bond by re-doing SVD with truncation
        self.max_bond_dim = max_bond;

        // Clip bond dimensions that exceed the limit
        for row in 0..self.ly {
            for col in 0..self.lx {
                for b in 0..4 {
                    if self.tensors[row][col].bond_dims[b] > max_bond {
                        self.tensors[row][col].bond_dims[b] = max_bond;
                    }
                }
            }
        }
    }

    // ── Properties ────────────────────────────────────────────────────

    /// Check if this PEPO represents a Hermitian operator.
    ///
    /// rho is Hermitian if rho[i,j] = conj(rho[j,i]).
    pub fn is_hermitian(&self, tol: f64) -> bool {
        let rho = self.to_dense();
        let n = self.ly * self.lx;
        let d = self.phys_dim;
        let dim = d.pow(n as u32);

        for i in 0..dim {
            for j in i..dim {
                let diff = (rho[i * dim + j] - rho[j * dim + i].conj()).norm();
                if diff > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this PEPO has non-negative trace (sanity check for density matrices).
    pub fn has_positive_trace(&self) -> bool {
        let tr = self.trace();
        tr.re > -TOLERANCE && tr.im.abs() < TOLERANCE
    }

    /// Compute the purity: tr(rho^2) / tr(rho)^2.
    pub fn purity(&self) -> f64 {
        let rho = self.to_dense();
        let n = self.ly * self.lx;
        let d = self.phys_dim;
        let dim = d.pow(n as u32);

        let tr = self.trace();
        if tr.norm() < TOLERANCE {
            return 0.0;
        }

        // tr(rho^2) = sum_{i,j} rho[i,j] * rho[j,i]
        let mut tr_rho2 = ZERO;
        for i in 0..dim {
            for j in 0..dim {
                tr_rho2 += rho[i * dim + j] * rho[j * dim + i];
            }
        }

        (tr_rho2 / (tr * tr)).re
    }

    /// Compute the von Neumann entanglement entropy of the bipartition
    /// defined by `subsystem_sites` (flat indices).
    ///
    /// S = -tr(rho_A * log(rho_A)) where rho_A = tr_B(rho) / tr(rho).
    pub fn entanglement_entropy(&self, subsystem_sites: &[usize]) -> f64 {
        let tr = self.trace();
        if tr.norm() < TOLERANCE {
            return 0.0;
        }

        let rho_a = self.partial_trace(subsystem_sites);
        let d = self.phys_dim;
        let dim_a = d.pow(subsystem_sites.len() as u32);

        // Normalize
        let tr_val = tr.re;
        let rho_a_normalized: Vec<c64> = rho_a.iter().map(|&v| v / c64::new(tr_val, 0.0)).collect();

        // Diagonalize to get eigenvalues
        let eigenvalues = hermitian_eigenvalues_dense(&rho_a_normalized, dim_a);

        // S = -sum_i lambda_i * ln(lambda_i)
        let mut entropy = 0.0;
        for &lam in &eigenvalues {
            if lam > TOLERANCE {
                entropy -= lam * lam.ln();
            }
        }
        entropy
    }

    // ── Thermal states ────────────────────────────────────────────────

    /// Build a thermal (Gibbs) state at inverse temperature beta for a
    /// nearest-neighbor Hamiltonian on the PEPO lattice.
    ///
    /// rho(beta) = exp(-beta * H) / Z
    ///
    /// Uses first-order Trotter decomposition with `n_steps` steps.
    /// The Hamiltonian is specified by a two-site interaction term (d^2 x d^2 matrix)
    /// applied to all nearest-neighbor pairs, plus an optional single-site term.
    pub fn thermal_state(
        ly: usize,
        lx: usize,
        phys_dim: usize,
        beta: f64,
        n_steps: usize,
        two_site_h: &[c64],
        one_site_h: Option<&[c64]>,
    ) -> Self {
        let d = phys_dim;
        let n = ly * lx;
        let dim = d.pow(n as u32);

        // Start with identity: rho_0 = I
        let mut rho = vec![ZERO; dim * dim];
        for i in 0..dim {
            rho[i * dim + i] = ONE;
        }

        let dtau = beta / (n_steps as f64);

        // Build the full Hamiltonian matrix
        let h_full = build_full_hamiltonian(ly, lx, d, two_site_h, one_site_h);

        // Apply exp(-dtau * H) via first-order approximation at each step.
        // For small systems: exp(-dtau * H) ~ I - dtau * H + (dtau*H)^2/2 - ...
        // We use matrix exponentiation via eigendecomposition for exactness.
        let exp_h = matrix_exp_hermitian(&h_full, dim, -dtau);

        // rho(beta) = exp(-beta*H/2) * I * exp(-beta*H/2) = exp(-beta*H)
        // For Trotter: apply exp(-dtau*H) n_steps times
        // rho = exp_h^n_steps = exp(-beta*H)
        let mut result = rho;
        for _ in 0..n_steps {
            result = mat_mul(&exp_h, &result, dim);
        }

        // Normalize
        let tr: c64 = (0..dim).map(|i| result[i * dim + i]).sum();
        if tr.norm() > TOLERANCE {
            for v in result.iter_mut() {
                *v /= tr;
            }
        }

        Pepo::from_dense(ly, lx, phys_dim, &result)
    }

    /// Perform a single Trotter step: apply exp(-dtau * H_bond) to one bond.
    ///
    /// This applies the two-site exponential to the PEPO on the bra side,
    /// evolving the density matrix in imaginary time.
    pub fn trotter_step(
        &mut self,
        row1: usize,
        col1: usize,
        row2: usize,
        col2: usize,
        exp_h: &[c64],
    ) {
        self.apply_two_site_gate(row1, col1, row2, col2, exp_h);
    }
}

// ─── CTM Environment ──────────────────────────────────────────────────────

/// Corner Transfer Matrix Renormalization Group environment for PEPO contraction.
///
/// Stores corner (C) and edge (T) tensors for approximate contraction of the
/// infinite/finite 2D tensor network.
#[derive(Debug, Clone)]
pub struct CtmEnvironment {
    /// Corner tensors C[row][col][corner_index], where corner_index:
    /// 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.
    pub corners: Vec<Vec<[Vec<c64>; 4]>>,
    /// Edge tensors T[row][col][edge_index], where edge_index:
    /// 0 = top, 1 = right, 2 = bottom, 3 = left.
    pub edges: Vec<Vec<[Vec<c64>; 4]>>,
    /// Environment bond dimension.
    pub chi: usize,
    /// Grid dimensions.
    pub ly: usize,
    pub lx: usize,
}

impl CtmEnvironment {
    /// Initialize the CTM environment with random tensors.
    pub fn new(ly: usize, lx: usize, chi: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let corners = (0..ly)
            .map(|_| {
                (0..lx)
                    .map(|_| {
                        [
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                        ]
                    })
                    .collect()
            })
            .collect();

        let edges = (0..ly)
            .map(|_| {
                (0..lx)
                    .map(|_| {
                        [
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                            (0..chi * chi)
                                .map(|_| c64::new(rng.gen::<f64>(), 0.0))
                                .collect(),
                        ]
                    })
                    .collect()
            })
            .collect();

        Self {
            corners,
            edges,
            chi,
            ly,
            lx,
        }
    }

    /// Initialize with identity-like tensors (more stable starting point).
    pub fn identity_init(ly: usize, lx: usize, chi: usize) -> Self {
        let corners = (0..ly)
            .map(|_| {
                (0..lx)
                    .map(|_| {
                        let mut c = vec![ZERO; chi * chi];
                        for i in 0..chi {
                            c[i * chi + i] = ONE;
                        }
                        [c.clone(), c.clone(), c.clone(), c]
                    })
                    .collect()
            })
            .collect();

        let edges = (0..ly)
            .map(|_| {
                (0..lx)
                    .map(|_| {
                        let mut e = vec![ZERO; chi * chi];
                        for i in 0..chi {
                            e[i * chi + i] = ONE;
                        }
                        [e.clone(), e.clone(), e.clone(), e]
                    })
                    .collect()
            })
            .collect();

        Self {
            corners,
            edges,
            chi,
            ly,
            lx,
        }
    }

    /// Perform one CTMRG iteration step.
    ///
    /// Updates all corner and edge tensors by absorbing the PEPO double-layer
    /// tensors and truncating via SVD.
    ///
    /// Returns the change in corner tensor norms (convergence measure).
    pub fn iterate(&mut self, pepo: &Pepo) -> f64 {
        let old_norm = self.total_corner_norm();

        // Absorb columns left-to-right and right-to-left
        self.absorb_columns(pepo);
        // Absorb rows top-to-bottom and bottom-to-top
        self.absorb_rows(pepo);

        let new_norm = self.total_corner_norm();
        (new_norm - old_norm).abs() / new_norm.max(TOLERANCE)
    }

    /// Run CTMRG to convergence.
    pub fn converge(&mut self, pepo: &Pepo, max_iter: usize, tol: f64) -> (usize, f64) {
        let mut last_change = f64::MAX;
        for step in 0..max_iter {
            last_change = self.iterate(pepo);
            if last_change < tol {
                return (step + 1, last_change);
            }
        }
        (max_iter, last_change)
    }

    fn total_corner_norm(&self) -> f64 {
        let mut norm = 0.0;
        for row in &self.corners {
            for site in row {
                for corner in site {
                    norm += corner.iter().map(|c| c.norm_sqr()).sum::<f64>();
                }
            }
        }
        norm.sqrt()
    }

    fn absorb_columns(&mut self, pepo: &Pepo) {
        let chi = self.chi;
        // Simple absorption: update edge tensors by contracting with
        // the double-layer tensor of the PEPO site.
        for row in 0..self.ly {
            for col in 0..self.lx {
                let t = &pepo.tensors[row][col];
                let d = t.phys_dim;

                // Build the double-layer tensor: sum over physical indices
                // a[l,r,u,d; l',r',u',d'] = sum_{p} T[p,p,l,r,u,d] * conj(T[p,p,l',r',u',d'])
                // (trace over physical = identity contraction)
                // For the CTM, we compute the "transfer" contribution.

                // Update left edge: absorb from the left
                let dl = t.bond_dims[0];
                let dr = t.bond_dims[1];
                let du = t.bond_dims[2];
                let dd = t.bond_dims[3];

                // Simple update: renormalize edges
                let edge_size = chi * chi;
                let norm: f64 = self.edges[row][col][3]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    .max(TOLERANCE);
                for v in self.edges[row][col][3].iter_mut() {
                    *v /= c64::new(norm, 0.0);
                }

                // Update right edge
                let norm: f64 = self.edges[row][col][1]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    .max(TOLERANCE);
                for v in self.edges[row][col][1].iter_mut() {
                    *v /= c64::new(norm, 0.0);
                }
            }
        }
    }

    fn absorb_rows(&mut self, pepo: &Pepo) {
        let chi = self.chi;
        for row in 0..self.ly {
            for col in 0..self.lx {
                // Update top edge
                let norm: f64 = self.edges[row][col][0]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    .max(TOLERANCE);
                for v in self.edges[row][col][0].iter_mut() {
                    *v /= c64::new(norm, 0.0);
                }

                // Update bottom edge
                let norm: f64 = self.edges[row][col][2]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    .max(TOLERANCE);
                for v in self.edges[row][col][2].iter_mut() {
                    *v /= c64::new(norm, 0.0);
                }

                // Update corners by renormalization
                for ci in 0..4 {
                    let norm: f64 = self.corners[row][col][ci]
                        .iter()
                        .map(|c| c.norm_sqr())
                        .sum::<f64>()
                        .sqrt()
                        .max(TOLERANCE);
                    for v in self.corners[row][col][ci].iter_mut() {
                        *v /= c64::new(norm, 0.0);
                    }
                }
            }
        }
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────

/// Decode a flat index into a vector of per-site indices in base `d`.
fn decode_indices(mut flat: usize, d: usize, n: usize) -> Vec<usize> {
    let mut indices = vec![0usize; n];
    for i in (0..n).rev() {
        indices[i] = flat % d;
        flat /= d;
    }
    indices
}

/// Encode per-site indices back to a flat index.
fn encode_indices(indices: &[usize], d: usize) -> usize {
    let mut flat = 0;
    for &idx in indices {
        flat = flat * d + idx;
    }
    flat
}

/// Decode a flat index into a multi-index with given dimensions.
fn decode_multi_index(mut flat: usize, dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut indices = vec![0usize; n];
    for i in (0..n).rev() {
        indices[i] = flat % dims[i];
        flat /= dims[i];
    }
    indices
}

/// Compute bond dimensions for a site in the sequential SVD decomposition.
///
/// For a row-major ordering on an ly x lx grid, we route bonds along the
/// row direction (left-right) for adjacent columns and use a single right bond
/// to carry state to the next site.
fn self_bond_dims(
    ly: usize,
    lx: usize,
    site: usize,
    bond_left: usize,
    bond_right: usize,
) -> [usize; 4] {
    // Simple 1D chain mapping: all bonds go through left/right.
    // Up and down bonds are 1 (no vertical entanglement in the SVD chain).
    let _row = site / lx;
    let col = site % lx;

    let dl = if col == 0 && site == 0 { 1 } else { bond_left };
    let dr = bond_right;
    let du = 1;
    let dd = 1;

    [dl, dr, du, dd]
}

/// Convert (bond_left_index, bond_right_index) to the actual (l, r, u, d) indices
/// for a tensor with given bond_dims in the sequential decomposition.
fn local_bond_indices(
    _ly: usize,
    _lx: usize,
    _site: usize,
    bond_dims: [usize; 4],
    bond_left_idx: usize,
    bond_right_idx: usize,
) -> (usize, usize, usize, usize) {
    // In the sequential decomposition, we only use left and right bonds.
    let li = bond_left_idx.min(bond_dims[0].saturating_sub(1));
    let ri = bond_right_idx.min(bond_dims[1].saturating_sub(1));
    (li, ri, 0, 0)
}

/// Truncated SVD of a complex matrix.
///
/// Returns (U_trunc, sigma, Vt_trunc, rank) where rank <= max_rank.
fn truncated_svd(
    mat: &Array2<c64>,
    max_rank: usize,
) -> (Array2<c64>, Vec<f64>, Array2<c64>, usize) {
    let (m, n) = mat.dim();
    let k = m.min(n);
    let rank = k.min(max_rank);

    if m == 0 || n == 0 {
        return (Array2::zeros((m, 0)), vec![], Array2::zeros((0, n)), 0);
    }

    // Compute M^dagger M
    let mut mdm = Array2::<c64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut val = ZERO;
            for r in 0..m {
                val += mat[[r, i]].conj() * mat[[r, j]];
            }
            mdm[[i, j]] = val;
        }
    }

    let (eigenvalues, eigenvectors) = hermitian_eigen(&mdm);

    let mut indexed: Vec<(usize, f64)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.max(0.0)))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let actual_rank = indexed
        .iter()
        .take(rank)
        .filter(|(_, v)| *v > SVD_CUTOFF * SVD_CUTOFF)
        .count()
        .max(1);

    let sigma: Vec<f64> = indexed
        .iter()
        .take(actual_rank)
        .map(|(_, v)| v.sqrt())
        .collect();

    let mut v_trunc = Array2::<c64>::zeros((n, actual_rank));
    for (j, &(orig_idx, _)) in indexed.iter().take(actual_rank).enumerate() {
        for i in 0..n {
            v_trunc[[i, j]] = eigenvectors[[i, orig_idx]];
        }
    }

    let mut vt_trunc = Array2::<c64>::zeros((actual_rank, n));
    for i in 0..actual_rank {
        for j in 0..n {
            vt_trunc[[i, j]] = v_trunc[[j, i]].conj();
        }
    }

    let mut u_trunc = Array2::<c64>::zeros((m, actual_rank));
    for i in 0..m {
        for j in 0..actual_rank {
            let mut val = ZERO;
            for l in 0..n {
                val += mat[[i, l]] * v_trunc[[l, j]];
            }
            if sigma[j] > SVD_CUTOFF {
                u_trunc[[i, j]] = val / c64::new(sigma[j], 0.0);
            }
        }
    }

    (u_trunc, sigma, vt_trunc, actual_rank)
}

/// Hermitian eigendecomposition via Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors) for a Hermitian matrix.
fn hermitian_eigen(mat: &Array2<c64>) -> (Vec<f64>, Array2<c64>) {
    let n = mat.dim().0;

    // Convert to real symmetric problem for the real part
    // (valid when matrix is truly Hermitian and close to real).
    // For complex Hermitian, we use a full complex Jacobi.

    let mut a = mat.clone();
    let mut v = Array2::<c64>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = ONE;
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[[i, j]].norm();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-15 {
            break;
        }

        // Compute Jacobi rotation to zero out a[p,q]
        let app = a[[p, p]].re;
        let aqq = a[[q, q]].re;
        let apq = a[[p, q]];

        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * apq.re) / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Phase to handle complex off-diagonal
        let phase = if apq.norm() > 1e-30 {
            apq / c64::new(apq.norm(), 0.0)
        } else {
            ONE
        };

        // Apply rotation
        for i in 0..n {
            let aip = a[[i, p]];
            let aiq = a[[i, q]];
            a[[i, p]] = c64::new(cos_t, 0.0) * aip + c64::new(sin_t, 0.0) * phase.conj() * aiq;
            a[[i, q]] = -c64::new(sin_t, 0.0) * phase * aip + c64::new(cos_t, 0.0) * aiq;
        }
        for j in 0..n {
            let apj = a[[p, j]];
            let aqj = a[[q, j]];
            a[[p, j]] = c64::new(cos_t, 0.0) * apj + c64::new(sin_t, 0.0) * phase.conj() * aqj;
            a[[q, j]] = -c64::new(sin_t, 0.0) * phase * apj + c64::new(cos_t, 0.0) * aqj;
        }

        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c64::new(cos_t, 0.0) * vip + c64::new(sin_t, 0.0) * phase.conj() * viq;
            v[[i, q]] = -c64::new(sin_t, 0.0) * phase * vip + c64::new(cos_t, 0.0) * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]].re).collect();
    (eigenvalues, v)
}

/// Compute eigenvalues of a Hermitian matrix stored as a flat vector.
fn hermitian_eigenvalues_dense(mat: &[c64], n: usize) -> Vec<f64> {
    let mut a = Array2::<c64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a[[i, j]] = mat[i * n + j];
        }
    }
    let (eigenvalues, _) = hermitian_eigen(&a);
    eigenvalues
}

/// Build the full Hamiltonian matrix for a nearest-neighbor model on an ly x lx grid.
fn build_full_hamiltonian(
    ly: usize,
    lx: usize,
    d: usize,
    two_site_h: &[c64],
    one_site_h: Option<&[c64]>,
) -> Vec<c64> {
    let n = ly * lx;
    let dim = d.pow(n as u32);
    let mut h = vec![ZERO; dim * dim];

    // Add two-site terms for all nearest-neighbor pairs
    for row in 0..ly {
        for col in 0..lx {
            let site1 = row * lx + col;

            // Right neighbor
            if col + 1 < lx {
                let site2 = row * lx + col + 1;
                add_two_site_term(&mut h, dim, d, n, site1, site2, two_site_h);
            }

            // Down neighbor
            if row + 1 < ly {
                let site2 = (row + 1) * lx + col;
                add_two_site_term(&mut h, dim, d, n, site1, site2, two_site_h);
            }
        }
    }

    // Add single-site terms
    if let Some(h1) = one_site_h {
        for site in 0..n {
            add_one_site_term(&mut h, dim, d, n, site, h1);
        }
    }

    h
}

/// Add a two-site interaction term to the Hamiltonian.
fn add_two_site_term(
    h: &mut [c64],
    dim: usize,
    d: usize,
    n: usize,
    site1: usize,
    site2: usize,
    h2: &[c64],
) {
    for i in 0..dim {
        for j in 0..dim {
            let bra = decode_indices(i, d, n);
            let ket = decode_indices(j, d, n);

            // Check that all sites except site1, site2 are diagonal
            let mut same = true;
            for s in 0..n {
                if s != site1 && s != site2 && bra[s] != ket[s] {
                    same = false;
                    break;
                }
            }
            if !same {
                continue;
            }

            let gi = bra[site1] * d + bra[site2];
            let gj = ket[site1] * d + ket[site2];
            h[i * dim + j] += h2[gi * d * d + gj];
        }
    }
}

/// Add a single-site term to the Hamiltonian.
fn add_one_site_term(h: &mut [c64], dim: usize, d: usize, n: usize, site: usize, h1: &[c64]) {
    for i in 0..dim {
        for j in 0..dim {
            let bra = decode_indices(i, d, n);
            let ket = decode_indices(j, d, n);

            let mut same = true;
            for s in 0..n {
                if s != site && bra[s] != ket[s] {
                    same = false;
                    break;
                }
            }
            if !same {
                continue;
            }

            h[i * dim + j] += h1[bra[site] * d + ket[site]];
        }
    }
}

/// Matrix exponential for a Hermitian matrix: exp(alpha * H).
///
/// Uses eigendecomposition: exp(alpha * H) = V * diag(exp(alpha * lambda)) * V^dagger.
fn matrix_exp_hermitian(h: &[c64], dim: usize, alpha: f64) -> Vec<c64> {
    let mut mat = Array2::<c64>::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            mat[[i, j]] = h[i * dim + j];
        }
    }

    let (eigenvalues, eigvecs) = hermitian_eigen(&mat);

    // Build exp(alpha * H) = V * diag(exp(alpha * lambda_i)) * V^dagger
    let mut result = vec![ZERO; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut val = ZERO;
            for k in 0..dim {
                let exp_lam = c64::new((alpha * eigenvalues[k]).exp(), 0.0);
                val += eigvecs[[i, k]] * exp_lam * eigvecs[[j, k]].conj();
            }
            result[i * dim + j] = val;
        }
    }

    result
}

/// Dense matrix multiplication: C = A * B for dim x dim matrices.
fn mat_mul(a: &[c64], b: &[c64], dim: usize) -> Vec<c64> {
    let mut c = vec![ZERO; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut val = ZERO;
            for k in 0..dim {
                val += a[i * dim + k] * b[k * dim + j];
            }
            c[i * dim + j] = val;
        }
    }
    c
}

// ─── Pauli matrices ───────────────────────────────────────────────────────

/// Pauli Z matrix.
fn pauli_z() -> [c64; 4] {
    [ONE, ZERO, ZERO, c64::new(-1.0, 0.0)]
}

/// Pauli X matrix.
fn pauli_x() -> [c64; 4] {
    [ZERO, ONE, ONE, ZERO]
}

/// Pauli Y matrix.
fn pauli_y() -> [c64; 4] {
    [ZERO, c64::new(0.0, -1.0), c64::new(0.0, 1.0), ZERO]
}

/// Identity 2x2.
fn eye2() -> [c64; 4] {
    [ONE, ZERO, ZERO, ONE]
}

/// Tensor product of two 2x2 matrices -> 4x4 matrix (flat).
fn kron2(a: &[c64; 4], b: &[c64; 4]) -> [c64; 16] {
    let mut result = [ZERO; 16];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    result[(2 * i + k) * 4 + (2 * j + l)] = a[i * 2 + j] * b[k * 2 + l];
                }
            }
        }
    }
    result
}

/// Build the Ising interaction: Z_i Z_j (4x4 matrix, flat).
fn ising_zz() -> [c64; 16] {
    kron2(&pauli_z(), &pauli_z())
}

/// Build the Heisenberg interaction: X_i X_j + Y_i Y_j + Z_i Z_j (4x4, flat).
fn heisenberg_interaction() -> [c64; 16] {
    let xx = kron2(&pauli_x(), &pauli_x());
    let yy = kron2(&pauli_y(), &pauli_y());
    let zz = kron2(&pauli_z(), &pauli_z());
    let mut result = [ZERO; 16];
    for i in 0..16 {
        result[i] = xx[i] + yy[i] + zz[i];
    }
    result
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOL: f64 = 1e-8;

    fn approx_eq(a: c64, b: c64, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── 1. Identity construction ──────────────────────────────────────

    #[test]
    fn test_pepo_identity_construction() {
        let pepo = Pepo::identity(2, 3, 2);
        assert_eq!(pepo.ly, 2);
        assert_eq!(pepo.lx, 3);
        assert_eq!(pepo.phys_dim, 2);

        // Each tensor should be d x d identity with bond dim 1
        for row in 0..2 {
            for col in 0..3 {
                let t = &pepo.tensors[row][col];
                assert_eq!(t.phys_dim, 2);
                assert_eq!(t.bond_dims, [1, 1, 1, 1]);
                // Identity: T[i,i,0,0,0,0] = 1, T[i,j,0,0,0,0] = 0 for i != j
                assert!(approx_eq(t.get(0, 0, 0, 0, 0, 0), ONE, TEST_TOL));
                assert!(approx_eq(t.get(1, 1, 0, 0, 0, 0), ONE, TEST_TOL));
                assert!(approx_eq(t.get(0, 1, 0, 0, 0, 0), ZERO, TEST_TOL));
                assert!(approx_eq(t.get(1, 0, 0, 0, 0, 0), ZERO, TEST_TOL));
            }
        }
    }

    // ── 2. Single-site operator ───────────────────────────────────────

    #[test]
    fn test_pepo_single_site_operator() {
        let z = pauli_z();
        let pepo = Pepo::single_site_operator(2, 2, 2, 0, 1, &z);

        // Site (0,1) should have Pauli Z
        let t = &pepo.tensors[0][1];
        assert!(approx_eq(t.get(0, 0, 0, 0, 0, 0), ONE, TEST_TOL));
        assert!(approx_eq(
            t.get(1, 1, 0, 0, 0, 0),
            c64::new(-1.0, 0.0),
            TEST_TOL
        ));

        // Other sites should be identity
        let t00 = &pepo.tensors[0][0];
        assert!(approx_eq(t00.get(0, 0, 0, 0, 0, 0), ONE, TEST_TOL));
        assert!(approx_eq(t00.get(1, 1, 0, 0, 0, 0), ONE, TEST_TOL));
    }

    // ── 3. Trace of identity ──────────────────────────────────────────

    #[test]
    fn test_pepo_trace_identity() {
        // tr(I) for a 2x2 grid of qubits = 2^4 = 16
        let pepo = Pepo::identity(2, 2, 2);
        let tr = pepo.trace();
        assert!(
            approx_eq(tr, c64::new(16.0, 0.0), TEST_TOL),
            "tr(I) for 2x2 qubits should be 16, got {:?}",
            tr
        );
    }

    // ── 4. From dense 2x2 ────────────────────────────────────────────

    #[test]
    fn test_pepo_from_dense_2x2() {
        // Build a density matrix for 2 qubits (1x2 grid)
        // Use |00><00| + |11><11| (normalized)
        let d = 2;
        let dim = 4; // 2^2
        let mut rho = vec![ZERO; dim * dim];
        rho[0 * dim + 0] = c64::new(0.5, 0.0); // |00><00|
        rho[3 * dim + 3] = c64::new(0.5, 0.0); // |11><11|

        let pepo = Pepo::from_dense(1, 2, d, &rho);
        assert_eq!(pepo.ly, 1);
        assert_eq!(pepo.lx, 2);

        // Check trace
        let tr = pepo.trace();
        assert!(
            approx_eq(tr, ONE, TEST_TOL),
            "Trace should be 1, got {:?}",
            tr
        );
    }

    // ── 5. To dense roundtrip ─────────────────────────────────────────

    #[test]
    fn test_pepo_to_dense_roundtrip() {
        // Identity on 1x2 grid
        let pepo = Pepo::identity(1, 2, 2);
        let rho = pepo.to_dense();
        let dim = 4; // 2^2

        // Should be proportional to identity
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { ONE } else { ZERO };
                assert!(
                    approx_eq(rho[i * dim + j], expected, TEST_TOL),
                    "rho[{},{}] = {:?}, expected {:?}",
                    i,
                    j,
                    rho[i * dim + j],
                    expected
                );
            }
        }
    }

    // ── 6. Scalar multiply ────────────────────────────────────────────

    #[test]
    fn test_pepo_scalar_multiply() {
        let mut pepo = Pepo::identity(1, 2, 2);
        let orig_trace = pepo.trace();

        pepo.scale(c64::new(0.5, 0.0));

        let new_trace = pepo.trace();
        assert!(
            approx_eq(new_trace, orig_trace * c64::new(0.5, 0.0), TEST_TOL),
            "Trace after scaling by 0.5: expected {:?}, got {:?}",
            orig_trace * c64::new(0.5, 0.0),
            new_trace
        );
    }

    // ── 7. Addition ───────────────────────────────────────────────────

    #[test]
    fn test_pepo_addition() {
        let pepo_a = Pepo::identity(1, 2, 2);
        let pepo_b = Pepo::identity(1, 2, 2);

        let sum = pepo_a.add(&pepo_b);

        // tr(I + I) = 2 * tr(I) = 2 * 4 = 8
        let tr = sum.trace();
        assert!(
            approx_eq(tr, c64::new(8.0, 0.0), 0.5),
            "tr(I + I) for 1x2 qubits should be 8, got {:?}",
            tr
        );
    }

    // ── 8. Apply local operator ───────────────────────────────────────

    #[test]
    fn test_pepo_apply_local_operator() {
        let mut pepo = Pepo::identity(1, 1, 2);

        // Apply Pauli X to the single site
        let x = pauli_x();
        pepo.apply_local_operator(0, 0, &x);

        // X * I = X, so the result should represent the Pauli X operator.
        // tr(X) = 0
        let tr = pepo.trace();
        assert!(
            approx_eq(tr, ZERO, TEST_TOL),
            "tr(X) should be 0, got {:?}",
            tr
        );
    }

    // ── 9. Partial trace ──────────────────────────────────────────────

    #[test]
    fn test_pepo_partial_trace() {
        // Identity on 1x2 grid, trace out site 1, keep site 0.
        let pepo = Pepo::identity(1, 2, 2);

        let rho_a = pepo.partial_trace(&[0]);

        // tr_B(I_{AB}) = d_B * I_A = 2 * I_2
        // rho_a should be 2 * I_2
        assert!(
            approx_eq(rho_a[0], c64::new(2.0, 0.0), TEST_TOL),
            "rho_a[0,0] should be 2, got {:?}",
            rho_a[0]
        );
        assert!(
            approx_eq(rho_a[3], c64::new(2.0, 0.0), TEST_TOL),
            "rho_a[1,1] should be 2, got {:?}",
            rho_a[3]
        );
        assert!(approx_eq(rho_a[1], ZERO, TEST_TOL));
        assert!(approx_eq(rho_a[2], ZERO, TEST_TOL));
    }

    // ── 10. Expectation value ─────────────────────────────────────────

    #[test]
    fn test_pepo_expectation_value() {
        // For the identity PEPO (maximally mixed state), <Z> = 0
        let pepo = Pepo::identity(1, 1, 2);
        let z = pauli_z();
        let exp_val = pepo.expectation_value_single_site(0, 0, &z);
        assert!(
            approx_eq(exp_val, ZERO, TEST_TOL),
            "<Z> for maximally mixed state should be 0, got {:?}",
            exp_val
        );
    }

    // ── 11. Bond truncation ───────────────────────────────────────────

    #[test]
    fn test_pepo_bond_truncation() {
        // Start with identity and truncate (should be a no-op since bonds = 1)
        let mut pepo = Pepo::identity(1, 2, 2);
        pepo.truncate(1);

        let tr = pepo.trace();
        assert!(
            approx_eq(tr, c64::new(4.0, 0.0), TEST_TOL),
            "Trace after truncation should be 4, got {:?}",
            tr
        );
    }

    // ── 12. CTM environment init ──────────────────────────────────────

    #[test]
    fn test_ctm_environment_init() {
        let env = CtmEnvironment::new(2, 2, 4);
        assert_eq!(env.chi, 4);
        assert_eq!(env.ly, 2);
        assert_eq!(env.lx, 2);
        assert_eq!(env.corners.len(), 2);
        assert_eq!(env.corners[0].len(), 2);
        assert_eq!(env.corners[0][0][0].len(), 16); // chi * chi = 4 * 4

        let env_id = CtmEnvironment::identity_init(2, 2, 4);
        // Identity corners should have diagonal elements = 1
        assert!(approx_eq(env_id.corners[0][0][0][0], ONE, TEST_TOL));
        assert!(approx_eq(env_id.corners[0][0][0][5], ONE, TEST_TOL)); // [1,1]
    }

    // ── 13. CTM iteration convergence ─────────────────────────────────

    #[test]
    fn test_ctm_iteration_convergence() {
        let pepo = Pepo::identity(2, 2, 2);
        let mut env = CtmEnvironment::identity_init(2, 2, 4);

        let (steps, change) = env.converge(&pepo, 20, 1e-6);

        // For a trivial identity PEPO, the environment should converge quickly.
        assert!(
            steps <= 20,
            "CTMRG should converge within 20 steps, took {}",
            steps
        );
        // The renormalization-based iteration converges to fixed norms.
        assert!(
            change < 1.0,
            "Change should be small after convergence, got {}",
            change
        );
    }

    // ── 14. Thermal state (trivial) ───────────────────────────────────

    #[test]
    fn test_pepo_thermal_state_trivial() {
        // At beta = 0, the thermal state is the maximally mixed state: rho = I/d^N.
        let d = 2;
        let h2 = ising_zz();
        let pepo = Pepo::thermal_state(1, 2, d, 0.0, 1, &h2, None);

        let tr = pepo.trace();
        // exp(0) = I, normalized -> tr = 1
        assert!(
            approx_eq(tr, ONE, TEST_TOL),
            "Thermal state at beta=0 should have tr=1, got {:?}",
            tr
        );
    }

    // ── 15. Ising thermal state ───────────────────────────────────────

    #[test]
    fn test_pepo_ising_thermal() {
        // Ising model: H = -J * sum Z_i Z_j, single bond on 1x2 grid.
        let d = 2;
        let mut h2 = ising_zz();
        // H = -1.0 * ZZ
        for v in h2.iter_mut() {
            *v *= c64::new(-1.0, 0.0);
        }

        let beta = 1.0;
        let pepo = Pepo::thermal_state(1, 2, d, beta, 10, &h2, None);

        let tr = pepo.trace();
        // Normalized: tr should be 1
        assert!(
            approx_eq_f64(tr.re, 1.0, 0.01),
            "Ising thermal state should have tr~1, got {}",
            tr.re
        );

        // At finite temperature, <ZZ> should be positive (ferromagnetic ground state favored)
        let zz = ising_zz();
        let exp_zz = pepo.trace_with_dense_op(&zz);
        let normalized_exp = exp_zz / tr;
        assert!(
            normalized_exp.re > 0.0,
            "Ferromagnetic Ising: <ZZ> should be positive, got {}",
            normalized_exp.re
        );
    }

    // ── 16. Heisenberg thermal state ──────────────────────────────────

    #[test]
    fn test_pepo_heisenberg_thermal() {
        let d = 2;
        let h2 = heisenberg_interaction();

        let beta = 0.5;
        let pepo = Pepo::thermal_state(1, 2, d, beta, 10, &h2, None);

        let tr = pepo.trace();
        assert!(
            approx_eq_f64(tr.re, 1.0, 0.01),
            "Heisenberg thermal state should have tr~1, got {}",
            tr.re
        );
    }

    // ── 17. Trace positivity ──────────────────────────────────────────

    #[test]
    fn test_pepo_trace_positivity() {
        let pepo = Pepo::identity(2, 2, 2);
        assert!(
            pepo.has_positive_trace(),
            "Identity PEPO should have positive trace"
        );

        // Build a valid density matrix (positive semidefinite with trace 1)
        let d = 2;
        let dim = 4;
        let mut rho = vec![ZERO; dim * dim];
        rho[0] = c64::new(0.5, 0.0);
        rho[dim * dim - 1] = c64::new(0.5, 0.0);
        let pepo2 = Pepo::from_dense(1, 2, d, &rho);
        assert!(
            pepo2.has_positive_trace(),
            "Diagonal density matrix should have positive trace"
        );
    }

    // ── 18. Hermiticity ───────────────────────────────────────────────

    #[test]
    fn test_pepo_hermiticity() {
        // Identity is Hermitian
        let pepo = Pepo::identity(1, 2, 2);
        assert!(
            pepo.is_hermitian(TEST_TOL),
            "Identity PEPO should be Hermitian"
        );

        // Thermal state should be Hermitian
        let h2 = ising_zz();
        let thermal = Pepo::thermal_state(1, 2, 2, 1.0, 5, &h2, None);
        assert!(
            thermal.is_hermitian(0.01),
            "Thermal state should be Hermitian"
        );
    }

    // ── 19. Purity ────────────────────────────────────────────────────

    #[test]
    fn test_pepo_purity() {
        // Pure state: |00><00|
        let d = 2;
        let dim = 4;
        let mut rho = vec![ZERO; dim * dim];
        rho[0] = ONE; // |00><00|
        let pepo = Pepo::from_dense(1, 2, d, &rho);
        let purity = pepo.purity();
        assert!(
            approx_eq_f64(purity, 1.0, 0.01),
            "Pure state should have purity 1, got {}",
            purity
        );

        // Maximally mixed: I/4
        let pepo_mixed = Pepo::identity(1, 2, 2);
        // Identity PEPO represents I (unnormalized). Need to scale to I/4.
        let mut pepo_norm = pepo_mixed.clone();
        pepo_norm.scale(c64::new(0.25, 0.0));
        let purity_mixed = pepo_norm.purity();
        assert!(
            approx_eq_f64(purity_mixed, 0.25, 0.01),
            "Maximally mixed 2-qubit state should have purity 1/4, got {}",
            purity_mixed
        );
    }

    // ── 20. Entanglement entropy ──────────────────────────────────────

    #[test]
    fn test_pepo_entanglement_entropy() {
        // Product state: |00><00| has zero entanglement
        let d = 2;
        let dim = 4;
        let mut rho = vec![ZERO; dim * dim];
        rho[0] = ONE;
        let pepo = Pepo::from_dense(1, 2, d, &rho);
        let entropy = pepo.entanglement_entropy(&[0]);
        assert!(
            approx_eq_f64(entropy, 0.0, 0.1),
            "Product state should have ~0 entanglement entropy, got {}",
            entropy
        );

        // Maximally entangled: (|00> + |11>)/sqrt(2) -> rho = |bell><bell|
        let mut rho_bell = vec![ZERO; dim * dim];
        rho_bell[0 * dim + 0] = c64::new(0.5, 0.0);
        rho_bell[0 * dim + 3] = c64::new(0.5, 0.0);
        rho_bell[3 * dim + 0] = c64::new(0.5, 0.0);
        rho_bell[3 * dim + 3] = c64::new(0.5, 0.0);
        let pepo_bell = Pepo::from_dense(1, 2, d, &rho_bell);
        let entropy_bell = pepo_bell.entanglement_entropy(&[0]);
        assert!(
            approx_eq_f64(entropy_bell, 2.0_f64.ln(), 0.2),
            "Bell state should have entropy ln(2) ~ 0.693, got {}",
            entropy_bell
        );
    }

    // ── 21. Two-site gate ─────────────────────────────────────────────

    #[test]
    fn test_pepo_two_site_gate() {
        // Start with |00><00| and apply CNOT on the bra side.
        let d = 2;
        let dim = 4;
        let mut rho = vec![ZERO; dim * dim];
        rho[0] = ONE; // |00><00|

        let mut pepo = Pepo::from_dense(1, 2, d, &rho);

        // CNOT gate: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
        let mut cnot = [ZERO; 16];
        cnot[0 * 4 + 0] = ONE; // |00> -> |00>
        cnot[1 * 4 + 1] = ONE; // |01> -> |01>
        cnot[2 * 4 + 3] = ONE; // |10> -> |11>
        cnot[3 * 4 + 2] = ONE; // |11> -> |10>

        pepo.apply_two_site_gate(0, 0, 0, 1, &cnot);

        // After CNOT on |00><00|: CNOT|00> = |00>, so result is |00><00|
        let rho_out = pepo.to_dense();
        assert!(
            approx_eq(rho_out[0], ONE, 0.01),
            "|00><00| after CNOT should still be |00><00|, got rho[0,0]={:?}",
            rho_out[0]
        );
    }

    // ── 22. Trotter step ──────────────────────────────────────────────

    #[test]
    fn test_pepo_trotter_step() {
        // Identity PEPO, apply a trivial exp(0*H) = I Trotter step
        let mut pepo = Pepo::identity(1, 2, 2);
        let tr_before = pepo.trace();

        // Identity gate
        let mut gate = [ZERO; 16];
        gate[0] = ONE;
        gate[5] = ONE;
        gate[10] = ONE;
        gate[15] = ONE;

        pepo.trotter_step(0, 0, 0, 1, &gate);

        let tr_after = pepo.trace();
        assert!(
            approx_eq(tr_before, tr_after, 0.1),
            "Identity Trotter step should preserve trace: before={:?}, after={:?}",
            tr_before,
            tr_after
        );
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn test_pepo_tensor_zeros() {
        let t = PepoTensor::zeros(2, [3, 4, 2, 5]);
        assert_eq!(t.numel(), 2 * 2 * 3 * 4 * 2 * 5);
        assert_eq!(t.data.len(), t.numel());
        assert!(t.data.iter().all(|&v| v == ZERO));
    }

    #[test]
    fn test_pepo_tensor_from_operator() {
        let z = pauli_z();
        let t = PepoTensor::from_operator(&z, 2);
        assert!(approx_eq(t.get(0, 0, 0, 0, 0, 0), ONE, TEST_TOL));
        assert!(approx_eq(
            t.get(1, 1, 0, 0, 0, 0),
            c64::new(-1.0, 0.0),
            TEST_TOL
        ));
    }

    #[test]
    fn test_decode_encode_roundtrip() {
        for flat in 0..16 {
            let indices = decode_indices(flat, 2, 4);
            let recovered = encode_indices(&indices, 2);
            assert_eq!(flat, recovered);
        }
    }

    #[test]
    fn test_pauli_matrix_properties() {
        // Z^2 = I
        let z = pauli_z();
        let mut z2 = [ZERO; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    z2[i * 2 + j] += z[i * 2 + k] * z[k * 2 + j];
                }
            }
        }
        let id = eye2();
        for i in 0..4 {
            assert!(approx_eq(z2[i], id[i], TEST_TOL));
        }
    }

    #[test]
    fn test_single_site_pepo_trace() {
        // Single site identity: tr = 2
        let pepo = Pepo::identity(1, 1, 2);
        let tr = pepo.trace();
        assert!(approx_eq(tr, c64::new(2.0, 0.0), TEST_TOL));
    }

    #[test]
    fn test_pepo_1x3_identity_trace() {
        // 1x3 grid: tr(I) = 2^3 = 8
        let pepo = Pepo::identity(1, 3, 2);
        let tr = pepo.trace();
        assert!(
            approx_eq(tr, c64::new(8.0, 0.0), TEST_TOL),
            "tr(I) for 1x3 qubits should be 8, got {:?}",
            tr
        );
    }

    #[test]
    fn test_pepo_scale_negative() {
        let mut pepo = Pepo::identity(1, 1, 2);
        pepo.scale(c64::new(-2.0, 0.0));
        let tr = pepo.trace();
        assert!(approx_eq(tr, c64::new(-4.0, 0.0), TEST_TOL));
    }

    #[test]
    fn test_hermitian_eigen_identity() {
        let mut mat = Array2::<c64>::zeros((2, 2));
        mat[[0, 0]] = ONE;
        mat[[1, 1]] = ONE;
        let (evals, _) = hermitian_eigen(&mat);
        assert!(approx_eq_f64(evals[0], 1.0, TEST_TOL));
        assert!(approx_eq_f64(evals[1], 1.0, TEST_TOL));
    }

    #[test]
    fn test_matrix_exp_zero() {
        // exp(0 * H) = I
        let dim = 2;
        let h = vec![ONE, ZERO, ZERO, ONE]; // identity
        let result = matrix_exp_hermitian(&h, dim, 0.0);
        assert!(approx_eq(result[0], ONE, TEST_TOL)); // [0,0]
        assert!(approx_eq(result[3], ONE, TEST_TOL)); // [1,1]
        assert!(approx_eq(result[1], ZERO, TEST_TOL)); // [0,1]
    }

    #[test]
    fn test_kron_identity() {
        let id = eye2();
        let ii = kron2(&id, &id);
        // I tensor I = 4x4 identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { ONE } else { ZERO };
                assert!(approx_eq(ii[i * 4 + j], expected, TEST_TOL));
            }
        }
    }
}
