//! Advanced Tensor Network Methods
//!
//! Beyond MPS: PEPS, Tree Tensor Networks, and MERA for efficient quantum simulation.
//!
//! **Methods**:
//! - **PEPS** (Projected Entangled Pair States): 2D systems, area law entanglement
//! - **TTN** (Tree Tensor Networks): Hierarchical entanglement structure
//! - **MERA** (Multiscale Entanglement Renormalization Ansatz): Critical systems
//!
//! **Scaling**:
//! - MPS: O(poly(n)) for 1D systems
//! - PEPS: O(exp(√n)) for 2D systems (exponential in boundary)
//! - TTN: O(n log n) for tree-like entanglement
//! - MERA: O(n log n) with better constants for critical systems

use crate::{QuantumState, C64};
use std::collections::HashMap;

/// Tensor network type selector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorNetworkType {
    /// Matrix Product States (1D chain).
    MPS,
    /// Projected Entangled Pair States (2D grid).
    PEPS,
    /// Tree Tensor Networks (hierarchical).
    TTN,
    /// Multiscale Entanglement Renormalization Ansatz (critical).
    MERA,
}

impl TensorNetworkType {
    /// Select optimal network type based on circuit characteristics.
    pub fn select_optimal(
        num_qubits: usize,
        _depth: usize,
        is_2d: bool,
        is_critical: bool,
    ) -> Self {
        if is_2d && num_qubits > 16 {
            // 2D system with many qubits → PEPS
            TensorNetworkType::PEPS
        } else if is_critical {
            // Critical system → MERA
            TensorNetworkType::MERA
        } else if num_qubits > 32 {
            // Large system with hierarchical structure → TTN
            TensorNetworkType::TTN
        } else {
            // Default → MPS
            TensorNetworkType::MPS
        }
    }
}

/// PEPS (Projected Entangled Pair State) for 2D systems.
pub struct PEPSTensorNetwork {
    /// Grid dimensions (rows, cols).
    grid_size: (usize, usize),
    /// Physical bond dimension.
    physical_dim: usize,
    /// Virtual bond dimension.
    virtual_dim: usize,
    /// 2D grid of tensors.
    tensors: HashMap<(usize, usize), PEPTensor>,
}

/// PEPS tensor at a single site with proper 5-index structure.
/// Indices: [physical, up, down, left, right].
/// Boundary sites use dimension 1 for missing neighbors.
#[derive(Clone, Debug)]
pub struct PEPTensor {
    pub data: Vec<C64>,
    pub position: (usize, usize),
    pub phys: usize,
    pub up: usize,
    pub down: usize,
    pub left: usize,
    pub right: usize,
}

impl PEPTensor {
    fn new(
        pos: (usize, usize),
        phys: usize,
        up: usize,
        down: usize,
        left: usize,
        right: usize,
    ) -> Self {
        let size = phys * up * down * left * right;
        Self {
            data: vec![C64::new(0.0, 0.0); size],
            position: pos,
            phys,
            up,
            down,
            left,
            right,
        }
    }

    #[inline]
    fn idx(&self, p: usize, u: usize, d: usize, l: usize, r: usize) -> usize {
        ((((p * self.up + u) * self.down + d) * self.left + l) * self.right) + r
    }

    #[inline]
    fn get(&self, p: usize, u: usize, d: usize, l: usize, r: usize) -> C64 {
        self.data[self.idx(p, u, d, l, r)]
    }

    #[inline]
    fn set(&mut self, p: usize, u: usize, d: usize, l: usize, r: usize, val: C64) {
        let i = self.idx(p, u, d, l, r);
        self.data[i] = val;
    }

    fn virtual_size(&self) -> usize {
        self.up * self.down * self.left * self.right
    }
}

impl PEPSTensorNetwork {
    /// Create a new PEPS for a 2D grid, initialized to |0...0⟩.
    pub fn new(rows: usize, cols: usize, bond_dim: usize) -> Self {
        let mut tensors = HashMap::new();
        let d = bond_dim;

        for r in 0..rows {
            for c in 0..cols {
                let up = if r == 0 { 1 } else { d };
                let down = if r == rows - 1 { 1 } else { d };
                let left = if c == 0 { 1 } else { d };
                let right = if c == cols - 1 { 1 } else { d };

                let mut tensor = PEPTensor::new((r, c), 2, up, down, left, right);
                // |0⟩ state: physical=0, all virtual indices at 0
                tensor.set(0, 0, 0, 0, 0, C64::new(1.0, 0.0));
                tensors.insert((r, c), tensor);
            }
        }

        Self {
            grid_size: (rows, cols),
            physical_dim: 2,
            virtual_dim: bond_dim,
            tensors,
        }
    }

    /// Get grid dimensions.
    pub fn grid_size(&self) -> (usize, usize) {
        self.grid_size
    }

    /// Get total number of sites.
    pub fn num_sites(&self) -> usize {
        self.grid_size.0 * self.grid_size.1
    }

    /// Contract PEPS to state vector via row-by-row boundary contraction.
    ///
    /// For each basis state, fixes physical indices and contracts the 2D virtual
    /// tensor network row by row. Exact for small grids; complexity O(2^n * R * C * D^{2C+2}).
    pub fn contract_to_state(&self) -> Result<QuantumState, String> {
        let (rows, cols) = self.grid_size;
        let n = rows * cols;
        let dim = 1usize << n;
        let d = self.virtual_dim;

        // Boundary vector size: D^cols (one vertical bond per column)
        let bnd_size = d.pow(cols as u32);

        let mut state = QuantumState::new(n);
        let amps = state.amplitudes_mut();

        for basis in 0..dim {
            // Extract physical index for each site
            let phys: Vec<Vec<usize>> = (0..rows)
                .map(|r| (0..cols).map(|c| (basis >> (r * cols + c)) & 1).collect())
                .collect();

            // Row 0: build initial boundary vector indexed by down bonds
            let mut boundary = vec![C64::new(0.0, 0.0); bnd_size];
            // Contract row 0 left-to-right
            // partial[right_bond] after processing columns 0..c
            // then fold down bonds into multi-index
            self.contract_row_into_boundary(&phys[0], 0, &mut boundary);

            // Rows 1..R-1: absorb each row into boundary
            for r in 1..rows {
                let mut new_boundary = vec![C64::new(0.0, 0.0); bnd_size];
                self.absorb_row_into_boundary(r, &phys[r], &boundary, &mut new_boundary);
                boundary = new_boundary;
            }

            // Final amplitude: sum boundary (last row has down=1, so bnd_size should collapse)
            // For last row, down dims are 1, so boundary has size 1^cols = 1 effectively.
            // But we still store it in bnd_size format. The only nonzero entry is at index 0.
            amps[basis] = boundary[0];
        }

        Ok(state)
    }

    /// Contract a single row left-to-right with fixed physical indices,
    /// producing a boundary vector indexed by the down bonds.
    fn contract_row_into_boundary(&self, phys: &[usize], row: usize, boundary: &mut [C64]) {
        let cols = self.grid_size.1;
        let d = self.virtual_dim;

        // partial: maps (accumulated_down_multi_index, current_right_bond) -> amplitude
        // After processing column c, accumulated includes down bonds for columns 0..=c.
        // down_multi_index = d_0 * D^c + d_1 * D^(c-1) + ... + d_c
        let mut partial: HashMap<(usize, usize), C64> = HashMap::new();

        // Column 0
        let t = &self.tensors[&(row, 0)];
        let p = phys[0];
        for dd in 0..t.down {
            for rr in 0..t.right {
                let val = t.get(p, 0, dd, 0, rr); // up=0, left=0 (dim 1)
                if val.norm_sqr() > 1e-30 {
                    partial.insert((dd, rr), val);
                }
            }
        }

        // Columns 1..C-1
        for c in 1..cols {
            let t = &self.tensors[&(row, c)];
            let p = phys[c];
            let mut new_partial: HashMap<(usize, usize), C64> = HashMap::new();

            for (&(down_idx, right_prev), &amp) in &partial {
                for dd in 0..t.down {
                    // new down multi-index: old * D + new_down
                    let new_down_idx = down_idx * d.max(t.down) + dd;
                    for rr in 0..t.right {
                        // Contract over left bond = right_prev
                        let val = t.get(p, 0, dd, right_prev, rr);
                        if val.norm_sqr() > 1e-30 {
                            let key = (new_down_idx, rr);
                            *new_partial.entry(key).or_insert(C64::new(0.0, 0.0)) += amp * val;
                        }
                    }
                }
            }
            partial = new_partial;
        }

        // Write to boundary: last column has right=1, so right_bond = 0
        for (&(down_idx, _right), &amp) in &partial {
            if down_idx < boundary.len() {
                boundary[down_idx] += amp;
            }
        }
    }

    /// Absorb a row into the boundary vector.
    /// old_boundary is indexed by up bonds (= previous row's down bonds).
    /// new_boundary is indexed by this row's down bonds.
    fn absorb_row_into_boundary(
        &self,
        row: usize,
        phys: &[usize],
        old_boundary: &[C64],
        new_boundary: &mut [C64],
    ) {
        let cols = self.grid_size.1;
        let d = self.virtual_dim;
        let bnd_size = old_boundary.len(); // D^cols

        // For each old_boundary configuration (up bonds for this row)
        for up_config in 0..bnd_size {
            let bnd_amp = old_boundary[up_config];
            if bnd_amp.norm_sqr() < 1e-30 {
                continue;
            }

            // Decode up_config into per-column up indices
            let mut up_indices = vec![0usize; cols];
            let mut tmp = up_config;
            for c in (0..cols).rev() {
                let t = &self.tensors[&(row, c)];
                up_indices[c] = tmp % t.up;
                tmp /= t.up.max(1);
            }

            // Contract this row left-to-right with these up indices
            let mut partial: HashMap<(usize, usize), C64> = HashMap::new();

            // Column 0
            let t = &self.tensors[&(row, 0)];
            let p = phys[0];
            let u = up_indices[0];
            for dd in 0..t.down {
                for rr in 0..t.right {
                    let val = t.get(p, u, dd, 0, rr);
                    if val.norm_sqr() > 1e-30 {
                        partial.insert((dd, rr), val);
                    }
                }
            }

            // Columns 1..C-1
            for c in 1..cols {
                let t = &self.tensors[&(row, c)];
                let p = phys[c];
                let u = up_indices[c];
                let mut new_partial: HashMap<(usize, usize), C64> = HashMap::new();

                for (&(down_idx, right_prev), &amp) in &partial {
                    for dd in 0..t.down {
                        let new_down_idx = down_idx * d.max(t.down) + dd;
                        for rr in 0..t.right {
                            let val = t.get(p, u, dd, right_prev, rr);
                            if val.norm_sqr() > 1e-30 {
                                let key = (new_down_idx, rr);
                                *new_partial.entry(key).or_insert(C64::new(0.0, 0.0)) += amp * val;
                            }
                        }
                    }
                }
                partial = new_partial;
            }

            // Accumulate into new_boundary
            for (&(down_idx, _right), &amp) in &partial {
                if down_idx < new_boundary.len() {
                    new_boundary[down_idx] += bnd_amp * amp;
                }
            }
        }
    }

    /// Apply single-qubit gate at site (row, col).
    /// T'[p'][u][d][l][r] = sum_p gate[p'][p] * T[p][u][d][l][r]
    pub fn apply_single_qubit_gate(
        &mut self,
        row: usize,
        col: usize,
        matrix: [[C64; 2]; 2],
    ) -> Result<(), String> {
        let tensor = self
            .tensors
            .get_mut(&(row, col))
            .ok_or_else(|| format!("No tensor at ({}, {})", row, col))?;

        for u in 0..tensor.up {
            for d in 0..tensor.down {
                for l in 0..tensor.left {
                    for r in 0..tensor.right {
                        let v0 = tensor.get(0, u, d, l, r);
                        let v1 = tensor.get(1, u, d, l, r);
                        tensor.set(0, u, d, l, r, matrix[0][0] * v0 + matrix[0][1] * v1);
                        tensor.set(1, u, d, l, r, matrix[1][0] * v0 + matrix[1][1] * v1);
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply two-qubit gate to adjacent sites.
    /// Contracts tensors over shared bond, applies gate, splits via SVD.
    pub fn apply_two_qubit_gate(
        &mut self,
        row1: usize,
        col1: usize,
        row2: usize,
        col2: usize,
        matrix: [[C64; 4]; 4],
    ) -> Result<(), String> {
        let dr = (row1 as i32 - row2 as i32).abs();
        let dc = (col1 as i32 - col2 as i32).abs();
        if dr + dc != 1 {
            return Err("Can only apply two-qubit gate to neighbors in PEPS".to_string());
        }

        // Determine bond direction
        let horizontal = row1 == row2;
        let t1 = self.tensors[&(row1, col1)].clone();
        let t2 = self.tensors[&(row2, col2)].clone();

        if horizontal {
            // Shared bond: t1.right = t2.left (dimension D)
            self.apply_horizontal_gate(row1, col1, col2, &t1, &t2, &matrix)?;
        } else {
            // Shared bond: t1.down = t2.up (dimension D)
            self.apply_vertical_gate(col1, row1, row2, &t1, &t2, &matrix)?;
        }
        Ok(())
    }

    fn apply_horizontal_gate(
        &mut self,
        row: usize,
        col1: usize,
        col2: usize,
        t1: &PEPTensor,
        t2: &PEPTensor,
        gate: &[[C64; 4]; 4],
    ) -> Result<(), String> {
        use nalgebra::{Complex as NComplex, DMatrix};

        let shared = t1.right; // = t2.left
                               // Theta[p1, u1, d1, l1, p2, u2, d2, r2] = sum_s T1[p1,u1,d1,l1,s]*T2[p2,u2,d2,s,r2]
        let dim_left = 2 * t1.up * t1.down * t1.left;
        let dim_right = 2 * t2.up * t2.down * t2.right;

        // Contract and apply gate, then form matrix for SVD
        let mut theta = vec![C64::new(0.0, 0.0); dim_left * dim_right];

        for p1 in 0..2 {
            for u1 in 0..t1.up {
                for d1 in 0..t1.down {
                    for l1 in 0..t1.left {
                        let row_idx = ((p1 * t1.up + u1) * t1.down + d1) * t1.left + l1;
                        for p2 in 0..2 {
                            for u2 in 0..t2.up {
                                for d2 in 0..t2.down {
                                    for r2 in 0..t2.right {
                                        let col_idx =
                                            ((p2 * t2.up + u2) * t2.down + d2) * t2.right + r2;
                                        // Contract over shared bond
                                        let mut val = C64::new(0.0, 0.0);
                                        for s in 0..shared {
                                            val += t1.get(p1, u1, d1, l1, s)
                                                * t2.get(p2, u2, d2, s, r2);
                                        }
                                        // Apply gate
                                        for gp1 in 0..2 {
                                            for gp2 in 0..2 {
                                                if p1 == gp1 && p2 == gp2 {
                                                    // This basis maps through gate
                                                }
                                            }
                                        }
                                        theta[row_idx * dim_right + col_idx] += val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply the gate on physical indices: rearrange so we can apply 4x4 gate
        let other1 = t1.up * t1.down * t1.left;
        let other2 = t2.up * t2.down * t2.right;
        let mut gated = vec![C64::new(0.0, 0.0); dim_left * dim_right];

        for o1 in 0..other1 {
            for o2 in 0..other2 {
                // Extract 2x2 block for this (o1, o2) pair
                let mut block = [[C64::new(0.0, 0.0); 2]; 2];
                for p1 in 0..2 {
                    for p2 in 0..2 {
                        let ri = p1 * other1 + o1;
                        let ci = p2 * other2 + o2;
                        block[p1][p2] = theta[ri * dim_right + ci];
                    }
                }
                // Apply gate
                for p1 in 0..2 {
                    for p2 in 0..2 {
                        let mut val = C64::new(0.0, 0.0);
                        for q1 in 0..2 {
                            for q2 in 0..2 {
                                val += gate[p1 * 2 + p2][q1 * 2 + q2] * block[q1][q2];
                            }
                        }
                        let ri = p1 * other1 + o1;
                        let ci = p2 * other2 + o2;
                        gated[ri * dim_right + ci] = val;
                    }
                }
            }
        }

        // SVD to split back
        let m = dim_left;
        let n = dim_right;
        let mut mat_data = vec![NComplex::<f64>::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let v = gated[i * n + j];
                mat_data[i + j * m] = NComplex::new(v.re, v.im); // column-major
            }
        }
        let mat = DMatrix::from_vec(m, n, mat_data);
        let svd = mat.svd(true, true);
        let u_full = svd.u.expect("SVD U");
        let vt_full = svd.v_t.expect("SVD V^T");
        let svals = &svd.singular_values;

        let chi_new = svals
            .iter()
            .filter(|&&s| s > 1e-12)
            .count()
            .max(1)
            .min(self.virtual_dim);

        // Rebuild tensors with new bond dimension
        let mut new_t1 = PEPTensor::new(t1.position, 2, t1.up, t1.down, t1.left, chi_new);
        let mut new_t2 = PEPTensor::new(t2.position, 2, t2.up, t2.down, chi_new, t2.right);

        for p1 in 0..2 {
            for u1 in 0..t1.up {
                for d1 in 0..t1.down {
                    for l1 in 0..t1.left {
                        let ri = ((p1 * t1.up + u1) * t1.down + d1) * t1.left + l1;
                        for k in 0..chi_new {
                            let u_val = u_full[(ri, k)];
                            let s = svals[k];
                            let val = C64::new(u_val.re * s.sqrt(), u_val.im * s.sqrt());
                            new_t1.set(p1, u1, d1, l1, k, val);
                        }
                    }
                }
            }
        }

        for p2 in 0..2 {
            for u2 in 0..t2.up {
                for d2 in 0..t2.down {
                    for r2 in 0..t2.right {
                        let ci = ((p2 * t2.up + u2) * t2.down + d2) * t2.right + r2;
                        for k in 0..chi_new {
                            let vt_val = vt_full[(k, ci)];
                            let s = svals[k];
                            let val = C64::new(vt_val.re * s.sqrt(), vt_val.im * s.sqrt());
                            new_t2.set(p2, u2, d2, k, r2, val);
                        }
                    }
                }
            }
        }

        self.tensors.insert((row, col1), new_t1);
        self.tensors.insert((row, col2), new_t2);
        Ok(())
    }

    fn apply_vertical_gate(
        &mut self,
        col: usize,
        row1: usize,
        row2: usize,
        t1: &PEPTensor,
        t2: &PEPTensor,
        gate: &[[C64; 4]; 4],
    ) -> Result<(), String> {
        use nalgebra::{Complex as NComplex, DMatrix};

        let shared = t1.down; // = t2.up
        let other1 = t1.up * t1.left * t1.right;
        let other2 = t2.down * t2.left * t2.right;
        let dim_left = 2 * other1;
        let dim_right = 2 * other2;

        // Contract over shared vertical bond and apply gate
        let mut gated = vec![C64::new(0.0, 0.0); dim_left * dim_right];

        for o1_u in 0..t1.up {
            for o1_l in 0..t1.left {
                for o1_r in 0..t1.right {
                    let o1 = (o1_u * t1.left + o1_l) * t1.right + o1_r;
                    for o2_d in 0..t2.down {
                        for o2_l in 0..t2.left {
                            for o2_r in 0..t2.right {
                                let o2 = (o2_d * t2.left + o2_l) * t2.right + o2_r;
                                // Contract then apply gate
                                let mut block = [[C64::new(0.0, 0.0); 2]; 2];
                                for p1 in 0..2 {
                                    for p2 in 0..2 {
                                        for s in 0..shared {
                                            block[p1][p2] += t1.get(p1, o1_u, s, o1_l, o1_r)
                                                * t2.get(p2, s, o2_d, o2_l, o2_r);
                                        }
                                    }
                                }
                                for p1 in 0..2 {
                                    for p2 in 0..2 {
                                        let mut val = C64::new(0.0, 0.0);
                                        for q1 in 0..2 {
                                            for q2 in 0..2 {
                                                val +=
                                                    gate[p1 * 2 + p2][q1 * 2 + q2] * block[q1][q2];
                                            }
                                        }
                                        let ri = p1 * other1 + o1;
                                        let ci = p2 * other2 + o2;
                                        gated[ri * dim_right + ci] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // SVD
        let m = dim_left;
        let n = dim_right;
        let mut mat_data = vec![NComplex::<f64>::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let v = gated[i * n + j];
                mat_data[i + j * m] = NComplex::new(v.re, v.im);
            }
        }
        let mat = DMatrix::from_vec(m, n, mat_data);
        let svd = mat.svd(true, true);
        let u_full = svd.u.expect("SVD U");
        let vt_full = svd.v_t.expect("SVD V^T");
        let svals = &svd.singular_values;

        let chi_new = svals
            .iter()
            .filter(|&&s| s > 1e-12)
            .count()
            .max(1)
            .min(self.virtual_dim);

        // Rebuild: t1's down becomes chi_new, t2's up becomes chi_new
        let mut new_t1 = PEPTensor::new(t1.position, 2, t1.up, chi_new, t1.left, t1.right);
        let mut new_t2 = PEPTensor::new(t2.position, 2, chi_new, t2.down, t2.left, t2.right);

        for p1 in 0..2 {
            for o1_u in 0..t1.up {
                for o1_l in 0..t1.left {
                    for o1_r in 0..t1.right {
                        let o1 = (o1_u * t1.left + o1_l) * t1.right + o1_r;
                        let ri = p1 * other1 + o1;
                        for k in 0..chi_new {
                            let u_val = u_full[(ri, k)];
                            let s = svals[k];
                            new_t1.set(
                                p1,
                                o1_u,
                                k,
                                o1_l,
                                o1_r,
                                C64::new(u_val.re * s.sqrt(), u_val.im * s.sqrt()),
                            );
                        }
                    }
                }
            }
        }

        for p2 in 0..2 {
            for o2_d in 0..t2.down {
                for o2_l in 0..t2.left {
                    for o2_r in 0..t2.right {
                        let o2 = (o2_d * t2.left + o2_l) * t2.right + o2_r;
                        let ci = p2 * other2 + o2;
                        for k in 0..chi_new {
                            let vt_val = vt_full[(k, ci)];
                            let s = svals[k];
                            new_t2.set(
                                p2,
                                k,
                                o2_d,
                                o2_l,
                                o2_r,
                                C64::new(vt_val.re * s.sqrt(), vt_val.im * s.sqrt()),
                            );
                        }
                    }
                }
            }
        }

        self.tensors.insert((row1, col), new_t1);
        self.tensors.insert((row2, col), new_t2);
        Ok(())
    }

    /// Estimate entanglement entropy of PEPS.
    pub fn entanglement_entropy(&self) -> f64 {
        let perimeter = 2.0 * (self.grid_size.0 + self.grid_size.1) as f64;
        let virtual_dim = self.virtual_dim as f64;
        perimeter * virtual_dim.ln()
    }
}

/// Tree Tensor Network (TTN) for hierarchical entanglement.
///
/// Binary tree of isometry tensors. Leaves are physical qubits (2D vectors).
/// Internal nodes are isometries mapping child bonds to parent bond.
/// Contraction: bottom-up, leaves return state vectors, internal nodes
/// contract children through isometry.
pub struct TreeTensorNetwork {
    root: TTNNode,
    num_leaves: usize,
    bond_dim: usize,
}

/// Node in TTN tree.
#[derive(Clone, Debug)]
enum TTNNode {
    /// Leaf node: 2-element vector [|0⟩, |1⟩].
    Leaf {
        tensor: Vec<C64>, // length = physical_dim (2)
    },
    /// Internal node: isometry tensor.
    /// Shape: [parent_bond, left_child_bond, right_child_bond].
    /// Stored flat: data[p * left_dim * right_dim + l * right_dim + r].
    Internal {
        data: Vec<C64>,
        parent_dim: usize,
        left_dim: usize,
        right_dim: usize,
        children: Vec<Box<TTNNode>>,
    },
}

impl TreeTensorNetwork {
    /// Create a new TTN with balanced binary tree, initialized to |0...0⟩.
    pub fn new_balanced(num_qubits: usize, bond_dim: usize) -> Self {
        let root = Self::build_balanced_tree(num_qubits, bond_dim);
        Self {
            root,
            num_leaves: num_qubits,
            bond_dim,
        }
    }

    fn build_balanced_tree(num_qubits: usize, bond_dim: usize) -> TTNNode {
        if num_qubits == 1 {
            TTNNode::Leaf {
                tensor: vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)], // |0⟩
            }
        } else {
            let left_q = num_qubits / 2;
            let right_q = num_qubits - left_q;
            let left = Box::new(Self::build_balanced_tree(left_q, bond_dim));
            let right = Box::new(Self::build_balanced_tree(right_q, bond_dim));

            // Child output dimensions
            let left_dim = Self::node_output_dim(&left, bond_dim);
            let right_dim = Self::node_output_dim(&right, bond_dim);
            // Parent bond = min(bond_dim, left_dim * right_dim) for the root or intermediates
            let parent_dim = if num_qubits == left_q + right_q && left_q + right_q > 2 {
                bond_dim.min(left_dim * right_dim)
            } else {
                left_dim * right_dim // no truncation at top
            };

            // Initialize isometry as identity-like: W[p, l, r] = delta(p, l*right_dim + r)
            // Maps product state of children to parent bond.
            let mut data = vec![C64::new(0.0, 0.0); parent_dim * left_dim * right_dim];
            for l in 0..left_dim {
                for r in 0..right_dim {
                    let combined = l * right_dim + r;
                    if combined < parent_dim {
                        data[combined * left_dim * right_dim + l * right_dim + r] =
                            C64::new(1.0, 0.0);
                    }
                }
            }

            TTNNode::Internal {
                data,
                parent_dim,
                left_dim,
                right_dim,
                children: vec![left, right],
            }
        }
    }

    fn node_output_dim(node: &TTNNode, _bond_dim: usize) -> usize {
        match node {
            TTNNode::Leaf { tensor } => tensor.len(), // 2
            TTNNode::Internal { parent_dim, .. } => *parent_dim,
        }
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Contract TTN bottom-up to produce state vector.
    ///
    /// Leaves return their 2-element vectors. Internal nodes contract:
    /// result[p] = sum_{l,r} W[p,l,r] * left_vec[l] * right_vec[r]
    /// Root's output is the full state vector (reshaped from bond indices).
    pub fn contract_to_state(&self) -> Result<QuantumState, String> {
        let result = self.contract_node(&self.root);
        let n = self.num_leaves;
        let dim = 1usize << n;
        let mut state = QuantumState::new(n);
        let amps = state.amplitudes_mut();
        for (i, &v) in result.iter().enumerate() {
            if i < dim {
                amps[i] = v;
            }
        }
        Ok(state)
    }

    /// Bottom-up recursive contraction. Returns the output vector for this subtree.
    fn contract_node(&self, node: &TTNNode) -> Vec<C64> {
        match node {
            TTNNode::Leaf { tensor } => tensor.clone(),
            TTNNode::Internal {
                data,
                parent_dim,
                left_dim,
                right_dim,
                children,
            } => {
                assert_eq!(children.len(), 2);
                let left_vec = self.contract_node(&children[0]);
                let right_vec = self.contract_node(&children[1]);

                // result[p] = sum_{l,r} W[p,l,r] * left[l] * right[r]
                let mut result = vec![C64::new(0.0, 0.0); *parent_dim];
                for p in 0..*parent_dim {
                    for l in 0..*left_dim {
                        if left_vec[l].norm_sqr() < 1e-30 {
                            continue;
                        }
                        for r in 0..*right_dim {
                            let w = data[p * left_dim * right_dim + l * right_dim + r];
                            result[p] += w * left_vec[l] * right_vec[r];
                        }
                    }
                }
                result
            }
        }
    }

    fn child_leaf_count(node: &TTNNode) -> usize {
        match node {
            TTNNode::Leaf { .. } => 1,
            TTNNode::Internal { children, .. } => {
                children.iter().map(|c| Self::child_leaf_count(c)).sum()
            }
        }
    }

    pub fn contraction_complexity(&self) -> usize {
        self.num_leaves * self.bond_dim.pow(3)
    }
}

/// MERA (Multiscale Entanglement Renormalization Ansatz).
///
/// Each layer has disentanglers (2-qubit unitaries, 4x4) and isometries (2-to-1 maps).
/// Top-down contraction: start from top tensor, expand through isometries, apply disentanglers.
pub struct MERANetwork {
    layers: Vec<MERALayer>,
    /// Top tensor: state vector at coarsest scale.
    top_tensor: Vec<C64>,
    num_physical_qubits: usize,
    bond_dim: usize,
}

/// Single MERA layer.
#[derive(Clone, Debug)]
pub struct MERALayer {
    /// Disentanglers: 4x4 unitaries stored flat [d][row*4+col]. Applied on pairs of sites.
    pub disentanglers: Vec<[C64; 16]>,
    /// Isometries: map 1 coarse site to 2 fine sites. Shape [2*chi, chi] stored flat.
    /// iso[fine_pair * chi + coarse] where fine_pair indexes (2 sites).
    pub isometries: Vec<Vec<C64>>,
    /// Number of fine-grained sites at this layer.
    pub num_sites: usize,
}

impl MERANetwork {
    /// Create a new MERA initialized to |0...0⟩ state.
    pub fn new(num_physical_qubits: usize, num_layers: usize, bond_dim: usize) -> Self {
        let mut layers = Vec::new();
        let mut num_sites = num_physical_qubits;

        for _ in 0..num_layers {
            let num_dis = num_sites / 2;
            let num_iso = num_sites / 2;

            // Identity disentanglers
            let mut id4 = [C64::new(0.0, 0.0); 16];
            for i in 0..4 {
                id4[i * 4 + i] = C64::new(1.0, 0.0);
            }

            // Identity isometries: map coarse |i⟩ to fine |i,0⟩
            // iso[fine_pair * chi + coarse]: fine_pair = fine0 * 2 + fine1
            let chi = bond_dim.min(num_sites);
            let fine_pairs = 4; // 2 sites, each dim 2
            let mut iso = vec![C64::new(0.0, 0.0); fine_pairs * chi];
            // Map coarse index c to fine pair (c, 0) = c*2 + 0
            for c in 0..chi.min(2) {
                let fine_idx = c * 2; // fine0 = c, fine1 = 0
                iso[fine_idx * chi + c] = C64::new(1.0, 0.0);
            }

            layers.push(MERALayer {
                disentanglers: vec![id4; num_dis],
                isometries: vec![iso; num_iso],
                num_sites,
            });

            num_sites = (num_sites + 1) / 2;
        }

        // Top tensor: |0⟩ state at coarsest scale
        let top_dim = 1 << num_sites;
        let mut top_tensor = vec![C64::new(0.0, 0.0); top_dim];
        top_tensor[0] = C64::new(1.0, 0.0);

        Self {
            layers,
            top_tensor,
            num_physical_qubits,
            bond_dim,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn num_physical_qubits(&self) -> usize {
        self.num_physical_qubits
    }

    /// Contract MERA top-down to produce state vector.
    ///
    /// Start from top tensor, work down: for each layer (from coarsest to finest),
    /// expand through isometries (1 coarse site → 2 fine sites), then apply disentanglers.
    pub fn contract_to_state(&self) -> Result<QuantumState, String> {
        // Start with top tensor
        let mut state_vec = self.top_tensor.clone();
        let mut current_sites = self.top_tensor.len().trailing_zeros() as usize;
        if current_sites == 0 && self.top_tensor.len() > 1 {
            current_sites = (self.top_tensor.len() as f64).log2().ceil() as usize;
        }
        if current_sites == 0 {
            current_sites = 1;
        }

        // Process layers from top (coarsest) to bottom (finest)
        for layer in self.layers.iter().rev() {
            let _num_iso = layer.isometries.len();
            let target_sites = layer.num_sites;

            // Expand through isometries: each coarse site becomes 2 fine sites
            let new_dim = 1usize << target_sites;
            let mut new_vec = vec![C64::new(0.0, 0.0); new_dim];

            // For each amplitude in current state
            for (coarse_basis, &amp) in state_vec.iter().enumerate() {
                if amp.norm_sqr() < 1e-30 {
                    continue;
                }

                // Expand each coarse site through its isometry
                self.expand_through_isometries(
                    coarse_basis,
                    current_sites,
                    amp,
                    &layer.isometries,
                    target_sites,
                    &mut new_vec,
                );
            }

            // Apply disentanglers (2-qubit unitaries on adjacent pairs)
            let mut disent_vec = vec![C64::new(0.0, 0.0); new_dim];
            self.apply_disentanglers(
                &new_vec,
                &layer.disentanglers,
                target_sites,
                &mut disent_vec,
            );

            state_vec = disent_vec;
            current_sites = target_sites;
        }

        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let mut state = QuantumState::new(n);
        let amps = state.amplitudes_mut();
        for (i, &v) in state_vec.iter().enumerate() {
            if i < dim {
                amps[i] = v;
            }
        }
        Ok(state)
    }

    /// Expand a coarse basis state through isometries to fine basis states.
    fn expand_through_isometries(
        &self,
        coarse_basis: usize,
        coarse_sites: usize,
        amplitude: C64,
        isometries: &[Vec<C64>],
        fine_sites: usize,
        output: &mut [C64],
    ) {
        // Each coarse site at position s maps to 2 fine sites at 2s, 2s+1
        // Isometry: |fine0, fine1⟩ = sum_c iso[(fine0*2+fine1)*chi + c] |c⟩
        // We need to iterate over all fine configurations
        let num_iso = isometries.len().min(coarse_sites);
        let _fine_dim = 1usize << fine_sites;

        // Recursive expansion: for each isometry, expand one coarse site into 2 fine sites
        let mut partial = vec![(coarse_basis, amplitude)];

        for s in 0..num_iso {
            let iso = &isometries[s];
            let chi = if iso.len() >= 4 { iso.len() / 4 } else { 1 };
            let _coarse_bit = (coarse_basis >> s) & (chi - 1).max(1);

            let mut new_partial = Vec::new();
            for &(basis, amp) in &partial {
                // Extract the coarse index for site s
                let c = (basis >> s) & 1; // simplified: treat as qubit
                                          // Clear old bit, make room for 2 fine bits
                let base_cleared = basis & !(1 << s);
                // Shift upper bits to make room
                let lower = base_cleared & ((1 << s) - 1);
                let upper = (base_cleared >> s) << (s + 1); // shift by 1 extra for new fine site

                for f0 in 0..2usize {
                    for f1 in 0..2 {
                        let fine_pair = f0 * 2 + f1;
                        let iso_idx = fine_pair * chi.max(1) + c.min(chi.max(1) - 1);
                        if iso_idx < iso.len() {
                            let w = iso[iso_idx];
                            if w.norm_sqr() > 1e-30 {
                                let new_basis = lower | (f0 << s) | (f1 << (s + 1)) | (upper << 1);
                                new_partial.push((new_basis, amp * w));
                            }
                        }
                    }
                }
            }
            partial = new_partial;
        }

        for &(basis, amp) in &partial {
            if basis < output.len() {
                output[basis] += amp;
            }
        }
    }

    /// Apply disentanglers to adjacent pairs.
    fn apply_disentanglers(
        &self,
        input: &[C64],
        disentanglers: &[[C64; 16]],
        num_sites: usize,
        output: &mut [C64],
    ) {
        if disentanglers.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        // Apply each disentangler on sites (2*d, 2*d+1) as a 2-qubit gate
        let mut current = input.to_vec();
        for (d, gate) in disentanglers.iter().enumerate() {
            let q0 = 2 * d;
            let q1 = 2 * d + 1;
            if q1 >= num_sites {
                break;
            }

            let mut new_state = vec![C64::new(0.0, 0.0); current.len()];
            let dim = current.len();
            for basis in 0..dim {
                if current[basis].norm_sqr() < 1e-30 {
                    continue;
                }
                let b0 = (basis >> q0) & 1;
                let b1 = (basis >> q1) & 1;
                let in_idx = b0 * 2 + b1;

                for out0 in 0..2usize {
                    for out1 in 0..2 {
                        let out_idx = out0 * 2 + out1;
                        let g = gate[out_idx * 4 + in_idx];
                        if g.norm_sqr() > 1e-30 {
                            let new_basis =
                                (basis & !(1 << q0) & !(1 << q1)) | (out0 << q0) | (out1 << q1);
                            new_state[new_basis] += current[basis] * g;
                        }
                    }
                }
            }
            current = new_state;
        }

        output.copy_from_slice(&current);
    }

    /// Apply gate at a specific MERA layer and site.
    pub fn apply_gate_scale_invariant(
        &mut self,
        layer: usize,
        site: usize,
        matrix: [[C64; 4]; 4],
    ) -> Result<(), String> {
        if layer >= self.layers.len() {
            return Err("Layer index out of bounds".to_string());
        }
        let num_dis = self.layers[layer].disentanglers.len();
        if site >= num_dis {
            return Err("Site index out of bounds".to_string());
        }
        // Flatten gate into disentangler format
        let mut flat = [C64::new(0.0, 0.0); 16];
        for i in 0..4 {
            for j in 0..4 {
                flat[i * 4 + j] = matrix[i][j];
            }
        }
        self.layers[layer].disentanglers[site] = flat;
        Ok(())
    }

    pub fn contraction_cost(&self) -> usize {
        self.num_physical_qubits * self.layers.len() * self.bond_dim.pow(4)
    }
}

/// Auto-selecting tensor network simulator.
pub struct AutoTensorNetworkSimulator {
    network_type: TensorNetworkType,
    mps: Option<crate::tensor_network::MPS>,
    peps: Option<PEPSTensorNetwork>,
    ttn: Option<TreeTensorNetwork>,
    mera: Option<MERANetwork>,
}

impl AutoTensorNetworkSimulator {
    /// Create auto-selecting tensor network simulator.
    pub fn new(num_qubits: usize, is_2d: bool, is_critical: bool, bond_dim: usize) -> Self {
        let network_type = TensorNetworkType::select_optimal(num_qubits, 0, is_2d, is_critical);

        let (mps, peps, ttn, mera) = match network_type {
            TensorNetworkType::MPS => (
                Some(crate::tensor_network::MPS::new(num_qubits, Some(bond_dim))),
                None,
                None,
                None,
            ),
            TensorNetworkType::PEPS => {
                let rows = (num_qubits as f64).sqrt().ceil() as usize;
                let cols = (num_qubits + rows - 1) / rows;
                (
                    None,
                    Some(PEPSTensorNetwork::new(rows, cols, bond_dim)),
                    None,
                    None,
                )
            }
            TensorNetworkType::TTN => (
                None,
                None,
                Some(TreeTensorNetwork::new_balanced(num_qubits, bond_dim)),
                None,
            ),
            TensorNetworkType::MERA => {
                let layers = (num_qubits as f64).log2().ceil() as usize;
                (
                    None,
                    None,
                    None,
                    Some(MERANetwork::new(num_qubits, layers, bond_dim)),
                )
            }
        };

        Self {
            network_type,
            mps,
            peps,
            ttn,
            mera,
        }
    }

    /// Get selected network type.
    pub fn network_type(&self) -> TensorNetworkType {
        self.network_type
    }

    /// Contract to state vector.
    pub fn contract_to_state(&self) -> Result<QuantumState, String> {
        match self.network_type {
            TensorNetworkType::MPS => {
                let state_vec = self
                    .mps
                    .as_ref()
                    .ok_or("MPS not initialized")?
                    .to_state_vector();

                // Convert state vector to QuantumState
                let num_qubits = self.mps.as_ref().ok_or("MPS not initialized")?.num_qubits();

                let mut state = QuantumState::new(num_qubits);
                let amplitudes = state.amplitudes_mut();
                for (i, &amp) in state_vec.iter().enumerate() {
                    if i < amplitudes.len() {
                        amplitudes[i] = crate::C64::new(amp.re, amp.im);
                    }
                }
                Ok(state)
            }
            TensorNetworkType::PEPS => self
                .peps
                .as_ref()
                .ok_or("PEPS not initialized")?
                .contract_to_state(),
            TensorNetworkType::TTN => self
                .ttn
                .as_ref()
                .ok_or("TTN not initialized")?
                .contract_to_state(),
            TensorNetworkType::MERA => self
                .mera
                .as_ref()
                .ok_or("MERA not initialized")?
                .contract_to_state(),
        }
    }

    /// Benchmark all tensor network methods.
    pub fn benchmark_all(
        num_qubits: usize,
        bond_dim: usize,
    ) -> Vec<(TensorNetworkType, f64, usize)> {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Advanced Tensor Network Benchmark: {} qubits", num_qubits);
        println!("═══════════════════════════════════════════════════════════════");

        let mut results = Vec::new();

        // Benchmark MPS
        let start = std::time::Instant::now();
        let mps = crate::tensor_network::MPS::new(num_qubits, Some(bond_dim));
        let _ = mps.to_state_vector();
        let mps_time = start.elapsed().as_secs_f64();
        // Estimate cost: n * chi^3
        let mps_cost = num_qubits * bond_dim.pow(3);
        results.push((TensorNetworkType::MPS, mps_time, mps_cost));

        // Benchmark TTN
        let start = std::time::Instant::now();
        let ttn = TreeTensorNetwork::new_balanced(num_qubits, bond_dim);
        let _ = ttn.contract_to_state();
        let ttn_time = start.elapsed().as_secs_f64();
        results.push((
            TensorNetworkType::TTN,
            ttn_time,
            ttn.contraction_complexity(),
        ));

        println!("Method          | Time (sec) | Complexity");
        println!("────────────────┼────────────┼────────────");
        for (ty, time, cost) in &results {
            println!("{:?} | {:.4}    | {}", ty, time, cost);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_network_type_selection() {
        let ty = TensorNetworkType::select_optimal(10, 5, false, false);
        assert_eq!(ty, TensorNetworkType::MPS);

        let ty = TensorNetworkType::select_optimal(50, 10, false, true);
        assert_eq!(ty, TensorNetworkType::MERA);

        let ty = TensorNetworkType::select_optimal(36, 15, true, true);
        assert_eq!(ty, TensorNetworkType::PEPS);
    }

    #[test]
    fn test_peps_creation() {
        let peps = PEPSTensorNetwork::new(4, 4, 4);
        assert_eq!(peps.grid_size(), (4, 4));
        assert_eq!(peps.num_sites(), 16);
    }

    #[test]
    fn test_peps_zero_state_2x2() {
        // 2x2 PEPS initialized to |0000⟩
        let peps = PEPSTensorNetwork::new(2, 2, 2);
        let state = peps.contract_to_state().unwrap();
        let amps = state.amplitudes_ref();
        // |0000⟩ = basis 0 should have amplitude 1
        assert!(
            (amps[0].norm_sqr() - 1.0).abs() < 1e-10,
            "|0000⟩ amplitude: {:?}",
            amps[0]
        );
        // All others should be 0
        for i in 1..amps.len() {
            assert!(
                amps[i].norm_sqr() < 1e-10,
                "basis {}: amplitude {:?}",
                i,
                amps[i]
            );
        }
    }

    #[test]
    fn test_peps_single_gate_2x2() {
        // Apply X gate to site (0,0) of 2x2 PEPS
        let mut peps = PEPSTensorNetwork::new(2, 2, 2);
        let x_gate = [
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        ];
        peps.apply_single_qubit_gate(0, 0, x_gate).unwrap();
        let state = peps.contract_to_state().unwrap();
        let amps = state.amplitudes_ref();
        // Site (0,0) = qubit at bit position 0, so |1000⟩ = basis 1
        assert!(
            (amps[1].norm_sqr() - 1.0).abs() < 1e-10,
            "|1000⟩ amplitude: {:?}",
            amps[1]
        );
    }

    #[test]
    fn test_peps_hadamard_2x2() {
        // Apply H gate to site (0,0) of 2x2 PEPS → (|0⟩+|1⟩)/√2 on site 0
        let mut peps = PEPSTensorNetwork::new(2, 2, 2);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h_gate = [
            [C64::new(inv_sqrt2, 0.0), C64::new(inv_sqrt2, 0.0)],
            [C64::new(inv_sqrt2, 0.0), C64::new(-inv_sqrt2, 0.0)],
        ];
        peps.apply_single_qubit_gate(0, 0, h_gate).unwrap();
        let state = peps.contract_to_state().unwrap();
        let amps = state.amplitudes_ref();
        // |0000⟩ and |1000⟩ should have amplitude 1/√2
        assert!((amps[0].re - inv_sqrt2).abs() < 1e-10);
        assert!((amps[1].re - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_peps_cnot_horizontal() {
        // Apply H to (0,0) then CNOT (0,0)->(0,1) → Bell state
        let mut peps = PEPSTensorNetwork::new(1, 2, 2);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let h_gate = [
            [C64::new(inv_sqrt2, 0.0), C64::new(inv_sqrt2, 0.0)],
            [C64::new(inv_sqrt2, 0.0), C64::new(-inv_sqrt2, 0.0)],
        ];
        peps.apply_single_qubit_gate(0, 0, h_gate).unwrap();

        let z = C64::new(0.0, 0.0);
        let o = C64::new(1.0, 0.0);
        let cnot = [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]];
        peps.apply_two_qubit_gate(0, 0, 0, 1, cnot).unwrap();

        let state = peps.contract_to_state().unwrap();
        let amps = state.amplitudes_ref();
        // Bell state: (|00⟩ + |11⟩)/√2
        assert!(
            (amps[0].re - inv_sqrt2).abs() < 1e-10,
            "|00⟩ = {:?}",
            amps[0]
        );
        assert!(
            (amps[3].re - inv_sqrt2).abs() < 1e-10,
            "|11⟩ = {:?}",
            amps[3]
        );
        assert!(amps[1].norm_sqr() < 1e-10);
        assert!(amps[2].norm_sqr() < 1e-10);
    }

    #[test]
    fn test_ttn_creation() {
        let ttn = TreeTensorNetwork::new_balanced(8, 4);
        assert_eq!(ttn.num_leaves(), 8);
    }

    #[test]
    fn test_ttn_zero_state_2q() {
        let ttn = TreeTensorNetwork::new_balanced(2, 4);
        let state = ttn.contract_to_state().unwrap();
        let amps = state.amplitudes_ref();
        assert!((amps[0].re - 1.0).abs() < 1e-10, "|00⟩ = {:?}", amps[0]);
        for i in 1..4 {
            assert!(amps[i].norm_sqr() < 1e-10, "basis {} = {:?}", i, amps[i]);
        }
    }

    #[test]
    fn test_ttn_zero_state_4q() {
        let ttn = TreeTensorNetwork::new_balanced(4, 4);
        let state = ttn.contract_to_state().unwrap();
        let amps = state.amplitudes_ref();
        assert!((amps[0].re - 1.0).abs() < 1e-10, "|0000⟩ = {:?}", amps[0]);
        let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10, "norm = {}", norm);
    }

    #[test]
    fn test_mera_creation() {
        let mera = MERANetwork::new(16, 4, 4);
        assert_eq!(mera.num_physical_qubits(), 16);
        assert_eq!(mera.num_layers(), 4);
    }

    #[test]
    fn test_mera_zero_state_4q() {
        let mera = MERANetwork::new(4, 2, 4);
        let state = mera.contract_to_state().unwrap();
        let amps = state.amplitudes_ref();
        assert!((amps[0].re - 1.0).abs() < 1e-10, "|0000⟩ = {:?}", amps[0]);
        let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10, "norm = {}", norm);
    }

    #[test]
    fn test_auto_simulator() {
        let sim = AutoTensorNetworkSimulator::new(16, false, false, 4);
        assert_eq!(sim.network_type(), TensorNetworkType::MPS);
    }
}
