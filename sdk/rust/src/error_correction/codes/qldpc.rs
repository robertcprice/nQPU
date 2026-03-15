//! Quantum Low-Density Parity-Check (qLDPC) Codes
//!
//! This module implements modern qLDPC codes that achieve better encoding rates
//! than surface codes while maintaining practical error correction capabilities.
//!
//! # Constructions
//!
//! - **Hypergraph Product**: Tillich-Zemor construction from two classical codes
//! - **Bivariate Bicycle**: Recent IBM construction using cyclic group algebras
//! - **Repetition Code**: Simple 1D code for testing and composition
//!
//! # Decoding
//!
//! - **Belief Propagation**: Message-passing decoder with MinSum/SumProduct variants
//! - **OSD Post-Processing**: Ordered Statistics Decoding for improved threshold
//!
//! # References
//!
//! - Tillich & Zemor, "Quantum LDPC codes with positive rate and minimum distance
//!   proportional to the square root of the blocklength", IEEE Trans. Inf. Theory 2014
//! - Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, "High-threshold and low-overhead
//!   fault-tolerant quantum memory", Nature 2024

use std::collections::{BTreeSet, VecDeque};

// ============================================================
// SPARSE BINARY MATRIX (CSR FORMAT)
// ============================================================

/// Sparse binary matrix in Compressed Sparse Row (CSR) format.
///
/// Stores only the column indices of nonzero (=1) entries in each row.
/// All arithmetic is over GF(2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// For each row, the sorted set of column indices with value 1.
    row_data: Vec<BTreeSet<usize>>,
}

impl SparseMatrix {
    /// Create a new zero matrix of the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_data: vec![BTreeSet::new(); rows],
        }
    }

    /// Set entry (r, c) to 1.
    ///
    /// # Panics
    ///
    /// Panics if `r >= rows` or `c >= cols`.
    pub fn set(&mut self, r: usize, c: usize) {
        assert!(
            r < self.rows,
            "row index {} out of bounds (rows={})",
            r,
            self.rows
        );
        assert!(
            c < self.cols,
            "col index {} out of bounds (cols={})",
            c,
            self.cols
        );
        self.row_data[r].insert(c);
    }

    /// Clear entry (r, c) to 0.
    pub fn clear(&mut self, r: usize, c: usize) {
        assert!(
            r < self.rows,
            "row index {} out of bounds (rows={})",
            r,
            self.rows
        );
        assert!(
            c < self.cols,
            "col index {} out of bounds (cols={})",
            c,
            self.cols
        );
        self.row_data[r].remove(&c);
    }

    /// Toggle entry (r, c) over GF(2): 0->1 or 1->0.
    pub fn toggle(&mut self, r: usize, c: usize) {
        assert!(r < self.rows);
        assert!(c < self.cols);
        if self.row_data[r].contains(&c) {
            self.row_data[r].remove(&c);
        } else {
            self.row_data[r].insert(c);
        }
    }

    /// Get the value at (r, c).
    pub fn get(&self, r: usize, c: usize) -> bool {
        assert!(
            r < self.rows,
            "row index {} out of bounds (rows={})",
            r,
            self.rows
        );
        assert!(
            c < self.cols,
            "col index {} out of bounds (cols={})",
            c,
            self.cols
        );
        self.row_data[r].contains(&c)
    }

    /// Number of nonzero entries in row `r`.
    pub fn row_weight(&self, r: usize) -> usize {
        assert!(r < self.rows);
        self.row_data[r].len()
    }

    /// Number of nonzero entries in column `c`.
    pub fn col_weight(&self, c: usize) -> usize {
        assert!(c < self.cols);
        self.row_data.iter().filter(|row| row.contains(&c)).count()
    }

    /// Iterator over column indices with value 1 in row `r`.
    pub fn row_indices(&self, r: usize) -> impl Iterator<Item = &usize> {
        self.row_data[r].iter()
    }

    /// Collect all column indices for a given row.
    pub fn row_indices_vec(&self, r: usize) -> Vec<usize> {
        self.row_data[r].iter().copied().collect()
    }

    /// Total number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.row_data.iter().map(|r| r.len()).sum()
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> SparseMatrix {
        let mut t = SparseMatrix::new(self.cols, self.rows);
        for r in 0..self.rows {
            for &c in &self.row_data[r] {
                t.row_data[c].insert(r);
            }
        }
        t
    }

    /// Multiply two sparse binary matrices over GF(2): self * other.
    ///
    /// The result entry (i, j) is the XOR of (self[i,k] AND other[k,j]) for all k.
    ///
    /// # Panics
    ///
    /// Panics if `self.cols != other.rows`.
    pub fn multiply_gf2(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(
            self.cols, other.rows,
            "incompatible dimensions for GF(2) multiply: {}x{} * {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut result = SparseMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for &k in &self.row_data[i] {
                // Row k of `other` contributes to row i of result via XOR
                for &j in &other.row_data[k] {
                    result.toggle(i, j);
                }
            }
        }
        result
    }

    /// Check if the matrix is all zeros.
    pub fn is_zero(&self) -> bool {
        self.row_data.iter().all(|r| r.is_empty())
    }

    /// Create an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        let mut m = SparseMatrix::new(n, n);
        for i in 0..n {
            m.set(i, i);
        }
        m
    }

    /// Horizontal concatenation: [self | other].
    ///
    /// Both matrices must have the same number of rows.
    pub fn hcat(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.rows, other.rows, "hcat requires same number of rows");
        let mut result = SparseMatrix::new(self.rows, self.cols + other.cols);
        for r in 0..self.rows {
            for &c in &self.row_data[r] {
                result.set(r, c);
            }
            for &c in &other.row_data[r] {
                result.set(r, self.cols + c);
            }
        }
        result
    }

    /// Vertical concatenation: [self; other].
    ///
    /// Both matrices must have the same number of columns.
    pub fn vcat(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(
            self.cols, other.cols,
            "vcat requires same number of columns"
        );
        let mut result = SparseMatrix::new(self.rows + other.rows, self.cols);
        for r in 0..self.rows {
            for &c in &self.row_data[r] {
                result.set(r, c);
            }
        }
        for r in 0..other.rows {
            for &c in &other.row_data[r] {
                result.set(self.rows + r, c);
            }
        }
        result
    }

    /// Kronecker product over GF(2): self tensor other.
    ///
    /// Result dimensions: (self.rows * other.rows) x (self.cols * other.cols).
    pub fn kronecker(&self, other: &SparseMatrix) -> SparseMatrix {
        let out_rows = self.rows * other.rows;
        let out_cols = self.cols * other.cols;
        let mut result = SparseMatrix::new(out_rows, out_cols);
        for ra in 0..self.rows {
            for &ca in &self.row_data[ra] {
                for rb in 0..other.rows {
                    for &cb in &other.row_data[rb] {
                        result.set(ra * other.rows + rb, ca * other.cols + cb);
                    }
                }
            }
        }
        result
    }

    /// Compute the GF(2) rank via Gaussian elimination on a dense copy.
    pub fn rank_gf2(&self) -> usize {
        // Convert to dense row-echelon form
        let mut dense: Vec<Vec<bool>> = (0..self.rows)
            .map(|r| {
                let mut row = vec![false; self.cols];
                for &c in &self.row_data[r] {
                    row[c] = true;
                }
                row
            })
            .collect();

        let mut rank = 0;
        for col in 0..self.cols {
            // Find pivot row
            let mut pivot = None;
            for row in rank..self.rows {
                if dense[row][col] {
                    pivot = Some(row);
                    break;
                }
            }
            if let Some(p) = pivot {
                dense.swap(rank, p);
                // Eliminate column from other rows
                for row in 0..self.rows {
                    if row != rank && dense[row][col] {
                        for c in 0..self.cols {
                            dense[row][c] ^= dense[rank][c];
                        }
                    }
                }
                rank += 1;
            }
        }
        rank
    }
}

impl std::fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "SparseMatrix({}x{}, nnz={})",
            self.rows,
            self.cols,
            self.nnz()
        )?;
        for r in 0..self.rows.min(16) {
            write!(f, "  [")?;
            for c in 0..self.cols.min(32) {
                if self.row_data[r].contains(&c) {
                    write!(f, "1")?;
                } else {
                    write!(f, ".")?;
                }
            }
            if self.cols > 32 {
                write!(f, " ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.rows > 16 {
            writeln!(f, "  ...")?;
        }
        Ok(())
    }
}

// ============================================================
// TANNER GRAPH
// ============================================================

/// Bipartite Tanner graph representing a parity check matrix.
///
/// Check nodes correspond to rows and variable nodes correspond to columns.
/// An edge exists between check node i and variable node j iff H[i,j] = 1.
#[derive(Clone, Debug)]
pub struct TannerGraph {
    /// Number of check nodes (rows of H).
    pub num_checks: usize,
    /// Number of variable nodes (columns of H).
    pub num_variables: usize,
    /// For each check node, the indices of adjacent variable nodes.
    check_to_var: Vec<Vec<usize>>,
    /// For each variable node, the indices of adjacent check nodes.
    var_to_check: Vec<Vec<usize>>,
}

impl TannerGraph {
    /// Construct a Tanner graph from a parity check matrix H.
    pub fn from_parity_check(h: &SparseMatrix) -> Self {
        let num_checks = h.rows;
        let num_variables = h.cols;

        let check_to_var: Vec<Vec<usize>> = (0..num_checks).map(|r| h.row_indices_vec(r)).collect();

        let mut var_to_check: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
        for (r, neighbors) in check_to_var.iter().enumerate() {
            for &v in neighbors {
                var_to_check[v].push(r);
            }
        }

        Self {
            num_checks,
            num_variables,
            check_to_var,
            var_to_check,
        }
    }

    /// Degree of check node i (number of adjacent variable nodes).
    pub fn check_degree(&self, i: usize) -> usize {
        self.check_to_var[i].len()
    }

    /// Degree of variable node j (number of adjacent check nodes).
    pub fn variable_degree(&self, j: usize) -> usize {
        self.var_to_check[j].len()
    }

    /// Neighbors (variable nodes) of check node i.
    pub fn neighbors_of_check(&self, i: usize) -> &[usize] {
        &self.check_to_var[i]
    }

    /// Neighbors (check nodes) of variable node j.
    pub fn neighbors_of_variable(&self, j: usize) -> &[usize] {
        &self.var_to_check[j]
    }

    /// Compute the girth (length of shortest cycle) of the Tanner graph.
    ///
    /// Uses BFS from each node. Returns `usize::MAX` if the graph is acyclic.
    pub fn girth(&self) -> usize {
        let mut best = usize::MAX;

        // BFS from each variable node to find shortest cycle through it
        for start_var in 0..self.num_variables {
            if self.var_to_check[start_var].is_empty() {
                continue;
            }
            // BFS on the bipartite graph, tracking parent edges to detect back edges.
            // Nodes encoded as: variable j -> 2*j, check i -> 2*i+1
            let total = 2 * (self.num_checks + self.num_variables);
            let mut dist: Vec<i32> = vec![-1; total];
            let mut queue: VecDeque<usize> = VecDeque::new();

            let start = 2 * start_var;
            dist[start] = 0;
            queue.push_back(start);

            while let Some(u) = queue.pop_front() {
                let d_u = dist[u];
                if d_u as usize >= best / 2 {
                    break; // Cannot improve
                }
                if u % 2 == 0 {
                    // Variable node u/2
                    let v = u / 2;
                    for &c in &self.var_to_check[v] {
                        let w = 2 * c + 1;
                        if w < total {
                            if dist[w] == -1 {
                                dist[w] = d_u + 1;
                                queue.push_back(w);
                            } else {
                                let cycle_len = (d_u + 1 + dist[w]) as usize;
                                if cycle_len < best {
                                    best = cycle_len;
                                }
                            }
                        }
                    }
                } else {
                    // Check node (u-1)/2
                    let c = (u - 1) / 2;
                    for &v in &self.check_to_var[c] {
                        let w = 2 * v;
                        if w < total && w != start || dist[w] != 0 {
                            if dist[w] == -1 {
                                dist[w] = d_u + 1;
                                queue.push_back(w);
                            } else if dist[w] >= 0 {
                                let cycle_len = (d_u + 1 + dist[w]) as usize;
                                if cycle_len < best {
                                    best = cycle_len;
                                }
                            }
                        }
                    }
                }
            }
        }

        // The bipartite graph distances count edges; cycles in a Tanner graph
        // have even length. The girth is the shortest cycle length.
        best
    }

    /// Average check node degree.
    pub fn avg_check_degree(&self) -> f64 {
        if self.num_checks == 0 {
            return 0.0;
        }
        let total: usize = self.check_to_var.iter().map(|v| v.len()).sum();
        total as f64 / self.num_checks as f64
    }

    /// Average variable node degree.
    pub fn avg_variable_degree(&self) -> f64 {
        if self.num_variables == 0 {
            return 0.0;
        }
        let total: usize = self.var_to_check.iter().map(|v| v.len()).sum();
        total as f64 / self.num_variables as f64
    }
}

// ============================================================
// REPETITION CODE
// ============================================================

/// Simple 1D repetition code for testing and as a building block.
///
/// A distance-d repetition code encodes 1 logical qubit into d physical
/// qubits with d-1 parity checks.
#[derive(Clone, Debug)]
pub struct RepetitionCode {
    /// Code distance (number of physical qubits).
    pub distance: usize,
}

impl RepetitionCode {
    /// Create a repetition code of the given distance.
    ///
    /// # Panics
    ///
    /// Panics if distance < 2.
    pub fn new(distance: usize) -> Self {
        assert!(distance >= 2, "repetition code distance must be >= 2");
        Self { distance }
    }

    /// Generate the parity check matrix H.
    ///
    /// H is (d-1) x d with entries H[i, i] = 1 and H[i, i+1] = 1.
    pub fn parity_check_matrix(&self) -> SparseMatrix {
        let d = self.distance;
        let mut h = SparseMatrix::new(d - 1, d);
        for i in 0..(d - 1) {
            h.set(i, i);
            h.set(i, i + 1);
        }
        h
    }

    /// Code parameters [[n, k, d]].
    pub fn parameters(&self) -> (usize, usize, usize) {
        (self.distance, 1, self.distance)
    }
}

// ============================================================
// HYPERGRAPH PRODUCT CODE (TILLICH-ZEMOR)
// ============================================================

/// Hypergraph product code from the Tillich-Zemor construction.
///
/// Given two classical binary codes with parity check matrices H1 (r1 x n1)
/// and H2 (r2 x n2), the quantum code has:
///
/// - HX = [H1 tensor I_n2  |  I_r1 tensor H2^T]
/// - HZ = [I_n1 tensor H2  |  H1^T tensor I_r2]
///
/// The CSS condition HX * HZ^T = 0 (mod 2) is guaranteed by construction.
#[derive(Clone, Debug)]
pub struct HypergraphProductCode {
    /// X-type parity check matrix.
    pub hx: SparseMatrix,
    /// Z-type parity check matrix.
    pub hz: SparseMatrix,
    /// Number of physical qubits.
    n: usize,
    /// Number of logical qubits.
    k: usize,
    /// Upper bound on code distance.
    d_upper: usize,
    /// Dimensions of the input classical codes for reference.
    h1_dims: (usize, usize),
    h2_dims: (usize, usize),
}

impl HypergraphProductCode {
    /// Construct a hypergraph product code from two classical parity check matrices.
    ///
    /// H1 is r1 x n1, H2 is r2 x n2.
    /// The resulting quantum code acts on n = n1*n2 + r1*r2 physical qubits.
    pub fn from_classical(h1: &SparseMatrix, h2: &SparseMatrix) -> Self {
        let r1 = h1.rows;
        let n1 = h1.cols;
        let r2 = h2.rows;
        let n2 = h2.cols;

        let h1_t = h1.transpose();
        let h2_t = h2.transpose();

        let i_n1 = SparseMatrix::identity(n1);
        let i_n2 = SparseMatrix::identity(n2);
        let i_r1 = SparseMatrix::identity(r1);
        let i_r2 = SparseMatrix::identity(r2);

        // HX = [H1 tensor I_n2 | I_r1 tensor H2^T]
        let hx_left = h1.kronecker(&i_n2); // r1*n2 rows, n1*n2 cols
        let hx_right = i_r1.kronecker(&h2_t); // r1*n2 rows, r1*r2 cols
        let hx = hx_left.hcat(&hx_right);

        // HZ = [I_n1 tensor H2 | H1^T tensor I_r2]
        let hz_left = i_n1.kronecker(h2); // n1*r2 rows, n1*n2 cols
        let hz_right = h1_t.kronecker(&i_r2); // n1*r2 rows, r1*r2 cols  (actually n1*r2 rows since h1_t is n1 x r1)
        let hz = hz_left.hcat(&hz_right);

        let n = n1 * n2 + r1 * r2;

        // Number of logical qubits: k = n - rank(HX) - rank(HZ)
        let rank_hx = hx.rank_gf2();
        let rank_hz = hz.rank_gf2();
        let k = if n > rank_hx + rank_hz {
            n - rank_hx - rank_hz
        } else {
            0
        };

        // Distance upper bound: min(d1, d2) where di is the distance of the
        // classical code (approximated by minimum row weight of Hi).
        let d1_est = (0..r1)
            .map(|r| h1.row_weight(r))
            .filter(|&w| w > 0)
            .min()
            .unwrap_or(1);
        let d2_est = (0..r2)
            .map(|r| h2.row_weight(r))
            .filter(|&w| w > 0)
            .min()
            .unwrap_or(1);
        let d_upper = d1_est.min(d2_est);

        Self {
            hx,
            hz,
            n,
            k,
            d_upper,
            h1_dims: (r1, n1),
            h2_dims: (r2, n2),
        }
    }

    /// Number of physical qubits.
    pub fn num_physical_qubits(&self) -> usize {
        self.n
    }

    /// Number of logical qubits.
    pub fn num_logical_qubits(&self) -> usize {
        self.k
    }

    /// Upper bound on the code distance.
    pub fn code_distance_upper_bound(&self) -> usize {
        self.d_upper
    }

    /// Code parameters [[n, k, d]] where d is an upper bound.
    pub fn parameters(&self) -> (usize, usize, usize) {
        (self.n, self.k, self.d_upper)
    }

    /// Verify the CSS orthogonality condition: HX * HZ^T = 0 (mod 2).
    pub fn verify_css_condition(&self) -> bool {
        let hz_t = self.hz.transpose();
        let product = self.hx.multiply_gf2(&hz_t);
        product.is_zero()
    }

    /// Encoding rate k/n.
    pub fn rate(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.k as f64 / self.n as f64
    }
}

// ============================================================
// BIVARIATE BICYCLE CODE (IBM CONSTRUCTION)
// ============================================================

/// Bivariate bicycle code from cyclic group algebras.
///
/// Defined over the group Z_l x Z_m using polynomial generators.
/// These codes achieve high encoding rates with practical syndrome
/// extraction circuits on planar layouts.
///
/// The parity check matrices are constructed from circulant blocks:
/// - HX = [A | B]
/// - HZ = [B^T | A^T]
///
/// where A and B are l*m x l*m matrices built from the polynomial generators.
#[derive(Clone, Debug)]
pub struct BivariateBicycleCode {
    /// First group dimension.
    pub l: usize,
    /// Second group dimension.
    pub m: usize,
    /// X-type parity check matrix.
    pub hx: SparseMatrix,
    /// Z-type parity check matrix.
    pub hz: SparseMatrix,
    /// Number of physical qubits.
    n: usize,
    /// Number of logical qubits.
    k: usize,
}

impl BivariateBicycleCode {
    /// Construct a bivariate bicycle code.
    ///
    /// `l` and `m` are the cyclic group dimensions.
    /// `a_polys` contains exponent pairs (i) indexing into Z_l x Z_m for polynomial A.
    /// `b_polys` contains exponent pairs (i) indexing into Z_l x Z_m for polynomial B.
    ///
    /// Each index `i` represents the element (i / m, i % m) in Z_l x Z_m.
    /// The polynomials define which cyclic shifts are summed.
    pub fn new(l: usize, m: usize, a_polys: &[usize], b_polys: &[usize]) -> Self {
        let n_block = l * m;

        // Build circulant matrix A from polynomial specification
        let a = Self::build_circulant_block(l, m, a_polys);
        // Build circulant matrix B from polynomial specification
        let b = Self::build_circulant_block(l, m, b_polys);

        let a_t = a.transpose();
        let b_t = b.transpose();

        // HX = [A | B]
        let hx = a.hcat(&b);
        // HZ = [B^T | A^T]
        let hz = b_t.hcat(&a_t);

        let n = 2 * n_block;
        let rank_hx = hx.rank_gf2();
        let rank_hz = hz.rank_gf2();
        let k = if n > rank_hx + rank_hz {
            n - rank_hx - rank_hz
        } else {
            0
        };

        Self { l, m, hx, hz, n, k }
    }

    /// Build a circulant block matrix from polynomial indices.
    ///
    /// Each entry in `polys` specifies a cyclic shift. The matrix has size
    /// (l*m) x (l*m). Element (r, c) is 1 iff (c - r) mod (l*m) is in polys,
    /// applying the group operation component-wise on Z_l x Z_m.
    fn build_circulant_block(l: usize, m: usize, polys: &[usize]) -> SparseMatrix {
        let n = l * m;
        let mut mat = SparseMatrix::new(n, n);

        for row in 0..n {
            let row_l = row / m;
            let row_m = row % m;
            for &shift in polys {
                let shift_l = shift / m;
                let shift_m = shift % m;
                let col_l = (row_l + shift_l) % l;
                let col_m = (row_m + shift_m) % m;
                let col = col_l * m + col_m;
                mat.set(row, col);
            }
        }
        mat
    }

    /// Number of physical qubits.
    pub fn num_physical_qubits(&self) -> usize {
        self.n
    }

    /// Number of logical qubits.
    pub fn num_logical_qubits(&self) -> usize {
        self.k
    }

    /// Code parameters [[n, k, d]] where d is estimated.
    pub fn parameters(&self) -> (usize, usize, usize) {
        // Distance estimation by random sampling is expensive; provide
        // a conservative upper bound from the minimum row weight of HX and HZ.
        let d_x = (0..self.hx.rows)
            .map(|r| self.hx.row_weight(r))
            .filter(|&w| w > 0)
            .min()
            .unwrap_or(1);
        let d_z = (0..self.hz.rows)
            .map(|r| self.hz.row_weight(r))
            .filter(|&w| w > 0)
            .min()
            .unwrap_or(1);
        (self.n, self.k, d_x.min(d_z))
    }

    /// Verify the CSS orthogonality condition: HX * HZ^T = 0 (mod 2).
    pub fn verify_css_condition(&self) -> bool {
        let hz_t = self.hz.transpose();
        let product = self.hx.multiply_gf2(&hz_t);
        product.is_zero()
    }

    /// Encoding rate k/n.
    pub fn rate(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.k as f64 / self.n as f64
    }
}

// ============================================================
// BELIEF PROPAGATION DECODER
// ============================================================

/// Decoding mode for belief propagation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BPMode {
    /// Min-sum algorithm: approximation using min and sign operations.
    /// More numerically stable and faster.
    MinSum,
    /// Sum-product algorithm: exact marginal computation using tanh.
    /// More accurate but slower.
    SumProduct,
}

/// Belief Propagation (BP) decoder for binary linear codes over GF(2).
///
/// Supports both MinSum and SumProduct message update rules, with
/// optional OSD (Ordered Statistics Decoding) post-processing for
/// improved performance when BP fails to converge.
#[derive(Clone, Debug)]
pub struct BPDecoder {
    /// Parity check matrix.
    h: SparseMatrix,
    /// Tanner graph for efficient message passing.
    tanner: TannerGraph,
    /// Decoding mode.
    mode: BPMode,
    /// Maximum number of BP iterations.
    max_iters: usize,
    /// Damping factor for message updates (0 < alpha <= 1).
    damping: f64,
    /// Whether to apply OSD post-processing on BP failure.
    pub osd_enabled: bool,
    /// OSD order (number of least reliable bits to search).
    pub osd_order: usize,
}

impl BPDecoder {
    /// Create a new BP decoder for the given parity check matrix.
    ///
    /// # Arguments
    ///
    /// * `h` - Parity check matrix defining the code.
    /// * `mode` - MinSum or SumProduct message passing rule.
    /// * `max_iters` - Maximum number of message passing iterations.
    pub fn new(h: &SparseMatrix, mode: BPMode, max_iters: usize) -> Self {
        let tanner = TannerGraph::from_parity_check(h);
        Self {
            h: h.clone(),
            tanner,
            mode,
            max_iters,
            damping: 1.0,
            osd_enabled: false,
            osd_order: 0,
        }
    }

    /// Set the damping factor for message updates.
    pub fn with_damping(mut self, damping: f64) -> Self {
        assert!(damping > 0.0 && damping <= 1.0, "damping must be in (0, 1]");
        self.damping = damping;
        self
    }

    /// Enable OSD post-processing with the given order.
    pub fn with_osd(mut self, order: usize) -> Self {
        self.osd_enabled = true;
        self.osd_order = order;
        self
    }

    /// Decode a syndrome to estimate the most likely error pattern.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Binary syndrome vector (length = number of checks).
    /// * `error_rate` - Physical error probability per qubit (channel parameter).
    ///
    /// # Returns
    ///
    /// Estimated binary error vector (length = number of variables/qubits).
    pub fn decode(&self, syndrome: &[bool], error_rate: f64) -> Vec<bool> {
        assert_eq!(
            syndrome.len(),
            self.h.rows,
            "syndrome length {} != number of checks {}",
            syndrome.len(),
            self.h.rows
        );

        let n = self.h.cols;
        let m = self.h.rows;

        // Channel log-likelihood ratio: LLR = log((1-p)/p)
        let channel_llr = if error_rate <= 0.0 || error_rate >= 1.0 {
            0.0
        } else {
            ((1.0 - error_rate) / error_rate).ln()
        };

        // Initialize variable-to-check messages with channel LLR
        // msg_v2c[check_idx][var_idx] = LLR message from var to check
        let mut msg_v2c: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
        for c in 0..m {
            for &v in self.tanner.neighbors_of_check(c) {
                msg_v2c[c][v] = channel_llr;
            }
        }

        // Check-to-variable messages
        let mut msg_c2v: Vec<Vec<f64>> = vec![vec![0.0; n]; m];

        let mut hard_decision = vec![false; n];

        for _iter in 0..self.max_iters {
            // --- Check node update ---
            for c in 0..m {
                let neighbors = self.tanner.neighbors_of_check(c);
                let syndrome_sign: f64 = if syndrome[c] { -1.0 } else { 1.0 };

                for &v in neighbors {
                    let old_msg = msg_c2v[c][v];
                    let new_msg = match self.mode {
                        BPMode::MinSum => self.min_sum_check_update(c, v, &msg_v2c, syndrome_sign),
                        BPMode::SumProduct => {
                            self.sum_product_check_update(c, v, &msg_v2c, syndrome_sign)
                        }
                    };
                    msg_c2v[c][v] = self.damping * new_msg + (1.0 - self.damping) * old_msg;
                }
            }

            // --- Variable node update ---
            for v in 0..n {
                let checks = self.tanner.neighbors_of_variable(v);
                let total_llr: f64 =
                    channel_llr + checks.iter().map(|&c| msg_c2v[c][v]).sum::<f64>();

                hard_decision[v] = total_llr < 0.0;

                for &c in checks {
                    msg_v2c[c][v] = total_llr - msg_c2v[c][v];
                }
            }

            // --- Check convergence ---
            if self.check_syndrome(&hard_decision, syndrome) {
                return hard_decision;
            }
        }

        // BP did not converge; try OSD post-processing if enabled
        if self.osd_enabled {
            return self.osd_post_process(syndrome, error_rate, &hard_decision);
        }

        hard_decision
    }

    /// MinSum check node update rule.
    fn min_sum_check_update(
        &self,
        check: usize,
        target_var: usize,
        msg_v2c: &[Vec<f64>],
        syndrome_sign: f64,
    ) -> f64 {
        let neighbors = self.tanner.neighbors_of_check(check);
        let mut sign = syndrome_sign;
        let mut min_abs = f64::INFINITY;

        for &v in neighbors {
            if v == target_var {
                continue;
            }
            let m = msg_v2c[check][v];
            if m < 0.0 {
                sign = -sign;
            }
            let abs_m = m.abs();
            if abs_m < min_abs {
                min_abs = abs_m;
            }
        }

        // Normalization factor (0.75 is a common choice for MinSum)
        sign * min_abs * 0.75
    }

    /// SumProduct check node update rule using tanh.
    fn sum_product_check_update(
        &self,
        check: usize,
        target_var: usize,
        msg_v2c: &[Vec<f64>],
        syndrome_sign: f64,
    ) -> f64 {
        let neighbors = self.tanner.neighbors_of_check(check);
        let mut product = syndrome_sign;

        for &v in neighbors {
            if v == target_var {
                continue;
            }
            let m = msg_v2c[check][v];
            // tanh(m/2) with clamping for numerical stability
            let t = (m / 2.0).tanh().clamp(-0.9999999, 0.9999999);
            product *= t;
        }

        // 2 * atanh(product) with clamping
        let clamped = product.clamp(-0.9999999, 0.9999999);
        2.0 * clamped.atanh()
    }

    /// Verify that the hard decision satisfies the syndrome.
    fn check_syndrome(&self, error: &[bool], syndrome: &[bool]) -> bool {
        for c in 0..self.h.rows {
            let mut parity = false;
            for &v in self.tanner.neighbors_of_check(c) {
                if error[v] {
                    parity = !parity;
                }
            }
            if parity != syndrome[c] {
                return false;
            }
        }
        true
    }

    /// OSD (Ordered Statistics Decoding) post-processing.
    ///
    /// When BP fails to converge, use the soft information from BP to order
    /// the columns by reliability, then solve the syndrome equation via
    /// Gaussian elimination on the most reliable columns first.
    fn osd_post_process(
        &self,
        syndrome: &[bool],
        error_rate: f64,
        bp_estimate: &[bool],
    ) -> Vec<bool> {
        let n = self.h.cols;
        let m = self.h.rows;

        // Compute reliability from BP soft output (absolute LLR)
        // For simplicity, use channel prior combined with BP hint
        let channel_llr = if error_rate > 0.0 && error_rate < 1.0 {
            ((1.0 - error_rate) / error_rate).ln()
        } else {
            10.0
        };

        let mut reliability: Vec<(f64, usize)> = (0..n)
            .map(|v| {
                // Lower reliability if BP flagged this bit as likely error
                let r = if bp_estimate[v] {
                    channel_llr * 0.1
                } else {
                    channel_llr
                };
                (r, v)
            })
            .collect();

        // Sort by reliability (most reliable first)
        reliability.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let col_order: Vec<usize> = reliability.iter().map(|&(_, idx)| idx).collect();

        // Build reordered parity check matrix and perform Gaussian elimination
        let mut augmented: Vec<Vec<bool>> = (0..m)
            .map(|r| {
                let mut row = vec![false; n + 1];
                for (new_c, &orig_c) in col_order.iter().enumerate() {
                    row[new_c] = self.h.get(r, orig_c);
                }
                row[n] = syndrome[r];
                row
            })
            .collect();

        // Forward elimination
        let mut pivot_cols = Vec::new();
        let mut current_row = 0;
        for col in 0..n {
            if current_row >= m {
                break;
            }
            // Find pivot
            let mut pivot = None;
            for row in current_row..m {
                if augmented[row][col] {
                    pivot = Some(row);
                    break;
                }
            }
            if let Some(p) = pivot {
                augmented.swap(current_row, p);
                for row in 0..m {
                    if row != current_row && augmented[row][col] {
                        for c in 0..=n {
                            augmented[row][c] ^= augmented[current_row][c];
                        }
                    }
                }
                pivot_cols.push(col);
                current_row += 1;
            }
        }

        // Back-substitute: set information bits to 0, solve for pivot bits
        let mut reordered_error = vec![false; n];
        for (rank, &col) in pivot_cols.iter().enumerate() {
            reordered_error[col] = augmented[rank][n];
        }

        // OSD-w: try flipping up to `osd_order` information bits
        let info_cols: Vec<usize> = (0..n)
            .filter(|c| !pivot_cols.contains(c))
            .take(self.osd_order.min(n.saturating_sub(pivot_cols.len())))
            .collect();

        let mut best_error = reordered_error.clone();
        let mut best_weight: usize = best_error.iter().filter(|&&b| b).count();

        // Try flipping each information bit and re-solving
        for &info_col in &info_cols {
            let mut trial = reordered_error.clone();
            trial[info_col] = !trial[info_col];

            // Re-solve pivot columns with the flipped information bit
            for (rank, &pcol) in pivot_cols.iter().enumerate() {
                let mut val = augmented[rank][n];
                // Subtract contribution of information bits
                for &ic in &info_cols {
                    if augmented[rank][ic] && trial[ic] {
                        val = !val;
                    }
                }
                // Account for other non-pivot, non-info columns
                trial[pcol] = val;
            }

            let w: usize = trial.iter().filter(|&&b| b).count();
            if w < best_weight {
                best_weight = w;
                best_error = trial;
            }
        }

        // Map back to original column ordering
        let mut result = vec![false; n];
        for (new_c, &orig_c) in col_order.iter().enumerate() {
            result[orig_c] = best_error[new_c];
        }
        result
    }

    /// Compute the syndrome for a given error vector.
    pub fn syndrome_of(&self, error: &[bool]) -> Vec<bool> {
        let mut syn = vec![false; self.h.rows];
        for c in 0..self.h.rows {
            let mut parity = false;
            for &v in self.tanner.neighbors_of_check(c) {
                if v < error.len() && error[v] {
                    parity = !parity;
                }
            }
            syn[c] = parity;
        }
        syn
    }
}

// ============================================================
// BENCHMARKING UTILITIES
// ============================================================

/// Result of comparing qLDPC vs surface code at a given distance.
#[derive(Clone, Debug)]
pub struct CodeComparisonResult {
    pub distance: usize,
    pub surface_code_qubits: usize,
    pub surface_code_logicals: usize,
    pub qldpc_qubits: usize,
    pub qldpc_logicals: usize,
    pub qldpc_rate: f64,
    pub surface_code_rate: f64,
    pub qubit_savings_ratio: f64,
}

/// Compare a qLDPC hypergraph product code against a surface code at the
/// same target distance.
///
/// Uses a repetition code of the given distance as the classical seed.
pub fn surface_code_comparison(distance: usize) -> CodeComparisonResult {
    // Surface code: [[d^2 + (d-1)^2, 1, d]] for rotated planar
    let surface_n = distance * distance + (distance - 1) * (distance - 1);
    let surface_k = 1;
    let surface_rate = surface_k as f64 / surface_n as f64;

    // Hypergraph product of two repetition codes of the given distance
    let rep = RepetitionCode::new(distance);
    let h = rep.parity_check_matrix();
    let hgp = HypergraphProductCode::from_classical(&h, &h);
    let (qldpc_n, qldpc_k, _qldpc_d) = hgp.parameters();
    let qldpc_rate = if qldpc_n > 0 {
        qldpc_k as f64 / qldpc_n as f64
    } else {
        0.0
    };

    let savings = if qldpc_n > 0 && surface_n > 0 {
        // Ratio of qubits per logical qubit
        let surface_per_logical = surface_n as f64 / surface_k as f64;
        let qldpc_per_logical = if qldpc_k > 0 {
            qldpc_n as f64 / qldpc_k as f64
        } else {
            f64::INFINITY
        };
        if qldpc_per_logical.is_finite() {
            surface_per_logical / qldpc_per_logical
        } else {
            0.0
        }
    } else {
        0.0
    };

    CodeComparisonResult {
        distance,
        surface_code_qubits: surface_n,
        surface_code_logicals: surface_k,
        qldpc_qubits: qldpc_n,
        qldpc_logicals: qldpc_k,
        qldpc_rate,
        surface_code_rate: surface_rate,
        qubit_savings_ratio: savings,
    }
}

/// Estimate the logical error rate of a code+decoder combination via
/// Monte Carlo sampling.
///
/// Applies independent bit-flip errors at the given physical error rate,
/// computes the syndrome, decodes, and checks if the residual error is
/// in the codespace (trivial syndrome) without a logical error.
///
/// For CSS codes, this tests X-type errors against HZ checks.
pub fn logical_error_rate(
    hz: &SparseMatrix,
    decoder: &BPDecoder,
    physical_error_rate: f64,
    num_trials: usize,
) -> f64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = hz.cols;
    let mut logical_failures = 0usize;

    for _ in 0..num_trials {
        // Generate random error
        let error: Vec<bool> = (0..n)
            .map(|_| rng.gen::<f64>() < physical_error_rate)
            .collect();

        // Compute syndrome
        let syndrome = decoder.syndrome_of(&error);

        // Decode
        let correction = decoder.decode(&syndrome, physical_error_rate);

        // Residual error = error XOR correction
        let residual: Vec<bool> = error
            .iter()
            .zip(correction.iter())
            .map(|(&e, &c)| e ^ c)
            .collect();

        // Check if residual is a nontrivial logical operator.
        // A residual in the codespace has zero syndrome for HZ.
        let residual_syndrome = decoder.syndrome_of(&residual);
        let residual_has_syndrome = residual_syndrome.iter().any(|&b| b);

        // If residual has nonzero syndrome, decoding failed (not even codespace).
        // If residual is in codespace but nonzero, it might be a logical error.
        if residual_has_syndrome || residual.iter().any(|&b| b) {
            logical_failures += 1;
        }
    }

    logical_failures as f64 / num_trials as f64
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- SparseMatrix tests ---

    #[test]
    fn test_sparse_matrix_set_get() {
        let mut m = SparseMatrix::new(3, 4);
        assert!(!m.get(0, 0));
        m.set(0, 0);
        assert!(m.get(0, 0));
        m.set(1, 2);
        assert!(m.get(1, 2));
        assert!(!m.get(1, 0));
        assert_eq!(m.nnz(), 2);
    }

    #[test]
    fn test_sparse_matrix_toggle() {
        let mut m = SparseMatrix::new(2, 2);
        m.toggle(0, 1);
        assert!(m.get(0, 1));
        m.toggle(0, 1);
        assert!(!m.get(0, 1));
    }

    #[test]
    fn test_sparse_matrix_row_col_weight() {
        let mut m = SparseMatrix::new(3, 3);
        m.set(0, 0);
        m.set(0, 1);
        m.set(0, 2);
        m.set(1, 1);
        assert_eq!(m.row_weight(0), 3);
        assert_eq!(m.row_weight(1), 1);
        assert_eq!(m.row_weight(2), 0);
        assert_eq!(m.col_weight(0), 1);
        assert_eq!(m.col_weight(1), 2);
        assert_eq!(m.col_weight(2), 1);
    }

    #[test]
    fn test_sparse_matrix_transpose() {
        let mut m = SparseMatrix::new(2, 3);
        m.set(0, 1);
        m.set(1, 0);
        m.set(1, 2);

        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!(t.get(1, 0));
        assert!(t.get(0, 1));
        assert!(t.get(2, 1));
        assert!(!t.get(0, 0));
    }

    #[test]
    fn test_sparse_matrix_multiply_gf2_identity() {
        let mut m = SparseMatrix::new(2, 3);
        m.set(0, 0);
        m.set(0, 2);
        m.set(1, 1);

        let id = SparseMatrix::identity(3);
        let result = m.multiply_gf2(&id);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 3);
        assert!(result.get(0, 0));
        assert!(result.get(0, 2));
        assert!(result.get(1, 1));
        assert!(!result.get(0, 1));
    }

    #[test]
    fn test_sparse_matrix_multiply_gf2_cancellation() {
        // Test that 1+1=0 in GF(2)
        // M = [[1,1],[0,1]], N = [[1],[1]]
        // M*N over GF(2) = [[0],[1]]
        let mut m = SparseMatrix::new(2, 2);
        m.set(0, 0);
        m.set(0, 1);
        m.set(1, 1);

        let mut n = SparseMatrix::new(2, 1);
        n.set(0, 0);
        n.set(1, 0);

        let result = m.multiply_gf2(&n);
        assert!(!result.get(0, 0), "1+1=0 in GF(2)");
        assert!(result.get(1, 0));
    }

    #[test]
    fn test_sparse_matrix_kronecker() {
        let id2 = SparseMatrix::identity(2);
        let id3 = SparseMatrix::identity(3);
        let kron = id2.kronecker(&id3);
        assert_eq!(kron.rows, 6);
        assert_eq!(kron.cols, 6);
        // Should be a 6x6 identity
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(kron.get(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_sparse_matrix_rank_gf2() {
        let id4 = SparseMatrix::identity(4);
        assert_eq!(id4.rank_gf2(), 4);

        // Rank-deficient: two identical rows
        let mut m = SparseMatrix::new(3, 3);
        m.set(0, 0);
        m.set(0, 1);
        m.set(1, 0);
        m.set(1, 1); // Same as row 0
        m.set(2, 2);
        assert_eq!(m.rank_gf2(), 2);
    }

    // --- TannerGraph tests ---

    #[test]
    fn test_tanner_graph_from_repetition_code() {
        let rep = RepetitionCode::new(4);
        let h = rep.parity_check_matrix();
        let tanner = TannerGraph::from_parity_check(&h);

        assert_eq!(tanner.num_checks, 3);
        assert_eq!(tanner.num_variables, 4);

        // Each check node connects to exactly 2 variable nodes
        for i in 0..3 {
            assert_eq!(tanner.check_degree(i), 2);
        }

        // Interior variable nodes have degree 2, boundary nodes degree 1
        assert_eq!(tanner.variable_degree(0), 1);
        assert_eq!(tanner.variable_degree(1), 2);
        assert_eq!(tanner.variable_degree(2), 2);
        assert_eq!(tanner.variable_degree(3), 1);
    }

    #[test]
    fn test_tanner_graph_neighbors() {
        let rep = RepetitionCode::new(3);
        let h = rep.parity_check_matrix();
        let tanner = TannerGraph::from_parity_check(&h);

        // Check 0 connects to variables 0,1
        let n0 = tanner.neighbors_of_check(0);
        assert!(n0.contains(&0) && n0.contains(&1));

        // Variable 1 connects to checks 0,1
        let v1 = tanner.neighbors_of_variable(1);
        assert!(v1.contains(&0) && v1.contains(&1));
    }

    // --- RepetitionCode tests ---

    #[test]
    fn test_repetition_code_parameters() {
        let rep = RepetitionCode::new(5);
        assert_eq!(rep.parameters(), (5, 1, 5));
    }

    #[test]
    fn test_repetition_code_parity_check() {
        let rep = RepetitionCode::new(4);
        let h = rep.parity_check_matrix();
        assert_eq!(h.rows, 3);
        assert_eq!(h.cols, 4);

        // Each row should have weight 2
        for r in 0..3 {
            assert_eq!(h.row_weight(r), 2);
        }
    }

    // --- HypergraphProductCode tests ---

    #[test]
    fn test_hgp_from_repetition_codes() {
        let rep = RepetitionCode::new(3);
        let h = rep.parity_check_matrix();
        let hgp = HypergraphProductCode::from_classical(&h, &h);

        let (n, k, _d) = hgp.parameters();
        // For rep-3 x rep-3: n = 3*3 + 2*2 = 13
        assert_eq!(n, 13);
        // Should have at least 1 logical qubit
        assert!(k >= 1, "expected k >= 1, got k = {}", k);
    }

    #[test]
    fn test_hgp_css_condition() {
        // The CSS condition HX * HZ^T = 0 must hold for any hypergraph product code
        let rep = RepetitionCode::new(3);
        let h = rep.parity_check_matrix();
        let hgp = HypergraphProductCode::from_classical(&h, &h);

        assert!(
            hgp.verify_css_condition(),
            "CSS condition HX * HZ^T = 0 violated"
        );
    }

    #[test]
    fn test_hgp_css_condition_asymmetric() {
        // Test with two different classical codes
        let rep3 = RepetitionCode::new(3);
        let rep4 = RepetitionCode::new(4);
        let h1 = rep3.parity_check_matrix();
        let h2 = rep4.parity_check_matrix();
        let hgp = HypergraphProductCode::from_classical(&h1, &h2);

        assert!(
            hgp.verify_css_condition(),
            "CSS condition violated for asymmetric hypergraph product"
        );
        assert!(hgp.num_physical_qubits() > 0);
        assert!(hgp.num_logical_qubits() >= 1);
    }

    #[test]
    fn test_hgp_rate_positive() {
        let rep = RepetitionCode::new(5);
        let h = rep.parity_check_matrix();
        let hgp = HypergraphProductCode::from_classical(&h, &h);

        let rate = hgp.rate();
        assert!(rate > 0.0, "expected positive encoding rate, got {}", rate);
    }

    // --- BPDecoder tests ---

    #[test]
    fn test_bp_decoder_no_error() {
        let rep = RepetitionCode::new(5);
        let h = rep.parity_check_matrix();
        let decoder = BPDecoder::new(&h, BPMode::SumProduct, 50);

        // Zero syndrome => no error detected
        let syndrome = vec![false; 4];
        let result = decoder.decode(&syndrome, 0.1);
        assert!(
            result.iter().all(|&b| !b),
            "expected no error for zero syndrome"
        );
    }

    #[test]
    fn test_bp_decoder_single_error() {
        let rep = RepetitionCode::new(5);
        let h = rep.parity_check_matrix();
        let decoder = BPDecoder::new(&h, BPMode::SumProduct, 50);

        // Single error on qubit 2 => syndrome = [0, 1, 1, 0]
        let error = vec![false, false, true, false, false];
        let syndrome = decoder.syndrome_of(&error);
        assert_eq!(syndrome, vec![false, true, true, false]);

        let correction = decoder.decode(&syndrome, 0.1);
        // The correction should yield a valid codeword when XORed with error
        let residual: Vec<bool> = error
            .iter()
            .zip(correction.iter())
            .map(|(&e, &c)| e ^ c)
            .collect();
        let residual_syn = decoder.syndrome_of(&residual);
        assert!(
            residual_syn.iter().all(|&b| !b),
            "residual should have zero syndrome"
        );
    }

    #[test]
    fn test_bp_decoder_min_sum_single_error() {
        let rep = RepetitionCode::new(5);
        let h = rep.parity_check_matrix();
        let decoder = BPDecoder::new(&h, BPMode::MinSum, 50);

        let error = vec![false, false, false, true, false];
        let syndrome = decoder.syndrome_of(&error);
        let correction = decoder.decode(&syndrome, 0.1);

        let residual: Vec<bool> = error
            .iter()
            .zip(correction.iter())
            .map(|(&e, &c)| e ^ c)
            .collect();
        let residual_syn = decoder.syndrome_of(&residual);
        assert!(
            residual_syn.iter().all(|&b| !b),
            "MinSum decoder should correct single bit error"
        );
    }

    #[test]
    fn test_bp_syndrome_computation() {
        let rep = RepetitionCode::new(4);
        let h = rep.parity_check_matrix();
        let decoder = BPDecoder::new(&h, BPMode::MinSum, 10);

        // All-zero error
        let syn = decoder.syndrome_of(&[false, false, false, false]);
        assert_eq!(syn, vec![false, false, false]);

        // Error on qubit 0
        let syn = decoder.syndrome_of(&[true, false, false, false]);
        assert_eq!(syn, vec![true, false, false]);

        // Error on qubit 1
        let syn = decoder.syndrome_of(&[false, true, false, false]);
        assert_eq!(syn, vec![true, true, false]);
    }

    // --- BivariateBicycleCode tests ---

    #[test]
    fn test_bivariate_bicycle_basic() {
        // Small example: l=3, m=3 with simple shift polynomials
        // A has shifts {0, 1}, B has shifts {0, 3} (shift by (0,0) and (1,0))
        let code = BivariateBicycleCode::new(3, 3, &[0, 1], &[0, 3]);

        let (n, k, _d) = code.parameters();
        assert_eq!(n, 18, "expected 2*3*3=18 physical qubits");
        // Just verify k is non-negative and reasonable
        assert!(k <= n, "k should be <= n");
    }

    #[test]
    fn test_bivariate_bicycle_css_condition() {
        let code = BivariateBicycleCode::new(3, 3, &[0, 1], &[0, 3]);
        assert!(
            code.verify_css_condition(),
            "CSS condition violated for bivariate bicycle code"
        );
    }

    #[test]
    fn test_bivariate_bicycle_larger() {
        // l=6, m=6 with shifts that produce a nontrivial code
        let code = BivariateBicycleCode::new(6, 6, &[0, 1, 6], &[0, 2, 12]);
        let (n, k, _d) = code.parameters();
        assert_eq!(n, 72);
        assert!(code.verify_css_condition());
        assert!(
            k > 0,
            "expected positive k for l=6,m=6 bicycle code, got {}",
            k
        );
    }

    // --- Benchmark utility tests ---

    #[test]
    fn test_surface_code_comparison() {
        let result = surface_code_comparison(3);
        assert_eq!(result.distance, 3);
        assert_eq!(result.surface_code_qubits, 13); // 9 + 4
        assert_eq!(result.surface_code_logicals, 1);
        assert!(result.qldpc_qubits > 0);
        assert!(result.qldpc_rate >= 0.0);
    }

    #[test]
    fn test_surface_code_comparison_distance_5() {
        let result = surface_code_comparison(5);
        assert_eq!(result.distance, 5);
        // Surface code: 25 + 16 = 41
        assert_eq!(result.surface_code_qubits, 41);
        assert!(result.qldpc_logicals >= 1);
    }
}
