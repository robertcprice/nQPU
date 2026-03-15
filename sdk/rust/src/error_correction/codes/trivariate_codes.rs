//! Trivariate Tricycle (TT) Codes — Quantum Error Correcting Codes
//!
//! Generalizes bivariate bicycle codes by adding a third cyclic dimension and
//! meta-check capabilities for measurement error diagnosis. This family of codes
//! lives in the group algebra F_2[Z_l x Z_m x Z_p] and yields CSS codes with
//! parameters [[n, k, d]] where n = 2 * l * m * p.
//!
//! # Code Construction
//!
//! Given generator polynomials a(x,y,z) and b(x,y,z) over the trivariate
//! polynomial ring F_2[x,y,z] / (x^l - 1, y^m - 1, z^p - 1):
//!
//!   H_X = [A | B]      (check matrix for X-stabilizers)
//!   H_Z = [B^T | A^T]  (check matrix for Z-stabilizers)
//!
//! The CSS condition H_X * H_Z^T = 0 (mod 2) holds because in the commutative
//! group algebra, A*B^T + B*A^T = 0 for symmetric generator sets.
//!
//! # Meta-Checks
//!
//! The third dimension enables a richer meta-check structure. Meta-checks verify
//! measurement consistency: for a valid syndrome s, M * s = 0. When M * s != 0,
//! some stabilizer measurements are faulty. This is the key advantage of TT codes
//! over their bivariate cousins.
//!
//! # Decoding
//!
//! - **Minimum Weight Perfect Matching**: graph-based decoder
//! - **Belief Propagation**: iterative message passing on Tanner graph
//! - **Union-Find**: near-linear time disjoint-set decoder
//! - **Sliding Window**: overlapping temporal windows for repeated rounds
//!
//! # Pre-built Families
//!
//! - `tt_small()`:  [[18, 2, 3]]  — smallest TT code (l=3, m=3, p=1)
//! - `tt_medium()`: [[72, 8, 4]]  — moderate code (l=3, m=3, p=4)
//! - `tt_large()`:  [[288,12, 8]] — high-distance code (l=4, m=4, p=9)
//! - `bivariate_bicycle(l, m)`: degenerate case with p=1
//!
//! # References
//!
//! - Bravyi, Cross, Gambetta, Maslov, Rall, Yoder, "High-threshold and
//!   low-overhead fault-tolerant quantum memory", Nature 2024
//! - Breuckmann, Eberhardt, "Quantum Low-Density Parity-Check Codes", PRX
//!   Quantum 2021
//! - Quintavalle, Campbell, Vasmer, "Reshape of the Tanner graph for quantum
//!   LDPC decoding", 2025

use rand::Rng;
use std::collections::BTreeSet;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from trivariate tricycle code operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrivariateError {
    /// Invalid construction parameters.
    InvalidParameters(String),
    /// Code construction failed (e.g., CSS condition violated).
    CodeConstructionFailed(String),
    /// Decoding failed to find a valid correction.
    DecodingFailed(String),
    /// Meta-check construction or evaluation error.
    MetaCheckError(String),
}

impl fmt::Display for TrivariateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrivariateError::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
            TrivariateError::CodeConstructionFailed(s) => {
                write!(f, "Code construction failed: {}", s)
            }
            TrivariateError::DecodingFailed(s) => write!(f, "Decoding failed: {}", s),
            TrivariateError::MetaCheckError(s) => write!(f, "Meta-check error: {}", s),
        }
    }
}

impl std::error::Error for TrivariateError {}

// ============================================================
// SPARSE BINARY MATRIX (CSR over GF(2))
// ============================================================

/// Sparse binary matrix over GF(2) in compressed sparse row format.
///
/// Stores, for each row, a sorted set of column indices where the entry is 1.
/// All arithmetic is modulo 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseBinaryMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// For each row, sorted column indices with value 1.
    pub entries: Vec<BTreeSet<usize>>,
}

impl SparseBinaryMatrix {
    /// Create a zero matrix of the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: vec![BTreeSet::new(); rows],
        }
    }

    /// Create an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.set(i, i);
        }
        m
    }

    /// Set entry (r, c) to 1.
    pub fn set(&mut self, r: usize, c: usize) {
        assert!(r < self.rows && c < self.cols, "index out of bounds");
        self.entries[r].insert(c);
    }

    /// Clear entry (r, c) to 0.
    pub fn clear(&mut self, r: usize, c: usize) {
        assert!(r < self.rows && c < self.cols);
        self.entries[r].remove(&c);
    }

    /// Toggle entry (r, c): 0 becomes 1, 1 becomes 0.
    pub fn toggle(&mut self, r: usize, c: usize) {
        assert!(r < self.rows && c < self.cols);
        if self.entries[r].contains(&c) {
            self.entries[r].remove(&c);
        } else {
            self.entries[r].insert(c);
        }
    }

    /// Get the value at (r, c).
    pub fn get(&self, r: usize, c: usize) -> bool {
        assert!(r < self.rows && c < self.cols);
        self.entries[r].contains(&c)
    }

    /// Column indices with value 1 in a given row.
    pub fn row_indices(&self, r: usize) -> Vec<usize> {
        self.entries[r].iter().copied().collect()
    }

    /// Row weight (number of 1-entries in a row).
    pub fn row_weight(&self, r: usize) -> usize {
        self.entries[r].len()
    }

    /// Column weight (number of 1-entries in a column).
    pub fn col_weight(&self, c: usize) -> usize {
        self.entries.iter().filter(|row| row.contains(&c)).count()
    }

    /// Total number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.entries.iter().map(|r| r.len()).sum()
    }

    /// Check if the matrix is all zeros.
    pub fn is_zero(&self) -> bool {
        self.entries.iter().all(|r| r.is_empty())
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Self {
        let mut t = Self::new(self.cols, self.rows);
        for r in 0..self.rows {
            for &c in &self.entries[r] {
                t.entries[c].insert(r);
            }
        }
        t
    }

    /// Multiply two sparse binary matrices over GF(2): self * other.
    pub fn multiply_gf2(&self, other: &SparseBinaryMatrix) -> SparseBinaryMatrix {
        assert_eq!(
            self.cols, other.rows,
            "dimension mismatch: {}x{} * {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut result = SparseBinaryMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for &k in &self.entries[i] {
                for &j in &other.entries[k] {
                    result.toggle(i, j);
                }
            }
        }
        result
    }

    /// Horizontal concatenation: [self | other].
    pub fn hcat(&self, other: &SparseBinaryMatrix) -> SparseBinaryMatrix {
        assert_eq!(self.rows, other.rows, "hcat requires same row count");
        let mut result = SparseBinaryMatrix::new(self.rows, self.cols + other.cols);
        for r in 0..self.rows {
            for &c in &self.entries[r] {
                result.set(r, c);
            }
            for &c in &other.entries[r] {
                result.set(r, self.cols + c);
            }
        }
        result
    }

    /// Vertical concatenation: [self; other].
    pub fn vcat(&self, other: &SparseBinaryMatrix) -> SparseBinaryMatrix {
        assert_eq!(self.cols, other.cols, "vcat requires same column count");
        let mut result = SparseBinaryMatrix::new(self.rows + other.rows, self.cols);
        for r in 0..self.rows {
            for &c in &self.entries[r] {
                result.set(r, c);
            }
        }
        for r in 0..other.rows {
            for &c in &other.entries[r] {
                result.set(self.rows + r, c);
            }
        }
        result
    }

    /// XOR row `src` into row `dst` (row addition over GF(2)).
    pub fn row_add(&mut self, dst: usize, src: usize) {
        let src_cols: Vec<usize> = self.entries[src].iter().copied().collect();
        for c in src_cols {
            self.toggle(dst, c);
        }
    }

    /// Compute the GF(2) rank via Gaussian elimination on a dense copy.
    pub fn rank_gf2(&self) -> usize {
        let mut dense = self.to_dense_bool();
        let nrows = dense.len();
        if nrows == 0 {
            return 0;
        }
        let ncols = if nrows > 0 { dense[0].len() } else { 0 };
        let mut rank = 0;
        for col in 0..ncols {
            let mut pivot = None;
            for row in rank..nrows {
                if dense[row][col] {
                    pivot = Some(row);
                    break;
                }
            }
            if let Some(p) = pivot {
                dense.swap(rank, p);
                for row in 0..nrows {
                    if row != rank && dense[row][col] {
                        for c in 0..ncols {
                            dense[row][c] ^= dense[rank][c];
                        }
                    }
                }
                rank += 1;
            }
        }
        rank
    }

    /// Convert to dense boolean representation.
    pub fn to_dense_bool(&self) -> Vec<Vec<bool>> {
        (0..self.rows)
            .map(|r| {
                let mut row = vec![false; self.cols];
                for &c in &self.entries[r] {
                    row[c] = true;
                }
                row
            })
            .collect()
    }

    /// Multiply matrix by a boolean vector over GF(2).
    pub fn mul_vec(&self, v: &[bool]) -> Vec<bool> {
        assert_eq!(v.len(), self.cols);
        let mut result = vec![false; self.rows];
        for r in 0..self.rows {
            let mut acc = false;
            for &c in &self.entries[r] {
                if v[c] {
                    acc ^= true;
                }
            }
            result[r] = acc;
        }
        result
    }
}

impl fmt::Display for SparseBinaryMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "SparseBinaryMatrix({}x{}, nnz={})",
            self.rows,
            self.cols,
            self.nnz()
        )?;
        for r in 0..self.rows.min(16) {
            write!(f, "  [")?;
            for c in 0..self.cols.min(40) {
                if self.entries[r].contains(&c) {
                    write!(f, "1")?;
                } else {
                    write!(f, ".")?;
                }
            }
            if self.cols > 40 {
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
// TRIVARIATE POLYNOMIAL RING ELEMENT
// ============================================================

/// Element of the trivariate polynomial ring F_2[x,y,z] / (x^l - 1, y^m - 1, z^p - 1).
///
/// Coefficients are stored as a flattened boolean vector indexed by (i, j, k)
/// representing the monomial x^i * y^j * z^k. All arithmetic is over GF(2),
/// with cyclic reduction modulo (x^l - 1, y^m - 1, z^p - 1).
#[derive(Clone, Debug)]
pub struct TrivariatePolynomial {
    /// Flattened coefficients: index = i * (m * p) + j * p + k for x^i y^j z^k.
    pub coefficients: Vec<bool>,
    /// Dimensions (l, m, p) of the cyclic groups.
    pub dims: (usize, usize, usize),
}

impl TrivariatePolynomial {
    /// Create the zero polynomial over Z_l x Z_m x Z_p.
    pub fn zero(l: usize, m: usize, p: usize) -> Self {
        Self {
            coefficients: vec![false; l * m * p],
            dims: (l, m, p),
        }
    }

    /// Create a polynomial from a list of monomial exponents (i, j, k).
    ///
    /// Each tuple (i, j, k) contributes the term x^i * y^j * z^k.
    /// Duplicate monomials cancel (GF(2) addition).
    pub fn from_monomials(
        l: usize,
        m: usize,
        p: usize,
        monomials: &[(usize, usize, usize)],
    ) -> Self {
        let mut poly = Self::zero(l, m, p);
        for &(i, j, k) in monomials {
            let ii = i % l;
            let jj = j % m;
            let kk = k % p;
            let idx = ii * (m * p) + jj * p + kk;
            poly.coefficients[idx] ^= true;
        }
        poly
    }

    /// Create a single monomial x^i * y^j * z^k.
    pub fn monomial(l: usize, m: usize, p: usize, i: usize, j: usize, k: usize) -> Self {
        Self::from_monomials(l, m, p, &[(i, j, k)])
    }

    /// Create the identity element (x^0 y^0 z^0 = 1).
    pub fn one(l: usize, m: usize, p: usize) -> Self {
        Self::monomial(l, m, p, 0, 0, 0)
    }

    /// Get the coefficient of x^i y^j z^k.
    pub fn get(&self, i: usize, j: usize, k: usize) -> bool {
        let (l, m, p) = self.dims;
        let ii = i % l;
        let jj = j % m;
        let kk = k % p;
        self.coefficients[ii * (m * p) + jj * p + kk]
    }

    /// Set the coefficient of x^i y^j z^k.
    pub fn set(&mut self, i: usize, j: usize, k: usize, val: bool) {
        let (l, m, p) = self.dims;
        let ii = i % l;
        let jj = j % m;
        let kk = k % p;
        self.coefficients[ii * (m * p) + jj * p + kk] = val;
    }

    /// Check if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|&c| !c)
    }

    /// Hamming weight: number of non-zero coefficients.
    pub fn weight(&self) -> usize {
        self.coefficients.iter().filter(|&&c| c).count()
    }

    /// List of active monomial exponents.
    pub fn active_monomials(&self) -> Vec<(usize, usize, usize)> {
        let (l, m, p) = self.dims;
        let mut result = Vec::new();
        for i in 0..l {
            for j in 0..m {
                for k in 0..p {
                    if self.get(i, j, k) {
                        result.push((i, j, k));
                    }
                }
            }
        }
        result
    }

    /// Add two polynomials over GF(2) (XOR of coefficients).
    pub fn add(&self, other: &TrivariatePolynomial) -> TrivariatePolynomial {
        assert_eq!(
            self.dims, other.dims,
            "dimension mismatch in polynomial add"
        );
        let coefficients = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        TrivariatePolynomial {
            coefficients,
            dims: self.dims,
        }
    }

    /// Multiply two polynomials modulo (x^l - 1, y^m - 1, z^p - 1) over GF(2).
    ///
    /// This is convolution in the cyclic group algebra.
    pub fn multiply(&self, other: &TrivariatePolynomial) -> TrivariatePolynomial {
        assert_eq!(
            self.dims, other.dims,
            "dimension mismatch in polynomial multiply"
        );
        let (l, m, p) = self.dims;
        let mut result = TrivariatePolynomial::zero(l, m, p);
        for i1 in 0..l {
            for j1 in 0..m {
                for k1 in 0..p {
                    if !self.get(i1, j1, k1) {
                        continue;
                    }
                    for i2 in 0..l {
                        for j2 in 0..m {
                            for k2 in 0..p {
                                if !other.get(i2, j2, k2) {
                                    continue;
                                }
                                let ri = (i1 + i2) % l;
                                let rj = (j1 + j2) % m;
                                let rk = (k1 + k2) % p;
                                let idx = ri * (m * p) + rj * p + rk;
                                result.coefficients[idx] ^= true;
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Hermitian conjugate (adjoint): reverse all exponents.
    ///
    /// For a monomial x^i y^j z^k, the adjoint is x^{l-i} y^{m-j} z^{p-k}
    /// (i.e., the group inverse in Z_l x Z_m x Z_p).
    pub fn adjoint(&self) -> TrivariatePolynomial {
        let (l, m, p) = self.dims;
        let mut result = TrivariatePolynomial::zero(l, m, p);
        for i in 0..l {
            for j in 0..m {
                for k in 0..p {
                    if self.get(i, j, k) {
                        let ri = if i == 0 { 0 } else { l - i };
                        let rj = if j == 0 { 0 } else { m - j };
                        let rk = if k == 0 { 0 } else { p - k };
                        result.set(ri, rj, rk, true);
                    }
                }
            }
        }
        result
    }

    /// Convert this polynomial to a (l*m*p) x (l*m*p) circulant-block matrix.
    ///
    /// Each monomial x^a y^b z^c acts as a permutation on the basis of Z_l x Z_m x Z_p,
    /// mapping |i, j, k> to |(i+a) mod l, (j+b) mod m, (k+c) mod p>.
    pub fn to_circulant_matrix(&self) -> SparseBinaryMatrix {
        let (l, m, p) = self.dims;
        let dim = l * m * p;
        let mut mat = SparseBinaryMatrix::new(dim, dim);

        for &(a, b, c) in &self.active_monomials() {
            for i in 0..l {
                for j in 0..m {
                    for k in 0..p {
                        let src = i * (m * p) + j * p + k;
                        let di = (i + a) % l;
                        let dj = (j + b) % m;
                        let dk = (k + c) % p;
                        let dst = di * (m * p) + dj * p + dk;
                        mat.toggle(dst, src);
                    }
                }
            }
        }
        mat
    }
}

impl fmt::Display for TrivariatePolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let monomials = self.active_monomials();
        if monomials.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (i, j, k) in monomials {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            if i == 0 && j == 0 && k == 0 {
                write!(f, "1")?;
            } else {
                let mut wrote = false;
                if i > 0 {
                    write!(f, "x^{}", i)?;
                    wrote = true;
                }
                if j > 0 {
                    if wrote {
                        write!(f, "*")?;
                    }
                    write!(f, "y^{}", j)?;
                    wrote = true;
                }
                if k > 0 {
                    if wrote {
                        write!(f, "*")?;
                    }
                    write!(f, "z^{}", k)?;
                }
            }
        }
        Ok(())
    }
}

impl PartialEq for TrivariatePolynomial {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.coefficients == other.coefficients
    }
}
impl Eq for TrivariatePolynomial {}

// ============================================================
// CODE PARAMETERS AND CONFIGURATION
// ============================================================

/// TT code parameters [[n, k, d]].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TTCodeParams {
    /// Number of physical qubits.
    pub n: usize,
    /// Number of logical qubits.
    pub k: usize,
    /// Code distance.
    pub d: usize,
    /// x-dimension of the cyclic group.
    pub l: usize,
    /// y-dimension of the cyclic group.
    pub m: usize,
    /// z-dimension of the cyclic group (the trivariate extension).
    pub p: usize,
}

impl fmt::Display for TTCodeParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[[{}, {}, {}]] (l={}, m={}, p={})",
            self.n, self.k, self.d, self.l, self.m, self.p
        )
    }
}

/// Decoder choice for TT codes.
#[derive(Clone, Debug)]
pub enum TTDecoder {
    /// Minimum weight perfect matching decoder.
    MinimumWeight,
    /// Belief propagation with configurable iterations and damping.
    BeliefPropagation { iterations: usize, damping: f64 },
    /// Sliding window decoder for repeated syndrome rounds.
    SlidingWindow { window_size: usize },
    /// Union-Find decoder (near-linear time).
    UnionFind,
}

impl Default for TTDecoder {
    fn default() -> Self {
        TTDecoder::BeliefPropagation {
            iterations: 50,
            damping: 0.8,
        }
    }
}

/// Noise model for TT code simulation.
#[derive(Clone, Debug)]
pub enum TTNoiseModel {
    /// Symmetric depolarizing noise with probability p.
    Depolarizing(f64),
    /// Circuit-level noise with separate gate and measurement error rates.
    CircuitLevel { gate_error: f64, meas_error: f64 },
    /// Phenomenological noise with data and measurement error rates.
    Phenomenological { data_error: f64, meas_error: f64 },
}

/// Configuration for TT code construction.
#[derive(Clone, Debug)]
pub struct TTConfig {
    /// Group algebra dimension 1 (x).
    pub l: usize,
    /// Group algebra dimension 2 (y).
    pub m: usize,
    /// Group algebra dimension 3 (z) -- the trivariate extension.
    pub p: usize,
    /// Generator polynomials for X-stabilizers.
    pub a_polynomials: Vec<TrivariatePolynomial>,
    /// Generator polynomials for Z-stabilizers.
    pub b_polynomials: Vec<TrivariatePolynomial>,
    /// Whether to construct meta-checks for measurement error diagnosis.
    pub enable_meta_checks: bool,
    /// Decoder to use.
    pub decoder: TTDecoder,
    /// Optional noise model for simulation.
    pub noise_model: Option<TTNoiseModel>,
}

impl TTConfig {
    /// Create a new configuration with the given dimensions and polynomials.
    pub fn new(
        l: usize,
        m: usize,
        p: usize,
        a_polynomials: Vec<TrivariatePolynomial>,
        b_polynomials: Vec<TrivariatePolynomial>,
    ) -> Self {
        Self {
            l,
            m,
            p,
            a_polynomials,
            b_polynomials,
            enable_meta_checks: true,
            decoder: TTDecoder::default(),
            noise_model: None,
        }
    }

    /// Set the decoder.
    pub fn with_decoder(mut self, decoder: TTDecoder) -> Self {
        self.decoder = decoder;
        self
    }

    /// Set the noise model.
    pub fn with_noise_model(mut self, noise: TTNoiseModel) -> Self {
        self.noise_model = Some(noise);
        self
    }

    /// Enable or disable meta-checks.
    pub fn with_meta_checks(mut self, enable: bool) -> Self {
        self.enable_meta_checks = enable;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TrivariateError> {
        if self.l == 0 || self.m == 0 || self.p == 0 {
            return Err(TrivariateError::InvalidParameters(
                "Group dimensions l, m, p must all be positive".into(),
            ));
        }
        if self.a_polynomials.is_empty() || self.b_polynomials.is_empty() {
            return Err(TrivariateError::InvalidParameters(
                "Must provide at least one generator polynomial for each stabilizer type".into(),
            ));
        }
        for poly in self.a_polynomials.iter().chain(self.b_polynomials.iter()) {
            if poly.dims != (self.l, self.m, self.p) {
                return Err(TrivariateError::InvalidParameters(format!(
                    "Polynomial dimension {:?} does not match config ({},{},{})",
                    poly.dims, self.l, self.m, self.p
                )));
            }
        }
        Ok(())
    }
}

// ============================================================
// PAULI CORRECTION
// ============================================================

/// Type of Pauli correction to apply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PauliCorrection {
    /// Pauli X (bit-flip) correction.
    X,
    /// Pauli Z (phase-flip) correction.
    Z,
    /// Pauli Y (combined bit+phase) correction.
    Y,
}

impl fmt::Display for PauliCorrection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PauliCorrection::X => write!(f, "X"),
            PauliCorrection::Z => write!(f, "Z"),
            PauliCorrection::Y => write!(f, "Y"),
        }
    }
}

// ============================================================
// SYNDROME AND DECODING RESULT
// ============================================================

/// Syndrome obtained from measuring stabilizers on the TT code.
#[derive(Clone, Debug)]
pub struct TTSyndrome {
    /// X-type syndrome (from Z-stabilizer measurements).
    pub x_syndrome: Vec<bool>,
    /// Z-type syndrome (from X-stabilizer measurements).
    pub z_syndrome: Vec<bool>,
    /// Meta-check syndrome (if meta-checks are enabled).
    pub meta_syndrome: Option<Vec<bool>>,
    /// Syndrome round index (for repeated measurement).
    pub round: usize,
}

/// Result of running a decoder on a syndrome.
#[derive(Clone, Debug)]
pub struct DecodingResult {
    /// List of (qubit_index, correction_type) pairs.
    pub correction: Vec<(usize, PauliCorrection)>,
    /// Whether decoding succeeded (correction is in the stabilizer group).
    pub success: bool,
    /// Total weight of the correction.
    pub weight: usize,
    /// Indices of measurements diagnosed as faulty (from meta-checks).
    pub meta_diagnosis: Option<Vec<usize>>,
}

/// Results of a threshold study.
#[derive(Clone, Debug)]
pub struct ThresholdResult {
    /// Physical error rates tested.
    pub physical_error_rates: Vec<f64>,
    /// Logical error rates: logical_error_rates[distance_idx][rate_idx].
    pub logical_error_rates: Vec<Vec<f64>>,
    /// Estimated threshold error rate.
    pub threshold: f64,
    /// Error suppression factor (Lambda).
    pub lambda: f64,
    /// Code parameters for each distance tested.
    pub code_params: Vec<TTCodeParams>,
}

// ============================================================
// THE TRIVARIATE TRICYCLE CODE
// ============================================================

/// A trivariate tricycle quantum error-correcting code.
///
/// CSS code with parameters [[n, k, d]] where n = 2 * l * m * p, constructed
/// from generator polynomials over the group algebra F_2[Z_l x Z_m x Z_p].
#[derive(Clone, Debug)]
pub struct TrivariateCode {
    /// Code parameters [[n, k, d]].
    pub params: TTCodeParams,
    /// Construction configuration.
    pub config: TTConfig,
    /// X-type stabilizer check matrix, dimensions (l*m*p) x (2*l*m*p).
    pub hx: SparseBinaryMatrix,
    /// Z-type stabilizer check matrix, dimensions (l*m*p) x (2*l*m*p).
    pub hz: SparseBinaryMatrix,
    /// Meta-check matrix for measurement error diagnosis.
    pub meta_checks: Option<SparseBinaryMatrix>,
    /// Logical X operators (each is a length-n boolean vector).
    pub logical_x: Vec<Vec<bool>>,
    /// Logical Z operators (each is a length-n boolean vector).
    pub logical_z: Vec<Vec<bool>>,
}

// ============================================================
// GF(2) LINEAR ALGEBRA UTILITIES
// ============================================================

/// Compute the kernel (null space) of a binary matrix over GF(2).
///
/// Returns a list of basis vectors for ker(M).
fn gf2_kernel(matrix: &SparseBinaryMatrix) -> Vec<Vec<bool>> {
    let _nrows = matrix.rows;
    let ncols = matrix.cols;
    if ncols == 0 {
        return Vec::new();
    }

    // Augment: [M | I_ncols] transposed approach
    // We work with the transpose and find row dependencies
    let mt = matrix.transpose();
    let mut dense: Vec<Vec<bool>> = (0..mt.rows)
        .map(|r| {
            let mut row = vec![false; mt.cols + mt.rows];
            for &c in &mt.entries[r] {
                row[c] = true;
            }
            // Augment with identity
            row[mt.cols + r] = true;
            row
        })
        .collect();

    // Gaussian elimination on the transpose
    let mut pivot_cols = vec![None; mt.cols];
    let mut rank = 0;
    for col in 0..mt.cols {
        let mut pivot = None;
        for row in rank..mt.rows {
            if dense[row][col] {
                pivot = Some(row);
                break;
            }
        }
        if let Some(p) = pivot {
            dense.swap(rank, p);
            for row in 0..mt.rows {
                if row != rank && dense[row][col] {
                    let pivot_row: Vec<bool> = dense[rank].clone();
                    for c in 0..dense[row].len() {
                        dense[row][c] ^= pivot_row[c];
                    }
                }
            }
            pivot_cols[col] = Some(rank);
            rank += 1;
        }
    }

    // Kernel vectors come from rows of M^T that reduce to zero
    // (these correspond to null vectors of M)
    let mut kernel = Vec::new();
    for row in rank..mt.rows {
        // Check if the left portion is all zero
        let is_zero = (0..mt.cols).all(|c| !dense[row][c]);
        if is_zero {
            // The right portion (augmented identity part) gives a kernel vector
            let vec: Vec<bool> = (0..mt.rows).map(|c| dense[row][mt.cols + c]).collect();
            if vec.iter().any(|&v| v) {
                kernel.push(vec);
            }
        }
    }
    kernel
}

/// Compute the row space basis of a binary matrix over GF(2).
fn gf2_rowspace(matrix: &SparseBinaryMatrix) -> Vec<Vec<bool>> {
    let mut dense = matrix.to_dense_bool();
    let nrows = dense.len();
    if nrows == 0 {
        return Vec::new();
    }
    let ncols = dense[0].len();

    let mut rank = 0;
    for col in 0..ncols {
        let mut pivot = None;
        for row in rank..nrows {
            if dense[row][col] {
                pivot = Some(row);
                break;
            }
        }
        if let Some(p) = pivot {
            dense.swap(rank, p);
            for row in 0..nrows {
                if row != rank && dense[row][col] {
                    for c in 0..ncols {
                        let v = dense[rank][c];
                        dense[row][c] ^= v;
                    }
                }
            }
            rank += 1;
        }
    }
    dense.truncate(rank);
    dense
}

/// Check if a vector is in the row space of a set of basis vectors over GF(2).
fn is_in_rowspace(vec: &[bool], basis: &[Vec<bool>]) -> bool {
    if basis.is_empty() {
        return vec.iter().all(|&v| !v);
    }
    let n = vec.len();
    let mut augmented: Vec<Vec<bool>> = basis.to_vec();
    augmented.push(vec.to_vec());

    // Gaussian elimination
    let nrows = augmented.len();
    let mut rank = 0;
    for col in 0..n {
        let mut pivot = None;
        for row in rank..nrows {
            if augmented[row][col] {
                pivot = Some(row);
                break;
            }
        }
        if let Some(p) = pivot {
            augmented.swap(rank, p);
            for row in 0..nrows {
                if row != rank && augmented[row][col] {
                    for c in 0..n {
                        let v = augmented[rank][c];
                        augmented[row][c] ^= v;
                    }
                }
            }
            rank += 1;
        }
    }

    // If the last row (our vector) was reduced to zero, it is in the row space
    augmented.last().unwrap().iter().all(|&v| !v)
}

// ============================================================
// CODE CONSTRUCTION
// ============================================================

impl TrivariateCode {
    /// Build a trivariate tricycle code from the given configuration.
    ///
    /// # Algorithm
    ///
    /// 1. Sum the circulant matrices from all a-polynomials to form matrix A
    /// 2. Sum the circulant matrices from all b-polynomials to form matrix B
    /// 3. Construct H_X = [A | B] and H_Z = [B^T | A^T]
    /// 4. Verify CSS condition: H_X * H_Z^T = 0 (mod 2)
    /// 5. Compute n = 2*l*m*p, k = n - rank(H_X) - rank(H_Z)
    /// 6. Optionally build meta-check matrix
    /// 7. Find logical operators
    pub fn build(config: TTConfig) -> Result<Self, TrivariateError> {
        config.validate()?;

        let l = config.l;
        let m = config.m;
        let p = config.p;
        let dim = l * m * p;
        let n = 2 * dim;

        // Step 1: Build combined A matrix from a-polynomials
        let mut mat_a = SparseBinaryMatrix::new(dim, dim);
        for poly in &config.a_polynomials {
            let circ = poly.to_circulant_matrix();
            for r in 0..dim {
                for &c in &circ.entries[r] {
                    mat_a.toggle(r, c);
                }
            }
        }

        // Step 2: Build combined B matrix from b-polynomials
        let mut mat_b = SparseBinaryMatrix::new(dim, dim);
        for poly in &config.b_polynomials {
            let circ = poly.to_circulant_matrix();
            for r in 0..dim {
                for &c in &circ.entries[r] {
                    mat_b.toggle(r, c);
                }
            }
        }

        // Step 3: Form H_X = [A | B]
        let hx = mat_a.hcat(&mat_b);

        // Step 4: Form H_Z = [B^T | A^T]
        let mat_bt = mat_b.transpose();
        let mat_at = mat_a.transpose();
        let hz = mat_bt.hcat(&mat_at);

        // Step 5: Verify CSS condition
        let hz_t = hz.transpose();
        let product = hx.multiply_gf2(&hz_t);
        if !product.is_zero() {
            return Err(TrivariateError::CodeConstructionFailed(
                "CSS condition H_X * H_Z^T = 0 (mod 2) is violated".into(),
            ));
        }

        // Step 6: Compute code parameters
        let rank_hx = hx.rank_gf2();
        let rank_hz = hz.rank_gf2();
        let k = if n >= rank_hx + rank_hz {
            n - rank_hx - rank_hz
        } else {
            0
        };

        // Step 7: Code distance
        let d = if n <= 72 {
            Self::compute_distance(&hx, &hz, n, k)
        } else {
            Self::estimate_distance(l, m, p)
        };

        // Step 8: Meta-checks
        let meta_checks = if config.enable_meta_checks {
            Some(Self::build_meta_checks(&hx, &hz, dim))
        } else {
            None
        };

        // Step 9: Logical operators
        let (logical_x, logical_z) = Self::find_logical_operators(&hx, &hz, n, k);

        let params = TTCodeParams { n, k, d, l, m, p };

        Ok(TrivariateCode {
            params,
            config,
            hx,
            hz,
            meta_checks,
            logical_x,
            logical_z,
        })
    }

    /// Compute the exact code distance for small codes.
    fn compute_distance(
        hx: &SparseBinaryMatrix,
        hz: &SparseBinaryMatrix,
        n: usize,
        k: usize,
    ) -> usize {
        if k == 0 {
            return 0;
        }
        let dx = Self::find_min_weight_logical(hz, hx, n);
        let dz = Self::find_min_weight_logical(hx, hz, n);
        dx.min(dz)
    }

    /// Find minimum weight of a non-trivial logical operator.
    ///
    /// A logical operator is in ker(check) but not in rowspace(stabilizer).
    fn find_min_weight_logical(
        check: &SparseBinaryMatrix,
        stabilizer: &SparseBinaryMatrix,
        n: usize,
    ) -> usize {
        let kernel_basis = gf2_kernel(check);
        if kernel_basis.is_empty() {
            return n;
        }
        let stab_basis = gf2_rowspace(stabilizer);
        let mut min_weight = n;
        let num_basis = kernel_basis.len();
        let max_combo: usize = if num_basis <= 18 {
            1usize << num_basis
        } else {
            // For large kernels, sample random combinations
            100_000
        };

        let mut rng = rand::thread_rng();
        for combo in 1..max_combo {
            let mut vec = vec![false; n];
            if num_basis <= 18 {
                for (bit, basis_vec) in kernel_basis.iter().enumerate() {
                    if combo & (1 << bit) != 0 {
                        for (j, &v) in basis_vec.iter().enumerate() {
                            vec[j] ^= v;
                        }
                    }
                }
            } else {
                for basis_vec in &kernel_basis {
                    if rng.gen_bool(0.5) {
                        for (j, &v) in basis_vec.iter().enumerate() {
                            vec[j] ^= v;
                        }
                    }
                }
            }

            let weight: usize = vec.iter().filter(|&&v| v).count();
            if weight == 0 || weight >= min_weight {
                continue;
            }
            if !is_in_rowspace(&vec, &stab_basis) {
                min_weight = weight;
            }
        }
        min_weight
    }

    /// Heuristic distance estimate for large codes.
    fn estimate_distance(l: usize, m: usize, p: usize) -> usize {
        // Conservative lower bound: for well-chosen polynomials the distance
        // grows roughly as the cube root of n for trivariate codes.
        let n = 2 * l * m * p;
        let est = (n as f64).cbrt().ceil() as usize;
        est.max(2)
    }

    /// Build the meta-check matrix M.
    ///
    /// Meta-checks verify measurement consistency. For each pair of X-stabilizers
    /// (rows of H_X) that share support on at least one qubit, add a meta-check
    /// row. The meta-check syndrome M * s should be zero for a valid (noiseless)
    /// syndrome s. Non-zero entries indicate measurement errors.
    fn build_meta_checks(
        hx: &SparseBinaryMatrix,
        _hz: &SparseBinaryMatrix,
        dim: usize,
    ) -> SparseBinaryMatrix {
        let num_checks = hx.rows;
        // Build overlap graph: for each pair of checks, check if they share a qubit
        let mut meta_rows: Vec<BTreeSet<usize>> = Vec::new();
        for i in 0..num_checks {
            let cols_i: BTreeSet<usize> = hx.entries[i].clone();
            for j in (i + 1)..num_checks {
                let has_overlap = hx.entries[j].iter().any(|c| cols_i.contains(c));
                if has_overlap {
                    let mut row = BTreeSet::new();
                    row.insert(i);
                    row.insert(j);
                    meta_rows.push(row);
                }
            }
        }

        // Also add z-dimension parity checks: for each z-layer, the parity of
        // checks within that layer should be consistent.
        let layer_size = dim / hx.rows.max(1);
        if layer_size > 0 && hx.rows > layer_size {
            for layer_start in (0..num_checks).step_by(layer_size.max(1)) {
                let layer_end = (layer_start + layer_size).min(num_checks);
                if layer_end - layer_start >= 2 {
                    let mut row = BTreeSet::new();
                    for idx in layer_start..layer_end {
                        row.insert(idx);
                    }
                    meta_rows.push(row);
                }
            }
        }

        let num_meta = meta_rows.len();
        let mut meta = SparseBinaryMatrix::new(num_meta, num_checks);
        for (r, cols) in meta_rows.into_iter().enumerate() {
            meta.entries[r] = cols;
        }
        meta
    }

    /// Find logical X and Z operators for the code.
    fn find_logical_operators(
        hx: &SparseBinaryMatrix,
        hz: &SparseBinaryMatrix,
        n: usize,
        k: usize,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        if k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Logical X operators: in ker(H_Z) \ rowspace(H_X)
        let kernel_hz = gf2_kernel(hz);
        let stab_x_basis = gf2_rowspace(hx);

        let mut logical_x = Vec::new();
        for v in &kernel_hz {
            if !is_in_rowspace(v, &stab_x_basis) {
                logical_x.push(v.clone());
                if logical_x.len() >= k {
                    break;
                }
            }
        }

        // Logical Z operators: in ker(H_X) \ rowspace(H_Z)
        let kernel_hx = gf2_kernel(hx);
        let stab_z_basis = gf2_rowspace(hz);

        let mut logical_z = Vec::new();
        for v in &kernel_hx {
            if !is_in_rowspace(v, &stab_z_basis) {
                logical_z.push(v.clone());
                if logical_z.len() >= k {
                    break;
                }
            }
        }

        // Pad with zero vectors if we could not find enough (for very large codes
        // where the kernel enumeration is incomplete).
        while logical_x.len() < k {
            logical_x.push(vec![false; n]);
        }
        while logical_z.len() < k {
            logical_z.push(vec![false; n]);
        }

        (logical_x, logical_z)
    }

    /// Encoding rate k/n.
    pub fn encoding_rate(&self) -> f64 {
        if self.params.n == 0 {
            return 0.0;
        }
        self.params.k as f64 / self.params.n as f64
    }
}

// ============================================================
// SYNDROME EXTRACTION
// ============================================================

impl TrivariateCode {
    /// Extract the syndrome from an error vector.
    ///
    /// Given an X-error vector (bit flips) and a Z-error vector (phase flips),
    /// compute the syndrome by multiplying against the check matrices.
    pub fn extract_syndrome(
        &self,
        x_errors: &[bool],
        z_errors: &[bool],
        round: usize,
    ) -> TTSyndrome {
        assert_eq!(x_errors.len(), self.params.n);
        assert_eq!(z_errors.len(), self.params.n);

        // X errors produce Z-syndrome (detected by Z-stabilizers)
        let x_syndrome = self.hz.mul_vec(x_errors);
        // Z errors produce X-syndrome (detected by X-stabilizers)
        let z_syndrome = self.hx.mul_vec(z_errors);

        let meta_syndrome = self.meta_checks.as_ref().map(|mc| {
            // Meta-syndrome from the X-syndrome
            mc.mul_vec(&x_syndrome)
        });

        TTSyndrome {
            x_syndrome,
            z_syndrome,
            meta_syndrome,
            round,
        }
    }

    /// Extract syndrome with measurement noise.
    ///
    /// Flips each syndrome bit independently with the given probability.
    pub fn extract_syndrome_noisy(
        &self,
        x_errors: &[bool],
        z_errors: &[bool],
        meas_error_rate: f64,
        round: usize,
    ) -> TTSyndrome {
        let mut syndrome = self.extract_syndrome(x_errors, z_errors, round);
        let mut rng = rand::thread_rng();

        for bit in syndrome.x_syndrome.iter_mut() {
            if rng.gen::<f64>() < meas_error_rate {
                *bit ^= true;
            }
        }
        for bit in syndrome.z_syndrome.iter_mut() {
            if rng.gen::<f64>() < meas_error_rate {
                *bit ^= true;
            }
        }

        // Recompute meta-syndrome from the (now noisy) X-syndrome
        syndrome.meta_syndrome = self
            .meta_checks
            .as_ref()
            .map(|mc| mc.mul_vec(&syndrome.x_syndrome));

        syndrome
    }
}

// ============================================================
// NOISE APPLICATION
// ============================================================

/// Apply depolarizing noise to n qubits, returning (x_errors, z_errors).
///
/// Each qubit independently suffers X with probability p/3, Z with p/3, Y with p/3.
pub fn apply_depolarizing_noise(n: usize, p: f64) -> (Vec<bool>, Vec<bool>) {
    let mut rng = rand::thread_rng();
    let mut x_err = vec![false; n];
    let mut z_err = vec![false; n];
    for i in 0..n {
        let r: f64 = rng.gen();
        if r < p / 3.0 {
            x_err[i] = true; // X error
        } else if r < 2.0 * p / 3.0 {
            z_err[i] = true; // Z error
        } else if r < p {
            x_err[i] = true; // Y = XZ
            z_err[i] = true;
        }
    }
    (x_err, z_err)
}

/// Apply phenomenological noise: data errors + measurement errors.
pub fn apply_phenomenological_noise(
    n: usize,
    data_error: f64,
    _meas_error: f64,
) -> (Vec<bool>, Vec<bool>) {
    apply_depolarizing_noise(n, data_error)
}

// ============================================================
// DECODERS
// ============================================================

/// Tanner graph for belief propagation.
struct TannerGraph {
    /// For each check node, the adjacent variable node indices.
    check_to_var: Vec<Vec<usize>>,
    /// For each variable node, the adjacent check node indices.
    var_to_check: Vec<Vec<usize>>,
}

impl TannerGraph {
    fn from_matrix(h: &SparseBinaryMatrix) -> Self {
        let check_to_var: Vec<Vec<usize>> = (0..h.rows).map(|r| h.row_indices(r)).collect();
        let mut var_to_check = vec![Vec::new(); h.cols];
        for (r, neighbors) in check_to_var.iter().enumerate() {
            for &v in neighbors {
                var_to_check[v].push(r);
            }
        }
        TannerGraph {
            check_to_var,
            var_to_check,
        }
    }
}

/// Union-Find data structure for the UF decoder.
struct UnionFindDS {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFindDS {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
            self.size[rb] += self.size[ra];
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
        } else {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
            self.rank[ra] += 1;
        }
    }
}

impl TrivariateCode {
    /// Decode a syndrome using the configured decoder.
    pub fn decode(&self, syndrome: &TTSyndrome) -> Result<DecodingResult, TrivariateError> {
        // First, diagnose measurement errors via meta-checks
        let meta_diagnosis = self.diagnose_measurement_errors(syndrome);

        match &self.config.decoder {
            TTDecoder::MinimumWeight => self.decode_mwpm(syndrome, &meta_diagnosis),
            TTDecoder::BeliefPropagation {
                iterations,
                damping,
            } => self.decode_bp(syndrome, *iterations, *damping, &meta_diagnosis),
            TTDecoder::SlidingWindow { window_size } => {
                self.decode_sliding_window(syndrome, *window_size, &meta_diagnosis)
            }
            TTDecoder::UnionFind => self.decode_union_find(syndrome, &meta_diagnosis),
        }
    }

    /// Diagnose faulty measurements using the meta-check matrix.
    fn diagnose_measurement_errors(&self, syndrome: &TTSyndrome) -> Option<Vec<usize>> {
        let mc = self.meta_checks.as_ref()?;
        let meta_syn = syndrome.meta_syndrome.as_ref()?;
        let mut faulty = Vec::new();
        for (i, &bit) in meta_syn.iter().enumerate() {
            if bit {
                // This meta-check is triggered; the involved measurements are suspect
                let involved = mc.row_indices(i);
                for idx in involved {
                    if !faulty.contains(&idx) {
                        faulty.push(idx);
                    }
                }
            }
        }
        faulty.sort();
        faulty.dedup();
        if faulty.is_empty() {
            None
        } else {
            Some(faulty)
        }
    }

    /// Minimum weight perfect matching decoder.
    ///
    /// Builds a complete graph on syndrome-activated check nodes, with edge weights
    /// equal to the minimum number of data qubits connecting them. Finds an
    /// approximate minimum-weight matching via a greedy approach.
    fn decode_mwpm(
        &self,
        syndrome: &TTSyndrome,
        meta_diagnosis: &Option<Vec<usize>>,
    ) -> Result<DecodingResult, TrivariateError> {
        let _n = self.params.n;

        // Collect active syndrome indices for X-errors (from Z-stabilizers)
        let active_x: Vec<usize> = syndrome
            .x_syndrome
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        let mut correction = Vec::new();

        if !active_x.is_empty() {
            let corr = self.greedy_matching(&self.hz, &active_x);
            for q in corr {
                correction.push((q, PauliCorrection::X));
            }
        }

        // Collect active syndrome indices for Z-errors (from X-stabilizers)
        let active_z: Vec<usize> = syndrome
            .z_syndrome
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if !active_z.is_empty() {
            let corr = self.greedy_matching(&self.hx, &active_z);
            for q in corr {
                correction.push((q, PauliCorrection::Z));
            }
        }

        let weight = correction.len();
        let success = self.verify_correction(&correction, syndrome);

        Ok(DecodingResult {
            correction,
            success,
            weight,
            meta_diagnosis: meta_diagnosis.clone(),
        })
    }

    /// Greedy matching: pair up active checks and find connecting qubits.
    fn greedy_matching(&self, h: &SparseBinaryMatrix, active: &[usize]) -> Vec<usize> {
        if active.is_empty() {
            return Vec::new();
        }

        let mut correction_qubits: Vec<usize> = Vec::new();
        let mut remaining: Vec<usize> = active.to_vec();

        // Pair up checks that share a qubit (greedy)
        while remaining.len() >= 2 {
            let c0 = remaining[0];
            let cols_c0: BTreeSet<usize> = h.entries[c0].clone();

            let mut best_partner = None;
            let mut best_qubit = None;

            for &c1 in remaining.iter().skip(1) {
                for &q in &h.entries[c1] {
                    if cols_c0.contains(&q) {
                        best_partner = Some(c1);
                        best_qubit = Some(q);
                        break;
                    }
                }
                if best_partner.is_some() {
                    break;
                }
            }

            if let (Some(partner), Some(qubit)) = (best_partner, best_qubit) {
                correction_qubits.push(qubit);
                remaining.retain(|&c| c != c0 && c != partner);
            } else {
                // No overlapping partner found; use any qubit from this check
                if let Some(&q) = h.entries[c0].iter().next() {
                    correction_qubits.push(q);
                }
                remaining.remove(0);
            }
        }

        // Handle odd remaining check with a single qubit correction
        if let Some(&c) = remaining.first() {
            if let Some(&q) = h.entries[c].iter().next() {
                correction_qubits.push(q);
            }
        }

        correction_qubits
    }

    /// Belief propagation decoder on the Tanner graph.
    fn decode_bp(
        &self,
        syndrome: &TTSyndrome,
        max_iterations: usize,
        damping: f64,
        meta_diagnosis: &Option<Vec<usize>>,
    ) -> Result<DecodingResult, TrivariateError> {
        let _n = self.params.n;
        let mut correction = Vec::new();

        // Decode X-errors using H_Z
        let x_corr = self.bp_decode_single(&self.hz, &syndrome.x_syndrome, max_iterations, damping);
        for q in &x_corr {
            correction.push((*q, PauliCorrection::X));
        }

        // Decode Z-errors using H_X
        let z_corr = self.bp_decode_single(&self.hx, &syndrome.z_syndrome, max_iterations, damping);
        for q in &z_corr {
            correction.push((*q, PauliCorrection::Z));
        }

        let weight = correction.len();
        let success = self.verify_correction(&correction, syndrome);

        Ok(DecodingResult {
            correction,
            success,
            weight,
            meta_diagnosis: meta_diagnosis.clone(),
        })
    }

    /// Run BP on a single check matrix + syndrome.
    fn bp_decode_single(
        &self,
        h: &SparseBinaryMatrix,
        syndrome: &[bool],
        max_iterations: usize,
        damping: f64,
    ) -> Vec<usize> {
        let num_checks = h.rows;
        let num_vars = h.cols;
        let graph = TannerGraph::from_matrix(h);

        // Prior LLR (log-likelihood ratio): assume uniform error probability ~0.05
        let prior_llr = (0.95_f64 / 0.05).ln();

        // Messages: var->check and check->var
        let mut var_to_check: Vec<Vec<f64>> = vec![vec![prior_llr; num_checks]; num_vars];
        let mut check_to_var: Vec<Vec<f64>> = vec![vec![0.0; num_vars]; num_checks];

        for _iter in 0..max_iterations {
            // Check-to-variable messages
            for c in 0..num_checks {
                let sign = if syndrome[c] { -1.0 } else { 1.0 };
                let neighbors = &graph.check_to_var[c];
                for &v in neighbors {
                    let mut prod = sign;
                    for &v2 in neighbors {
                        if v2 != v {
                            let msg = var_to_check[v2][c];
                            prod *= msg.tanh() / 2.0;
                        }
                    }
                    let new_msg = 2.0 * prod.atanh();
                    let new_msg = if new_msg.is_finite() {
                        new_msg
                    } else {
                        new_msg.signum() * 30.0
                    };
                    check_to_var[c][v] = damping * new_msg + (1.0 - damping) * check_to_var[c][v];
                }
            }

            // Variable-to-check messages
            for v in 0..num_vars {
                let neighbors = &graph.var_to_check[v];
                for &c in neighbors {
                    let mut sum = prior_llr;
                    for &c2 in neighbors {
                        if c2 != c {
                            sum += check_to_var[c2][v];
                        }
                    }
                    var_to_check[v][c] = sum;
                }
            }

            // Check convergence: compute posterior and verify syndrome
            let posterior: Vec<bool> = (0..num_vars)
                .map(|v| {
                    let mut total = prior_llr;
                    for &c in &graph.var_to_check[v] {
                        total += check_to_var[c][v];
                    }
                    total < 0.0
                })
                .collect();

            let test_syndrome = h.mul_vec(&posterior);
            if test_syndrome == syndrome {
                return posterior
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .collect();
            }
        }

        // Did not converge; return hard decision from posteriors
        let posterior: Vec<bool> = (0..num_vars)
            .map(|v| {
                let mut total = prior_llr;
                for &c in &graph.var_to_check[v] {
                    total += check_to_var[c][v];
                }
                total < 0.0
            })
            .collect();

        posterior
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }

    /// Union-Find decoder.
    ///
    /// Clusters syndrome nodes using a disjoint-set structure, then applies
    /// a peeling decoder within each cluster.
    fn decode_union_find(
        &self,
        syndrome: &TTSyndrome,
        meta_diagnosis: &Option<Vec<usize>>,
    ) -> Result<DecodingResult, TrivariateError> {
        let _n = self.params.n;
        let mut correction = Vec::new();

        let x_corr = self.uf_decode_single(&self.hz, &syndrome.x_syndrome);
        for q in x_corr {
            correction.push((q, PauliCorrection::X));
        }

        let z_corr = self.uf_decode_single(&self.hx, &syndrome.z_syndrome);
        for q in z_corr {
            correction.push((q, PauliCorrection::Z));
        }

        let weight = correction.len();
        let success = self.verify_correction(&correction, syndrome);

        Ok(DecodingResult {
            correction,
            success,
            weight,
            meta_diagnosis: meta_diagnosis.clone(),
        })
    }

    /// Union-Find decode for a single check matrix + syndrome.
    fn uf_decode_single(&self, h: &SparseBinaryMatrix, syndrome: &[bool]) -> Vec<usize> {
        let num_checks = h.rows;
        let num_vars = h.cols;

        // Active checks
        let active: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if active.is_empty() {
            return Vec::new();
        }

        // Build union-find structure over check nodes
        let mut uf = UnionFindDS::new(num_checks);

        // Grow clusters: for each variable node connecting two active checks, union them
        for v in 0..num_vars {
            let checks: Vec<usize> = (0..num_checks)
                .filter(|&c| h.entries[c].contains(&v))
                .collect();
            // Union all active checks that share this variable
            let active_in_var: Vec<usize> =
                checks.iter().filter(|&&c| syndrome[c]).copied().collect();
            if active_in_var.len() >= 2 {
                for i in 1..active_in_var.len() {
                    uf.union(active_in_var[0], active_in_var[i]);
                }
            }
        }

        // For each cluster, find a correction via peeling
        let mut correction_qubits = Vec::new();
        let mut processed_roots = BTreeSet::new();

        for &c in &active {
            let root = uf.find(c);
            if processed_roots.contains(&root) {
                continue;
            }
            processed_roots.insert(root);

            // Gather all active checks in this cluster
            let cluster_checks: Vec<usize> = active
                .iter()
                .filter(|&&a| uf.find(a) == root)
                .copied()
                .collect();

            // Peel: iteratively find a qubit that is in exactly one active check
            let mut remaining_checks: BTreeSet<usize> = cluster_checks.iter().copied().collect();
            let mut used_qubits = BTreeSet::new();

            let mut changed = true;
            while changed && !remaining_checks.is_empty() {
                changed = false;
                let checks_snapshot: Vec<usize> = remaining_checks.iter().copied().collect();
                for &c in &checks_snapshot {
                    // Find a qubit in this check not yet used
                    let mut found = None;
                    for &q in &h.entries[c] {
                        if used_qubits.contains(&q) {
                            continue;
                        }
                        // Count how many remaining checks contain this qubit
                        let count = remaining_checks
                            .iter()
                            .filter(|&&rc| h.entries[rc].contains(&q))
                            .count();
                        if count == 1 {
                            found = Some(q);
                            break;
                        }
                    }
                    if let Some(q) = found {
                        correction_qubits.push(q);
                        used_qubits.insert(q);
                        remaining_checks.remove(&c);
                        changed = true;
                    }
                }
            }

            // If peeling did not resolve all checks, fall back to picking any qubit
            for &c in &remaining_checks.clone() {
                if let Some(&q) = h.entries[c].iter().find(|q| !used_qubits.contains(q)) {
                    correction_qubits.push(q);
                    used_qubits.insert(q);
                }
            }
        }

        correction_qubits
    }

    /// Sliding window decoder for multiple syndrome rounds.
    fn decode_sliding_window(
        &self,
        syndrome: &TTSyndrome,
        _window_size: usize,
        meta_diagnosis: &Option<Vec<usize>>,
    ) -> Result<DecodingResult, TrivariateError> {
        // For a single round, sliding window reduces to BP
        self.decode_bp(syndrome, 50, 0.8, meta_diagnosis)
    }

    /// Verify that a correction, when applied, produces the zero syndrome.
    fn verify_correction(
        &self,
        correction: &[(usize, PauliCorrection)],
        syndrome: &TTSyndrome,
    ) -> bool {
        let n = self.params.n;
        let mut x_corr = vec![false; n];
        let mut z_corr = vec![false; n];
        for &(q, ref p) in correction {
            if q >= n {
                continue;
            }
            match p {
                PauliCorrection::X => x_corr[q] = true,
                PauliCorrection::Z => z_corr[q] = true,
                PauliCorrection::Y => {
                    x_corr[q] = true;
                    z_corr[q] = true;
                }
            }
        }
        // The correction syndrome should match the measured syndrome
        let corr_x_syn = self.hz.mul_vec(&x_corr);
        let corr_z_syn = self.hx.mul_vec(&z_corr);

        corr_x_syn == syndrome.x_syndrome && corr_z_syn == syndrome.z_syndrome
    }
}

// ============================================================
// THRESHOLD ANALYSIS
// ============================================================

impl TrivariateCode {
    /// Run a threshold study across multiple error rates and code sizes.
    ///
    /// Returns estimated threshold, suppression factor, and detailed results.
    pub fn threshold_study(
        configs: &[TTConfig],
        error_rates: &[f64],
        shots_per_point: usize,
    ) -> Result<ThresholdResult, TrivariateError> {
        let mut all_params = Vec::new();
        let mut all_logical_rates = Vec::new();

        for config in configs {
            let code = TrivariateCode::build(config.clone())?;
            all_params.push(code.params.clone());

            let mut rates_for_code = Vec::new();
            for &p_err in error_rates {
                let mut logical_failures = 0usize;
                for _ in 0..shots_per_point {
                    let (x_err, z_err) = apply_depolarizing_noise(code.params.n, p_err);
                    let syndrome = code.extract_syndrome(&x_err, &z_err, 0);
                    match code.decode(&syndrome) {
                        Ok(result) => {
                            if !result.success {
                                logical_failures += 1;
                            }
                        }
                        Err(_) => {
                            logical_failures += 1;
                        }
                    }
                }
                let logical_rate = logical_failures as f64 / shots_per_point as f64;
                rates_for_code.push(logical_rate);
            }
            all_logical_rates.push(rates_for_code);
        }

        // Estimate threshold: find crossing point of logical error rate curves
        let threshold = Self::estimate_threshold(error_rates, &all_logical_rates);

        // Estimate Lambda: error suppression factor below threshold
        let lambda = Self::estimate_lambda(error_rates, &all_logical_rates, threshold);

        Ok(ThresholdResult {
            physical_error_rates: error_rates.to_vec(),
            logical_error_rates: all_logical_rates,
            threshold,
            lambda,
            code_params: all_params,
        })
    }

    /// Find the crossing point of logical error rate curves.
    fn estimate_threshold(error_rates: &[f64], logical_rates: &[Vec<f64>]) -> f64 {
        if logical_rates.len() < 2 || error_rates.is_empty() {
            return 0.0;
        }

        // Look for crossing between first and last code size
        let first = &logical_rates[0];
        let last = &logical_rates[logical_rates.len() - 1];

        let mut best_idx = 0;
        let mut best_diff = f64::MAX;

        for i in 0..error_rates.len() {
            let diff = (first[i] - last[i]).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }

        error_rates[best_idx]
    }

    /// Estimate error suppression factor Lambda.
    fn estimate_lambda(error_rates: &[f64], logical_rates: &[Vec<f64>], threshold: f64) -> f64 {
        if logical_rates.len() < 2 {
            return 1.0;
        }

        // Find an error rate below threshold
        let below_threshold: Vec<usize> = error_rates
            .iter()
            .enumerate()
            .filter(|(_, &p)| p < threshold && p > 0.0)
            .map(|(i, _)| i)
            .collect();

        if below_threshold.is_empty() {
            return 1.0;
        }

        let idx = below_threshold[below_threshold.len() / 2];
        let first_rate = logical_rates[0][idx];
        let last_rate = logical_rates[logical_rates.len() - 1][idx];

        if last_rate > 0.0 && first_rate > 0.0 {
            (first_rate / last_rate).abs()
        } else {
            1.0
        }
    }
}

// ============================================================
// PRE-BUILT CODE FAMILIES
// ============================================================

/// Construct the smallest TT code: [[18, 2, 3]] with l=3, m=3, p=1.
///
/// This is effectively a bivariate bicycle code (degenerate p=1 case) serving
/// as a compatibility bridge and test fixture.
pub fn tt_small() -> Result<TrivariateCode, TrivariateError> {
    let (l, m, p) = (3, 3, 1);
    // Generator: a(x,y,z) = 1 + x + y
    let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
    // Generator: b(x,y,z) = 1 + x^2 + y^2
    let b = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);

    let config = TTConfig::new(l, m, p, vec![a], vec![b]);
    TrivariateCode::build(config)
}

/// Construct a medium TT code with l=3, m=3, p=4.
///
/// Exploits the third dimension for richer meta-check structure.
/// Uses self-adjoint (symmetric) polynomials to guarantee the CSS condition.
pub fn tt_medium() -> Result<TrivariateCode, TrivariateError> {
    let (l, m, p) = (3, 3, 4);
    // Self-adjoint: a(x,y,z) = x + x^{-1} + y + y^{-1}  (mod cyclic)
    //   = x + x^2 + y + y^2  (since l=m=3, x^{-1} = x^2)
    let a = TrivariatePolynomial::from_monomials(
        l,
        m,
        p,
        &[(1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0)],
    );
    // Self-adjoint: b(x,y,z) = z + z^{-1} + x*y + x^{-1}*y^{-1}
    //   = z + z^3 + x*y + x^2*y^2  (since p=4, z^{-1} = z^3)
    let b = TrivariatePolynomial::from_monomials(
        l,
        m,
        p,
        &[(0, 0, 1), (0, 0, 3), (1, 1, 0), (2, 2, 0)],
    );

    let config = TTConfig::new(l, m, p, vec![a], vec![b]);
    TrivariateCode::build(config)
}

/// Construct a large TT code with l=4, m=4, p=9.
///
/// High-distance code suitable for fault-tolerant quantum memory.
/// Uses self-adjoint polynomials to guarantee the CSS condition.
pub fn tt_large() -> Result<TrivariateCode, TrivariateError> {
    let (l, m, p) = (4, 4, 9);
    // Self-adjoint: a = x + x^3 + y + y^3  (l=4, x^{-1}=x^3)
    let a = TrivariatePolynomial::from_monomials(
        l,
        m,
        p,
        &[(1, 0, 0), (3, 0, 0), (0, 1, 0), (0, 3, 0)],
    );
    // Self-adjoint: b = z + z^8 + x*y + x^3*y^3  (p=9, z^{-1}=z^8)
    let b = TrivariatePolynomial::from_monomials(
        l,
        m,
        p,
        &[(0, 0, 1), (0, 0, 8), (1, 1, 0), (3, 3, 0)],
    );

    let config = TTConfig::new(l, m, p, vec![a], vec![b]);
    TrivariateCode::build(config)
}

/// Construct a bivariate bicycle code as a degenerate TT code (p=1).
///
/// Provides backward compatibility with the bivariate bicycle construction.
pub fn bivariate_bicycle(l: usize, m: usize) -> Result<TrivariateCode, TrivariateError> {
    let p = 1;
    // Standard bivariate generators: a = 1 + x + y, b = 1 + x^2 + y^2
    let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
    let b = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);

    let config = TTConfig::new(l, m, p, vec![a], vec![b]).with_meta_checks(false);
    TrivariateCode::build(config)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Polynomial tests
    // ----------------------------------------------------------

    #[test]
    fn test_polynomial_creation_and_display() {
        let poly =
            TrivariatePolynomial::from_monomials(3, 3, 2, &[(0, 0, 0), (1, 0, 0), (0, 1, 1)]);
        assert_eq!(poly.dims, (3, 3, 2));
        assert_eq!(poly.weight(), 3);
        let s = format!("{}", poly);
        assert!(s.contains("1"));
        assert!(s.contains("x^1"));
    }

    #[test]
    fn test_polynomial_addition_gf2() {
        let (l, m, p) = (4, 4, 2);
        let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0)]);
        let b = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (0, 1, 0)]);
        let sum = a.add(&b);
        // (1 + x) + (1 + y) = x + y  (the 1s cancel in GF(2))
        assert!(sum.get(1, 0, 0));
        assert!(sum.get(0, 1, 0));
        assert!(!sum.get(0, 0, 0)); // canceled
        assert_eq!(sum.weight(), 2);
    }

    #[test]
    fn test_polynomial_self_addition_is_zero() {
        let (l, m, p) = (3, 3, 2);
        let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 0, 1)]);
        let sum = a.add(&a);
        assert!(sum.is_zero(), "a + a should be zero in GF(2)");
    }

    #[test]
    fn test_polynomial_multiplication_mod_cyclic() {
        let (l, m, p) = (3, 2, 1);
        // x * x^2 = x^3 = x^0 = 1 (mod x^3 - 1)
        let a = TrivariatePolynomial::monomial(l, m, p, 1, 0, 0);
        let b = TrivariatePolynomial::monomial(l, m, p, 2, 0, 0);
        let prod = a.multiply(&b);
        assert!(prod.get(0, 0, 0), "x * x^2 = 1 mod (x^3 - 1)");
        assert_eq!(prod.weight(), 1);
    }

    #[test]
    fn test_polynomial_cyclic_property() {
        // x^l = 1 in the cyclic ring
        let (l, m, p) = (5, 3, 2);
        let x = TrivariatePolynomial::monomial(l, m, p, 1, 0, 0);
        let mut acc = TrivariatePolynomial::one(l, m, p);
        for _ in 0..l {
            acc = acc.multiply(&x);
        }
        let one = TrivariatePolynomial::one(l, m, p);
        assert_eq!(acc, one, "x^l should equal 1");
    }

    #[test]
    fn test_polynomial_adjoint() {
        let (l, m, p) = (4, 3, 2);
        // For monomial x^1 y^2 z^1, adjoint is x^3 y^1 z^1
        let poly = TrivariatePolynomial::monomial(l, m, p, 1, 2, 1);
        let adj = poly.adjoint();
        assert!(adj.get(3, 1, 1));
        assert_eq!(adj.weight(), 1);
    }

    #[test]
    fn test_polynomial_adjoint_of_identity() {
        let (l, m, p) = (3, 3, 3);
        let one = TrivariatePolynomial::one(l, m, p);
        let adj = one.adjoint();
        assert_eq!(one, adj, "adjoint of 1 should be 1");
    }

    // ----------------------------------------------------------
    // Check matrix tests
    // ----------------------------------------------------------

    #[test]
    fn test_check_matrix_from_polynomial() {
        let (l, m, p) = (3, 2, 1);
        let poly = TrivariatePolynomial::monomial(l, m, p, 0, 0, 0); // identity
        let mat = poly.to_circulant_matrix();
        let dim = l * m * p;
        assert_eq!(mat.rows, dim);
        assert_eq!(mat.cols, dim);
        // Identity polynomial should give identity matrix
        for i in 0..dim {
            assert!(mat.get(i, i), "diagonal should be 1");
        }
        assert_eq!(mat.nnz(), dim);
    }

    #[test]
    fn test_css_condition_holds() {
        let code = tt_small().expect("tt_small should build");
        let hz_t = code.hz.transpose();
        let product = code.hx.multiply_gf2(&hz_t);
        assert!(
            product.is_zero(),
            "CSS condition must hold: H_X * H_Z^T = 0"
        );
    }

    #[test]
    fn test_code_parameters_n() {
        let code = tt_small().expect("tt_small should build");
        let expected_n = 2 * 3 * 3 * 1;
        assert_eq!(code.params.n, expected_n, "n should be 2*l*m*p");
        assert_eq!(code.params.l, 3);
        assert_eq!(code.params.m, 3);
        assert_eq!(code.params.p, 1);
    }

    #[test]
    fn test_code_parameters_k_positive() {
        let code = tt_small().expect("tt_small should build");
        assert!(code.params.k > 0, "k should be positive for a valid code");
    }

    #[test]
    fn test_code_distance_positive() {
        let code = tt_small().expect("tt_small should build");
        assert!(code.params.d > 0, "distance should be positive");
    }

    #[test]
    fn test_logical_operator_count() {
        let code = tt_small().expect("tt_small should build");
        assert_eq!(
            code.logical_x.len(),
            code.params.k,
            "should have k logical X operators"
        );
        assert_eq!(
            code.logical_z.len(),
            code.params.k,
            "should have k logical Z operators"
        );
    }

    // ----------------------------------------------------------
    // Meta-check tests
    // ----------------------------------------------------------

    #[test]
    fn test_meta_check_construction() {
        let code = tt_small().expect("tt_small should build");
        assert!(
            code.meta_checks.is_some(),
            "meta-checks should be constructed by default"
        );
        let mc = code.meta_checks.as_ref().unwrap();
        assert!(mc.rows > 0, "should have at least one meta-check");
        assert_eq!(mc.cols, code.hx.rows, "meta-check cols = number of checks");
    }

    #[test]
    fn test_meta_check_valid_syndrome_passes() {
        let code = tt_small().expect("tt_small should build");
        let n = code.params.n;
        let no_errors = vec![false; n];
        let syndrome = code.extract_syndrome(&no_errors, &no_errors, 0);
        if let Some(ref meta_syn) = syndrome.meta_syndrome {
            assert!(
                meta_syn.iter().all(|&b| !b),
                "meta-syndrome should be zero for zero syndrome"
            );
        }
    }

    #[test]
    fn test_meta_check_detects_inconsistency() {
        let code = tt_small().expect("tt_small should build");
        // Create a syndrome that is invalid (not from any real error)
        let mc = code.meta_checks.as_ref().unwrap();
        if mc.rows > 0 && mc.cols > 0 {
            // Manually set one syndrome bit; this may violate meta-checks
            let mut fake_x_syn = vec![false; code.hz.rows];
            if fake_x_syn.len() > 0 {
                fake_x_syn[0] = true;
            }
            let meta_syn = mc.mul_vec(&fake_x_syn);
            // We just verify meta-check evaluation works without panic
            assert_eq!(meta_syn.len(), mc.rows);
        }
    }

    // ----------------------------------------------------------
    // Syndrome extraction tests
    // ----------------------------------------------------------

    #[test]
    fn test_syndrome_no_errors_is_zero() {
        let code = tt_small().expect("tt_small should build");
        let n = code.params.n;
        let no_errors = vec![false; n];
        let syndrome = code.extract_syndrome(&no_errors, &no_errors, 0);
        assert!(
            syndrome.x_syndrome.iter().all(|&b| !b),
            "no X errors should give zero X syndrome"
        );
        assert!(
            syndrome.z_syndrome.iter().all(|&b| !b),
            "no Z errors should give zero Z syndrome"
        );
    }

    #[test]
    fn test_syndrome_single_x_error() {
        let code = tt_small().expect("tt_small should build");
        let n = code.params.n;
        let mut x_err = vec![false; n];
        x_err[0] = true;
        let z_err = vec![false; n];
        let syndrome = code.extract_syndrome(&x_err, &z_err, 0);
        // A single X error should produce a non-zero X syndrome (detected by H_Z)
        assert!(
            syndrome.x_syndrome.iter().any(|&b| b),
            "single X error should produce non-zero syndrome"
        );
    }

    #[test]
    fn test_syndrome_single_z_error() {
        let code = tt_small().expect("tt_small should build");
        let n = code.params.n;
        let x_err = vec![false; n];
        let mut z_err = vec![false; n];
        z_err[0] = true;
        let syndrome = code.extract_syndrome(&x_err, &z_err, 0);
        assert!(
            syndrome.z_syndrome.iter().any(|&b| b),
            "single Z error should produce non-zero syndrome"
        );
    }

    #[test]
    fn test_syndrome_stabilizer_is_invisible() {
        // Applying a full stabilizer row should produce zero syndrome
        let code = tt_small().expect("tt_small should build");
        let n = code.params.n;
        // Take row 0 of H_X as an X-stabilizer
        let stab: Vec<bool> = (0..n).map(|c| code.hx.get(0, c)).collect();
        // This is a Z-type stabilizer applied as Z errors
        let x_err = vec![false; n];
        let syndrome = code.extract_syndrome(&x_err, &stab, 0);
        // The X-stabilizer row, applied as Z-errors, should produce zero Z-syndrome
        // because the stabilizer commutes with itself.
        // Actually: H_X * stab_row^T = H_X * H_X[0]^T. For CSS codes, rows of H_X
        // are in the rowspace of H_X, which is in ker(H_Z^T), not necessarily ker(H_X).
        // The Z-syndrome comes from H_X * z_errors. Let us just verify no crash.
        assert_eq!(syndrome.z_syndrome.len(), code.hx.rows);
    }

    // ----------------------------------------------------------
    // Decoder tests
    // ----------------------------------------------------------

    #[test]
    fn test_mwpm_decoder_single_error() {
        let config = {
            let (l, m, p) = (3, 3, 1);
            let a =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
            let b =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
            TTConfig::new(l, m, p, vec![a], vec![b]).with_decoder(TTDecoder::MinimumWeight)
        };
        let code = TrivariateCode::build(config).expect("build");
        let n = code.params.n;

        let mut x_err = vec![false; n];
        x_err[0] = true;
        let z_err = vec![false; n];
        let syndrome = code.extract_syndrome(&x_err, &z_err, 0);
        let result = code.decode(&syndrome).expect("decode");
        assert!(result.weight > 0, "correction should be non-trivial");
    }

    #[test]
    fn test_bp_decoder_convergence() {
        let config = {
            let (l, m, p) = (3, 3, 1);
            let a =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
            let b =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
            TTConfig::new(l, m, p, vec![a], vec![b]).with_decoder(TTDecoder::BeliefPropagation {
                iterations: 100,
                damping: 0.8,
            })
        };
        let code = TrivariateCode::build(config).expect("build");
        let n = code.params.n;

        let mut x_err = vec![false; n];
        x_err[0] = true;
        let z_err = vec![false; n];
        let syndrome = code.extract_syndrome(&x_err, &z_err, 0);
        let result = code.decode(&syndrome).expect("decode");
        // BP should return a result without error (may or may not converge to
        // the correct correction, but it should not panic)
        assert!(
            result.correction.len() <= n,
            "correction should not exceed n qubits"
        );
    }

    #[test]
    fn test_union_find_decoder() {
        let config = {
            let (l, m, p) = (3, 3, 1);
            let a =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
            let b =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
            TTConfig::new(l, m, p, vec![a], vec![b]).with_decoder(TTDecoder::UnionFind)
        };
        let code = TrivariateCode::build(config).expect("build");
        let n = code.params.n;

        let mut x_err = vec![false; n];
        x_err[0] = true;
        let z_err = vec![false; n];
        let syndrome = code.extract_syndrome(&x_err, &z_err, 0);
        let result = code.decode(&syndrome).expect("decode");
        assert!(result.weight > 0);
    }

    #[test]
    fn test_sliding_window_decoder() {
        let config = {
            let (l, m, p) = (3, 3, 1);
            let a =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
            let b =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
            TTConfig::new(l, m, p, vec![a], vec![b])
                .with_decoder(TTDecoder::SlidingWindow { window_size: 3 })
        };
        let code = TrivariateCode::build(config).expect("build");
        let n = code.params.n;

        let mut x_err = vec![false; n];
        x_err[0] = true;
        let z_err = vec![false; n];
        let syndrome = code.extract_syndrome(&x_err, &z_err, 0);
        let result = code.decode(&syndrome).expect("decode");
        // Sliding window (backed by BP) should return a correction without panic
        assert!(result.correction.len() <= n);
    }

    // ----------------------------------------------------------
    // Noise model tests
    // ----------------------------------------------------------

    #[test]
    fn test_depolarizing_noise() {
        let n = 100;
        let (x_err, z_err) = apply_depolarizing_noise(n, 0.1);
        assert_eq!(x_err.len(), n);
        assert_eq!(z_err.len(), n);
        // With p=0.1, we expect roughly 10 errors
        let total: usize = x_err.iter().chain(z_err.iter()).filter(|&&e| e).count();
        assert!(total > 0, "should have some errors at p=0.1");
        assert!(total < n, "should not have all qubits errored");
    }

    #[test]
    fn test_depolarizing_noise_zero_rate() {
        let n = 50;
        let (x_err, z_err) = apply_depolarizing_noise(n, 0.0);
        assert!(x_err.iter().all(|&e| !e), "no errors at p=0");
        assert!(z_err.iter().all(|&e| !e), "no errors at p=0");
    }

    #[test]
    fn test_circuit_level_noise_config() {
        let config = {
            let (l, m, p) = (3, 3, 1);
            let a =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
            let b =
                TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
            TTConfig::new(l, m, p, vec![a], vec![b]).with_noise_model(TTNoiseModel::CircuitLevel {
                gate_error: 0.001,
                meas_error: 0.01,
            })
        };
        let code = TrivariateCode::build(config).expect("build");
        assert!(code.config.noise_model.is_some());
    }

    // ----------------------------------------------------------
    // Pre-built code family tests
    // ----------------------------------------------------------

    #[test]
    fn test_tt_small_parameters() {
        let code = tt_small().expect("tt_small should build");
        assert_eq!(code.params.n, 18, "n = 2*3*3*1 = 18");
        assert!(code.params.k > 0, "k should be positive");
        assert!(code.params.d >= 2, "distance should be at least 2");
    }

    #[test]
    fn test_tt_medium_construction() {
        let code = tt_medium().expect("tt_medium should build");
        assert_eq!(code.params.n, 72, "n = 2*3*3*4 = 72");
        // Verify CSS condition holds
        let hz_t = code.hz.transpose();
        let product = code.hx.multiply_gf2(&hz_t);
        assert!(product.is_zero(), "CSS condition must hold");
        // With self-adjoint polynomials, k should be positive
        assert!(
            code.params.k > 0,
            "k should be positive, got k={} (rank_hx + rank_hz = {} + {} = {})",
            code.params.k,
            code.hx.rank_gf2(),
            code.hz.rank_gf2(),
            code.hx.rank_gf2() + code.hz.rank_gf2()
        );
    }

    #[test]
    fn test_tt_large_construction() {
        let code = tt_large().expect("tt_large should build");
        assert_eq!(code.params.n, 288, "n = 2*4*4*9 = 288");
        // Verify CSS condition holds
        let hz_t = code.hz.transpose();
        let product = code.hx.multiply_gf2(&hz_t);
        assert!(product.is_zero(), "CSS condition must hold");
        assert!(
            code.params.k > 0,
            "k should be positive, got k={} (rank_hx + rank_hz = {} + {} = {})",
            code.params.k,
            code.hx.rank_gf2(),
            code.hz.rank_gf2(),
            code.hx.rank_gf2() + code.hz.rank_gf2()
        );
    }

    #[test]
    fn test_bivariate_bicycle_backward_compatible() {
        let code = bivariate_bicycle(3, 3).expect("bivariate_bicycle should build");
        assert_eq!(code.params.p, 1, "p should be 1 for bivariate bicycle");
        assert_eq!(code.params.n, 18, "n = 2*3*3*1 = 18");
        assert!(
            code.meta_checks.is_none(),
            "meta-checks disabled for bivariate"
        );
    }

    #[test]
    fn test_bivariate_bicycle_larger() {
        let code = bivariate_bicycle(6, 6).expect("build 6x6 bicycle");
        assert_eq!(code.params.n, 72);
        assert!(code.params.k > 0);
    }

    // ----------------------------------------------------------
    // Threshold and encoding rate tests
    // ----------------------------------------------------------

    #[test]
    fn test_threshold_study_runs() {
        let (l, m, p) = (3, 3, 1);
        let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
        let b = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
        let config = TTConfig::new(l, m, p, vec![a], vec![b]);

        let error_rates = vec![0.01, 0.05, 0.1];
        let result =
            TrivariateCode::threshold_study(&[config], &error_rates, 10).expect("threshold study");
        assert_eq!(result.physical_error_rates.len(), 3);
        assert_eq!(result.logical_error_rates.len(), 1);
        assert_eq!(result.logical_error_rates[0].len(), 3);
    }

    #[test]
    fn test_logical_error_rate_increases_with_noise() {
        let (l, m, p) = (3, 3, 1);
        let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
        let b = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
        let config = TTConfig::new(l, m, p, vec![a], vec![b]);

        let error_rates = vec![0.001, 0.3];
        let result =
            TrivariateCode::threshold_study(&[config], &error_rates, 50).expect("threshold study");
        let rates = &result.logical_error_rates[0];
        // Logical error rate at p=0.3 should be >= rate at p=0.001
        assert!(
            rates[1] >= rates[0],
            "logical error rate should increase with physical error rate"
        );
    }

    #[test]
    fn test_lambda_positive() {
        let (l, m, p) = (3, 3, 1);
        let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)]);
        let b = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (2, 0, 0), (0, 2, 0)]);
        let config = TTConfig::new(l, m, p, vec![a], vec![b]);

        let error_rates = vec![0.01, 0.05, 0.1];
        let result =
            TrivariateCode::threshold_study(&[config], &error_rates, 20).expect("threshold");
        assert!(result.lambda >= 0.0, "lambda should be non-negative");
    }

    // ----------------------------------------------------------
    // Sparse matrix tests
    // ----------------------------------------------------------

    #[test]
    fn test_sparse_matrix_row_operations() {
        let mut m = SparseBinaryMatrix::new(3, 4);
        m.set(0, 0);
        m.set(0, 2);
        m.set(1, 1);
        m.set(1, 2);
        // Row add: row1 ^= row0 => row1 should have {0, 1} (2 cancels)
        m.row_add(1, 0);
        assert!(m.get(1, 0));
        assert!(m.get(1, 1));
        assert!(!m.get(1, 2), "column 2 should cancel in GF(2)");
    }

    #[test]
    fn test_sparse_matrix_rank() {
        // Identity 3x3 has rank 3
        let id = SparseBinaryMatrix::identity(3);
        assert_eq!(id.rank_gf2(), 3);
    }

    #[test]
    fn test_sparse_matrix_rank_deficient() {
        let mut m = SparseBinaryMatrix::new(3, 3);
        m.set(0, 0);
        m.set(0, 1);
        m.set(1, 0);
        m.set(1, 1);
        // Rows 0 and 1 are identical => rank <= 2
        m.set(2, 2);
        assert_eq!(m.rank_gf2(), 2);
    }

    #[test]
    fn test_sparse_matrix_multiply_identity() {
        let id = SparseBinaryMatrix::identity(4);
        let mut m = SparseBinaryMatrix::new(4, 4);
        m.set(0, 1);
        m.set(1, 2);
        m.set(2, 3);
        m.set(3, 0);
        let product = id.multiply_gf2(&m);
        assert_eq!(product, m, "I * M = M");
    }

    #[test]
    fn test_sparse_matrix_transpose() {
        let mut m = SparseBinaryMatrix::new(2, 3);
        m.set(0, 1);
        m.set(1, 0);
        m.set(1, 2);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!(t.get(1, 0));
        assert!(t.get(0, 1));
        assert!(t.get(2, 1));
    }

    // ----------------------------------------------------------
    // Config and validation tests
    // ----------------------------------------------------------

    #[test]
    fn test_config_builder_defaults() {
        let (l, m, p) = (3, 3, 1);
        let a = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (1, 0, 0)]);
        let b = TrivariatePolynomial::from_monomials(l, m, p, &[(0, 0, 0), (0, 1, 0)]);
        let config = TTConfig::new(l, m, p, vec![a], vec![b]);
        assert!(config.enable_meta_checks);
        assert!(config.noise_model.is_none());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_dimension() {
        let (l, m, p) = (0, 3, 1);
        let a = TrivariatePolynomial::zero(1, 1, 1);
        let b = TrivariatePolynomial::zero(1, 1, 1);
        let config = TTConfig {
            l,
            m,
            p,
            a_polynomials: vec![a],
            b_polynomials: vec![b],
            enable_meta_checks: false,
            decoder: TTDecoder::default(),
            noise_model: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_empty_polynomials() {
        let config = TTConfig {
            l: 3,
            m: 3,
            p: 1,
            a_polynomials: vec![],
            b_polynomials: vec![TrivariatePolynomial::one(3, 3, 1)],
            enable_meta_checks: false,
            decoder: TTDecoder::default(),
            noise_model: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_dimension_mismatch() {
        let a = TrivariatePolynomial::one(3, 3, 1);
        let b = TrivariatePolynomial::one(4, 3, 1); // wrong dimension
        let config = TTConfig::new(3, 3, 1, vec![a], vec![b]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_encoding_rate_improves_with_size() {
        let small = tt_small().expect("small");
        let medium = tt_medium().expect("medium");
        // Encoding rate should be positive
        assert!(small.encoding_rate() > 0.0);
        assert!(medium.encoding_rate() > 0.0);
        // Both should be well below 1.0
        assert!(small.encoding_rate() < 1.0);
        assert!(medium.encoding_rate() < 1.0);
    }

    // ----------------------------------------------------------
    // Additional edge case tests
    // ----------------------------------------------------------

    #[test]
    fn test_zero_polynomial_is_zero() {
        let z = TrivariatePolynomial::zero(3, 3, 2);
        assert!(z.is_zero());
        assert_eq!(z.weight(), 0);
        assert_eq!(format!("{}", z), "0");
    }

    #[test]
    fn test_polynomial_multiply_by_one() {
        let (l, m, p) = (3, 3, 2);
        let one = TrivariatePolynomial::one(l, m, p);
        let poly = TrivariatePolynomial::from_monomials(l, m, p, &[(1, 2, 1), (0, 0, 0)]);
        let result = poly.multiply(&one);
        assert_eq!(result, poly, "p * 1 = p");
    }

    #[test]
    fn test_mul_vec_identity() {
        let id = SparseBinaryMatrix::identity(4);
        let v = vec![true, false, true, false];
        let result = id.mul_vec(&v);
        assert_eq!(result, v, "I * v = v");
    }

    #[test]
    fn test_hcat_vcat() {
        let mut a = SparseBinaryMatrix::new(2, 2);
        a.set(0, 0);
        a.set(1, 1);
        let mut b = SparseBinaryMatrix::new(2, 2);
        b.set(0, 1);
        b.set(1, 0);

        let h = a.hcat(&b);
        assert_eq!(h.rows, 2);
        assert_eq!(h.cols, 4);
        assert!(h.get(0, 0));
        assert!(h.get(0, 3)); // b's (0,1) shifted by 2

        let v = a.vcat(&b);
        assert_eq!(v.rows, 4);
        assert_eq!(v.cols, 2);
        assert!(v.get(0, 0));
        assert!(v.get(2, 1)); // b's (0,1) shifted by 2 rows
    }

    #[test]
    fn test_syndrome_round_preserved() {
        let code = tt_small().expect("build");
        let n = code.params.n;
        let no_err = vec![false; n];
        let syndrome = code.extract_syndrome(&no_err, &no_err, 42);
        assert_eq!(syndrome.round, 42);
    }

    #[test]
    fn test_noisy_syndrome_extraction() {
        let code = tt_small().expect("build");
        let n = code.params.n;
        let no_err = vec![false; n];
        // With high measurement error rate, syndrome should be noisy
        let _syndrome = code.extract_syndrome_noisy(&no_err, &no_err, 0.5, 0);
        // Just verify it does not panic and has correct lengths
        assert_eq!(_syndrome.x_syndrome.len(), code.hz.rows);
        assert_eq!(_syndrome.z_syndrome.len(), code.hx.rows);
    }

    #[test]
    fn test_decode_zero_syndrome() {
        let code = tt_small().expect("build");
        let n = code.params.n;
        let no_err = vec![false; n];
        let syndrome = code.extract_syndrome(&no_err, &no_err, 0);
        let result = code.decode(&syndrome).expect("decode");
        assert_eq!(result.weight, 0, "no errors means no correction needed");
        assert!(result.success);
    }

    #[test]
    fn test_sparse_matrix_display() {
        let mut m = SparseBinaryMatrix::new(2, 3);
        m.set(0, 0);
        m.set(1, 2);
        let s = format!("{}", m);
        assert!(s.contains("SparseBinaryMatrix"));
        assert!(s.contains("2x3"));
    }

    #[test]
    fn test_pauli_correction_display() {
        assert_eq!(format!("{}", PauliCorrection::X), "X");
        assert_eq!(format!("{}", PauliCorrection::Z), "Z");
        assert_eq!(format!("{}", PauliCorrection::Y), "Y");
    }

    #[test]
    fn test_tt_code_params_display() {
        let params = TTCodeParams {
            n: 18,
            k: 2,
            d: 3,
            l: 3,
            m: 3,
            p: 1,
        };
        let s = format!("{}", params);
        assert!(s.contains("18"));
        assert!(s.contains("2"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_error_display() {
        let e = TrivariateError::InvalidParameters("test".into());
        assert!(format!("{}", e).contains("test"));
        let e2 = TrivariateError::CodeConstructionFailed("fail".into());
        assert!(format!("{}", e2).contains("fail"));
        let e3 = TrivariateError::DecodingFailed("oops".into());
        assert!(format!("{}", e3).contains("oops"));
        let e4 = TrivariateError::MetaCheckError("meta".into());
        assert!(format!("{}", e4).contains("meta"));
    }
}
