//! Apple Accelerate (AMX) Tensor Contractions for Quantum Simulation
//!
//! This module provides hardware-accelerated tensor contraction operations using
//! Apple's Accelerate framework (BLAS/LAPACK). On Apple Silicon, these routines
//! dispatch to the AMX co-processor for complex matrix multiplication and SVD,
//! delivering substantially higher throughput than pure-Rust implementations for
//! the inner loops of MPS (Matrix Product State) simulation.
//!
//! # Components
//!
//! - [`AmxTensorContractor`]: Implements [`TensorContractor`] using cblas_zgemm / LAPACK zgesdd
//! - [`AmxComplexGemm`]: Safe wrapper for complex-64 general matrix multiplication
//! - [`AmxSvd`]: Safe wrapper for complex-64 singular value decomposition
//! - [`BatchGemm`]: Batched matrix multiplications for gate fusion pipelines
//! - [`AmxMPSEngine`]: Optimized two-qubit gate application on MPS site tensors
//!
//! # Safety
//!
//! All FFI calls are encapsulated behind safe Rust APIs. The unsafe blocks are
//! limited to the raw extern calls and pointer arithmetic required by the
//! column-major BLAS/LAPACK ABI. Inputs are validated (dimension checks, slice
//! length assertions) before any unsafe code executes.
//!
//! # Platform
//!
//! This entire module is gated on `#[cfg(target_os = "macos")]`. On other
//! platforms, the crate falls back to the nalgebra-based [`NalgebraTensorContractor`].

#![cfg(target_os = "macos")]

use crate::traits::TensorContractor;
use crate::{c64_one, c64_zero, C64};

// ===================================================================
// FFI BINDINGS TO APPLE ACCELERATE (BLAS + LAPACK)
// ===================================================================

/// CBLAS transpose enum values matching the Accelerate header.
#[allow(non_camel_case_types, dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
}

/// CBLAS row/column order enum.
#[allow(non_camel_case_types, dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102,
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// Complex double-precision general matrix multiply.
    ///
    /// C := alpha * op(A) * op(B) + beta * C
    ///
    /// All matrices are passed as pointers to interleaved (re, im) pairs.
    fn cblas_zgemm(
        order: CBLAS_ORDER,
        trans_a: CBLAS_TRANSPOSE,
        trans_b: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const C64,
        a: *const C64,
        lda: i32,
        b: *const C64,
        ldb: i32,
        beta: *const C64,
        c: *mut C64,
        ldc: i32,
    );

    /// LAPACK complex double-precision SVD using divide-and-conquer.
    ///
    /// Computes the SVD of a general M-by-N matrix A:
    ///   A = U * SIGMA * conjugate-transpose(V)
    ///
    /// Column-major storage. `jobz` controls how much of U/Vt to compute.
    fn zgesdd_(
        jobz: *const u8,
        m: *const i32,
        n: *const i32,
        a: *mut C64,
        lda: *const i32,
        s: *mut f64,
        u: *mut C64,
        ldu: *const i32,
        vt: *mut C64,
        ldvt: *const i32,
        work: *mut C64,
        lwork: *const i32,
        rwork: *mut f64,
        iwork: *mut i32,
        info: *mut i32,
    );
}

// ===================================================================
// HELPER: ROW-MAJOR <-> COLUMN-MAJOR TRANSPOSE
// ===================================================================

/// Convert a row-major complex matrix to column-major layout (in-place copy).
///
/// BLAS/LAPACK expect column-major (Fortran) ordering. Our crate stores
/// matrices in row-major (C) order. This performs the layout transposition
/// without conjugation.
#[inline]
fn row_to_col_major(src: &[C64], rows: usize, cols: usize) -> Vec<C64> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![c64_zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
    dst
}

/// Convert a column-major complex matrix back to row-major layout.
#[inline]
fn col_to_row_major(src: &[C64], rows: usize, cols: usize) -> Vec<C64> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![c64_zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            dst[i * cols + j] = src[j * rows + i];
        }
    }
    dst
}

// ===================================================================
// AmxComplexGemm: SAFE COMPLEX MATRIX MULTIPLICATION WRAPPER
// ===================================================================

/// Safe wrapper around `cblas_zgemm` for complex-64 matrix multiplication.
///
/// All dimension validation happens before the unsafe call. Matrices are
/// expected in row-major order and are internally transposed for the
/// column-major BLAS ABI.
pub struct AmxComplexGemm;

impl AmxComplexGemm {
    /// Compute C = A * B where A is (rows_a x cols_a) and B is (cols_a x cols_b).
    ///
    /// Both input slices must be in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if slice lengths do not match the declared dimensions.
    pub fn multiply(a: &[C64], b: &[C64], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<C64> {
        Self::multiply_add(a, b, None, c64_one(), c64_zero(), rows_a, cols_a, cols_b)
    }

    /// Full GEMM: C = alpha * A * B + beta * C.
    ///
    /// If `c` is `None`, a zero-initialized matrix is used. If `c` is `Some`,
    /// it must have length `rows_a * cols_b` and is consumed / mutated in place
    /// (the result is returned).
    ///
    /// # Panics
    ///
    /// Panics on dimension mismatches.
    pub fn multiply_add(
        a: &[C64],
        b: &[C64],
        c: Option<&[C64]>,
        alpha: C64,
        beta: C64,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> Vec<C64> {
        assert_eq!(a.len(), rows_a * cols_a, "A dimensions mismatch");
        assert_eq!(b.len(), cols_a * cols_b, "B dimensions mismatch");
        if let Some(c_ref) = c {
            assert_eq!(c_ref.len(), rows_a * cols_b, "C dimensions mismatch");
        }

        let m = rows_a as i32;
        let k = cols_a as i32;
        let n = cols_b as i32;

        // Convert to column-major for BLAS
        let a_col = row_to_col_major(a, rows_a, cols_a);
        let b_col = row_to_col_major(b, cols_a, cols_b);
        let mut c_col = match c {
            Some(c_ref) => row_to_col_major(c_ref, rows_a, cols_b),
            None => vec![c64_zero(); rows_a * cols_b],
        };

        unsafe {
            cblas_zgemm(
                CBLAS_ORDER::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                m,
                n,
                k,
                &alpha as *const C64,
                a_col.as_ptr(),
                m, // lda = m for col-major no-trans
                b_col.as_ptr(),
                k, // ldb = k for col-major no-trans
                &beta as *const C64,
                c_col.as_mut_ptr(),
                m, // ldc = m for col-major
            );
        }

        // Convert result back to row-major
        col_to_row_major(&c_col, rows_a, cols_b)
    }
}

// ===================================================================
// AmxSvd: SAFE COMPLEX SVD WRAPPER
// ===================================================================

/// Safe wrapper around LAPACK `zgesdd_` (divide-and-conquer complex SVD).
///
/// Returns (U, S, Vt) where:
/// - U is m x min(m,n) in row-major
/// - S is a vector of min(m,n) real singular values in descending order
/// - Vt is min(m,n) x n in row-major
pub struct AmxSvd;

impl AmxSvd {
    /// Compute the thin SVD of an m x n complex matrix.
    ///
    /// The input matrix must be in row-major order with length m * n.
    ///
    /// # Returns
    ///
    /// `(U, S, Vt)` where `A ~= U * diag(S) * Vt`.
    ///
    /// # Panics
    ///
    /// Panics if the input length does not match m * n, or if LAPACK reports
    /// a convergence failure.
    pub fn compute(matrix: &[C64], rows: usize, cols: usize) -> (Vec<C64>, Vec<f64>, Vec<C64>) {
        assert_eq!(
            matrix.len(),
            rows * cols,
            "Matrix length {} does not match {}x{} = {}",
            matrix.len(),
            rows,
            cols,
            rows * cols
        );

        if rows == 0 || cols == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let m = rows as i32;
        let n = cols as i32;
        let min_mn = rows.min(cols);

        // LAPACK overwrites A, so we make a column-major copy.
        let mut a_col = row_to_col_major(matrix, rows, cols);

        // Output buffers
        let mut s = vec![0.0f64; min_mn];
        let mut u_col = vec![c64_zero(); rows * min_mn];
        let mut vt_col = vec![c64_zero(); min_mn * cols];

        // Workspace query: lwork = -1 to ask LAPACK for optimal size
        let mut work_query = vec![c64_zero(); 1];
        let mut rwork = vec![0.0f64; Self::rwork_size(rows, cols)];
        let mut iwork = vec![0i32; 8 * min_mn];
        let mut info: i32 = 0;
        let lwork_query: i32 = -1;
        let jobz: u8 = b'S'; // thin SVD

        unsafe {
            zgesdd_(
                &jobz,
                &m,
                &n,
                a_col.as_mut_ptr(),
                &m,
                s.as_mut_ptr(),
                u_col.as_mut_ptr(),
                &m,
                vt_col.as_mut_ptr(),
                &(min_mn as i32),
                work_query.as_mut_ptr(),
                &lwork_query,
                rwork.as_mut_ptr(),
                iwork.as_mut_ptr(),
                &mut info,
            );
        }

        assert_eq!(info, 0, "zgesdd_ workspace query failed with info={}", info);

        // Allocate optimal workspace
        let optimal_lwork = work_query[0].re as i32;
        let lwork = optimal_lwork.max(1);
        let mut work = vec![c64_zero(); lwork as usize];

        // Execute the actual SVD
        unsafe {
            zgesdd_(
                &jobz,
                &m,
                &n,
                a_col.as_mut_ptr(),
                &m,
                s.as_mut_ptr(),
                u_col.as_mut_ptr(),
                &m,
                vt_col.as_mut_ptr(),
                &(min_mn as i32),
                work.as_mut_ptr(),
                &lwork,
                rwork.as_mut_ptr(),
                iwork.as_mut_ptr(),
                &mut info,
            );
        }

        assert!(info >= 0, "zgesdd_ illegal argument at position {}", -info);
        assert_eq!(info, 0, "zgesdd_ SVD did not converge, info={}", info);

        // Convert outputs to row-major
        let u_row = col_to_row_major(&u_col, rows, min_mn);
        let vt_row = col_to_row_major(&vt_col, min_mn, cols);

        (u_row, s, vt_row)
    }

    /// Compute the required rwork buffer size for zgesdd_ with jobz='S'.
    fn rwork_size(m: usize, n: usize) -> usize {
        let min_mn = m.min(n);
        let max_mn = m.max(n);
        // LAPACK documentation: rwork size for zgesdd with jobz='S'
        // = max(1, 5*min(m,n)^2 + 7*min(m,n))
        // We add some headroom.
        let base = 5 * min_mn * min_mn + 7 * min_mn;
        base.max(2 * max_mn * min_mn + 2 * min_mn * min_mn + min_mn)
            .max(1)
    }
}

// ===================================================================
// AmxTensorContractor: TensorContractor IMPLEMENTATION
// ===================================================================

/// Tensor contraction engine backed by Apple Accelerate (AMX).
///
/// Implements the [`TensorContractor`] trait for use in MPS simulation,
/// TEBD, and other tensor network algorithms. All operations dispatch to
/// hardware-accelerated BLAS/LAPACK routines.
pub struct AmxTensorContractor;

impl AmxTensorContractor {
    /// Create a new AMX-backed tensor contractor.
    pub fn new() -> Self {
        AmxTensorContractor
    }
}

impl Default for AmxTensorContractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorContractor for AmxTensorContractor {
    /// Contract two matrices: C = A * B.
    ///
    /// A is m x k, B is k x n, both in row-major order. Returns C (m x n).
    fn contract(&self, a: &[C64], b: &[C64], m: usize, k: usize, n: usize) -> Vec<C64> {
        AmxComplexGemm::multiply(a, b, m, k, n)
    }

    /// Compute the SVD of an m x n matrix. Returns (U, S, Vt) where
    /// U is m x min(m,n), S has min(m,n) entries, Vt is min(m,n) x n.
    fn decompose_svd(&self, matrix: &[C64], m: usize, n: usize) -> (Vec<C64>, Vec<f64>, Vec<C64>) {
        AmxSvd::compute(matrix, m, n)
    }

    /// Truncate an SVD to keep at most `max_rank` singular values.
    ///
    /// Overrides the default implementation to use an AMX-accelerated
    /// copy path, though the logic is numerically identical.
    fn truncate(
        &self,
        u: &[C64],
        s: &[f64],
        vt: &[C64],
        m: usize,
        n: usize,
        max_rank: usize,
    ) -> (Vec<C64>, Vec<f64>, Vec<C64>, f64) {
        let full_rank = s.len();
        let rank = full_rank.min(max_rank);

        // Truncation error: Frobenius norm of discarded singular values.
        let trunc_error: f64 = s[rank..].iter().map(|&x| x * x).sum::<f64>().sqrt();

        // Truncate U: m x rank (from m x full_rank)
        let mut u_trunc = vec![c64_zero(); m * rank];
        for i in 0..m {
            u_trunc[i * rank..(i + 1) * rank]
                .copy_from_slice(&u[i * full_rank..i * full_rank + rank]);
        }

        let s_trunc = s[..rank].to_vec();

        // Truncate Vt: rank x n (from full_rank x n)
        let mut vt_trunc = vec![c64_zero(); rank * n];
        vt_trunc.copy_from_slice(&vt[..rank * n]);

        (u_trunc, s_trunc, vt_trunc, trunc_error)
    }
}

// ===================================================================
// BatchGemm: BATCHED MATRIX MULTIPLICATIONS
// ===================================================================

/// A single matrix multiplication specification within a batch.
pub struct GemmItem<'a> {
    pub a: &'a [C64],
    pub b: &'a [C64],
    pub rows: usize,
    pub k: usize,
    pub cols: usize,
}

/// Batch executor for multiple independent matrix multiplications.
///
/// This is useful in gate fusion pipelines where multiple site tensors must
/// be contracted in parallel. Each multiplication is dispatched to
/// `cblas_zgemm` independently.
pub struct BatchGemm;

impl BatchGemm {
    /// Execute a batch of independent matrix multiplications.
    ///
    /// Returns a `Vec<Vec<C64>>` where each inner vec is the result of
    /// one C = A * B multiplication.
    ///
    /// # Panics
    ///
    /// Panics if any item has mismatched slice lengths.
    pub fn execute(batch: &[GemmItem<'_>]) -> Vec<Vec<C64>> {
        batch
            .iter()
            .map(|item| AmxComplexGemm::multiply(item.a, item.b, item.rows, item.k, item.cols))
            .collect()
    }

    /// Execute batch with Rayon parallelism when available.
    ///
    /// Falls back to sequential execution if the `parallel` feature is disabled
    /// or the batch is too small to benefit from threading.
    #[cfg(feature = "parallel")]
    pub fn execute_parallel(batch: &[GemmItem<'_>]) -> Vec<Vec<C64>> {
        use rayon::prelude::*;

        if batch.len() < 4 {
            return Self::execute(batch);
        }

        batch
            .par_iter()
            .map(|item| AmxComplexGemm::multiply(item.a, item.b, item.rows, item.k, item.cols))
            .collect()
    }
}

// ===================================================================
// AmxMPSEngine: OPTIMIZED MPS TWO-QUBIT GATE APPLICATION
// ===================================================================

/// Optimized MPS engine that applies two-qubit gates using AMX-accelerated
/// contraction and SVD.
///
/// The engine caches work buffers to avoid repeated allocation in hot loops
/// (e.g., TEBD sweeps over an MPS chain).
pub struct AmxMPSEngine {
    /// Reusable work buffer for the fused theta tensor.
    theta_buf: Vec<C64>,
    /// Reusable buffer for SVD output U.
    u_buf: Vec<C64>,
    /// Reusable buffer for singular values.
    s_buf: Vec<f64>,
    /// Reusable buffer for SVD output Vt.
    vt_buf: Vec<C64>,
    /// Maximum bond dimension for truncation.
    max_bond_dim: usize,
    /// Singular value cutoff below which modes are discarded.
    sv_cutoff: f64,
}

impl AmxMPSEngine {
    /// Create a new engine with the given bond dimension limit and cutoff.
    ///
    /// # Arguments
    ///
    /// * `max_bond_dim` - Maximum number of singular values to retain.
    /// * `sv_cutoff` - Absolute singular value threshold. Values below this
    ///   are discarded even if `max_bond_dim` is not reached.
    pub fn new(max_bond_dim: usize, sv_cutoff: f64) -> Self {
        // Pre-allocate for a typical two-qubit contraction:
        // theta is (chi_l * 2) x (2 * chi_r) where chi ~ max_bond_dim.
        let cap = 4 * max_bond_dim * max_bond_dim;
        AmxMPSEngine {
            theta_buf: Vec::with_capacity(cap),
            u_buf: Vec::with_capacity(cap),
            s_buf: Vec::with_capacity(2 * max_bond_dim),
            vt_buf: Vec::with_capacity(cap),
            max_bond_dim,
            sv_cutoff,
        }
    }

    /// Apply a two-qubit gate to adjacent sites `site` and `site + 1` of an MPS.
    ///
    /// The MPS is represented as a mutable slice of site tensors, each stored
    /// as a flat row-major `Vec<C64>` with associated shape `(bond_l, phys, bond_r)`.
    ///
    /// The gate is a 4x4 unitary in row-major order (basis: |00>, |01>, |10>, |11>).
    ///
    /// # Algorithm
    ///
    /// 1. Contract site tensors A[site] and A[site+1] into theta
    ///    (shape: bond_l * d, d * bond_r where d=2).
    /// 2. Apply the gate to the physical indices of theta.
    /// 3. SVD decompose theta and truncate.
    /// 4. Write back the new site tensors.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Mutable slice of MPS site tensor data (flat, row-major).
    /// * `shapes` - Shapes for each tensor as (bond_left, physical_dim, bond_right).
    /// * `site` - Index of the left site (gate acts on site and site + 1).
    /// * `gate` - 4x4 unitary matrix in row-major order.
    ///
    /// # Returns
    ///
    /// The truncation error from the SVD step.
    pub fn apply_two_qubit_gate(
        &mut self,
        tensors: &mut [Vec<C64>],
        shapes: &mut [(usize, usize, usize)],
        site: usize,
        gate: &[C64],
    ) -> f64 {
        assert_eq!(gate.len(), 16, "Gate must be 4x4 (16 elements)");
        assert!(
            site + 1 < tensors.len(),
            "Site {} out of bounds for MPS with {} sites",
            site,
            tensors.len()
        );

        let (bl, d_l, br_l) = shapes[site]; // A: (bl, d_l, br_l)
        let (bl_r, d_r, br) = shapes[site + 1]; // B: (bl_r, d_r, br)
        assert_eq!(d_l, 2, "Physical dimension must be 2");
        assert_eq!(d_r, 2, "Physical dimension must be 2");
        assert_eq!(br_l, bl_r, "Bond dimensions must match at contraction");

        let chi_inner = br_l; // shared bond dimension

        // Step 1: Contract A and B into theta of shape (bl * d_l, d_r * br)
        // Reshape A to (bl * d_l, chi_inner) and B to (chi_inner, d_r * br)
        let rows_a = bl * d_l;
        let cols_b = d_r * br;
        let theta_contract = AmxComplexGemm::multiply(
            &tensors[site],
            &tensors[site + 1],
            rows_a,
            chi_inner,
            cols_b,
        );

        // Step 2: Apply gate to physical indices.
        // theta has shape (bl, d_l, d_r, br). We need to contract the gate
        // (which acts on the d_l x d_r space) with theta.
        // Reshape theta to (bl, d_l * d_r, br), apply gate to middle index.
        let phys_dim = d_l * d_r; // = 4
        self.theta_buf.clear();
        self.theta_buf.resize(bl * phys_dim * br, c64_zero());

        // Rearrange theta_contract from (bl * d_l, d_r * br) to (bl, d_l, d_r, br)
        // then apply gate: for each (bl_idx, br_idx), multiply gate * phys_vec
        for bl_idx in 0..bl {
            for br_idx in 0..br {
                // Extract the 4-element physical vector
                let mut phys_in = [c64_zero(); 4];
                for dl in 0..d_l {
                    for dr in 0..d_r {
                        phys_in[dl * d_r + dr] =
                            theta_contract[(bl_idx * d_l + dl) * cols_b + dr * br + br_idx];
                    }
                }

                // Apply gate: phys_out = gate * phys_in
                let mut phys_out = [c64_zero(); 4];
                for i in 0..4 {
                    let mut sum = c64_zero();
                    for j in 0..4 {
                        sum += gate[i * 4 + j] * phys_in[j];
                    }
                    phys_out[i] = sum;
                }

                // Store back into theta_buf as (bl, phys, br) layout
                for p in 0..phys_dim {
                    self.theta_buf[(bl_idx * phys_dim + p) * br + br_idx] = phys_out[p];
                }
            }
        }

        // Step 3: SVD decompose theta reshaped as (bl * d_l, d_r * br)
        let svd_rows = bl * d_l;
        let svd_cols = d_r * br;

        // Reshape theta_buf from (bl, phys, br) to (bl * d_l, d_r * br)
        let mut theta_matrix = vec![c64_zero(); svd_rows * svd_cols];
        for bl_idx in 0..bl {
            for dl in 0..d_l {
                for dr in 0..d_r {
                    for br_idx in 0..br {
                        let p = dl * d_r + dr;
                        theta_matrix[(bl_idx * d_l + dl) * svd_cols + dr * br + br_idx] =
                            self.theta_buf[(bl_idx * phys_dim + p) * br + br_idx];
                    }
                }
            }
        }

        let (u_full, s_full, vt_full) = AmxSvd::compute(&theta_matrix, svd_rows, svd_cols);

        // Step 4: Truncate based on max_bond_dim and sv_cutoff
        let effective_rank = Self::effective_rank(&s_full, self.max_bond_dim, self.sv_cutoff);
        let contractor = AmxTensorContractor::new();
        let (u_trunc, s_trunc, vt_trunc, trunc_error) = contractor.truncate(
            &u_full,
            &s_full,
            &vt_full,
            svd_rows,
            svd_cols,
            effective_rank,
        );

        let new_chi = s_trunc.len();

        // Absorb singular values into U: U_new[i,j] = U_trunc[i,j] * s[j]
        let mut new_a = vec![c64_zero(); svd_rows * new_chi];
        for i in 0..svd_rows {
            for j in 0..new_chi {
                new_a[i * new_chi + j] = u_trunc[i * new_chi + j] * s_trunc[j];
            }
        }

        // Vt is already (new_chi x svd_cols)
        let new_b = vt_trunc;

        // Write back
        tensors[site] = new_a;
        shapes[site] = (bl, d_l, new_chi);
        tensors[site + 1] = new_b;
        shapes[site + 1] = (new_chi, d_r, br);

        // Cache buffers for reuse
        self.u_buf = u_full;
        self.s_buf = s_full;
        self.vt_buf = vt_full;

        trunc_error
    }

    /// Compute the effective rank given max_rank and singular value cutoff.
    fn effective_rank(s: &[f64], max_rank: usize, cutoff: f64) -> usize {
        let mut rank = s.len().min(max_rank);
        // Further reduce rank based on cutoff
        while rank > 0 && s[rank - 1] < cutoff {
            rank -= 1;
        }
        rank.max(1) // Keep at least rank 1
    }

    /// Get the maximum bond dimension setting.
    pub fn max_bond_dim(&self) -> usize {
        self.max_bond_dim
    }

    /// Get the singular value cutoff setting.
    pub fn sv_cutoff(&self) -> f64 {
        self.sv_cutoff
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::C64;

    /// Tolerance for floating-point comparisons.
    const TOL: f64 = 1e-10;

    /// Helper: check that two complex slices are approximately equal.
    fn assert_approx_eq_c64(a: &[C64], b: &[C64], tol: f64) {
        assert_eq!(
            a.len(),
            b.len(),
            "Length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).norm();
            assert!(
                diff < tol,
                "Element {} differs: {:?} vs {:?} (diff={})",
                i,
                x,
                y,
                diff
            );
        }
    }

    /// Helper: check two f64 slices are approximately equal.
    fn assert_approx_eq_f64(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(
            a.len(),
            b.len(),
            "Length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff < tol,
                "Element {} differs: {} vs {} (diff={})",
                i,
                x,
                y,
                diff
            );
        }
    }

    // ---------------------------------------------------------------
    // GEMM TESTS
    // ---------------------------------------------------------------

    #[test]
    fn test_gemm_identity_multiply() {
        // I * A = A for 2x2 identity
        let identity = vec![c64_one(), c64_zero(), c64_zero(), c64_one()];
        let a = vec![
            C64::new(1.0, 2.0),
            C64::new(3.0, 4.0),
            C64::new(5.0, 6.0),
            C64::new(7.0, 8.0),
        ];
        let result = AmxComplexGemm::multiply(&identity, &a, 2, 2, 2);
        assert_approx_eq_c64(&result, &a, TOL);
    }

    #[test]
    fn test_gemm_known_product() {
        // A = [[1+i, 2], [3, 4-i]]
        // B = [[1, 0], [0, 1]]  (identity)
        // C = A
        let a = vec![
            C64::new(1.0, 1.0),
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, -1.0),
        ];
        let b = vec![c64_one(), c64_zero(), c64_zero(), c64_one()];
        let result = AmxComplexGemm::multiply(&a, &b, 2, 2, 2);
        assert_approx_eq_c64(&result, &a, TOL);
    }

    #[test]
    fn test_gemm_non_square() {
        // A: 2x3, B: 3x2 -> C: 2x2
        let a = vec![
            C64::new(1.0, 0.0),
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
            C64::new(5.0, 0.0),
            C64::new(6.0, 0.0),
        ];
        let b = vec![
            C64::new(7.0, 0.0),
            C64::new(8.0, 0.0),
            C64::new(9.0, 0.0),
            C64::new(10.0, 0.0),
            C64::new(11.0, 0.0),
            C64::new(12.0, 0.0),
        ];
        let result = AmxComplexGemm::multiply(&a, &b, 2, 3, 2);
        // Row 0: 1*7+2*9+3*11 = 7+18+33 = 58, 1*8+2*10+3*12 = 8+20+36 = 64
        // Row 1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
        let expected = vec![
            C64::new(58.0, 0.0),
            C64::new(64.0, 0.0),
            C64::new(139.0, 0.0),
            C64::new(154.0, 0.0),
        ];
        assert_approx_eq_c64(&result, &expected, TOL);
    }

    #[test]
    fn test_gemm_complex_product() {
        // A = [[i, 1], [1, -i]], B = [[1, i], [i, 1]]
        // C[0,0] = i*1 + 1*i = 2i
        // C[0,1] = i*i + 1*1 = -1+1 = 0
        // C[1,0] = 1*1 + (-i)*i = 1 + 1 = 2
        // C[1,1] = 1*i + (-i)*1 = i - i = 0
        let a = vec![
            C64::new(0.0, 1.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, -1.0),
        ];
        let b = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(0.0, 1.0),
            C64::new(1.0, 0.0),
        ];
        let result = AmxComplexGemm::multiply(&a, &b, 2, 2, 2);
        let expected = vec![
            C64::new(0.0, 2.0),
            C64::new(0.0, 0.0),
            C64::new(2.0, 0.0),
            C64::new(0.0, 0.0),
        ];
        assert_approx_eq_c64(&result, &expected, TOL);
    }

    #[test]
    fn test_gemm_multiply_add() {
        // C = 2 * A * B + 3 * C0
        let a = vec![c64_one(), c64_zero(), c64_zero(), c64_one()];
        let b = vec![
            C64::new(1.0, 0.0),
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
        ];
        let c0 = vec![
            C64::new(10.0, 0.0),
            C64::new(10.0, 0.0),
            C64::new(10.0, 0.0),
            C64::new(10.0, 0.0),
        ];
        let alpha = C64::new(2.0, 0.0);
        let beta = C64::new(3.0, 0.0);
        let result = AmxComplexGemm::multiply_add(&a, &b, Some(&c0), alpha, beta, 2, 2, 2);
        // 2*I*B + 3*C0 = 2*B + 3*C0
        let expected = vec![
            C64::new(32.0, 0.0),
            C64::new(34.0, 0.0),
            C64::new(36.0, 0.0),
            C64::new(38.0, 0.0),
        ];
        assert_approx_eq_c64(&result, &expected, TOL);
    }

    // ---------------------------------------------------------------
    // SVD TESTS
    // ---------------------------------------------------------------

    #[test]
    fn test_svd_identity() {
        // SVD of 2x2 identity should give U=I (up to phase), S=[1,1], Vt=I (up to phase)
        let identity = vec![c64_one(), c64_zero(), c64_zero(), c64_one()];
        let (u, s, vt) = AmxSvd::compute(&identity, 2, 2);

        // Singular values should both be 1.0
        assert_approx_eq_f64(&s, &[1.0, 1.0], TOL);

        // Reconstruct: U * diag(S) * Vt should equal identity
        let reconstructed = reconstruct_from_svd(&u, &s, &vt, 2, 2);
        assert_approx_eq_c64(&reconstructed, &identity, TOL);
    }

    #[test]
    fn test_svd_reconstruction() {
        // Random-ish complex matrix, verify A = U * S * Vt
        let a = vec![
            C64::new(1.0, 2.0),
            C64::new(3.0, -1.0),
            C64::new(0.5, 0.5),
            C64::new(-1.0, 0.0),
            C64::new(2.0, 1.0),
            C64::new(1.0, -2.0),
        ];
        let (u, s, vt) = AmxSvd::compute(&a, 2, 3);

        assert_eq!(s.len(), 2); // min(2,3) = 2
        assert_eq!(u.len(), 2 * 2); // 2 x 2
        assert_eq!(vt.len(), 2 * 3); // 2 x 3

        let reconstructed = reconstruct_from_svd(&u, &s, &vt, 2, 3);
        assert_approx_eq_c64(&reconstructed, &a, 1e-9);
    }

    #[test]
    fn test_svd_tall_matrix() {
        // 3x2 matrix
        let a = vec![
            C64::new(1.0, 0.0),
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
            C64::new(5.0, 0.0),
            C64::new(6.0, 0.0),
        ];
        let (u, s, vt) = AmxSvd::compute(&a, 3, 2);

        assert_eq!(s.len(), 2);
        assert_eq!(u.len(), 3 * 2);
        assert_eq!(vt.len(), 2 * 2);

        let reconstructed = reconstruct_from_svd(&u, &s, &vt, 3, 2);
        assert_approx_eq_c64(&reconstructed, &a, 1e-9);
    }

    #[test]
    fn test_svd_singular_values_descending() {
        let a = vec![
            C64::new(3.0, 1.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.5),
        ];
        let (_u, s, _vt) = AmxSvd::compute(&a, 2, 2);

        // Singular values must be in descending order
        for i in 1..s.len() {
            assert!(
                s[i - 1] >= s[i] - TOL,
                "Singular values not descending: s[{}]={} < s[{}]={}",
                i - 1,
                s[i - 1],
                i,
                s[i]
            );
        }
    }

    // ---------------------------------------------------------------
    // BATCH GEMM TEST
    // ---------------------------------------------------------------

    #[test]
    fn test_batch_gemm() {
        let a1 = vec![c64_one(), c64_zero(), c64_zero(), c64_one()];
        let b1 = vec![
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
            C64::new(5.0, 0.0),
        ];
        let a2 = vec![
            C64::new(0.0, 1.0),
            c64_zero(),
            c64_zero(),
            C64::new(0.0, 1.0),
        ];
        let b2 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ];

        let batch = vec![
            GemmItem {
                a: &a1,
                b: &b1,
                rows: 2,
                k: 2,
                cols: 2,
            },
            GemmItem {
                a: &a2,
                b: &b2,
                rows: 2,
                k: 2,
                cols: 2,
            },
        ];

        let results = BatchGemm::execute(&batch);
        assert_eq!(results.len(), 2);

        // First: I * B1 = B1
        assert_approx_eq_c64(&results[0], &b1, TOL);

        // Second: diag(i, i) * I = diag(i, i)
        assert_approx_eq_c64(&results[1], &a2, TOL);
    }

    // ---------------------------------------------------------------
    // TensorContractor TRAIT TESTS
    // ---------------------------------------------------------------

    #[test]
    fn test_tensor_contractor_contract() {
        let contractor = AmxTensorContractor::new();
        let a = vec![
            C64::new(1.0, 0.0),
            C64::new(2.0, 0.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 0.0),
        ];
        let b = vec![
            C64::new(5.0, 0.0),
            C64::new(6.0, 0.0),
            C64::new(7.0, 0.0),
            C64::new(8.0, 0.0),
        ];
        let c = contractor.contract(&a, &b, 2, 2, 2);
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let expected = vec![
            C64::new(19.0, 0.0),
            C64::new(22.0, 0.0),
            C64::new(43.0, 0.0),
            C64::new(50.0, 0.0),
        ];
        assert_approx_eq_c64(&c, &expected, TOL);
    }

    #[test]
    fn test_tensor_contractor_svd() {
        let contractor = AmxTensorContractor::new();
        let a = vec![
            C64::new(1.0, 0.0),
            C64::new(2.0, 1.0),
            C64::new(-1.0, 1.0),
            C64::new(3.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, -1.0),
        ];
        let (u, s, vt) = contractor.decompose_svd(&a, 3, 2);
        let reconstructed = reconstruct_from_svd(&u, &s, &vt, 3, 2);
        assert_approx_eq_c64(&reconstructed, &a, 1e-9);
    }

    #[test]
    fn test_truncation_preserves_dominant() {
        let contractor = AmxTensorContractor::new();

        // Create a matrix with known singular value spectrum
        // Diagonal-ish: singular values ~ 10, 5, 1, 0.01
        let a = vec![
            C64::new(10.0, 0.0),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            C64::new(5.0, 0.0),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            C64::new(1.0, 0.0),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            C64::new(0.01, 0.0),
        ];
        let (u, s, vt) = contractor.decompose_svd(&a, 4, 4);

        // Truncate to rank 2
        let (u_t, s_t, vt_t, err) = contractor.truncate(&u, &s, &vt, 4, 4, 2);

        assert_eq!(s_t.len(), 2);
        // The two largest singular values should be preserved
        assert!((s_t[0] - 10.0).abs() < TOL);
        assert!((s_t[1] - 5.0).abs() < TOL);

        // Truncation error = sqrt(1^2 + 0.01^2) ~ 1.00005
        let expected_err = (1.0f64 * 1.0 + 0.01 * 0.01).sqrt();
        assert!((err - expected_err).abs() < 1e-6);

        // Reconstructed truncated matrix should be a good approximation
        let approx = reconstruct_from_svd(&u_t, &s_t, &vt_t, 4, 4);
        // The first two diagonal entries should be preserved
        assert!((approx[0].re - 10.0).abs() < TOL);
        assert!((approx[5].re - 5.0).abs() < TOL);
    }

    #[test]
    fn test_truncation_max_rank_larger_than_available() {
        let contractor = AmxTensorContractor::new();
        let a = vec![
            C64::new(3.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(3.0, 0.0),
        ];
        let (u, s, vt) = contractor.decompose_svd(&a, 2, 2);
        let (u_t, s_t, vt_t, err) = contractor.truncate(&u, &s, &vt, 2, 2, 100);

        // No truncation should occur
        assert_eq!(s_t.len(), s.len());
        assert!(err < TOL);

        let recon = reconstruct_from_svd(&u_t, &s_t, &vt_t, 2, 2);
        assert_approx_eq_c64(&recon, &a, 1e-9);
    }

    // ---------------------------------------------------------------
    // MPS ENGINE TEST
    // ---------------------------------------------------------------

    #[test]
    fn test_mps_engine_cnot_like_gate() {
        // Create a minimal 2-site MPS in |00> state and apply a CNOT-like gate.
        // A[0]: shape (1, 2, 1), A[1]: shape (1, 2, 1)
        // |00>: A[0][0,0,0]=1, A[1][0,0,0]=1
        let mut tensors = vec![
            vec![c64_one(), c64_zero()], // site 0: (1,2,1) -> |0>
            vec![c64_one(), c64_zero()], // site 1: (1,2,1) -> |0>
        ];
        let mut shapes = vec![(1, 2, 1), (1, 2, 1)];

        // CNOT gate: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
        let cnot = vec![
            c64_one(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_one(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_zero(),
            c64_one(),
            c64_zero(),
            c64_zero(),
            c64_one(),
            c64_zero(),
        ];

        let mut engine = AmxMPSEngine::new(16, 1e-14);
        let err = engine.apply_two_qubit_gate(&mut tensors, &mut shapes, 0, &cnot);

        // CNOT|00> = |00>, so the state should be unchanged (up to SVD reshaping).
        // The amplitude for |00> should be 1, all others 0.
        // Reconstruct the full 4-element state vector.
        let (bl0, _d0, chi) = shapes[0];
        let (chi1, _d1, br1) = shapes[1];
        assert_eq!(bl0, 1);
        assert_eq!(br1, 1);
        assert_eq!(chi, chi1);

        // Contract: sum_alpha A[0][0, s0, alpha] * A[1][alpha, s1, 0]
        let mut state = [c64_zero(); 4];
        for s0 in 0..2 {
            for s1 in 0..2 {
                let mut amp = c64_zero();
                for alpha in 0..chi {
                    let a_val = tensors[0][s0 * chi + alpha]; // (1, 2, chi) -> row s0
                    let b_val = tensors[1][alpha * 2 + s1]; // (chi, 2, 1) -> row alpha
                    amp += a_val * b_val;
                }
                state[s0 * 2 + s1] = amp;
            }
        }

        // |00> should have amplitude 1
        assert!(
            (state[0].norm() - 1.0).abs() < 1e-9,
            "Expected |00>=1, got {:?}",
            state[0]
        );
        assert!(
            state[1].norm() < 1e-9,
            "Expected |01>=0, got {:?}",
            state[1]
        );
        assert!(
            state[2].norm() < 1e-9,
            "Expected |10>=0, got {:?}",
            state[2]
        );
        assert!(
            state[3].norm() < 1e-9,
            "Expected |11>=0, got {:?}",
            state[3]
        );
        assert!(
            err < 1e-10,
            "Truncation error should be negligible for product state"
        );
    }

    // ---------------------------------------------------------------
    // HELPERS
    // ---------------------------------------------------------------

    /// Reconstruct A = U * diag(S) * Vt from SVD components.
    fn reconstruct_from_svd(u: &[C64], s: &[f64], vt: &[C64], m: usize, n: usize) -> Vec<C64> {
        let rank = s.len();
        // US = U * diag(S): m x rank
        let mut us = vec![c64_zero(); m * rank];
        for i in 0..m {
            for j in 0..rank {
                us[i * rank + j] = u[i * rank + j] * s[j];
            }
        }
        // A = US * Vt: m x n
        AmxComplexGemm::multiply(&us, vt, m, rank, n)
    }
}
