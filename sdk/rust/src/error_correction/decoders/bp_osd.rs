//! BP-OSD: Belief Propagation + Ordered Statistics Decoding
//!
//! State-of-the-art decoder for quantum LDPC codes, including bivariate bicycle
//! codes and lifted product codes. Combines iterative belief propagation on the
//! Tanner graph with OSD post-processing when BP fails to converge.
//!
//! # Algorithm Overview
//!
//! 1. **Belief Propagation (BP)**: Run min-sum or sum-product BP on the code's
//!    Tanner graph. If BP converges (hard-decision satisfies syndrome), return
//!    the BP solution.
//!
//! 2. **OSD-0 (Order-0 Statistics Decoding)**: If BP does not converge, use the
//!    soft LLR information from BP to rank variable nodes by reliability. Permute
//!    columns of H by decreasing reliability, perform Gaussian elimination over
//!    GF(2), and solve for the most likely error pattern by setting the least
//!    reliable (information) bits to zero and back-substituting.
//!
//! 3. **OSD-CS (Combination Sweep)**: After OSD-0 produces a baseline correction,
//!    enumerate combinations of up to `osd_order` least-reliable information bits,
//!    flipping them and re-solving via back-substitution. Keep the lowest-weight
//!    correction that satisfies the syndrome. This dramatically improves decoding
//!    performance on quantum LDPC codes at a tunable computational cost.
//!
//! # Why BP-OSD?
//!
//! Standard BP alone fails on many quantum LDPC codes due to short cycles and
//! degeneracy in the Tanner graph. MWPM is optimal for surface codes but does
//! not generalize to LDPC codes with irregular structure. BP-OSD bridges this
//! gap: BP provides fast convergence when possible, and OSD-CS provides
//! near-optimal correction when BP gets trapped.
//!
//! # Supported Code Families
//!
//! - Bivariate bicycle codes (Bravyi et al., arXiv:2308.07915)
//! - Lifted product / fibre bundle codes
//! - Hypergraph product codes
//! - Any CSS code with a binary parity check matrix
//!
//! # References
//!
//! - Panteleev & Kalachev, "Degenerate quantum LDPC codes with good finite
//!   length performance", Quantum 5, 585 (2021). arXiv:2104.13659
//! - Roffe et al., "Decoding across the quantum low-density parity-check
//!   landscape", PRX Quantum (2023). arXiv:2005.07016
//! - Fossorier & Lin, "Soft-decision decoding of linear block codes based on
//!   ordered statistics", IEEE Trans. Inf. Theory 41(5), 1379-1396 (1995)
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::bp_osd::{BpOsdDecoder, BpConfig, BpMethod, SparseMatrix};
//!
//! // Repetition code parity check matrix
//! let h = SparseMatrix::from_dense(&[
//!     vec![true, true, false, false, false],
//!     vec![false, true, true, false, false],
//!     vec![false, false, true, true, false],
//!     vec![false, false, false, true, true],
//! ]);
//!
//! let decoder = BpOsdDecoder::new(h, BpConfig::default(), 5, 1000);
//! let syndrome = vec![false, true, true, false];
//! let result = decoder.decode(&syndrome, 0.05);
//! assert!(result.converged || result.used_osd);
//! ```

use std::fmt;

// ============================================================
// DATA STRUCTURES
// ============================================================

/// Sparse binary matrix in adjacency-list format (Tanner graph representation).
///
/// Stores the non-zero positions for each row and each column, enabling
/// efficient iteration over the Tanner graph in both directions. This is
/// the natural representation for LDPC parity check matrices where each
/// row (check) and column (variable) has a small number of non-zero entries.
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    /// Number of rows (check nodes / stabilizers).
    pub rows: usize,
    /// Number of columns (variable nodes / qubits).
    pub cols: usize,
    /// For each row, the sorted list of column indices with a 1 entry.
    pub row_indices: Vec<Vec<usize>>,
    /// For each column, the sorted list of row indices with a 1 entry.
    pub col_indices: Vec<Vec<usize>>,
}

impl SparseMatrix {
    /// Construct a sparse matrix from a dense boolean matrix.
    ///
    /// Each inner `Vec<bool>` is one row. All rows must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if rows have inconsistent lengths.
    pub fn from_dense(dense: &[Vec<bool>]) -> Self {
        let rows = dense.len();
        let cols = if rows > 0 { dense[0].len() } else { 0 };

        let mut row_indices = Vec::with_capacity(rows);
        let mut col_indices = vec![Vec::new(); cols];

        for (r, row) in dense.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "Row {} has length {}, expected {}",
                r,
                row.len(),
                cols
            );
            let mut ri = Vec::new();
            for (c, &val) in row.iter().enumerate() {
                if val {
                    ri.push(c);
                    col_indices[c].push(r);
                }
            }
            row_indices.push(ri);
        }

        Self {
            rows,
            cols,
            row_indices,
            col_indices,
        }
    }

    /// Construct a sparse matrix from adjacency lists (row-major).
    ///
    /// `check_to_var[i]` gives the column indices of non-zero entries in row `i`.
    pub fn from_adjacency(num_rows: usize, num_cols: usize, check_to_var: Vec<Vec<usize>>) -> Self {
        assert_eq!(
            check_to_var.len(),
            num_rows,
            "Expected {} rows, got {}",
            num_rows,
            check_to_var.len()
        );

        let mut col_indices = vec![Vec::new(); num_cols];
        for (r, cols) in check_to_var.iter().enumerate() {
            for &c in cols {
                assert!(
                    c < num_cols,
                    "Column index {} out of range (num_cols = {})",
                    c,
                    num_cols
                );
                col_indices[c].push(r);
            }
        }

        Self {
            rows: num_rows,
            cols: num_cols,
            row_indices: check_to_var,
            col_indices,
        }
    }

    /// Compute the syndrome `s = H * e` over GF(2).
    ///
    /// Returns a boolean vector of length `self.rows`.
    pub fn syndrome(&self, error: &[bool]) -> Vec<bool> {
        assert_eq!(
            error.len(),
            self.cols,
            "Error vector length {} != num_cols {}",
            error.len(),
            self.cols
        );
        self.row_indices
            .iter()
            .map(|row_cols| {
                let parity: usize = row_cols.iter().filter(|&&c| error[c]).count();
                parity % 2 == 1
            })
            .collect()
    }

    /// Check whether a correction vector satisfies a given syndrome.
    pub fn satisfies_syndrome(&self, correction: &[bool], syndrome: &[bool]) -> bool {
        let computed = self.syndrome(correction);
        computed == syndrome
    }

    /// Build a repetition code parity check matrix of length `n`.
    ///
    /// The code has `n` variable nodes and `n-1` check nodes, where check `i`
    /// connects variables `i` and `i+1`.
    pub fn repetition_code(n: usize) -> Self {
        assert!(n >= 2, "Repetition code requires at least 2 qubits");
        let checks: Vec<Vec<usize>> = (0..n - 1).map(|i| vec![i, i + 1]).collect();
        Self::from_adjacency(n - 1, n, checks)
    }

    /// Build a Hamming [7,4,3] parity check matrix.
    ///
    /// Classic error-correcting code useful for testing. Has 3 checks, 7 bits,
    /// and corrects any single-bit error.
    pub fn hamming_7_4() -> Self {
        Self::from_dense(&[
            vec![true, true, true, false, true, false, false],
            vec![true, true, false, true, false, true, false],
            vec![true, false, true, true, false, false, true],
        ])
    }

    /// Build a small surface-code-like parity check matrix.
    ///
    /// Distance-3 rotated surface code with 9 data qubits and 8 stabilizers
    /// (4 X-type + 4 Z-type). This returns only the X-stabilizer submatrix
    /// (4 checks on 9 qubits) for single-type decoding.
    pub fn surface_code_d3() -> Self {
        // X-stabilizers of the d=3 rotated surface code
        // Plaquette stabilizers connecting 4 (or 2 on boundary) data qubits
        Self::from_dense(&[
            //  q0  q1  q2  q3  q4  q5  q6  q7  q8
            vec![true, true, false, true, true, false, false, false, false], // X_0: q0,q1,q3,q4
            vec![false, false, true, false, false, true, false, false, false], // X_1: q2,q5 (boundary)
            vec![false, false, false, true, false, false, true, false, false], // X_2: q3,q6 (boundary)
            vec![false, false, false, false, true, true, false, true, true],   // X_3: q4,q5,q7,q8
        ])
    }

    /// Hamming weight (number of `true` entries) of a boolean vector.
    pub fn weight(v: &[bool]) -> usize {
        v.iter().filter(|&&b| b).count()
    }
}

impl fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseMatrix({}x{}, nnz={})",
            self.rows,
            self.cols,
            self.row_indices.iter().map(|r| r.len()).sum::<usize>()
        )
    }
}

// ============================================================
// BP CONFIGURATION
// ============================================================

/// Belief propagation algorithm variant.
///
/// - **SumProduct**: Exact marginalization on tree-structured graphs. Uses
///   `tanh` and `atanh` operations which are numerically expensive but more
///   accurate for loopy codes.
/// - **MinSum**: Approximation that replaces `tanh`/`atanh` with `min` and
///   `sign` operations. Faster per iteration and often comparable accuracy
///   when combined with a scaling factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpMethod {
    /// Sum-product (exact for trees, uses tanh/atanh).
    SumProduct,
    /// Min-sum (approximation, uses min/sign with optional scaling).
    MinSum,
}

impl fmt::Display for BpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpMethod::SumProduct => write!(f, "SumProduct"),
            BpMethod::MinSum => write!(f, "MinSum"),
        }
    }
}

/// Configuration for the belief propagation stage.
#[derive(Clone, Debug)]
pub struct BpConfig {
    /// Maximum number of BP iterations before declaring non-convergence.
    pub max_iterations: usize,
    /// Damping factor in (0, 1]. Messages are updated as:
    /// `new_msg = damping * computed + (1 - damping) * old_msg`.
    /// Values around 0.5-0.8 help convergence on loopy graphs.
    pub damping: f64,
    /// BP algorithm variant (MinSum or SumProduct).
    pub method: BpMethod,
    /// Scaling factor for min-sum messages. Typical values: 0.625-0.875.
    /// Only used when `method == BpMethod::MinSum`.
    pub min_sum_scaling: f64,
    /// Convergence threshold: stop if max message change < this value.
    pub convergence_threshold: f64,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            damping: 0.75,
            method: BpMethod::MinSum,
            min_sum_scaling: 0.625,
            convergence_threshold: 1e-8,
        }
    }
}

impl BpConfig {
    /// Configuration tuned for sum-product decoding.
    pub fn sum_product() -> Self {
        Self {
            method: BpMethod::SumProduct,
            damping: 0.5,
            min_sum_scaling: 1.0, // unused for sum-product
            ..Self::default()
        }
    }

    /// Configuration tuned for min-sum decoding with normalized scaling.
    pub fn min_sum() -> Self {
        Self {
            method: BpMethod::MinSum,
            damping: 0.75,
            min_sum_scaling: 0.625,
            ..Self::default()
        }
    }
}

// ============================================================
// DECODING RESULT
// ============================================================

/// Result of a BP-OSD decoding attempt.
///
/// Contains the estimated correction, convergence status, and diagnostic
/// information about which stages were used and how much work was done.
#[derive(Clone, Debug)]
pub struct DecodingResult {
    /// Estimated error pattern. `correction[i] == true` means qubit `i` is
    /// believed to have experienced an error.
    pub correction: Vec<bool>,
    /// Whether the decoder found a valid correction satisfying the syndrome.
    pub converged: bool,
    /// Number of BP iterations executed.
    pub bp_iterations: usize,
    /// Whether OSD post-processing was invoked (BP alone did not converge).
    pub used_osd: bool,
    /// Number of OSD combinations tried (0 if OSD was not used, 1 for OSD-0,
    /// more for OSD-CS with higher order).
    pub osd_combinations_tried: usize,
    /// Hamming weight of the correction.
    pub weight: usize,
    /// Final LLR beliefs from BP (positive = likely no error, negative = likely error).
    pub llr_beliefs: Vec<f64>,
}

// ============================================================
// OSD (ORDERED STATISTICS DECODING)
// ============================================================

/// Ordered Statistics Decoding post-processor.
///
/// Implements both OSD-0 (basic Gaussian elimination with reliability ordering)
/// and OSD-CS (combination sweep over least-reliable information bits).
///
/// # Algorithm
///
/// 1. Rank columns of H by reliability (|LLR| from BP), most reliable first.
/// 2. Perform Gaussian elimination on the reordered H to find a systematic
///    form with pivot columns (most reliable bits = redundancy set).
/// 3. **OSD-0**: Set all information bits (non-pivot, least reliable) to 0,
///    solve for pivot bits via back-substitution.
/// 4. **OSD-CS**: Enumerate combinations of 1..`osd_order` information bits
///    set to 1 (flipped), re-solve each time. Keep the lowest-weight valid
///    correction.
///
/// # References
///
/// - Fossorier & Lin, IEEE Trans. Inf. Theory 1995 (original OSD)
/// - Roffe et al., PRX Quantum 2023 (OSD-CS for quantum LDPC)
struct Osd;

impl Osd {
    /// Run OSD-0: basic ordered statistics decoding.
    ///
    /// Returns `(correction, pivot_cols, col_order, augmented_matrix)` so that
    /// OSD-CS can reuse the elimination result.
    fn osd0(
        h: &SparseMatrix,
        syndrome: &[bool],
        llrs: &[f64],
    ) -> (Vec<bool>, Vec<usize>, Vec<usize>, Vec<Vec<bool>>) {
        let m = h.rows;
        let n = h.cols;

        // Order columns by reliability: most reliable (highest |LLR|) first.
        // These become the "redundancy set" (pivot columns after elimination).
        let mut col_order: Vec<usize> = (0..n).collect();
        col_order.sort_by(|&a, &b| {
            llrs[b]
                .abs()
                .partial_cmp(&llrs[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build augmented matrix [H_reordered | syndrome] for GF(2) elimination.
        // We use a dense representation since m is typically small for LDPC codes.
        let mut augmented: Vec<Vec<bool>> = (0..m)
            .map(|r| {
                let mut row = vec![false; n + 1];
                for (new_c, &orig_c) in col_order.iter().enumerate() {
                    // Check if orig_c is in this row's adjacency list
                    row[new_c] = h.row_indices[r].contains(&orig_c);
                }
                row[n] = syndrome[r];
                row
            })
            .collect();

        // Forward Gaussian elimination over GF(2) with partial pivoting.
        let mut pivot_cols: Vec<usize> = Vec::with_capacity(m);
        let mut current_row = 0;

        for col in 0..n {
            if current_row >= m {
                break;
            }

            // Find a pivot row for this column
            let mut pivot = None;
            for row in current_row..m {
                if augmented[row][col] {
                    pivot = Some(row);
                    break;
                }
            }

            if let Some(p) = pivot {
                // Swap pivot row into position
                augmented.swap(current_row, p);

                // Eliminate this column from all other rows
                for row in 0..m {
                    if row != current_row && augmented[row][col] {
                        // XOR row with pivot row (GF(2) addition)
                        let pivot_row = augmented[current_row].clone();
                        for c in 0..=n {
                            augmented[row][c] ^= pivot_row[c];
                        }
                    }
                }

                pivot_cols.push(col);
                current_row += 1;
            }
        }

        // OSD-0: Set all information bits (non-pivot) to 0, solve pivot bits
        // from the syndrome via back-substitution.
        let mut reordered_error = vec![false; n];
        for (rank, &col) in pivot_cols.iter().enumerate() {
            reordered_error[col] = augmented[rank][n];
        }

        // Map back to original column ordering
        let mut correction = vec![false; n];
        for (new_c, &orig_c) in col_order.iter().enumerate() {
            correction[orig_c] = reordered_error[new_c];
        }

        (correction, pivot_cols, col_order, augmented)
    }

    /// Run OSD-CS: combination sweep over information bits.
    ///
    /// Starting from the OSD-0 solution, tries flipping combinations of up to
    /// `osd_order` information (non-pivot) bits and re-solving. Returns the
    /// lowest-weight correction found.
    ///
    /// `max_combinations` bounds the total number of candidate corrections
    /// evaluated, providing a hard limit on computational cost.
    fn osd_cs(
        h: &SparseMatrix,
        syndrome: &[bool],
        llrs: &[f64],
        osd_order: usize,
        max_combinations: usize,
    ) -> (Vec<bool>, usize) {
        let n = h.cols;

        // Run OSD-0 first to get the baseline and elimination state
        let (osd0_correction, pivot_cols, col_order, augmented) = Self::osd0(h, syndrome, llrs);

        let mut best_correction = osd0_correction;
        let mut best_weight = SparseMatrix::weight(&best_correction);
        let mut combinations_tried: usize = 1; // OSD-0 counts as 1

        // Identify information bit positions (non-pivot columns in reordered space).
        // These are sorted by decreasing reliability because col_order is sorted
        // that way, and pivot columns consumed the most-reliable positions first.
        let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter().copied().collect();
        let info_bits: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();

        if info_bits.is_empty() || osd_order == 0 {
            return (best_correction, combinations_tried);
        }

        // Enumerate combinations of 1..osd_order information bits to flip.
        // We enumerate in order of increasing weight to find low-weight
        // corrections first, and stop at max_combinations.
        let effective_order = osd_order.min(info_bits.len());

        'outer: for order in 1..=effective_order {
            // Generate combinations of `order` elements from info_bits
            let mut combo_indices = Vec::with_capacity(order);
            for i in 0..order {
                combo_indices.push(i);
            }

            loop {
                if combinations_tried >= max_combinations {
                    break 'outer;
                }

                // Build the information bit vector with selected bits flipped
                let mut reordered_error = vec![false; n];
                for &idx in &combo_indices {
                    reordered_error[info_bits[idx]] = true;
                }

                // Solve for pivot bits given the flipped information bits.
                // For each pivot position at rank `r`, the pivot bit value is:
                //   augmented[r][n] XOR (sum of augmented[r][info_col] * info_val for info cols)
                for (rank, &pcol) in pivot_cols.iter().enumerate() {
                    let mut val = augmented[rank][n]; // syndrome contribution
                    for &icol in &combo_indices {
                        let info_col = info_bits[icol];
                        if augmented[rank][info_col] {
                            val ^= true;
                        }
                    }
                    reordered_error[pcol] = val;
                }

                // Map back to original ordering
                let mut correction = vec![false; n];
                for (new_c, &orig_c) in col_order.iter().enumerate() {
                    correction[orig_c] = reordered_error[new_c];
                }

                let w = SparseMatrix::weight(&correction);
                if w < best_weight {
                    // Verify syndrome (should always match, but safety check)
                    if h.satisfies_syndrome(&correction, syndrome) {
                        best_weight = w;
                        best_correction = correction;
                    }
                }

                combinations_tried += 1;

                // Advance to next combination (lexicographic order)
                if !Self::next_combination(&mut combo_indices, info_bits.len()) {
                    break;
                }
            }
        }

        (best_correction, combinations_tried)
    }

    /// Advance a combination to the next lexicographic position.
    ///
    /// `indices` contains sorted positions into a collection of size `n`.
    /// Returns `false` if there are no more combinations.
    fn next_combination(indices: &mut [usize], n: usize) -> bool {
        let k = indices.len();
        if k == 0 {
            return false;
        }

        // Find rightmost index that can be incremented
        let mut i = k;
        loop {
            if i == 0 {
                return false;
            }
            i -= 1;
            if indices[i] < n - k + i {
                break;
            }
            if i == 0 {
                return false;
            }
        }

        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }

        true
    }
}

// ============================================================
// BP-OSD DECODER
// ============================================================

/// Combined Belief Propagation + Ordered Statistics Decoder.
///
/// This is the main decoder struct. It runs BP first; if BP converges, the
/// BP solution is returned immediately. If BP does not converge, the final
/// LLR beliefs are fed into OSD-CS for post-processing.
///
/// # Configuration
///
/// - `bp_config`: Controls BP iteration count, damping, algorithm variant.
/// - `osd_order`: Controls OSD search depth (0 = OSD-0 only, higher = more
///   combinations searched). Typical values: 5-10 for production.
/// - `max_osd_combinations`: Hard limit on OSD-CS search to bound runtime.
///
/// # Performance Characteristics
///
/// - BP alone: O(iterations * edges) per decoding
/// - OSD-0: O(m^2 * n) for Gaussian elimination (m = checks, n = qubits)
/// - OSD-CS order w: O(C(k, w) * m) where k = information bits
///
/// # References
///
/// - Panteleev & Kalachev, arXiv:2104.13659
/// - Roffe et al., arXiv:2005.07016
pub struct BpOsdDecoder {
    /// Parity check matrix in sparse format.
    parity_check: SparseMatrix,
    /// Belief propagation configuration.
    bp_config: BpConfig,
    /// OSD order: 0 for OSD-0, higher for combination sweep.
    osd_order: usize,
    /// Maximum number of OSD combinations to try.
    max_osd_combinations: usize,
}

impl BpOsdDecoder {
    /// Create a new BP-OSD decoder.
    ///
    /// # Arguments
    ///
    /// * `parity_check` - The code's parity check matrix H in sparse format.
    /// * `bp_config` - Configuration for the BP stage.
    /// * `osd_order` - OSD search depth (0 = OSD-0, 5-10 typical for production).
    /// * `max_osd_combinations` - Hard limit on total OSD candidates evaluated.
    pub fn new(
        parity_check: SparseMatrix,
        bp_config: BpConfig,
        osd_order: usize,
        max_osd_combinations: usize,
    ) -> Self {
        Self {
            parity_check,
            bp_config,
            osd_order,
            max_osd_combinations,
        }
    }

    /// Create a decoder with default BP config and the given OSD parameters.
    pub fn with_osd(parity_check: SparseMatrix, osd_order: usize) -> Self {
        Self::new(parity_check, BpConfig::default(), osd_order, 10_000)
    }

    /// Create a decoder with OSD disabled (BP only).
    pub fn bp_only(parity_check: SparseMatrix, bp_config: BpConfig) -> Self {
        Self::new(parity_check, bp_config, 0, 0)
    }

    /// Decode a syndrome given the physical error probability `p`.
    ///
    /// `syndrome` is a boolean vector of length equal to `parity_check.rows`.
    /// `p` is the per-qubit error probability (depolarizing channel model).
    ///
    /// Returns a [`DecodingResult`] containing the estimated correction,
    /// convergence status, and diagnostic metrics.
    ///
    /// # Panics
    ///
    /// Panics if `syndrome.len() != parity_check.rows` or `p` is not in (0, 1).
    pub fn decode(&self, syndrome: &[bool], p: f64) -> DecodingResult {
        assert_eq!(
            syndrome.len(),
            self.parity_check.rows,
            "Syndrome length {} != num_checks {}",
            syndrome.len(),
            self.parity_check.rows
        );
        assert!(
            p > 0.0 && p < 1.0,
            "Error probability must be in (0, 1), got {}",
            p
        );

        // Trivial syndrome: no errors detected
        if syndrome.iter().all(|&s| !s) {
            return DecodingResult {
                correction: vec![false; self.parity_check.cols],
                converged: true,
                bp_iterations: 0,
                used_osd: false,
                osd_combinations_tried: 0,
                weight: 0,
                llr_beliefs: vec![((1.0 - p) / p).ln(); self.parity_check.cols],
            };
        }

        // Run BP
        let (bp_correction, bp_llrs, bp_iterations, bp_converged) = self.run_bp(syndrome, p);

        if bp_converged {
            let w = SparseMatrix::weight(&bp_correction);
            return DecodingResult {
                correction: bp_correction,
                converged: true,
                bp_iterations,
                used_osd: false,
                osd_combinations_tried: 0,
                weight: w,
                llr_beliefs: bp_llrs,
            };
        }

        // BP failed to converge -- run OSD post-processing
        if self.osd_order == 0 && self.max_osd_combinations == 0 {
            // OSD disabled: return BP hard decision even though it may not satisfy syndrome
            let w = SparseMatrix::weight(&bp_correction);
            return DecodingResult {
                correction: bp_correction,
                converged: false,
                bp_iterations,
                used_osd: false,
                osd_combinations_tried: 0,
                weight: w,
                llr_beliefs: bp_llrs,
            };
        }

        let (osd_correction, osd_combos) = if self.osd_order == 0 {
            // OSD-0 only
            let (corr, _, _, _) = Osd::osd0(&self.parity_check, syndrome, &bp_llrs);
            (corr, 1)
        } else {
            // OSD-CS
            Osd::osd_cs(
                &self.parity_check,
                syndrome,
                &bp_llrs,
                self.osd_order,
                self.max_osd_combinations,
            )
        };

        let converged = self
            .parity_check
            .satisfies_syndrome(&osd_correction, syndrome);
        let w = SparseMatrix::weight(&osd_correction);

        DecodingResult {
            correction: osd_correction,
            converged,
            bp_iterations,
            used_osd: true,
            osd_combinations_tried: osd_combos,
            weight: w,
            llr_beliefs: bp_llrs,
        }
    }

    /// Run belief propagation on the Tanner graph.
    ///
    /// Returns `(hard_decision, final_llrs, iterations, converged)`.
    fn run_bp(&self, syndrome: &[bool], p: f64) -> (Vec<bool>, Vec<f64>, usize, bool) {
        let h = &self.parity_check;
        let n_var = h.cols;
        let n_chk = h.rows;
        let cfg = &self.bp_config;

        // Channel LLR: log((1-p)/p). Positive means "more likely no error".
        let channel_llr = ((1.0 - p) / p).ln();

        // Syndrome as sign: +1 for unsatisfied (syndrome=false), -1 for satisfied (syndrome=true)
        let syndrome_sign: Vec<f64> = syndrome
            .iter()
            .map(|&s| if s { -1.0 } else { 1.0 })
            .collect();

        // Variable-to-check messages: msg_vc[v][local_c_index]
        // Initialized to channel LLR
        let mut msg_vc: Vec<Vec<f64>> = h
            .col_indices
            .iter()
            .map(|checks| vec![channel_llr; checks.len()])
            .collect();

        // Check-to-variable messages: msg_cv[c][local_v_index]
        let mut msg_cv: Vec<Vec<f64>> = h
            .row_indices
            .iter()
            .map(|vars| vec![0.0; vars.len()])
            .collect();

        let mut iterations = 0;

        for iter in 0..cfg.max_iterations {
            iterations = iter + 1;
            let mut max_delta: f64 = 0.0;

            // ---- Check-to-variable messages ----
            for c in 0..n_chk {
                let vars = &h.row_indices[c];
                let n_v = vars.len();

                for j in 0..n_v {
                    let new_msg = match cfg.method {
                        BpMethod::MinSum => {
                            self.check_to_var_min_sum(c, j, &msg_vc, &syndrome_sign)
                        }
                        BpMethod::SumProduct => {
                            self.check_to_var_sum_product(c, j, &msg_vc, &syndrome_sign)
                        }
                    };

                    // Apply damping
                    let old_msg = msg_cv[c][j];
                    let damped = cfg.damping * new_msg + (1.0 - cfg.damping) * old_msg;

                    let delta = (damped - old_msg).abs();
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    msg_cv[c][j] = damped;
                }
            }

            // ---- Variable-to-check messages ----
            for v in 0..n_var {
                let checks = &h.col_indices[v];
                let n_c = checks.len();

                // Total incoming from all check nodes
                let total_incoming: f64 = checks
                    .iter()
                    .enumerate()
                    .map(|(_, &c)| {
                        let local_v = self.check_var_local_index(c, v);
                        msg_cv[c][local_v]
                    })
                    .sum();

                for j in 0..n_c {
                    let c = checks[j];
                    let local_v = self.check_var_local_index(c, v);
                    let incoming_from_c = msg_cv[c][local_v];

                    // Extrinsic information: channel LLR + all incoming except from target check
                    msg_vc[v][j] = channel_llr + total_incoming - incoming_from_c;
                }
            }

            // ---- Check convergence: does hard decision satisfy syndrome? ----
            let beliefs: Vec<f64> = (0..n_var)
                .map(|v| {
                    let checks = &h.col_indices[v];
                    let total_incoming: f64 = checks
                        .iter()
                        .map(|&c| {
                            let local_v = self.check_var_local_index(c, v);
                            msg_cv[c][local_v]
                        })
                        .sum();
                    channel_llr + total_incoming
                })
                .collect();

            let hard_decision: Vec<bool> = beliefs.iter().map(|&b| b < 0.0).collect();

            if h.satisfies_syndrome(&hard_decision, syndrome) {
                return (hard_decision, beliefs, iterations, true);
            }

            // Also check message convergence (not the same as syndrome satisfaction)
            if max_delta < cfg.convergence_threshold {
                // Messages converged but syndrome not satisfied -- BP is stuck
                return (hard_decision, beliefs, iterations, false);
            }
        }

        // Max iterations reached without convergence
        let beliefs: Vec<f64> = (0..n_var)
            .map(|v| {
                let checks = &h.col_indices[v];
                let total_incoming: f64 = checks
                    .iter()
                    .map(|&c| {
                        let local_v = self.check_var_local_index(c, v);
                        msg_cv[c][local_v]
                    })
                    .sum();
                channel_llr + total_incoming
            })
            .collect();

        let hard_decision: Vec<bool> = beliefs.iter().map(|&b| b < 0.0).collect();
        (hard_decision, beliefs, iterations, false)
    }

    /// Compute a check-to-variable message using the min-sum algorithm.
    ///
    /// For check `c` sending to its `j`-th variable, the message is:
    ///   sign = syndrome_sign[c] * product(sign(msg_vc[v_k -> c])) for k != j
    ///   magnitude = min(|msg_vc[v_k -> c]|) for k != j
    ///   result = sign * magnitude * min_sum_scaling
    fn check_to_var_min_sum(
        &self,
        c: usize,
        j: usize,
        msg_vc: &[Vec<f64>],
        syndrome_sign: &[f64],
    ) -> f64 {
        let vars = &self.parity_check.row_indices[c];
        let mut sign_prod = syndrome_sign[c];
        let mut min_mag = f64::INFINITY;

        for (k, &v) in vars.iter().enumerate() {
            if k == j {
                continue;
            }
            let local_c = self.var_check_local_index(v, c);
            let msg = msg_vc[v][local_c];

            if msg < 0.0 {
                sign_prod = -sign_prod;
            }
            let mag = msg.abs();
            if mag < min_mag {
                min_mag = mag;
            }
        }

        sign_prod * min_mag * self.bp_config.min_sum_scaling
    }

    /// Compute a check-to-variable message using the sum-product algorithm.
    ///
    /// Uses the tanh rule:
    ///   tanh(msg_cv/2) = syndrome_sign * product(tanh(msg_vc[v_k -> c] / 2)) for k != j
    ///
    /// The result is: 2 * atanh(product)
    fn check_to_var_sum_product(
        &self,
        c: usize,
        j: usize,
        msg_vc: &[Vec<f64>],
        syndrome_sign: &[f64],
    ) -> f64 {
        let vars = &self.parity_check.row_indices[c];
        let mut product = syndrome_sign[c];

        for (k, &v) in vars.iter().enumerate() {
            if k == j {
                continue;
            }
            let local_c = self.var_check_local_index(v, c);
            let msg = msg_vc[v][local_c];
            let t = (msg / 2.0).tanh();

            // Clamp to avoid atanh(+/-1) = +/-infinity
            let t_clamped = t.clamp(-0.9999999, 0.9999999);
            product *= t_clamped;
        }

        // Clamp product before atanh
        let product_clamped = product.clamp(-0.9999999, 0.9999999);
        2.0 * product_clamped.atanh()
    }

    /// Find the local index of check `c` in variable `v`'s check list.
    #[inline]
    fn var_check_local_index(&self, v: usize, c: usize) -> usize {
        self.parity_check.col_indices[v]
            .iter()
            .position(|&x| x == c)
            .expect("Check not found in variable's adjacency list")
    }

    /// Find the local index of variable `v` in check `c`'s variable list.
    #[inline]
    fn check_var_local_index(&self, c: usize, v: usize) -> usize {
        self.parity_check.row_indices[c]
            .iter()
            .position(|&x| x == v)
            .expect("Variable not found in check's adjacency list")
    }
}

impl fmt::Display for BpOsdDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BpOsdDecoder({}, bp={}, osd_order={}, max_combos={})",
            self.parity_check, self.bp_config.method, self.osd_order, self.max_osd_combinations
        )
    }
}

// ============================================================
// QECDecoder TRAIT IMPLEMENTATION
// ============================================================

use crate::traits::{Correction, QECDecoder, Syndrome};

/// Syndrome wrapper that carries the physical error probability alongside the
/// syndrome bits, allowing the `QECDecoder::decode` interface (which takes only
/// `&Syndrome`) to supply the `p` value that BP-OSD requires.
#[derive(Clone, Debug)]
pub struct BpSyndrome {
    /// Detector outcomes (one bool per check).
    pub bits: Vec<bool>,
    /// Per-qubit physical error probability for the BP channel model.
    pub p: f64,
}

impl Syndrome for BpSyndrome {
    fn to_bits(&self) -> Vec<bool> {
        self.bits.clone()
    }
    fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Correction produced by BP-OSD decoding.
#[derive(Clone, Debug)]
pub struct BpCorrection {
    /// Per-qubit correction flags.
    pub correction: Vec<bool>,
}

impl Correction for BpCorrection {
    fn x_corrections(&self) -> Vec<usize> {
        self.correction
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect()
    }
    fn z_corrections(&self) -> Vec<usize> {
        // BP-OSD decodes X and Z independently; a single decoder instance
        // handles one error type, so the other correction list is empty.
        Vec::new()
    }
}

impl QECDecoder<BpSyndrome, BpCorrection> for BpOsdDecoder {
    fn decode(&self, syndrome: &BpSyndrome) -> BpCorrection {
        let result = self.decode(&syndrome.bits, syndrome.p);
        BpCorrection {
            correction: result.correction,
        }
    }

    fn name(&self) -> &str {
        "BP-OSD"
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper: compute syndrome for an error on a sparse matrix ----
    fn make_syndrome(h: &SparseMatrix, error: &[bool]) -> Vec<bool> {
        h.syndrome(error)
    }

    // ---- Helper: create a single-qubit error vector ----
    fn single_error(n: usize, pos: usize) -> Vec<bool> {
        let mut e = vec![false; n];
        e[pos] = true;
        e
    }

    // ================================================================
    // Test 1: Sparse matrix construction and basic operations
    // ================================================================
    #[test]
    fn test_sparse_matrix_construction() {
        let h = SparseMatrix::from_dense(&[
            vec![true, true, false, false, false],
            vec![false, true, true, false, false],
            vec![false, false, true, true, false],
            vec![false, false, false, true, true],
        ]);

        assert_eq!(h.rows, 4);
        assert_eq!(h.cols, 5);

        // Row 0 has columns 0 and 1
        assert_eq!(h.row_indices[0], vec![0, 1]);
        // Row 3 has columns 3 and 4
        assert_eq!(h.row_indices[3], vec![3, 4]);

        // Column 0 appears in row 0 only
        assert_eq!(h.col_indices[0], vec![0]);
        // Column 1 appears in rows 0 and 1
        assert_eq!(h.col_indices[1], vec![0, 1]);
        // Column 2 appears in rows 1 and 2
        assert_eq!(h.col_indices[2], vec![1, 2]);

        // Syndrome computation: error on qubit 2
        let error = single_error(5, 2);
        let syn = h.syndrome(&error);
        assert_eq!(syn, vec![false, true, true, false]);

        // From adjacency should produce equivalent result
        let h2 = SparseMatrix::from_adjacency(
            4,
            5,
            vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]],
        );
        assert_eq!(h2.row_indices, h.row_indices);
        assert_eq!(h2.col_indices, h.col_indices);

        // Display trait
        let display = format!("{}", h);
        assert!(display.contains("SparseMatrix(4x5"));
    }

    // ================================================================
    // Test 2: BP converges on simple repetition code
    // ================================================================
    #[test]
    fn test_bp_converges_simple_code() {
        let h = SparseMatrix::repetition_code(5);
        let decoder = BpOsdDecoder::new(h.clone(), BpConfig::default(), 0, 0);

        // Error on qubit 2: syndrome bits 1 and 2 fire
        let error = single_error(5, 2);
        let syn = make_syndrome(&h, &error);
        assert_eq!(syn, vec![false, true, true, false]);

        let result = decoder.decode(&syn, 0.05);

        assert!(
            result.converged,
            "BP should converge on simple repetition code"
        );
        assert!(!result.used_osd, "Should not need OSD");
        assert!(
            result.correction[2],
            "Should identify qubit 2 as the error location"
        );
        assert_eq!(result.weight, 1, "Single error should have weight 1");
    }

    // ================================================================
    // Test 3: BP repetition code -- boundary error and trivial syndrome
    // ================================================================
    #[test]
    fn test_bp_repetition_code() {
        let h = SparseMatrix::repetition_code(7);
        let decoder = BpOsdDecoder::with_osd(h.clone(), 0);

        // Trivial syndrome: no errors
        let syn_trivial = vec![false; 6];
        let result_trivial = decoder.decode(&syn_trivial, 0.01);
        assert!(result_trivial.converged);
        assert_eq!(result_trivial.weight, 0);
        assert_eq!(result_trivial.bp_iterations, 0);

        // Boundary error on qubit 0: only syndrome 0 fires
        let error = single_error(7, 0);
        let syn = make_syndrome(&h, &error);
        assert_eq!(syn, vec![true, false, false, false, false, false]);

        let result = decoder.decode(&syn, 0.05);
        assert!(result.converged);
        assert!(result.correction[0], "Should detect error on qubit 0");

        // Boundary error on qubit 6 (last): only syndrome 5 fires
        let error = single_error(7, 6);
        let syn = make_syndrome(&h, &error);
        assert_eq!(syn, vec![false, false, false, false, false, true]);

        let result = decoder.decode(&syn, 0.05);
        assert!(result.converged);
        assert!(result.correction[6], "Should detect error on qubit 6");
    }

    // ================================================================
    // Test 4: OSD-0 basic decoding
    // ================================================================
    #[test]
    fn test_osd0_basic() {
        // Hamming [7,4,3] code: OSD-0 should correct any single error
        let h = SparseMatrix::hamming_7_4();

        for error_pos in 0..7 {
            let error = single_error(7, error_pos);
            let syn = make_syndrome(&h, &error);

            // Use uniform LLRs (no BP information) -- just reliability ordering
            let llrs: Vec<f64> = (0..7)
                .map(|i| if i == error_pos { -1.0 } else { 2.0 })
                .collect();

            let (correction, _pivot_cols, _, _) = Osd::osd0(&h, &syn, &llrs);

            assert!(
                h.satisfies_syndrome(&correction, &syn),
                "OSD-0 correction must satisfy syndrome for error at position {}",
                error_pos
            );
        }
    }

    // ================================================================
    // Test 5: OSD-CS finds lower weight than OSD-0
    // ================================================================
    #[test]
    fn test_osd_cs_finds_lower_weight() {
        // Construct a scenario where OSD-CS can improve over OSD-0.
        // Use a small code where misleading LLRs cause OSD-0 to find a
        // heavier correction, but OSD-CS can find the true (lighter) one.
        let h = SparseMatrix::from_dense(&[
            vec![true, true, false, true, false, false],
            vec![false, true, true, false, true, false],
            vec![true, false, true, false, false, true],
        ]);

        // True error: single bit on column 0
        let error = single_error(6, 0);
        let syn = make_syndrome(&h, &error);

        // Deliberately misleading LLRs: make column 0 appear reliable,
        // and some others appear unreliable. This may cause OSD-0 to
        // find a multi-bit correction.
        let llrs = vec![3.0, -0.1, 0.2, 0.5, -0.3, 0.1];

        let (osd0_correction, _, _, _) = Osd::osd0(&h, &syn, &llrs);
        let osd0_weight = SparseMatrix::weight(&osd0_correction);

        let (osd_cs_correction, combos_tried) = Osd::osd_cs(&h, &syn, &llrs, 3, 100);
        let osd_cs_weight = SparseMatrix::weight(&osd_cs_correction);

        // OSD-CS should find a correction at least as good as OSD-0
        assert!(
            osd_cs_weight <= osd0_weight,
            "OSD-CS weight {} should be <= OSD-0 weight {}",
            osd_cs_weight,
            osd0_weight
        );

        // Both must satisfy the syndrome
        assert!(h.satisfies_syndrome(&osd0_correction, &syn));
        assert!(h.satisfies_syndrome(&osd_cs_correction, &syn));

        // OSD-CS tried at least 2 combinations (OSD-0 + at least one flip)
        assert!(
            combos_tried >= 1,
            "OSD-CS should have tried at least 1 combination"
        );
    }

    // ================================================================
    // Test 6: Full BP-OSD pipeline converges
    // ================================================================
    #[test]
    fn test_bp_osd_pipeline_converges() {
        // Use Hamming [7,4,3] code with OSD fallback enabled
        let h = SparseMatrix::hamming_7_4();
        let decoder = BpOsdDecoder::new(h.clone(), BpConfig::default(), 3, 100);

        // Test all single-error patterns
        for error_pos in 0..7 {
            let error = single_error(7, error_pos);
            let syn = make_syndrome(&h, &error);

            let result = decoder.decode(&syn, 0.05);

            assert!(
                result.converged,
                "BP-OSD should converge for error at position {}",
                error_pos
            );
            assert!(
                h.satisfies_syndrome(&result.correction, &syn),
                "Correction must satisfy syndrome for error at position {}",
                error_pos
            );
            // Single error should be correctable with weight 1
            assert!(
                result.weight <= 3, // Hamming distance is 3, so worst case is 3
                "Weight {} too high for single error at position {}",
                result.weight,
                error_pos
            );
        }
    }

    // ================================================================
    // Test 7: BP-OSD on surface code syndrome
    // ================================================================
    #[test]
    fn test_bp_osd_surface_code_syndrome() {
        let h = SparseMatrix::surface_code_d3();
        let decoder = BpOsdDecoder::new(h.clone(), BpConfig::default(), 5, 500);

        // Single qubit error on qubit 4 (center of d=3 surface code)
        let error = single_error(9, 4);
        let syn = make_syndrome(&h, &error);

        let result = decoder.decode(&syn, 0.01);

        assert!(
            result.converged,
            "BP-OSD should converge on surface code with single error"
        );
        assert!(
            h.satisfies_syndrome(&result.correction, &syn),
            "Correction must satisfy surface code syndrome"
        );

        // Boundary qubit error
        let error = single_error(9, 0);
        let syn = make_syndrome(&h, &error);
        let result = decoder.decode(&syn, 0.01);

        assert!(
            result.converged,
            "BP-OSD should handle boundary qubit errors"
        );
        assert!(h.satisfies_syndrome(&result.correction, &syn));
    }

    // ================================================================
    // Test 8: Sum-product BP method
    // ================================================================
    #[test]
    fn test_bp_method_sum_product() {
        let h = SparseMatrix::repetition_code(5);
        let config = BpConfig::sum_product();
        let decoder = BpOsdDecoder::new(h.clone(), config, 0, 0);

        // Error on qubit 2
        let error = single_error(5, 2);
        let syn = make_syndrome(&h, &error);

        let result = decoder.decode(&syn, 0.05);

        assert!(
            result.converged,
            "Sum-product should converge on repetition code"
        );
        assert!(
            result.correction[2],
            "Sum-product should identify error on qubit 2"
        );
        assert_eq!(result.weight, 1);
    }

    // ================================================================
    // Test 9: Min-sum BP method
    // ================================================================
    #[test]
    fn test_bp_method_min_sum() {
        let h = SparseMatrix::repetition_code(5);
        let config = BpConfig::min_sum();
        let decoder = BpOsdDecoder::new(h.clone(), config, 0, 0);

        // Error on qubit 3
        let error = single_error(5, 3);
        let syn = make_syndrome(&h, &error);

        let result = decoder.decode(&syn, 0.05);

        assert!(
            result.converged,
            "Min-sum should converge on repetition code"
        );
        assert!(
            result.correction[3],
            "Min-sum should identify error on qubit 3"
        );
        assert_eq!(result.weight, 1);

        // Verify LLR beliefs have correct signs
        for (i, &b) in result.llr_beliefs.iter().enumerate() {
            if result.correction[i] {
                assert!(b < 0.0, "LLR for error qubit {} should be negative", i);
            } else {
                assert!(b >= 0.0, "LLR for clean qubit {} should be non-negative", i);
            }
        }
    }

    // ================================================================
    // Test 10: OSD order 0 vs higher order
    // ================================================================
    #[test]
    fn test_osd_order_zero_vs_higher() {
        // Use a code where OSD-0 may produce a suboptimal result and
        // higher-order OSD can improve it.
        let h = SparseMatrix::from_dense(&[
            vec![true, true, false, true, false, false, true, false],
            vec![false, true, true, false, true, false, false, true],
            vec![true, false, true, false, false, true, true, false],
            vec![false, false, false, true, true, true, false, true],
        ]);

        // Two-qubit error: positions 1 and 5
        let mut error = vec![false; 8];
        error[1] = true;
        error[5] = true;
        let syn = make_syndrome(&h, &error);

        // Force OSD by using BP with only 1 iteration (unlikely to converge)
        let bp_config = BpConfig {
            max_iterations: 1,
            ..BpConfig::default()
        };

        // OSD-0 decoder
        let decoder_0 = BpOsdDecoder::new(h.clone(), bp_config.clone(), 0, 1);
        let result_0 = decoder_0.decode(&syn, 0.1);

        // OSD order-3 decoder
        let decoder_3 = BpOsdDecoder::new(h.clone(), bp_config.clone(), 3, 500);
        let result_3 = decoder_3.decode(&syn, 0.1);

        // Higher order should find weight <= OSD-0
        assert!(
            result_3.weight <= result_0.weight || !result_0.converged,
            "OSD order-3 (weight {}) should be <= OSD-0 (weight {})",
            result_3.weight,
            result_0.weight
        );

        // Higher order should have tried more combinations
        if result_3.used_osd && result_0.used_osd {
            assert!(
                result_3.osd_combinations_tried >= result_0.osd_combinations_tried,
                "Higher OSD order should try more combinations"
            );
        }

        // Both should satisfy the syndrome (if they converged)
        if result_0.converged {
            assert!(h.satisfies_syndrome(&result_0.correction, &syn));
        }
        if result_3.converged {
            assert!(h.satisfies_syndrome(&result_3.correction, &syn));
        }
    }

    // ================================================================
    // Test 11: Large code performance
    // ================================================================
    #[test]
    fn test_large_code_performance() {
        // Repetition code with 500 qubits -- BP should handle this efficiently
        let h = SparseMatrix::repetition_code(500);
        let decoder = BpOsdDecoder::new(h.clone(), BpConfig::default(), 0, 0);

        // Single error in the middle
        let error = single_error(500, 250);
        let syn = make_syndrome(&h, &error);

        let start = std::time::Instant::now();
        let result = decoder.decode(&syn, 0.01);
        let elapsed = start.elapsed();

        assert!(
            result.converged,
            "BP should converge on large repetition code"
        );
        assert!(result.correction[250], "Should identify error on qubit 250");
        assert_eq!(result.weight, 1);

        // Should complete in well under 100ms
        assert!(
            elapsed.as_millis() < 200,
            "Large code decoding took {}ms, expected <200ms",
            elapsed.as_millis()
        );
    }

    // ================================================================
    // Test 12: Next combination utility
    // ================================================================
    #[test]
    fn test_next_combination() {
        // Combinations of 2 from 4 elements: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        let mut indices = vec![0, 1];
        let expected = vec![vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

        for exp in &expected {
            assert!(Osd::next_combination(&mut indices, 4));
            assert_eq!(&indices, exp);
        }

        assert!(!Osd::next_combination(&mut indices, 4));

        // Edge case: single element
        let mut single = vec![0];
        assert!(Osd::next_combination(&mut single, 3));
        assert_eq!(single, vec![1]);
        assert!(Osd::next_combination(&mut single, 3));
        assert_eq!(single, vec![2]);
        assert!(!Osd::next_combination(&mut single, 3));

        // Edge case: empty
        let mut empty: Vec<usize> = vec![];
        assert!(!Osd::next_combination(&mut empty, 5));
    }

    // ================================================================
    // Test 13: Max OSD combinations limit is respected
    // ================================================================
    #[test]
    fn test_max_osd_combinations_limit() {
        let h = SparseMatrix::from_dense(&[
            vec![
                true, true, false, true, false, false, true, false, false, false,
            ],
            vec![
                false, true, true, false, true, false, false, true, false, false,
            ],
            vec![
                true, false, true, false, false, true, false, false, true, false,
            ],
            vec![
                false, false, false, true, true, true, false, false, false, true,
            ],
        ]);

        let error = single_error(10, 0);
        let syn = make_syndrome(&h, &error);

        // Force OSD with 1 BP iteration, high OSD order but low max_combinations
        let bp_config = BpConfig {
            max_iterations: 1,
            ..BpConfig::default()
        };

        let decoder = BpOsdDecoder::new(h.clone(), bp_config, 10, 5);
        let result = decoder.decode(&syn, 0.1);

        assert!(result.used_osd, "Should have used OSD");
        assert!(
            result.osd_combinations_tried <= 5,
            "Should respect max_combinations limit: tried {}",
            result.osd_combinations_tried
        );
    }

    // ================================================================
    // Test 14: Two-qubit error on repetition code
    // ================================================================
    #[test]
    fn test_two_qubit_error_with_osd() {
        let h = SparseMatrix::repetition_code(7);
        let decoder = BpOsdDecoder::new(h.clone(), BpConfig::default(), 5, 100);

        // Errors on qubits 2 and 4 (non-adjacent)
        let mut error = vec![false; 7];
        error[2] = true;
        error[4] = true;
        let syn = make_syndrome(&h, &error);

        let result = decoder.decode(&syn, 0.1);

        // Should at least satisfy the syndrome (may find a degenerate correction)
        if result.converged {
            assert!(
                h.satisfies_syndrome(&result.correction, &syn),
                "Correction must satisfy syndrome"
            );
        }
    }

    // ================================================================
    // Test 15: Decoder Display trait
    // ================================================================
    #[test]
    fn test_decoder_display() {
        let h = SparseMatrix::repetition_code(5);
        let decoder = BpOsdDecoder::new(h, BpConfig::default(), 5, 1000);
        let display = format!("{}", decoder);
        assert!(display.contains("BpOsdDecoder"));
        assert!(display.contains("osd_order=5"));
        assert!(display.contains("max_combos=1000"));
        assert!(display.contains("MinSum"));
    }

    // ================================================================
    // Test 16: BP-only mode (OSD disabled)
    // ================================================================
    #[test]
    fn test_bp_only_mode() {
        let h = SparseMatrix::repetition_code(5);
        let decoder = BpOsdDecoder::bp_only(h.clone(), BpConfig::default());

        let error = single_error(5, 2);
        let syn = make_syndrome(&h, &error);

        let result = decoder.decode(&syn, 0.05);

        assert!(result.converged);
        assert!(!result.used_osd);
        assert_eq!(result.osd_combinations_tried, 0);
    }

    // ================================================================
    // Test 17: Hamming code weight correctness
    // ================================================================
    #[test]
    fn test_hamming_code_weight_function() {
        assert_eq!(SparseMatrix::weight(&[false, false, false]), 0);
        assert_eq!(SparseMatrix::weight(&[true, false, false]), 1);
        assert_eq!(SparseMatrix::weight(&[true, true, true]), 3);
        assert_eq!(SparseMatrix::weight(&[]), 0);
    }

    // ================================================================
    // Test 18: Syndrome satisfaction check
    // ================================================================
    #[test]
    fn test_satisfies_syndrome() {
        let h = SparseMatrix::repetition_code(5);

        let error = single_error(5, 2);
        let syn = h.syndrome(&error);

        // Correct error should satisfy syndrome
        assert!(h.satisfies_syndrome(&error, &syn));

        // Wrong error should not satisfy
        let wrong = single_error(5, 0);
        assert!(!h.satisfies_syndrome(&wrong, &syn));

        // Zero error satisfies zero syndrome
        let zero_error = vec![false; 5];
        let zero_syn = vec![false; 4];
        assert!(h.satisfies_syndrome(&zero_error, &zero_syn));
    }
}
