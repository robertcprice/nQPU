//! Grover's Quantum Search Algorithm
//!
//! Implements Grover's algorithm for unstructured database search, providing
//! a quadratic speedup over classical brute-force search. Given an oracle that
//! marks M "good" states out of N = 2^n total basis states, Grover's algorithm
//! finds a marked state with high probability in O(sqrt(N/M)) oracle queries.
//!
//! # Algorithm Overview
//!
//! 1. **Initialize** the state to a uniform superposition: H^n |0...0>
//! 2. **Iterate** (repeat ~pi/4 * sqrt(N/M) times):
//!    - **Oracle**: Flip the phase of all marked states (|x> -> -|x> if x is marked)
//!    - **Diffusion**: Reflect about the uniform superposition (2|s><s| - I)
//! 3. **Measure**: The marked state(s) appear with near-certainty
//!
//! The diffusion operator is applied efficiently via the identity:
//!   (2|s><s| - I)|psi> = 2*mean(psi) - psi[i] for each amplitude i,
//! avoiding materialization of the full N x N matrix.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::grover::{GroverOracle, GroverSearch};
//!
//! let oracle = GroverOracle::single_target(3, 5);
//! let search = GroverSearch::new(oracle);
//! let result = search.search();
//! assert_eq!(result.found_state, 5);
//! assert!(result.success_probability > 0.9);
//! ```
//!
//! # References
//!
//! - Grover, L. K. "A fast quantum mechanical algorithm for database search"
//!   (1996). Proceedings of STOC, pp. 212-219.
//! - Boyer, Brassard, Hoyer, Tapp. "Tight bounds on quantum searching" (1998).
//!   Fortschritte der Physik 46(4-5), pp. 493-505.

use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// LOCAL HELPERS
// ============================================================

type C64 = Complex64;

#[inline]
fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

#[inline]
fn c64_zero() -> C64 {
    c64(0.0, 0.0)
}

#[inline]
fn c64_one() -> C64 {
    c64(1.0, 0.0)
}

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors that can occur during Grover's search.
#[derive(Debug, Clone, PartialEq)]
pub enum GroverError {
    /// The number of qubits is zero.
    ZeroQubits,
    /// A marked state index is out of range for the given number of qubits.
    MarkedStateOutOfRange {
        state: usize,
        max_state: usize,
    },
    /// No marked states were provided.
    NoMarkedStates,
    /// All states are marked (search is trivial but degenerate for Grover iteration).
    AllStatesMarked,
}

impl std::fmt::Display for GroverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GroverError::ZeroQubits => write!(f, "Number of qubits must be at least 1"),
            GroverError::MarkedStateOutOfRange { state, max_state } => {
                write!(
                    f,
                    "Marked state {} is out of range (max {})",
                    state, max_state
                )
            }
            GroverError::NoMarkedStates => write!(f, "At least one marked state is required"),
            GroverError::AllStatesMarked => write!(f, "All states are marked; search is trivial"),
        }
    }
}

impl std::error::Error for GroverError {}

// ============================================================
// ORACLE
// ============================================================

/// Defines which basis states are "marked" (solutions) for Grover's search.
///
/// The oracle encodes a boolean function f: {0, ..., N-1} -> {0, 1} where
/// f(x) = 1 for marked states and f(x) = 0 otherwise. During the search,
/// the oracle operator flips the phase of marked states: |x> -> (-1)^f(x) |x>.
#[derive(Debug, Clone)]
pub struct GroverOracle {
    /// Number of qubits defining the search space (N = 2^num_qubits).
    pub num_qubits: usize,
    /// Basis state indices that are "good" (solutions to the search problem).
    pub marked_states: Vec<usize>,
}

impl GroverOracle {
    /// Create a new oracle with explicit marked states.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits (search space size N = 2^num_qubits).
    /// * `marked_states` - Indices of basis states that the oracle marks as solutions.
    ///
    /// # Errors
    ///
    /// Returns `GroverError` if `num_qubits` is zero, `marked_states` is empty,
    /// any index exceeds 2^num_qubits - 1, or all states are marked.
    pub fn new(num_qubits: usize, marked_states: Vec<usize>) -> Result<Self, GroverError> {
        if num_qubits == 0 {
            return Err(GroverError::ZeroQubits);
        }
        if marked_states.is_empty() {
            return Err(GroverError::NoMarkedStates);
        }
        let n = 1usize << num_qubits;
        for &s in &marked_states {
            if s >= n {
                return Err(GroverError::MarkedStateOutOfRange {
                    state: s,
                    max_state: n - 1,
                });
            }
        }
        if marked_states.len() >= n {
            return Err(GroverError::AllStatesMarked);
        }
        Ok(Self {
            num_qubits,
            marked_states,
        })
    }

    /// Convenience constructor for searching for a single target state.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits.
    /// * `target` - The single basis state index to search for.
    pub fn single_target(num_qubits: usize, target: usize) -> Result<Self, GroverError> {
        Self::new(num_qubits, vec![target])
    }

    /// Build an oracle from a boolean predicate function.
    ///
    /// Evaluates `predicate(x)` for all x in {0, ..., 2^num_qubits - 1} and
    /// marks every state where the predicate returns true.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits.
    /// * `predicate` - A function that returns `true` for states that should be marked.
    pub fn from_predicate<F>(num_qubits: usize, predicate: F) -> Result<Self, GroverError>
    where
        F: Fn(usize) -> bool,
    {
        if num_qubits == 0 {
            return Err(GroverError::ZeroQubits);
        }
        let n = 1usize << num_qubits;
        let marked: Vec<usize> = (0..n).filter(|&x| predicate(x)).collect();
        if marked.is_empty() {
            return Err(GroverError::NoMarkedStates);
        }
        if marked.len() >= n {
            return Err(GroverError::AllStatesMarked);
        }
        Ok(Self {
            num_qubits,
            marked_states: marked,
        })
    }

    /// The total number of basis states in the search space (N = 2^num_qubits).
    #[inline]
    pub fn search_space_size(&self) -> usize {
        1usize << self.num_qubits
    }

    /// The number of marked (solution) states.
    #[inline]
    pub fn num_marked(&self) -> usize {
        self.marked_states.len()
    }
}

// ============================================================
// RESULT TYPE
// ============================================================

/// Result of a Grover search execution.
#[derive(Debug, Clone)]
pub struct GroverResult {
    /// The basis state index found (highest-probability state after measurement).
    pub found_state: usize,
    /// Binary string representation of the found state (e.g., "101" for state 5 with 3 qubits).
    pub found_state_binary: String,
    /// Probability of measuring the found state.
    pub success_probability: f64,
    /// Number of Grover iterations performed.
    pub num_iterations: usize,
    /// Number of oracle calls (equal to num_iterations).
    pub num_oracle_calls: usize,
    /// Full probability distribution over all basis states after the search.
    pub state_probabilities: Vec<f64>,
}

// ============================================================
// GROVER SEARCH
// ============================================================

/// Grover's quantum search algorithm.
///
/// Executes the Grover iterate (oracle + diffusion) on a uniform superposition
/// to amplify the amplitude of marked states. The number of iterations can be
/// specified explicitly or computed automatically for optimal success probability.
///
/// The simulation uses direct state-vector evolution with O(N) per iteration
/// by exploiting the structure of both operators:
/// - **Oracle**: diagonal operator, O(N) element-wise phase flip
/// - **Diffusion**: 2|s><s| - I applied as 2*mean - amplitude[i], also O(N)
#[derive(Debug, Clone)]
pub struct GroverSearch {
    /// The oracle defining which states are marked.
    pub oracle: GroverOracle,
    /// Number of iterations to perform. `None` means auto-compute the optimal count.
    pub num_iterations: Option<usize>,
}

impl GroverSearch {
    /// Create a new Grover search with automatically computed optimal iterations.
    ///
    /// The optimal number of iterations is floor(pi/4 * sqrt(N/M)) where
    /// N = 2^num_qubits and M is the number of marked states.
    pub fn new(oracle: GroverOracle) -> Self {
        Self {
            oracle,
            num_iterations: None,
        }
    }

    /// Create a Grover search with an explicit iteration count.
    ///
    /// Use this when you want to study the algorithm's behavior at specific
    /// iteration counts, or when the optimal count is known externally.
    pub fn with_iterations(oracle: GroverOracle, iterations: usize) -> Self {
        Self {
            oracle,
            num_iterations: Some(iterations),
        }
    }

    /// Compute the optimal number of Grover iterations for the given parameters.
    ///
    /// Returns floor(pi/4 * sqrt(N/M)) which maximizes the probability of
    /// measuring a marked state. For N/M very large, this approaches the
    /// theoretical O(sqrt(N/M)) query complexity.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits (search space N = 2^num_qubits).
    /// * `num_marked` - Number of marked (solution) states M.
    ///
    /// # Returns
    ///
    /// The optimal iteration count, with a minimum of 1.
    pub fn optimal_iterations(num_qubits: usize, num_marked: usize) -> usize {
        if num_marked == 0 {
            return 0;
        }
        let n = (1usize << num_qubits) as f64;
        let m = num_marked as f64;
        let theta = (m / n).sqrt().asin();
        if theta < 1e-15 {
            return 0;
        }
        // Optimal k = round(pi/(4*theta) - 1/2) but the standard formula
        // floor(pi/4 * sqrt(N/M)) is more commonly cited and equivalent
        // for the regime where M << N.
        let k = (PI / (4.0 * theta)).floor() as usize;
        k.max(1)
    }

    /// Compute the theoretical success probability after `iterations` Grover steps.
    ///
    /// The probability of measuring a marked state after k iterations is:
    ///   P(k) = sin^2((2k + 1) * theta)
    /// where theta = arcsin(sqrt(M/N)).
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of Grover iterations (oracle + diffusion applications).
    pub fn success_probability(&self, iterations: usize) -> f64 {
        let n = self.oracle.search_space_size() as f64;
        let m = self.oracle.num_marked() as f64;
        let theta = (m / n).sqrt().asin();
        let angle = (2 * iterations + 1) as f64 * theta;
        angle.sin().powi(2)
    }

    /// Execute the Grover search and return the result.
    ///
    /// Simulates the full state-vector evolution:
    /// 1. Prepare uniform superposition H^n |0...0>
    /// 2. Repeat for `num_iterations` steps:
    ///    a. Apply oracle (phase flip marked states)
    ///    b. Apply diffusion operator (2|s><s| - I)
    /// 3. Measure (find the basis state with maximum probability)
    pub fn search(&self) -> GroverResult {
        let num_qubits = self.oracle.num_qubits;
        let n = self.oracle.search_space_size();
        let num_marked = self.oracle.num_marked();

        // Determine iteration count
        let iterations = self
            .num_iterations
            .unwrap_or_else(|| Self::optimal_iterations(num_qubits, num_marked));

        // Step 1: Initialize to uniform superposition |s> = H^n |0>
        let mut state = uniform_superposition(num_qubits);

        // Build the oracle diagonal: -1 for marked states, +1 otherwise
        let oracle_diag = oracle_operator(&self.oracle);

        // Step 2: Grover iteration
        for _ in 0..iterations {
            // 2a: Apply oracle (element-wise multiply by diagonal)
            apply_diagonal(&mut state, &oracle_diag);

            // 2b: Apply diffusion operator via the mean formula
            //     new_amp[i] = 2 * mean(amp) - amp[i]
            apply_diffusion(&mut state);
        }

        // Step 3: Extract probabilities and find the most probable state
        let probabilities: Vec<f64> = state.iter().map(|a| a.norm_sqr()).collect();

        let (found_state, &success_probability) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let found_state_binary = format!("{:0>width$b}", found_state, width = num_qubits);

        GroverResult {
            found_state,
            found_state_binary,
            success_probability,
            num_iterations: iterations,
            num_oracle_calls: iterations,
            state_probabilities: probabilities,
        }
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Prepare the uniform superposition state H^n |0...0>.
///
/// Returns a state vector of length 2^num_qubits where every amplitude
/// is 1/sqrt(2^num_qubits), corresponding to applying Hadamard gates
/// to all qubits initialized in |0>.
pub fn uniform_superposition(num_qubits: usize) -> Vec<C64> {
    let n = 1usize << num_qubits;
    let amp = c64(1.0 / (n as f64).sqrt(), 0.0);
    vec![amp; n]
}

/// Build the oracle operator as a diagonal vector.
///
/// Returns a length-N vector where entry i is -1 if state i is marked
/// and +1 otherwise. This is the diagonal of the oracle unitary
/// O = I - 2 * sum_{m in marked} |m><m|.
pub fn oracle_operator(oracle: &GroverOracle) -> Vec<C64> {
    let n = oracle.search_space_size();
    let mut diag = vec![c64_one(); n];
    for &m in &oracle.marked_states {
        diag[m] = c64(-1.0, 0.0);
    }
    diag
}

/// Build the full diffusion operator matrix 2|s><s| - I as a flat row-major matrix.
///
/// This is primarily useful for testing and verification. The actual search
/// uses the O(N) mean-based formula instead of materializing this N x N matrix.
///
/// The diffusion operator D has entries:
///   D[i][j] = 2/N - delta_{ij}
/// where delta_{ij} is the Kronecker delta.
pub fn diffusion_operator(num_qubits: usize) -> Vec<C64> {
    let n = 1usize << num_qubits;
    let two_over_n = c64(2.0 / n as f64, 0.0);
    let mut matrix = vec![c64_zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            matrix[i * n + j] = if i == j {
                two_over_n - c64_one()
            } else {
                two_over_n
            };
        }
    }
    matrix
}

/// Apply a diagonal operator to a state vector in-place (element-wise multiply).
///
/// This is O(N) and avoids the full N x N matrix-vector product.
#[inline]
pub fn apply_diagonal(state: &mut [C64], diagonal: &[C64]) {
    debug_assert_eq!(state.len(), diagonal.len());
    for (s, &d) in state.iter_mut().zip(diagonal.iter()) {
        *s *= d;
    }
}

/// Apply the Grover diffusion operator to a state vector in-place.
///
/// Uses the identity: (2|s><s| - I)|psi>[i] = 2 * mean(psi) - psi[i],
/// which is O(N) rather than the O(N^2) of full matrix-vector multiplication.
fn apply_diffusion(state: &mut [C64]) {
    let n = state.len();
    // Compute the mean amplitude
    let mean: C64 = state.iter().copied().sum::<C64>() / c64(n as f64, 0.0);
    let two_mean = mean * c64(2.0, 0.0);
    // Apply: new_amp[i] = 2*mean - amp[i]
    for s in state.iter_mut() {
        *s = two_mean - *s;
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // ----------------------------------------------------------
    // 1. Single target search (3 qubits, target=5)
    // ----------------------------------------------------------
    #[test]
    fn test_single_target_search() {
        let oracle = GroverOracle::single_target(3, 5).unwrap();
        let search = GroverSearch::new(oracle);
        let result = search.search();

        assert_eq!(
            result.found_state, 5,
            "Should find target state 5, got {}",
            result.found_state
        );
        assert!(
            result.success_probability > 0.9,
            "Success probability {:.4} should be > 0.9",
            result.success_probability
        );
        assert_eq!(result.state_probabilities.len(), 8, "N=8 for 3 qubits");
    }

    // ----------------------------------------------------------
    // 2. Multiple targets (4 qubits, targets=[3,7,11])
    // ----------------------------------------------------------
    #[test]
    fn test_multiple_targets() {
        let targets = vec![3, 7, 11];
        let oracle = GroverOracle::new(4, targets.clone()).unwrap();
        let search = GroverSearch::new(oracle);
        let result = search.search();

        assert!(
            targets.contains(&result.found_state),
            "Found state {} should be one of {:?}",
            result.found_state,
            targets
        );
        // Total probability on marked states should be high
        let total_marked_prob: f64 = targets
            .iter()
            .map(|&t| result.state_probabilities[t])
            .sum();
        assert!(
            total_marked_prob > 0.9,
            "Total marked state probability {:.4} should be > 0.9",
            total_marked_prob
        );
    }

    // ----------------------------------------------------------
    // 3. Optimal iterations formula
    // ----------------------------------------------------------
    #[test]
    fn test_optimal_iterations() {
        // N=1024 (10 qubits), M=1: floor(pi/4 * sqrt(1024)) ~ 25
        let iters = GroverSearch::optimal_iterations(10, 1);
        assert_eq!(
            iters, 25,
            "optimal_iterations(10, 1) should be 25, got {}",
            iters
        );

        // N=8 (3 qubits), M=1: floor(pi/4 * sqrt(8)) ~ 2
        let iters_3 = GroverSearch::optimal_iterations(3, 1);
        assert_eq!(
            iters_3, 2,
            "optimal_iterations(3, 1) should be 2, got {}",
            iters_3
        );

        // N=4 (2 qubits), M=1: floor(pi/4 * sqrt(4)) ~ 1
        let iters_2 = GroverSearch::optimal_iterations(2, 1);
        assert_eq!(
            iters_2, 1,
            "optimal_iterations(2, 1) should be 1, got {}",
            iters_2
        );
    }

    // ----------------------------------------------------------
    // 4. Success probability > 0.9 at optimal iterations
    // ----------------------------------------------------------
    #[test]
    fn test_success_probability_at_optimal() {
        // N=8, M=1
        let oracle = GroverOracle::single_target(3, 0).unwrap();
        let search = GroverSearch::new(oracle);
        let optimal = GroverSearch::optimal_iterations(3, 1);
        let prob = search.success_probability(optimal);
        assert!(
            prob > 0.9,
            "Success probability at optimal iterations ({}) should be > 0.9, got {:.6}",
            optimal,
            prob
        );
    }

    // ----------------------------------------------------------
    // 5. Two qubits, one target (simplest nontrivial case)
    // ----------------------------------------------------------
    #[test]
    fn test_two_qubits_single_target() {
        // N=4, M=1: optimal iterations = 1
        let oracle = GroverOracle::single_target(2, 2).unwrap();
        let search = GroverSearch::new(oracle);
        let result = search.search();

        assert_eq!(result.num_iterations, 1, "Optimal iterations should be 1");
        assert_eq!(result.found_state, 2, "Should find target state 2");
        // For N=4, M=1: sin^2(3*theta) where theta=arcsin(1/2)=pi/6
        // = sin^2(3*pi/6) = sin^2(pi/2) = 1.0
        assert!(
            (result.success_probability - 1.0).abs() < TOL,
            "Success probability should be ~1.0, got {:.6}",
            result.success_probability
        );
    }

    // ----------------------------------------------------------
    // 6. Success probability formula matches measured probability
    // ----------------------------------------------------------
    #[test]
    fn test_success_probability_formula_matches_simulation() {
        let oracle = GroverOracle::single_target(3, 6).unwrap();
        let search = GroverSearch::new(oracle.clone());

        // Check at several iteration counts
        for k in 0..6 {
            let s = GroverSearch::with_iterations(oracle.clone(), k);
            let result = s.search();
            let theoretical = search.success_probability(k);
            let measured: f64 = oracle
                .marked_states
                .iter()
                .map(|&m| result.state_probabilities[m])
                .sum();

            assert!(
                (theoretical - measured).abs() < TOL,
                "At k={}: theoretical {:.6} != measured {:.6}",
                k,
                theoretical,
                measured
            );
        }
    }

    // ----------------------------------------------------------
    // 7. Binary string representation
    // ----------------------------------------------------------
    #[test]
    fn test_binary_string_representation() {
        let oracle = GroverOracle::single_target(4, 10).unwrap();
        let search = GroverSearch::new(oracle);
        let result = search.search();

        assert_eq!(
            result.found_state_binary.len(),
            4,
            "Binary string should have length equal to num_qubits (4)"
        );
        assert_eq!(
            result.found_state_binary, "1010",
            "State 10 in binary with 4 bits should be '1010'"
        );

        // Verify consistency between found_state and found_state_binary
        let parsed = usize::from_str_radix(&result.found_state_binary, 2).unwrap();
        assert_eq!(
            parsed, result.found_state,
            "Binary string should parse back to found_state"
        );
    }

    // ----------------------------------------------------------
    // 8. From predicate (mark states divisible by 4)
    // ----------------------------------------------------------
    #[test]
    fn test_from_predicate() {
        // Use divisible-by-4 to avoid M=N/2 which makes Grover degenerate.
        // For 4 qubits (N=16), divisible by 4: {0, 4, 8, 12} -> M=4, N/M=4
        let oracle = GroverOracle::from_predicate(4, |x| x % 4 == 0).unwrap();

        assert_eq!(oracle.num_marked(), 4);
        assert!(oracle.marked_states.contains(&0));
        assert!(oracle.marked_states.contains(&4));
        assert!(oracle.marked_states.contains(&8));
        assert!(oracle.marked_states.contains(&12));
        assert!(!oracle.marked_states.contains(&1));

        // Search should find one of the marked states
        let search = GroverSearch::new(oracle.clone());
        let result = search.search();
        assert!(
            result.found_state % 4 == 0,
            "Found state {} should be divisible by 4",
            result.found_state
        );
    }

    // ----------------------------------------------------------
    // 9. Large search space (10 qubits, N=1024, single target)
    // ----------------------------------------------------------
    #[test]
    fn test_large_search_space() {
        let target = 777;
        let oracle = GroverOracle::single_target(10, target).unwrap();
        let search = GroverSearch::new(oracle);
        let result = search.search();

        assert_eq!(
            result.found_state, target,
            "Should find target {} in N=1024, got {}",
            target, result.found_state
        );
        assert!(
            result.success_probability > 0.9,
            "Success probability {:.4} should be > 0.9 for 10-qubit search",
            result.success_probability
        );
        assert_eq!(result.state_probabilities.len(), 1024);
        assert_eq!(result.num_iterations, 25, "Optimal iterations for N=1024, M=1");
    }

    // ----------------------------------------------------------
    // 10. Zero iterations (no oracle calls)
    // ----------------------------------------------------------
    #[test]
    fn test_zero_iterations() {
        let oracle = GroverOracle::single_target(3, 3).unwrap();
        let search = GroverSearch::with_iterations(oracle, 0);
        let result = search.search();

        assert_eq!(result.num_iterations, 0);
        assert_eq!(result.num_oracle_calls, 0);

        // With zero iterations, all states should have equal probability (uniform superposition)
        let expected_prob = 1.0 / 8.0;
        for (i, &p) in result.state_probabilities.iter().enumerate() {
            assert!(
                (p - expected_prob).abs() < TOL,
                "State {} probability {:.6} should be {:.6} (uniform)",
                i,
                p,
                expected_prob
            );
        }
    }

    // ----------------------------------------------------------
    // 11. Diffusion operator test (verify 2*mean - x formula)
    // ----------------------------------------------------------
    #[test]
    fn test_diffusion_operator_consistency() {
        let num_qubits = 3;
        let n = 1usize << num_qubits;

        // Create a non-uniform test state
        let mut state: Vec<C64> = (0..n)
            .map(|i| c64((i as f64 + 1.0) / 10.0, 0.0))
            .collect();
        // Normalize it
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for s in state.iter_mut() {
            *s /= c64(norm, 0.0);
        }

        // Apply diffusion via full matrix
        let d_matrix = diffusion_operator(num_qubits);
        let mut state_matrix = vec![c64_zero(); n];
        for i in 0..n {
            let mut sum = c64_zero();
            for j in 0..n {
                sum += d_matrix[i * n + j] * state[j];
            }
            state_matrix[i] = sum;
        }

        // Apply diffusion via mean formula
        let mut state_mean = state.clone();
        apply_diffusion(&mut state_mean);

        // Both methods should produce the same result
        for i in 0..n {
            assert!(
                (state_matrix[i] - state_mean[i]).norm_sqr() < TOL,
                "Diffusion mismatch at index {}: matrix={:?} vs mean={:?}",
                i,
                state_matrix[i],
                state_mean[i]
            );
        }
    }

    // ----------------------------------------------------------
    // 12. Oracle validation errors
    // ----------------------------------------------------------
    #[test]
    fn test_oracle_validation_zero_qubits() {
        let result = GroverOracle::new(0, vec![0]);
        assert_eq!(result.unwrap_err(), GroverError::ZeroQubits);
    }

    #[test]
    fn test_oracle_validation_out_of_range() {
        let result = GroverOracle::new(2, vec![5]);
        assert!(matches!(
            result.unwrap_err(),
            GroverError::MarkedStateOutOfRange { state: 5, max_state: 3 }
        ));
    }

    #[test]
    fn test_oracle_validation_no_marked() {
        let result = GroverOracle::new(2, vec![]);
        assert_eq!(result.unwrap_err(), GroverError::NoMarkedStates);
    }

    #[test]
    fn test_oracle_validation_all_marked() {
        let result = GroverOracle::new(2, vec![0, 1, 2, 3]);
        assert_eq!(result.unwrap_err(), GroverError::AllStatesMarked);
    }

    // ----------------------------------------------------------
    // 13. Uniform superposition correctness
    // ----------------------------------------------------------
    #[test]
    fn test_uniform_superposition() {
        let state = uniform_superposition(3);
        assert_eq!(state.len(), 8);

        let expected_amp = 1.0 / (8.0_f64).sqrt();
        for (i, &amp) in state.iter().enumerate() {
            assert!(
                (amp.re - expected_amp).abs() < TOL && amp.im.abs() < TOL,
                "State[{}] = ({:.6}, {:.6}), expected ({:.6}, 0.0)",
                i,
                amp.re,
                amp.im,
                expected_amp
            );
        }

        // Verify normalization
        let norm_sq: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < TOL,
            "Superposition norm^2 = {:.6}, should be 1.0",
            norm_sq
        );
    }

    // ----------------------------------------------------------
    // 14. Oracle operator structure
    // ----------------------------------------------------------
    #[test]
    fn test_oracle_operator_structure() {
        let oracle = GroverOracle::new(3, vec![2, 5]).unwrap();
        let diag = oracle_operator(&oracle);

        assert_eq!(diag.len(), 8);
        for (i, &d) in diag.iter().enumerate() {
            let expected = if i == 2 || i == 5 { -1.0 } else { 1.0 };
            assert!(
                (d.re - expected).abs() < TOL && d.im.abs() < TOL,
                "Oracle diagonal[{}] = ({:.1}, {:.1}), expected ({:.1}, 0.0)",
                i,
                d.re,
                d.im,
                expected
            );
        }
    }

    // ----------------------------------------------------------
    // 15. Probability oscillation (over-rotation)
    // ----------------------------------------------------------
    #[test]
    fn test_probability_oscillation() {
        // For N=8, M=1, the success probability oscillates as we increase iterations.
        // After optimal (2 iterations) it should peak, then decrease.
        let oracle = GroverOracle::single_target(3, 0).unwrap();
        let search = GroverSearch::new(oracle.clone());

        let prob_0 = search.success_probability(0);
        let prob_1 = search.success_probability(1);
        let prob_2 = search.success_probability(2);
        let prob_3 = search.success_probability(3);

        // Probability should increase toward the optimal point
        assert!(
            prob_1 > prob_0,
            "prob(k=1)={:.4} should > prob(k=0)={:.4}",
            prob_1,
            prob_0
        );
        assert!(
            prob_2 > prob_1,
            "prob(k=2)={:.4} should > prob(k=1)={:.4}",
            prob_2,
            prob_1
        );
        // After optimal, probability should decrease (over-rotation)
        assert!(
            prob_3 < prob_2,
            "prob(k=3)={:.4} should < prob(k=2)={:.4} (over-rotation)",
            prob_3,
            prob_2
        );
    }

    // ----------------------------------------------------------
    // 16. Multiple marked states with from_predicate
    // ----------------------------------------------------------
    #[test]
    fn test_from_predicate_multiples_of_three() {
        let oracle = GroverOracle::from_predicate(4, |x| x % 3 == 0 && x > 0).unwrap();
        // For 4 qubits (0..16), multiples of 3 excluding 0: {3, 6, 9, 12, 15}
        assert_eq!(oracle.num_marked(), 5);
        assert!(oracle.marked_states.contains(&3));
        assert!(oracle.marked_states.contains(&6));
        assert!(oracle.marked_states.contains(&9));
        assert!(oracle.marked_states.contains(&12));
        assert!(oracle.marked_states.contains(&15));

        let search = GroverSearch::new(oracle.clone());
        let result = search.search();
        assert!(
            result.found_state % 3 == 0 && result.found_state > 0,
            "Found state {} should be a positive multiple of 3",
            result.found_state
        );
    }

    // ----------------------------------------------------------
    // 17. Search space size helper
    // ----------------------------------------------------------
    #[test]
    fn test_search_space_size() {
        let oracle = GroverOracle::single_target(5, 0).unwrap();
        assert_eq!(oracle.search_space_size(), 32);
        assert_eq!(oracle.num_marked(), 1);
    }

    // ----------------------------------------------------------
    // 18. Normalization preserved across iterations
    // ----------------------------------------------------------
    #[test]
    fn test_normalization_preserved() {
        let oracle = GroverOracle::new(3, vec![1, 4, 7]).unwrap();
        let search = GroverSearch::with_iterations(oracle, 5);
        let result = search.search();

        let total_prob: f64 = result.state_probabilities.iter().sum();
        assert!(
            (total_prob - 1.0).abs() < TOL,
            "Total probability {:.10} should be 1.0 after 5 iterations",
            total_prob
        );
    }
}
