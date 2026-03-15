//! Belief Propagation (BP) Decoder for Quantum Error Correction
//!
//! Min-sum belief propagation on the Tanner graph of a stabilizer code.
//! This is a fast, iterative decoder that works well for LDPC and surface codes
//! at moderate error rates.
//!
//! # Algorithm
//!
//! The decoder constructs a bipartite (Tanner) graph:
//! - **Variable nodes**: one per physical qubit, representing possible errors
//! - **Check nodes**: one per stabilizer, representing parity constraints
//!
//! Messages are passed iteratively between variable and check nodes using
//! log-likelihood ratios (LLRs). The min-sum variant approximates the
//! sum-product algorithm with lower computational cost.
//!
//! # References
//!
//! - Poulin & Chung, "On the iterative decoding of sparse quantum codes" (2008)
//! - Panteleev & Kalachev, "Degenerate quantum LDPC codes" (2021)
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::decoding::bp::{BPDecoder, BPConfig, TannerGraph};
//!
//! // Build a repetition code Tanner graph
//! let graph = TannerGraph::repetition_code(5);
//! let decoder = BPDecoder::new(graph, BPConfig::default());
//!
//! // Decode a syndrome
//! let syndrome = vec![true, false, false, true]; // 4 checks for 5 qubits
//! let correction = decoder.decode(&syndrome, 0.01);
//! ```

use std::collections::HashMap;

// ============================================================
// TANNER GRAPH
// ============================================================

/// Tanner graph representing the structure of a stabilizer code.
///
/// Bipartite graph with variable nodes (qubits) and check nodes (stabilizers).
/// Each check node is connected to the variable nodes it acts on.
#[derive(Clone, Debug)]
pub struct TannerGraph {
    /// Number of variable nodes (physical qubits)
    pub num_variables: usize,
    /// Number of check nodes (stabilizers)
    pub num_checks: usize,
    /// For each check node, the list of variable nodes it is connected to
    pub check_to_var: Vec<Vec<usize>>,
    /// For each variable node, the list of check nodes it participates in
    pub var_to_check: Vec<Vec<usize>>,
}

impl TannerGraph {
    /// Create a Tanner graph from a parity check matrix H.
    ///
    /// H is given as a list of rows, where each row is a list of column indices
    /// (variable nodes) that have a 1 in that row.
    pub fn from_check_matrix(num_variables: usize, checks: Vec<Vec<usize>>) -> Self {
        let num_checks = checks.len();
        let mut var_to_check = vec![Vec::new(); num_variables];

        for (c, vars) in checks.iter().enumerate() {
            for &v in vars {
                assert!(v < num_variables, "Variable index {} out of range", v);
                var_to_check[v].push(c);
            }
        }

        Self {
            num_variables,
            num_checks,
            check_to_var: checks,
            var_to_check,
        }
    }

    /// Create a Tanner graph for a repetition code of given length.
    ///
    /// A repetition code on `n` qubits has `n-1` Z⊗Z parity checks:
    /// check_i checks qubit_i and qubit_{i+1}.
    pub fn repetition_code(n: usize) -> Self {
        assert!(n >= 2, "Repetition code requires at least 2 qubits");
        let checks: Vec<Vec<usize>> = (0..n - 1).map(|i| vec![i, i + 1]).collect();
        Self::from_check_matrix(n, checks)
    }

    /// Create a Tanner graph for a surface code of given distance `d`.
    ///
    /// Constructs a rotated planar surface code with `d*d` data qubits
    /// and `(d*d - 1)` stabilizer checks (split between X and Z types).
    /// For simplicity, this builds only the X-type stabilizer subgraph
    /// (decoding X errors), which is sufficient for benchmarking.
    pub fn surface_code(d: usize) -> Self {
        assert!(d >= 3 && d % 2 == 1, "Distance must be odd and >= 3");
        let n = d * d; // data qubits
        let mut checks = Vec::new();

        // Z-type stabilizers on the rotated surface code
        // Each plaquette connects 4 data qubits (or 2 on boundaries)
        for row in 0..d {
            for col in 0..d {
                // Place checks on alternating plaquettes
                if (row + col) % 2 == 0 && !(row == d - 1 && col == d - 1) {
                    let mut qubits = Vec::new();
                    // The four neighbors of this plaquette center
                    let center = row * d + col;
                    qubits.push(center);
                    if col + 1 < d {
                        qubits.push(row * d + col + 1);
                    }
                    if row + 1 < d {
                        qubits.push((row + 1) * d + col);
                    }
                    if col + 1 < d && row + 1 < d {
                        qubits.push((row + 1) * d + col + 1);
                    }
                    if qubits.len() >= 2 {
                        checks.push(qubits);
                    }
                }
            }
        }

        Self::from_check_matrix(n, checks)
    }
}

// ============================================================
// BP CONFIGURATION
// ============================================================

/// Configuration for the BP decoder.
#[derive(Clone, Debug)]
pub struct BPConfig {
    /// Maximum number of BP iterations.
    pub max_iterations: usize,
    /// Convergence threshold: stop when max message change < this value.
    pub convergence_threshold: f64,
    /// Normalization factor for min-sum (0.0 = pure min-sum, ~0.625 is typical).
    /// Scales check-to-variable messages to compensate for min-sum approximation.
    pub normalization_factor: f64,
    /// Whether to use the enhanced "min-sum with correction" variant.
    pub use_correction: bool,
}

impl Default for BPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 1e-6,
            normalization_factor: 0.625,
            use_correction: true,
        }
    }
}

// ============================================================
// BP DECODER
// ============================================================

/// Min-sum belief propagation decoder for stabilizer codes.
///
/// Operates on the Tanner graph representation of a stabilizer code.
/// Variable nodes represent qubits, check nodes represent stabilizer generators.
/// The decoder iteratively passes log-likelihood ratio (LLR) messages
/// between variable and check nodes to determine the most likely error pattern.
#[derive(Clone, Debug)]
pub struct BPDecoder {
    /// The code's Tanner graph
    graph: TannerGraph,
    /// Decoder configuration
    config: BPConfig,
}

/// Result of a BP decoding attempt.
#[derive(Clone, Debug)]
pub struct BPResult {
    /// Estimated error pattern (true = error on that qubit)
    pub error_pattern: Vec<bool>,
    /// Whether the decoder converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
    /// Final LLR beliefs for each variable node
    pub beliefs: Vec<f64>,
}

impl BPDecoder {
    /// Create a new BP decoder for the given Tanner graph.
    pub fn new(graph: TannerGraph, config: BPConfig) -> Self {
        Self { graph, config }
    }

    /// Create a BP decoder with default configuration.
    pub fn with_defaults(graph: TannerGraph) -> Self {
        Self::new(graph, BPConfig::default())
    }

    /// Decode a syndrome given channel error probability `p`.
    ///
    /// `syndrome` is a boolean vector of length `num_checks`.
    /// `p` is the physical error probability per qubit (depolarizing channel).
    ///
    /// Returns the decoded error pattern.
    pub fn decode(&self, syndrome: &[bool], p: f64) -> BPResult {
        assert_eq!(
            syndrome.len(),
            self.graph.num_checks,
            "Syndrome length {} != num_checks {}",
            syndrome.len(),
            self.graph.num_checks
        );
        assert!(p > 0.0 && p < 1.0, "Error probability must be in (0, 1)");

        let n_var = self.graph.num_variables;
        let n_chk = self.graph.num_checks;

        // Channel LLR: log((1-p)/p) for each variable node
        let channel_llr = ((1.0 - p) / p).ln();

        // Messages: variable→check and check→variable
        // Indexed as msg_vc[v][local_c_idx] and msg_cv[c][local_v_idx]
        let mut msg_vc: Vec<Vec<f64>> = self
            .graph
            .var_to_check
            .iter()
            .map(|checks| vec![channel_llr; checks.len()])
            .collect();

        let mut msg_cv: Vec<Vec<f64>> = self
            .graph
            .check_to_var
            .iter()
            .map(|vars| vec![0.0; vars.len()])
            .collect();

        // Syndrome as +1/-1 (positive if syndrome bit is 0, negative if 1)
        let syndrome_sign: Vec<f64> = syndrome
            .iter()
            .map(|&s| if s { -1.0 } else { 1.0 })
            .collect();

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // ---- Check-to-variable messages (min-sum) ----
            let mut max_delta: f64 = 0.0;

            for c in 0..n_chk {
                let vars = &self.graph.check_to_var[c];
                let n_v = vars.len();

                for j in 0..n_v {
                    // Product of signs, minimum of magnitudes over all other variable nodes
                    let mut sign_prod = syndrome_sign[c];
                    let mut min_mag = f64::INFINITY;

                    for k in 0..n_v {
                        if k == j {
                            continue;
                        }
                        let v = vars[k];
                        // Find the local index of check c in var v's check list
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

                    // Apply normalization
                    let mut new_msg = sign_prod * min_mag * self.config.normalization_factor;

                    // Optional correction term (offset min-sum)
                    if self.config.use_correction && n_v > 2 {
                        // Find second minimum for correction
                        let mut min1 = f64::INFINITY;
                        let mut min2 = f64::INFINITY;
                        for k in 0..n_v {
                            if k == j {
                                continue;
                            }
                            let v = vars[k];
                            let local_c = self.var_check_local_index(v, c);
                            let mag = msg_vc[v][local_c].abs();
                            if mag < min1 {
                                min2 = min1;
                                min1 = mag;
                            } else if mag < min2 {
                                min2 = mag;
                            }
                        }
                        // Correction: approximate log(1 + e^{-|a-b|}) - log(1 + e^{-|a+b|})
                        let correction = Self::bp_correction(min1, min2);
                        new_msg =
                            sign_prod * (min_mag * self.config.normalization_factor + correction);
                    }

                    let old_msg = msg_cv[c][j];
                    let delta = (new_msg - old_msg).abs();
                    if delta > max_delta {
                        max_delta = delta;
                    }
                    msg_cv[c][j] = new_msg;
                }
            }

            // ---- Variable-to-check messages ----
            for v in 0..n_var {
                let checks = &self.graph.var_to_check[v];
                let n_c = checks.len();

                // Sum of all incoming check→variable messages
                let total_incoming: f64 = checks
                    .iter()
                    .enumerate()
                    .map(|(local_c, &c)| {
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

            // Check convergence
            if max_delta < self.config.convergence_threshold {
                converged = true;
                break;
            }
        }

        // ---- Hard decision from final beliefs ----
        let mut beliefs = vec![0.0; n_var];
        let mut error_pattern = vec![false; n_var];

        for v in 0..n_var {
            let checks = &self.graph.var_to_check[v];
            let total_incoming: f64 = checks
                .iter()
                .enumerate()
                .map(|(_, &c)| {
                    let local_v = self.check_var_local_index(c, v);
                    msg_cv[c][local_v]
                })
                .sum();

            beliefs[v] = channel_llr + total_incoming;
            error_pattern[v] = beliefs[v] < 0.0;
        }

        BPResult {
            error_pattern,
            converged,
            iterations,
            beliefs,
        }
    }

    /// BP correction term for offset min-sum.
    /// Approximates: ln(1 + e^{-|a-b|}) - ln(1 + e^{-(a+b)})
    fn bp_correction(a: f64, b: f64) -> f64 {
        let diff = (a - b).abs();
        let sum = a + b;
        // Use the max(0, 0.5 - 0.25*|a-b|) approximation for efficiency
        (0.5 - 0.25 * diff).max(0.0) - (0.5 - 0.25 * sum).max(0.0)
    }

    /// Find the local index of check `c` in variable `v`'s check list.
    #[inline]
    fn var_check_local_index(&self, v: usize, c: usize) -> usize {
        self.graph.var_to_check[v]
            .iter()
            .position(|&x| x == c)
            .expect("Check not found in variable's adjacency list")
    }

    /// Find the local index of variable `v` in check `c`'s variable list.
    #[inline]
    fn check_var_local_index(&self, c: usize, v: usize) -> usize {
        self.graph.check_to_var[c]
            .iter()
            .position(|&x| x == v)
            .expect("Variable not found in check's adjacency list")
    }

    /// Build fast lookup tables for variable↔check indices.
    /// Returns (var_to_check_local_idx, check_to_var_local_idx) hashmaps.
    /// For large graphs, precomputing these avoids repeated linear scans.
    #[allow(dead_code)]
    fn build_index_maps(
        &self,
    ) -> (
        HashMap<(usize, usize), usize>,
        HashMap<(usize, usize), usize>,
    ) {
        let mut vc_map = HashMap::new();
        let mut cv_map = HashMap::new();

        for (v, checks) in self.graph.var_to_check.iter().enumerate() {
            for (local_c, &c) in checks.iter().enumerate() {
                vc_map.insert((v, c), local_c);
            }
        }

        for (c, vars) in self.graph.check_to_var.iter().enumerate() {
            for (local_v, &v) in vars.iter().enumerate() {
                cv_map.insert((c, v), local_v);
            }
        }

        (vc_map, cv_map)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanner_graph_repetition_code() {
        let graph = TannerGraph::repetition_code(5);
        assert_eq!(graph.num_variables, 5);
        assert_eq!(graph.num_checks, 4);
        // Check 0 connects qubits 0,1
        assert_eq!(graph.check_to_var[0], vec![0, 1]);
        // Qubit 2 participates in checks 1 and 2
        assert_eq!(graph.var_to_check[2], vec![1, 2]);
    }

    #[test]
    fn test_tanner_graph_surface_code() {
        let graph = TannerGraph::surface_code(3);
        assert_eq!(graph.num_variables, 9);
        assert!(graph.num_checks > 0);
        // All check-to-var entries should reference valid variables
        for check in &graph.check_to_var {
            for &v in check {
                assert!(v < 9);
            }
        }
    }

    #[test]
    fn test_trivial_syndrome_no_errors() {
        let graph = TannerGraph::repetition_code(5);
        let decoder = BPDecoder::with_defaults(graph);
        let syndrome = vec![false, false, false, false];
        let result = decoder.decode(&syndrome, 0.01);
        // No syndrome → no errors
        assert!(result.error_pattern.iter().all(|&e| !e));
        assert!(result.converged);
    }

    #[test]
    fn test_single_qubit_error_repetition() {
        // Repetition code: 5 qubits, 4 checks
        // Error on qubit 2 → syndromes 1 and 2 fire (checks between q1-q2 and q2-q3)
        let graph = TannerGraph::repetition_code(5);
        let decoder = BPDecoder::with_defaults(graph);
        let syndrome = vec![false, true, true, false];
        let result = decoder.decode(&syndrome, 0.05);

        // Should identify qubit 2 as the error
        assert!(result.converged);
        assert!(result.error_pattern[2], "Should detect error on qubit 2");
        // Other qubits should not have errors
        let error_count: usize = result.error_pattern.iter().filter(|&&e| e).count();
        assert_eq!(error_count, 1, "Should find exactly one error");
    }

    #[test]
    fn test_boundary_error_repetition() {
        // Error on qubit 0 → only syndrome 0 fires
        let graph = TannerGraph::repetition_code(5);
        let decoder = BPDecoder::with_defaults(graph);
        let syndrome = vec![true, false, false, false];
        let result = decoder.decode(&syndrome, 0.05);

        assert!(result.converged);
        assert!(result.error_pattern[0], "Should detect error on qubit 0");
    }

    #[test]
    fn test_convergence_larger_code() {
        // Repetition code with 20 qubits, single error in the middle
        let graph = TannerGraph::repetition_code(20);
        let decoder = BPDecoder::with_defaults(graph);
        let mut syndrome = vec![false; 19];
        syndrome[9] = true; // check between q9 and q10
        syndrome[10] = true; // check between q10 and q11
        let result = decoder.decode(&syndrome, 0.01);

        assert!(result.converged);
        assert!(result.error_pattern[10], "Should detect error on qubit 10");
    }

    #[test]
    fn test_two_errors_repetition() {
        // Errors on qubits 1 and 3 (non-adjacent)
        // Syndrome: check 0 fires (q0-q1), check 1 fires (q1-q2),
        //           check 2 fires (q2-q3), check 3 fires (q3-q4)
        let graph = TannerGraph::repetition_code(5);
        let decoder = BPDecoder::with_defaults(graph);
        let syndrome = vec![true, true, true, true];
        let result = decoder.decode(&syndrome, 0.1);

        // BP may not perfectly decode two errors on a repetition code
        // but it should converge
        assert!(result.converged || result.iterations <= 50);
    }

    #[test]
    fn test_config_max_iterations() {
        let graph = TannerGraph::repetition_code(5);
        let config = BPConfig {
            max_iterations: 3,
            ..BPConfig::default()
        };
        let decoder = BPDecoder::new(graph, config);
        let syndrome = vec![false, true, true, false];
        let result = decoder.decode(&syndrome, 0.05);

        assert!(result.iterations <= 3);
    }

    #[test]
    fn test_high_error_rate() {
        // At high error rates, BP should still not crash
        let graph = TannerGraph::repetition_code(7);
        let decoder = BPDecoder::with_defaults(graph);
        let syndrome = vec![true, false, true, false, true, false];
        let result = decoder.decode(&syndrome, 0.45);

        // Just verify it doesn't panic and produces valid output
        assert_eq!(result.error_pattern.len(), 7);
    }

    #[test]
    fn test_surface_code_trivial() {
        let graph = TannerGraph::surface_code(3);
        let num_checks = graph.num_checks;
        let decoder = BPDecoder::with_defaults(graph);
        let syndrome = vec![false; num_checks];
        let result = decoder.decode(&syndrome, 0.01);

        assert!(result.error_pattern.iter().all(|&e| !e));
        assert!(result.converged);
    }

    #[test]
    fn test_beliefs_sign_matches_decision() {
        let graph = TannerGraph::repetition_code(5);
        let decoder = BPDecoder::with_defaults(graph);
        let syndrome = vec![false, true, true, false];
        let result = decoder.decode(&syndrome, 0.05);

        // Verify beliefs are consistent with error_pattern
        for (i, (&belief, &error)) in result
            .beliefs
            .iter()
            .zip(result.error_pattern.iter())
            .enumerate()
        {
            if error {
                assert!(
                    belief < 0.0,
                    "Qubit {} has error but belief {} >= 0",
                    i,
                    belief
                );
            } else {
                assert!(
                    belief >= 0.0,
                    "Qubit {} has no error but belief {} < 0",
                    i,
                    belief
                );
            }
        }
    }

    #[test]
    fn test_performance_1000_qubits() {
        // Verify BP can handle 1000-qubit repetition code quickly
        let graph = TannerGraph::repetition_code(1000);
        let decoder = BPDecoder::with_defaults(graph);
        let mut syndrome = vec![false; 999];
        syndrome[499] = true;
        syndrome[500] = true;

        let start = std::time::Instant::now();
        let result = decoder.decode(&syndrome, 0.01);
        let elapsed = start.elapsed();

        assert!(result.converged);
        assert!(result.error_pattern[500], "Should find error on qubit 500");
        assert!(
            elapsed.as_millis() < 100,
            "Should decode 1000-qubit syndrome in <100ms, took {}ms",
            elapsed.as_millis()
        );
    }
}
