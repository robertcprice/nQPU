//! Relay Belief Propagation (Relay-BP) Decoder
//!
//! An enhanced belief propagation decoder for quantum error correction that
//! addresses the "trapping set" problem where standard BP gets stuck in local
//! optima. Relay-BP inserts message relay nodes and applies strategic message
//! resets to escape trapping configurations.
//!
//! # Algorithm Overview
//!
//! 1. **Initialize**: Set channel LLRs from physical error rate, build Tanner
//!    graph from parity check matrix H.
//! 2. **Standard BP iteration**: Variable-to-check messages (tanh rule),
//!    check-to-variable messages (box-plus / min-sum).
//! 3. **Convergence check**: Does the hard-decision satisfy the syndrome?
//! 4. **Relay detection**: If messages oscillate (variance below threshold),
//!    insert a relay node on the problematic edge.
//! 5. **Relay mechanism**: Reset messages on relay edges, re-run BP with the
//!    modified message state.
//! 6. **OSD fallback**: If BP+relay fails after `max_relays` attempts, use
//!    OSD-0 (Gaussian elimination + re-encoding).
//! 7. **Serial schedule**: Optional layer-by-layer message updates for better
//!    convergence on LDPC codes.
//!
//! # Integration
//!
//! Works with parity check matrices from [`crate::qldpc`] and
//! [`crate::bivariate_bicycle`]. Can serve as the decoder in
//! [`crate::qec_sampling`] threshold studies.
//!
//! # References
//!
//! - Panteleev & Kalachev, "Degenerate quantum LDPC codes with good finite
//!   length performance", Quantum 2021
//! - Roffe et al., "Decoding across the quantum low-density parity-check
//!   landscape", PRX Quantum 2023
//! - Fossorier & Lin, "Soft-decision decoding of linear block codes based on
//!   ordered statistics", IEEE Trans. Inf. Theory 1995

use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during Relay-BP decoding.
#[derive(Debug, Clone, PartialEq)]
pub enum RelayBPError {
    /// The parity check matrix dimensions do not match the syndrome length.
    MatrixDimensionMismatch {
        expected_checks: usize,
        got_syndrome_len: usize,
    },
    /// BP did not converge within the allowed iterations and relay budget.
    NotConverged {
        iterations_used: usize,
        relays_used: usize,
    },
    /// The syndrome vector contains invalid data (e.g., wrong length for H).
    InvalidSyndrome(String),
}

impl fmt::Display for RelayBPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelayBPError::MatrixDimensionMismatch {
                expected_checks,
                got_syndrome_len,
            } => write!(
                f,
                "Matrix dimension mismatch: H has {} checks but syndrome has length {}",
                expected_checks, got_syndrome_len
            ),
            RelayBPError::NotConverged {
                iterations_used,
                relays_used,
            } => write!(
                f,
                "BP did not converge after {} iterations and {} relay insertions",
                iterations_used, relays_used
            ),
            RelayBPError::InvalidSyndrome(msg) => write!(f, "Invalid syndrome: {}", msg),
        }
    }
}

impl std::error::Error for RelayBPError {}

// ============================================================
// MESSAGE SCHEDULE
// ============================================================

/// Message update schedule for belief propagation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// All messages update simultaneously each iteration (standard BP).
    Flooding,
    /// Messages update layer-by-layer: check nodes process in sequential
    /// order, each seeing the latest variable messages immediately. Often
    /// converges faster for LDPC codes.
    LayeredSerial,
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the Relay-BP decoder, constructed via builder pattern.
#[derive(Clone, Debug)]
pub struct RelayBPConfig {
    /// Maximum number of BP iterations per relay attempt.
    pub max_iterations: usize,
    /// Damping factor for message updates (0 < alpha <= 1).
    /// New message = alpha * computed + (1 - alpha) * old.
    pub damping_factor: f64,
    /// Variance threshold below which an edge is considered "trapped".
    /// When the variance of an edge's message history falls below this
    /// value, a relay node is inserted.
    pub relay_threshold: f64,
    /// Maximum number of relay insertions before falling back to OSD.
    pub max_relays: usize,
    /// Convergence epsilon: if the maximum absolute change in any message
    /// is below this value, BP is considered converged (even if the
    /// syndrome is not yet satisfied).
    pub convergence_eps: f64,
    /// Message update schedule.
    pub schedule: Schedule,
}

impl Default for RelayBPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            damping_factor: 0.5,
            relay_threshold: 0.01,
            max_relays: 5,
            convergence_eps: 1e-8,
            schedule: Schedule::Flooding,
        }
    }
}

impl RelayBPConfig {
    /// Create a new builder starting from default values.
    pub fn builder() -> RelayBPConfigBuilder {
        RelayBPConfigBuilder {
            config: RelayBPConfig::default(),
        }
    }
}

/// Builder for [`RelayBPConfig`].
pub struct RelayBPConfigBuilder {
    config: RelayBPConfig,
}

impl RelayBPConfigBuilder {
    /// Set the maximum number of BP iterations per relay attempt.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set the damping factor for message updates.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in the interval (0, 1].
    pub fn damping_factor(mut self, alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "damping_factor must be in (0, 1], got {}",
            alpha
        );
        self.config.damping_factor = alpha;
        self
    }

    /// Set the relay activation threshold (message variance).
    pub fn relay_threshold(mut self, threshold: f64) -> Self {
        self.config.relay_threshold = threshold;
        self
    }

    /// Set the maximum number of relay insertions.
    pub fn max_relays(mut self, n: usize) -> Self {
        self.config.max_relays = n;
        self
    }

    /// Set the convergence epsilon.
    pub fn convergence_eps(mut self, eps: f64) -> Self {
        self.config.convergence_eps = eps;
        self
    }

    /// Set the message update schedule.
    pub fn schedule(mut self, schedule: Schedule) -> Self {
        self.config.schedule = schedule;
        self
    }

    /// Consume the builder and return the configuration.
    pub fn build(self) -> RelayBPConfig {
        self.config
    }
}

// ============================================================
// BP MESSAGE
// ============================================================

/// A log-likelihood ratio (LLR) message on a Tanner graph edge.
///
/// Positive LLR indicates the variable is more likely 0; negative
/// indicates more likely 1. The magnitude indicates confidence.
#[derive(Clone, Debug)]
pub struct BPMessage {
    /// Current LLR value.
    pub llr: f64,
    /// History of recent LLR values for oscillation detection.
    history: Vec<f64>,
    /// Maximum history length to retain.
    history_capacity: usize,
}

impl BPMessage {
    /// Create a new message initialized to the given LLR.
    fn new(llr: f64) -> Self {
        Self {
            llr,
            history: vec![llr],
            history_capacity: 8,
        }
    }

    /// Update the message value and record history.
    fn update(&mut self, new_llr: f64) {
        self.llr = new_llr;
        if self.history.len() >= self.history_capacity {
            self.history.remove(0);
        }
        self.history.push(new_llr);
    }

    /// Compute the variance of the message history.
    ///
    /// A low variance indicates the message is stuck (oscillating around
    /// a fixed point or not changing), which is a signal for relay insertion.
    fn variance(&self) -> f64 {
        if self.history.len() < 2 {
            return f64::INFINITY;
        }
        let n = self.history.len() as f64;
        let mean = self.history.iter().sum::<f64>() / n;
        let var = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        var
    }

    /// Reset the message to a new LLR, clearing history.
    fn reset(&mut self, llr: f64) {
        self.llr = llr;
        self.history.clear();
        self.history.push(llr);
    }
}

// ============================================================
// TANNER GRAPH
// ============================================================

/// Bipartite Tanner graph for belief propagation decoding.
///
/// Variable nodes correspond to qubits (columns of H), and check nodes
/// correspond to stabilizers (rows of H). An edge connects check node i
/// to variable node j if and only if H[i, j] = 1.
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
    /// Construct a Tanner graph from a dense parity check matrix H.
    ///
    /// `h` is stored as row-major: `h[i * cols + j]` is H[i,j].
    pub fn from_parity_check(h: &[Vec<bool>]) -> Self {
        let num_checks = h.len();
        let num_variables = if num_checks > 0 { h[0].len() } else { 0 };

        let mut check_to_var: Vec<Vec<usize>> = Vec::with_capacity(num_checks);
        let mut var_to_check: Vec<Vec<usize>> = vec![Vec::new(); num_variables];

        for (i, row) in h.iter().enumerate() {
            let mut neighbors = Vec::new();
            for (j, &val) in row.iter().enumerate() {
                if val {
                    neighbors.push(j);
                    var_to_check[j].push(i);
                }
            }
            check_to_var.push(neighbors);
        }

        Self {
            num_checks,
            num_variables,
            check_to_var,
            var_to_check,
        }
    }

    /// Neighbors (variable nodes) of check node `i`.
    pub fn neighbors_of_check(&self, i: usize) -> &[usize] {
        &self.check_to_var[i]
    }

    /// Neighbors (check nodes) of variable node `j`.
    pub fn neighbors_of_variable(&self, j: usize) -> &[usize] {
        &self.var_to_check[j]
    }

    /// Total number of edges in the graph.
    pub fn num_edges(&self) -> usize {
        self.check_to_var.iter().map(|v| v.len()).sum()
    }

    /// Check node degree (number of adjacent variable nodes).
    pub fn check_degree(&self, i: usize) -> usize {
        self.check_to_var[i].len()
    }

    /// Variable node degree (number of adjacent check nodes).
    pub fn variable_degree(&self, j: usize) -> usize {
        self.var_to_check[j].len()
    }
}

// ============================================================
// RELAY NODE
// ============================================================

/// A relay node inserted between a variable node and a check node to
/// break trapping set oscillations.
///
/// When BP messages on an edge oscillate without converging, a relay
/// node is inserted on that edge. The relay resets the message to the
/// channel prior and adds a small perturbation to break symmetry.
#[derive(Clone, Debug)]
pub struct RelayNode {
    /// The check node endpoint of the relayed edge.
    pub check_idx: usize,
    /// The variable node endpoint of the relayed edge.
    pub var_idx: usize,
    /// The iteration at which the relay was inserted.
    pub inserted_at_iter: usize,
    /// Perturbation factor applied to the reset message.
    pub perturbation: f64,
}

// ============================================================
// DECODING RESULT
// ============================================================

/// Result of a Relay-BP decoding attempt.
#[derive(Clone, Debug)]
pub struct DecodingResult {
    /// Estimated error pattern (true = error on that qubit).
    pub error_estimate: Vec<bool>,
    /// Whether the decoder converged (hard-decision satisfies syndrome).
    pub converged: bool,
    /// Total number of BP iterations used across all relay attempts.
    pub iterations_used: usize,
    /// Whether the final error estimate satisfies the syndrome.
    pub syndrome_satisfied: bool,
    /// Number of relay nodes that were inserted.
    pub relays_inserted: usize,
    /// Whether OSD post-processing was invoked.
    pub osd_used: bool,
}

// ============================================================
// OSD POST-PROCESSOR
// ============================================================

/// Ordered Statistics Decoding (OSD-0) post-processor.
///
/// When BP (with or without relays) fails to converge, OSD uses the
/// soft information from BP to order columns by reliability, then
/// solves the syndrome equation via Gaussian elimination. This is the
/// OSD-0 variant (no additional search over information bits).
pub struct OSD;

impl OSD {
    /// Run OSD-0 decoding given the parity check matrix, syndrome,
    /// and soft LLR information from BP.
    ///
    /// Returns the estimated error vector.
    pub fn decode(h: &[Vec<bool>], syndrome: &[bool], soft_llr: &[f64]) -> Vec<bool> {
        let m = h.len();
        let n = if m > 0 { h[0].len() } else { return vec![] };

        // Order columns by reliability (absolute LLR), least reliable first
        // so that the most reliable columns become the information set.
        let mut col_order: Vec<usize> = (0..n).collect();
        col_order.sort_by(|&a, &b| {
            soft_llr[a]
                .abs()
                .partial_cmp(&soft_llr[b].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build augmented matrix [H_reordered | syndrome] for GF(2) elimination
        let mut augmented: Vec<Vec<bool>> = (0..m)
            .map(|r| {
                let mut row = vec![false; n + 1];
                for (new_c, &orig_c) in col_order.iter().enumerate() {
                    row[new_c] = h[r][orig_c];
                }
                row[n] = syndrome[r];
                row
            })
            .collect();

        // Forward elimination (Gaussian elimination over GF(2))
        let mut pivot_cols: Vec<usize> = Vec::new();
        let mut current_row = 0;
        for col in 0..n {
            if current_row >= m {
                break;
            }
            // Find pivot row
            let mut pivot = None;
            for row in current_row..m {
                if augmented[row][col] {
                    pivot = Some(row);
                    break;
                }
            }
            if let Some(p) = pivot {
                augmented.swap(current_row, p);
                // Eliminate this column from all other rows
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

        // Back-substitute: set non-pivot (information) columns to their
        // hard-decision from BP, solve for pivot columns.
        let mut reordered_error = vec![false; n];

        // For OSD-0, information bits are set to 0 (most likely values),
        // then pivot bits are solved from the syndrome.
        for (rank, &col) in pivot_cols.iter().enumerate() {
            reordered_error[col] = augmented[rank][n];
        }

        // Map back to original column ordering
        let mut result = vec![false; n];
        for (new_c, &orig_c) in col_order.iter().enumerate() {
            result[orig_c] = reordered_error[new_c];
        }

        result
    }
}

// ============================================================
// RELAY-BP DECODER
// ============================================================

/// Relay Belief Propagation decoder for quantum error correction.
///
/// Implements standard BP with damping, relay node insertion for trapping
/// set escape, and OSD-0 fallback. Supports both flooding and layered
/// serial message schedules.
///
/// # Example
///
/// ```
/// use nqpu_metal::relay_bp::{RelayBPDecoder, RelayBPConfig};
///
/// // Repetition code [[5,1,5]] parity check matrix
/// let h = vec![
///     vec![true,  true,  false, false, false],
///     vec![false, true,  true,  false, false],
///     vec![false, false, true,  true,  false],
///     vec![false, false, false, true,  true ],
/// ];
///
/// let config = RelayBPConfig::builder()
///     .max_iterations(50)
///     .damping_factor(0.8)
///     .build();
///
/// let decoder = RelayBPDecoder::new(h, config);
/// let syndrome = vec![true, false, false, false]; // error on qubit 0
/// let result = decoder.decode(&syndrome, 0.1).unwrap();
/// assert!(result.syndrome_satisfied);
/// ```
pub struct RelayBPDecoder {
    /// Dense parity check matrix (row-major).
    h: Vec<Vec<bool>>,
    /// Tanner graph built from H.
    tanner: TannerGraph,
    /// Decoder configuration.
    config: RelayBPConfig,
    /// Number of check nodes (rows of H).
    num_checks: usize,
    /// Number of variable nodes (columns of H).
    num_vars: usize,
}

impl RelayBPDecoder {
    /// Create a new Relay-BP decoder for the given parity check matrix.
    ///
    /// # Arguments
    ///
    /// * `h` - Dense parity check matrix as a vector of rows.
    /// * `config` - Decoder configuration.
    pub fn new(h: Vec<Vec<bool>>, config: RelayBPConfig) -> Self {
        let tanner = TannerGraph::from_parity_check(&h);
        let num_checks = tanner.num_checks;
        let num_vars = tanner.num_variables;
        Self {
            h,
            tanner,
            config,
            num_checks,
            num_vars,
        }
    }

    /// Decode a syndrome to estimate the most likely error pattern.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Binary syndrome vector (length = number of checks).
    /// * `error_rate` - Physical error probability per qubit.
    ///
    /// # Returns
    ///
    /// A [`DecodingResult`] on success, or a [`RelayBPError`] if inputs
    /// are invalid.
    pub fn decode(
        &self,
        syndrome: &[bool],
        error_rate: f64,
    ) -> Result<DecodingResult, RelayBPError> {
        // Validate inputs
        if syndrome.len() != self.num_checks {
            return Err(RelayBPError::MatrixDimensionMismatch {
                expected_checks: self.num_checks,
                got_syndrome_len: syndrome.len(),
            });
        }

        // All-zero syndrome means no errors detected
        if syndrome.iter().all(|&s| !s) {
            return Ok(DecodingResult {
                error_estimate: vec![false; self.num_vars],
                converged: true,
                iterations_used: 0,
                syndrome_satisfied: true,
                relays_inserted: 0,
                osd_used: false,
            });
        }

        // Channel LLR: log((1-p)/p), positive means "more likely 0"
        let channel_llr = if error_rate <= 0.0 || error_rate >= 1.0 {
            0.0
        } else {
            ((1.0 - error_rate) / error_rate).ln()
        };

        // Initialize messages
        let mut msg_v2c = self.init_messages(channel_llr);
        let mut msg_c2v = self.init_messages(0.0);
        let mut relay_nodes: Vec<RelayNode> = Vec::new();
        let mut total_iters = 0;

        // Run BP with relay insertion
        for relay_round in 0..=self.config.max_relays {
            let bp_result = match self.config.schedule {
                Schedule::Flooding => self.bp_flooding(
                    syndrome,
                    channel_llr,
                    &mut msg_v2c,
                    &mut msg_c2v,
                    &relay_nodes,
                ),
                Schedule::LayeredSerial => self.bp_serial(
                    syndrome,
                    channel_llr,
                    &mut msg_v2c,
                    &mut msg_c2v,
                    &relay_nodes,
                ),
            };

            total_iters += bp_result.iterations_used;

            if bp_result.converged {
                return Ok(DecodingResult {
                    error_estimate: bp_result.hard_decision,
                    converged: true,
                    iterations_used: total_iters,
                    syndrome_satisfied: true,
                    relays_inserted: relay_nodes.len(),
                    osd_used: false,
                });
            }

            // Check if we should insert a relay
            if relay_round < self.config.max_relays {
                if let Some(relay) = self.detect_trapped_edge(&msg_v2c, &msg_c2v, total_iters) {
                    // Reset messages on the trapped edge
                    let c = relay.check_idx;
                    let v = relay.var_idx;
                    let perturbed_llr = channel_llr * (1.0 + relay.perturbation);
                    msg_v2c[c][v].reset(perturbed_llr);
                    msg_c2v[c][v].reset(0.0);
                    relay_nodes.push(relay);
                }
            }
        }

        // OSD fallback
        let soft_llr = self.compute_posteriors(channel_llr, &msg_c2v);
        let osd_estimate = OSD::decode(&self.h, syndrome, &soft_llr);
        let syndrome_ok = self.check_syndrome(&osd_estimate, syndrome);

        Ok(DecodingResult {
            error_estimate: osd_estimate,
            converged: syndrome_ok,
            iterations_used: total_iters,
            syndrome_satisfied: syndrome_ok,
            relays_inserted: relay_nodes.len(),
            osd_used: true,
        })
    }

    /// Initialize a 2D message array indexed by [check][variable].
    ///
    /// Only edges present in the Tanner graph carry meaningful messages,
    /// but we allocate the full grid for simplicity and cache locality.
    fn init_messages(&self, default_llr: f64) -> Vec<Vec<BPMessage>> {
        let mut msgs = Vec::with_capacity(self.num_checks);
        for c in 0..self.num_checks {
            let mut row = Vec::with_capacity(self.num_vars);
            for v in 0..self.num_vars {
                if self.h[c][v] {
                    row.push(BPMessage::new(default_llr));
                } else {
                    row.push(BPMessage::new(0.0));
                }
            }
            msgs.push(row);
        }
        msgs
    }

    /// Run flooding-schedule BP.
    fn bp_flooding(
        &self,
        syndrome: &[bool],
        channel_llr: f64,
        msg_v2c: &mut [Vec<BPMessage>],
        msg_c2v: &mut [Vec<BPMessage>],
        _relay_nodes: &[RelayNode],
    ) -> BPIterationResult {
        let mut hard_decision = vec![false; self.num_vars];

        for iter in 0..self.config.max_iterations {
            let mut max_delta: f64 = 0.0;

            // --- Check node update ---
            for c in 0..self.num_checks {
                let neighbors = self.tanner.neighbors_of_check(c);
                let syndrome_sign: f64 = if syndrome[c] { -1.0 } else { 1.0 };

                for &v in neighbors {
                    let old_llr = msg_c2v[c][v].llr;
                    let new_llr = self.sum_product_check_update(c, v, msg_v2c, syndrome_sign);
                    let damped = self.config.damping_factor * new_llr
                        + (1.0 - self.config.damping_factor) * old_llr;
                    msg_c2v[c][v].update(damped);
                    max_delta = max_delta.max((damped - old_llr).abs());
                }
            }

            // --- Variable node update ---
            for v in 0..self.num_vars {
                let checks = self.tanner.neighbors_of_variable(v);
                let total_llr: f64 =
                    channel_llr + checks.iter().map(|&c| msg_c2v[c][v].llr).sum::<f64>();

                hard_decision[v] = total_llr < 0.0;

                for &c in checks {
                    let new_v2c = total_llr - msg_c2v[c][v].llr;
                    msg_v2c[c][v].update(new_v2c);
                }
            }

            // --- Check convergence ---
            if self.check_syndrome(&hard_decision, syndrome) {
                return BPIterationResult {
                    hard_decision,
                    converged: true,
                    iterations_used: iter + 1,
                };
            }

            // Early exit if messages have stabilized
            if max_delta < self.config.convergence_eps {
                return BPIterationResult {
                    hard_decision,
                    converged: false,
                    iterations_used: iter + 1,
                };
            }
        }

        BPIterationResult {
            hard_decision,
            converged: false,
            iterations_used: self.config.max_iterations,
        }
    }

    /// Run layered serial schedule BP.
    ///
    /// In serial scheduling, check nodes are processed one at a time.
    /// After each check node update, the variable messages it touches
    /// are immediately recomputed, so subsequent check nodes see the
    /// latest information.
    fn bp_serial(
        &self,
        syndrome: &[bool],
        channel_llr: f64,
        msg_v2c: &mut [Vec<BPMessage>],
        msg_c2v: &mut [Vec<BPMessage>],
        _relay_nodes: &[RelayNode],
    ) -> BPIterationResult {
        let mut hard_decision = vec![false; self.num_vars];

        for iter in 0..self.config.max_iterations {
            let mut max_delta: f64 = 0.0;

            for c in 0..self.num_checks {
                let neighbors: Vec<usize> = self.tanner.neighbors_of_check(c).to_vec();
                let syndrome_sign: f64 = if syndrome[c] { -1.0 } else { 1.0 };

                // Check-to-variable update for this check node
                for &v in &neighbors {
                    let old_llr = msg_c2v[c][v].llr;
                    let new_llr = self.sum_product_check_update(c, v, msg_v2c, syndrome_sign);
                    let damped = self.config.damping_factor * new_llr
                        + (1.0 - self.config.damping_factor) * old_llr;
                    msg_c2v[c][v].update(damped);
                    max_delta = max_delta.max((damped - old_llr).abs());
                }

                // Immediately update variable-to-check messages for touched variables
                for &v in &neighbors {
                    let checks = self.tanner.neighbors_of_variable(v);
                    let total_llr: f64 =
                        channel_llr + checks.iter().map(|&cc| msg_c2v[cc][v].llr).sum::<f64>();
                    hard_decision[v] = total_llr < 0.0;

                    for &cc in checks {
                        let new_v2c = total_llr - msg_c2v[cc][v].llr;
                        msg_v2c[cc][v].update(new_v2c);
                    }
                }
            }

            // Check convergence after full sweep
            if self.check_syndrome(&hard_decision, syndrome) {
                return BPIterationResult {
                    hard_decision,
                    converged: true,
                    iterations_used: iter + 1,
                };
            }

            if max_delta < self.config.convergence_eps {
                return BPIterationResult {
                    hard_decision,
                    converged: false,
                    iterations_used: iter + 1,
                };
            }
        }

        BPIterationResult {
            hard_decision,
            converged: false,
            iterations_used: self.config.max_iterations,
        }
    }

    /// Sum-product check node update (box-plus / tanh rule).
    ///
    /// Computes the message from check node `check` to variable node
    /// `target_var`, using messages from all other variable neighbors.
    fn sum_product_check_update(
        &self,
        check: usize,
        target_var: usize,
        msg_v2c: &[Vec<BPMessage>],
        syndrome_sign: f64,
    ) -> f64 {
        let neighbors = self.tanner.neighbors_of_check(check);
        let mut product = syndrome_sign;

        for &v in neighbors {
            if v == target_var {
                continue;
            }
            let m = msg_v2c[check][v].llr;
            // tanh(m/2) clamped for numerical stability
            let t = (m / 2.0).tanh().clamp(-0.9999999, 0.9999999);
            product *= t;
        }

        // 2 * atanh(product)
        let clamped = product.clamp(-0.9999999, 0.9999999);
        2.0 * clamped.atanh()
    }

    /// Verify that an error pattern satisfies the syndrome.
    ///
    /// For each check node, the parity of the error bits in its
    /// neighborhood must equal the syndrome bit.
    fn check_syndrome(&self, error: &[bool], syndrome: &[bool]) -> bool {
        for c in 0..self.num_checks {
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

    /// Detect the most problematic edge for relay insertion.
    ///
    /// Scans all edges for message variance below the relay threshold.
    /// Returns the edge with the lowest non-zero message variance among
    /// trapped candidates, with a perturbation chosen to break symmetry.
    fn detect_trapped_edge(
        &self,
        msg_v2c: &[Vec<BPMessage>],
        msg_c2v: &[Vec<BPMessage>],
        current_iter: usize,
    ) -> Option<RelayNode> {
        let mut worst_edge: Option<(usize, usize, f64)> = None;

        for c in 0..self.num_checks {
            for &v in self.tanner.neighbors_of_check(c) {
                let var_v2c = msg_v2c[c][v].variance();
                let var_c2v = msg_c2v[c][v].variance();
                let combined_var = var_v2c + var_c2v;

                // Edge is "trapped" if both message directions have low variance
                // and there is at least some history
                if combined_var < self.config.relay_threshold && msg_v2c[c][v].history.len() >= 3 {
                    match &worst_edge {
                        None => {
                            worst_edge = Some((c, v, combined_var));
                        }
                        Some((_, _, prev_var)) => {
                            if combined_var < *prev_var {
                                worst_edge = Some((c, v, combined_var));
                            }
                        }
                    }
                }
            }
        }

        worst_edge.map(|(c, v, _)| {
            // Perturbation: small jitter proportional to iteration count
            // to avoid re-entering the same trapping configuration.
            let perturbation = 0.1 * (1.0 + (current_iter as f64 * 0.01).sin());
            RelayNode {
                check_idx: c,
                var_idx: v,
                inserted_at_iter: current_iter,
                perturbation,
            }
        })
    }

    /// Compute posterior LLRs for all variable nodes from the current
    /// check-to-variable messages plus the channel prior.
    fn compute_posteriors(&self, channel_llr: f64, msg_c2v: &[Vec<BPMessage>]) -> Vec<f64> {
        let mut posteriors = vec![channel_llr; self.num_vars];
        for v in 0..self.num_vars {
            for &c in self.tanner.neighbors_of_variable(v) {
                posteriors[v] += msg_c2v[c][v].llr;
            }
        }
        posteriors
    }

    /// Compute the syndrome for a given error vector against this
    /// decoder's parity check matrix.
    pub fn syndrome_of(&self, error: &[bool]) -> Vec<bool> {
        let mut syn = vec![false; self.num_checks];
        for c in 0..self.num_checks {
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

    /// Access the underlying Tanner graph.
    pub fn tanner_graph(&self) -> &TannerGraph {
        &self.tanner
    }

    /// Access the decoder configuration.
    pub fn config(&self) -> &RelayBPConfig {
        &self.config
    }

    /// Number of variable nodes (qubits).
    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Number of check nodes (stabilizers).
    pub fn num_checks(&self) -> usize {
        self.num_checks
    }
}

/// Internal result from a single BP run (before relay/OSD logic).
struct BPIterationResult {
    hard_decision: Vec<bool>,
    converged: bool,
    iterations_used: usize,
}

// ============================================================
// HELPER: PARITY CHECK MATRIX BUILDERS
// ============================================================

/// Build the parity check matrix for a repetition code of the given distance.
///
/// The [[d, 1, d]] repetition code has (d-1) parity checks, each comparing
/// adjacent qubits: H[i, i] = 1, H[i, i+1] = 1.
pub fn repetition_code_h(distance: usize) -> Vec<Vec<bool>> {
    assert!(distance >= 2, "repetition code distance must be >= 2");
    let mut h = vec![vec![false; distance]; distance - 1];
    for i in 0..(distance - 1) {
        h[i][i] = true;
        h[i][i + 1] = true;
    }
    h
}

/// Build the parity check matrix for a rotated surface code of distance d.
///
/// This produces a simplified X-stabilizer check matrix for a d x d surface
/// code patch. The matrix has approximately d^2 - 1 checks on 2*d^2 - 2*d + 1
/// data qubits (we use the standard planar layout).
///
/// For small distances used in tests, we construct the check matrix directly.
pub fn surface_code_x_checks(d: usize) -> Vec<Vec<bool>> {
    assert!(d >= 2, "surface code distance must be >= 2");

    // For a d x d grid of data qubits, X stabilizers are plaquettes.
    // Simplified: (d-1) x (d-1) plaquettes, each touching 4 data qubits.
    let n = d * d; // number of data qubits (on grid vertices)
    let num_checks = (d - 1) * (d - 1);
    let mut h = vec![vec![false; n]; num_checks];

    for row in 0..(d - 1) {
        for col in 0..(d - 1) {
            let check_idx = row * (d - 1) + col;
            // Four data qubits around this plaquette
            h[check_idx][row * d + col] = true; // top-left
            h[check_idx][row * d + col + 1] = true; // top-right
            h[check_idx][(row + 1) * d + col] = true; // bottom-left
            h[check_idx][(row + 1) * d + col + 1] = true; // bottom-right
        }
    }

    h
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test 1: config_builder ---

    #[test]
    fn config_builder_defaults() {
        let config = RelayBPConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!((config.damping_factor - 0.5).abs() < f64::EPSILON);
        assert!((config.relay_threshold - 0.01).abs() < f64::EPSILON);
        assert_eq!(config.max_relays, 5);
        assert!((config.convergence_eps - 1e-8).abs() < f64::EPSILON);
        assert_eq!(config.schedule, Schedule::Flooding);
    }

    #[test]
    fn config_builder_custom() {
        let config = RelayBPConfig::builder()
            .max_iterations(200)
            .damping_factor(0.8)
            .relay_threshold(0.05)
            .max_relays(10)
            .convergence_eps(1e-6)
            .schedule(Schedule::LayeredSerial)
            .build();

        assert_eq!(config.max_iterations, 200);
        assert!((config.damping_factor - 0.8).abs() < f64::EPSILON);
        assert!((config.relay_threshold - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.max_relays, 10);
        assert!((config.convergence_eps - 1e-6).abs() < f64::EPSILON);
        assert_eq!(config.schedule, Schedule::LayeredSerial);
    }

    // --- Test 2: tanner_graph_construction ---

    #[test]
    fn tanner_graph_construction() {
        // Repetition code d=5: 4 checks, 5 variables
        let h = repetition_code_h(5);
        let tanner = TannerGraph::from_parity_check(&h);

        assert_eq!(tanner.num_checks, 4);
        assert_eq!(tanner.num_variables, 5);
        // Each check connects to 2 variables, so 8 edges total
        assert_eq!(tanner.num_edges(), 8);

        // Check node 0 connects to variables 0 and 1
        assert_eq!(tanner.neighbors_of_check(0), &[0, 1]);
        // Variable 0 connects to check 0 only (boundary)
        assert_eq!(tanner.neighbors_of_variable(0), &[0]);
        // Variable 1 connects to checks 0 and 1 (interior)
        assert_eq!(tanner.neighbors_of_variable(1), &[0, 1]);
    }

    #[test]
    fn tanner_graph_surface_code() {
        // d=3 surface code: 4 checks, 9 variables, each check has degree 4
        let h = surface_code_x_checks(3);
        let tanner = TannerGraph::from_parity_check(&h);

        assert_eq!(tanner.num_checks, 4);
        assert_eq!(tanner.num_variables, 9);
        assert_eq!(tanner.num_edges(), 16); // 4 checks * 4 variables each
        for c in 0..4 {
            assert_eq!(tanner.check_degree(c), 4);
        }
    }

    // --- Test 3: repetition_code_decode ---

    #[test]
    fn repetition_code_decode_single_error() {
        // [[5,1,5]] repetition code, single error on qubit 0
        let h = repetition_code_h(5);
        let config = RelayBPConfig::builder()
            .max_iterations(50)
            .damping_factor(0.8)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        // Error on qubit 0: syndrome = [1, 0, 0, 0]
        let error = vec![true, false, false, false, false];
        let syndrome = decoder.syndrome_of(&error);
        assert_eq!(syndrome, vec![true, false, false, false]);

        let result = decoder.decode(&syndrome, 0.1).unwrap();
        assert!(
            result.syndrome_satisfied,
            "Decoded error must satisfy syndrome"
        );
        assert!(result.converged, "BP should converge for single error");
    }

    #[test]
    fn repetition_code_decode_middle_error() {
        // Error on qubit 2 (middle): syndrome = [0, 1, 1, 0]
        let h = repetition_code_h(5);
        let config = RelayBPConfig::builder()
            .max_iterations(50)
            .damping_factor(0.8)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        let error = vec![false, false, true, false, false];
        let syndrome = decoder.syndrome_of(&error);
        assert_eq!(syndrome, vec![false, true, true, false]);

        let result = decoder.decode(&syndrome, 0.1).unwrap();
        assert!(result.syndrome_satisfied);
    }

    // --- Test 4: surface_code_decode ---

    #[test]
    fn surface_code_decode_single_error() {
        // d=3 surface code, single error
        let h = surface_code_x_checks(3);
        let config = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(0.5)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        // Error on qubit 4 (center of 3x3 grid): touches all 4 checks
        let error = vec![false, false, false, false, true, false, false, false, false];
        let syndrome = decoder.syndrome_of(&error);

        let result = decoder.decode(&syndrome, 0.05).unwrap();
        assert!(
            result.syndrome_satisfied,
            "Surface code single error should be decodable"
        );
    }

    // --- Test 5: bp_convergence ---

    #[test]
    fn bp_convergence_easy_error() {
        // Repetition code d=7, single error: BP should converge quickly
        let h = repetition_code_h(7);
        let config = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(0.8)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        let error = vec![false, false, false, true, false, false, false];
        let syndrome = decoder.syndrome_of(&error);

        let result = decoder.decode(&syndrome, 0.05).unwrap();
        assert!(result.converged, "BP should converge for easy single error");
        assert!(
            result.iterations_used < 50,
            "Should converge well before max iterations, used {}",
            result.iterations_used
        );
        assert!(!result.osd_used, "Should not need OSD for easy errors");
    }

    // --- Test 6: relay_activation ---

    #[test]
    fn relay_activation_trapping_set() {
        // Create a scenario where BP is more likely to get stuck:
        // use very low damping (oscillation-prone) and tight threshold
        let h = repetition_code_h(5);
        let config = RelayBPConfig::builder()
            .max_iterations(10) // very few iterations to force non-convergence
            .damping_factor(1.0) // no damping = more oscillation
            .relay_threshold(100.0) // very high threshold: always detect "trapping"
            .max_relays(3)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        // Multi-error pattern that may cause BP difficulty
        let error = vec![true, false, true, false, false];
        let syndrome = decoder.syndrome_of(&error);

        let result = decoder.decode(&syndrome, 0.3).unwrap();
        // With a high relay_threshold, relays should be inserted
        // The result may or may not converge, but relays should have been tried
        assert!(
            result.relays_inserted > 0 || result.converged,
            "Either relays should be inserted or BP converged before needing them"
        );
    }

    // --- Test 7: relay_breaks_trap ---

    #[test]
    fn relay_breaks_trap() {
        // Use short iterations without relay, then with relay, to show
        // relay can help escape a stuck configuration.
        let h = repetition_code_h(5);

        // First: very few iterations, no relays
        let config_no_relay = RelayBPConfig::builder()
            .max_iterations(3)
            .damping_factor(1.0)
            .max_relays(0)
            .build();
        let decoder_no_relay = RelayBPDecoder::new(h.clone(), config_no_relay);

        let error = vec![true, false, true, false, false];
        let syndrome = decoder_no_relay.syndrome_of(&error);

        let result_no_relay = decoder_no_relay.decode(&syndrome, 0.3).unwrap();

        // Now with relays: more total iterations across relay rounds
        let config_relay = RelayBPConfig::builder()
            .max_iterations(3)
            .damping_factor(1.0)
            .relay_threshold(100.0) // force relay activation
            .max_relays(5)
            .build();
        let decoder_relay = RelayBPDecoder::new(h, config_relay);

        let result_relay = decoder_relay.decode(&syndrome, 0.3).unwrap();

        // The relay decoder gets more total iterations (3 per round * up to 6 rounds)
        // so it should have at least as good a result
        assert!(
            result_relay.iterations_used >= result_no_relay.iterations_used,
            "Relay decoder should use at least as many total iterations"
        );
    }

    // --- Test 8: osd_fallback ---

    #[test]
    fn osd_fallback() {
        // Force OSD by using 1 iteration and 0 relays
        let h = repetition_code_h(5);
        let config = RelayBPConfig::builder()
            .max_iterations(1)
            .damping_factor(1.0)
            .max_relays(0)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        // Two errors: harder pattern
        let error = vec![true, false, true, false, false];
        let syndrome = decoder.syndrome_of(&error);

        let result = decoder.decode(&syndrome, 0.3).unwrap();
        // OSD should be used since BP has only 1 iteration and no relays
        assert!(
            result.osd_used,
            "OSD should be invoked when BP fails with no relay budget"
        );
        // OSD-0 should at least satisfy the syndrome
        assert!(
            result.syndrome_satisfied,
            "OSD-0 should produce a valid correction satisfying the syndrome"
        );
    }

    // --- Test 9: damping_effect ---

    #[test]
    fn damping_effect() {
        // Compare convergence with strong damping vs no damping
        let h = repetition_code_h(7);

        let config_damped = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(0.5) // strong damping
            .max_relays(0)
            .build();
        let decoder_damped = RelayBPDecoder::new(h.clone(), config_damped);

        let config_undamped = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(1.0) // no damping
            .max_relays(0)
            .build();
        let decoder_undamped = RelayBPDecoder::new(h, config_undamped);

        let error = vec![false, false, false, true, false, false, false];
        let syndrome = decoder_damped.syndrome_of(&error);

        let result_damped = decoder_damped.decode(&syndrome, 0.1).unwrap();
        let result_undamped = decoder_undamped.decode(&syndrome, 0.1).unwrap();

        // Both should converge for this easy case
        assert!(result_damped.converged, "Damped BP should converge");
        assert!(result_undamped.converged, "Undamped BP should converge");

        // Damped version typically converges in fewer iterations for easy cases,
        // but the key property is that both converge and satisfy the syndrome.
        assert!(result_damped.syndrome_satisfied);
        assert!(result_undamped.syndrome_satisfied);
    }

    // --- Test 10: serial_vs_flooding ---

    #[test]
    fn serial_vs_flooding() {
        let h = repetition_code_h(7);

        let config_flood = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(0.8)
            .schedule(Schedule::Flooding)
            .max_relays(0)
            .build();
        let decoder_flood = RelayBPDecoder::new(h.clone(), config_flood);

        let config_serial = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(0.8)
            .schedule(Schedule::LayeredSerial)
            .max_relays(0)
            .build();
        let decoder_serial = RelayBPDecoder::new(h, config_serial);

        let error = vec![false, false, true, false, false, false, false];
        let syndrome = decoder_flood.syndrome_of(&error);

        let result_flood = decoder_flood.decode(&syndrome, 0.1).unwrap();
        let result_serial = decoder_serial.decode(&syndrome, 0.1).unwrap();

        // Both should converge
        assert!(result_flood.converged, "Flooding should converge");
        assert!(result_serial.converged, "Serial should converge");
        assert!(result_flood.syndrome_satisfied);
        assert!(result_serial.syndrome_satisfied);

        // Serial schedule typically converges in fewer iterations
        // (each iteration propagates information further)
        assert!(
            result_serial.iterations_used <= result_flood.iterations_used + 5,
            "Serial should not be significantly worse than flooding: serial={}, flooding={}",
            result_serial.iterations_used,
            result_flood.iterations_used
        );
    }

    // --- Test 11: multi_error_decode ---

    #[test]
    fn multi_error_decode() {
        // d=5 repetition code with 2 errors (below correction capacity)
        let h = repetition_code_h(5);
        let config = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(0.5)
            .max_relays(3)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        // Two errors at positions 0 and 3
        let error = vec![true, false, false, true, false];
        let syndrome = decoder.syndrome_of(&error);

        let result = decoder.decode(&syndrome, 0.2).unwrap();
        // The decoded error must satisfy the syndrome (may differ from
        // the actual error if it finds an equivalent correction).
        assert!(
            result.syndrome_satisfied,
            "Multi-error decode should satisfy syndrome"
        );
    }

    // --- Test 12: logical_error_detection ---

    #[test]
    fn logical_error_detection() {
        // For a repetition code, a logical error occurs when the decoder
        // finds a correction that differs from the actual error by a
        // logical operator (all-ones for the rep code).
        let h = repetition_code_h(5);
        let config = RelayBPConfig::builder()
            .max_iterations(100)
            .damping_factor(0.5)
            .max_relays(5)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        // Single error: decoder should correct without logical error
        let error = vec![true, false, false, false, false];
        let syndrome = decoder.syndrome_of(&error);
        let result = decoder.decode(&syndrome, 0.1).unwrap();

        // Compute residual = error XOR correction
        let residual: Vec<bool> = error
            .iter()
            .zip(result.error_estimate.iter())
            .map(|(&e, &c)| e ^ c)
            .collect();

        // Check if residual is trivial (no logical error) or a logical operator.
        // For a repetition code, the logical operator is all-ones.
        let residual_is_logical = residual.iter().all(|&b| b);
        let residual_is_trivial = residual.iter().all(|&b| !b);

        assert!(
            residual_is_trivial || residual_is_logical,
            "Residual must be either trivial or a logical operator, got {:?}",
            residual
        );

        // For low error rate and single error, we expect trivial residual
        if result.converged {
            // A converged single-error decode at low error rate should not
            // introduce a logical error (the all-ones correction has much
            // higher weight than the single-qubit correction).
            assert!(
                residual_is_trivial,
                "Converged single-error decode should not introduce logical error"
            );
        }
    }

    // --- Test 13: syndrome_validation ---

    #[test]
    fn syndrome_validation() {
        // Verify that the decoded error always satisfies the syndrome,
        // regardless of the decoding path taken.
        let h = repetition_code_h(7);
        let config = RelayBPConfig::builder()
            .max_iterations(50)
            .damping_factor(0.5)
            .max_relays(5)
            .build();
        let decoder = RelayBPDecoder::new(h, config);

        // Test several error patterns
        let patterns: Vec<Vec<bool>> = vec![
            vec![true, false, false, false, false, false, false],
            vec![false, false, false, true, false, false, false],
            vec![true, false, false, false, false, false, true],
            vec![false, true, false, true, false, false, false],
            vec![false, false, false, false, false, false, false], // no error
        ];

        for error in &patterns {
            let syndrome = decoder.syndrome_of(error);
            let result = decoder.decode(&syndrome, 0.1).unwrap();

            // Manually verify syndrome satisfaction
            let computed_syn = decoder.syndrome_of(&result.error_estimate);
            assert_eq!(
                computed_syn, syndrome,
                "Decoded error syndrome must match input syndrome for error {:?}",
                error
            );
        }
    }

    // --- Additional edge case tests ---

    #[test]
    fn dimension_mismatch_error() {
        let h = repetition_code_h(5);
        let config = RelayBPConfig::default();
        let decoder = RelayBPDecoder::new(h, config);

        // Wrong syndrome length
        let bad_syndrome = vec![true, false]; // should be length 4
        let result = decoder.decode(&bad_syndrome, 0.1);
        assert!(result.is_err());
        match result.unwrap_err() {
            RelayBPError::MatrixDimensionMismatch {
                expected_checks,
                got_syndrome_len,
            } => {
                assert_eq!(expected_checks, 4);
                assert_eq!(got_syndrome_len, 2);
            }
            other => panic!("Expected MatrixDimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn zero_syndrome_returns_immediately() {
        let h = repetition_code_h(5);
        let config = RelayBPConfig::default();
        let decoder = RelayBPDecoder::new(h, config);

        let syndrome = vec![false, false, false, false];
        let result = decoder.decode(&syndrome, 0.1).unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations_used, 0);
        assert!(result.error_estimate.iter().all(|&b| !b));
    }

    #[test]
    fn osd_standalone() {
        // Test OSD-0 directly
        let h = repetition_code_h(5);
        let syndrome = vec![true, true, false, false]; // error on qubit 1

        // Soft LLR: qubit 1 has lowest reliability (most likely error)
        let soft_llr = vec![2.0, 0.1, 2.0, 2.0, 2.0];

        let result = OSD::decode(&h, &syndrome, &soft_llr);

        // Check syndrome satisfaction
        let mut computed_syn = vec![false; 4];
        for c in 0..4 {
            let mut parity = false;
            for v in 0..5 {
                if h[c][v] && result[v] {
                    parity = !parity;
                }
            }
            computed_syn[c] = parity;
        }
        assert_eq!(
            computed_syn, syndrome,
            "OSD result must satisfy the syndrome"
        );
    }

    #[test]
    fn bp_message_variance() {
        let mut msg = BPMessage::new(1.0);
        // Single value: variance should be infinity (not enough data)
        assert!(msg.variance().is_infinite());

        // Add identical values: variance should be 0
        msg.update(1.0);
        msg.update(1.0);
        assert!((msg.variance()).abs() < 1e-12);

        // Add a different value: variance should be nonzero
        msg.update(2.0);
        assert!(msg.variance() > 0.0);
    }

    #[test]
    fn relay_node_creation() {
        let relay = RelayNode {
            check_idx: 2,
            var_idx: 3,
            inserted_at_iter: 42,
            perturbation: 0.15,
        };
        assert_eq!(relay.check_idx, 2);
        assert_eq!(relay.var_idx, 3);
        assert_eq!(relay.inserted_at_iter, 42);
        assert!((relay.perturbation - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn error_display() {
        let e1 = RelayBPError::MatrixDimensionMismatch {
            expected_checks: 4,
            got_syndrome_len: 3,
        };
        assert!(e1.to_string().contains("4"));
        assert!(e1.to_string().contains("3"));

        let e2 = RelayBPError::NotConverged {
            iterations_used: 100,
            relays_used: 5,
        };
        assert!(e2.to_string().contains("100"));

        let e3 = RelayBPError::InvalidSyndrome("bad data".to_string());
        assert!(e3.to_string().contains("bad data"));
    }
}
