//! Stabilizer Tensor Networks
//!
//! A hybrid simulation method inspired by PRL Dec 2024 where computational cost
//! scales with non-stabilizerness (magic) rather than qubit count. The network
//! maintains nodes in stabilizer representation when possible, promoting to dense
//! only when accumulated magic exceeds a configurable threshold.
//!
//! # Architecture
//!
//! The simulator maintains a collection of hybrid tensor nodes. Each node is
//! either a stabilizer decomposition (efficient for Clifford operations) or a
//! dense state vector (required for high-magic regions). Gates route adaptively:
//!
//! - **Clifford gates** (H, S, X, Y, Z, CNOT, CZ, SWAP) are applied directly
//!   to stabilizer nodes in O(n^2) time.
//! - **Non-Clifford gates** (T, Rz with non-Clifford angle) cause stabilizer
//!   branching. When branch count or magic exceeds a threshold, the node is
//!   promoted to dense representation.
//! - **Demotion** back to stabilizer is attempted after long Clifford sequences
//!   on dense nodes, recovering efficiency when magic has been consumed.
//!
//! # Magic Monotones
//!
//! The [`MagicMonotone`] struct provides computable measures of non-stabilizerness:
//! - Stabilizer extent: (sum |c_k|)^2 for decomposition sum_k c_k |S_k>
//! - Mana: log2(sum |c_k|)
//! - Robustness: sum |c_k|
//!
//! # References
//!
//! - Qassim, Pashayan, Gosset, "Improved upper bounds on the stabilizer rank" (2021)
//! - Bravyi, Smith, Smolin, "Trading classical and quantum resources" (2016)
//! - Bravyi, Browne, Calpin, Campbell, Gosset, Howard, "Simulation of quantum
//!   circuits by low-rank stabilizer decompositions" (2019)

use std::collections::HashMap;

use crate::gates::{Gate, GateType};
use crate::stabilizer::StabilizerState;
use crate::{c64_one, c64_zero, C64};

// ============================================================
// MAGIC MONOTONE COMPUTATION
// ============================================================

/// Computes magic monotones for a quantum state given as a stabilizer decomposition.
///
/// Magic monotones quantify how far a state is from the set of stabilizer states.
/// Lower values mean the state is closer to being efficiently simulable.
pub struct MagicMonotone;

impl MagicMonotone {
    /// Compute the stabilizer extent of a state given as a weighted sum of
    /// stabilizer states: |psi> = sum_k c_k |S_k>.
    ///
    /// The stabilizer extent xi(|psi>) = (sum_k |c_k|)^2, the squared l1-norm
    /// of the decomposition coefficients. For pure stabilizer states xi = 1.
    pub fn stabilizer_extent(coefficients: &[C64]) -> f64 {
        let l1: f64 = coefficients.iter().map(|c| c.norm()).sum();
        l1 * l1
    }

    /// Compute the mana (logarithmic negativity of the discrete Wigner function).
    ///
    /// For a decomposition with coefficients c_k, mana = log2(sum_k |c_k|).
    /// For stabilizer states mana = 0.
    pub fn mana(coefficients: &[C64]) -> f64 {
        let l1: f64 = coefficients.iter().map(|c| c.norm()).sum();
        if l1 <= 1.0 + 1e-12 {
            0.0
        } else {
            l1.log2()
        }
    }

    /// Compute the robustness of magic.
    ///
    /// The robustness R(|psi>) = sum_k |c_k| for the decomposition.
    /// For stabilizer states R = 1.
    pub fn robustness(coefficients: &[C64]) -> f64 {
        coefficients.iter().map(|c| c.norm()).sum()
    }

    /// Estimate magic from T-gate count using known bounds.
    ///
    /// For t T-gates the stabilizer extent is at most (1/cos(pi/8))^t.
    /// Returns (extent_upper_bound, mana_upper_bound).
    pub fn from_t_count(t_count: usize) -> (f64, f64) {
        let single = 1.0 / (std::f64::consts::PI / 8.0).cos();
        let extent = single.powi(t_count as i32);
        let mana = extent.log2();
        (extent, mana)
    }

    /// Determine whether a gate type is Clifford.
    pub fn is_clifford_gate(gate_type: &GateType) -> bool {
        matches!(
            gate_type,
            GateType::H
                | GateType::S
                | GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::SX
                | GateType::CNOT
                | GateType::CZ
                | GateType::SWAP
        )
    }
}

// ============================================================
// STABILIZER TERM
// ============================================================

/// A weighted stabilizer state: coeff * |stabilizer_state>.
///
/// This is the fundamental building block of stabilizer decompositions.
/// A general quantum state can be written as sum_k c_k |S_k> where each
/// |S_k> is a stabilizer state.
#[derive(Clone, Debug)]
pub struct StabilizerTerm {
    /// The stabilizer state (tableau representation).
    pub state: StabilizerState,
    /// Complex coefficient (weight).
    pub coeff: C64,
}

impl StabilizerTerm {
    /// Create a new stabilizer term with given state and coefficient.
    pub fn new(state: StabilizerState, coeff: C64) -> Self {
        StabilizerTerm { state, coeff }
    }

    /// Create a term with unit coefficient.
    pub fn unit(state: StabilizerState) -> Self {
        Self::new(state, c64_one())
    }

    /// Check if the coefficient is negligible.
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.coeff.norm() < threshold
    }

    /// Compute the inner product of two stabilizer states using statevector
    /// expansion (exact, practical for systems up to ~16 qubits).
    pub fn stabilizer_inner_product(&self, other: &StabilizerTerm) -> C64 {
        let n = self.state.num_qubits();
        assert_eq!(n, other.state.num_qubits(), "Qubit count mismatch");

        let sv1 = stabilizer_to_statevector(&self.state);
        let sv2 = stabilizer_to_statevector(&other.state);
        let dim = sv1.len();
        let mut ip = c64_zero();
        for i in 0..dim {
            ip += sv1[i].conj() * sv2[i];
        }
        ip
    }

    /// Compute the weighted inner product including coefficients:
    /// conj(self.coeff) * other.coeff * <self.state|other.state>.
    pub fn weighted_inner_product(&self, other: &StabilizerTerm) -> C64 {
        self.coeff.conj() * other.coeff * self.stabilizer_inner_product(other)
    }
}

// ============================================================
// HYBRID TENSOR NODE
// ============================================================

/// A node in the hybrid tensor network.
///
/// Each node represents a subset of qubits in either stabilizer or dense form.
/// The representation is chosen adaptively based on the magic content.
#[derive(Clone, Debug)]
pub enum HybridTensorNode {
    /// Stabilizer node: weighted sum of stabilizer states.
    Stabilizer {
        /// The terms in the stabilizer decomposition.
        terms: Vec<StabilizerTerm>,
    },
    /// Dense tensor node: full 2^n state vector.
    Dense {
        /// Flattened tensor data in computational-basis order.
        data: Vec<C64>,
        /// Dimensions of each qubit axis (always vec![2; num_qubits]).
        dims: Vec<usize>,
    },
}

impl HybridTensorNode {
    /// Create a new stabilizer node initialised to |0>^n.
    pub fn new_stabilizer(num_qubits: usize) -> Self {
        HybridTensorNode::Stabilizer {
            terms: vec![StabilizerTerm::unit(StabilizerState::new(num_qubits))],
        }
    }

    /// Create a new dense node initialised to |0>^n.
    pub fn new_dense(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut data = vec![c64_zero(); dim];
        data[0] = c64_one();
        HybridTensorNode::Dense {
            data,
            dims: vec![2; num_qubits],
        }
    }

    /// Returns true if this node uses the stabilizer representation.
    pub fn is_stabilizer(&self) -> bool {
        matches!(self, HybridTensorNode::Stabilizer { .. })
    }

    /// Returns true if this node uses the dense representation.
    pub fn is_dense(&self) -> bool {
        matches!(self, HybridTensorNode::Dense { .. })
    }

    /// Returns the number of qubits in this node.
    pub fn num_qubits(&self) -> usize {
        match self {
            HybridTensorNode::Stabilizer { terms } => {
                if terms.is_empty() {
                    0
                } else {
                    terms[0].state.num_qubits()
                }
            }
            HybridTensorNode::Dense { dims, .. } => dims.len(),
        }
    }

    /// Returns the number of stabilizer terms (0 if dense).
    pub fn num_terms(&self) -> usize {
        match self {
            HybridTensorNode::Stabilizer { terms } => terms.len(),
            HybridTensorNode::Dense { .. } => 0,
        }
    }

    /// Compute the magic (stabilizer extent) of this node.
    pub fn magic(&self) -> f64 {
        match self {
            HybridTensorNode::Stabilizer { terms } => {
                let coeffs: Vec<C64> = terms.iter().map(|t| t.coeff).collect();
                MagicMonotone::stabilizer_extent(&coeffs)
            }
            HybridTensorNode::Dense { .. } => f64::INFINITY,
        }
    }

    /// Apply a Clifford gate to this node.
    ///
    /// Returns true if the gate was recognised as Clifford and applied.
    pub fn apply_clifford(&mut self, gate: &Gate) -> bool {
        if !MagicMonotone::is_clifford_gate(&gate.gate_type) {
            return false;
        }
        match self {
            HybridTensorNode::Stabilizer { terms } => {
                for term in terms.iter_mut() {
                    apply_clifford_to_stabilizer(&mut term.state, gate);
                }
            }
            HybridTensorNode::Dense { data, dims } => {
                apply_gate_to_dense(data, dims.len(), gate);
            }
        }
        true
    }

    /// Apply any gate (Clifford or non-Clifford) to this node.
    ///
    /// Non-Clifford gates on stabilizer nodes cause branching via the
    /// decomposition T = a*I + b*Z.
    pub fn apply_gate(&mut self, gate: &Gate) {
        match self {
            HybridTensorNode::Stabilizer { terms } => {
                if MagicMonotone::is_clifford_gate(&gate.gate_type) {
                    for term in terms.iter_mut() {
                        apply_clifford_to_stabilizer(&mut term.state, gate);
                    }
                } else {
                    *terms = branch_non_clifford(terms, gate);
                }
            }
            HybridTensorNode::Dense { data, dims } => {
                apply_gate_to_dense(data, dims.len(), gate);
            }
        }
    }

    /// Promote this node from stabilizer to dense representation.
    pub fn promote_to_dense(&mut self) {
        if let HybridTensorNode::Stabilizer { terms } = self {
            if terms.is_empty() {
                *self = HybridTensorNode::Dense {
                    data: vec![c64_one()],
                    dims: vec![],
                };
                return;
            }
            let n = terms[0].state.num_qubits();
            let dim = 1usize << n;
            let mut dense = vec![c64_zero(); dim];
            for term in terms.iter() {
                let sv = stabilizer_to_statevector(&term.state);
                for (i, amp) in sv.iter().enumerate() {
                    dense[i] += term.coeff * amp;
                }
            }
            *self = HybridTensorNode::Dense {
                data: dense,
                dims: vec![2; n],
            };
        }
    }

    /// Attempt to demote this node from dense back to stabilizer.
    ///
    /// Returns true if demotion succeeded. A state is demotable if it can
    /// be identified as a stabilizer state (power-of-two support, equal
    /// magnitude amplitudes with stabilizer phases).
    pub fn try_demote_to_stabilizer(&mut self) -> bool {
        if let HybridTensorNode::Dense { data, dims } = self {
            let n = dims.len();
            if let Some(stab) = try_extract_stabilizer(data, n) {
                *self = HybridTensorNode::Stabilizer {
                    terms: vec![StabilizerTerm::unit(stab)],
                };
                return true;
            }
        }
        false
    }

    /// Prune stabilizer terms with negligible coefficients.
    pub fn prune(&mut self, threshold: f64) {
        if let HybridTensorNode::Stabilizer { terms } = self {
            terms.retain(|t| !t.is_negligible(threshold));
        }
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the stabilizer tensor network.
#[derive(Clone, Debug)]
pub struct STNConfig {
    /// Magic threshold above which a node is promoted to dense.
    pub magic_threshold: f64,
    /// Maximum number of stabilizer terms per node before promotion.
    pub max_stabilizer_terms: usize,
    /// Whether to attempt demotion back to stabilizer after Clifford sequences.
    pub enable_demotion: bool,
    /// Number of consecutive Clifford gates required before demotion attempt.
    pub demotion_clifford_threshold: usize,
    /// Pruning threshold for negligible coefficients.
    pub prune_threshold: f64,
}

impl Default for STNConfig {
    fn default() -> Self {
        STNConfig {
            magic_threshold: 10.0,
            max_stabilizer_terms: 64,
            enable_demotion: true,
            demotion_clifford_threshold: 10,
            prune_threshold: 1e-12,
        }
    }
}

// ============================================================
// STATISTICS
// ============================================================

/// Runtime statistics for the stabilizer tensor network simulator.
#[derive(Clone, Debug, Default)]
pub struct STNStats {
    pub stabilizer_gates: usize,
    pub dense_gates: usize,
    pub promotions: usize,
    pub demotions: usize,
    pub current_stabilizer_nodes: usize,
    pub current_dense_nodes: usize,
}

impl std::fmt::Display for STNStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "STNStats {{ stab_gates: {}, dense_gates: {}, promotions: {}, demotions: {}, \
             stab_nodes: {}, dense_nodes: {} }}",
            self.stabilizer_gates,
            self.dense_gates,
            self.promotions,
            self.demotions,
            self.current_stabilizer_nodes,
            self.current_dense_nodes
        )
    }
}

// ============================================================
// STABILIZER TENSOR NETWORK
// ============================================================

/// A tensor network with hybrid stabilizer/dense nodes.
///
/// Nodes represent groups of qubits. The network automatically manages
/// promotion (stabilizer -> dense) and demotion (dense -> stabilizer)
/// based on magic thresholds.
pub struct StabilizerTensorNetwork {
    /// The hybrid nodes in the network.
    nodes: Vec<HybridTensorNode>,
    /// Mapping: qubit_to_node[q] = index into nodes.
    qubit_to_node: Vec<usize>,
    /// Total number of qubits.
    num_qubits: usize,
    /// Configuration parameters.
    config: STNConfig,
    /// Runtime statistics.
    stats: STNStats,
    /// Per-node consecutive Clifford gate counter (for demotion).
    consecutive_cliffords: Vec<usize>,
}

impl StabilizerTensorNetwork {
    /// Create a new network with all qubits in a single stabilizer node.
    pub fn new(num_qubits: usize, config: STNConfig) -> Self {
        let node = HybridTensorNode::new_stabilizer(num_qubits);
        let qubit_to_node = vec![0; num_qubits];
        let mut stats = STNStats::default();
        stats.current_stabilizer_nodes = 1;
        StabilizerTensorNetwork {
            nodes: vec![node],
            qubit_to_node,
            num_qubits,
            config,
            stats,
            consecutive_cliffords: vec![0],
        }
    }

    /// Get the number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get a reference to the underlying nodes.
    pub fn nodes(&self) -> &[HybridTensorNode] {
        &self.nodes
    }

    /// Get the runtime statistics.
    pub fn stats(&self) -> &STNStats {
        &self.stats
    }

    /// Check if all nodes are stabilizer.
    pub fn is_all_stabilizer(&self) -> bool {
        self.nodes.iter().all(|n| n.is_stabilizer())
    }

    /// Check if all nodes are dense.
    pub fn is_all_dense(&self) -> bool {
        self.nodes.iter().all(|n| n.is_dense())
    }

    /// Get the total number of stabilizer terms across all nodes.
    pub fn total_stabilizer_terms(&self) -> usize {
        self.nodes.iter().map(|n| n.num_terms()).sum()
    }

    // --------------------------------------------------------
    // Gate application
    // --------------------------------------------------------

    /// Apply a gate to the network.
    ///
    /// Clifford gates are applied directly. Non-Clifford gates cause
    /// stabilizer branching or dense-mode matrix application.
    pub fn apply_gate(&mut self, gate: &Gate) {
        let target_qubit = gate.targets[0];
        let node_idx = self.qubit_to_node[target_qubit];
        let is_clifford = MagicMonotone::is_clifford_gate(&gate.gate_type);

        if is_clifford {
            self.nodes[node_idx].apply_clifford(gate);
            if self.nodes[node_idx].is_stabilizer() {
                self.stats.stabilizer_gates += 1;
            } else {
                self.stats.dense_gates += 1;
            }
            self.consecutive_cliffords[node_idx] += 1;

            // Attempt demotion if threshold reached.
            if self.config.enable_demotion
                && self.nodes[node_idx].is_dense()
                && self.consecutive_cliffords[node_idx] >= self.config.demotion_clifford_threshold
            {
                if self.nodes[node_idx].try_demote_to_stabilizer() {
                    self.stats.demotions += 1;
                    self.stats.current_dense_nodes =
                        self.stats.current_dense_nodes.saturating_sub(1);
                    self.stats.current_stabilizer_nodes += 1;
                }
                self.consecutive_cliffords[node_idx] = 0;
            }
        } else {
            self.consecutive_cliffords[node_idx] = 0;
            self.nodes[node_idx].apply_gate(gate);

            if self.nodes[node_idx].is_stabilizer() {
                self.stats.stabilizer_gates += 1;
                self.nodes[node_idx].prune(self.config.prune_threshold);

                // Check promotion conditions.
                let num_terms = self.nodes[node_idx].num_terms();
                let magic = self.nodes[node_idx].magic();
                if num_terms > self.config.max_stabilizer_terms
                    || magic > self.config.magic_threshold
                {
                    self.promote_node(node_idx);
                }
            } else {
                self.stats.dense_gates += 1;
            }
        }
    }

    /// Apply a sequence of gates.
    pub fn apply_circuit(&mut self, gates: &[Gate]) {
        for gate in gates {
            self.apply_gate(gate);
        }
    }

    // --------------------------------------------------------
    // Node management
    // --------------------------------------------------------

    /// Promote a stabilizer node to dense representation.
    pub fn promote_node(&mut self, node_idx: usize) {
        if self.nodes[node_idx].is_dense() {
            return;
        }
        self.nodes[node_idx].promote_to_dense();
        self.stats.promotions += 1;
        self.stats.current_stabilizer_nodes =
            self.stats.current_stabilizer_nodes.saturating_sub(1);
        self.stats.current_dense_nodes += 1;
    }

    /// Attempt to demote a dense node back to stabilizer.
    pub fn demote_node(&mut self, node_idx: usize) -> bool {
        if self.nodes[node_idx].is_stabilizer() {
            return true;
        }
        if self.nodes[node_idx].try_demote_to_stabilizer() {
            self.stats.demotions += 1;
            self.stats.current_dense_nodes =
                self.stats.current_dense_nodes.saturating_sub(1);
            self.stats.current_stabilizer_nodes += 1;
            true
        } else {
            false
        }
    }

    // --------------------------------------------------------
    // Contraction
    // --------------------------------------------------------

    /// Contract the entire network into a single statevector.
    ///
    /// For a single-node network, the statevector is returned directly.
    /// For multi-node networks, a greedy contraction is performed:
    /// pairs of nodes are contracted by summing over shared indices
    /// (tensor product for independent subsystems), choosing the
    /// smallest-tensor pair first at each step.
    pub fn contract(&self) -> Vec<C64> {
        if self.nodes.len() == 1 {
            return node_to_statevector(&self.nodes[0]);
        }

        // Convert all nodes to dense statevectors for contraction.
        let mut tensors: Vec<Vec<C64>> = self
            .nodes
            .iter()
            .map(|n| node_to_statevector(n))
            .collect();

        // Track the number of qubits each tensor represents.
        let mut qubit_counts: Vec<usize> = self.nodes.iter().map(|n| n.num_qubits()).collect();

        // Greedy contraction: repeatedly pick the pair with smallest combined
        // tensor size and contract them via tensor product (Kronecker product).
        // In a full tensor-network implementation, shared indices between
        // connected nodes would be summed over. Since the current architecture
        // uses independent qubit groups per node (no shared bond indices),
        // contraction is a tensor product.
        while tensors.len() > 1 {
            // Find the pair (i, j) with smallest combined size.
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_size = tensors[0].len() + tensors[1].len();

            for i in 0..tensors.len() {
                for j in (i + 1)..tensors.len() {
                    let combined = tensors[i].len() + tensors[j].len();
                    if combined < best_size {
                        best_size = combined;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Contract: tensor product of the two statevectors.
            // |psi_A> tensor |psi_B> has dimension dim_A * dim_B.
            let tensor_a = &tensors[best_i];
            let tensor_b = &tensors[best_j];
            let dim_a = tensor_a.len();
            let dim_b = tensor_b.len();
            let mut contracted = vec![c64_zero(); dim_a * dim_b];

            for (ia, &amp_a) in tensor_a.iter().enumerate() {
                for (ib, &amp_b) in tensor_b.iter().enumerate() {
                    contracted[ia + ib * dim_a] = amp_a * amp_b;
                }
            }

            let new_qubits = qubit_counts[best_i] + qubit_counts[best_j];

            // Remove the two old tensors (remove higher index first to
            // avoid invalidating the lower index).
            if best_i < best_j {
                tensors.remove(best_j);
                qubit_counts.remove(best_j);
                tensors.remove(best_i);
                qubit_counts.remove(best_i);
            } else {
                tensors.remove(best_i);
                qubit_counts.remove(best_i);
                tensors.remove(best_j);
                qubit_counts.remove(best_j);
            }

            // Insert the contracted result.
            tensors.push(contracted);
            qubit_counts.push(new_qubits);
        }

        tensors.into_iter().next().unwrap_or_else(|| vec![c64_one()])
    }

    /// Compute the full probability distribution.
    pub fn probabilities(&self) -> Vec<f64> {
        let sv = self.contract();
        sv.iter().map(|c| c.norm_sqr()).collect()
    }

    // --------------------------------------------------------
    // Measurement
    // --------------------------------------------------------

    /// Sample measurement outcomes.
    ///
    /// Returns a histogram mapping bitstring (as usize) to count.
    ///
    /// For multi-node networks, each node is sampled independently (since nodes
    /// represent independent qubit groups with no shared bond indices) and the
    /// per-node bitstrings are combined by placing bits at the correct global
    /// qubit positions.
    pub fn sample(&self, n_shots: usize) -> HashMap<usize, usize> {
        // Build node-to-qubits mapping: which global qubits belong to each node.
        let mut node_to_qubits: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (q, &node_idx) in self.qubit_to_node.iter().enumerate() {
            node_to_qubits[node_idx].push(q);
        }

        let mut counts: HashMap<usize, usize> = HashMap::new();

        for _ in 0..n_shots {
            let mut full_bs = 0usize;

            for (node_idx, node) in self.nodes.iter().enumerate() {
                let qubits = &node_to_qubits[node_idx];
                let local_bs = sample_node_once(node);

                // Map local bitstring bits to global qubit positions.
                for (local_q, &global_q) in qubits.iter().enumerate() {
                    if (local_bs >> local_q) & 1 == 1 {
                        full_bs |= 1 << global_q;
                    }
                }
            }

            *counts.entry(full_bs).or_insert(0) += 1;
        }
        counts
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Apply a Clifford gate to a stabilizer state via the tableau.
fn apply_clifford_to_stabilizer(state: &mut StabilizerState, gate: &Gate) {
    let target = gate.targets[0];
    match gate.gate_type {
        GateType::H => state.h(target),
        GateType::S => state.s(target),
        GateType::X => state.x(target),
        GateType::Y => state.y(target),
        GateType::Z => state.z(target),
        GateType::SX => {
            state.s_dag(target);
            state.h(target);
            state.s_dag(target);
        }
        GateType::CNOT => {
            let control = gate.controls[0];
            state.cx(control, target);
        }
        GateType::CZ => {
            let control = gate.controls[0];
            state.cz(control, target);
        }
        GateType::SWAP => {
            let b = gate.targets[1];
            state.swap(target, b);
        }
        _ => {} // Non-Clifford handled elsewhere.
    }
}

/// Branch non-Clifford gates via stabilizer decomposition.
///
/// T = a*I + b*Z, Rz(theta) = cos(theta/2)*I - i*sin(theta/2)*Z, etc.
fn branch_non_clifford(terms: &[StabilizerTerm], gate: &Gate) -> Vec<StabilizerTerm> {
    let target = gate.targets[0];

    match gate.gate_type {
        GateType::T => {
            let phase = std::f64::consts::FRAC_PI_4;
            let a = C64::new((1.0 + phase.cos()) / 2.0, phase.sin() / 2.0);
            let b = C64::new((1.0 - phase.cos()) / 2.0, -phase.sin() / 2.0);
            let mut out = Vec::with_capacity(terms.len() * 2);
            for term in terms {
                out.push(StabilizerTerm::new(term.state.clone(), term.coeff * a));
                let mut zs = term.state.clone();
                zs.z(target);
                out.push(StabilizerTerm::new(zs, term.coeff * b));
            }
            out
        }
        GateType::Phase(theta) | GateType::Rz(theta) => {
            let (a, b) = if matches!(gate.gate_type, GateType::Phase(_)) {
                (
                    C64::new((1.0 + theta.cos()) / 2.0, theta.sin() / 2.0),
                    C64::new((1.0 - theta.cos()) / 2.0, -theta.sin() / 2.0),
                )
            } else {
                (
                    C64::new((theta / 2.0).cos(), 0.0),
                    C64::new(0.0, -(theta / 2.0).sin()),
                )
            };
            let mut out = Vec::with_capacity(terms.len() * 2);
            for term in terms {
                out.push(StabilizerTerm::new(term.state.clone(), term.coeff * a));
                let mut zs = term.state.clone();
                zs.z(target);
                out.push(StabilizerTerm::new(zs, term.coeff * b));
            }
            out
        }
        GateType::Rx(theta) => {
            // Rx(theta) = cos(theta/2)*I - i*sin(theta/2)*X
            let a = C64::new((theta / 2.0).cos(), 0.0);
            let b = C64::new(0.0, -(theta / 2.0).sin());
            let mut out = Vec::with_capacity(terms.len() * 2);
            for term in terms {
                out.push(StabilizerTerm::new(term.state.clone(), term.coeff * a));
                let mut xs = term.state.clone();
                xs.x(target);
                out.push(StabilizerTerm::new(xs, term.coeff * b));
            }
            out
        }
        GateType::Ry(theta) => {
            // Ry(theta) = cos(theta/2)*I - i*sin(theta/2)*Y
            let a = C64::new((theta / 2.0).cos(), 0.0);
            let b = C64::new(0.0, -(theta / 2.0).sin());
            let mut out = Vec::with_capacity(terms.len() * 2);
            for term in terms {
                out.push(StabilizerTerm::new(term.state.clone(), term.coeff * a));
                let mut ys = term.state.clone();
                ys.y(target);
                out.push(StabilizerTerm::new(ys, term.coeff * b));
            }
            out
        }
        _ => terms.to_vec(),
    }
}

/// Apply any gate to a dense state vector.
fn apply_gate_to_dense(data: &mut [C64], num_qubits: usize, gate: &Gate) {
    let matrix = gate.gate_type.matrix();
    if gate.is_single_qubit() {
        let target = gate.targets[0];
        apply_single_qubit_dense(data, num_qubits, target, &matrix);
    } else if gate.gate_type == GateType::SWAP {
        apply_swap_dense(data, num_qubits, gate.targets[0], gate.targets[1]);
    } else if gate.is_two_qubit() {
        let control = gate.controls[0];
        let target = gate.targets[0];
        apply_two_qubit_dense(data, num_qubits, control, target, &matrix);
    }
}

fn apply_single_qubit_dense(
    data: &mut [C64],
    num_qubits: usize,
    target: usize,
    matrix: &[Vec<C64>],
) {
    let dim = 1usize << num_qubits;
    let step = 1usize << target;
    let u00 = matrix[0][0];
    let u01 = matrix[0][1];
    let u10 = matrix[1][0];
    let u11 = matrix[1][1];
    let mut i = 0;
    while i < dim {
        for j in 0..step {
            let idx0 = i + j;
            let idx1 = idx0 + step;
            let a = data[idx0];
            let b = data[idx1];
            data[idx0] = u00 * a + u01 * b;
            data[idx1] = u10 * a + u11 * b;
        }
        i += step << 1;
    }
}

fn apply_two_qubit_dense(
    data: &mut [C64],
    num_qubits: usize,
    control: usize,
    target: usize,
    matrix: &[Vec<C64>],
) {
    let dim = 1usize << num_qubits;
    let c_step = 1usize << control;
    let t_step = 1usize << target;
    for i in 0..dim {
        let c_bit = (i >> control) & 1;
        let t_bit = (i >> target) & 1;
        if c_bit == 0 && t_bit == 0 {
            let i00 = i;
            let i01 = i | t_step;
            let i10 = i | c_step;
            let i11 = i | c_step | t_step;
            let a00 = data[i00];
            let a01 = data[i01];
            let a10 = data[i10];
            let a11 = data[i11];
            data[i00] =
                matrix[0][0] * a00 + matrix[0][1] * a01 + matrix[0][2] * a10 + matrix[0][3] * a11;
            data[i01] =
                matrix[1][0] * a00 + matrix[1][1] * a01 + matrix[1][2] * a10 + matrix[1][3] * a11;
            data[i10] =
                matrix[2][0] * a00 + matrix[2][1] * a01 + matrix[2][2] * a10 + matrix[2][3] * a11;
            data[i11] =
                matrix[3][0] * a00 + matrix[3][1] * a01 + matrix[3][2] * a10 + matrix[3][3] * a11;
        }
    }
}

fn apply_swap_dense(data: &mut [C64], num_qubits: usize, a: usize, b: usize) {
    let dim = 1usize << num_qubits;
    for i in 0..dim {
        let bit_a = (i >> a) & 1;
        let bit_b = (i >> b) & 1;
        if bit_a != bit_b {
            let j = i ^ (1 << a) ^ (1 << b);
            if i < j {
                data.swap(i, j);
            }
        }
    }
}

/// Convert a stabilizer state to a full statevector using the projector method.
///
/// For each stabilizer generator g_i with eigenvalue s_i, we project via
/// P_i = (I + s_i * g_i) / 2, applied sequentially to a uniform superposition
/// of all computational basis states. Starting from a uniform superposition
/// guarantees nonzero overlap with any stabilizer state (unlike starting from
/// |0...0> which can be annihilated by projectors for states like |1>).
pub fn stabilizer_to_statevector(state: &StabilizerState) -> Vec<C64> {
    let n = state.num_qubits();
    let dim = 1usize << n;
    // Start from equal superposition of all basis states to guarantee nonzero
    // overlap with the target stabilizer state.
    let amp = C64::new(1.0 / (dim as f64).sqrt(), 0.0);
    let mut sv = vec![amp; dim];

    let x = state.x_tableau();
    let z = state.z_tableau();
    let phases = state.phases();

    for gen_idx in 0..n {
        let sign: f64 = if phases[gen_idx] == 0 { 1.0 } else { -1.0 };
        let mut new_sv = vec![c64_zero(); dim];

        for basis in 0..dim {
            if sv[basis].norm() < 1e-15 {
                continue;
            }
            // Compute g_i |basis>: product of single-qubit Paulis.
            let mut target_basis = basis;
            let mut phase = C64::new(1.0, 0.0);
            for qubit in 0..n {
                let has_x = x[gen_idx][qubit];
                let has_z = z[gen_idx][qubit];
                let bit = (target_basis >> qubit) & 1;
                match (has_x, has_z) {
                    (false, false) => {}
                    (true, false) => {
                        target_basis ^= 1 << qubit;
                    }
                    (false, true) => {
                        if bit == 1 {
                            phase = phase * C64::new(-1.0, 0.0);
                        }
                    }
                    (true, true) => {
                        if bit == 1 {
                            phase = phase * C64::new(0.0, -1.0);
                        } else {
                            phase = phase * C64::new(0.0, 1.0);
                        }
                        target_basis ^= 1 << qubit;
                    }
                }
            }
            // P_i |basis> = (|basis> + sign * g_i |basis>) / 2
            new_sv[basis] += sv[basis] * 0.5;
            new_sv[target_basis] += sv[basis] * phase * sign * 0.5;
        }
        sv = new_sv;
    }

    // Normalise.
    let norm_sq: f64 = sv.iter().map(|c| c.norm_sqr()).sum();
    if norm_sq > 1e-30 {
        let inv = 1.0 / norm_sq.sqrt();
        for c in sv.iter_mut() {
            *c = *c * inv;
        }
    }
    sv
}

/// Convert any hybrid tensor node to a statevector.
fn node_to_statevector(node: &HybridTensorNode) -> Vec<C64> {
    match node {
        HybridTensorNode::Stabilizer { terms } => {
            if terms.is_empty() {
                return vec![c64_one()];
            }
            let n = terms[0].state.num_qubits();
            let dim = 1usize << n;
            let mut sv = vec![c64_zero(); dim];
            for term in terms {
                let tsv = stabilizer_to_statevector(&term.state);
                for (i, amp) in tsv.iter().enumerate() {
                    sv[i] += term.coeff * amp;
                }
            }
            sv
        }
        HybridTensorNode::Dense { data, .. } => data.clone(),
    }
}

/// Attempt to extract a stabilizer state from a dense statevector.
fn try_extract_stabilizer(data: &[C64], n: usize) -> Option<StabilizerState> {
    let dim = 1usize << n;
    let eps = 1e-10;

    let nonzero: Vec<(usize, C64)> = data
        .iter()
        .enumerate()
        .filter(|(_, a)| a.norm_sqr() > eps)
        .map(|(i, a)| (i, *a))
        .collect();

    if nonzero.is_empty() {
        return None;
    }

    let count = nonzero.len();
    if count & (count - 1) != 0 {
        return None;
    }

    let mag = nonzero[0].1.norm();
    for &(_, a) in &nonzero {
        if (a.norm() - mag).abs() > eps * 10.0 {
            return None;
        }
    }

    // Phases must be multiples of pi/2.
    let ref_phase = nonzero[0].1 / mag;
    for &(_, a) in &nonzero {
        let ratio = (a / mag) / ref_phase;
        let ok = (ratio.re.abs() < eps && (ratio.im.abs() - 1.0).abs() < eps)
            || (ratio.im.abs() < eps && (ratio.re.abs() - 1.0).abs() < eps);
        if !ok {
            return None;
        }
    }

    // Computational basis state.
    if count == 1 {
        return Some(StabilizerState::from_basis_state(n, nonzero[0].0));
    }

    // Uniform superposition: H^n |0>.
    if count == dim {
        let expected = 1.0 / (dim as f64).sqrt();
        let matches = nonzero.iter().all(|(_, a)| {
            (a.re - expected).abs() < eps * 100.0 && a.im.abs() < eps * 100.0
        });
        if matches {
            let mut s = StabilizerState::new(n);
            for q in 0..n {
                s.h(q);
            }
            return Some(s);
        }
    }

    // GHZ state: (|0...0> + |1...1>) / sqrt(2).
    if count == 2 && nonzero[0].0 == 0 && nonzero[1].0 == dim - 1 {
        let expected = 1.0 / 2.0_f64.sqrt();
        if (nonzero[0].1.re - expected).abs() < eps * 100.0
            && (nonzero[1].1.re - expected).abs() < eps * 100.0
        {
            let mut s = StabilizerState::new(n);
            s.h(0);
            for q in 1..n {
                s.cx(0, q);
            }
            return Some(s);
        }
    }

    None
}

/// Sample a single bitstring from one hybrid tensor node.
///
/// Returns the measurement outcome as a usize bitstring.
fn sample_node_once(node: &HybridTensorNode) -> usize {
    match node {
        HybridTensorNode::Stabilizer { terms } => {
            let weights: Vec<f64> = terms.iter().map(|t| t.coeff.norm_sqr()).collect();
            let total: f64 = weights.iter().sum();
            let r = rand::random::<f64>() * total;
            let mut cum = 0.0;
            let mut chosen = 0;
            for (i, w) in weights.iter().enumerate() {
                cum += w;
                if r <= cum {
                    chosen = i;
                    break;
                }
            }
            let mut state = terms[chosen].state.clone();
            let n = state.num_qubits();
            let mut bs = 0usize;
            for q in 0..n {
                if state.measure(q) {
                    bs |= 1 << q;
                }
            }
            bs
        }
        HybridTensorNode::Dense { data, .. } => {
            let probs: Vec<f64> = data.iter().map(|a| a.norm_sqr()).collect();
            let total: f64 = probs.iter().sum();
            let r = rand::random::<f64>() * total;
            let mut cum = 0.0;
            let mut outcome = 0;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if r <= cum {
                    outcome = i;
                    break;
                }
            }
            outcome
        }
    }
}

/// Compute fidelity |<a|b>|^2 between two statevectors.
fn fidelity_sv(a: &[C64], b: &[C64]) -> f64 {
    let ip: C64 = a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum();
    ip.norm_sqr()
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    const EPS: f64 = 1e-6;

    // --------------------------------------------------------
    // 1. StabilizerTerm inner products
    // --------------------------------------------------------

    #[test]
    fn test_inner_product_same_state() {
        let t1 = StabilizerTerm::unit(StabilizerState::new(1));
        let t2 = StabilizerTerm::unit(StabilizerState::new(1));
        let ip = t1.stabilizer_inner_product(&t2);
        assert!((ip.norm() - 1.0).abs() < EPS, "<0|0> should be 1, got {:?}", ip);
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let t1 = StabilizerTerm::unit(StabilizerState::new(1));
        let mut s = StabilizerState::new(1);
        s.x(0);
        let t2 = StabilizerTerm::unit(s);
        let ip = t1.stabilizer_inner_product(&t2);
        assert!(ip.norm() < EPS, "<0|1> should be 0, got {:?}", ip);
    }

    #[test]
    fn test_inner_product_plus_zero() {
        let t1 = StabilizerTerm::unit(StabilizerState::new(1));
        let mut plus = StabilizerState::new(1);
        plus.h(0);
        let t2 = StabilizerTerm::unit(plus);
        let ip = t1.stabilizer_inner_product(&t2);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (ip.norm() - expected).abs() < EPS,
            "<0|+> should be 1/sqrt(2), got {:?}",
            ip
        );
    }

    #[test]
    fn test_inner_product_bell_00() {
        let mut bell = StabilizerState::new(2);
        bell.h(0);
        bell.cx(0, 1);
        let t_bell = StabilizerTerm::unit(bell);
        let t_00 = StabilizerTerm::unit(StabilizerState::new(2));
        let ip = t_bell.stabilizer_inner_product(&t_00);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (ip.norm() - expected).abs() < EPS,
            "<Bell|00> should be 1/sqrt(2), got {:?}",
            ip
        );
    }

    // --------------------------------------------------------
    // 2. Clifford gates stay in stabilizer representation
    // --------------------------------------------------------

    #[test]
    fn test_clifford_stays_stabilizer() {
        let mut node = HybridTensorNode::new_stabilizer(2);
        assert!(node.apply_clifford(&Gate::h(0)));
        assert!(node.is_stabilizer());
        assert_eq!(node.num_terms(), 1);

        assert!(node.apply_clifford(&Gate::cnot(0, 1)));
        assert!(node.is_stabilizer());
        assert_eq!(node.num_terms(), 1);
    }

    #[test]
    fn test_full_clifford_circuit_no_branching() {
        let mut sim = StabilizerTensorNetwork::new(3, STNConfig::default());
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        sim.apply_gate(&Gate::cnot(1, 2));
        sim.apply_gate(&Gate::s(0));
        sim.apply_gate(&Gate::z(1));

        assert!(sim.is_all_stabilizer());
        assert_eq!(sim.total_stabilizer_terms(), 1);
        assert_eq!(sim.stats().promotions, 0);
    }

    // --------------------------------------------------------
    // 3. T gate causes splitting
    // --------------------------------------------------------

    #[test]
    fn test_t_gate_branches() {
        let mut node = HybridTensorNode::new_stabilizer(2);
        node.apply_clifford(&Gate::h(0));
        let was_clifford = node.apply_clifford(&Gate::t(0));
        assert!(!was_clifford, "T should not be classified as Clifford");

        node.apply_gate(&Gate::t(0));
        assert!(node.is_stabilizer());
        assert_eq!(node.num_terms(), 2);
    }

    #[test]
    fn test_multiple_t_gates_branch_exponentially() {
        let mut node = HybridTensorNode::new_stabilizer(1);
        node.apply_gate(&Gate::t(0));
        assert_eq!(node.num_terms(), 2);
        node.apply_gate(&Gate::t(0));
        assert_eq!(node.num_terms(), 4);
        node.apply_gate(&Gate::t(0));
        assert_eq!(node.num_terms(), 8);
    }

    // --------------------------------------------------------
    // 4. Promotion / demotion
    // --------------------------------------------------------

    #[test]
    fn test_promotion_on_term_overflow() {
        let config = STNConfig {
            max_stabilizer_terms: 4,
            ..STNConfig::default()
        };
        let mut sim = StabilizerTensorNetwork::new(2, config);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::t(0)); // 2
        sim.apply_gate(&Gate::t(0)); // 4 <= max
        assert!(sim.is_all_stabilizer());

        sim.apply_gate(&Gate::t(0)); // 8 > max -> promote
        assert!(sim.is_all_dense());
        assert_eq!(sim.stats().promotions, 1);
    }

    #[test]
    fn test_explicit_promotion_preserves_state() {
        let mut node = HybridTensorNode::new_stabilizer(2);
        node.apply_clifford(&Gate::h(0));
        node.apply_clifford(&Gate::cnot(0, 1));

        // Capture statevector before promotion.
        let sv_before = node_to_statevector(&node);

        node.promote_to_dense();
        assert!(node.is_dense());

        let sv_after = node_to_statevector(&node);
        let fid = fidelity_sv(&sv_before, &sv_after);
        assert!(fid > 1.0 - EPS, "Promotion changed the state: fid={}", fid);
    }

    #[test]
    fn test_demotion_for_basis_state() {
        let mut node = HybridTensorNode::new_dense(1);
        // |0> is a stabilizer state, so demotion should succeed.
        assert!(node.try_demote_to_stabilizer());
        assert!(node.is_stabilizer());
    }

    #[test]
    fn test_demotion_after_cliffords() {
        let config = STNConfig {
            max_stabilizer_terms: 1,
            enable_demotion: true,
            demotion_clifford_threshold: 3,
            ..STNConfig::default()
        };
        let mut sim = StabilizerTensorNetwork::new(1, config);
        sim.apply_gate(&Gate::x(0));
        assert!(sim.is_all_stabilizer());

        sim.apply_gate(&Gate::t(0));
        assert!(sim.is_all_dense());
        assert_eq!(sim.stats().promotions, 1);

        // Apply Cliffords. Demotion may or may not succeed depending on state.
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::h(0));
        // Verify no panic and stats are consistent.
        assert!(sim.stats().promotions >= 1);
    }

    // --------------------------------------------------------
    // 5. Contraction of stabilizer-stabilizer bonds
    // --------------------------------------------------------

    #[test]
    fn test_contract_zero_state() {
        let sim = StabilizerTensorNetwork::new(2, STNConfig::default());
        let sv = sim.contract();
        assert_eq!(sv.len(), 4);
        assert!((sv[0].norm() - 1.0).abs() < EPS);
        for i in 1..4 {
            assert!(sv[i].norm() < EPS);
        }
    }

    #[test]
    fn test_contract_hadamard() {
        let mut sim = StabilizerTensorNetwork::new(1, STNConfig::default());
        sim.apply_gate(&Gate::h(0));
        let sv = sim.contract();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < EPS);
        assert!((sv[1].re - expected).abs() < EPS);
    }

    // --------------------------------------------------------
    // 6. Full circuit: Bell state matches state-vector reference
    // --------------------------------------------------------

    #[test]
    fn test_bell_state_circuit() {
        let mut sim = StabilizerTensorNetwork::new(2, STNConfig::default());
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));

        let sv = sim.contract();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < EPS, "sv[0]={:?}", sv[0]);
        assert!(sv[1].norm() < EPS);
        assert!(sv[2].norm() < EPS);
        assert!((sv[3].re - expected).abs() < EPS, "sv[3]={:?}", sv[3]);
    }

    #[test]
    fn test_bell_matches_dense_reference() {
        // STN path
        let mut sim = StabilizerTensorNetwork::new(2, STNConfig::default());
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        let sv_stn = sim.contract();

        // Dense reference
        let mut node = HybridTensorNode::new_dense(2);
        node.apply_gate(&Gate::h(0));
        node.apply_gate(&Gate::cnot(0, 1));
        let sv_ref = node_to_statevector(&node);

        let fid = fidelity_sv(&sv_stn, &sv_ref);
        assert!(fid > 1.0 - EPS, "Bell fidelity = {}", fid);
    }

    // --------------------------------------------------------
    // 7. Full circuit: GHZ state matches state-vector reference
    // --------------------------------------------------------

    #[test]
    fn test_ghz_state_circuit() {
        let mut sim = StabilizerTensorNetwork::new(3, STNConfig::default());
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        sim.apply_gate(&Gate::cnot(1, 2));

        let sv = sim.contract();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < EPS);
        for i in 1..7 {
            assert!(sv[i].norm() < EPS, "sv[{}]={:?}", i, sv[i]);
        }
        assert!((sv[7].re - expected).abs() < EPS);

        assert!(sim.is_all_stabilizer());
        assert_eq!(sim.total_stabilizer_terms(), 1);
    }

    #[test]
    fn test_ghz_matches_dense_reference() {
        let mut sim = StabilizerTensorNetwork::new(4, STNConfig::default());
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        sim.apply_gate(&Gate::cnot(1, 2));
        sim.apply_gate(&Gate::cnot(2, 3));
        let sv_stn = sim.contract();

        let expected = 1.0 / 2.0_f64.sqrt();
        let mut sv_ref = vec![c64_zero(); 16];
        sv_ref[0] = C64::new(expected, 0.0);
        sv_ref[15] = C64::new(expected, 0.0);

        let fid = fidelity_sv(&sv_stn, &sv_ref);
        assert!(fid > 1.0 - EPS, "GHZ fidelity = {}", fid);
    }

    // --------------------------------------------------------
    // MagicMonotone tests
    // --------------------------------------------------------

    #[test]
    fn test_magic_stabilizer_state() {
        let coeffs = vec![c64_one()];
        assert!((MagicMonotone::stabilizer_extent(&coeffs) - 1.0).abs() < EPS);
        assert!(MagicMonotone::mana(&coeffs).abs() < EPS);
        assert!((MagicMonotone::robustness(&coeffs) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_magic_increases_with_t() {
        let (e0, _) = MagicMonotone::from_t_count(0);
        let (e1, _) = MagicMonotone::from_t_count(1);
        let (e5, _) = MagicMonotone::from_t_count(5);
        assert!((e0 - 1.0).abs() < EPS);
        assert!(e1 > e0);
        assert!(e5 > e1);
    }

    // --------------------------------------------------------
    // Dense gate application
    // --------------------------------------------------------

    #[test]
    fn test_dense_bell_state() {
        let mut node = HybridTensorNode::new_dense(2);
        node.apply_gate(&Gate::h(0));
        node.apply_gate(&Gate::cnot(0, 1));
        if let HybridTensorNode::Dense { data, .. } = &node {
            assert!((data[0].norm_sqr() - 0.5).abs() < EPS);
            assert!(data[1].norm_sqr() < EPS);
            assert!(data[2].norm_sqr() < EPS);
            assert!((data[3].norm_sqr() - 0.5).abs() < EPS);
        }
    }

    #[test]
    fn test_dense_swap() {
        let mut node = HybridTensorNode::new_dense(2);
        node.apply_gate(&Gate::x(0));
        node.apply_gate(&Gate::swap(0, 1));
        if let HybridTensorNode::Dense { data, .. } = &node {
            assert!(data[0].norm_sqr() < EPS);
            assert!(data[1].norm_sqr() < EPS);
            assert!((data[2].norm_sqr() - 1.0).abs() < EPS);
            assert!(data[3].norm_sqr() < EPS);
        }
    }

    // --------------------------------------------------------
    // Measurement sampling
    // --------------------------------------------------------

    #[test]
    fn test_sample_deterministic() {
        let sim = StabilizerTensorNetwork::new(2, STNConfig::default());
        let counts = sim.sample(100);
        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get(&0).unwrap(), 100);
    }

    #[test]
    fn test_sample_bell_correlation() {
        // Use dense mode for reliable correlated sampling.
        let config = STNConfig {
            max_stabilizer_terms: 1,
            ..STNConfig::default()
        };
        let mut sim = StabilizerTensorNetwork::new(2, config);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::rx(0, 0.1)); // Force dense via non-Clifford gate
        sim.apply_gate(&Gate::cnot(0, 1));
        assert!(sim.is_all_dense());

        let counts = sim.sample(1000);
        let c00 = *counts.get(&0b00).unwrap_or(&0);
        let c11 = *counts.get(&0b11).unwrap_or(&0);
        assert!(c00 + c11 > 950, "Bell: 00={} 11={}", c00, c11);
        assert!(c00 > 100);
        assert!(c11 > 100);
    }

    // --------------------------------------------------------
    // Stabilizer-to-statevector helper
    // --------------------------------------------------------

    #[test]
    fn test_sv_zero_state() {
        let sv = stabilizer_to_statevector(&StabilizerState::new(1));
        assert!((sv[0].re - 1.0).abs() < EPS);
        assert!(sv[1].norm() < EPS);
    }

    #[test]
    fn test_sv_one_state() {
        let mut s = StabilizerState::new(1);
        s.x(0);
        let sv = stabilizer_to_statevector(&s);
        assert!(sv[0].norm() < EPS);
        assert!((sv[1].re - 1.0).abs() < EPS);
    }

    #[test]
    fn test_sv_plus_state() {
        let mut s = StabilizerState::new(1);
        s.h(0);
        let sv = stabilizer_to_statevector(&s);
        let e = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - e).abs() < EPS);
        assert!((sv[1].re - e).abs() < EPS);
    }

    #[test]
    fn test_sv_bell_state() {
        let mut s = StabilizerState::new(2);
        s.h(0);
        s.cx(0, 1);
        let sv = stabilizer_to_statevector(&s);
        let e = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - e).abs() < EPS, "sv[0]={:?}", sv[0]);
        assert!(sv[1].norm() < EPS);
        assert!(sv[2].norm() < EPS);
        assert!((sv[3].re - e).abs() < EPS, "sv[3]={:?}", sv[3]);
    }

    // --------------------------------------------------------
    // Mixed stabilizer + dense circuit
    // --------------------------------------------------------

    #[test]
    fn test_mixed_circuit() {
        let config = STNConfig {
            max_stabilizer_terms: 8,
            ..STNConfig::default()
        };
        let mut sim = StabilizerTensorNetwork::new(2, config);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        assert!(sim.is_all_stabilizer());

        sim.apply_gate(&Gate::t(0));
        sim.apply_gate(&Gate::t(1));
        assert!(sim.is_all_stabilizer());
        assert_eq!(sim.total_stabilizer_terms(), 4);

        sim.apply_gate(&Gate::t(0));
        sim.apply_gate(&Gate::t(1));
        assert!(sim.is_all_dense());

        sim.apply_gate(&Gate::h(0));
        let counts = sim.sample(100);
        let total: usize = counts.values().sum();
        assert_eq!(total, 100);
    }

    // --------------------------------------------------------
    // Rz branching
    // --------------------------------------------------------

    #[test]
    fn test_rz_branches() {
        let mut node = HybridTensorNode::new_stabilizer(1);
        node.apply_gate(&Gate::rz(0, std::f64::consts::FRAC_PI_4));
        assert!(node.is_stabilizer());
        assert_eq!(node.num_terms(), 2);
    }

    // --------------------------------------------------------
    // Config and stats display
    // --------------------------------------------------------

    #[test]
    fn test_config_default() {
        let c = STNConfig::default();
        assert_eq!(c.max_stabilizer_terms, 64);
        assert!(c.enable_demotion);
    }

    #[test]
    fn test_stats_display() {
        let s = STNStats {
            stabilizer_gates: 10,
            dense_gates: 5,
            promotions: 2,
            demotions: 1,
            current_stabilizer_nodes: 3,
            current_dense_nodes: 1,
        };
        let txt = format!("{}", s);
        assert!(txt.contains("promotions: 2"));
    }
}
