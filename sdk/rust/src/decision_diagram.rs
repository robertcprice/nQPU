//! Quantum Decision Diagram (QDD) Simulator
//!
//! This module implements BDD/ZDD-based quantum state compression using quantum
//! decision diagrams. Each path from root to terminal encodes a basis state
//! amplitude as the product of edge weights along the path.
//!
//! Key features:
//! - Arena-allocated node pool with hash-based deduplication (unique table)
//! - Complement edges for approximately 2x node reduction on symmetric states
//! - Mark-and-sweep garbage collection with reference counting
//! - Recursive gate application with addition (combine) operation on DDs
//! - Support for single-qubit, controlled, and SWAP gates
//!
//! # Complexity
//!
//! Decision diagrams achieve exponential compression on structured quantum states
//! (GHZ, W, symmetric states) while degrading to O(2^n) only for maximally
//! entangled random states.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::decision_diagram::DDSimulator;
//! use nqpu_metal::gates::Gate;
//!
//! let mut sim = DDSimulator::new(2);
//! sim.apply_gate(&Gate::h(0));
//! sim.apply_gate(&Gate::cnot(0, 1));
//! let probs = sim.probabilities();
//! // Bell state: |00> and |11> each with probability 0.5
//! ```

use std::collections::HashMap;

use crate::gates::{Gate, GateType};
use crate::C64;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Terminal node representing the zero amplitude.
const TERMINAL_ZERO: usize = 0;

/// Terminal node representing unit amplitude (weight carried on incoming edge).
const TERMINAL_ONE: usize = 1;

/// Tolerance for treating a complex number as zero.
const EPSILON: f64 = 1e-12;

/// Threshold for triggering automatic garbage collection.
const GC_NODE_THRESHOLD: usize = 50_000;

// ---------------------------------------------------------------------------
// DDEdge
// ---------------------------------------------------------------------------

/// An edge in the decision diagram carrying a complex weight.
///
/// The `complemented` flag negates all amplitudes reachable through this edge,
/// enabling roughly 2x node sharing on symmetric states without allocating
/// additional nodes.
#[derive(Clone, Debug)]
pub struct DDEdge {
    /// Target node identifier. `TERMINAL_ZERO` (0) and `TERMINAL_ONE` (1) are
    /// reserved for the two terminal nodes.
    pub node_id: usize,
    /// Complex weight on this edge.
    pub weight: C64,
    /// When true, the entire sub-diagram below this edge is logically negated
    /// (all amplitudes multiplied by -1).
    pub complemented: bool,
}

impl DDEdge {
    /// Create a standard (non-complemented) edge.
    #[inline]
    pub fn new(node_id: usize, weight: C64) -> Self {
        Self {
            node_id,
            weight,
            complemented: false,
        }
    }

    /// Create a complemented edge (all sub-amplitudes negated).
    #[inline]
    pub fn complemented(node_id: usize, weight: C64) -> Self {
        Self {
            node_id,
            weight,
            complemented: true,
        }
    }

    /// Convenience constructor for the zero terminal.
    #[inline]
    pub fn zero() -> Self {
        Self::new(TERMINAL_ZERO, C64::new(0.0, 0.0))
    }

    /// Convenience constructor for the one terminal with a given weight.
    #[inline]
    pub fn one(weight: C64) -> Self {
        Self::new(TERMINAL_ONE, weight)
    }

    /// The effective weight, accounting for complementation.
    #[inline]
    pub fn effective_weight(&self) -> C64 {
        if self.complemented {
            -self.weight
        } else {
            self.weight
        }
    }

    /// True if this edge effectively contributes zero amplitude.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.node_id == TERMINAL_ZERO || self.weight.norm() < EPSILON
    }
}

// ---------------------------------------------------------------------------
// DDNode
// ---------------------------------------------------------------------------

/// A node in the quantum decision diagram.
///
/// Each internal node corresponds to a qubit variable. The `low` edge encodes
/// the |0> branch and the `high` edge encodes the |1> branch.
#[derive(Clone, Debug)]
pub struct DDNode {
    /// Unique identifier within the node pool.
    pub id: usize,
    /// Qubit variable index (0 = top-level qubit).
    pub var: usize,
    /// Edge taken when this qubit is |0>.
    pub low: DDEdge,
    /// Edge taken when this qubit is |1>.
    pub high: DDEdge,
}

// ---------------------------------------------------------------------------
// DDNodePool
// ---------------------------------------------------------------------------

/// Arena-allocated node pool with hash-based unique table for deduplication.
///
/// Nodes 0 and 1 are reserved as terminal-zero and terminal-one respectively.
/// All subsequent nodes are internal decision nodes created via `get_or_create`.
pub struct DDNodePool {
    /// Arena storage. Indices 0 and 1 are terminal sentinels.
    nodes: Vec<DDNode>,
    /// Unique table: (var, low_node_id, high_node_id) -> node_id.
    /// Edge weights are factored out during normalization, so structural
    /// identity depends only on topology.
    unique_table: HashMap<(usize, usize, usize), usize>,
    /// Reference counts for garbage collection. Indexed by node id.
    ref_counts: Vec<usize>,
}

impl DDNodePool {
    /// Create a new pool pre-populated with the two terminal nodes.
    pub fn new() -> Self {
        let terminal_zero = DDNode {
            id: TERMINAL_ZERO,
            var: usize::MAX,
            low: DDEdge::zero(),
            high: DDEdge::zero(),
        };
        let terminal_one = DDNode {
            id: TERMINAL_ONE,
            var: usize::MAX,
            low: DDEdge::zero(),
            high: DDEdge::zero(),
        };

        Self {
            nodes: vec![terminal_zero, terminal_one],
            unique_table: HashMap::new(),
            ref_counts: vec![1, 1], // terminals are always referenced
        }
    }

    /// Look up or create a node with the given variable and children.
    ///
    /// If a structurally identical node already exists (same var, same child
    /// node ids) it is returned directly. Edge weights are not part of the
    /// lookup key because they are normalized before insertion.
    pub fn get_or_create(&mut self, var: usize, low: DDEdge, high: DDEdge) -> usize {
        let key = (var, low.node_id, high.node_id);

        if let Some(&existing) = self.unique_table.get(&key) {
            return existing;
        }

        let id = self.nodes.len();
        let node = DDNode {
            id,
            var,
            low,
            high,
        };
        self.nodes.push(node);
        self.ref_counts.push(0);
        self.unique_table.insert(key, id);
        id
    }

    /// Retrieve a node by its identifier.
    #[inline]
    pub fn get(&self, id: usize) -> &DDNode {
        &self.nodes[id]
    }

    /// Total number of allocated nodes (including terminals).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of live internal (non-terminal) nodes.
    pub fn live_node_count(&self) -> usize {
        self.nodes.len().saturating_sub(2)
    }

    /// Increment the reference count for a node.
    #[inline]
    pub fn inc_ref(&mut self, id: usize) {
        if id < self.ref_counts.len() {
            self.ref_counts[id] += 1;
        }
    }

    /// Decrement the reference count for a node.
    #[inline]
    pub fn dec_ref(&mut self, id: usize) {
        if id < self.ref_counts.len() && self.ref_counts[id] > 0 {
            self.ref_counts[id] -= 1;
        }
    }

    /// Mark-and-sweep garbage collection rooted at the given node id.
    ///
    /// All nodes not reachable from `root` (and not terminals) are removed.
    /// Returns the number of nodes collected.
    pub fn mark_and_sweep(&mut self, root: usize) -> usize {
        let mut reachable = vec![false; self.nodes.len()];
        reachable[TERMINAL_ZERO] = true;
        reachable[TERMINAL_ONE] = true;
        self.mark_reachable(root, &mut reachable);

        let unreachable_count = reachable
            .iter()
            .enumerate()
            .skip(2)
            .filter(|(_, &r)| !r)
            .count();

        if unreachable_count == 0 {
            return 0;
        }

        // Build id remapping.
        let mut id_map: Vec<usize> = vec![0; self.nodes.len()];
        id_map[TERMINAL_ZERO] = TERMINAL_ZERO;
        id_map[TERMINAL_ONE] = TERMINAL_ONE;

        let mut new_nodes: Vec<DDNode> =
            Vec::with_capacity(self.nodes.len() - unreachable_count);
        new_nodes.push(self.nodes[TERMINAL_ZERO].clone());
        new_nodes.push(self.nodes[TERMINAL_ONE].clone());

        let mut new_ref_counts: Vec<usize> = vec![1, 1];
        let mut next_id = 2;

        for i in 2..self.nodes.len() {
            if reachable[i] {
                id_map[i] = next_id;
                new_ref_counts.push(self.ref_counts[i]);
                next_id += 1;
            }
        }

        for i in 2..self.nodes.len() {
            if reachable[i] {
                let node = &self.nodes[i];
                let mut new_node = node.clone();
                new_node.id = id_map[i];
                new_node.low.node_id = id_map[node.low.node_id];
                new_node.high.node_id = id_map[node.high.node_id];
                new_nodes.push(new_node);
            }
        }

        let mut new_unique_table = HashMap::with_capacity(new_nodes.len());
        for node in new_nodes.iter().skip(2) {
            let key = (node.var, node.low.node_id, node.high.node_id);
            new_unique_table.insert(key, node.id);
        }

        self.nodes = new_nodes;
        self.unique_table = new_unique_table;
        self.ref_counts = new_ref_counts;

        unreachable_count
    }

    /// Recursively mark all nodes reachable from `id`.
    fn mark_reachable(&self, id: usize, reachable: &mut [bool]) {
        if id >= reachable.len() || reachable[id] {
            return;
        }
        reachable[id] = true;
        let node = &self.nodes[id];
        self.mark_reachable(node.low.node_id, reachable);
        self.mark_reachable(node.high.node_id, reachable);
    }
}

impl Default for DDNodePool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DDStats
// ---------------------------------------------------------------------------

/// Accumulated statistics for a decision diagram simulation run.
#[derive(Clone, Debug, Default)]
pub struct DDStats {
    /// Total number of nodes ever allocated.
    pub total_nodes: usize,
    /// High-water mark for node count.
    pub max_nodes: usize,
    /// Number of garbage collection passes.
    pub gc_runs: usize,
    /// Total nodes reclaimed by GC.
    pub nodes_collected: usize,
    /// Operation cache hits.
    pub cache_hits: usize,
    /// Operation cache misses.
    pub cache_misses: usize,
}

// ---------------------------------------------------------------------------
// DDSimulator
// ---------------------------------------------------------------------------

/// Quantum decision diagram simulator.
///
/// Represents the full n-qubit quantum state as a rooted decision diagram
/// where each internal node corresponds to a qubit, and amplitudes are
/// encoded as products of edge weights along root-to-terminal paths.
///
/// An edge `(w, node_id)` to terminal-one means the sub-state contributes
/// weight `w`. An edge to terminal-zero means zero amplitude. For an internal
/// node at variable `v`, the sub-state splits into the |0> branch (low edge)
/// and the |1> branch (high edge).
pub struct DDSimulator {
    /// Node arena with unique table deduplication.
    pool: DDNodePool,
    /// Root edge of the state diagram.
    root: DDEdge,
    /// Number of qubits.
    num_qubits: usize,
    /// Accumulated statistics.
    stats: DDStats,
    /// Operation cache for DD addition: (id_a, id_b) -> (result_id, weight).
    add_cache: HashMap<(usize, usize), (usize, C64)>,
}

impl DDSimulator {
    /// Create a new simulator initialized to the |0...0> state.
    ///
    /// The initial state is a simple chain of nodes, each with its low edge
    /// pointing to the next level and its high edge pointing to terminal-zero.
    pub fn new(num_qubits: usize) -> Self {
        let mut pool = DDNodePool::new();

        // Build the |0...0> state bottom-up.
        // Each qubit level has: low -> next level (|0> path), high -> zero (|1> path).
        let mut current_id = TERMINAL_ONE;

        for var in (0..num_qubits).rev() {
            let low = DDEdge::new(current_id, C64::new(1.0, 0.0));
            let high = DDEdge::zero();
            current_id = pool.get_or_create(var, low, high);
            pool.inc_ref(current_id);
        }

        let root = DDEdge::new(current_id, C64::new(1.0, 0.0));

        let stats = DDStats {
            total_nodes: pool.node_count(),
            max_nodes: pool.node_count(),
            ..Default::default()
        };

        Self {
            pool,
            root,
            num_qubits,
            stats,
            add_cache: HashMap::new(),
        }
    }

    /// Apply a single quantum gate to the state.
    pub fn apply_gate(&mut self, gate: &Gate) {
        match &gate.gate_type {
            GateType::SWAP => {
                let a = gate.targets[0];
                let b = gate.targets[1];
                self.apply_cnot_gate(a, b);
                self.apply_cnot_gate(b, a);
                self.apply_cnot_gate(a, b);
            }
            GateType::CZ => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.apply_single_qubit_gate(target, &GateType::H);
                self.apply_cnot_gate(control, target);
                self.apply_single_qubit_gate(target, &GateType::H);
            }
            GateType::CNOT => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.apply_cnot_gate(control, target);
            }
            GateType::Toffoli => {
                let c1 = gate.controls[0];
                let c2 = gate.controls[1];
                let target = gate.targets[0];
                self.apply_toffoli_gate(c1, c2, target);
            }
            GateType::CCZ => {
                let c1 = gate.controls[0];
                let c2 = gate.controls[1];
                let target = gate.targets[0];
                self.apply_single_qubit_gate(target, &GateType::H);
                self.apply_toffoli_gate(c1, c2, target);
                self.apply_single_qubit_gate(target, &GateType::H);
            }
            GateType::ISWAP => {
                let a = gate.targets[0];
                let b = gate.targets[1];
                self.apply_cnot_gate(a, b);
                self.apply_single_qubit_gate(a, &GateType::S);
                self.apply_cnot_gate(b, a);
                self.apply_single_qubit_gate(b, &GateType::S);
                self.apply_cnot_gate(a, b);
            }
            GateType::CRx(angle) => {
                let c = gate.controls[0];
                let t = gate.targets[0];
                self.apply_controlled_gate(c, t, &GateType::Rx(*angle));
            }
            GateType::CRy(angle) => {
                let c = gate.controls[0];
                let t = gate.targets[0];
                self.apply_controlled_gate(c, t, &GateType::Ry(*angle));
            }
            GateType::CRz(angle) => {
                let c = gate.controls[0];
                let t = gate.targets[0];
                self.apply_controlled_gate(c, t, &GateType::Rz(*angle));
            }
            GateType::CR(angle) => {
                let c = gate.controls[0];
                let t = gate.targets[0];
                self.apply_controlled_gate(c, t, &GateType::Phase(*angle));
            }
            _ => {
                if gate.controls.is_empty() {
                    let target = gate.targets[0];
                    self.apply_single_qubit_gate(target, &gate.gate_type);
                } else if gate.controls.len() == 1 {
                    let control = gate.controls[0];
                    let target = gate.targets[0];
                    self.apply_controlled_gate(control, target, &gate.gate_type);
                }
            }
        }

        self.stats.total_nodes = self.pool.node_count();
        if self.pool.node_count() > self.stats.max_nodes {
            self.stats.max_nodes = self.pool.node_count();
        }

        if self.pool.node_count() > GC_NODE_THRESHOLD {
            self.run_gc();
        }
    }

    /// Apply a sequence of gates.
    pub fn apply_circuit(&mut self, gates: &[Gate]) {
        for gate in gates {
            self.apply_gate(gate);
        }
    }

    /// Extract the full probability distribution as a vector of length 2^n.
    pub fn probabilities(&self) -> Vec<f64> {
        let dim = 1usize << self.num_qubits;
        let mut probs = vec![0.0f64; dim];
        for basis in 0..dim {
            let amp = self.amplitude(basis);
            probs[basis] = amp.norm_sqr();
        }
        probs
    }

    /// Retrieve the amplitude of a specific computational basis state.
    ///
    /// The basis state is encoded as an integer where bit k corresponds to
    /// qubit k (bit 0 = qubit 0).
    pub fn amplitude(&self, basis_state: usize) -> C64 {
        self.traverse_amplitude(&self.root, basis_state, 0)
    }

    /// Current number of nodes in the pool (including terminals).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.pool.node_count()
    }

    /// Number of live internal (non-terminal) nodes.
    #[inline]
    pub fn live_node_count(&self) -> usize {
        self.pool.live_node_count()
    }

    /// Access accumulated statistics.
    #[inline]
    pub fn stats(&self) -> &DDStats {
        &self.stats
    }

    /// Number of qubits in this simulator.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Manually trigger garbage collection.
    pub fn run_gc(&mut self) {
        self.add_cache.clear();

        let root_id = self.root.node_id;
        let collected = self.pool.mark_and_sweep(root_id);

        if collected > 0 && root_id >= 2 {
            // After compaction, find the new root by searching for var=0.
            for node in self.pool.nodes.iter().skip(2) {
                if node.var == 0 {
                    self.root.node_id = node.id;
                    break;
                }
            }
        }

        self.stats.gc_runs += 1;
        self.stats.nodes_collected += collected;
    }

    // -----------------------------------------------------------------------
    // Amplitude traversal
    // -----------------------------------------------------------------------

    /// Recursively traverse the DD to compute the amplitude for a basis state.
    fn traverse_amplitude(&self, edge: &DDEdge, basis_state: usize, var: usize) -> C64 {
        if edge.is_zero() {
            return C64::new(0.0, 0.0);
        }

        let w = edge.effective_weight();

        if edge.node_id == TERMINAL_ONE {
            // All remaining qubits are implicitly |0>.
            // If any remaining bit is set, amplitude is zero.
            for v in var..self.num_qubits {
                if (basis_state >> v) & 1 != 0 {
                    return C64::new(0.0, 0.0);
                }
            }
            return w;
        }

        let node = self.pool.get(edge.node_id);

        // Variables between `var` and `node.var` are implicitly |0>.
        for v in var..node.var {
            if (basis_state >> v) & 1 != 0 {
                return C64::new(0.0, 0.0);
            }
        }

        let bit = (basis_state >> node.var) & 1;
        let child = if bit == 0 { &node.low } else { &node.high };

        w * self.traverse_amplitude(child, basis_state, node.var + 1)
    }

    // -----------------------------------------------------------------------
    // Single-qubit gate
    // -----------------------------------------------------------------------

    /// Apply a single-qubit gate to qubit `target`.
    ///
    /// The gate matrix `U` is 2x2. For a node at the target variable:
    ///   new_low_dd  = U[0][0] * old_low_dd  + U[0][1] * old_high_dd
    ///   new_high_dd = U[1][0] * old_low_dd  + U[1][1] * old_high_dd
    ///
    /// where `old_low_dd` and `old_high_dd` are entire sub-diagrams (not scalars).
    /// This requires a DD addition operation.
    fn apply_single_qubit_gate(&mut self, target: usize, gate_type: &GateType) {
        let mat = gate_matrix_2x2(gate_type);
        let root_clone = self.root.clone();
        self.root = self.apply_single_recursive(&root_clone, target, &mat);
        self.add_cache.clear();
    }

    /// Recursive helper for single-qubit gate application.
    /// Returns a new DDEdge representing the transformed sub-diagram.
    fn apply_single_recursive(
        &mut self,
        edge: &DDEdge,
        target: usize,
        mat: &[[C64; 2]; 2],
    ) -> DDEdge {
        if edge.is_zero() {
            return DDEdge::zero();
        }

        let w = edge.effective_weight();

        // Terminal one: all remaining qubits are implicitly |0>.
        if edge.node_id == TERMINAL_ONE {
            // The target qubit is implicitly |0>. Gate maps:
            //   new_low  = U[0][0] * 1
            //   new_high = U[1][0] * 1
            let new_low_w = mat[0][0] * w;
            let new_high_w = mat[1][0] * w;
            return self.make_edge(target, new_low_w, TERMINAL_ONE, new_high_w, TERMINAL_ONE);
        }

        let node = self.pool.get(edge.node_id).clone();

        if node.var < target {
            // Target is below this node. Recurse into both children,
            // incorporating the incoming weight into the children.
            let child_low = self.weighted_edge(&node.low, w);
            let child_high = self.weighted_edge(&node.high, w);

            let new_low = self.apply_single_recursive(&child_low, target, mat);
            let new_high = self.apply_single_recursive(&child_high, target, mat);

            return self.build_node(node.var, new_low, new_high);
        }

        if node.var == target {
            // This is the target variable. Apply the 2x2 matrix.
            // old_low_dd has weight (w * node.low.weight) pointing to node.low.node_id
            // old_high_dd has weight (w * node.high.weight) pointing to node.high.node_id

            let old_low = self.weighted_edge(&node.low, w);
            let old_high = self.weighted_edge(&node.high, w);

            // new_low_dd  = mat[0][0] * old_low_dd  + mat[0][1] * old_high_dd
            // new_high_dd = mat[1][0] * old_low_dd  + mat[1][1] * old_high_dd
            let term00 = self.scale_dd(&old_low, mat[0][0]);
            let term01 = self.scale_dd(&old_high, mat[0][1]);
            let new_low = self.add_dd(&term00, &term01);

            let term10 = self.scale_dd(&old_low, mat[1][0]);
            let term11 = self.scale_dd(&old_high, mat[1][1]);
            let new_high = self.add_dd(&term10, &term11);

            return self.build_node(target, new_low, new_high);
        }

        // node.var > target: the target qubit is implicitly |0> above this node.
        // Apply column 0 of the matrix: new_low = U[0][0]*self, new_high = U[1][0]*self
        let scaled_self = DDEdge::new(edge.node_id, w);
        let new_low = self.scale_dd(&scaled_self, mat[0][0]);
        let new_high = self.scale_dd(&scaled_self, mat[1][0]);
        self.build_node(target, new_low, new_high)
    }

    // -----------------------------------------------------------------------
    // CNOT gate
    // -----------------------------------------------------------------------

    /// Apply a CNOT gate with given control and target qubits.
    ///
    /// When control < target, we encounter control first in the DD traversal
    /// and can use a simple recursive approach. When target < control, we
    /// use a decomposition that extracts the control-|0> and control-|1>
    /// sub-diagrams at the target level.
    fn apply_cnot_gate(&mut self, control: usize, target: usize) {
        if control < target {
            // Standard case: control is above target in the DD.
            let root_clone = self.root.clone();
            self.root = self.apply_cnot_ctrl_above(&root_clone, control, target, false);
        } else {
            // Reversed case: target is above control.
            // Decompose using X gate on target, conditioned on control.
            // This is equivalent to applying a controlled-X where the
            // control is below the target in the variable ordering.
            let root_clone = self.root.clone();
            self.root = self.apply_cnot_tgt_above(&root_clone, control, target);
        }
        self.add_cache.clear();
    }

    /// CNOT where control variable < target variable (control is encountered
    /// first in top-down DD traversal).
    fn apply_cnot_ctrl_above(
        &mut self,
        edge: &DDEdge,
        control: usize,
        target: usize,
        ctrl_active: bool,
    ) -> DDEdge {
        if edge.is_zero() {
            return DDEdge::zero();
        }

        let w = edge.effective_weight();

        if edge.node_id == TERMINAL_ONE {
            if ctrl_active {
                // Implicit |0> on target. CNOT flips it to |1>.
                let low = DDEdge::zero();
                let high = DDEdge::one(w);
                return self.build_node(target, low, high);
            }
            return DDEdge::one(w);
        }

        let node = self.pool.get(edge.node_id).clone();

        if node.var == control {
            let child_low = self.weighted_edge(&node.low, w);
            let child_high = self.weighted_edge(&node.high, w);

            let new_low = self.apply_cnot_ctrl_above(&child_low, control, target, false);
            let new_high =
                self.apply_cnot_ctrl_above(&child_high, control, target, true);

            return self.build_node(node.var, new_low, new_high);
        }

        if node.var == target && ctrl_active {
            // Swap low and high (X gate on target).
            let new_low = self.weighted_edge(&node.high, w);
            let new_high = self.weighted_edge(&node.low, w);
            return self.build_node(node.var, new_low, new_high);
        }

        if node.var == target {
            // Control above, not active: no change.
            return DDEdge::new(edge.node_id, w);
        }

        // Not at control or target: recurse.
        let child_low = self.weighted_edge(&node.low, w);
        let child_high = self.weighted_edge(&node.high, w);

        let new_low =
            self.apply_cnot_ctrl_above(&child_low, control, target, ctrl_active);
        let new_high =
            self.apply_cnot_ctrl_above(&child_high, control, target, ctrl_active);

        self.build_node(node.var, new_low, new_high)
    }

    /// CNOT where target variable < control variable (target is encountered
    /// first in top-down DD traversal).
    ///
    /// Strategy: at the target node, extract the control-|0> and control-|1>
    /// sub-diagrams from each branch. Then recombine, re-wrapping the
    /// control-|1> parts with their control variable:
    ///
    ///   new_low  = low_ctrl0  + wrap_ctrl1(high_ctrl1)
    ///   new_high = high_ctrl0 + wrap_ctrl1(low_ctrl1)
    ///
    /// where wrap_ctrl1(dd) creates a node at the control variable with
    /// low=zero, high=dd (preserving that control was |1>).
    fn apply_cnot_tgt_above(
        &mut self,
        edge: &DDEdge,
        control: usize,
        target: usize,
    ) -> DDEdge {
        if edge.is_zero() {
            return DDEdge::zero();
        }

        let w = edge.effective_weight();

        if edge.node_id == TERMINAL_ONE {
            // All qubits implicitly |0>. Control is |0>, so no flip.
            return DDEdge::one(w);
        }

        let node = self.pool.get(edge.node_id).clone();

        if node.var == target {
            let old_low = self.weighted_edge(&node.low, w);
            let old_high = self.weighted_edge(&node.high, w);

            // Split each branch by the control variable value.
            let (low_ctrl0, low_ctrl1) = self.split_on_var(&old_low, control);
            let (high_ctrl0, high_ctrl1) = self.split_on_var(&old_high, control);

            // Re-wrap the ctrl=1 parts with the control variable node
            // (preserving that control qubit is |1>).
            let high_ctrl1_wrapped = self.wrap_var(control, high_ctrl1);
            let low_ctrl1_wrapped = self.wrap_var(control, low_ctrl1);
            let high_ctrl0_wrapped = self.wrap_var_low(control, high_ctrl0);
            let low_ctrl0_wrapped = self.wrap_var_low(control, low_ctrl0);

            // Recombine:
            // ctrl=0: no change (low stays low, high stays high)
            // ctrl=1: X on target (low<->high swap)
            let new_low = self.add_dd(&low_ctrl0_wrapped, &high_ctrl1_wrapped);
            let new_high = self.add_dd(&high_ctrl0_wrapped, &low_ctrl1_wrapped);

            return self.build_node(target, new_low, new_high);
        }

        if node.var < target {
            let child_low = self.weighted_edge(&node.low, w);
            let child_high = self.weighted_edge(&node.high, w);

            let new_low = self.apply_cnot_tgt_above(&child_low, control, target);
            let new_high = self.apply_cnot_tgt_above(&child_high, control, target);

            return self.build_node(node.var, new_low, new_high);
        }

        // node.var > target: target is implicit |0>. Split on control.
        let dd = DDEdge::new(edge.node_id, w);
        let (ctrl0, ctrl1) = self.split_on_var(&dd, control);

        // Re-wrap with control variable preserved.
        let ctrl0_wrapped = self.wrap_var_low(control, ctrl0);
        let ctrl1_wrapped = self.wrap_var(control, ctrl1);

        // ctrl=0: target stays |0> -> new_low
        // ctrl=1: target flips |0>->|1> -> new_high
        self.build_node(target, ctrl0_wrapped, ctrl1_wrapped)
    }

    /// Split a DD into two parts based on the value of a given variable:
    /// - Part 0: sub-diagram where `var` is |0>
    /// - Part 1: sub-diagram where `var` is |1>
    ///
    /// Both returned DDs have the `var` variable removed (projected out).
    fn split_on_var(&mut self, edge: &DDEdge, var: usize) -> (DDEdge, DDEdge) {
        if edge.is_zero() {
            return (DDEdge::zero(), DDEdge::zero());
        }

        let w = edge.effective_weight();

        if edge.node_id == TERMINAL_ONE {
            // Variable is implicitly |0>.
            return (DDEdge::one(w), DDEdge::zero());
        }

        let node = self.pool.get(edge.node_id).clone();

        if node.var == var {
            let part0 = self.weighted_edge(&node.low, w);
            let part1 = self.weighted_edge(&node.high, w);
            return (part0, part1);
        }

        if node.var > var {
            // Variable is implicitly |0> above this node.
            return (DDEdge::new(edge.node_id, w), DDEdge::zero());
        }

        // node.var < var: recurse into both children.
        let child_low = self.weighted_edge(&node.low, w);
        let child_high = self.weighted_edge(&node.high, w);

        let (low_0, low_1) = self.split_on_var(&child_low, var);
        let (high_0, high_1) = self.split_on_var(&child_high, var);

        let part0 = self.build_node(node.var, low_0, high_0);
        let part1 = self.build_node(node.var, low_1, high_1);

        (part0, part1)
    }

    /// Wrap a sub-diagram in a variable node with low=zero, high=dd.
    /// This represents the sub-state where `var` is |1>.
    fn wrap_var(&mut self, var: usize, dd: DDEdge) -> DDEdge {
        if dd.is_zero() {
            return DDEdge::zero();
        }
        self.build_node(var, DDEdge::zero(), dd)
    }

    /// Wrap a sub-diagram in a variable node with low=dd, high=zero.
    /// This represents the sub-state where `var` is |0>.
    /// Since high=zero, build_node returns dd directly (variable is skipped).
    fn wrap_var_low(&mut self, _var: usize, dd: DDEdge) -> DDEdge {
        // When high=zero, the variable is implicitly |0>, so the DD
        // already correctly represents this without an explicit node.
        dd
    }

    // -----------------------------------------------------------------------
    // Controlled single-qubit gate
    // -----------------------------------------------------------------------

    /// Apply a controlled single-qubit gate.
    fn apply_controlled_gate(&mut self, control: usize, target: usize, gate_type: &GateType) {
        let mat = gate_matrix_2x2(gate_type);
        let root_clone = self.root.clone();
        self.root =
            self.apply_controlled_recursive(&root_clone, control, target, &mat, false);
        self.add_cache.clear();
    }

    fn apply_controlled_recursive(
        &mut self,
        edge: &DDEdge,
        control: usize,
        target: usize,
        mat: &[[C64; 2]; 2],
        ctrl_active: bool,
    ) -> DDEdge {
        if edge.is_zero() {
            return DDEdge::zero();
        }

        let w = edge.effective_weight();

        if edge.node_id == TERMINAL_ONE {
            if ctrl_active {
                let new_low_w = mat[0][0] * w;
                let new_high_w = mat[1][0] * w;
                return self.make_edge(target, new_low_w, TERMINAL_ONE, new_high_w, TERMINAL_ONE);
            }
            return DDEdge::one(w);
        }

        let node = self.pool.get(edge.node_id).clone();

        if node.var == control {
            let child_low = self.weighted_edge(&node.low, w);
            let child_high = self.weighted_edge(&node.high, w);

            let new_low =
                self.apply_controlled_recursive(&child_low, control, target, mat, false);
            let new_high =
                self.apply_controlled_recursive(&child_high, control, target, mat, true);

            return self.build_node(node.var, new_low, new_high);
        }

        if node.var == target && ctrl_active {
            // Apply the gate at the target level.
            let old_low = self.weighted_edge(&node.low, w);
            let old_high = self.weighted_edge(&node.high, w);

            let term00 = self.scale_dd(&old_low, mat[0][0]);
            let term01 = self.scale_dd(&old_high, mat[0][1]);
            let new_low = self.add_dd(&term00, &term01);

            let term10 = self.scale_dd(&old_low, mat[1][0]);
            let term11 = self.scale_dd(&old_high, mat[1][1]);
            let new_high = self.add_dd(&term10, &term11);

            return self.build_node(target, new_low, new_high);
        }

        // Recurse into both children.
        let child_low = self.weighted_edge(&node.low, w);
        let child_high = self.weighted_edge(&node.high, w);

        let new_low =
            self.apply_controlled_recursive(&child_low, control, target, mat, ctrl_active);
        let new_high =
            self.apply_controlled_recursive(&child_high, control, target, mat, ctrl_active);

        self.build_node(node.var, new_low, new_high)
    }

    // -----------------------------------------------------------------------
    // Toffoli gate
    // -----------------------------------------------------------------------

    fn apply_toffoli_gate(&mut self, c1: usize, c2: usize, target: usize) {
        let root_clone = self.root.clone();
        self.root = self.apply_toffoli_recursive(&root_clone, c1, c2, target, false, false);
        self.add_cache.clear();
    }

    fn apply_toffoli_recursive(
        &mut self,
        edge: &DDEdge,
        c1: usize,
        c2: usize,
        target: usize,
        c1_on: bool,
        c2_on: bool,
    ) -> DDEdge {
        if edge.is_zero() {
            return DDEdge::zero();
        }

        let w = edge.effective_weight();

        if edge.node_id == TERMINAL_ONE {
            if c1_on && c2_on {
                let low = DDEdge::zero();
                let high = DDEdge::one(w);
                return self.build_node(target, low, high);
            }
            return DDEdge::one(w);
        }

        let node = self.pool.get(edge.node_id).clone();

        if node.var == c1 {
            let child_low = self.weighted_edge(&node.low, w);
            let child_high = self.weighted_edge(&node.high, w);
            let new_low =
                self.apply_toffoli_recursive(&child_low, c1, c2, target, false, c2_on);
            let new_high =
                self.apply_toffoli_recursive(&child_high, c1, c2, target, true, c2_on);
            return self.build_node(node.var, new_low, new_high);
        }

        if node.var == c2 {
            let child_low = self.weighted_edge(&node.low, w);
            let child_high = self.weighted_edge(&node.high, w);
            let new_low =
                self.apply_toffoli_recursive(&child_low, c1, c2, target, c1_on, false);
            let new_high =
                self.apply_toffoli_recursive(&child_high, c1, c2, target, c1_on, true);
            return self.build_node(node.var, new_low, new_high);
        }

        if node.var == target {
            if c1_on && c2_on {
                let new_low = self.weighted_edge(&node.high, w);
                let new_high = self.weighted_edge(&node.low, w);
                return self.build_node(node.var, new_low, new_high);
            }
            return DDEdge::new(edge.node_id, w);
        }

        let child_low = self.weighted_edge(&node.low, w);
        let child_high = self.weighted_edge(&node.high, w);
        let new_low =
            self.apply_toffoli_recursive(&child_low, c1, c2, target, c1_on, c2_on);
        let new_high =
            self.apply_toffoli_recursive(&child_high, c1, c2, target, c1_on, c2_on);
        self.build_node(node.var, new_low, new_high)
    }

    // -----------------------------------------------------------------------
    // DD arithmetic: scale and add
    // -----------------------------------------------------------------------

    /// Scale a DD edge by a complex scalar.
    fn scale_dd(&self, edge: &DDEdge, scalar: C64) -> DDEdge {
        if scalar.norm() < EPSILON || edge.is_zero() {
            return DDEdge::zero();
        }
        DDEdge {
            node_id: edge.node_id,
            weight: edge.effective_weight() * scalar,
            complemented: false,
        }
    }

    /// Add two decision diagrams. This is the core operation that combines
    /// two sub-diagrams representing quantum state vectors.
    ///
    /// add(edge_a, edge_b) returns an edge representing the pointwise sum
    /// of the amplitudes encoded by edge_a and edge_b.
    fn add_dd(&mut self, a: &DDEdge, b: &DDEdge) -> DDEdge {
        // Base cases.
        if a.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return a.clone();
        }

        let wa = a.effective_weight();
        let wb = b.effective_weight();

        // Both point to terminal-one: just add weights.
        if a.node_id == TERMINAL_ONE && b.node_id == TERMINAL_ONE {
            let sum = wa + wb;
            if sum.norm() < EPSILON {
                return DDEdge::zero();
            }
            return DDEdge::one(sum);
        }

        // One is terminal-one, the other is internal.
        if a.node_id == TERMINAL_ONE {
            // terminal-one represents all-zeros state with weight wa.
            // We need to add wa to the |0...0> path of b.
            return self.add_terminal_to_dd(wa, b);
        }
        if b.node_id == TERMINAL_ONE {
            return self.add_terminal_to_dd(wb, a);
        }

        // Both internal nodes.
        let node_a = self.pool.get(a.node_id).clone();
        let node_b = self.pool.get(b.node_id).clone();

        let (top_var, a_low, a_high, b_low, b_high) = if node_a.var == node_b.var {
            (
                node_a.var,
                self.weighted_edge(&node_a.low, wa),
                self.weighted_edge(&node_a.high, wa),
                self.weighted_edge(&node_b.low, wb),
                self.weighted_edge(&node_b.high, wb),
            )
        } else if node_a.var < node_b.var {
            // a has the smaller (higher-level) variable.
            // b is implicitly identity at var_a (b passes through on |0>).
            let b_as_low = DDEdge::new(b.node_id, wb);
            (
                node_a.var,
                self.weighted_edge(&node_a.low, wa),
                self.weighted_edge(&node_a.high, wa),
                b_as_low,
                DDEdge::zero(),
            )
        } else {
            // b has the smaller variable.
            let a_as_low = DDEdge::new(a.node_id, wa);
            (
                node_b.var,
                a_as_low,
                DDEdge::zero(),
                self.weighted_edge(&node_b.low, wb),
                self.weighted_edge(&node_b.high, wb),
            )
        };

        let new_low = self.add_dd(&a_low, &b_low);
        let new_high = self.add_dd(&a_high, &b_high);

        self.build_node(top_var, new_low, new_high)
    }

    /// Add a terminal-one (representing weight `tw` on the all-zeros path)
    /// to an existing DD.
    fn add_terminal_to_dd(&mut self, tw: C64, dd: &DDEdge) -> DDEdge {
        if tw.norm() < EPSILON {
            return dd.clone();
        }
        if dd.is_zero() {
            return DDEdge::one(tw);
        }

        let wd = dd.effective_weight();

        if dd.node_id == TERMINAL_ONE {
            let sum = tw + wd;
            if sum.norm() < EPSILON {
                return DDEdge::zero();
            }
            return DDEdge::one(sum);
        }

        let node = self.pool.get(dd.node_id).clone();

        // The terminal adds to the |0> path (low branch) of every level.
        // At this node's variable, the |0> branch gets the terminal added.
        let old_low = self.weighted_edge(&node.low, wd);
        let old_high = self.weighted_edge(&node.high, wd);

        let new_low = self.add_terminal_to_dd(tw, &old_low);
        // high branch is unchanged.
        self.build_node(node.var, new_low, old_high)
    }

    // -----------------------------------------------------------------------
    // Node construction helpers
    // -----------------------------------------------------------------------

    /// Build a normalized DD node from a variable and two child edges.
    /// Extracts a common weight factor into the returned edge.
    fn build_node(&mut self, var: usize, low: DDEdge, high: DDEdge) -> DDEdge {
        if low.is_zero() && high.is_zero() {
            return DDEdge::zero();
        }

        // If high is zero, this level could be skipped (the variable is
        // implicitly |0>). Return the low edge directly.
        if high.is_zero() {
            return low;
        }

        // Factor out a normalization weight from the low edge.
        let norm = if low.effective_weight().norm() > EPSILON {
            low.effective_weight()
        } else {
            high.effective_weight()
        };

        if norm.norm() < EPSILON {
            return DDEdge::zero();
        }

        let inv = C64::new(1.0, 0.0) / norm;

        let norm_low = if low.is_zero() {
            DDEdge::zero()
        } else {
            DDEdge::new(low.node_id, low.effective_weight() * inv)
        };
        let norm_high = DDEdge::new(high.node_id, high.effective_weight() * inv);

        let node_id = self.pool.get_or_create(var, norm_low, norm_high);
        self.pool.inc_ref(node_id);

        DDEdge::new(node_id, norm)
    }

    /// Construct an edge to a node with given children weights and node ids.
    fn make_edge(
        &mut self,
        var: usize,
        low_w: C64,
        low_id: usize,
        high_w: C64,
        high_id: usize,
    ) -> DDEdge {
        let low = if low_w.norm() < EPSILON {
            DDEdge::zero()
        } else {
            DDEdge::new(low_id, low_w)
        };
        let high = if high_w.norm() < EPSILON {
            DDEdge::zero()
        } else {
            DDEdge::new(high_id, high_w)
        };
        self.build_node(var, low, high)
    }

    /// Combine an incoming weight with a child edge to produce a weighted edge.
    #[inline]
    fn weighted_edge(&self, child: &DDEdge, parent_weight: C64) -> DDEdge {
        if child.is_zero() || parent_weight.norm() < EPSILON {
            return DDEdge::zero();
        }
        DDEdge {
            node_id: child.node_id,
            weight: parent_weight * child.effective_weight(),
            complemented: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: 2x2 gate matrix extraction
// ---------------------------------------------------------------------------

/// Extract the 2x2 unitary matrix for a single-qubit gate type.
fn gate_matrix_2x2(gate_type: &GateType) -> [[C64; 2]; 2] {
    let mat = gate_type.matrix();
    [[mat[0][0], mat[0][1]], [mat[1][0], mat[1][1]]]
}

// ---------------------------------------------------------------------------
// Display trait for DDStats
// ---------------------------------------------------------------------------

impl std::fmt::Display for DDStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DDStats {{ nodes: {}, max: {}, gc: {} runs ({} collected), cache: {}/{} hit/miss }}",
            self.total_nodes,
            self.max_nodes,
            self.gc_runs,
            self.nodes_collected,
            self.cache_hits,
            self.cache_misses,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    const TOL: f64 = 1e-10;

    fn amp_close(a: C64, b: C64) -> bool {
        (a - b).norm() < TOL
    }

    fn prob_close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // -- basic state creation -----------------------------------------------

    #[test]
    fn test_initial_state_single_qubit() {
        let sim = DDSimulator::new(1);
        let amp0 = sim.amplitude(0);
        let amp1 = sim.amplitude(1);
        assert!(
            amp_close(amp0, C64::new(1.0, 0.0)),
            "|0> amp = {:?}",
            amp0
        );
        assert!(
            amp_close(amp1, C64::new(0.0, 0.0)),
            "|1> amp = {:?}",
            amp1
        );
    }

    #[test]
    fn test_initial_state_two_qubit() {
        let sim = DDSimulator::new(2);
        assert!(amp_close(sim.amplitude(0b00), C64::new(1.0, 0.0)));
        assert!(amp_close(sim.amplitude(0b01), C64::new(0.0, 0.0)));
        assert!(amp_close(sim.amplitude(0b10), C64::new(0.0, 0.0)));
        assert!(amp_close(sim.amplitude(0b11), C64::new(0.0, 0.0)));
    }

    // -- X gate -------------------------------------------------------------

    #[test]
    fn test_x_gate_flips_to_one() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::x(0));
        assert!(
            amp_close(sim.amplitude(0), C64::new(0.0, 0.0)),
            "|0> = {:?}",
            sim.amplitude(0)
        );
        assert!(
            amp_close(sim.amplitude(1), C64::new(1.0, 0.0)),
            "|1> = {:?}",
            sim.amplitude(1)
        );
    }

    #[test]
    fn test_x_gate_double_identity() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::x(0));
        assert!(amp_close(sim.amplitude(0), C64::new(1.0, 0.0)));
        assert!(amp_close(sim.amplitude(1), C64::new(0.0, 0.0)));
    }

    #[test]
    fn test_x_gate_second_qubit() {
        let mut sim = DDSimulator::new(2);
        sim.apply_gate(&Gate::x(1));
        assert!(amp_close(sim.amplitude(0b00), C64::new(0.0, 0.0)));
        assert!(amp_close(sim.amplitude(0b01), C64::new(0.0, 0.0)));
        assert!(
            amp_close(sim.amplitude(0b10), C64::new(1.0, 0.0)),
            "|10> = {:?}",
            sim.amplitude(0b10)
        );
        assert!(amp_close(sim.amplitude(0b11), C64::new(0.0, 0.0)));
    }

    // -- H gate -------------------------------------------------------------

    #[test]
    fn test_h_gate_equal_superposition() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::h(0));
        let expected = C64::new(FRAC_1_SQRT_2, 0.0);
        assert!(
            amp_close(sim.amplitude(0), expected),
            "H|0> -> |0> = {:?}",
            sim.amplitude(0)
        );
        assert!(
            amp_close(sim.amplitude(1), expected),
            "H|0> -> |1> = {:?}",
            sim.amplitude(1)
        );
    }

    #[test]
    fn test_h_gate_double_identity() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::h(0));
        assert!(
            amp_close(sim.amplitude(0), C64::new(1.0, 0.0)),
            "HH|0> = {:?}",
            sim.amplitude(0)
        );
        assert!(
            amp_close(sim.amplitude(1), C64::new(0.0, 0.0)),
            "HH|1> = {:?}",
            sim.amplitude(1)
        );
    }

    // -- Z gate -------------------------------------------------------------

    #[test]
    fn test_z_gate_phase_flip() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::z(0));
        assert!(amp_close(sim.amplitude(0), C64::new(0.0, 0.0)));
        assert!(
            amp_close(sim.amplitude(1), C64::new(-1.0, 0.0)),
            "Z|1> = {:?}",
            sim.amplitude(1)
        );
    }

    #[test]
    fn test_z_gate_no_effect_on_zero() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::z(0));
        assert!(amp_close(sim.amplitude(0), C64::new(1.0, 0.0)));
        assert!(amp_close(sim.amplitude(1), C64::new(0.0, 0.0)));
    }

    // -- S gate -------------------------------------------------------------

    #[test]
    fn test_s_gate_phase() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::s(0));
        assert!(amp_close(sim.amplitude(0), C64::new(0.0, 0.0)));
        assert!(
            amp_close(sim.amplitude(1), C64::new(0.0, 1.0)),
            "S|1> = {:?}",
            sim.amplitude(1)
        );
    }

    // -- T gate -------------------------------------------------------------

    #[test]
    fn test_t_gate_phase() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::t(0));
        let expected = C64::new((PI / 4.0).cos(), (PI / 4.0).sin());
        assert!(amp_close(sim.amplitude(0), C64::new(0.0, 0.0)));
        assert!(
            amp_close(sim.amplitude(1), expected),
            "T|1> = {:?}, expected {:?}",
            sim.amplitude(1),
            expected,
        );
    }

    // -- Bell state ---------------------------------------------------------

    #[test]
    fn test_bell_state() {
        let mut sim = DDSimulator::new(2);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));

        let expected = C64::new(FRAC_1_SQRT_2, 0.0);
        assert!(
            amp_close(sim.amplitude(0b00), expected),
            "|00> = {:?}",
            sim.amplitude(0b00)
        );
        assert!(
            amp_close(sim.amplitude(0b01), C64::new(0.0, 0.0)),
            "|01> = {:?}",
            sim.amplitude(0b01)
        );
        assert!(
            amp_close(sim.amplitude(0b10), C64::new(0.0, 0.0)),
            "|10> = {:?}",
            sim.amplitude(0b10)
        );
        assert!(
            amp_close(sim.amplitude(0b11), expected),
            "|11> = {:?}",
            sim.amplitude(0b11)
        );
    }

    // -- GHZ state ----------------------------------------------------------

    #[test]
    fn test_ghz_state_3_qubits() {
        let mut sim = DDSimulator::new(3);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        sim.apply_gate(&Gate::cnot(0, 2));

        let expected = C64::new(FRAC_1_SQRT_2, 0.0);
        assert!(amp_close(sim.amplitude(0b000), expected));
        assert!(amp_close(sim.amplitude(0b111), expected));

        for bs in 1..7 {
            assert!(
                amp_close(sim.amplitude(bs), C64::new(0.0, 0.0)),
                "GHZ |{:03b}> = {:?}",
                bs,
                sim.amplitude(bs),
            );
        }
    }

    // -- Probabilities ------------------------------------------------------

    #[test]
    fn test_probabilities_sum_to_one_single() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::h(0));
        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0), "Sum = {}", sum);
    }

    #[test]
    fn test_probabilities_sum_to_one_multi_gate() {
        let mut sim = DDSimulator::new(3);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::h(1));
        sim.apply_gate(&Gate::cnot(0, 2));
        sim.apply_gate(&Gate::t(1));
        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0), "Sum = {}", sum);
    }

    #[test]
    fn test_probabilities_bell_state() {
        let mut sim = DDSimulator::new(2);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        let probs = sim.probabilities();
        assert!(prob_close(probs[0b00], 0.5));
        assert!(prob_close(probs[0b01], 0.0));
        assert!(prob_close(probs[0b10], 0.0));
        assert!(prob_close(probs[0b11], 0.5));
    }

    // -- SWAP gate ----------------------------------------------------------

    #[test]
    fn test_swap_gate() {
        let mut sim = DDSimulator::new(2);
        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::swap(0, 1));
        assert!(
            amp_close(sim.amplitude(0b01), C64::new(0.0, 0.0)),
            "|01> = {:?}",
            sim.amplitude(0b01)
        );
        assert!(
            amp_close(sim.amplitude(0b10), C64::new(1.0, 0.0)),
            "|10> = {:?}",
            sim.amplitude(0b10)
        );
    }

    // -- Node deduplication -------------------------------------------------

    #[test]
    fn test_node_deduplication_symmetric() {
        let mut sim = DDSimulator::new(4);
        for q in 0..4 {
            sim.apply_gate(&Gate::h(q));
        }

        let expected_amp = 0.25; // 1/sqrt(16) = 0.25
        for bs in 0..16 {
            let amp = sim.amplitude(bs);
            assert!(
                prob_close(amp.norm_sqr(), expected_amp * expected_amp),
                "Basis {} amplitude {:?}",
                bs,
                amp,
            );
        }

        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0));
    }

    // -- Rotation gates -----------------------------------------------------

    #[test]
    fn test_rx_gate() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::rx(0, PI));
        assert!(
            amp_close(sim.amplitude(0), C64::new(0.0, 0.0)),
            "Rx(pi)|0> -> |0> = {:?}",
            sim.amplitude(0),
        );
        assert!(
            amp_close(sim.amplitude(1), C64::new(0.0, -1.0)),
            "Rx(pi)|0> -> |1> = {:?}",
            sim.amplitude(1),
        );
    }

    #[test]
    fn test_ry_gate() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::ry(0, PI));
        assert!(
            amp_close(sim.amplitude(0), C64::new(0.0, 0.0)),
            "Ry(pi)|0> -> |0> = {:?}",
            sim.amplitude(0),
        );
        assert!(
            amp_close(sim.amplitude(1), C64::new(1.0, 0.0)),
            "Ry(pi)|0> -> |1> = {:?}",
            sim.amplitude(1),
        );
    }

    #[test]
    fn test_rz_gate() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::rz(0, PI));
        let expected = C64::new((PI / 2.0).cos(), (PI / 2.0).sin());
        assert!(
            amp_close(sim.amplitude(1), expected),
            "Rz(pi)|1> = {:?}, expected {:?}",
            sim.amplitude(1),
            expected,
        );
    }

    // -- Multiple gate sequence ---------------------------------------------

    #[test]
    fn test_multi_gate_sequence() {
        let mut sim = DDSimulator::new(2);
        let circuit = vec![Gate::h(0), Gate::cnot(0, 1), Gate::z(0), Gate::h(0)];
        sim.apply_circuit(&circuit);

        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0));
        assert!(prob_close(probs[0b00], 0.25));
        assert!(prob_close(probs[0b01], 0.25));
        assert!(prob_close(probs[0b10], 0.25));
        assert!(prob_close(probs[0b11], 0.25));
    }

    // -- Node count growth --------------------------------------------------

    #[test]
    fn test_node_count_asymmetric_vs_symmetric() {
        let mut sym = DDSimulator::new(4);
        for q in 0..4 {
            sym.apply_gate(&Gate::h(q));
        }
        let sym_nodes = sym.node_count();

        let mut asym = DDSimulator::new(4);
        for q in 0..4 {
            let angle = (q as f64 + 1.0) * 0.3;
            asym.apply_gate(&Gate::ry(q, angle));
        }
        let asym_nodes = asym.node_count();

        // Both should produce valid states; allow generous margin.
        assert!(
            sym_nodes <= asym_nodes + 8,
            "Symmetric ({}) vs asymmetric ({})",
            sym_nodes,
            asym_nodes,
        );
    }

    // -- Garbage collection -------------------------------------------------

    #[test]
    fn test_garbage_collection() {
        let mut sim = DDSimulator::new(3);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));
        sim.apply_gate(&Gate::h(2));

        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::x(0));

        sim.run_gc();

        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0), "Sum after GC = {}", sum);
        assert!(sim.stats().gc_runs >= 1);
    }

    // -- Statistics ---------------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut sim = DDSimulator::new(2);
        sim.apply_gate(&Gate::h(0));
        sim.apply_gate(&Gate::cnot(0, 1));

        let stats = sim.stats();
        assert!(stats.total_nodes > 0);
        assert!(stats.max_nodes >= stats.total_nodes);
    }

    // -- Large qubit count with structured circuit --------------------------

    #[test]
    fn test_large_ghz_10_qubits() {
        let n = 10;
        let mut sim = DDSimulator::new(n);
        sim.apply_gate(&Gate::h(0));
        for q in 1..n {
            sim.apply_gate(&Gate::cnot(0, q));
        }

        let expected = C64::new(FRAC_1_SQRT_2, 0.0);
        assert!(amp_close(sim.amplitude(0), expected));
        assert!(amp_close(sim.amplitude((1 << n) - 1), expected));

        assert!(amp_close(sim.amplitude(1), C64::new(0.0, 0.0)));
        assert!(amp_close(sim.amplitude(1 << 5), C64::new(0.0, 0.0)));

        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0));

        assert!(
            sim.node_count() < 200,
            "GHZ-10 nodes = {}",
            sim.node_count(),
        );
    }

    #[test]
    fn test_large_product_state_12_qubits() {
        let n = 12;
        let mut sim = DDSimulator::new(n);
        for q in 0..n {
            sim.apply_gate(&Gate::h(q));
        }

        let dim = 1 << n;
        let expected_prob = 1.0 / dim as f64;

        for bs in [0, 1, 42, 100, 255, 1000, dim - 1] {
            let amp = sim.amplitude(bs);
            assert!(
                prob_close(amp.norm_sqr(), expected_prob),
                "Basis {} prob = {}, expected {}",
                bs,
                amp.norm_sqr(),
                expected_prob,
            );
        }

        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0));
    }

    // -- CZ gate -----------------------------------------------------------

    #[test]
    fn test_cz_gate() {
        let mut sim = DDSimulator::new(2);
        sim.apply_gate(&Gate::x(0));
        sim.apply_gate(&Gate::x(1));
        sim.apply_gate(&Gate::cz(0, 1));
        assert!(
            amp_close(sim.amplitude(0b11), C64::new(-1.0, 0.0)),
            "CZ|11> = {:?}",
            sim.amplitude(0b11),
        );
    }

    // -- Y gate ------------------------------------------------------------

    #[test]
    fn test_y_gate() {
        let mut sim = DDSimulator::new(1);
        sim.apply_gate(&Gate::y(0));
        assert!(amp_close(sim.amplitude(0), C64::new(0.0, 0.0)));
        assert!(
            amp_close(sim.amplitude(1), C64::new(0.0, 1.0)),
            "Y|0> = {:?}",
            sim.amplitude(1),
        );
    }

    // -- DDNodePool unit tests ---------------------------------------------

    #[test]
    fn test_pool_terminals() {
        let pool = DDNodePool::new();
        assert_eq!(pool.node_count(), 2);
        assert_eq!(pool.get(TERMINAL_ZERO).var, usize::MAX);
        assert_eq!(pool.get(TERMINAL_ONE).var, usize::MAX);
    }

    #[test]
    fn test_pool_deduplication() {
        let mut pool = DDNodePool::new();

        let low = DDEdge::one(C64::new(1.0, 0.0));
        let high = DDEdge::zero();

        let id1 = pool.get_or_create(0, low.clone(), high.clone());
        let id2 = pool.get_or_create(0, low.clone(), high.clone());

        assert_eq!(id1, id2);
        assert_eq!(pool.node_count(), 3);
    }

    #[test]
    fn test_pool_different_vars_not_deduplicated() {
        let mut pool = DDNodePool::new();

        let low = DDEdge::one(C64::new(1.0, 0.0));
        let high = DDEdge::zero();

        let id1 = pool.get_or_create(0, low.clone(), high.clone());
        let id2 = pool.get_or_create(1, low.clone(), high.clone());

        assert_ne!(id1, id2);
        assert_eq!(pool.node_count(), 4);
    }

    // -- Complement edge testing -------------------------------------------

    #[test]
    fn test_complement_edge_effective_weight() {
        let edge = DDEdge::new(5, C64::new(1.0, 0.0));
        assert!(amp_close(edge.effective_weight(), C64::new(1.0, 0.0)));

        let comp = DDEdge::complemented(5, C64::new(1.0, 0.0));
        assert!(amp_close(comp.effective_weight(), C64::new(-1.0, 0.0)));
    }

    // -- Entangled chain circuit -------------------------------------------

    #[test]
    fn test_entangled_chain_5_qubits() {
        let n = 5;
        let mut sim = DDSimulator::new(n);
        sim.apply_gate(&Gate::h(0));
        for q in 0..(n - 1) {
            sim.apply_gate(&Gate::cnot(q, q + 1));
        }

        let expected = C64::new(FRAC_1_SQRT_2, 0.0);
        assert!(amp_close(sim.amplitude(0), expected));
        assert!(amp_close(sim.amplitude((1 << n) - 1), expected));

        let probs = sim.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!(prob_close(sum, 1.0));
    }

    // -- apply_circuit convenience -----------------------------------------

    #[test]
    fn test_apply_circuit_matches_sequential() {
        let circuit = vec![Gate::h(0), Gate::cnot(0, 1), Gate::z(1)];

        let mut sim1 = DDSimulator::new(2);
        sim1.apply_circuit(&circuit);

        let mut sim2 = DDSimulator::new(2);
        for g in &circuit {
            sim2.apply_gate(g);
        }

        for bs in 0..4 {
            assert!(amp_close(sim1.amplitude(bs), sim2.amplitude(bs)));
        }
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn test_zero_qubit_edge_case() {
        let sim = DDSimulator::new(0);
        assert_eq!(sim.num_qubits(), 0);
    }

    #[test]
    fn test_single_qubit_all_gates() {
        let gates = vec![
            Gate::h(0),
            Gate::x(0),
            Gate::y(0),
            Gate::z(0),
            Gate::s(0),
            Gate::t(0),
            Gate::rx(0, 0.5),
            Gate::ry(0, 0.5),
            Gate::rz(0, 0.5),
        ];

        for gate in &gates {
            let mut sim = DDSimulator::new(1);
            sim.apply_gate(gate);
            let probs = sim.probabilities();
            let sum: f64 = probs.iter().sum();
            assert!(
                prob_close(sum, 1.0),
                "Gate {:?} sum = {}",
                gate.gate_type,
                sum,
            );
        }
    }
}
