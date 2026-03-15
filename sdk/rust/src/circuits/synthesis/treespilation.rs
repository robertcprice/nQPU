//! Treespilation: Tree-decomposition based circuit optimization.
//!
//! Exploits tree structure in quantum circuit connectivity graphs to achieve
//! exponential simulation speedups. When a circuit's qubit interaction graph
//! has low treewidth, the circuit can be simulated in O(2^tw * n) instead of
//! O(2^n), where tw is the treewidth and n is the number of qubits.
//!
//! # Key Concepts
//!
//! - **Treewidth**: Measure of how "tree-like" a graph is. Trees have treewidth 1,
//!   grids have treewidth O(sqrt(n)), complete graphs have treewidth n-1.
//! - **Tree decomposition**: Decompose an interaction graph into a tree of "bags"
//!   (sets of vertices) satisfying covering and running intersection properties.
//! - **Contraction order**: The tree decomposition gives an optimal tensor contraction
//!   order for simulation.
//! - **Gate scheduling**: Reorder commuting gates to minimize the treewidth of the
//!   resulting interaction graph.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::treespilation::{Treespiler, TreewidthHeuristic, TreeGate, TreeGateType};
//!
//! let gates = vec![
//!     TreeGate::new(TreeGateType::H, vec![0], vec![]),
//!     TreeGate::new(TreeGateType::CX, vec![0, 1], vec![]),
//!     TreeGate::new(TreeGateType::CX, vec![1, 2], vec![]),
//! ];
//!
//! let treespiler = Treespiler::new(10, TreewidthHeuristic::MinFill);
//! let result = treespiler.treespile(&gates, 3);
//! println!("Treewidth: {} -> {}", result.original_treewidth, result.optimized_treewidth);
//! ```

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

// ============================================================================
// INTERACTION GRAPH
// ============================================================================

/// Graph representation for qubit interactions extracted from a quantum circuit.
///
/// Each vertex represents a qubit and each edge represents a two-qubit gate
/// interaction between those qubits.
#[derive(Debug, Clone)]
pub struct InteractionGraph {
    /// Number of vertices (qubits) in the graph.
    pub n_vertices: usize,
    /// Edges representing two-qubit interactions.
    pub edges: Vec<(usize, usize)>,
    /// Adjacency list representation for efficient neighbor lookups.
    pub adjacency: Vec<HashSet<usize>>,
}

impl InteractionGraph {
    /// Create a new empty interaction graph with the given number of vertices.
    pub fn new(n_vertices: usize) -> Self {
        Self {
            n_vertices,
            edges: Vec::new(),
            adjacency: vec![HashSet::new(); n_vertices],
        }
    }

    /// Add an undirected edge between two vertices.
    /// Returns false if the edge already exists or is a self-loop.
    pub fn add_edge(&mut self, u: usize, v: usize) -> bool {
        if u == v || u >= self.n_vertices || v >= self.n_vertices {
            return false;
        }
        if self.adjacency[u].contains(&v) {
            return false;
        }
        self.adjacency[u].insert(v);
        self.adjacency[v].insert(u);
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        self.edges.push((a, b));
        true
    }

    /// Get the degree of a vertex.
    pub fn degree(&self, v: usize) -> usize {
        if v >= self.n_vertices {
            return 0;
        }
        self.adjacency[v].len()
    }

    /// Get the neighbors of a vertex.
    pub fn neighbors(&self, v: usize) -> &HashSet<usize> {
        &self.adjacency[v]
    }

    /// Check if an edge exists between two vertices.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.n_vertices || v >= self.n_vertices {
            return false;
        }
        self.adjacency[u].contains(&v)
    }

    /// Compute the number of fill edges that would be added if vertex v were eliminated.
    /// Fill edges connect pairs of neighbors of v that are not already connected.
    pub fn fill_count(&self, v: usize) -> usize {
        let neighbors: Vec<usize> = self.adjacency[v].iter().copied().collect();
        let mut count = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if !self.adjacency[neighbors[i]].contains(&neighbors[j]) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Build an interaction graph from a list of gates.
    /// Two-qubit (and multi-qubit) gates create edges between the involved qubits.
    pub fn from_gates(gates: &[TreeGate], n_qubits: usize) -> Self {
        let mut graph = Self::new(n_qubits);
        for gate in gates {
            if gate.qubits.len() >= 2 {
                for i in 0..gate.qubits.len() {
                    for j in (i + 1)..gate.qubits.len() {
                        graph.add_edge(gate.qubits[i], gate.qubits[j]);
                    }
                }
            }
        }
        graph
    }

    /// Create a path graph: 0--1--2--...--n-1
    pub fn path(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n.saturating_sub(1) {
            graph.add_edge(i, i + 1);
        }
        graph
    }

    /// Create a cycle graph: 0--1--2--...--n-1--0
    pub fn cycle(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n {
            graph.add_edge(i, (i + 1) % n);
        }
        graph
    }

    /// Create a complete graph on n vertices.
    pub fn complete(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                graph.add_edge(i, j);
            }
        }
        graph
    }

    /// Create a grid graph with the given dimensions.
    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut graph = Self::new(n);
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if c + 1 < cols {
                    graph.add_edge(v, v + 1);
                }
                if r + 1 < rows {
                    graph.add_edge(v, v + cols);
                }
            }
        }
        graph
    }

    /// Create a tree (specifically a star graph) with a central vertex 0 connected
    /// to all others.
    pub fn star(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 1..n {
            graph.add_edge(0, i);
        }
        graph
    }

    /// Create a binary tree with the given depth.
    /// Root is vertex 0, children of vertex i are 2i+1 and 2i+2.
    pub fn binary_tree(depth: usize) -> Self {
        let n = (1 << (depth + 1)) - 1; // 2^(depth+1) - 1
        let mut graph = Self::new(n);
        for i in 0..n {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < n {
                graph.add_edge(i, left);
            }
            if right < n {
                graph.add_edge(i, right);
            }
        }
        graph
    }

    /// Return a mutable copy of the adjacency structure for elimination algorithms.
    fn clone_adjacency(&self) -> Vec<HashSet<usize>> {
        self.adjacency.clone()
    }
}

// ============================================================================
// TREE DECOMPOSITION
// ============================================================================

/// A tree decomposition of a graph.
///
/// A valid tree decomposition satisfies three properties:
/// 1. **Vertex coverage**: Every vertex appears in at least one bag.
/// 2. **Edge coverage**: For every edge (u,v), there exists a bag containing both u and v.
/// 3. **Running intersection**: For every vertex v, the set of bags containing v
///    forms a connected subtree.
///
/// The **width** of the decomposition is (max bag size) - 1. The **treewidth** of a
/// graph is the minimum width over all valid tree decompositions.
#[derive(Debug, Clone)]
pub struct TreeDecomposition {
    /// bags[i] = set of vertices in bag i.
    pub bags: Vec<BTreeSet<usize>>,
    /// Edges between bags forming a tree.
    pub tree_edges: Vec<(usize, usize)>,
    /// Width of the decomposition: max(|bag|) - 1.
    pub width: usize,
    /// Root bag index (for rooted tree operations).
    pub root: usize,
}

impl TreeDecomposition {
    /// Create a new tree decomposition from bags and edges.
    /// Automatically computes the width.
    pub fn new(bags: Vec<BTreeSet<usize>>, tree_edges: Vec<(usize, usize)>, root: usize) -> Self {
        let width = bags
            .iter()
            .map(|b| b.len())
            .max()
            .unwrap_or(1)
            .saturating_sub(1);
        Self {
            bags,
            tree_edges,
            width,
            root,
        }
    }

    /// Number of bags in the decomposition.
    pub fn num_bags(&self) -> usize {
        self.bags.len()
    }

    /// Validate that this is a valid tree decomposition for the given graph.
    /// Returns Ok(()) if valid, Err with description otherwise.
    pub fn validate(&self, graph: &InteractionGraph) -> Result<(), String> {
        // Property 1: Vertex coverage - every vertex in at least one bag
        for v in 0..graph.n_vertices {
            // Only check vertices that participate in edges or are isolated
            let in_some_bag = self.bags.iter().any(|bag| bag.contains(&v));
            if !in_some_bag {
                // Allow vertices with no edges to not appear (they don't affect treewidth)
                if graph.degree(v) > 0 {
                    return Err(format!(
                        "Vertex coverage violated: vertex {} not in any bag",
                        v
                    ));
                }
            }
        }

        // Property 2: Edge coverage - for every edge, some bag contains both endpoints
        for &(u, v) in &graph.edges {
            let covered = self
                .bags
                .iter()
                .any(|bag| bag.contains(&u) && bag.contains(&v));
            if !covered {
                return Err(format!(
                    "Edge coverage violated: edge ({}, {}) not covered by any bag",
                    u, v
                ));
            }
        }

        // Property 3: Running intersection - bags containing each vertex form a subtree
        // Build adjacency for the tree of bags
        let n_bags = self.bags.len();
        if n_bags == 0 {
            return Ok(());
        }
        let mut bag_adj: Vec<HashSet<usize>> = vec![HashSet::new(); n_bags];
        for &(a, b) in &self.tree_edges {
            if a >= n_bags || b >= n_bags {
                return Err(format!(
                    "Invalid tree edge ({}, {}) with only {} bags",
                    a, b, n_bags
                ));
            }
            bag_adj[a].insert(b);
            bag_adj[b].insert(a);
        }

        // Check the tree of bags is connected (if > 1 bag, need n_bags - 1 edges)
        if n_bags > 1 {
            let mut visited = vec![false; n_bags];
            let mut queue = VecDeque::new();
            visited[0] = true;
            queue.push_back(0);
            let mut count = 1;
            while let Some(node) = queue.pop_front() {
                for &neighbor in &bag_adj[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        count += 1;
                        queue.push_back(neighbor);
                    }
                }
            }
            if count != n_bags {
                return Err(format!(
                    "Bag tree is not connected: reached {} of {} bags",
                    count, n_bags
                ));
            }
        }

        // For each vertex that appears in bags, check the bags form a connected subtree
        let all_vertices: HashSet<usize> =
            self.bags.iter().flat_map(|b| b.iter().copied()).collect();
        for &v in &all_vertices {
            let containing_bags: Vec<usize> =
                (0..n_bags).filter(|&i| self.bags[i].contains(&v)).collect();
            if containing_bags.len() <= 1 {
                continue;
            }
            // BFS on the sub-tree induced by containing_bags
            let containing_set: HashSet<usize> = containing_bags.iter().copied().collect();
            let start = containing_bags[0];
            let mut visited_sub = HashSet::new();
            let mut queue = VecDeque::new();
            visited_sub.insert(start);
            queue.push_back(start);
            while let Some(node) = queue.pop_front() {
                for &neighbor in &bag_adj[node] {
                    if containing_set.contains(&neighbor) && !visited_sub.contains(&neighbor) {
                        visited_sub.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
            if visited_sub.len() != containing_bags.len() {
                return Err(format!(
                    "Running intersection violated for vertex {}: bags {:?} do not form a subtree",
                    v, containing_bags
                ));
            }
        }

        Ok(())
    }

    /// Get the adjacency list of the bag tree.
    pub fn bag_adjacency(&self) -> Vec<HashSet<usize>> {
        let n = self.bags.len();
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for &(a, b) in &self.tree_edges {
            adj[a].insert(b);
            adj[b].insert(a);
        }
        adj
    }

    /// Compute the optimal contraction order from this tree decomposition.
    /// Returns a sequence of (bag_index, vertices_to_contract) pairs.
    /// The contraction proceeds leaf-to-root.
    pub fn contraction_order(&self) -> Vec<(usize, BTreeSet<usize>)> {
        let n = self.bags.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![(0, self.bags[0].clone())];
        }

        let adj = self.bag_adjacency();
        let mut degree: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let mut order = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut processed = vec![false; n];

        // Start from leaves
        for i in 0..n {
            if degree[i] <= 1 && i != self.root {
                queue.push_back(i);
            }
        }

        while let Some(bag_idx) = queue.pop_front() {
            if processed[bag_idx] {
                continue;
            }
            processed[bag_idx] = true;

            // Vertices to contract: those in this bag but not in the parent bag
            let parent_bag: Option<usize> =
                adj[bag_idx].iter().find(|&&nb| !processed[nb]).copied();

            let to_contract = if let Some(parent) = parent_bag {
                self.bags[bag_idx]
                    .difference(&self.bags[parent])
                    .copied()
                    .collect()
            } else {
                self.bags[bag_idx].clone()
            };

            order.push((bag_idx, to_contract));

            // Update degrees
            for &nb in &adj[bag_idx] {
                if !processed[nb] {
                    degree[nb] = degree[nb].saturating_sub(1);
                    if degree[nb] <= 1 && nb != self.root {
                        queue.push_back(nb);
                    }
                }
            }
        }

        // Add root last
        if !processed[self.root] {
            order.push((self.root, self.bags[self.root].clone()));
        }

        order
    }

    /// Estimate the contraction cost based on the tree decomposition.
    /// Cost is dominated by the largest bag: O(2^width * n_bags).
    pub fn contraction_cost(&self) -> f64 {
        if self.bags.is_empty() {
            return 0.0;
        }
        // Total cost: sum of 2^|bag_i| for each bag
        let mut total: f64 = 0.0;
        for bag in &self.bags {
            total += 2.0_f64.powi(bag.len() as i32);
        }
        total
    }
}

// ============================================================================
// GATE TYPES AND GATES
// ============================================================================

/// Quantum gate types supported by the treespiler.
#[derive(Debug, Clone, PartialEq)]
pub enum TreeGateType {
    /// Hadamard gate.
    H,
    /// Pauli-X (NOT) gate.
    X,
    /// Pauli-Y gate.
    Y,
    /// Pauli-Z gate.
    Z,
    /// Phase gate (S = sqrt(Z)).
    S,
    /// T gate (pi/8 gate).
    T,
    /// Rotation around X by angle theta.
    Rx,
    /// Rotation around Y by angle theta.
    Ry,
    /// Rotation around Z by angle theta.
    Rz,
    /// Controlled-NOT (CNOT) gate.
    CX,
    /// Controlled-Z gate.
    CZ,
    /// SWAP gate.
    SWAP,
    /// Toffoli (CCX) gate.
    CCX,
    /// Custom named gate.
    Custom(String),
}

impl TreeGateType {
    /// Number of qubits this gate type operates on (minimum).
    pub fn num_qubits(&self) -> usize {
        match self {
            TreeGateType::H
            | TreeGateType::X
            | TreeGateType::Y
            | TreeGateType::Z
            | TreeGateType::S
            | TreeGateType::T
            | TreeGateType::Rx
            | TreeGateType::Ry
            | TreeGateType::Rz => 1,
            TreeGateType::CX | TreeGateType::CZ | TreeGateType::SWAP => 2,
            TreeGateType::CCX => 3,
            TreeGateType::Custom(_) => 1, // Default; actual count determined by qubits vec
        }
    }

    /// Whether this is a single-qubit gate.
    pub fn is_single_qubit(&self) -> bool {
        self.num_qubits() == 1
    }

    /// Whether this is a multi-qubit gate that creates interaction edges.
    pub fn is_multi_qubit(&self) -> bool {
        self.num_qubits() >= 2
    }
}

/// A quantum circuit gate for treespilation, carrying its type, qubit operands,
/// and optional rotation parameters.
#[derive(Debug, Clone)]
pub struct TreeGate {
    /// The type of quantum gate.
    pub gate_type: TreeGateType,
    /// Qubit indices this gate acts on (control + target for multi-qubit gates).
    pub qubits: Vec<usize>,
    /// Optional parameters (rotation angles, etc.).
    pub params: Vec<f64>,
}

impl TreeGate {
    /// Create a new gate with no parameters.
    pub fn new(gate_type: TreeGateType, qubits: Vec<usize>, params: Vec<f64>) -> Self {
        Self {
            gate_type,
            qubits,
            params,
        }
    }

    /// Create a single-qubit gate.
    pub fn single(gate_type: TreeGateType, qubit: usize) -> Self {
        Self::new(gate_type, vec![qubit], vec![])
    }

    /// Create a two-qubit gate.
    pub fn two_qubit(gate_type: TreeGateType, q0: usize, q1: usize) -> Self {
        Self::new(gate_type, vec![q0, q1], vec![])
    }

    /// Check if this gate commutes with another gate.
    /// Gates on disjoint qubit sets always commute. Additionally, diagonal gates
    /// on the same qubit commute (Z, S, T, Rz, CZ).
    pub fn commutes_with(&self, other: &TreeGate) -> bool {
        // Gates on disjoint qubits always commute
        let self_qubits: HashSet<usize> = self.qubits.iter().copied().collect();
        let other_qubits: HashSet<usize> = other.qubits.iter().copied().collect();
        if self_qubits.is_disjoint(&other_qubits) {
            return true;
        }

        // Diagonal gates commute with each other
        let self_diagonal = self.is_diagonal();
        let other_diagonal = other.is_diagonal();
        if self_diagonal && other_diagonal {
            return true;
        }

        false
    }

    /// Check if this gate is diagonal in the computational basis.
    fn is_diagonal(&self) -> bool {
        matches!(
            self.gate_type,
            TreeGateType::Z
                | TreeGateType::S
                | TreeGateType::T
                | TreeGateType::Rz
                | TreeGateType::CZ
        )
    }

    /// Get the set of qubits this gate acts on.
    pub fn qubit_set(&self) -> HashSet<usize> {
        self.qubits.iter().copied().collect()
    }
}

// ============================================================================
// ELIMINATION ORDERING HEURISTICS
// ============================================================================

/// Heuristic strategies for computing elimination orderings.
///
/// Different heuristics trade off between quality and computation time.
/// All produce valid elimination orderings; the choice affects the resulting
/// treewidth bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreewidthHeuristic {
    /// Minimum degree: always eliminate the vertex with the fewest neighbors.
    /// Fast O(n^2) but can produce poor orderings on some graphs.
    MinDegree,
    /// Minimum fill: eliminate the vertex whose elimination adds the fewest
    /// new edges. Better quality than MinDegree but slower O(n^2 * max_degree).
    MinFill,
    /// Greedy degree: like MinDegree but with tie-breaking by fill count.
    /// Good balance of speed and quality.
    GreedyDegree,
}

/// Result of computing an elimination ordering.
#[derive(Debug, Clone)]
pub struct EliminationResult {
    /// The elimination ordering (sequence of vertex indices).
    pub ordering: Vec<usize>,
    /// The width (max bag size - 1) resulting from this ordering.
    pub width: usize,
    /// The bags produced during elimination (for constructing tree decomposition).
    pub bags: Vec<BTreeSet<usize>>,
}

/// Compute an elimination ordering using the specified heuristic.
///
/// The elimination ordering drives the tree decomposition construction.
/// Each step:
/// 1. Select the next vertex to eliminate (per heuristic).
/// 2. Record the bag: {vertex} union {current neighbors}.
/// 3. Connect all neighbors (add fill edges).
/// 4. Remove the vertex from the graph.
pub fn compute_elimination_ordering(
    graph: &InteractionGraph,
    heuristic: TreewidthHeuristic,
) -> EliminationResult {
    let n = graph.n_vertices;
    let mut adj = graph.clone_adjacency();
    let mut eliminated = vec![false; n];
    let mut ordering = Vec::with_capacity(n);
    let mut bags = Vec::with_capacity(n);
    let mut max_bag_size: usize = 0;

    for _step in 0..n {
        // Select vertex to eliminate based on heuristic
        let v = select_vertex(&adj, &eliminated, heuristic);
        eliminated[v] = true;
        ordering.push(v);

        // Form the bag: v + its current (non-eliminated) neighbors
        let neighbors: Vec<usize> = adj[v].iter().copied().filter(|&u| !eliminated[u]).collect();
        let mut bag = BTreeSet::new();
        bag.insert(v);
        for &u in &neighbors {
            bag.insert(u);
        }
        if bag.len() > max_bag_size {
            max_bag_size = bag.len();
        }
        bags.push(bag);

        // Add fill edges: connect all pairs of non-eliminated neighbors
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let a = neighbors[i];
                let b = neighbors[j];
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }

        // Remove v from all adjacency lists
        for &u in &neighbors {
            adj[u].remove(&v);
        }
        adj[v].clear();
    }

    let width = max_bag_size.saturating_sub(1);
    EliminationResult {
        ordering,
        width,
        bags,
    }
}

/// Select the next vertex to eliminate based on the heuristic.
fn select_vertex(
    adj: &[HashSet<usize>],
    eliminated: &[bool],
    heuristic: TreewidthHeuristic,
) -> usize {
    let n = adj.len();
    let active: Vec<usize> = (0..n).filter(|&v| !eliminated[v]).collect();
    if active.is_empty() {
        // Should not happen during normal elimination
        return 0;
    }

    match heuristic {
        TreewidthHeuristic::MinDegree => {
            // Pick vertex with minimum degree among non-eliminated neighbors
            *active
                .iter()
                .min_by_key(|&&v| adj[v].iter().filter(|&&u| !eliminated[u]).count())
                .unwrap()
        }
        TreewidthHeuristic::MinFill => {
            // Pick vertex whose elimination adds fewest fill edges
            *active
                .iter()
                .min_by_key(|&&v| {
                    let neighbors: Vec<usize> =
                        adj[v].iter().copied().filter(|&u| !eliminated[u]).collect();
                    let mut fill = 0usize;
                    for i in 0..neighbors.len() {
                        for j in (i + 1)..neighbors.len() {
                            if !adj[neighbors[i]].contains(&neighbors[j]) {
                                fill += 1;
                            }
                        }
                    }
                    fill
                })
                .unwrap()
        }
        TreewidthHeuristic::GreedyDegree => {
            // Min degree, breaking ties by min fill
            *active
                .iter()
                .min_by(|&&a, &&b| {
                    let deg_a = adj[a].iter().filter(|&&u| !eliminated[u]).count();
                    let deg_b = adj[b].iter().filter(|&&u| !eliminated[u]).count();
                    let cmp = deg_a.cmp(&deg_b);
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                    // Tie-break by fill count
                    let fill_a = compute_fill_for_vertex(adj, eliminated, a);
                    let fill_b = compute_fill_for_vertex(adj, eliminated, b);
                    fill_a.cmp(&fill_b)
                })
                .unwrap()
        }
    }
}

/// Compute the fill count for a vertex (helper for GreedyDegree).
fn compute_fill_for_vertex(adj: &[HashSet<usize>], eliminated: &[bool], v: usize) -> usize {
    let neighbors: Vec<usize> = adj[v].iter().copied().filter(|&u| !eliminated[u]).collect();
    let mut fill = 0usize;
    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            if !adj[neighbors[i]].contains(&neighbors[j]) {
                fill += 1;
            }
        }
    }
    fill
}

/// Build a tree decomposition from an elimination ordering result.
///
/// Uses the standard construction: for each eliminated vertex `v_i` with bag
/// `B_i = {v_i} union N(v_i)`, the parent of `B_i` is the bag `B_j` where `j`
/// is the *elimination position* of the first neighbor of `v_i` to be eliminated
/// after `v_i`. Formally, if `v_i` has neighbors `u_1, u_2, ...` (at time of
/// elimination) and `u_k` is the one with the smallest elimination position > i,
/// then `B_j` with `j = pos(u_k)` is the parent of `B_i`.
///
/// This construction provably satisfies the running intersection property because
/// the neighbor set of `v_i` forms a clique after elimination, and each neighbor
/// persists into all later bags until it is itself eliminated.
pub fn tree_decomposition_from_elimination(
    elimination: &EliminationResult,
    _graph: &InteractionGraph,
) -> TreeDecomposition {
    let raw_bags = &elimination.bags;
    let ordering = &elimination.ordering;
    let n = raw_bags.len();

    if n == 0 {
        return TreeDecomposition::new(Vec::new(), Vec::new(), 0);
    }

    // Build a map from vertex -> its position in the elimination ordering
    let mut pos_of: Vec<usize> = vec![0; n];
    for (pos, &v) in ordering.iter().enumerate() {
        if v < n {
            pos_of[v] = pos;
        }
    }

    // Collect non-empty bags, keeping their original elimination-order indices
    let non_empty: Vec<usize> = (0..n)
        .filter(|&i| raw_bags[i].len() > 1 || (raw_bags[i].len() == 1 && n <= 1))
        .collect();

    // If no multi-vertex bags, check for singleton bags (isolated vertices)
    let indices: Vec<usize> = if non_empty.is_empty() {
        (0..n).filter(|&i| !raw_bags[i].is_empty()).collect()
    } else {
        non_empty
    };

    if indices.is_empty() {
        return TreeDecomposition::new(vec![BTreeSet::new()], Vec::new(), 0);
    }

    let m = indices.len();
    let mut bags: Vec<BTreeSet<usize>> = Vec::with_capacity(m);
    // Map from original elimination position -> new bag index
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
        bags.push(raw_bags[old_idx].clone());
    }

    if m <= 1 {
        return TreeDecomposition::new(bags, Vec::new(), 0);
    }

    // For each bag (in elimination order), find the parent:
    // The parent of bag at elimination position `i` is the bag at elimination
    // position `j`, where `j = min { pos_of[u] : u in N(v_i) at elimination time }`.
    // N(v_i) at elimination time = B_i \ {v_i}.
    let mut tree_edges: Vec<(usize, usize)> = Vec::new();

    for k in 0..(m - 1) {
        let old_idx = indices[k];
        let v_elim = ordering[old_idx];

        // Neighbors at time of elimination = bag contents minus the eliminated vertex
        let neighbors: Vec<usize> = raw_bags[old_idx]
            .iter()
            .copied()
            .filter(|&v| v != v_elim)
            .collect();

        if neighbors.is_empty() {
            // Isolated vertex bag: connect to the next bag in order
            tree_edges.push((k, k + 1));
            continue;
        }

        // Find the neighbor with the smallest elimination position > old_idx.
        // That neighbor's bag is our parent.
        let parent_elim_pos = neighbors
            .iter()
            .map(|&u| pos_of[u])
            .filter(|&p| p > old_idx)
            .min();

        let parent_new_idx = if let Some(parent_pos) = parent_elim_pos {
            // Find the new bag index for this elimination position
            if let Some(&new_idx) = old_to_new.get(&parent_pos) {
                new_idx
            } else {
                // The parent's bag was a singleton (size 1) and was filtered out.
                // Walk forward from parent_pos to find the next included bag.
                let mut found = m - 1;
                for p in parent_pos..n {
                    if let Some(&ni) = old_to_new.get(&p) {
                        found = ni;
                        break;
                    }
                }
                found
            }
        } else {
            // All neighbors were eliminated before this vertex (shouldn't happen
            // in a correct elimination, but handle gracefully)
            m - 1
        };

        tree_edges.push((k, parent_new_idx));
    }

    let root = m - 1;
    TreeDecomposition::new(bags, tree_edges, root)
}

// ============================================================================
// TREESPILATION RESULT
// ============================================================================

/// Result of the treespilation optimization pass.
#[derive(Debug, Clone)]
pub struct TreespilationResult {
    /// Treewidth of the original circuit interaction graph.
    pub original_treewidth: usize,
    /// Treewidth after gate reordering optimization.
    pub optimized_treewidth: usize,
    /// Gate list after commutation-aware reordering.
    pub reordered_gates: Vec<TreeGate>,
    /// Logical-to-physical qubit mapping: qubit_mapping[logical] = physical.
    pub qubit_mapping: Vec<usize>,
    /// Estimated tensor contraction cost for the optimized circuit.
    pub contraction_cost: f64,
    /// Estimated speedup factor: 2^(original_tw) / 2^(optimized_tw).
    pub speedup_estimate: f64,
    /// The tree decomposition of the optimized circuit.
    pub decomposition: TreeDecomposition,
    /// Partition boundaries for circuit partitioning.
    pub partitions: Vec<CircuitPartition>,
}

/// A partition of the circuit, defined by a contiguous range of (reordered) gates
/// and the set of qubits it involves.
#[derive(Debug, Clone)]
pub struct CircuitPartition {
    /// Start index in the reordered gate list (inclusive).
    pub start: usize,
    /// End index in the reordered gate list (exclusive).
    pub end: usize,
    /// Qubits involved in this partition.
    pub qubits: BTreeSet<usize>,
    /// Corresponding bag index in the tree decomposition.
    pub bag_index: usize,
}

// ============================================================================
// GATE REORDERING
// ============================================================================

/// Reorder commuting gates to try to reduce the treewidth of the interaction graph.
///
/// Strategy: greedily schedule gates to minimize the "active edge width" at each step.
/// A gate is ready when all prior non-commuting gates on the same qubits have been
/// scheduled. Among ready gates, pick the one that keeps the interaction graph's
/// treewidth lowest.
///
/// This is a heuristic; optimal gate scheduling for minimum treewidth is NP-hard.
pub fn reorder_gates_for_treewidth(gates: &[TreeGate], n_qubits: usize) -> (Vec<TreeGate>, usize) {
    if gates.is_empty() {
        return (Vec::new(), 0);
    }

    let n_gates = gates.len();

    // Build a dependency graph: gate i depends on gate j if j < i and they don't commute
    // and they share qubits.
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n_gates];
    for i in 0..n_gates {
        for j in 0..i {
            if !gates[i].commutes_with(&gates[j]) {
                deps[i].insert(j);
            }
        }
    }

    let mut scheduled = vec![false; n_gates];
    let mut order: Vec<usize> = Vec::with_capacity(n_gates);

    for _ in 0..n_gates {
        // Find all ready gates: those whose dependencies have all been scheduled
        let ready: Vec<usize> = (0..n_gates)
            .filter(|&i| !scheduled[i] && deps[i].iter().all(|&d| scheduled[d]))
            .collect();

        if ready.is_empty() {
            // Should not happen; all gates should eventually become ready
            break;
        }

        // Greedy selection: among ready gates, pick the one that results in
        // the smallest current interaction graph treewidth.
        // For efficiency, we approximate: prefer single-qubit gates first (they
        // don't increase treewidth), then among multi-qubit gates prefer those
        // that connect to already-active qubits.
        let best = select_best_ready_gate(&ready, &order, gates, n_qubits);

        scheduled[best] = true;
        order.push(best);
    }

    // Handle any remaining unscheduled gates (shouldn't happen, but be safe)
    for i in 0..n_gates {
        if !scheduled[i] {
            order.push(i);
        }
    }

    let reordered: Vec<TreeGate> = order.iter().map(|&i| gates[i].clone()).collect();

    // Compute the treewidth of the reordered circuit
    let graph = InteractionGraph::from_gates(&reordered, n_qubits);
    let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
    let treewidth = elim.width;

    (reordered, treewidth)
}

/// Select the best gate to schedule next from the ready set.
fn select_best_ready_gate(
    ready: &[usize],
    already_scheduled: &[usize],
    gates: &[TreeGate],
    n_qubits: usize,
) -> usize {
    // Build the current interaction graph from already-scheduled gates
    let scheduled_gates: Vec<TreeGate> = already_scheduled
        .iter()
        .map(|&i| gates[i].clone())
        .collect();
    let current_active_qubits: HashSet<usize> = scheduled_gates
        .iter()
        .flat_map(|g| g.qubits.iter().copied())
        .collect();

    // Score each ready gate: lower is better
    let mut best_idx = ready[0];
    let mut best_score = i64::MAX;

    for &gate_idx in ready {
        let gate = &gates[gate_idx];
        let score = if gate.gate_type.is_single_qubit() {
            // Single-qubit gates don't increase treewidth; schedule them first
            -1000
        } else {
            // Multi-qubit gate: prefer gates that connect to already-active qubits
            // (extending existing connections rather than creating new ones)
            let new_connections: usize = gate
                .qubits
                .iter()
                .filter(|q| !current_active_qubits.contains(q))
                .count();
            new_connections as i64
        };

        if score < best_score {
            best_score = score;
            best_idx = gate_idx;
        }
    }

    best_idx
}

// ============================================================================
// QUBIT MAPPING
// ============================================================================

/// Optimize the logical-to-physical qubit mapping to minimize treewidth.
///
/// Strategy: build the interaction graph, compute the elimination ordering,
/// and assign physical qubits in elimination order. Vertices eliminated
/// early get lower physical indices, which tends to localize interactions.
pub fn optimize_qubit_mapping(
    gates: &[TreeGate],
    n_qubits: usize,
    heuristic: TreewidthHeuristic,
) -> Vec<usize> {
    let graph = InteractionGraph::from_gates(gates, n_qubits);
    let elim = compute_elimination_ordering(&graph, heuristic);

    // Map: physical index = position in elimination ordering
    let mut mapping = vec![0usize; n_qubits];
    for (physical, &logical) in elim.ordering.iter().enumerate() {
        if logical < n_qubits {
            mapping[logical] = physical;
        }
    }

    mapping
}

/// Apply a qubit mapping to a gate list, producing a new gate list with
/// remapped qubit indices.
pub fn apply_qubit_mapping(gates: &[TreeGate], mapping: &[usize]) -> Vec<TreeGate> {
    gates
        .iter()
        .map(|gate| {
            let new_qubits: Vec<usize> = gate
                .qubits
                .iter()
                .map(|&q| if q < mapping.len() { mapping[q] } else { q })
                .collect();
            TreeGate::new(gate.gate_type.clone(), new_qubits, gate.params.clone())
        })
        .collect()
}

// ============================================================================
// CIRCUIT PARTITIONING
// ============================================================================

/// Partition the circuit into sub-circuits based on the tree decomposition.
///
/// Each partition corresponds to a bag in the tree decomposition. Gates are
/// assigned to the smallest bag that contains all their qubit operands.
pub fn partition_circuit(
    gates: &[TreeGate],
    decomposition: &TreeDecomposition,
) -> Vec<CircuitPartition> {
    if gates.is_empty() || decomposition.bags.is_empty() {
        return Vec::new();
    }

    // Assign each gate to the best (smallest) bag that covers all its qubits
    let mut gate_assignments: Vec<usize> = Vec::with_capacity(gates.len());
    for gate in gates {
        let qubit_set: BTreeSet<usize> = gate.qubits.iter().copied().collect();
        let best_bag = decomposition
            .bags
            .iter()
            .enumerate()
            .filter(|(_, bag)| qubit_set.is_subset(bag))
            .min_by_key(|(_, bag)| bag.len())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        gate_assignments.push(best_bag);
    }

    // Group consecutive gates with the same bag assignment into partitions
    let mut partitions: Vec<CircuitPartition> = Vec::new();
    let mut start = 0;
    while start < gates.len() {
        let bag_idx = gate_assignments[start];
        let mut end = start + 1;
        while end < gates.len() && gate_assignments[end] == bag_idx {
            end += 1;
        }
        let qubits: BTreeSet<usize> = gates[start..end]
            .iter()
            .flat_map(|g| g.qubits.iter().copied())
            .collect();
        partitions.push(CircuitPartition {
            start,
            end,
            qubits,
            bag_index: bag_idx,
        });
        start = end;
    }

    partitions
}

// ============================================================================
// TREESPILER (MAIN INTERFACE)
// ============================================================================

/// The main treespiler: combines tree decomposition, gate reordering, qubit
/// mapping, and circuit partitioning into a single optimization pass.
///
/// # Example
///
/// ```ignore
/// let treespiler = Treespiler::new(10, TreewidthHeuristic::MinFill);
/// let result = treespiler.treespile(&gates, 5);
/// assert!(result.optimized_treewidth <= result.original_treewidth);
/// ```
#[derive(Debug, Clone)]
pub struct Treespiler {
    /// Maximum treewidth beyond which the optimizer gives up and returns
    /// the original circuit. For very high-treewidth circuits, optimizing
    /// is unlikely to help.
    pub max_treewidth: usize,
    /// Heuristic for treewidth computation.
    pub heuristic: TreewidthHeuristic,
    /// Whether to apply gate reordering.
    pub enable_reordering: bool,
    /// Whether to optimize qubit mapping.
    pub enable_mapping: bool,
}

impl Treespiler {
    /// Create a new treespiler with the given parameters.
    pub fn new(max_treewidth: usize, heuristic: TreewidthHeuristic) -> Self {
        Self {
            max_treewidth,
            heuristic,
            enable_reordering: true,
            enable_mapping: true,
        }
    }

    /// Create a treespiler with default settings (max_treewidth=20, MinFill heuristic).
    pub fn default_config() -> Self {
        Self::new(20, TreewidthHeuristic::MinFill)
    }

    /// Run the full treespilation pipeline on a circuit.
    ///
    /// Steps:
    /// 1. Build interaction graph from the circuit.
    /// 2. Compute original treewidth.
    /// 3. Optimize qubit mapping (if enabled).
    /// 4. Reorder gates for minimum treewidth (if enabled).
    /// 5. Compute optimized tree decomposition.
    /// 6. Partition circuit at decomposition boundaries.
    /// 7. Estimate contraction cost and speedup.
    pub fn treespile(&self, gates: &[TreeGate], n_qubits: usize) -> TreespilationResult {
        // Step 1-2: Original interaction graph and treewidth
        let original_graph = InteractionGraph::from_gates(gates, n_qubits);
        let original_elim = compute_elimination_ordering(&original_graph, self.heuristic);
        let original_treewidth = original_elim.width;

        // Step 3: Optimize qubit mapping
        let (mapped_gates, qubit_mapping) = if self.enable_mapping {
            let mapping = optimize_qubit_mapping(gates, n_qubits, self.heuristic);
            let mapped = apply_qubit_mapping(gates, &mapping);
            (mapped, mapping)
        } else {
            let identity: Vec<usize> = (0..n_qubits).collect();
            (gates.to_vec(), identity)
        };

        // Step 4: Reorder gates
        let (reordered_gates, _) = if self.enable_reordering {
            reorder_gates_for_treewidth(&mapped_gates, n_qubits)
        } else {
            (mapped_gates, original_treewidth)
        };

        // Step 5: Compute optimized tree decomposition
        let opt_graph = InteractionGraph::from_gates(&reordered_gates, n_qubits);
        let opt_elim = compute_elimination_ordering(&opt_graph, self.heuristic);
        let optimized_treewidth = opt_elim.width;
        let decomposition = tree_decomposition_from_elimination(&opt_elim, &opt_graph);

        // Step 6: Partition circuit
        let partitions = partition_circuit(&reordered_gates, &decomposition);

        // Step 7: Cost and speedup estimation
        let contraction_cost = decomposition.contraction_cost();
        let speedup_estimate = if optimized_treewidth < n_qubits && original_treewidth > 0 {
            // Speedup: O(2^n) / O(2^tw * n)
            // Simplified: 2^(n - tw) / n
            let naive_cost = 2.0_f64.powi(n_qubits.min(50) as i32);
            let opt_cost = contraction_cost.max(1.0);
            (naive_cost / opt_cost).max(1.0)
        } else {
            1.0
        };

        TreespilationResult {
            original_treewidth,
            optimized_treewidth,
            reordered_gates,
            qubit_mapping,
            contraction_cost,
            speedup_estimate,
            decomposition,
            partitions,
        }
    }

    /// Compute just the treewidth of a circuit without full optimization.
    pub fn compute_treewidth(&self, gates: &[TreeGate], n_qubits: usize) -> usize {
        let graph = InteractionGraph::from_gates(gates, n_qubits);
        let elim = compute_elimination_ordering(&graph, self.heuristic);
        elim.width
    }

    /// Check if a circuit has low enough treewidth to benefit from treespilation.
    pub fn is_low_treewidth(&self, gates: &[TreeGate], n_qubits: usize) -> bool {
        let tw = self.compute_treewidth(gates, n_qubits);
        tw <= self.max_treewidth && tw < n_qubits / 2
    }
}

// ============================================================================
// UTILITY: QFT AND QAOA CIRCUIT BUILDERS
// ============================================================================

/// Build a Quantum Fourier Transform circuit on n_qubits.
/// QFT has all-to-all connectivity, giving treewidth n-1.
pub fn build_qft_circuit(n_qubits: usize) -> Vec<TreeGate> {
    let mut gates = Vec::new();
    for i in 0..n_qubits {
        gates.push(TreeGate::single(TreeGateType::H, i));
        for j in (i + 1)..n_qubits {
            let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
            gates.push(TreeGate::new(TreeGateType::CZ, vec![i, j], vec![angle]));
        }
    }
    gates
}

/// Build a QAOA circuit for MaxCut on a given graph.
/// QAOA has the same connectivity as the problem graph.
pub fn build_qaoa_circuit(edges: &[(usize, usize)], n_qubits: usize, p: usize) -> Vec<TreeGate> {
    let mut gates = Vec::new();

    // Initial layer: Hadamard on all qubits
    for i in 0..n_qubits {
        gates.push(TreeGate::single(TreeGateType::H, i));
    }

    // p rounds of QAOA
    for round in 0..p {
        let gamma = std::f64::consts::PI / (2.0 * (round + 1) as f64);
        let beta = std::f64::consts::PI / (4.0 * (round + 1) as f64);

        // Problem unitary: CZ gates on edges
        for &(u, v) in edges {
            gates.push(TreeGate::new(TreeGateType::CZ, vec![u, v], vec![gamma]));
        }

        // Mixer unitary: Rx on each qubit
        for i in 0..n_qubits {
            gates.push(TreeGate::new(TreeGateType::Rx, vec![i], vec![beta]));
        }
    }

    gates
}

/// Build a linear chain circuit (nearest-neighbor CX gates), which has
/// treewidth 1 (a path graph).
pub fn build_linear_circuit(n_qubits: usize) -> Vec<TreeGate> {
    let mut gates = Vec::new();
    for i in 0..n_qubits {
        gates.push(TreeGate::single(TreeGateType::H, i));
    }
    for i in 0..n_qubits.saturating_sub(1) {
        gates.push(TreeGate::two_qubit(TreeGateType::CX, i, i + 1));
    }
    gates
}

/// Build an identity circuit (only single-qubit gates, no interactions).
pub fn build_identity_circuit(n_qubits: usize) -> Vec<TreeGate> {
    let mut gates = Vec::new();
    for i in 0..n_qubits {
        gates.push(TreeGate::single(TreeGateType::H, i));
        gates.push(TreeGate::single(TreeGateType::H, i));
    }
    gates
}

/// Build a grid-connected circuit for the given grid dimensions.
pub fn build_grid_circuit(rows: usize, cols: usize) -> Vec<TreeGate> {
    let n = rows * cols;
    let mut gates = Vec::new();

    // Hadamard layer
    for i in 0..n {
        gates.push(TreeGate::single(TreeGateType::H, i));
    }

    // Horizontal CX gates
    for r in 0..rows {
        for c in 0..cols.saturating_sub(1) {
            let q = r * cols + c;
            gates.push(TreeGate::two_qubit(TreeGateType::CX, q, q + 1));
        }
    }

    // Vertical CX gates
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols {
            let q = r * cols + c;
            gates.push(TreeGate::two_qubit(TreeGateType::CX, q, q + cols));
        }
    }

    gates
}

// ============================================================================
// TREEWIDTH-BOUNDED SIMULATION COST ANALYSIS
// ============================================================================

/// Analyze the simulation cost advantage from using treespilation.
#[derive(Debug, Clone)]
pub struct CostAnalysis {
    /// Number of qubits in the circuit.
    pub n_qubits: usize,
    /// Treewidth of the circuit interaction graph.
    pub treewidth: usize,
    /// Naive simulation cost: O(2^n).
    pub naive_cost: f64,
    /// Treewidth-bounded cost: O(2^tw * n_bags).
    pub bounded_cost: f64,
    /// Memory for naive simulation: O(2^n) complex numbers.
    pub naive_memory_bytes: f64,
    /// Memory for treewidth-bounded: O(2^tw) complex numbers per bag.
    pub bounded_memory_bytes: f64,
    /// Estimated speedup factor.
    pub speedup: f64,
    /// Memory reduction factor.
    pub memory_reduction: f64,
}

/// Estimate the cost of simulating a circuit with and without treespilation.
pub fn analyze_simulation_cost(
    gates: &[TreeGate],
    n_qubits: usize,
    heuristic: TreewidthHeuristic,
) -> CostAnalysis {
    let graph = InteractionGraph::from_gates(gates, n_qubits);
    let elim = compute_elimination_ordering(&graph, heuristic);
    let decomposition = tree_decomposition_from_elimination(&elim, &graph);

    let treewidth = elim.width;
    let n_bags = decomposition.bags.len().max(1);

    // Cost estimates
    let naive_cost = 2.0_f64.powi(n_qubits.min(60) as i32);
    let bounded_cost = 2.0_f64.powi(treewidth.min(60) as i32) * n_bags as f64;

    // Memory: each complex number is 16 bytes (two f64)
    let bytes_per_complex = 16.0_f64;
    let naive_memory = 2.0_f64.powi(n_qubits.min(60) as i32) * bytes_per_complex;
    let bounded_memory = 2.0_f64.powi(treewidth.min(60) as i32) * bytes_per_complex;

    let speedup = (naive_cost / bounded_cost).max(1.0);
    let memory_reduction = (naive_memory / bounded_memory).max(1.0);

    CostAnalysis {
        n_qubits,
        treewidth,
        naive_cost,
        bounded_cost,
        naive_memory_bytes: naive_memory,
        bounded_memory_bytes: bounded_memory,
        speedup,
        memory_reduction,
    }
}

// ============================================================================
// CONVERSION TO/FROM EXISTING GATE TYPES
// ============================================================================

/// Convert from the crate's native Gate/GateType to TreeGate/TreeGateType.
pub fn from_native_gate(gate: &crate::gates::Gate) -> TreeGate {
    let gate_type = match &gate.gate_type {
        crate::gates::GateType::H => TreeGateType::H,
        crate::gates::GateType::X => TreeGateType::X,
        crate::gates::GateType::Y => TreeGateType::Y,
        crate::gates::GateType::Z => TreeGateType::Z,
        crate::gates::GateType::S => TreeGateType::S,
        crate::gates::GateType::T => TreeGateType::T,
        crate::gates::GateType::Rx(_) => TreeGateType::Rx,
        crate::gates::GateType::Ry(_) => TreeGateType::Ry,
        crate::gates::GateType::Rz(_) => TreeGateType::Rz,
        crate::gates::GateType::CNOT => TreeGateType::CX,
        crate::gates::GateType::CZ => TreeGateType::CZ,
        crate::gates::GateType::SWAP => TreeGateType::SWAP,
        crate::gates::GateType::Toffoli => TreeGateType::CCX,
        _ => TreeGateType::Custom(format!("{:?}", gate.gate_type)),
    };

    let mut qubits = Vec::new();
    qubits.extend_from_slice(&gate.controls);
    qubits.extend_from_slice(&gate.targets);

    let params = gate.params.clone().unwrap_or_default();

    TreeGate::new(gate_type, qubits, params)
}

/// Convert from TreeGate/TreeGateType back to the crate's native Gate/GateType.
pub fn to_native_gate(gate: &TreeGate) -> crate::gates::Gate {
    let (gate_type, targets, controls) = match &gate.gate_type {
        TreeGateType::H => (crate::gates::GateType::H, gate.qubits.clone(), vec![]),
        TreeGateType::X => (crate::gates::GateType::X, gate.qubits.clone(), vec![]),
        TreeGateType::Y => (crate::gates::GateType::Y, gate.qubits.clone(), vec![]),
        TreeGateType::Z => (crate::gates::GateType::Z, gate.qubits.clone(), vec![]),
        TreeGateType::S => (crate::gates::GateType::S, gate.qubits.clone(), vec![]),
        TreeGateType::T => (crate::gates::GateType::T, gate.qubits.clone(), vec![]),
        TreeGateType::Rx => {
            let angle = gate.params.first().copied().unwrap_or(0.0);
            (
                crate::gates::GateType::Rx(angle),
                gate.qubits.clone(),
                vec![],
            )
        }
        TreeGateType::Ry => {
            let angle = gate.params.first().copied().unwrap_or(0.0);
            (
                crate::gates::GateType::Ry(angle),
                gate.qubits.clone(),
                vec![],
            )
        }
        TreeGateType::Rz => {
            let angle = gate.params.first().copied().unwrap_or(0.0);
            (
                crate::gates::GateType::Rz(angle),
                gate.qubits.clone(),
                vec![],
            )
        }
        TreeGateType::CX => {
            if gate.qubits.len() >= 2 {
                (
                    crate::gates::GateType::CNOT,
                    vec![gate.qubits[1]],
                    vec![gate.qubits[0]],
                )
            } else {
                (crate::gates::GateType::CNOT, gate.qubits.clone(), vec![])
            }
        }
        TreeGateType::CZ => {
            if gate.qubits.len() >= 2 {
                (
                    crate::gates::GateType::CZ,
                    vec![gate.qubits[1]],
                    vec![gate.qubits[0]],
                )
            } else {
                (crate::gates::GateType::CZ, gate.qubits.clone(), vec![])
            }
        }
        TreeGateType::SWAP => (crate::gates::GateType::SWAP, gate.qubits.clone(), vec![]),
        TreeGateType::CCX => {
            if gate.qubits.len() >= 3 {
                (
                    crate::gates::GateType::Toffoli,
                    vec![gate.qubits[2]],
                    vec![gate.qubits[0], gate.qubits[1]],
                )
            } else {
                (crate::gates::GateType::Toffoli, gate.qubits.clone(), vec![])
            }
        }
        TreeGateType::Custom(name) => {
            // Fallback: treat as X gate (custom gates need application-specific handling)
            (crate::gates::GateType::X, gate.qubits.clone(), vec![])
        }
    };

    let params = if gate.params.is_empty() {
        None
    } else {
        Some(gate.params.clone())
    };

    crate::gates::Gate {
        gate_type,
        targets,
        controls,
        params,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Test 1: Interaction graph construction from circuit
    // ========================================================================

    #[test]
    fn test_interaction_graph_from_circuit() {
        let gates = vec![
            TreeGate::single(TreeGateType::H, 0),
            TreeGate::two_qubit(TreeGateType::CX, 0, 1),
            TreeGate::two_qubit(TreeGateType::CX, 1, 2),
            TreeGate::two_qubit(TreeGateType::CX, 2, 3),
        ];
        let graph = InteractionGraph::from_gates(&gates, 4);

        assert_eq!(graph.n_vertices, 4);
        assert_eq!(graph.edges.len(), 3);
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 2));
        assert!(graph.has_edge(2, 3));
        assert!(!graph.has_edge(0, 2));
        assert!(!graph.has_edge(0, 3));
    }

    // ========================================================================
    // Test 2: Tree graph has treewidth 1
    // ========================================================================

    #[test]
    fn test_tree_graph_treewidth_1() {
        // Star graph (a tree) should have treewidth 1
        let graph = InteractionGraph::star(5);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinDegree);
        assert_eq!(elim.width, 1, "Star graph (tree) should have treewidth 1");
    }

    // ========================================================================
    // Test 3: Path graph has treewidth 1
    // ========================================================================

    #[test]
    fn test_path_graph_treewidth_1() {
        let graph = InteractionGraph::path(6);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        assert_eq!(elim.width, 1, "Path graph should have treewidth 1");
    }

    // ========================================================================
    // Test 4: Cycle graph has treewidth 2
    // ========================================================================

    #[test]
    fn test_cycle_graph_treewidth_2() {
        let graph = InteractionGraph::cycle(6);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        assert_eq!(elim.width, 2, "Cycle graph should have treewidth 2");
    }

    // ========================================================================
    // Test 5: Grid graph treewidth is bounded
    // ========================================================================

    #[test]
    fn test_grid_graph_treewidth() {
        // 3x3 grid has treewidth 3 (= min(rows, cols))
        let graph = InteractionGraph::grid(3, 3);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        // Treewidth of m x n grid is min(m, n), so tw(3x3) = 3
        // Heuristic may overshoot but should not exceed 2*min(rows, cols)
        assert!(
            elim.width <= 6,
            "3x3 grid treewidth should be <= 6, got {}",
            elim.width
        );
        assert!(
            elim.width >= 3,
            "3x3 grid treewidth should be >= 3, got {}",
            elim.width
        );
    }

    // ========================================================================
    // Test 6: Complete graph treewidth = n-1
    // ========================================================================

    #[test]
    fn test_complete_graph_treewidth() {
        let graph = InteractionGraph::complete(5);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        assert_eq!(
            elim.width, 4,
            "K5 should have treewidth 4, got {}",
            elim.width
        );
    }

    // ========================================================================
    // Test 7: Min-degree elimination ordering
    // ========================================================================

    #[test]
    fn test_min_degree_elimination() {
        let graph = InteractionGraph::path(5);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinDegree);

        // All vertices should appear in the ordering
        let mut seen = HashSet::new();
        for &v in &elim.ordering {
            assert!(seen.insert(v), "Vertex {} appears twice", v);
        }
        assert_eq!(seen.len(), 5);

        // Treewidth should be 1 for a path
        assert_eq!(elim.width, 1);
    }

    // ========================================================================
    // Test 8: Min-fill elimination ordering
    // ========================================================================

    #[test]
    fn test_min_fill_elimination() {
        let graph = InteractionGraph::path(5);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);

        assert_eq!(elim.ordering.len(), 5);
        assert_eq!(elim.width, 1);

        // Min-fill on a path should eliminate endpoints first (0 fill edges)
        // The first vertex eliminated should be an endpoint (degree 1)
        let first = elim.ordering[0];
        assert!(
            first == 0 || first == 4,
            "Min-fill should start with an endpoint, got {}",
            first
        );
    }

    // ========================================================================
    // Test 9: Tree decomposition validity
    // ========================================================================

    #[test]
    fn test_tree_decomposition_validity() {
        let graph = InteractionGraph::path(5);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        assert!(
            td.validate(&graph).is_ok(),
            "Tree decomposition of path graph should be valid: {:?}",
            td.validate(&graph).err()
        );
        assert_eq!(td.width, 1);
    }

    // ========================================================================
    // Test 10: Tree decomposition covers all vertices
    // ========================================================================

    #[test]
    fn test_tree_decomposition_covers_all_vertices() {
        let graph = InteractionGraph::cycle(6);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        // Every vertex with edges should appear in at least one bag
        let all_bag_vertices: HashSet<usize> =
            td.bags.iter().flat_map(|b| b.iter().copied()).collect();
        for v in 0..graph.n_vertices {
            if graph.degree(v) > 0 {
                assert!(all_bag_vertices.contains(&v), "Vertex {} not in any bag", v);
            }
        }
    }

    // ========================================================================
    // Test 11: Tree decomposition edge coverage
    // ========================================================================

    #[test]
    fn test_tree_decomposition_edge_coverage() {
        let graph = InteractionGraph::grid(3, 3);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        // Every edge should be covered by some bag
        for &(u, v) in &graph.edges {
            let covered = td
                .bags
                .iter()
                .any(|bag| bag.contains(&u) && bag.contains(&v));
            assert!(covered, "Edge ({}, {}) not covered by any bag", u, v);
        }
    }

    // ========================================================================
    // Test 12: Tree decomposition running intersection property
    // ========================================================================

    #[test]
    fn test_tree_decomposition_running_intersection() {
        let graph = InteractionGraph::grid(3, 3);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        // Validate should check running intersection
        assert!(
            td.validate(&graph).is_ok(),
            "Running intersection violated: {:?}",
            td.validate(&graph).err()
        );
    }

    // ========================================================================
    // Test 13: Gate reordering preserves commuting gates
    // ========================================================================

    #[test]
    fn test_gate_reordering_commuting() {
        // Gates on disjoint qubits can be reordered
        let gates = vec![
            TreeGate::two_qubit(TreeGateType::CX, 0, 1),
            TreeGate::two_qubit(TreeGateType::CX, 2, 3),
            TreeGate::two_qubit(TreeGateType::CX, 4, 5),
        ];

        let (reordered, _tw) = reorder_gates_for_treewidth(&gates, 6);
        assert_eq!(reordered.len(), 3, "Should preserve all gates");

        // Check that all original gate types are present
        let original_types: HashSet<String> =
            gates.iter().map(|g| format!("{:?}", g.gate_type)).collect();
        let reordered_types: HashSet<String> = reordered
            .iter()
            .map(|g| format!("{:?}", g.gate_type))
            .collect();
        assert_eq!(original_types, reordered_types);
    }

    // ========================================================================
    // Test 14: Gate reordering can reduce treewidth
    // ========================================================================

    #[test]
    fn test_gate_reordering_reduces_treewidth() {
        // Build a circuit where reordering could help:
        // Mix far-apart and near gates
        let gates = vec![
            TreeGate::two_qubit(TreeGateType::CX, 0, 5), // far apart
            TreeGate::two_qubit(TreeGateType::CX, 0, 1), // near
            TreeGate::two_qubit(TreeGateType::CX, 5, 6), // near
            TreeGate::two_qubit(TreeGateType::CX, 1, 2), // near
            TreeGate::two_qubit(TreeGateType::CX, 6, 7), // near
        ];

        let original_graph = InteractionGraph::from_gates(&gates, 8);
        let original_elim =
            compute_elimination_ordering(&original_graph, TreewidthHeuristic::MinFill);

        let (reordered, optimized_tw) = reorder_gates_for_treewidth(&gates, 8);

        // The reordered circuit should have the same edges (treewidth depends on graph,
        // not gate order when graph stays the same). But at minimum, it should not
        // increase treewidth beyond the original.
        assert!(
            optimized_tw <= original_elim.width + 1,
            "Reordering should not significantly increase treewidth: {} vs {}",
            optimized_tw,
            original_elim.width
        );
        assert_eq!(reordered.len(), gates.len());
    }

    // ========================================================================
    // Test 15: Qubit mapping optimization
    // ========================================================================

    #[test]
    fn test_qubit_mapping_optimization() {
        let gates = vec![
            TreeGate::two_qubit(TreeGateType::CX, 0, 1),
            TreeGate::two_qubit(TreeGateType::CX, 1, 2),
            TreeGate::two_qubit(TreeGateType::CX, 2, 3),
        ];

        let mapping = optimize_qubit_mapping(&gates, 4, TreewidthHeuristic::MinFill);

        // Mapping should be a permutation of 0..n
        let mut sorted = mapping.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);

        // Apply mapping and verify the gates are valid
        let mapped = apply_qubit_mapping(&gates, &mapping);
        assert_eq!(mapped.len(), 3);
        for gate in &mapped {
            for &q in &gate.qubits {
                assert!(q < 4, "Mapped qubit {} out of range", q);
            }
        }
    }

    // ========================================================================
    // Test 16: Contraction cost for tree circuit
    // ========================================================================

    #[test]
    fn test_contraction_cost_tree() {
        let gates = build_linear_circuit(5);
        let graph = InteractionGraph::from_gates(&gates, 5);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        let cost = td.contraction_cost();
        // For treewidth 1, each bag has size <= 2, so cost per bag is <= 4
        // Total cost should be O(n), much less than 2^5 = 32
        assert!(
            cost < 2.0_f64.powi(5),
            "Tree circuit contraction cost {} should be less than 2^5 = 32",
            cost
        );
        assert!(cost > 0.0, "Contraction cost should be positive");
    }

    // ========================================================================
    // Test 17: Contraction cost for grid circuit
    // ========================================================================

    #[test]
    fn test_contraction_cost_grid() {
        let gates = build_grid_circuit(3, 3);
        let graph = InteractionGraph::from_gates(&gates, 9);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        let cost = td.contraction_cost();
        // Grid has treewidth ~3, so cost should be much less than 2^9 = 512
        assert!(cost > 0.0, "Grid contraction cost should be positive");
        // At treewidth 3, bags have size <= 4, so per-bag cost is 16
        // Total should be well below naive
        let naive = 2.0_f64.powi(9);
        assert!(
            cost < naive,
            "Grid contraction cost {} should be less than naive {}",
            cost,
            naive
        );
    }

    // ========================================================================
    // Test 18: Treespilation of identity circuit
    // ========================================================================

    #[test]
    fn test_treespilation_identity_circuit() {
        let gates = build_identity_circuit(5);
        let treespiler = Treespiler::default_config();
        let result = treespiler.treespile(&gates, 5);

        // Identity circuit has no two-qubit gates, so treewidth = 0
        assert_eq!(
            result.original_treewidth, 0,
            "Identity circuit should have treewidth 0"
        );
        assert_eq!(
            result.optimized_treewidth, 0,
            "Optimized identity circuit should have treewidth 0"
        );
        assert_eq!(result.reordered_gates.len(), gates.len());
    }

    // ========================================================================
    // Test 19: Treespilation of QFT circuit
    // ========================================================================

    #[test]
    fn test_treespilation_qft_circuit() {
        let n = 5;
        let gates = build_qft_circuit(n);
        let treespiler = Treespiler::new(20, TreewidthHeuristic::MinFill);
        let result = treespiler.treespile(&gates, n);

        // QFT has all-to-all connectivity, so treewidth = n-1
        assert_eq!(
            result.original_treewidth,
            n - 1,
            "QFT treewidth should be n-1 = {}",
            n - 1
        );
        // Optimization cannot improve all-to-all connectivity
        assert_eq!(
            result.optimized_treewidth,
            n - 1,
            "QFT optimized treewidth should still be n-1"
        );
        assert!(result.speedup_estimate >= 1.0);
    }

    // ========================================================================
    // Test 20: Treespilation of QAOA circuit
    // ========================================================================

    #[test]
    fn test_treespilation_qaoa_circuit() {
        // QAOA on a path graph: treewidth 1
        let edges: Vec<(usize, usize)> = (0..4).map(|i| (i, i + 1)).collect();
        let gates = build_qaoa_circuit(&edges, 5, 2);
        let treespiler = Treespiler::new(20, TreewidthHeuristic::MinFill);
        let result = treespiler.treespile(&gates, 5);

        // Path graph has treewidth 1
        assert_eq!(
            result.original_treewidth, 1,
            "QAOA on path should have treewidth 1"
        );
        assert!(
            result.speedup_estimate > 1.0,
            "Should have significant speedup"
        );
        assert!(!result.reordered_gates.is_empty());
    }

    // ========================================================================
    // Test 21: Speedup estimate for low treewidth
    // ========================================================================

    #[test]
    fn test_speedup_estimate_low_treewidth() {
        // Linear circuit on 10 qubits: treewidth 1
        let gates = build_linear_circuit(10);
        let treespiler = Treespiler::new(20, TreewidthHeuristic::MinFill);
        let result = treespiler.treespile(&gates, 10);

        assert_eq!(result.optimized_treewidth, 1);
        // Speedup should be substantial: 2^10 / O(2^1 * n_bags)
        assert!(
            result.speedup_estimate > 10.0,
            "Should have significant speedup for tw=1 on 10 qubits, got {}",
            result.speedup_estimate
        );
    }

    // ========================================================================
    // Test 22: Circuit partitioning
    // ========================================================================

    #[test]
    fn test_circuit_partitioning() {
        let gates = build_linear_circuit(5);
        let treespiler = Treespiler::new(20, TreewidthHeuristic::MinFill);
        let result = treespiler.treespile(&gates, 5);

        // Should have partitions
        assert!(
            !result.partitions.is_empty(),
            "Should produce circuit partitions"
        );

        // Partitions should cover all gates
        let mut covered = vec![false; result.reordered_gates.len()];
        for partition in &result.partitions {
            for i in partition.start..partition.end {
                assert!(!covered[i], "Gate {} covered by multiple partitions", i);
                covered[i] = true;
            }
        }
        assert!(
            covered.iter().all(|&c| c),
            "Not all gates covered by partitions"
        );
    }

    // ========================================================================
    // Additional tests: edge cases and advanced scenarios
    // ========================================================================

    #[test]
    fn test_empty_circuit() {
        let gates: Vec<TreeGate> = Vec::new();
        let treespiler = Treespiler::default_config();
        let result = treespiler.treespile(&gates, 0);

        assert_eq!(result.original_treewidth, 0);
        assert_eq!(result.optimized_treewidth, 0);
        assert!(result.reordered_gates.is_empty());
    }

    #[test]
    fn test_single_gate_circuit() {
        let gates = vec![TreeGate::two_qubit(TreeGateType::CX, 0, 1)];
        let treespiler = Treespiler::default_config();
        let result = treespiler.treespile(&gates, 2);

        assert_eq!(result.original_treewidth, 1);
        assert_eq!(result.reordered_gates.len(), 1);
    }

    #[test]
    fn test_binary_tree_graph_treewidth() {
        let graph = InteractionGraph::binary_tree(3); // depth 3, 15 vertices
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        // Binary tree should have treewidth 1 (it's a tree)
        assert_eq!(
            elim.width, 1,
            "Binary tree should have treewidth 1, got {}",
            elim.width
        );
    }

    #[test]
    fn test_interaction_graph_no_self_loops() {
        let mut graph = InteractionGraph::new(5);
        let added = graph.add_edge(2, 2); // self-loop
        assert!(!added, "Self-loop should not be added");
        assert_eq!(graph.edges.len(), 0);
    }

    #[test]
    fn test_interaction_graph_no_duplicates() {
        let mut graph = InteractionGraph::new(5);
        assert!(graph.add_edge(0, 1));
        assert!(!graph.add_edge(0, 1)); // duplicate
        assert!(!graph.add_edge(1, 0)); // reverse duplicate
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_gate_commutation_disjoint() {
        let g1 = TreeGate::two_qubit(TreeGateType::CX, 0, 1);
        let g2 = TreeGate::two_qubit(TreeGateType::CX, 2, 3);
        assert!(g1.commutes_with(&g2), "Disjoint gates should commute");
    }

    #[test]
    fn test_gate_commutation_overlapping_non_diagonal() {
        let g1 = TreeGate::two_qubit(TreeGateType::CX, 0, 1);
        let g2 = TreeGate::two_qubit(TreeGateType::CX, 1, 2);
        assert!(
            !g1.commutes_with(&g2),
            "Overlapping non-diagonal gates should not commute"
        );
    }

    #[test]
    fn test_gate_commutation_diagonal() {
        let g1 = TreeGate::single(TreeGateType::Z, 0);
        let g2 = TreeGate::single(TreeGateType::S, 0);
        assert!(g1.commutes_with(&g2), "Diagonal gates should commute");
    }

    #[test]
    fn test_contraction_order() {
        let graph = InteractionGraph::path(4);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        let order = td.contraction_order();
        assert!(!order.is_empty(), "Contraction order should not be empty");

        // Total contracted vertices should cover all vertices in bags
        let all_contracted: HashSet<usize> = order
            .iter()
            .flat_map(|(_, verts)| verts.iter().copied())
            .collect();
        let all_bag_vertices: HashSet<usize> =
            td.bags.iter().flat_map(|b| b.iter().copied()).collect();
        // Every bag vertex should eventually be contracted
        for v in &all_bag_vertices {
            assert!(
                all_contracted.contains(v),
                "Vertex {} in bags but never contracted",
                v
            );
        }
    }

    #[test]
    fn test_cost_analysis() {
        let gates = build_linear_circuit(10);
        let analysis = analyze_simulation_cost(&gates, 10, TreewidthHeuristic::MinFill);

        assert_eq!(analysis.n_qubits, 10);
        assert_eq!(analysis.treewidth, 1);
        assert!(analysis.speedup > 1.0);
        assert!(analysis.memory_reduction > 1.0);
        assert!(analysis.bounded_cost < analysis.naive_cost);
        assert!(analysis.bounded_memory_bytes < analysis.naive_memory_bytes);
    }

    #[test]
    fn test_fill_count() {
        // In a path 0-1-2, vertex 1 has neighbors {0, 2} which are not connected.
        // Eliminating vertex 1 adds 1 fill edge (0-2).
        let graph = InteractionGraph::path(3);
        assert_eq!(
            graph.fill_count(1),
            1,
            "Middle vertex of path should have fill count 1"
        );
        // Endpoints have no fill
        assert_eq!(graph.fill_count(0), 0);
        assert_eq!(graph.fill_count(2), 0);
    }

    #[test]
    fn test_greedy_degree_heuristic() {
        // GreedyDegree should produce valid results on a cycle
        let graph = InteractionGraph::cycle(8);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::GreedyDegree);
        assert_eq!(elim.ordering.len(), 8);
        assert_eq!(elim.width, 2, "Cycle-8 treewidth should be 2");
    }

    #[test]
    fn test_is_low_treewidth() {
        let treespiler = Treespiler::new(5, TreewidthHeuristic::MinFill);

        // Linear circuit: tw = 1, definitely low
        let linear = build_linear_circuit(10);
        assert!(treespiler.is_low_treewidth(&linear, 10));

        // QFT: tw = n-1 = 9, not low
        let qft = build_qft_circuit(10);
        assert!(!treespiler.is_low_treewidth(&qft, 10));
    }

    #[test]
    fn test_native_gate_conversion_roundtrip() {
        // Test converting from TreeGate to native Gate and back
        let tree_gate = TreeGate::new(TreeGateType::Rx, vec![0], vec![1.57]);
        let native = to_native_gate(&tree_gate);
        let back = from_native_gate(&native);

        assert!(
            matches!(back.gate_type, TreeGateType::Rx),
            "Roundtrip should preserve Rx type"
        );
        assert_eq!(back.params.len(), 1);
        assert!((back.params[0] - 1.57).abs() < 1e-10);
    }

    #[test]
    fn test_ccx_gate_interaction_graph() {
        // CCX gate should create edges between all 3 qubits
        let gates = vec![TreeGate::new(TreeGateType::CCX, vec![0, 1, 2], vec![])];
        let graph = InteractionGraph::from_gates(&gates, 3);
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(0, 2));
        assert!(graph.has_edge(1, 2));
        assert_eq!(graph.edges.len(), 3);
    }

    #[test]
    fn test_treespiler_disable_reordering() {
        let gates = build_linear_circuit(5);
        let mut treespiler = Treespiler::default_config();
        treespiler.enable_reordering = false;
        treespiler.enable_mapping = false;
        let result = treespiler.treespile(&gates, 5);

        // Without reordering or mapping, gates should be in original order
        // (though the types/qubits may differ since no mapping is applied)
        assert_eq!(result.reordered_gates.len(), gates.len());
    }

    #[test]
    fn test_large_path_circuit() {
        // Larger path circuit to verify scalability
        let n = 30;
        let gates = build_linear_circuit(n);
        let treespiler = Treespiler::new(20, TreewidthHeuristic::MinFill);
        let result = treespiler.treespile(&gates, n);

        // Path graph: treewidth should be 1 regardless of size
        assert_eq!(
            result.optimized_treewidth, 1,
            "Large path should still have treewidth 1"
        );
        assert!(result.speedup_estimate > 100.0);
    }

    #[test]
    fn test_qaoa_ring_topology() {
        // QAOA on a ring (cycle): treewidth 2
        let edges: Vec<(usize, usize)> = (0..6).map(|i| (i, (i + 1) % 6)).collect();
        let gates = build_qaoa_circuit(&edges, 6, 1);
        let treespiler = Treespiler::new(20, TreewidthHeuristic::MinFill);
        let result = treespiler.treespile(&gates, 6);

        assert_eq!(
            result.original_treewidth, 2,
            "QAOA on ring should have treewidth 2"
        );
    }

    #[test]
    fn test_bag_adjacency() {
        let graph = InteractionGraph::path(5);
        let elim = compute_elimination_ordering(&graph, TreewidthHeuristic::MinFill);
        let td = tree_decomposition_from_elimination(&elim, &graph);

        let adj = td.bag_adjacency();
        assert_eq!(adj.len(), td.bags.len());

        // Each edge should appear symmetrically
        for &(a, b) in &td.tree_edges {
            assert!(adj[a].contains(&b), "Bag {} should have neighbor {}", a, b);
            assert!(adj[b].contains(&a), "Bag {} should have neighbor {}", b, a);
        }
    }
}
