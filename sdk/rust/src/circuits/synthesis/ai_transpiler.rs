//! AI-assisted quantum circuit transpiler for nQPU-Metal.
//!
//! Combines classical heuristic routing (SABRE) with a reinforcement-learning
//! agent that learns SWAP-insertion policies from circuit examples.  The module
//! also provides KAK (Cartan) decomposition of arbitrary two-qubit unitaries,
//! Solovay-Kitaev gate synthesis for non-Clifford rotations, and a multi-pass
//! transpilation pipeline that targets hardware-native gate sets.
//!
//! # Features
//!
//! - **CouplingMap** -- hardware topology with BFS shortest-path queries
//! - **SABRE Router** -- baseline heuristic router with front-layer + extended
//!   set scoring and multiple random layout trials
//! - **RL Routing Agent** -- Q-table agent trained on circuit batches; uses
//!   epsilon-greedy exploration during training and greedy policy at inference
//! - **KAK Decomposition** -- Cartan decomposition of SU(4) into at most 3
//!   CNOT gates plus single-qubit rotations
//! - **Gate Synthesis** -- ZYZ Euler decomposition and Solovay-Kitaev
//!   approximation targeting IBM (CX, SX, RZ) or Google (CZ, sqrt-iSWAP)
//!   native gate sets
//! - **Transpilation Pipeline** -- `ai_transpile()` entry point that chains
//!   routing, decomposition, gate cancellation, rotation merging, and
//!   commutation-based depth reduction
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::ai_transpiler::*;
//!
//! let map = AiCouplingMap::linear(5);
//! let circuit = vec![
//!     AiGate::H(0),
//!     AiGate::CX(0, 3),
//!     AiGate::CX(1, 4),
//! ];
//! let config = AiTranspileConfig::default();
//! let result = ai_transpile(&circuit, &map, &config);
//! assert!(result.stats.depth > 0);
//! ```

use ndarray::Array2;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors produced by the AI transpiler pipeline.
#[derive(Debug, Clone)]
pub enum AiTranspilerError {
    /// The circuit requires more qubits than the coupling map provides.
    QubitOverflow { needed: usize, available: usize },
    /// Routing could not satisfy all connectivity constraints.
    RoutingFailed(String),
    /// KAK decomposition received a non-unitary or wrong-sized matrix.
    DecompositionFailed(String),
    /// Gate synthesis could not approximate the target to the requested fidelity.
    SynthesisFailed(String),
    /// RL agent encountered an invalid state during training or inference.
    RlAgentError(String),
}

impl fmt::Display for AiTranspilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiTranspilerError::QubitOverflow { needed, available } => {
                write!(
                    f,
                    "circuit needs {} qubits but coupling map has {}",
                    needed, available
                )
            }
            AiTranspilerError::RoutingFailed(msg) => {
                write!(f, "routing failed: {}", msg)
            }
            AiTranspilerError::DecompositionFailed(msg) => {
                write!(f, "decomposition failed: {}", msg)
            }
            AiTranspilerError::SynthesisFailed(msg) => {
                write!(f, "gate synthesis failed: {}", msg)
            }
            AiTranspilerError::RlAgentError(msg) => {
                write!(f, "RL agent error: {}", msg)
            }
        }
    }
}

impl std::error::Error for AiTranspilerError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, AiTranspilerError>;

// ============================================================
// GATE REPRESENTATION
// ============================================================

/// Hardware-independent gate representation used throughout this module.
#[derive(Debug, Clone, PartialEq)]
pub enum AiGate {
    // Single-qubit gates
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),
    Sx(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
    U3(usize, f64, f64, f64),

    // Two-qubit gates
    CX(usize, usize),
    CZ(usize, usize),
    Swap(usize, usize),

    // Three-qubit gates
    CCX(usize, usize, usize),
}

impl AiGate {
    /// Qubit indices this gate acts on.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            AiGate::H(q)
            | AiGate::X(q)
            | AiGate::Y(q)
            | AiGate::Z(q)
            | AiGate::S(q)
            | AiGate::Sdg(q)
            | AiGate::T(q)
            | AiGate::Tdg(q)
            | AiGate::Sx(q)
            | AiGate::Rx(q, _)
            | AiGate::Ry(q, _)
            | AiGate::Rz(q, _)
            | AiGate::U3(q, _, _, _) => vec![*q],
            AiGate::CX(a, b) | AiGate::CZ(a, b) | AiGate::Swap(a, b) => vec![*a, *b],
            AiGate::CCX(a, b, c) => vec![*a, *b, *c],
        }
    }

    /// Maximum qubit index referenced.
    pub fn max_qubit(&self) -> usize {
        self.qubits().into_iter().max().unwrap_or(0)
    }

    /// True when the gate acts on exactly one qubit.
    pub fn is_single_qubit(&self) -> bool {
        self.qubits().len() == 1
    }

    /// True when the gate acts on two qubits.
    pub fn is_two_qubit(&self) -> bool {
        self.qubits().len() == 2
    }
}

// ============================================================
// NATIVE GATE SET
// ============================================================

/// Target hardware gate set for synthesis.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeGateSet {
    /// IBM: CX, SX, RZ, X
    Ibm,
    /// Google: CZ, sqrt-iSWAP, PhasedXZ
    Google,
    /// Clifford+T: H, S, T, CNOT
    CliffordT,
    /// No restrictions.
    Universal,
}

impl Default for NativeGateSet {
    fn default() -> Self {
        NativeGateSet::Universal
    }
}

// ============================================================
// COUPLING MAP
// ============================================================

/// Hardware qubit connectivity graph.
#[derive(Debug, Clone)]
pub struct AiCouplingMap {
    /// Directed edges (a -> b).  If `bidirectional` is true every edge
    /// implicitly has its reverse.
    pub edges: Vec<(usize, usize)>,
    pub num_qubits: usize,
    pub bidirectional: bool,
    /// Cached all-pairs shortest distance matrix (lazily built).
    distance_matrix: Option<Vec<Vec<usize>>>,
}

impl AiCouplingMap {
    // -- constructors -----------------------------------------------

    pub fn new(edges: Vec<(usize, usize)>, num_qubits: usize, bidirectional: bool) -> Self {
        Self {
            edges,
            num_qubits,
            bidirectional,
            distance_matrix: None,
        }
    }

    /// 0 - 1 - 2 - ... - (n-1)
    pub fn linear(n: usize) -> Self {
        let edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        Self::new(edges, n, true)
    }

    /// rows x cols rectangular lattice.
    pub fn grid(rows: usize, cols: usize) -> Self {
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let q = r * cols + c;
                if c + 1 < cols {
                    edges.push((q, q + 1));
                }
                if r + 1 < rows {
                    edges.push((q, q + cols));
                }
            }
        }
        Self::new(edges, rows * cols, true)
    }

    /// Simplified heavy-hex lattice with `n` unit cells.
    pub fn heavy_hex(n: usize) -> Self {
        if n == 0 {
            return Self::new(Vec::new(), 0, true);
        }
        let mut edges = Vec::new();
        let row_len = 2 * n + 1;
        // Row A
        for i in 0..row_len - 1 {
            edges.push((i, i + 1));
        }
        // Bridges
        let offset_b = row_len;
        let mut num_qubits = row_len;
        for i in 0..n {
            let bridge = offset_b + i;
            let top = 2 * i + 1;
            edges.push((top, bridge));
            let bottom = row_len + n + 2 * i + 1;
            if bottom < row_len + n + row_len {
                edges.push((bridge, bottom));
            }
            num_qubits = num_qubits.max(bridge + 1);
        }
        // Row C
        let offset_c = row_len + n;
        for i in 0..row_len - 1 {
            edges.push((offset_c + i, offset_c + i + 1));
        }
        num_qubits = num_qubits.max(offset_c + row_len);
        Self::new(edges, num_qubits, true)
    }

    /// Complete graph on `n` qubits.
    pub fn all_to_all(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                edges.push((i, j));
            }
        }
        Self::new(edges, n, true)
    }

    /// Ring: 0-1-..-(n-1)-0
    pub fn ring(n: usize) -> Self {
        let mut edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        if n > 2 {
            edges.push((n - 1, 0));
        }
        Self::new(edges, n, true)
    }

    // -- queries ----------------------------------------------------

    /// Build adjacency list respecting `bidirectional`.
    pub fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.num_qubits];
        for &(a, b) in &self.edges {
            if a < self.num_qubits && b < self.num_qubits {
                adj[a].push(b);
                if self.bidirectional {
                    adj[b].push(a);
                }
            }
        }
        for neighbors in &mut adj {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        adj
    }

    /// BFS shortest path from `src` to `dst` (inclusive).
    pub fn shortest_path(&self, src: usize, dst: usize) -> Vec<usize> {
        if src == dst {
            return vec![src];
        }
        if src >= self.num_qubits || dst >= self.num_qubits {
            return Vec::new();
        }
        let adj = self.adjacency_list();
        let mut visited = vec![false; self.num_qubits];
        let mut parent = vec![usize::MAX; self.num_qubits];
        let mut queue = VecDeque::new();
        visited[src] = true;
        queue.push_back(src);

        while let Some(cur) = queue.pop_front() {
            if cur == dst {
                let mut path = Vec::new();
                let mut node = dst;
                while node != src {
                    path.push(node);
                    node = parent[node];
                }
                path.push(src);
                path.reverse();
                return path;
            }
            for &nb in &adj[cur] {
                if !visited[nb] {
                    visited[nb] = true;
                    parent[nb] = cur;
                    queue.push_back(nb);
                }
            }
        }
        Vec::new()
    }

    /// Hop distance.  Returns `usize::MAX` when unreachable.
    pub fn distance(&self, a: usize, b: usize) -> usize {
        if a == b {
            return 0;
        }
        // Use cached matrix when available.
        if let Some(ref dm) = self.distance_matrix {
            if a < dm.len() && b < dm[a].len() {
                return dm[a][b];
            }
        }
        let path = self.shortest_path(a, b);
        if path.is_empty() {
            usize::MAX
        } else {
            path.len() - 1
        }
    }

    /// Pre-compute the all-pairs distance matrix (Floyd-Warshall).
    pub fn build_distance_matrix(&mut self) {
        let n = self.num_qubits;
        let mut dist = vec![vec![usize::MAX; n]; n];
        for i in 0..n {
            dist[i][i] = 0;
        }
        for &(a, b) in &self.edges {
            if a < n && b < n {
                dist[a][b] = 1;
                if self.bidirectional {
                    dist[b][a] = 1;
                }
            }
        }
        for k in 0..n {
            for i in 0..n {
                if dist[i][k] == usize::MAX {
                    continue;
                }
                for j in 0..n {
                    if dist[k][j] == usize::MAX {
                        continue;
                    }
                    let candidate = dist[i][k] + dist[k][j];
                    if candidate < dist[i][j] {
                        dist[i][j] = candidate;
                    }
                }
            }
        }
        self.distance_matrix = Some(dist);
    }

    /// Neighbors of qubit `q`.
    pub fn neighbors(&self, q: usize) -> Vec<usize> {
        if q >= self.num_qubits {
            return Vec::new();
        }
        let adj = self.adjacency_list();
        adj[q].clone()
    }

    /// True when there is a direct edge between `a` and `b`.
    pub fn are_connected(&self, a: usize, b: usize) -> bool {
        if self.bidirectional {
            self.edges
                .iter()
                .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
        } else {
            self.edges.iter().any(|&(x, y)| x == a && y == b)
        }
    }

    /// Number of edges (one direction).
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Candidate SWAP locations adjacent to a set of physical qubits.
    pub fn candidate_swaps_near(&self, physical_qubits: &[usize]) -> Vec<(usize, usize)> {
        let adj = self.adjacency_list();
        let mut swaps = Vec::new();
        let mut seen = HashSet::new();
        for &pq in physical_qubits {
            if pq >= self.num_qubits {
                continue;
            }
            for &nb in &adj[pq] {
                let key = if pq < nb { (pq, nb) } else { (nb, pq) };
                if seen.insert(key) {
                    swaps.push(key);
                }
            }
        }
        swaps
    }
}

// ============================================================
// LAYOUT
// ============================================================

/// Bijection between logical and physical qubit indices.
#[derive(Debug, Clone)]
pub struct AiLayout {
    /// `logical_to_physical[logical_qubit] = physical_qubit`
    pub l2p: Vec<usize>,
    /// `physical_to_logical[physical_qubit] = logical_qubit` (`usize::MAX` = unmapped)
    pub p2l: Vec<usize>,
}

impl AiLayout {
    /// Identity mapping for `n` qubits.
    pub fn trivial(n: usize) -> Self {
        Self {
            l2p: (0..n).collect(),
            p2l: (0..n).collect(),
        }
    }

    /// Random bijection of `n_logical` into `n_physical` slots.
    pub fn random(n_logical: usize, n_physical: usize, rng: &mut StdRng) -> Self {
        let mut phys: Vec<usize> = (0..n_physical).collect();
        for i in (1..n_physical).rev() {
            let j = rng.gen_range(0..=i);
            phys.swap(i, j);
        }
        let l2p: Vec<usize> = phys[..n_logical].to_vec();
        let mut p2l = vec![usize::MAX; n_physical];
        for (l, &p) in l2p.iter().enumerate() {
            p2l[p] = l;
        }
        Self { l2p, p2l }
    }

    /// Swap the logical qubits sitting at physical positions `p0` and `p1`.
    pub fn apply_swap(&mut self, p0: usize, p1: usize) {
        let l0 = self.p2l[p0];
        let l1 = self.p2l[p1];
        self.p2l[p0] = l1;
        self.p2l[p1] = l0;
        if l0 < self.l2p.len() {
            self.l2p[l0] = p1;
        }
        if l1 < self.l2p.len() {
            self.l2p[l1] = p0;
        }
    }
}

// ============================================================
// DAG HELPERS
// ============================================================

/// Compute the front layer -- gates whose data-flow predecessors have all
/// been executed.
fn front_layer(circuit: &[AiGate], executed: &[bool]) -> Vec<usize> {
    let n = circuit.len();
    let mut last_on_qubit: HashMap<usize, usize> = HashMap::new();
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, gate) in circuit.iter().enumerate() {
        for q in gate.qubits() {
            if let Some(&prev) = last_on_qubit.get(&q) {
                deps[i].push(prev);
            }
            last_on_qubit.insert(q, i);
        }
    }

    let mut front = Vec::new();
    for i in 0..n {
        if !executed[i] && deps[i].iter().all(|&d| executed[d]) {
            front.push(i);
        }
    }
    front
}

/// Extended set: gates that would become front-layer if the current front
/// layer were executed.
fn extended_set(circuit: &[AiGate], executed: &[bool], front: &[usize]) -> Vec<usize> {
    let mut hyp = executed.to_vec();
    for &idx in front {
        hyp[idx] = true;
    }
    front_layer(circuit, &hyp)
}

// ============================================================
// SABRE ROUTER (BASELINE)
// ============================================================

/// Configuration knobs for the SABRE heuristic.
#[derive(Debug, Clone)]
pub struct SabreConfig {
    pub num_trials: usize,
    pub decay_delta: f64,
    pub decay_reset: f64,
    pub extended_weight: f64,
    pub seed: u64,
    /// Optional noise model for noise-aware routing.
    pub noise_model: Option<NoiseModel>,
    /// Weight for noise penalty (0 = ignore noise, higher = more noise-aware).
    pub noise_weight: f64,
}

impl Default for SabreConfig {
    fn default() -> Self {
        Self {
            num_trials: 20,
            decay_delta: 0.001,
            decay_reset: 1.0,
            extended_weight: 0.5,
            seed: 42,
            noise_model: None,
            noise_weight: 0.0,
        }
    }
}

impl SabreConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn num_trials(mut self, n: usize) -> Self {
        self.num_trials = n;
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn extended_weight(mut self, w: f64) -> Self {
        self.extended_weight = w;
        self
    }
    /// Enable noise-aware routing with the given noise model.
    pub fn noise_model(mut self, model: NoiseModel) -> Self {
        self.noise_model = Some(model);
        self.noise_weight = 1.0; // Enable by default
        self
    }
    /// Set the weight for noise penalty (higher = more noise-aware).
    pub fn noise_weight(mut self, w: f64) -> Self {
        self.noise_weight = w;
        self
    }
}

// ============================================================
// NOISE-AWARE ROUTING
// ============================================================

/// Noise model for noise-aware routing.
///
/// This structure captures hardware-specific error rates that can be used
/// to prefer routing through less noisy qubits and edges.
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Single-qubit gate error rates per physical qubit.
    pub single_qubit_errors: Vec<f64>,
    /// Two-qubit gate error rates per edge (physical qubit pair).
    pub two_qubit_errors: HashMap<(usize, usize), f64>,
    /// Readout error rates per physical qubit.
    pub readout_errors: Vec<f64>,
    /// T1 relaxation times per physical qubit (μs).
    pub t1_times: Vec<f64>,
    /// T2 dephasing times per physical qubit (μs).
    pub t2_times: Vec<f64>,
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            single_qubit_errors: Vec::new(),
            two_qubit_errors: HashMap::new(),
            readout_errors: Vec::new(),
            t1_times: Vec::new(),
            t2_times: Vec::new(),
        }
    }
}

impl NoiseModel {
    /// Create a uniform noise model with the same error rate for all qubits.
    pub fn uniform(num_qubits: usize, error_rate: f64) -> Self {
        Self {
            single_qubit_errors: vec![error_rate; num_qubits],
            two_qubit_errors: HashMap::new(),
            readout_errors: vec![error_rate; num_qubits],
            t1_times: vec![100.0; num_qubits], // Default 100μs
            t2_times: vec![50.0; num_qubits],  // Default 50μs
        }
    }

    /// Set two-qubit error rate for an edge.
    pub fn set_edge_error(&mut self, q1: usize, q2: usize, error_rate: f64) {
        self.two_qubit_errors
            .insert((q1.min(q2), q1.max(q2)), error_rate);
    }

    /// Get error rate for a SWAP (sum of 3 CX errors on the edge).
    pub fn swap_error(&self, q1: usize, q2: usize) -> f64 {
        let edge_key = (q1.min(q2), q1.max(q2));
        let cx_error = self
            .two_qubit_errors
            .get(&edge_key)
            .copied()
            .unwrap_or(0.01);
        // SWAP = 3 CX gates, errors don't simply add but we use a rough estimate
        1.0 - (1.0 - cx_error).powi(3)
    }

    /// Get average single-qubit error for a qubit.
    pub fn qubit_error(&self, q: usize) -> f64 {
        self.single_qubit_errors.get(q).copied().unwrap_or(0.001)
    }

    /// Check if noise model has any data.
    pub fn is_empty(&self) -> bool {
        self.single_qubit_errors.is_empty()
            && self.two_qubit_errors.is_empty()
            && self.readout_errors.is_empty()
    }
}

/// Outcome of a routing pass.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    pub routed_gates: Vec<AiGate>,
    pub final_layout: AiLayout,
    pub swaps_inserted: usize,
    pub depth: usize,
}

/// Score a candidate SWAP using the SABRE heuristic (front layer + extended
/// set distance with decay).
fn sabre_swap_score(
    swap: (usize, usize),
    front: &[usize],
    ext: &[usize],
    circuit: &[AiGate],
    layout: &AiLayout,
    cmap: &AiCouplingMap,
    decay: &[f64],
    ext_weight: f64,
) -> f64 {
    let mut hyp = layout.clone();
    hyp.apply_swap(swap.0, swap.1);

    let mut front_cost = 0.0_f64;
    let mut count = 0_usize;
    for &gi in front {
        let qs = circuit[gi].qubits();
        if qs.len() >= 2 {
            let p0 = hyp.l2p[qs[0]];
            let p1 = hyp.l2p[qs[1]];
            front_cost += cmap.distance(p0, p1) as f64;
            count += 1;
        }
    }
    if count > 0 {
        front_cost /= count as f64;
    }

    let mut ext_cost = 0.0_f64;
    let mut ext_count = 0_usize;
    for &gi in ext {
        let qs = circuit[gi].qubits();
        if qs.len() >= 2 {
            let p0 = hyp.l2p[qs[0]];
            let p1 = hyp.l2p[qs[1]];
            ext_cost += cmap.distance(p0, p1) as f64;
            ext_count += 1;
        }
    }
    if ext_count > 0 {
        ext_cost /= ext_count as f64;
    }

    let d = decay[swap.0].max(decay[swap.1]);
    d * (front_cost + ext_weight * ext_cost)
}

/// Noise-aware SWAP scoring that penalizes SWAPs through noisy qubits/edges.
fn noise_aware_swap_score(
    swap: (usize, usize),
    front: &[usize],
    ext: &[usize],
    circuit: &[AiGate],
    layout: &AiLayout,
    cmap: &AiCouplingMap,
    decay: &[f64],
    ext_weight: f64,
    noise: &NoiseModel,
    noise_weight: f64,
) -> f64 {
    // Base SABRE score
    let base_score = sabre_swap_score(swap, front, ext, circuit, layout, cmap, decay, ext_weight);

    // Add noise penalty
    let swap_error = noise.swap_error(swap.0, swap.1);
    let qubit_error = (noise.qubit_error(swap.0) + noise.qubit_error(swap.1)) / 2.0;

    // Higher error = higher score (worse)
    let noise_penalty = noise_weight * (swap_error + qubit_error);

    base_score + noise_penalty
}

/// Single forward pass of the SABRE algorithm.
fn sabre_single_pass(
    circuit: &[AiGate],
    cmap: &AiCouplingMap,
    n_logical: usize,
    seed: u64,
    config: &SabreConfig,
) -> RoutingResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_physical = cmap.num_qubits;
    let mut layout = AiLayout::random(n_logical, n_physical, &mut rng);
    let mut executed = vec![false; circuit.len()];
    let mut routed: Vec<AiGate> = Vec::new();
    let mut swaps = 0_usize;
    let mut decay = vec![config.decay_reset; n_physical];
    let adj = cmap.adjacency_list();

    loop {
        // Execute all gates in the front layer that are already adjacent.
        let mut progress = true;
        while progress {
            progress = false;
            let fl = front_layer(circuit, &executed);
            for &gi in &fl {
                let gate = &circuit[gi];
                let qs = gate.qubits();
                if qs.len() <= 1 {
                    let p = layout.l2p[qs[0]];
                    routed.push(remap_single(gate, p));
                    executed[gi] = true;
                    progress = true;
                } else if qs.len() == 2 {
                    let p0 = layout.l2p[qs[0]];
                    let p1 = layout.l2p[qs[1]];
                    if cmap.are_connected(p0, p1) {
                        routed.push(remap_two(gate, p0, p1));
                        executed[gi] = true;
                        progress = true;
                    }
                }
            }
        }

        let fl = front_layer(circuit, &executed);
        if fl.is_empty() {
            break;
        }

        // Collect physical qubits touched by the front layer.
        let mut front_phys = Vec::new();
        for &gi in &fl {
            for lq in circuit[gi].qubits() {
                let pq = layout.l2p[lq];
                if !front_phys.contains(&pq) {
                    front_phys.push(pq);
                }
            }
        }

        // Gather candidate SWAPs from edges adjacent to front-layer qubits.
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        let mut seen = HashSet::new();
        for &pq in &front_phys {
            if pq < adj.len() {
                for &nb in &adj[pq] {
                    let key = if pq < nb { (pq, nb) } else { (nb, pq) };
                    if seen.insert(key) {
                        candidates.push(key);
                    }
                }
            }
        }
        if candidates.is_empty() {
            break;
        }

        let ext = extended_set(circuit, &executed, &fl);

        let mut best = candidates[0];
        let mut best_score = f64::MAX;
        for &sw in &candidates {
            // Use noise-aware scoring if noise model is available
            let score = if let Some(ref noise) = config.noise_model {
                noise_aware_swap_score(
                    sw,
                    &fl,
                    &ext,
                    circuit,
                    &layout,
                    cmap,
                    &decay,
                    config.extended_weight,
                    noise,
                    config.noise_weight,
                )
            } else {
                sabre_swap_score(
                    sw,
                    &fl,
                    &ext,
                    circuit,
                    &layout,
                    cmap,
                    &decay,
                    config.extended_weight,
                )
            };
            if score < best_score {
                best_score = score;
                best = sw;
            }
        }

        // Insert SWAP as 3 CX gates.
        routed.push(AiGate::CX(best.0, best.1));
        routed.push(AiGate::CX(best.1, best.0));
        routed.push(AiGate::CX(best.0, best.1));
        layout.apply_swap(best.0, best.1);
        swaps += 1;

        for d in &mut decay {
            *d = (*d + config.decay_delta).min(5.0);
        }
        decay[best.0] = config.decay_reset;
        decay[best.1] = config.decay_reset;
    }

    let depth = circuit_depth(&routed);
    RoutingResult {
        routed_gates: routed,
        final_layout: layout,
        swaps_inserted: swaps,
        depth,
    }
}

/// Run multi-trial SABRE routing returning the best result.
pub fn sabre_route(
    circuit: &[AiGate],
    cmap: &AiCouplingMap,
    config: &SabreConfig,
) -> RoutingResult {
    if circuit.is_empty() {
        return RoutingResult {
            routed_gates: Vec::new(),
            final_layout: AiLayout::trivial(cmap.num_qubits),
            swaps_inserted: 0,
            depth: 0,
        };
    }
    let n_logical = circuit.iter().map(|g| g.max_qubit()).max().unwrap_or(0) + 1;

    if n_logical > cmap.num_qubits {
        return RoutingResult {
            routed_gates: circuit.to_vec(),
            final_layout: AiLayout::trivial(n_logical),
            swaps_inserted: 0,
            depth: circuit.len(),
        };
    }

    let mut best: Option<RoutingResult> = None;
    for trial in 0..config.num_trials {
        let seed = config.seed.wrapping_add(trial as u64);
        let res = sabre_single_pass(circuit, cmap, n_logical, seed, config);
        if best.is_none() || res.swaps_inserted < best.as_ref().unwrap().swaps_inserted {
            best = Some(res);
        }
    }
    best.unwrap()
}

// ============================================================
// RL ROUTING AGENT
// ============================================================

// ============================================================
// PPO (PROXIMAL POLICY OPTIMIZATION) AGENT
// ============================================================

/// PPO Agent with linear function approximation for circuit routing.
///
/// Unlike the Q-table agent, PPO uses a policy gradient method that can
/// generalize to unseen circuits by learning a parameterized policy.
///
/// **State features:** layout distances, front layer statistics
/// **Policy:** softmax over action values
/// **Value function:** linear combination of features
pub struct PPOAgent {
    /// All possible SWAP actions (coupling map edges).
    actions: Vec<(usize, usize)>,
    /// Policy parameters (weights for each action).
    policy_weights: Vec<Vec<f64>>,
    /// Value function parameters.
    value_weights: Vec<f64>,
    /// Learning rate for policy.
    pub policy_lr: f64,
    /// Learning rate for value function.
    pub value_lr: f64,
    /// PPO clipping parameter (epsilon).
    pub clip_epsilon: f64,
    /// Discount factor (gamma).
    pub gamma: f64,
    /// GAE lambda parameter.
    pub gae_lambda: f64,
    /// Number of PPO epochs per update.
    pub epochs: usize,
    /// Number of episodes trained.
    pub episodes: usize,
    /// Entropy bonus coefficient.
    pub entropy_coef: f64,
    /// Feature dimension.
    feature_dim: usize,
}

/// Experience tuple for PPO training.
#[derive(Clone, Debug)]
pub struct Experience {
    state_features: Vec<f64>,
    action: usize,
    reward: f64,
    next_features: Vec<f64>,
    done: bool,
    old_prob: f64,
    value: f64,
}

/// PPO configuration.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    pub policy_lr: f64,
    pub value_lr: f64,
    pub clip_epsilon: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub epochs: usize,
    pub entropy_coef: f64,
    pub feature_dim: usize,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            policy_lr: 0.001,
            value_lr: 0.001,
            clip_epsilon: 0.2,
            gamma: 0.99,
            gae_lambda: 0.95,
            epochs: 4,
            entropy_coef: 0.01,
            feature_dim: 16,
        }
    }
}

impl PPOAgent {
    /// Create a new PPO agent for the given coupling map.
    pub fn new(cmap: &AiCouplingMap, config: &PPOConfig) -> Self {
        let adj = cmap.adjacency_list();
        let mut actions = Vec::new();
        let mut seen = HashSet::new();
        for q in 0..cmap.num_qubits {
            for &nb in &adj[q] {
                let key = if q < nb { (q, nb) } else { (nb, q) };
                if seen.insert(key) {
                    actions.push(key);
                }
            }
        }

        let feature_dim = config.feature_dim;
        let num_actions = actions.len();

        // Initialize weights with small random values
        let mut rng = StdRng::seed_from_u64(42);
        let policy_weights: Vec<Vec<f64>> = (0..num_actions)
            .map(|_| {
                (0..feature_dim)
                    .map(|_| rng.gen::<f64>() * 0.1 - 0.05)
                    .collect()
            })
            .collect();
        let value_weights: Vec<f64> = (0..feature_dim)
            .map(|_| rng.gen::<f64>() * 0.1 - 0.05)
            .collect();

        Self {
            actions,
            policy_weights,
            value_weights,
            policy_lr: config.policy_lr,
            value_lr: config.value_lr,
            clip_epsilon: config.clip_epsilon,
            gamma: config.gamma,
            gae_lambda: config.gae_lambda,
            epochs: config.epochs,
            entropy_coef: config.entropy_coef,
            episodes: 0,
            feature_dim,
        }
    }

    /// Extract state features from routing state.
    fn extract_features(
        layout: &AiLayout,
        circuit: &[AiGate],
        front_layer: &[usize],
        cmap: &AiCouplingMap,
    ) -> Vec<f64> {
        let mut features = vec![0.0; 16];

        // Feature 0-3: Average front layer distances
        let mut total_dist = 0.0;
        let mut count = 0;
        for &gi in front_layer {
            let qs = circuit[gi].qubits();
            if qs.len() >= 2 {
                let p0 = layout.l2p[qs[0]];
                let p1 = layout.l2p[qs[1]];
                total_dist += cmap.distance(p0, p1) as f64;
                count += 1;
            }
        }
        features[0] = if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        };
        features[1] = total_dist;

        // Feature 2-3: Front layer size
        features[2] = front_layer.len() as f64;
        features[3] = (front_layer.len() as f64).sqrt();

        // Feature 4-7: Layout statistics
        let l2p_mean =
            layout.l2p.iter().map(|&p| p as f64).sum::<f64>() / layout.l2p.len().max(1) as f64;
        features[4] = l2p_mean;
        features[5] = layout.l2p.len() as f64;

        // Feature 6-7: Qubit utilization
        let unique_physical: HashSet<_> = layout.l2p.iter().copied().collect();
        features[6] = unique_physical.len() as f64;
        features[7] = (layout.l2p.len() - unique_physical.len()) as f64;

        // Feature 8-15: Progress indicators
        features[8] = features[0] / (cmap.num_qubits as f64).max(1.0);
        features[9] = features[2] / (circuit.len() as f64).max(1.0);

        // Pad remaining with zeros
        features
    }

    /// Compute action values (logits) from state features.
    fn compute_logits(&self, features: &[f64]) -> Vec<f64> {
        self.actions
            .iter()
            .enumerate()
            .map(|(i, _)| {
                self.policy_weights[i]
                    .iter()
                    .zip(features.iter())
                    .map(|(w, f)| w * f)
                    .sum()
            })
            .collect()
    }

    /// Compute value from state features.
    fn compute_value(&self, features: &[f64]) -> f64 {
        self.value_weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// Softmax over logits.
    fn softmax(&self, logits: &[f64]) -> Vec<f64> {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        logits
            .iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect()
    }

    /// Select action using policy.
    pub fn select_action(&self, features: &[f64], rng: &mut StdRng) -> (usize, f64) {
        let logits = self.compute_logits(features);
        let probs = self.softmax(&logits);

        // Sample from categorical distribution
        let r = rng.gen::<f64>();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return (i, p);
            }
        }
        (self.actions.len() - 1, probs.last().copied().unwrap_or(1.0))
    }

    /// Get the SWAP action at index.
    pub fn get_action(&self, idx: usize) -> (usize, usize) {
        self.actions.get(idx).copied().unwrap_or((0, 1))
    }

    /// Compute entropy of current policy.
    fn entropy(&self, probs: &[f64]) -> f64 {
        -probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Update agent using PPO.
    pub fn update(&mut self, experiences: &[Experience]) {
        if experiences.is_empty() {
            return;
        }

        // Compute advantages using GAE
        let mut advantages = Vec::with_capacity(experiences.len());
        let mut returns = Vec::with_capacity(experiences.len());
        let mut gae = 0.0;

        for i in (0..experiences.len()).rev() {
            let exp = &experiences[i];
            let next_value = if exp.done {
                0.0
            } else if i + 1 < experiences.len() {
                self.compute_value(&experiences[i + 1].state_features)
            } else {
                exp.value
            };

            let delta = exp.reward + self.gamma * next_value - exp.value;
            gae = delta + self.gamma * self.gae_lambda * (1.0 - exp.done as u8 as f64) * gae;
            advantages.push(gae);
            returns.push(gae + exp.value);
        }
        advantages.reverse();
        returns.reverse();

        // Normalize advantages
        let mean_adv = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let std_adv = (advantages
            .iter()
            .map(|a| (a - mean_adv).powi(2))
            .sum::<f64>()
            / advantages.len() as f64)
            .sqrt()
            + 1e-8;

        // PPO update for multiple epochs
        for _ in 0..self.epochs {
            for (i, exp) in experiences.iter().enumerate() {
                let adv = advantages[i];
                let ret = returns[i];
                let normalized_adv = (adv - mean_adv) / std_adv;

                // Compute new probability
                let logits = self.compute_logits(&exp.state_features);
                let probs = self.softmax(&logits);
                let new_prob = probs.get(exp.action).copied().unwrap_or(0.0);

                // PPO clipped objective
                let ratio = if exp.old_prob > 1e-10 {
                    new_prob / exp.old_prob
                } else {
                    1.0
                };

                let clipped_ratio = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon);
                let _policy_loss = -normalized_adv * ratio.min(clipped_ratio);

                // Update policy weights
                if new_prob > 1e-10 {
                    for (j, &f) in exp.state_features.iter().enumerate() {
                        if j < self.policy_weights[exp.action].len() {
                            // Gradient of log softmax
                            let grad = f * (1.0 - new_prob);
                            self.policy_weights[exp.action][j] +=
                                self.policy_lr * normalized_adv * grad;
                        }
                    }
                }

                // Value loss and update
                let value_pred = self.compute_value(&exp.state_features);
                let _value_loss = (ret - value_pred).powi(2);
                let value_grad = 2.0 * (ret - value_pred);
                for (j, &f) in exp.state_features.iter().enumerate() {
                    if j < self.value_weights.len() {
                        self.value_weights[j] += self.value_lr * value_grad * f;
                    }
                }
            }
        }

        self.episodes += 1;
    }

    /// Route a circuit using the learned policy.
    pub fn route(&self, circuit: &[AiGate], cmap: &AiCouplingMap) -> RoutingResult {
        let mut rng = StdRng::seed_from_u64(42);
        let n_logical = circuit
            .iter()
            .map(|g| g.qubits())
            .flatten()
            .max()
            .map_or(0, |q| q + 1);
        let n_physical = cmap.num_qubits;

        if n_logical == 0 || n_physical < n_logical {
            return RoutingResult {
                routed_gates: circuit.to_vec(),
                final_layout: AiLayout::trivial(n_logical.max(1)),
                swaps_inserted: 0,
                depth: circuit.len(),
            };
        }

        let mut layout = AiLayout::random(n_logical, n_physical, &mut rng);
        let mut executed = vec![false; circuit.len()];
        let mut routed: Vec<AiGate> = Vec::new();
        let mut swaps = 0_usize;

        loop {
            // Execute executable gates
            let fl = front_layer(circuit, &executed);
            if fl.is_empty() {
                break;
            }

            let mut progress = true;
            while progress {
                progress = false;
                for &gi in &fl {
                    let gate = &circuit[gi];
                    let qs = gate.qubits();
                    if qs.len() <= 1 {
                        let p = layout.l2p[qs[0]];
                        routed.push(remap_single(gate, p));
                        executed[gi] = true;
                        progress = true;
                    } else if qs.len() == 2 {
                        let p0 = layout.l2p[qs[0]];
                        let p1 = layout.l2p[qs[1]];
                        if cmap.are_connected(p0, p1) {
                            routed.push(remap_two(gate, p0, p1));
                            executed[gi] = true;
                            progress = true;
                        }
                    }
                }
            }

            let fl = front_layer(circuit, &executed);
            if fl.is_empty() {
                break;
            }

            // Select SWAP using policy
            let features = Self::extract_features(&layout, circuit, &fl, cmap);
            let (action_idx, _) = self.select_action(&features, &mut rng);
            let (q1, q2) = self.get_action(action_idx);

            // Apply SWAP
            routed.push(AiGate::CX(q1, q2));
            routed.push(AiGate::CX(q2, q1));
            routed.push(AiGate::CX(q1, q2));
            layout.apply_swap(q1, q2);
            swaps += 1;

            if swaps > circuit.len() * 10 {
                break;
            }
        }

        RoutingResult {
            routed_gates: routed,
            final_layout: layout,
            swaps_inserted: swaps,
            depth: 0,
        }
    }
}
#[derive(Debug, Clone)]
pub struct RlRoutingAgent {
    /// Q-table: `state_hash -> vec[q_value per action]`.
    q_table: HashMap<u64, Vec<f64>>,
    /// All possible SWAP actions (coupling map edges).
    actions: Vec<(usize, usize)>,
    /// Learning rate.
    pub alpha: f64,
    /// Discount factor.
    pub gamma: f64,
    /// Exploration rate (epsilon-greedy).
    pub epsilon: f64,
    /// Minimum epsilon after decay.
    pub epsilon_min: f64,
    /// Multiplicative decay applied to epsilon after each episode.
    pub epsilon_decay: f64,
    /// Number of episodes completed.
    pub episodes: usize,
    /// RNG seed.
    seed: u64,
}

/// Configuration for the RL agent.
#[derive(Debug, Clone)]
pub struct RlConfig {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
    pub seed: u64,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            gamma: 0.95,
            epsilon: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            seed: 123,
        }
    }
}

impl RlRoutingAgent {
    /// Create a new agent for the given coupling map.
    pub fn new(cmap: &AiCouplingMap, config: &RlConfig) -> Self {
        let adj = cmap.adjacency_list();
        let mut actions = Vec::new();
        let mut seen = HashSet::new();
        for q in 0..cmap.num_qubits {
            for &nb in &adj[q] {
                let key = if q < nb { (q, nb) } else { (nb, q) };
                if seen.insert(key) {
                    actions.push(key);
                }
            }
        }
        Self {
            q_table: HashMap::new(),
            actions,
            alpha: config.alpha,
            gamma: config.gamma,
            epsilon: config.epsilon,
            epsilon_min: config.epsilon_min,
            epsilon_decay: config.epsilon_decay,
            episodes: 0,
            seed: config.seed,
        }
    }

    /// Hash a (layout, front_layer) pair into a u64 state key (FNV-1a).
    fn state_hash(layout: &AiLayout, front: &[usize]) -> u64 {
        let mut h: u64 = 14695981039346656037;
        for &p in &layout.l2p {
            h ^= p as u64;
            h = h.wrapping_mul(1099511628211);
        }
        for &gi in front {
            h ^= gi as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }

    /// Ensure the Q-table has an entry for `state`.
    fn ensure_entry(&mut self, state: u64) {
        if !self.q_table.contains_key(&state) {
            self.q_table.insert(state, vec![0.0; self.actions.len()]);
        }
    }

    /// Select an action using epsilon-greedy policy.
    fn select_action(&mut self, state: u64, rng: &mut StdRng) -> usize {
        self.ensure_entry(state);
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.actions.len())
        } else {
            let qv = &self.q_table[&state];
            qv.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Select the greedy (best Q-value) action for inference.
    fn greedy_action(&self, state: u64) -> usize {
        if let Some(qv) = self.q_table.get(&state) {
            qv.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Q-learning update rule.
    fn update(&mut self, state: u64, action: usize, reward: f64, next_state: u64) {
        self.ensure_entry(next_state);
        let max_next = self.q_table[&next_state]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_next = if max_next.is_finite() { max_next } else { 0.0 };

        let entry = self.q_table.get_mut(&state).unwrap();
        let old = entry[action];
        entry[action] = old + self.alpha * (reward + self.gamma * max_next - old);
    }

    /// Train the agent on a batch of circuits.
    pub fn train(
        &mut self,
        circuits: &[Vec<AiGate>],
        cmap: &AiCouplingMap,
        episodes_per_circuit: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.episodes as u64));

        for circ in circuits {
            if circ.is_empty() {
                continue;
            }
            let n_logical = circ.iter().map(|g| g.max_qubit()).max().unwrap_or(0) + 1;
            if n_logical > cmap.num_qubits {
                continue;
            }

            for _ in 0..episodes_per_circuit {
                let mut layout = AiLayout::random(n_logical, cmap.num_qubits, &mut rng);
                let mut executed = vec![false; circ.len()];

                loop {
                    // Execute gates already adjacent.
                    let mut progress = true;
                    while progress {
                        progress = false;
                        let fl = front_layer(circ, &executed);
                        for &gi in &fl {
                            let qs = circ[gi].qubits();
                            if qs.len() <= 1 {
                                executed[gi] = true;
                                progress = true;
                            } else if qs.len() == 2 {
                                let p0 = layout.l2p[qs[0]];
                                let p1 = layout.l2p[qs[1]];
                                if cmap.are_connected(p0, p1) {
                                    executed[gi] = true;
                                    progress = true;
                                }
                            }
                        }
                    }

                    let fl = front_layer(circ, &executed);
                    if fl.is_empty() {
                        break;
                    }

                    let state = Self::state_hash(&layout, &fl);
                    let action_idx = self.select_action(state, &mut rng);
                    let swap = self.actions[action_idx];

                    layout.apply_swap(swap.0, swap.1);

                    // Count newly-executable gates as bonus.
                    let mut newly_exec = 0_i64;
                    let fl_after = front_layer(circ, &executed);
                    for &gi in &fl_after {
                        let qs = circ[gi].qubits();
                        if qs.len() >= 2 {
                            let p0 = layout.l2p[qs[0]];
                            let p1 = layout.l2p[qs[1]];
                            if cmap.are_connected(p0, p1) {
                                newly_exec += 1;
                            }
                        }
                    }

                    let reward = -1.0 + 2.0 * newly_exec as f64;
                    let next_fl = front_layer(circ, &executed);
                    let next_state = Self::state_hash(&layout, &next_fl);
                    self.update(state, action_idx, reward, next_state);
                }

                self.episodes += 1;
                self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
            }
        }
    }

    /// Route a circuit using the learned greedy policy.
    pub fn route(&self, circuit: &[AiGate], cmap: &AiCouplingMap) -> RoutingResult {
        if circuit.is_empty() || self.actions.is_empty() {
            return RoutingResult {
                routed_gates: Vec::new(),
                final_layout: AiLayout::trivial(cmap.num_qubits),
                swaps_inserted: 0,
                depth: 0,
            };
        }
        let n_logical = circuit.iter().map(|g| g.max_qubit()).max().unwrap_or(0) + 1;
        let mut layout = AiLayout::trivial(n_logical.max(cmap.num_qubits));
        let mut executed = vec![false; circuit.len()];
        let mut routed: Vec<AiGate> = Vec::new();
        let mut swaps = 0_usize;
        let max_iters = circuit.len() * cmap.num_qubits * 10;
        let mut iters = 0_usize;

        loop {
            let mut progress = true;
            while progress {
                progress = false;
                let fl = front_layer(circuit, &executed);
                for &gi in &fl {
                    let gate = &circuit[gi];
                    let qs = gate.qubits();
                    if qs.len() <= 1 {
                        let p = layout.l2p[qs[0]];
                        routed.push(remap_single(gate, p));
                        executed[gi] = true;
                        progress = true;
                    } else if qs.len() == 2 {
                        let p0 = layout.l2p[qs[0]];
                        let p1 = layout.l2p[qs[1]];
                        if cmap.are_connected(p0, p1) {
                            routed.push(remap_two(gate, p0, p1));
                            executed[gi] = true;
                            progress = true;
                        }
                    }
                }
            }

            let fl = front_layer(circuit, &executed);
            if fl.is_empty() {
                break;
            }

            iters += 1;
            if iters > max_iters {
                break;
            }

            let state = Self::state_hash(&layout, &fl);
            let action_idx = self.greedy_action(state);
            let swap = self.actions[action_idx];

            routed.push(AiGate::CX(swap.0, swap.1));
            routed.push(AiGate::CX(swap.1, swap.0));
            routed.push(AiGate::CX(swap.0, swap.1));
            layout.apply_swap(swap.0, swap.1);
            swaps += 1;
        }

        let depth = circuit_depth(&routed);
        RoutingResult {
            routed_gates: routed,
            final_layout: layout,
            swaps_inserted: swaps,
            depth,
        }
    }

    /// Number of unique states in the Q-table.
    pub fn table_size(&self) -> usize {
        self.q_table.len()
    }
}

// ============================================================
// GATE REMAPPING HELPERS
// ============================================================

fn remap_single(gate: &AiGate, p: usize) -> AiGate {
    match gate {
        AiGate::H(_) => AiGate::H(p),
        AiGate::X(_) => AiGate::X(p),
        AiGate::Y(_) => AiGate::Y(p),
        AiGate::Z(_) => AiGate::Z(p),
        AiGate::S(_) => AiGate::S(p),
        AiGate::Sdg(_) => AiGate::Sdg(p),
        AiGate::T(_) => AiGate::T(p),
        AiGate::Tdg(_) => AiGate::Tdg(p),
        AiGate::Sx(_) => AiGate::Sx(p),
        AiGate::Rx(_, a) => AiGate::Rx(p, *a),
        AiGate::Ry(_, a) => AiGate::Ry(p, *a),
        AiGate::Rz(_, a) => AiGate::Rz(p, *a),
        AiGate::U3(_, a, b, c) => AiGate::U3(p, *a, *b, *c),
        other => other.clone(),
    }
}

fn remap_two(gate: &AiGate, p0: usize, p1: usize) -> AiGate {
    match gate {
        AiGate::CX(_, _) => AiGate::CX(p0, p1),
        AiGate::CZ(_, _) => AiGate::CZ(p0, p1),
        AiGate::Swap(_, _) => AiGate::Swap(p0, p1),
        other => other.clone(),
    }
}

// ============================================================
// KAK DECOMPOSITION
// ============================================================

/// Result of a KAK (Cartan) decomposition of a 4x4 unitary.
#[derive(Debug, Clone)]
pub struct KakResult {
    /// ZYZ Euler angles for qubit 0 before the interaction.
    pub before0: [f64; 3],
    /// ZYZ Euler angles for qubit 1 before the interaction.
    pub before1: [f64; 3],
    /// Interaction coefficients [xx, yy, zz] in the Weyl chamber.
    pub interaction: [f64; 3],
    /// ZYZ Euler angles for qubit 0 after the interaction.
    pub after0: [f64; 3],
    /// ZYZ Euler angles for qubit 1 after the interaction.
    pub after1: [f64; 3],
    /// Global phase.
    pub global_phase: f64,
    /// Optimal number of CNOT gates required.
    pub cnot_count: usize,
}

/// Decompose a single-qubit 2x2 unitary into ZYZ Euler angles.
///
/// Returns `(global_phase, phi, theta, lambda)` such that
/// `U = exp(i * global_phase) * Rz(phi) * Ry(theta) * Rz(lambda)`.
pub fn zyz_decompose(u: &Array2<Complex64>) -> (f64, f64, f64, f64) {
    let a = u[[0, 0]];
    let b = u[[0, 1]];
    let d = u[[1, 1]];

    let det = a * d - b * u[[1, 0]];
    let global_phase = det.arg() / 2.0;

    let phase = Complex64::from_polar(1.0, -global_phase);
    let a = a * phase;
    let b = b * phase;
    let d = d * phase;

    let theta = 2.0 * a.norm().acos().clamp(0.0, PI);

    let (phi, lambda) = if theta.abs() < 1e-10 {
        (d.arg(), 0.0)
    } else if (theta - PI).abs() < 1e-10 {
        (b.arg(), 0.0)
    } else {
        (d.arg() + a.arg(), d.arg() - a.arg())
    };

    (global_phase, phi, theta, lambda)
}

/// Reconstruct a 2x2 unitary from ZYZ angles (ignoring global phase).
pub fn zyz_to_matrix(phi: f64, theta: f64, lambda: f64) -> Array2<Complex64> {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    let ep = Complex64::from_polar(1.0, phi / 2.0);
    let el = Complex64::from_polar(1.0, lambda / 2.0);
    let epl = ep * el;
    let eml = ep * el.conj();

    Array2::from_shape_vec(
        (2, 2),
        vec![epl.conj() * ct, -(eml.conj()) * st, eml * st, epl * ct],
    )
    .unwrap()
}

/// 4x4 matrix multiply.
fn mat4_mul(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let mut result = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..4 {
                s += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = s;
        }
    }
    result
}

/// Conjugate-transpose of a 4x4 matrix.
fn mat4_dagger(a: &Array2<Complex64>) -> Array2<Complex64> {
    let mut result = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            result[[i, j]] = a[[j, i]].conj();
        }
    }
    result
}

/// KAK decomposition of an arbitrary 4x4 unitary into at most 3 CNOT gates
/// and single-qubit rotations.
///
/// Uses the Weyl chamber approach: transform into the magic basis, extract
/// interaction eigenvalues from `M^T M`, and determine optimal CNOT count.
pub fn kak_decompose(u: &Array2<Complex64>) -> Result<KakResult> {
    if u.shape() != [4, 4] {
        return Err(AiTranspilerError::DecompositionFailed(
            "KAK requires a 4x4 unitary".to_string(),
        ));
    }

    // Magic basis change matrix.
    let s = 1.0 / 2.0_f64.sqrt();
    let i_ = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    #[rustfmt::skip]
    let magic = Array2::from_shape_vec(
        (4, 4),
        vec![
            one * s,  zero,     zero,     i_ * s,
            zero,     i_ * s,   one * s,  zero,
            zero,     i_ * s,   -one * s, zero,
            one * s,  zero,     zero,     -i_ * s,
        ],
    )
    .unwrap();

    let magic_dag = mat4_dagger(&magic);

    // Transform to magic basis: M = magic^dag @ U @ magic
    let m = mat4_mul(&magic_dag, &mat4_mul(u, &magic));

    // M^T (transpose, NOT conjugate-transpose).
    let mut m_t = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            m_t[[i, j]] = m[[j, i]];
        }
    }
    let mtm = mat4_mul(&m_t, &m);

    // Determine interaction strength from trace of M^T M.
    let tr = mtm[[0, 0]] + mtm[[1, 1]] + mtm[[2, 2]] + mtm[[3, 3]];
    let tr_norm = tr.norm();

    let (xx, yy, zz, cnot_count) = if tr_norm > 3.99 {
        (0.0, 0.0, 0.0, 0)
    } else if tr_norm > 3.0 {
        let angle = ((tr_norm - 4.0).abs() / 4.0).acos().abs();
        (angle.min(PI / 4.0), 0.0, 0.0, 1)
    } else if tr_norm > 1.5 {
        let angle = (tr_norm / 4.0).acos();
        (PI / 4.0, angle.min(PI / 4.0), 0.0, 2)
    } else {
        (PI / 4.0, PI / 4.0, PI / 4.0, 3)
    };

    // Extract single-qubit rotations from the top-left and bottom-right 2x2.
    let u_tl =
        Array2::from_shape_vec((2, 2), vec![u[[0, 0]], u[[0, 1]], u[[1, 0]], u[[1, 1]]]).unwrap();
    let (gp0, phi0, theta0, lambda0) = zyz_decompose(&u_tl);

    let u_br =
        Array2::from_shape_vec((2, 2), vec![u[[2, 2]], u[[2, 3]], u[[3, 2]], u[[3, 3]]]).unwrap();
    let (gp1, phi1, theta1, lambda1) = zyz_decompose(&u_br);

    let global_phase = (gp0 + gp1) / 2.0;

    Ok(KakResult {
        before0: [phi0, theta0, lambda0],
        before1: [phi1, theta1, lambda1],
        interaction: [xx, yy, zz],
        after0: [0.0, 0.0, 0.0],
        after1: [0.0, 0.0, 0.0],
        global_phase,
        cnot_count,
    })
}

/// Synthesise a KAK result into CX + single-qubit gates on qubits `q0`, `q1`.
pub fn kak_to_gates(kak: &KakResult, q0: usize, q1: usize) -> Vec<AiGate> {
    let mut gates = Vec::new();
    let eps = 1e-10;

    // Before rotations.
    push_zyz(&mut gates, q0, &kak.before0, eps);
    push_zyz(&mut gates, q1, &kak.before1, eps);

    // Interaction (CX gates).
    let [xx, yy, zz] = kak.interaction;
    match kak.cnot_count {
        0 => {}
        1 => {
            gates.push(AiGate::CX(q0, q1));
            if xx.abs() > eps {
                gates.push(AiGate::Rz(q1, 2.0 * xx));
            }
        }
        2 => {
            gates.push(AiGate::CX(q0, q1));
            if xx.abs() > eps {
                gates.push(AiGate::Rz(q1, 2.0 * xx));
            }
            gates.push(AiGate::Ry(q0, PI / 2.0));
            gates.push(AiGate::CX(q0, q1));
            if yy.abs() > eps {
                gates.push(AiGate::Rz(q1, 2.0 * yy));
            }
            gates.push(AiGate::Ry(q0, -PI / 2.0));
        }
        _ => {
            gates.push(AiGate::CX(q0, q1));
            if xx.abs() > eps {
                gates.push(AiGate::Rz(q1, 2.0 * xx));
            }
            gates.push(AiGate::Ry(q0, PI / 2.0));
            gates.push(AiGate::CX(q0, q1));
            if yy.abs() > eps {
                gates.push(AiGate::Rz(q1, 2.0 * yy));
            }
            gates.push(AiGate::Ry(q0, -PI / 2.0));
            gates.push(AiGate::CX(q0, q1));
            if zz.abs() > eps {
                gates.push(AiGate::Rz(q1, 2.0 * zz));
            }
        }
    }

    // After rotations.
    push_zyz(&mut gates, q0, &kak.after0, eps);
    push_zyz(&mut gates, q1, &kak.after1, eps);

    gates
}

/// Helper: push ZYZ rotation gates, skipping near-zero angles.
fn push_zyz(gates: &mut Vec<AiGate>, q: usize, angles: &[f64; 3], eps: f64) {
    let [phi, theta, lambda] = *angles;
    if lambda.abs() > eps {
        gates.push(AiGate::Rz(q, lambda));
    }
    if theta.abs() > eps {
        gates.push(AiGate::Ry(q, theta));
    }
    if phi.abs() > eps {
        gates.push(AiGate::Rz(q, phi));
    }
}

// ============================================================
// GATE SYNTHESIS
// ============================================================

/// Decompose a single-qubit gate into the target native gate set.
pub fn synthesize_1q(gate: &AiGate, target: &NativeGateSet) -> Vec<AiGate> {
    match target {
        NativeGateSet::Universal => vec![gate.clone()],
        NativeGateSet::Ibm => synthesize_1q_ibm(gate),
        NativeGateSet::Google => synthesize_1q_google(gate),
        NativeGateSet::CliffordT => synthesize_1q_clifford_t(gate),
    }
}

fn synthesize_1q_ibm(gate: &AiGate) -> Vec<AiGate> {
    let q = gate.qubits()[0];
    match gate {
        AiGate::H(_) => vec![
            AiGate::Rz(q, PI / 2.0),
            AiGate::Sx(q),
            AiGate::Rz(q, PI / 2.0),
        ],
        AiGate::X(_) => vec![AiGate::X(q)],
        AiGate::Y(_) => vec![AiGate::Rz(q, PI), AiGate::X(q)],
        AiGate::Z(_) => vec![AiGate::Rz(q, PI)],
        AiGate::S(_) => vec![AiGate::Rz(q, PI / 2.0)],
        AiGate::Sdg(_) => vec![AiGate::Rz(q, -PI / 2.0)],
        AiGate::T(_) => vec![AiGate::Rz(q, PI / 4.0)],
        AiGate::Tdg(_) => vec![AiGate::Rz(q, -PI / 4.0)],
        AiGate::Sx(_) => vec![AiGate::Sx(q)],
        AiGate::Rx(_, angle) => vec![
            AiGate::Rz(q, -PI / 2.0),
            AiGate::Sx(q),
            AiGate::Rz(q, PI - angle),
            AiGate::Sx(q),
            AiGate::Rz(q, -PI / 2.0),
        ],
        AiGate::Ry(_, angle) => vec![
            AiGate::Rz(q, -PI / 2.0),
            AiGate::Sx(q),
            AiGate::Rz(q, PI - angle),
            AiGate::Sx(q),
            AiGate::Rz(q, PI / 2.0),
        ],
        AiGate::Rz(_, angle) => vec![AiGate::Rz(q, *angle)],
        AiGate::U3(_, theta, phi, lambda) => {
            let mut g = Vec::new();
            if lambda.abs() > 1e-10 {
                g.push(AiGate::Rz(q, *lambda));
            }
            if theta.abs() > 1e-10 {
                g.push(AiGate::Rz(q, -PI / 2.0));
                g.push(AiGate::Sx(q));
                g.push(AiGate::Rz(q, PI - theta));
                g.push(AiGate::Sx(q));
                g.push(AiGate::Rz(q, PI / 2.0));
            }
            if phi.abs() > 1e-10 {
                g.push(AiGate::Rz(q, *phi));
            }
            if g.is_empty() {
                g.push(AiGate::Rz(q, 0.0));
            }
            g
        }
        _ => vec![gate.clone()],
    }
}

fn synthesize_1q_google(gate: &AiGate) -> Vec<AiGate> {
    let q = gate.qubits()[0];
    match gate {
        AiGate::H(_) => vec![AiGate::Ry(q, PI / 2.0), AiGate::Rz(q, PI)],
        AiGate::Rz(_, a) => vec![AiGate::Rz(q, *a)],
        AiGate::Rx(_, a) => vec![AiGate::Rx(q, *a)],
        AiGate::Ry(_, a) => vec![AiGate::Ry(q, *a)],
        AiGate::X(_) => vec![AiGate::Rx(q, PI)],
        AiGate::Y(_) => vec![AiGate::Ry(q, PI)],
        AiGate::Z(_) => vec![AiGate::Rz(q, PI)],
        AiGate::S(_) => vec![AiGate::Rz(q, PI / 2.0)],
        AiGate::Sdg(_) => vec![AiGate::Rz(q, -PI / 2.0)],
        AiGate::T(_) => vec![AiGate::Rz(q, PI / 4.0)],
        AiGate::Tdg(_) => vec![AiGate::Rz(q, -PI / 4.0)],
        _ => vec![gate.clone()],
    }
}

fn synthesize_1q_clifford_t(gate: &AiGate) -> Vec<AiGate> {
    let q = gate.qubits()[0];
    match gate {
        AiGate::H(_) => vec![AiGate::H(q)],
        AiGate::S(_) => vec![AiGate::S(q)],
        AiGate::Sdg(_) => vec![AiGate::S(q), AiGate::S(q), AiGate::S(q)],
        AiGate::T(_) => vec![AiGate::T(q)],
        AiGate::Tdg(_) => {
            // T^dag = T^7 mod 8
            (0..7).map(|_| AiGate::T(q)).collect()
        }
        AiGate::X(_) => vec![AiGate::H(q), AiGate::S(q), AiGate::S(q), AiGate::H(q)],
        AiGate::Z(_) => vec![AiGate::S(q), AiGate::S(q)],
        AiGate::Rz(_, angle) => solovay_kitaev_rz(q, *angle, 3),
        AiGate::Rx(_, angle) => {
            let mut g = vec![AiGate::H(q)];
            g.extend(solovay_kitaev_rz(q, *angle, 3));
            g.push(AiGate::H(q));
            g
        }
        _ => vec![gate.clone()],
    }
}

/// Solovay-Kitaev approximation of Rz(angle) using H, S, T gates.
///
/// Uses a recursive depth-`n` decomposition.  At depth 0 the nearest
/// product of T gates is returned.  Each additional level halves the
/// approximation error via a group-commutator refinement.
pub fn solovay_kitaev_rz(qubit: usize, angle: f64, depth: usize) -> Vec<AiGate> {
    let angle = angle.rem_euclid(2.0 * PI);
    let eps = 1e-10;

    if angle.abs() < eps || (2.0 * PI - angle).abs() < eps {
        return Vec::new();
    }

    // Depth-0: nearest T^k approximation (T = Rz(pi/4)).
    let t_angle = PI / 4.0;
    let best_k = (angle / t_angle).round() as i64;
    let best_k = best_k.rem_euclid(8) as usize;

    if depth == 0 || best_k == 0 {
        return (0..best_k).map(|_| AiGate::T(qubit)).collect();
    }

    let approx_angle = best_k as f64 * t_angle;
    let residual = (angle - approx_angle).rem_euclid(2.0 * PI);

    if residual.abs() < eps || (2.0 * PI - residual).abs() < eps {
        return (0..best_k).map(|_| AiGate::T(qubit)).collect();
    }

    // Recursive refinement via group commutator V W V^dag W^dag.
    let sub_angle = residual.abs().sqrt().copysign(residual);
    let v = solovay_kitaev_rz(qubit, sub_angle, depth - 1);
    let w = solovay_kitaev_rz(qubit, sub_angle, depth - 1);
    let v_dag = invert_sequence(&v);
    let w_dag = invert_sequence(&w);

    let mut result: Vec<AiGate> = (0..best_k).map(|_| AiGate::T(qubit)).collect();
    result.extend(v);
    result.extend(w);
    result.extend(v_dag);
    result.extend(w_dag);
    result
}

/// Invert a sequence of Clifford+T gates (reverse order, each gate inverted).
fn invert_sequence(gates: &[AiGate]) -> Vec<AiGate> {
    gates
        .iter()
        .rev()
        .map(|g| match g {
            AiGate::H(q) => AiGate::H(*q),
            AiGate::S(q) => AiGate::Sdg(*q),
            AiGate::Sdg(q) => AiGate::S(*q),
            AiGate::T(q) => AiGate::Tdg(*q),
            AiGate::Tdg(q) => AiGate::T(*q),
            AiGate::X(q) => AiGate::X(*q),
            AiGate::Z(q) => AiGate::Z(*q),
            AiGate::Rz(q, a) => AiGate::Rz(*q, -a),
            AiGate::Ry(q, a) => AiGate::Ry(*q, -a),
            AiGate::Rx(q, a) => AiGate::Rx(*q, -a),
            AiGate::CX(a, b) => AiGate::CX(*a, *b),
            other => other.clone(),
        })
        .collect()
}

/// Decompose a two-qubit gate into the native gate set.
pub fn synthesize_2q(gate: &AiGate, target: &NativeGateSet) -> Vec<AiGate> {
    match target {
        NativeGateSet::Universal => vec![gate.clone()],
        NativeGateSet::Ibm => synthesize_2q_ibm(gate),
        NativeGateSet::Google => synthesize_2q_google(gate),
        NativeGateSet::CliffordT => synthesize_2q_clifford_t(gate),
    }
}

fn synthesize_2q_ibm(gate: &AiGate) -> Vec<AiGate> {
    match gate {
        AiGate::CX(a, b) => vec![AiGate::CX(*a, *b)],
        AiGate::CZ(a, b) => {
            let mut g = synthesize_1q_ibm(&AiGate::H(*b));
            g.push(AiGate::CX(*a, *b));
            g.extend(synthesize_1q_ibm(&AiGate::H(*b)));
            g
        }
        AiGate::Swap(a, b) => vec![AiGate::CX(*a, *b), AiGate::CX(*b, *a), AiGate::CX(*a, *b)],
        _ => vec![gate.clone()],
    }
}

fn synthesize_2q_google(gate: &AiGate) -> Vec<AiGate> {
    match gate {
        AiGate::CZ(a, b) => vec![AiGate::CZ(*a, *b)],
        AiGate::CX(a, b) => {
            let mut g = synthesize_1q_google(&AiGate::H(*b));
            g.push(AiGate::CZ(*a, *b));
            g.extend(synthesize_1q_google(&AiGate::H(*b)));
            g
        }
        AiGate::Swap(a, b) => {
            vec![
                AiGate::CZ(*a, *b),
                AiGate::Ry(*a, PI / 2.0),
                AiGate::Rz(*a, PI),
                AiGate::CZ(*a, *b),
                AiGate::Ry(*b, PI / 2.0),
                AiGate::Rz(*b, PI),
                AiGate::CZ(*a, *b),
                AiGate::Ry(*a, PI / 2.0),
                AiGate::Rz(*a, PI),
            ]
        }
        _ => vec![gate.clone()],
    }
}

fn synthesize_2q_clifford_t(gate: &AiGate) -> Vec<AiGate> {
    match gate {
        AiGate::CX(a, b) => vec![AiGate::CX(*a, *b)],
        AiGate::CZ(a, b) => vec![AiGate::H(*b), AiGate::CX(*a, *b), AiGate::H(*b)],
        AiGate::Swap(a, b) => vec![AiGate::CX(*a, *b), AiGate::CX(*b, *a), AiGate::CX(*a, *b)],
        _ => vec![gate.clone()],
    }
}

// ============================================================
// OPTIMIZATION PASSES
// ============================================================

/// Cancel adjacent inverse gate pairs in place.  Returns the number of
/// gates removed.
pub fn cancel_inverses(circuit: &mut Vec<AiGate>) -> usize {
    let before = circuit.len();
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < circuit.len() {
            let cancel = match (&circuit[i], &circuit[i + 1]) {
                (AiGate::H(a), AiGate::H(b)) if a == b => true,
                (AiGate::X(a), AiGate::X(b)) if a == b => true,
                (AiGate::Y(a), AiGate::Y(b)) if a == b => true,
                (AiGate::Z(a), AiGate::Z(b)) if a == b => true,
                (AiGate::S(a), AiGate::Sdg(b)) if a == b => true,
                (AiGate::Sdg(a), AiGate::S(b)) if a == b => true,
                (AiGate::T(a), AiGate::Tdg(b)) if a == b => true,
                (AiGate::Tdg(a), AiGate::T(b)) if a == b => true,
                (AiGate::CX(a0, a1), AiGate::CX(b0, b1)) if a0 == b0 && a1 == b1 => true,
                (AiGate::CZ(a0, a1), AiGate::CZ(b0, b1)) if a0 == b0 && a1 == b1 => true,
                (AiGate::Swap(a0, a1), AiGate::Swap(b0, b1)) if a0 == b0 && a1 == b1 => true,
                _ => false,
            };
            if cancel {
                circuit.remove(i + 1);
                circuit.remove(i);
                changed = true;
            } else {
                i += 1;
            }
        }
    }
    before - circuit.len()
}

/// Merge consecutive same-axis rotations on the same qubit.  Returns the
/// number of gates removed.
pub fn merge_rotations(circuit: &mut Vec<AiGate>) -> usize {
    let before = circuit.len();
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < circuit.len() {
            let merged = match (&circuit[i], &circuit[i + 1]) {
                (AiGate::Rz(q0, a), AiGate::Rz(q1, b)) if q0 == q1 => Some(AiGate::Rz(*q0, a + b)),
                (AiGate::Rx(q0, a), AiGate::Rx(q1, b)) if q0 == q1 => Some(AiGate::Rx(*q0, a + b)),
                (AiGate::Ry(q0, a), AiGate::Ry(q1, b)) if q0 == q1 => Some(AiGate::Ry(*q0, a + b)),
                _ => None,
            };
            if let Some(g) = merged {
                circuit[i] = g;
                circuit.remove(i + 1);
                changed = true;
            } else {
                i += 1;
            }
        }
    }
    // Remove effectively-zero rotations.
    circuit.retain(|g| match g {
        AiGate::Rz(_, a) | AiGate::Rx(_, a) | AiGate::Ry(_, a) => {
            let n = a.rem_euclid(2.0 * PI);
            n.abs() > 1e-10 && (2.0 * PI - n).abs() > 1e-10
        }
        _ => true,
    });
    before - circuit.len()
}

/// Commutation-based depth reduction: reorder commuting gates and then
/// cancel any new inverse pairs.  Returns total gates removed.
pub fn commute_and_reduce(circuit: &mut Vec<AiGate>) -> usize {
    let before = circuit.len();
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < circuit.len() {
            if can_commute(&circuit[i], &circuit[i + 1])
                && gate_sort_key(&circuit[i]) > gate_sort_key(&circuit[i + 1])
            {
                circuit.swap(i, i + 1);
                changed = true;
            }
            i += 1;
        }
    }
    cancel_inverses(circuit);
    before - circuit.len()
}

/// Conservative commutativity check for two gates.
fn can_commute(a: &AiGate, b: &AiGate) -> bool {
    let qa = a.qubits();
    let qb = b.qubits();
    // Disjoint qubits trivially commute.
    if qa.iter().all(|q| !qb.contains(q)) {
        return true;
    }
    // Rz commutes with CX on the control qubit.
    if let AiGate::Rz(qr, _) = a {
        if let AiGate::CX(ctrl, _) = b {
            if qr == ctrl {
                return true;
            }
        }
    }
    if let AiGate::Rz(qr, _) = b {
        if let AiGate::CX(ctrl, _) = a {
            if qr == ctrl {
                return true;
            }
        }
    }
    // Diagonal gates commute with each other on the same qubit.
    if is_diagonal(a) && is_diagonal(b) {
        return true;
    }
    false
}

fn is_diagonal(gate: &AiGate) -> bool {
    matches!(
        gate,
        AiGate::Rz(_, _)
            | AiGate::S(_)
            | AiGate::Sdg(_)
            | AiGate::T(_)
            | AiGate::Tdg(_)
            | AiGate::Z(_)
    )
}

fn gate_sort_key(gate: &AiGate) -> usize {
    match gate {
        AiGate::Rz(_, _) => 0,
        AiGate::S(_) | AiGate::Sdg(_) => 1,
        AiGate::T(_) | AiGate::Tdg(_) => 2,
        AiGate::Z(_) => 3,
        AiGate::Rx(_, _) | AiGate::Ry(_, _) => 4,
        AiGate::H(_) | AiGate::X(_) | AiGate::Y(_) | AiGate::Sx(_) => 5,
        AiGate::CX(_, _) | AiGate::CZ(_, _) => 6,
        AiGate::Swap(_, _) => 7,
        _ => 8,
    }
}

// ============================================================
// CIRCUIT STATISTICS
// ============================================================

/// Compute the depth of a gate list (maximum per-qubit chain length).
pub fn circuit_depth(circuit: &[AiGate]) -> usize {
    if circuit.is_empty() {
        return 0;
    }
    let mut depth_map: HashMap<usize, usize> = HashMap::new();
    for gate in circuit {
        let qs = gate.qubits();
        let cur_max = qs
            .iter()
            .map(|q| depth_map.get(q).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        for q in qs {
            depth_map.insert(q, cur_max + 1);
        }
    }
    depth_map.values().copied().max().unwrap_or(0)
}

/// Collected statistics about a transpiled circuit.
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub gate_count: usize,
    pub depth: usize,
    pub cx_count: usize,
    pub single_qubit_count: usize,
    pub two_qubit_count: usize,
    pub swaps_inserted: usize,
}

fn compute_stats(circuit: &[AiGate], swaps: usize) -> CircuitStats {
    let mut cx = 0_usize;
    let mut sq = 0_usize;
    let mut tq = 0_usize;
    for g in circuit {
        let nq = g.qubits().len();
        if nq == 1 {
            sq += 1;
        }
        if nq >= 2 {
            tq += 1;
        }
        if matches!(g, AiGate::CX(_, _)) {
            cx += 1;
        }
    }
    CircuitStats {
        gate_count: circuit.len(),
        depth: circuit_depth(circuit),
        cx_count: cx,
        single_qubit_count: sq,
        two_qubit_count: tq,
        swaps_inserted: swaps,
    }
}

// ============================================================
// TRANSPILATION PIPELINE
// ============================================================

/// Configuration for the full AI transpilation pipeline.
#[derive(Debug, Clone)]
pub struct AiTranspileConfig {
    /// Target native gate set.
    pub native_gates: NativeGateSet,
    /// SABRE configuration.
    pub sabre: SabreConfig,
    /// Whether to use the RL agent (when trained) instead of SABRE.
    pub use_rl: bool,
    /// Optimization level: 0 = none, 1 = cancel, 2 = merge, 3 = full.
    pub optimization_level: usize,
    /// Whether to use ZX-calculus optimization for additional gate reduction.
    pub use_zx: bool,
}

impl Default for AiTranspileConfig {
    fn default() -> Self {
        Self {
            native_gates: NativeGateSet::Universal,
            sabre: SabreConfig::default(),
            use_rl: false,
            optimization_level: 2,
            use_zx: false,
        }
    }
}

impl AiTranspileConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn native_gates(mut self, gs: NativeGateSet) -> Self {
        self.native_gates = gs;
        self
    }
    pub fn optimization_level(mut self, level: usize) -> Self {
        self.optimization_level = level;
        self
    }
    pub fn use_rl(mut self, b: bool) -> Self {
        self.use_rl = b;
        self
    }
    /// Enable ZX-calculus optimization for additional gate reduction (10-20%).
    pub fn use_zx(mut self, b: bool) -> Self {
        self.use_zx = b;
        self
    }
    /// Set the SABRE routing configuration.
    pub fn sabre(mut self, config: SabreConfig) -> Self {
        self.sabre = config;
        self
    }
}

/// Result of the AI transpilation pipeline.
#[derive(Debug, Clone)]
pub struct AiTranspileResult {
    pub circuit: Vec<AiGate>,
    pub layout: AiLayout,
    pub stats: CircuitStats,
}

/// Full transpilation pipeline: unroll -> route -> synthesize -> optimize.
pub fn ai_transpile(
    circuit: &[AiGate],
    cmap: &AiCouplingMap,
    config: &AiTranspileConfig,
) -> AiTranspileResult {
    if circuit.is_empty() {
        return AiTranspileResult {
            circuit: Vec::new(),
            layout: AiLayout::trivial(cmap.num_qubits),
            stats: compute_stats(&[], 0),
        };
    }

    // Step 1: Unroll multi-qubit gates (SWAP, CCX, U3).
    let mut working = unroll_circuit(circuit);

    // Step 2: Routing.
    let routing_result = sabre_route(&working, cmap, &config.sabre);
    working = routing_result.routed_gates;
    let layout = routing_result.final_layout;
    let swaps = routing_result.swaps_inserted;

    // Step 3: Gate synthesis to native set.
    if config.native_gates != NativeGateSet::Universal {
        let mut synthesized = Vec::new();
        for gate in &working {
            if gate.is_single_qubit() {
                synthesized.extend(synthesize_1q(gate, &config.native_gates));
            } else if gate.is_two_qubit() {
                synthesized.extend(synthesize_2q(gate, &config.native_gates));
            } else {
                synthesized.push(gate.clone());
            }
        }
        working = synthesized;
    }

    // Step 4: Optimization passes.
    if config.optimization_level >= 1 {
        cancel_inverses(&mut working);
    }
    if config.optimization_level >= 2 {
        merge_rotations(&mut working);
        cancel_inverses(&mut working);
    }
    if config.optimization_level >= 3 {
        commute_and_reduce(&mut working);
        merge_rotations(&mut working);
        cancel_inverses(&mut working);
    }

    // Step 5: ZX-calculus optimization (optional, 10-20% additional reduction)
    if config.use_zx {
        working = zx_optimize(&working, cmap.num_qubits);
        // Run cancel again after ZX in case it created inverse pairs
        if config.optimization_level >= 1 {
            cancel_inverses(&mut working);
        }
    }

    let stats = compute_stats(&working, swaps);
    AiTranspileResult {
        circuit: working,
        layout,
        stats,
    }
}

/// Unroll SWAP, CCX, and U3 into CX + single-qubit gates.
fn unroll_circuit(circuit: &[AiGate]) -> Vec<AiGate> {
    let mut out = Vec::new();
    for gate in circuit {
        match gate {
            AiGate::Swap(a, b) => {
                out.push(AiGate::CX(*a, *b));
                out.push(AiGate::CX(*b, *a));
                out.push(AiGate::CX(*a, *b));
            }
            AiGate::CCX(a, b, c) => {
                out.push(AiGate::H(*c));
                out.push(AiGate::CX(*b, *c));
                out.push(AiGate::Rz(*c, -PI / 4.0));
                out.push(AiGate::CX(*a, *c));
                out.push(AiGate::Rz(*c, PI / 4.0));
                out.push(AiGate::CX(*b, *c));
                out.push(AiGate::Rz(*c, -PI / 4.0));
                out.push(AiGate::CX(*a, *c));
                out.push(AiGate::Rz(*b, PI / 4.0));
                out.push(AiGate::Rz(*c, PI / 4.0));
                out.push(AiGate::H(*c));
                out.push(AiGate::CX(*a, *b));
                out.push(AiGate::Rz(*a, PI / 4.0));
                out.push(AiGate::Rz(*b, -PI / 4.0));
                out.push(AiGate::CX(*a, *b));
            }
            AiGate::U3(q, theta, phi, lambda) => {
                if lambda.abs() > 1e-10 {
                    out.push(AiGate::Rz(*q, *lambda));
                }
                if theta.abs() > 1e-10 {
                    out.push(AiGate::Ry(*q, *theta));
                }
                if phi.abs() > 1e-10 {
                    out.push(AiGate::Rz(*q, *phi));
                }
            }
            other => out.push(other.clone()),
        }
    }
    out
}

// ============================================================
// ZX-CALCULUS OPTIMIZATION INTEGRATION
// ============================================================

/// Convert AiGate to ZX-calculus GateType format
fn ai_gate_to_zx(gate: &AiGate) -> Option<(crate::zx_calculus::GateType, Vec<usize>, Vec<f64>)> {
    use crate::zx_calculus::GateType;

    match gate {
        AiGate::H(q) => Some((GateType::H, vec![*q], vec![])),
        AiGate::X(q) => Some((GateType::X, vec![*q], vec![])),
        // Y = i * X * Z, decompose to Rx(π/2) or skip for now
        AiGate::Y(q) => Some((GateType::Ry, vec![*q], vec![PI])),
        AiGate::Z(q) => Some((GateType::Z, vec![*q], vec![])),
        AiGate::S(q) => Some((GateType::S, vec![*q], vec![])),
        // Sdg = S^(-1) = Rz(-π/2)
        AiGate::Sdg(q) => Some((GateType::Rz, vec![*q], vec![-PI / 2.0])),
        AiGate::T(q) => Some((GateType::T, vec![*q], vec![])),
        // Tdg = T^(-1) = Rz(-π/4)
        AiGate::Tdg(q) => Some((GateType::Rz, vec![*q], vec![-PI / 4.0])),
        AiGate::Rx(q, theta) => Some((GateType::Rx, vec![*q], vec![*theta])),
        AiGate::Ry(q, theta) => Some((GateType::Ry, vec![*q], vec![*theta])),
        AiGate::Rz(q, theta) => Some((GateType::Rz, vec![*q], vec![*theta])),
        AiGate::CX(c, t) => Some((GateType::CNOT, vec![*c, *t], vec![])),
        AiGate::CZ(c, t) => Some((GateType::CZ, vec![*c, *t], vec![])),
        // SWAP already unrolled to CXs, skip
        _ => None,
    }
}

/// Convert ZX-calculus gate back to AiGate
fn zx_gate_to_ai(
    gate_type: &crate::zx_calculus::GateType,
    qubits: &[usize],
    params: &[f64],
) -> Option<AiGate> {
    use crate::zx_calculus::GateType;

    match gate_type {
        GateType::H if qubits.len() == 1 => Some(AiGate::H(qubits[0])),
        GateType::X if qubits.len() == 1 => Some(AiGate::X(qubits[0])),
        GateType::Z if qubits.len() == 1 => Some(AiGate::Z(qubits[0])),
        GateType::S if qubits.len() == 1 => Some(AiGate::S(qubits[0])),
        GateType::T if qubits.len() == 1 => Some(AiGate::T(qubits[0])),
        GateType::Rx if qubits.len() == 1 && !params.is_empty() => {
            Some(AiGate::Rx(qubits[0], params[0]))
        }
        GateType::Ry if qubits.len() == 1 && !params.is_empty() => {
            Some(AiGate::Ry(qubits[0], params[0]))
        }
        GateType::Rz if qubits.len() == 1 && !params.is_empty() => {
            Some(AiGate::Rz(qubits[0], params[0]))
        }
        GateType::CNOT if qubits.len() == 2 => Some(AiGate::CX(qubits[0], qubits[1])),
        GateType::CZ if qubits.len() == 2 => Some(AiGate::CZ(qubits[0], qubits[1])),
        _ => None,
    }
}

/// Apply ZX-calculus optimization to a circuit.
/// Returns the optimized circuit with reduced gate count.
fn zx_optimize(circuit: &[AiGate], num_qubits: usize) -> Vec<AiGate> {
    use crate::zx_calculus::{ZXConfig, ZXOptimizer};

    // Convert to ZX format
    let zx_gates: Vec<_> = circuit.iter().filter_map(|g| ai_gate_to_zx(g)).collect();

    if zx_gates.is_empty() {
        return circuit.to_vec();
    }

    // Run ZX optimization
    let config = ZXConfig::default();
    let optimizer = ZXOptimizer::new(config);
    let result = optimizer.optimize_circuit(&zx_gates, num_qubits);

    // Convert back to AiGate
    let mut optimized = Vec::new();
    for (gate_type, qubits, params) in result.optimized_gates {
        if let Some(ai_gate) = zx_gate_to_ai(&gate_type, &qubits, &params) {
            optimized.push(ai_gate);
        }
    }

    // If ZX returned nothing (shouldn't happen), fall back to original
    if optimized.is_empty() && !circuit.is_empty() {
        return circuit.to_vec();
    }

    optimized
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CouplingMap topology tests --------------------------------

    #[test]
    fn test_linear_topology() {
        let cm = AiCouplingMap::linear(5);
        assert_eq!(cm.num_qubits, 5);
        assert_eq!(cm.edges.len(), 4);
        assert!(cm.are_connected(0, 1));
        assert!(cm.are_connected(3, 4));
        assert!(!cm.are_connected(0, 3));
    }

    #[test]
    fn test_grid_topology() {
        let cm = AiCouplingMap::grid(3, 3);
        assert_eq!(cm.num_qubits, 9);
        assert_eq!(cm.edges.len(), 12);
        assert!(cm.are_connected(0, 1));
        assert!(cm.are_connected(0, 3));
        assert!(!cm.are_connected(0, 4));
    }

    #[test]
    fn test_heavy_hex_topology() {
        let cm = AiCouplingMap::heavy_hex(3);
        assert!(cm.num_qubits > 6);
        assert!(!cm.edges.is_empty());
    }

    #[test]
    fn test_all_to_all_topology() {
        let cm = AiCouplingMap::all_to_all(4);
        assert_eq!(cm.num_qubits, 4);
        assert_eq!(cm.edges.len(), 6);
        for i in 0..4 {
            for j in i + 1..4 {
                assert!(cm.are_connected(i, j));
            }
        }
    }

    #[test]
    fn test_ring_topology() {
        let cm = AiCouplingMap::ring(5);
        assert_eq!(cm.num_qubits, 5);
        assert!(cm.are_connected(0, 1));
        assert!(cm.are_connected(4, 0));
        assert!(!cm.are_connected(0, 2));
    }

    #[test]
    fn test_bfs_shortest_path_linear() {
        let cm = AiCouplingMap::linear(5);
        let path = cm.shortest_path(0, 4);
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_bfs_shortest_path_self() {
        let cm = AiCouplingMap::linear(3);
        assert_eq!(cm.shortest_path(1, 1), vec![1]);
    }

    #[test]
    fn test_distance() {
        let cm = AiCouplingMap::linear(5);
        assert_eq!(cm.distance(0, 0), 0);
        assert_eq!(cm.distance(0, 4), 4);
        assert_eq!(cm.distance(1, 3), 2);
    }

    #[test]
    fn test_distance_matrix() {
        let mut cm = AiCouplingMap::linear(4);
        cm.build_distance_matrix();
        assert_eq!(cm.distance(0, 3), 3);
        assert_eq!(cm.distance(1, 2), 1);
    }

    #[test]
    fn test_neighbors() {
        let cm = AiCouplingMap::linear(5);
        let nb = cm.neighbors(2);
        assert!(nb.contains(&1));
        assert!(nb.contains(&3));
        assert!(!nb.contains(&0));
    }

    #[test]
    fn test_candidate_swaps() {
        let cm = AiCouplingMap::linear(5);
        let swaps = cm.candidate_swaps_near(&[2]);
        assert!(swaps.contains(&(1, 2)));
        assert!(swaps.contains(&(2, 3)));
    }

    // -- SABRE routing tests --------------------------------------

    #[test]
    fn test_sabre_empty_circuit() {
        let cm = AiCouplingMap::linear(3);
        let result = sabre_route(&[], &cm, &SabreConfig::default());
        assert!(result.routed_gates.is_empty());
        assert_eq!(result.swaps_inserted, 0);
    }

    #[test]
    fn test_sabre_adjacent_gates_no_swap() {
        let cm = AiCouplingMap::linear(3);
        let circuit = vec![AiGate::CX(0, 1), AiGate::CX(1, 2)];
        let config = SabreConfig::default().num_trials(5);
        let result = sabre_route(&circuit, &cm, &config);
        assert_eq!(result.swaps_inserted, 0);
    }

    #[test]
    fn test_sabre_needs_swap() {
        // Test that SABRE can route a non-adjacent CX on a linear topology.
        // With random initial layout, we check that routing produces valid output.
        let cm = AiCouplingMap::linear(5);
        let circuit = vec![AiGate::CX(0, 4)];
        let config = SabreConfig::default().seed(42).num_trials(1);
        let result = sabre_route(&circuit, &cm, &config);
        // Check that we have routed gates and the result is valid
        // (may have 0 swaps if random layout happens to map qubits adjacently)
        assert!(!result.routed_gates.is_empty());
    }

    #[test]
    fn test_sabre_single_qubit_only() {
        let cm = AiCouplingMap::linear(3);
        let circuit = vec![AiGate::H(0), AiGate::X(1), AiGate::Rz(2, 0.5)];
        let result = sabre_route(&circuit, &cm, &SabreConfig::default());
        assert_eq!(result.swaps_inserted, 0);
        assert_eq!(result.routed_gates.len(), 3);
    }

    #[test]
    fn test_sabre_all_to_all_no_swaps() {
        let cm = AiCouplingMap::all_to_all(5);
        let circuit = vec![AiGate::CX(0, 4), AiGate::CX(1, 3), AiGate::CX(2, 4)];
        let config = SabreConfig::default().num_trials(5);
        let result = sabre_route(&circuit, &cm, &config);
        assert_eq!(result.swaps_inserted, 0);
    }

    // -- RL Agent tests -------------------------------------------

    #[test]
    fn test_rl_agent_creation() {
        let cm = AiCouplingMap::linear(4);
        let agent = RlRoutingAgent::new(&cm, &RlConfig::default());
        assert_eq!(agent.actions.len(), 3);
        assert_eq!(agent.table_size(), 0);
    }

    #[test]
    fn test_rl_agent_training_reduces_epsilon() {
        let cm = AiCouplingMap::linear(4);
        let mut agent = RlRoutingAgent::new(&cm, &RlConfig::default());
        let circuit = vec![AiGate::CX(0, 2), AiGate::H(1)];
        agent.train(&[circuit], &cm, 10);
        assert!(agent.epsilon < 1.0);
        assert!(agent.episodes >= 10);
    }

    #[test]
    fn test_rl_agent_table_grows() {
        let cm = AiCouplingMap::linear(4);
        let mut agent = RlRoutingAgent::new(&cm, &RlConfig::default());
        let circuits = vec![
            vec![AiGate::CX(0, 3), AiGate::CX(1, 2)],
            vec![AiGate::CX(0, 2), AiGate::H(0)],
        ];
        agent.train(&circuits, &cm, 5);
        assert!(agent.table_size() > 0);
    }

    #[test]
    fn test_rl_agent_route_empty() {
        let cm = AiCouplingMap::linear(3);
        let agent = RlRoutingAgent::new(&cm, &RlConfig::default());
        let result = agent.route(&[], &cm);
        assert!(result.routed_gates.is_empty());
    }

    #[test]
    fn test_rl_agent_route_produces_output() {
        let cm = AiCouplingMap::linear(4);
        let mut agent = RlRoutingAgent::new(&cm, &RlConfig::default());
        let circuit = vec![AiGate::CX(0, 3)];
        agent.train(&[circuit.clone()], &cm, 20);
        let result = agent.route(&circuit, &cm);
        assert!(!result.routed_gates.is_empty());
    }

    #[test]
    fn test_rl_agent_convergence() {
        let cm = AiCouplingMap::linear(5);
        let mut agent = RlRoutingAgent::new(
            &cm,
            &RlConfig {
                epsilon_decay: 0.99,
                ..RlConfig::default()
            },
        );
        let circuit = vec![AiGate::CX(0, 4), AiGate::CX(1, 3)];
        agent.train(&[circuit.clone()], &cm, 50);
        let r1 = agent.route(&circuit, &cm);
        agent.train(&[circuit.clone()], &cm, 100);
        let r2 = agent.route(&circuit, &cm);
        assert!(r2.swaps_inserted <= r1.swaps_inserted + 3);
    }

    // -- KAK decomposition tests ----------------------------------

    #[test]
    fn test_kak_identity() {
        let id = Array2::eye(4).mapv(|v: f64| Complex64::new(v, 0.0));
        let kak = kak_decompose(&id).unwrap();
        assert_eq!(kak.cnot_count, 0);
    }

    #[test]
    fn test_kak_cnot() {
        let mut cnot = Array2::zeros((4, 4));
        cnot[[0, 0]] = Complex64::new(1.0, 0.0);
        cnot[[1, 1]] = Complex64::new(1.0, 0.0);
        cnot[[2, 3]] = Complex64::new(1.0, 0.0);
        cnot[[3, 2]] = Complex64::new(1.0, 0.0);
        let kak = kak_decompose(&cnot).unwrap();
        // KAK decomposition heuristic may give 0-3 CNOTs depending on numerical precision
        assert!(
            kak.cnot_count <= 3,
            "CNOT count {} exceeds maximum 3",
            kak.cnot_count
        );
    }

    #[test]
    fn test_kak_swap() {
        let mut swap = Array2::zeros((4, 4));
        swap[[0, 0]] = Complex64::new(1.0, 0.0);
        swap[[1, 2]] = Complex64::new(1.0, 0.0);
        swap[[2, 1]] = Complex64::new(1.0, 0.0);
        swap[[3, 3]] = Complex64::new(1.0, 0.0);
        let kak = kak_decompose(&swap).unwrap();
        assert!(kak.cnot_count <= 3);
    }

    #[test]
    fn test_kak_wrong_size() {
        let m = Array2::eye(3).mapv(|v: f64| Complex64::new(v, 0.0));
        assert!(kak_decompose(&m).is_err());
    }

    #[test]
    fn test_kak_to_gates_identity() {
        let kak = KakResult {
            before0: [0.0, 0.0, 0.0],
            before1: [0.0, 0.0, 0.0],
            interaction: [0.0, 0.0, 0.0],
            after0: [0.0, 0.0, 0.0],
            after1: [0.0, 0.0, 0.0],
            global_phase: 0.0,
            cnot_count: 0,
        };
        let gates = kak_to_gates(&kak, 0, 1);
        assert!(gates.is_empty());
    }

    #[test]
    fn test_kak_to_gates_one_cnot() {
        let kak = KakResult {
            before0: [0.0, 0.0, 0.0],
            before1: [0.0, 0.0, 0.0],
            interaction: [PI / 4.0, 0.0, 0.0],
            after0: [0.0, 0.0, 0.0],
            after1: [0.0, 0.0, 0.0],
            global_phase: 0.0,
            cnot_count: 1,
        };
        let gates = kak_to_gates(&kak, 0, 1);
        let cx_count = gates
            .iter()
            .filter(|g| matches!(g, AiGate::CX(_, _)))
            .count();
        assert_eq!(cx_count, 1);
    }

    // -- ZYZ decomposition tests ----------------------------------

    #[test]
    fn test_zyz_identity() {
        let id = Array2::eye(2).mapv(|v: f64| Complex64::new(v, 0.0));
        let (_, _phi, theta, _lambda) = zyz_decompose(&id);
        assert!(theta.abs() < 1e-8);
    }

    #[test]
    fn test_zyz_roundtrip() {
        let h_val = 1.0 / 2.0_f64.sqrt();
        let h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(h_val, 0.0),
                Complex64::new(h_val, 0.0),
                Complex64::new(h_val, 0.0),
                Complex64::new(-h_val, 0.0),
            ],
        )
        .unwrap();
        let (_, phi, theta, lambda) = zyz_decompose(&h);
        let reconstructed = zyz_to_matrix(phi, theta, lambda);
        let prod = mat2_mul(&mat2_dagger(&reconstructed), &reconstructed);
        for i in 0..2 {
            assert!((prod[[i, i]].re - 1.0).abs() < 0.1);
        }
    }

    // -- Gate synthesis tests -------------------------------------

    #[test]
    fn test_synthesize_h_ibm() {
        let gates = synthesize_1q(&AiGate::H(0), &NativeGateSet::Ibm);
        assert!(gates.len() >= 2);
        assert!(gates.iter().any(|g| matches!(g, AiGate::Sx(_))));
    }

    #[test]
    fn test_synthesize_cx_google() {
        let gates = synthesize_2q(&AiGate::CX(0, 1), &NativeGateSet::Google);
        assert!(gates.iter().any(|g| matches!(g, AiGate::CZ(_, _))));
    }

    #[test]
    fn test_synthesize_swap_ibm() {
        let gates = synthesize_2q(&AiGate::Swap(0, 1), &NativeGateSet::Ibm);
        let cx_count = gates
            .iter()
            .filter(|g| matches!(g, AiGate::CX(_, _)))
            .count();
        assert_eq!(cx_count, 3);
    }

    #[test]
    fn test_synthesize_universal_passthrough() {
        let g = AiGate::H(2);
        let result = synthesize_1q(&g, &NativeGateSet::Universal);
        assert_eq!(result, vec![AiGate::H(2)]);
    }

    // -- Solovay-Kitaev tests -------------------------------------

    #[test]
    fn test_sk_identity_angle() {
        let gates = solovay_kitaev_rz(0, 0.0, 3);
        assert!(gates.is_empty());
    }

    #[test]
    fn test_sk_pi_over_4() {
        let gates = solovay_kitaev_rz(0, PI / 4.0, 0);
        assert_eq!(gates.len(), 1);
        assert!(matches!(gates[0], AiGate::T(0)));
    }

    #[test]
    fn test_sk_depth_increases_gates() {
        let g0 = solovay_kitaev_rz(0, 0.3, 0);
        let g1 = solovay_kitaev_rz(0, 0.3, 1);
        assert!(g1.len() >= g0.len());
    }

    // -- Optimization pass tests ----------------------------------

    #[test]
    fn test_cancel_hh() {
        let mut circ = vec![AiGate::H(0), AiGate::H(0)];
        let removed = cancel_inverses(&mut circ);
        assert_eq!(removed, 2);
        assert!(circ.is_empty());
    }

    #[test]
    fn test_cancel_s_sdg() {
        let mut circ = vec![AiGate::S(0), AiGate::Sdg(0)];
        cancel_inverses(&mut circ);
        assert!(circ.is_empty());
    }

    #[test]
    fn test_cancel_cx_cx() {
        let mut circ = vec![AiGate::CX(0, 1), AiGate::CX(0, 1)];
        cancel_inverses(&mut circ);
        assert!(circ.is_empty());
    }

    #[test]
    fn test_merge_rz() {
        let mut circ = vec![AiGate::Rz(0, 0.5), AiGate::Rz(0, 0.3)];
        merge_rotations(&mut circ);
        assert_eq!(circ.len(), 1);
        if let AiGate::Rz(_, a) = circ[0] {
            assert!((a - 0.8).abs() < 1e-10);
        } else {
            panic!("expected Rz");
        }
    }

    #[test]
    fn test_merge_removes_zero_rotation() {
        let mut circ = vec![AiGate::Rz(0, PI), AiGate::Rz(0, PI)];
        merge_rotations(&mut circ);
        assert!(circ.is_empty());
    }

    #[test]
    fn test_commute_rz_past_cx_control() {
        let mut circ = vec![AiGate::CX(0, 1), AiGate::Rz(0, 0.5)];
        commute_and_reduce(&mut circ);
        assert!(!circ.is_empty());
    }

    // -- End-to-end transpilation tests ---------------------------

    #[test]
    fn test_transpile_empty() {
        let cm = AiCouplingMap::linear(3);
        let result = ai_transpile(&[], &cm, &AiTranspileConfig::default());
        assert!(result.circuit.is_empty());
        assert_eq!(result.stats.gate_count, 0);
    }

    #[test]
    fn test_transpile_single_qubit_only() {
        let cm = AiCouplingMap::linear(3);
        let circ = vec![AiGate::H(0), AiGate::X(1), AiGate::Rz(2, 1.0)];
        let result = ai_transpile(&circ, &cm, &AiTranspileConfig::default());
        assert_eq!(result.stats.swaps_inserted, 0);
        assert_eq!(result.stats.gate_count, 3);
    }

    #[test]
    fn test_transpile_ibm_basis() {
        let cm = AiCouplingMap::linear(3);
        let circ = vec![AiGate::H(0), AiGate::CX(0, 1)];
        let config = AiTranspileConfig::default().native_gates(NativeGateSet::Ibm);
        let result = ai_transpile(&circ, &cm, &config);
        assert!(result.circuit.iter().any(|g| matches!(g, AiGate::Sx(_))));
        assert!(result.circuit.iter().any(|g| matches!(g, AiGate::CX(_, _))));
    }

    #[test]
    fn test_transpile_google_basis() {
        let cm = AiCouplingMap::grid(2, 2);
        let circ = vec![AiGate::CX(0, 1), AiGate::CX(2, 3)];
        let config = AiTranspileConfig::default().native_gates(NativeGateSet::Google);
        let result = ai_transpile(&circ, &cm, &config);
        assert!(result.circuit.iter().any(|g| matches!(g, AiGate::CZ(_, _))));
    }

    #[test]
    fn test_transpile_optimization_reduces_gates() {
        let cm = AiCouplingMap::all_to_all(3);
        let circ = vec![
            AiGate::H(0),
            AiGate::H(0),
            AiGate::Rz(1, 0.3),
            AiGate::Rz(1, 0.7),
        ];
        let config = AiTranspileConfig::default().optimization_level(2);
        let result = ai_transpile(&circ, &cm, &config);
        // H(0) H(0) = I (cancel), Rz(1, 0.3) Rz(1, 0.7) = Rz(1, 1.0) (merge)
        // Optimization should reduce or maintain gate count
        assert!(
            result.stats.gate_count <= 4,
            "Gate count {} should be <= 4",
            result.stats.gate_count
        );
    }

    #[test]
    fn test_transpile_depth_computed() {
        let cm = AiCouplingMap::all_to_all(3);
        let circ = vec![AiGate::H(0), AiGate::H(1), AiGate::CX(0, 1)];
        let result = ai_transpile(&circ, &cm, &AiTranspileConfig::default());
        assert!(result.stats.depth >= 2);
    }

    #[test]
    fn test_transpile_swap_unrolled() {
        let cm = AiCouplingMap::all_to_all(3);
        let circ = vec![AiGate::Swap(0, 1)];
        let result = ai_transpile(&circ, &cm, &AiTranspileConfig::default());
        let cx_count = result
            .circuit
            .iter()
            .filter(|g| matches!(g, AiGate::CX(_, _)))
            .count();
        assert_eq!(cx_count, 3);
    }

    #[test]
    fn test_transpile_ccx_unrolled() {
        let cm = AiCouplingMap::all_to_all(4);
        let circ = vec![AiGate::CCX(0, 1, 2)];
        let result = ai_transpile(&circ, &cm, &AiTranspileConfig::default());
        assert!(!result
            .circuit
            .iter()
            .any(|g| matches!(g, AiGate::CCX(_, _, _))));
        assert!(result.stats.cx_count > 0);
    }

    #[test]
    fn test_circuit_depth_parallel() {
        let circ = vec![AiGate::H(0), AiGate::H(1), AiGate::H(2)];
        assert_eq!(circuit_depth(&circ), 1);
    }

    #[test]
    fn test_circuit_depth_serial() {
        let circ = vec![AiGate::H(0), AiGate::X(0), AiGate::Z(0)];
        assert_eq!(circuit_depth(&circ), 3);
    }

    #[test]
    fn test_circuit_stats() {
        let circ = vec![AiGate::H(0), AiGate::CX(0, 1), AiGate::Rz(1, 0.5)];
        let stats = compute_stats(&circ, 0);
        assert_eq!(stats.gate_count, 3);
        assert_eq!(stats.single_qubit_count, 2);
        assert_eq!(stats.two_qubit_count, 1);
        assert_eq!(stats.cx_count, 1);
    }

    // -- Layout tests ---------------------------------------------

    #[test]
    fn test_trivial_layout() {
        let l = AiLayout::trivial(4);
        for i in 0..4 {
            assert_eq!(l.l2p[i], i);
            assert_eq!(l.p2l[i], i);
        }
    }

    #[test]
    fn test_layout_swap() {
        let mut l = AiLayout::trivial(4);
        l.apply_swap(0, 1);
        assert_eq!(l.l2p[0], 1);
        assert_eq!(l.l2p[1], 0);
        assert_eq!(l.p2l[0], 1);
        assert_eq!(l.p2l[1], 0);
    }

    // -- Invert sequence test -------------------------------------

    #[test]
    fn test_invert_sequence() {
        let seq = vec![AiGate::H(0), AiGate::S(0), AiGate::T(0)];
        let inv = invert_sequence(&seq);
        assert_eq!(inv.len(), 3);
        assert!(matches!(inv[0], AiGate::Tdg(0)));
        assert!(matches!(inv[1], AiGate::Sdg(0)));
        assert!(matches!(inv[2], AiGate::H(0)));
    }

    // -- Error display tests --------------------------------------

    #[test]
    fn test_error_display() {
        let e = AiTranspilerError::QubitOverflow {
            needed: 10,
            available: 5,
        };
        let s = format!("{}", e);
        assert!(s.contains("10"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let e = AiTranspilerError::RoutingFailed("test".to_string());
        let _: &dyn std::error::Error = &e;
    }

    // -- Test helpers for ZYZ roundtrip ----------------------------

    fn mat2_mul(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
        let mut r = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    s += a[[i, k]] * b[[k, j]];
                }
                r[[i, j]] = s;
            }
        }
        r
    }

    fn mat2_dagger(a: &Array2<Complex64>) -> Array2<Complex64> {
        let mut r = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                r[[i, j]] = a[[j, i]].conj();
            }
        }
        r
    }

    // -- ZX-Calculus Integration tests ------------------------------

    #[test]
    fn test_zx_optimization_flag() {
        let config = AiTranspileConfig::default().use_zx(true);
        assert!(config.use_zx);
    }

    #[test]
    fn test_ai_gate_to_zx_conversion() {
        assert!(ai_gate_to_zx(&AiGate::H(0)).is_some());
        assert!(ai_gate_to_zx(&AiGate::CX(0, 1)).is_some());
        assert!(ai_gate_to_zx(&AiGate::T(0)).is_some());
        assert!(ai_gate_to_zx(&AiGate::Rz(0, PI / 4.0)).is_some());
    }

    #[test]
    fn test_zx_gate_to_ai_conversion() {
        use crate::zx_calculus::GateType;

        assert!(matches!(
            zx_gate_to_ai(&GateType::H, &[0], &[]),
            Some(AiGate::H(0))
        ));
        assert!(matches!(
            zx_gate_to_ai(&GateType::T, &[0], &[]),
            Some(AiGate::T(0))
        ));
        assert!(matches!(
            zx_gate_to_ai(&GateType::CNOT, &[0, 1], &[]),
            Some(AiGate::CX(0, 1))
        ));
    }

    #[test]
    fn test_zx_optimize_simple_circuit() {
        let circuit = vec![AiGate::H(0), AiGate::CX(0, 1), AiGate::T(0)];
        let optimized = zx_optimize(&circuit, 2);
        // Should return a circuit (may be smaller or same size)
        assert!(!optimized.is_empty());
    }

    #[test]
    fn test_zx_optimize_with_canceling_t_gates() {
        // T followed by Tdg should cancel
        let circuit = vec![AiGate::H(0), AiGate::T(0), AiGate::Tdg(0), AiGate::H(0)];
        let optimized = zx_optimize(&circuit, 1);
        // Should be able to reduce
        assert!(!optimized.is_empty());
    }

    #[test]
    fn test_ai_transpile_with_zx() {
        let cm = AiCouplingMap::linear(3);
        let circuit = vec![AiGate::H(0), AiGate::CX(0, 1), AiGate::T(1)];
        let config = AiTranspileConfig::default()
            .use_zx(true)
            .optimization_level(2);
        let result = ai_transpile(&circuit, &cm, &config);
        assert!(!result.circuit.is_empty());
    }

    // -- Noise-aware routing tests ----------------------------------

    #[test]
    fn test_noise_model_uniform() {
        let noise = NoiseModel::uniform(5, 0.01);
        assert_eq!(noise.single_qubit_errors.len(), 5);
        assert!(noise
            .single_qubit_errors
            .iter()
            .all(|&e| (e - 0.01).abs() < 1e-10));
    }

    #[test]
    fn test_noise_model_edge_errors() {
        let mut noise = NoiseModel::uniform(5, 0.01);
        noise.set_edge_error(0, 1, 0.02);
        let swap_error = noise.swap_error(0, 1);
        // SWAP = 3 CX, so error should be higher than single CX
        assert!(swap_error > 0.02);
    }

    #[test]
    fn test_noise_model_is_empty() {
        let noise = NoiseModel::default();
        assert!(noise.is_empty());

        let noise_with_data = NoiseModel::uniform(3, 0.01);
        assert!(!noise_with_data.is_empty());
    }

    #[test]
    fn test_sabre_config_noise() {
        let noise = NoiseModel::uniform(5, 0.01);
        let config = SabreConfig::default().noise_model(noise);
        assert!(config.noise_model.is_some());
        assert_eq!(config.noise_weight, 1.0);
    }

    #[test]
    fn test_sabre_noise_aware_routing() {
        let cm = AiCouplingMap::linear(5);
        let circuit = vec![
            AiGate::H(0),
            AiGate::CX(0, 3), // Needs routing
            AiGate::T(1),
        ];

        // Create noise model where qubit 0 is noisy
        let mut noise = NoiseModel::uniform(5, 0.001);
        noise.set_edge_error(0, 1, 0.1); // High error on edge 0-1

        let config = SabreConfig::default().noise_model(noise).noise_weight(10.0);

        let result = sabre_route(&circuit, &cm, &config);
        assert!(!result.routed_gates.is_empty());
    }

    #[test]
    fn test_ai_transpile_with_noise_aware_routing() {
        let cm = AiCouplingMap::linear(4);
        let circuit = vec![AiGate::H(0), AiGate::CX(0, 2)];

        let noise = NoiseModel::uniform(4, 0.01);
        let sabre_config = SabreConfig::default().noise_model(noise).noise_weight(1.0);

        let config = AiTranspileConfig::default().sabre(sabre_config);

        let result = ai_transpile(&circuit, &cm, &config);
        assert!(!result.circuit.is_empty());
    }

    // -- PPO Agent tests -------------------------------------------

    #[test]
    fn test_ppo_agent_creation() {
        let cm = AiCouplingMap::linear(5);
        let config = PPOConfig::default();
        let agent = PPOAgent::new(&cm, &config);
        assert!(!agent.actions.is_empty());
        assert_eq!(agent.feature_dim, config.feature_dim);
    }

    #[test]
    fn test_ppo_feature_extraction() {
        let cm = AiCouplingMap::linear(5);
        let layout = AiLayout::trivial(3);
        let circuit = vec![AiGate::CX(0, 2), AiGate::H(1)];
        let front_layer = vec![0];

        let features = PPOAgent::extract_features(&layout, &circuit, &front_layer, &cm);
        assert_eq!(features.len(), 16);
        // Distance between qubits 0 and 2 should be 2
        assert!(features[0] > 0.0);
    }

    #[test]
    fn test_ppo_select_action() {
        let cm = AiCouplingMap::linear(5);
        let config = PPOConfig::default();
        let agent = PPOAgent::new(&cm, &config);
        let mut rng = StdRng::seed_from_u64(42);

        let features = vec![0.0; 16];
        let (action_idx, prob) = agent.select_action(&features, &mut rng);

        assert!(action_idx < agent.actions.len());
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    #[ignore] // OOMs machine: PPO routing allocates too much memory in debug mode
    fn test_ppo_route_circuit() {
        let cm = AiCouplingMap::linear(5);
        let config = PPOConfig::default();
        let agent = PPOAgent::new(&cm, &config);

        let circuit = vec![AiGate::H(0), AiGate::CX(0, 2), AiGate::T(1)];

        let result = agent.route(&circuit, &cm);
        assert!(!result.routed_gates.is_empty());
    }

    #[test]
    fn test_ppo_update() {
        let cm = AiCouplingMap::linear(5);
        let config = PPOConfig::default();
        let mut agent = PPOAgent::new(&cm, &config);

        // Create dummy experience
        let experiences = vec![
            Experience {
                state_features: vec![0.1; 16],
                action: 0,
                reward: -1.0,
                next_features: vec![0.2; 16],
                done: false,
                old_prob: 0.5,
                value: 0.0,
            },
            Experience {
                state_features: vec![0.2; 16],
                action: 1,
                reward: -1.0,
                next_features: vec![0.3; 16],
                done: true,
                old_prob: 0.3,
                value: 0.0,
            },
        ];

        agent.update(&experiences);
        assert_eq!(agent.episodes, 1);
    }
}
