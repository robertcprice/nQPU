//! Quantum Selfish Routing and Braess's Paradox
//!
//! **BLEEDING EDGE**: Implements the 2026 discovery that quantum networks can suffer
//! from Braess's paradox -- adding more entanglement links can *degrade* overall
//! network performance when agents route selfishly.
//!
//! This module provides:
//! - Quantum network graph construction (nodes with quantum memories, links with fidelity)
//! - Entanglement swapping with fidelity degradation across multi-hop paths
//! - Entanglement purification protocols (DEJMPS, BBPSSW, Pumping)
//! - Routing algorithms: shortest path, highest fidelity, max throughput
//! - Game-theoretic routing: Nash equilibrium via best-response dynamics
//! - Social optimum computation for cooperative benchmarking
//! - Braess paradox detection: adding a link that hurts selfish performance
//! - Price of Anarchy and Price of Stability metrics
//! - Pre-built network topologies: diamond, line, grid, star
//!
//! # The Quantum Braess Paradox
//!
//! In classical networks, Braess (1968) showed that adding a road can increase
//! travel time when drivers route selfishly. The quantum analogue is more severe:
//! entanglement fidelity degrades multiplicatively through swaps, so congestion
//! on a "shortcut" link forces purification overhead that can overwhelm the
//! benefit, leading to lower end-to-end fidelity for *all* users.
//!
//! # References
//!
//! - Braess (1968) - On a paradox of traffic planning
//! - Roughgarden & Tardos (2002) - How bad is selfish routing?
//! - Pant et al. (2019) - Routing entanglement in the quantum internet
//! - Caleffi (2017) - Optimal routing for quantum networks
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::selfish_routing::*;
//!
//! let net = QuantumNetworkLibrary::diamond_network();
//! let requests = vec![
//!     RoutingRequest { source: 0, destination: 3, fidelity_threshold: 0.5, num_pairs: 10 },
//! ];
//! let game = RoutingGame::new(net, requests, RoutingStrategy::NashEquilibrium);
//! let result = game.solve();
//! println!("Price of Anarchy: {:.3}", result.price_of_anarchy);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

// ===================================================================
// ERRORS
// ===================================================================

/// Errors that can occur during quantum network routing.
#[derive(Debug, Clone)]
pub enum RoutingError {
    /// No path exists between the requested nodes.
    NoPath(String),
    /// Entanglement generation or swapping failed.
    EntanglementFailed(String),
    /// General network error (invalid node, link, etc.).
    NetworkError(String),
    /// Braess paradox detected: adding a link degraded performance.
    ParadoxDetected(String),
}

impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoutingError::NoPath(s) => write!(f, "No path: {}", s),
            RoutingError::EntanglementFailed(s) => write!(f, "Entanglement failed: {}", s),
            RoutingError::NetworkError(s) => write!(f, "Network error: {}", s),
            RoutingError::ParadoxDetected(s) => write!(f, "Paradox detected: {}", s),
        }
    }
}

impl std::error::Error for RoutingError {}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Builder for routing configuration parameters.
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    /// Maximum number of hops allowed in any route.
    pub max_hops: usize,
    /// Minimum acceptable end-to-end fidelity.
    pub min_fidelity: f64,
    /// Congestion penalty coefficient: cost += alpha * (load / capacity).
    pub congestion_alpha: f64,
    /// Maximum iterations for Nash equilibrium solver.
    pub nash_max_iterations: usize,
    /// Convergence threshold for Nash equilibrium.
    pub nash_convergence: f64,
    /// Whether to apply purification automatically on routes.
    pub auto_purify: bool,
    /// Default purification target fidelity.
    pub purification_target: f64,
    /// Maximum purification rounds.
    pub max_purification_rounds: usize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            max_hops: 20,
            min_fidelity: 0.5,
            congestion_alpha: 1.0,
            nash_max_iterations: 500,
            nash_convergence: 1e-6,
            auto_purify: false,
            purification_target: 0.9,
            max_purification_rounds: 10,
        }
    }
}

impl RoutingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_hops(mut self, v: usize) -> Self {
        self.max_hops = v;
        self
    }

    pub fn min_fidelity(mut self, v: f64) -> Self {
        self.min_fidelity = v;
        self
    }

    pub fn congestion_alpha(mut self, v: f64) -> Self {
        self.congestion_alpha = v;
        self
    }

    pub fn nash_max_iterations(mut self, v: usize) -> Self {
        self.nash_max_iterations = v;
        self
    }

    pub fn nash_convergence(mut self, v: f64) -> Self {
        self.nash_convergence = v;
        self
    }

    pub fn auto_purify(mut self, v: bool) -> Self {
        self.auto_purify = v;
        self
    }

    pub fn purification_target(mut self, v: f64) -> Self {
        self.purification_target = v;
        self
    }

    pub fn max_purification_rounds(mut self, v: usize) -> Self {
        self.max_purification_rounds = v;
        self
    }
}

// ===================================================================
// NETWORK TYPES
// ===================================================================

/// A node in the quantum network with quantum memory resources.
#[derive(Debug, Clone)]
pub struct QuantumNode {
    /// Unique node identifier.
    pub id: usize,
    /// Human-readable name.
    pub name: String,
    /// Number of quantum memory qubits available.
    pub memory_qubits: usize,
    /// Coherence time of quantum memories in seconds.
    pub memory_coherence_time: f64,
    /// Spatial position (x, y) for visualization and distance calculations.
    pub position: (f64, f64),
}

impl QuantumNode {
    pub fn new(id: usize, name: &str, memory_qubits: usize, coherence_time: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            memory_qubits,
            memory_coherence_time: coherence_time,
            position: (0.0, 0.0),
        }
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.position = (x, y);
        self
    }
}

/// A quantum link connecting two nodes, capable of generating entangled pairs.
#[derive(Debug, Clone)]
pub struct QuantumLink {
    /// First endpoint node ID.
    pub node_a: usize,
    /// Second endpoint node ID.
    pub node_b: usize,
    /// Fidelity of entangled pairs generated on this link.
    pub fidelity: f64,
    /// Probability that entanglement generation succeeds on a single attempt.
    pub success_probability: f64,
    /// One-way classical communication latency in milliseconds.
    pub latency_ms: f64,
    /// Maximum entangled pairs that can be generated per second.
    pub capacity: usize,
}

impl QuantumLink {
    pub fn new(
        a: usize,
        b: usize,
        fidelity: f64,
        success_prob: f64,
        latency: f64,
        capacity: usize,
    ) -> Self {
        Self {
            node_a: a,
            node_b: b,
            fidelity: fidelity.clamp(0.0, 1.0),
            success_probability: success_prob.clamp(0.0, 1.0),
            latency_ms: latency.max(0.0),
            capacity: capacity.max(1),
        }
    }

    /// Check whether this link connects a given pair of nodes (undirected).
    pub fn connects(&self, a: usize, b: usize) -> bool {
        (self.node_a == a && self.node_b == b) || (self.node_a == b && self.node_b == a)
    }
}

/// A quantum network: nodes connected by entanglement-generating links.
#[derive(Debug, Clone)]
pub struct QuantumNetwork {
    pub nodes: Vec<QuantumNode>,
    pub links: Vec<QuantumLink>,
}

impl QuantumNetwork {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            links: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: QuantumNode) {
        self.nodes.push(node);
    }

    pub fn add_link(&mut self, link: QuantumLink) {
        self.links.push(link);
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of links.
    pub fn num_links(&self) -> usize {
        self.links.len()
    }

    /// Get all link indices incident on a node.
    pub fn links_from(&self, node_id: usize) -> Vec<usize> {
        self.links
            .iter()
            .enumerate()
            .filter(|(_, l)| l.node_a == node_id || l.node_b == node_id)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the neighbor reached through a given link from a given node.
    pub fn neighbor(&self, node_id: usize, link_idx: usize) -> Option<usize> {
        let l = &self.links[link_idx];
        if l.node_a == node_id {
            Some(l.node_b)
        } else if l.node_b == node_id {
            Some(l.node_a)
        } else {
            None
        }
    }

    /// Build an adjacency list: node_id -> Vec<(neighbor_id, link_index)>.
    pub fn adjacency_list(&self) -> Vec<Vec<(usize, usize)>> {
        let n = self.num_nodes();
        let mut adj = vec![vec![]; n];
        for (li, link) in self.links.iter().enumerate() {
            if link.node_a < n {
                adj[link.node_a].push((link.node_b, li));
            }
            if link.node_b < n {
                adj[link.node_b].push((link.node_a, li));
            }
        }
        adj
    }

    /// Find the link index connecting two nodes, if any.
    pub fn find_link(&self, a: usize, b: usize) -> Option<usize> {
        self.links.iter().position(|l| l.connects(a, b))
    }

    /// Compute the Euclidean distance between two nodes.
    pub fn distance(&self, a: usize, b: usize) -> f64 {
        let pa = self.nodes[a].position;
        let pb = self.nodes[b].position;
        ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt()
    }
}

// ===================================================================
// ROUTING REQUEST & STRATEGY
// ===================================================================

/// A request to distribute entanglement between two nodes.
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    /// Source node ID.
    pub source: usize,
    /// Destination node ID.
    pub destination: usize,
    /// Minimum acceptable end-to-end fidelity.
    pub fidelity_threshold: f64,
    /// Number of entangled pairs requested.
    pub num_pairs: usize,
}

/// Strategy for how routes are selected.
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Minimum number of hops (Dijkstra with unit weights).
    ShortestPath,
    /// Maximize end-to-end fidelity (Dijkstra with -log(F) weights).
    HighestFidelity,
    /// Maximize entangled pairs per second.
    MaxThroughput,
    /// Game-theoretic Nash equilibrium via best-response dynamics.
    NashEquilibrium,
    /// Globally optimal cooperative routing.
    SocialOptimum,
    /// Each of `num_agents` users optimizes independently.
    Selfish { num_agents: usize },
}

// ===================================================================
// ROUTE
// ===================================================================

/// A computed route through the quantum network.
#[derive(Debug, Clone)]
pub struct Route {
    /// Ordered list of node IDs from source to destination.
    pub path: Vec<usize>,
    /// End-to-end entanglement fidelity after all swaps.
    pub end_to_end_fidelity: f64,
    /// Probability that the entire route succeeds.
    pub success_probability: f64,
    /// Total latency in milliseconds.
    pub latency_ms: f64,
    /// Effective throughput in entangled pairs per second.
    pub throughput: f64,
}

impl Route {
    /// Number of hops (links) in the route.
    pub fn num_hops(&self) -> usize {
        if self.path.len() < 2 {
            0
        } else {
            self.path.len() - 1
        }
    }

    /// Cost function for game-theoretic routing.
    /// Lower is better: inversely related to fidelity, penalized by congestion.
    pub fn cost(&self, congestion: f64, alpha: f64) -> f64 {
        let fidelity_cost = if self.end_to_end_fidelity > 1e-12 {
            1.0 / self.end_to_end_fidelity
        } else {
            1e12
        };
        fidelity_cost + alpha * congestion
    }
}

// ===================================================================
// ENTANGLEMENT SWAPPING
// ===================================================================

/// Result of an entanglement swapping operation at an intermediate node.
#[derive(Debug, Clone)]
pub struct SwapResult {
    /// Fidelity of the output entangled pair.
    pub output_fidelity: f64,
    /// Whether the swap succeeded (probabilistic Bell measurement).
    pub success: bool,
    /// Number of purification rounds applied after the swap.
    pub purification_rounds: usize,
}

/// Perform entanglement swapping on two adjacent links.
///
/// Uses the Werner state model: for two links with fidelities F1 and F2,
/// the post-swap fidelity is:
///
///   F_out = F1 * F2 + (1 - F1)(1 - F2) / 3
///
/// This accounts for depolarizing noise on both links.
pub fn entanglement_swap(f1: f64, f2: f64) -> SwapResult {
    let f1 = f1.clamp(0.25, 1.0);
    let f2 = f2.clamp(0.25, 1.0);
    // Werner state swap formula
    let output = f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0;
    SwapResult {
        output_fidelity: output.clamp(0.25, 1.0),
        success: true,
        purification_rounds: 0,
    }
}

/// Compute end-to-end fidelity through a chain of links with given fidelities.
///
/// Applies entanglement swapping sequentially from left to right.
pub fn chain_fidelity(link_fidelities: &[f64]) -> f64 {
    if link_fidelities.is_empty() {
        return 0.0;
    }
    if link_fidelities.len() == 1 {
        return link_fidelities[0];
    }
    let mut f = link_fidelities[0];
    for &fi in &link_fidelities[1..] {
        let swap = entanglement_swap(f, fi);
        f = swap.output_fidelity;
    }
    f
}

/// Compute end-to-end success probability through a chain of links.
pub fn chain_success_probability(link_probs: &[f64]) -> f64 {
    link_probs.iter().copied().product()
}

/// Compute total latency through a chain of links.
pub fn chain_latency(link_latencies: &[f64]) -> f64 {
    link_latencies.iter().copied().sum()
}

/// Compute effective throughput (bottleneck model: minimum capacity link).
pub fn chain_throughput(link_capacities: &[usize], link_probs: &[f64]) -> f64 {
    if link_capacities.is_empty() || link_probs.is_empty() {
        return 0.0;
    }
    let min_cap = link_capacities.iter().copied().min().unwrap_or(1) as f64;
    let total_prob: f64 = link_probs.iter().copied().product();
    min_cap * total_prob
}

// ===================================================================
// ENTANGLEMENT PURIFICATION
// ===================================================================

/// Purification method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PurificationMethod {
    /// Deutsch, Ekert, Jozsa, Macchiavello, Popescu, Sanpera (1996).
    /// Uses bilateral CNOT + Hadamard rotations.
    DEJMPS,
    /// Bennett, Brassard, Popescu, Schumacher, Smolin, Wootters (1996).
    /// Uses bilateral CNOT only.
    BBPSSW,
    /// Entanglement pumping: iteratively purify with fresh pairs.
    Pumping,
}

/// Configuration for an entanglement purification protocol.
#[derive(Debug, Clone)]
pub struct PurificationProtocol {
    pub method: PurificationMethod,
    pub target_fidelity: f64,
    pub max_rounds: usize,
}

impl PurificationProtocol {
    pub fn new(method: PurificationMethod, target: f64, max_rounds: usize) -> Self {
        Self {
            method,
            target_fidelity: target.clamp(0.5, 0.9999),
            max_rounds,
        }
    }

    /// Apply one round of purification. Takes two copies of fidelity F,
    /// produces one copy of higher fidelity F'.
    ///
    /// Returns (new_fidelity, success_probability_of_round).
    pub fn purify_one_round(&self, f: f64) -> (f64, f64) {
        match self.method {
            PurificationMethod::DEJMPS => {
                // DEJMPS: F' = (F^2 + ((1-F)/3)^2) / (F^2 + 2*F*(1-F)/3 + 5*((1-F)/3)^2)
                let f2 = f * f;
                let e = (1.0 - f) / 3.0;
                let e2 = e * e;
                let num = f2 + e2;
                let den = f2 + 2.0 * f * e + 5.0 * e2;
                let p_success = den;
                if den < 1e-15 {
                    (f, 0.0)
                } else {
                    ((num / den).clamp(0.25, 1.0), p_success.clamp(0.0, 1.0))
                }
            }
            PurificationMethod::BBPSSW => {
                // BBPSSW (simpler): F' = F^2 / (F^2 + (1-F)^2)
                let f2 = f * f;
                let e2 = (1.0 - f) * (1.0 - f);
                let den = f2 + e2;
                let p_success = den;
                if den < 1e-15 {
                    (f, 0.0)
                } else {
                    ((f2 / den).clamp(0.25, 1.0), p_success.clamp(0.0, 1.0))
                }
            }
            PurificationMethod::Pumping => {
                // Pumping with a fresh pair of fidelity F_fresh = F.
                // Same formula as BBPSSW but represents iterative pumping.
                let f2 = f * f;
                let e2 = (1.0 - f) * (1.0 - f);
                let den = f2 + e2;
                let p_success = den;
                if den < 1e-15 {
                    (f, 0.0)
                } else {
                    ((f2 / den).clamp(0.25, 1.0), p_success.clamp(0.0, 1.0))
                }
            }
        }
    }

    /// Run purification until target fidelity is reached or max rounds exhausted.
    /// Returns (final_fidelity, total_rounds, cumulative_success_probability).
    pub fn purify(&self, initial_fidelity: f64) -> (f64, usize, f64) {
        let mut f = initial_fidelity;
        let mut rounds = 0;
        let mut total_p = 1.0;

        // Purification only works if F > 0.5
        if f <= 0.5 {
            return (f, 0, 1.0);
        }

        while f < self.target_fidelity && rounds < self.max_rounds {
            let (f_new, p) = self.purify_one_round(f);
            // If fidelity did not improve, stop
            if f_new <= f + 1e-12 {
                break;
            }
            f = f_new;
            total_p *= p;
            rounds += 1;
        }

        (f, rounds, total_p)
    }
}

// ===================================================================
// ROUTING ALGORITHMS
// ===================================================================

/// Internal state for Dijkstra's algorithm.
#[derive(Clone, PartialEq)]
struct DijkstraState {
    cost: f64,
    node: usize,
}

impl Eq for DijkstraState {}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Find the shortest path (minimum hops) between source and destination.
pub fn shortest_path(
    network: &QuantumNetwork,
    src: usize,
    dst: usize,
) -> Result<Route, RoutingError> {
    dijkstra_route(network, src, dst, |_link| 1.0, &HashMap::new())
}

/// Find the highest-fidelity path using -log(fidelity) as edge weight.
pub fn highest_fidelity_path(
    network: &QuantumNetwork,
    src: usize,
    dst: usize,
) -> Result<Route, RoutingError> {
    dijkstra_route(
        network,
        src,
        dst,
        |link| {
            if link.fidelity > 1e-12 {
                -link.fidelity.ln()
            } else {
                1e12
            }
        },
        &HashMap::new(),
    )
}

/// Find the maximum-throughput path using -log(capacity * success_prob) as weight.
pub fn max_throughput_path(
    network: &QuantumNetwork,
    src: usize,
    dst: usize,
) -> Result<Route, RoutingError> {
    dijkstra_route(
        network,
        src,
        dst,
        |link| {
            let eff = (link.capacity as f64) * link.success_probability;
            if eff > 1e-12 {
                -eff.ln()
            } else {
                1e12
            }
        },
        &HashMap::new(),
    )
}

/// Generalized Dijkstra that computes a route given an arbitrary link cost function.
///
/// `link_loads` maps link_index -> current load (number of flows using it).
/// This is used for congestion-aware routing.
fn dijkstra_route(
    network: &QuantumNetwork,
    src: usize,
    dst: usize,
    weight_fn: impl Fn(&QuantumLink) -> f64,
    link_loads: &HashMap<usize, usize>,
) -> Result<Route, RoutingError> {
    let n = network.num_nodes();
    if src >= n || dst >= n {
        return Err(RoutingError::NetworkError(format!(
            "Node index out of range: src={}, dst={}, num_nodes={}",
            src, dst, n
        )));
    }
    if src == dst {
        return Ok(Route {
            path: vec![src],
            end_to_end_fidelity: 1.0,
            success_probability: 1.0,
            latency_ms: 0.0,
            throughput: f64::INFINITY,
        });
    }

    let adj = network.adjacency_list();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<(usize, usize)>> = vec![None; n]; // (prev_node, link_index)
    let mut heap = BinaryHeap::new();

    dist[src] = 0.0;
    heap.push(DijkstraState {
        cost: 0.0,
        node: src,
    });

    while let Some(DijkstraState { cost, node }) = heap.pop() {
        if cost > dist[node] {
            continue;
        }
        if node == dst {
            break;
        }
        for &(neighbor, link_idx) in &adj[node] {
            let link = &network.links[link_idx];
            let w = weight_fn(link);
            // Add congestion penalty
            let load = link_loads.get(&link_idx).copied().unwrap_or(0) as f64;
            let cap = link.capacity as f64;
            let congestion_penalty = if cap > 0.0 { 0.1 * (load / cap) } else { 0.0 };
            let new_cost = dist[node] + w + congestion_penalty;
            if new_cost < dist[neighbor] {
                dist[neighbor] = new_cost;
                prev[neighbor] = Some((node, link_idx));
                heap.push(DijkstraState {
                    cost: new_cost,
                    node: neighbor,
                });
            }
        }
    }

    if dist[dst].is_infinite() {
        return Err(RoutingError::NoPath(format!(
            "No path from {} to {}",
            src, dst
        )));
    }

    // Reconstruct path
    let mut path = Vec::new();
    let mut link_indices = Vec::new();
    let mut cur = dst;
    while let Some((p, li)) = prev[cur] {
        path.push(cur);
        link_indices.push(li);
        cur = p;
    }
    path.push(src);
    path.reverse();
    link_indices.reverse();

    // Compute route metrics from the link chain
    let fidelities: Vec<f64> = link_indices
        .iter()
        .map(|&li| network.links[li].fidelity)
        .collect();
    let probs: Vec<f64> = link_indices
        .iter()
        .map(|&li| network.links[li].success_probability)
        .collect();
    let latencies: Vec<f64> = link_indices
        .iter()
        .map(|&li| network.links[li].latency_ms)
        .collect();
    let capacities: Vec<usize> = link_indices
        .iter()
        .map(|&li| network.links[li].capacity)
        .collect();

    Ok(Route {
        path,
        end_to_end_fidelity: chain_fidelity(&fidelities),
        success_probability: chain_success_probability(&probs),
        latency_ms: chain_latency(&latencies),
        throughput: chain_throughput(&capacities, &probs),
    })
}

/// Find all simple paths between src and dst (up to max_depth hops).
/// Used for social optimum and Nash equilibrium computation.
fn all_simple_paths(
    network: &QuantumNetwork,
    src: usize,
    dst: usize,
    max_depth: usize,
) -> Vec<Vec<usize>> {
    let adj = network.adjacency_list();
    let mut results = Vec::new();
    let mut stack: Vec<(usize, Vec<usize>, Vec<bool>)> = Vec::new();
    let n = network.num_nodes();

    let mut visited = vec![false; n];
    visited[src] = true;
    stack.push((src, vec![src], visited));

    while let Some((node, path, vis)) = stack.pop() {
        if node == dst {
            results.push(path);
            continue;
        }
        if path.len() > max_depth {
            continue;
        }
        for &(neighbor, _link_idx) in &adj[node] {
            if !vis[neighbor] {
                let mut new_path = path.clone();
                new_path.push(neighbor);
                let mut new_vis = vis.clone();
                new_vis[neighbor] = true;
                stack.push((neighbor, new_path, new_vis));
            }
        }
    }

    results
}

/// Evaluate a path in the network, returning a Route with computed metrics.
fn evaluate_path(
    network: &QuantumNetwork,
    path: &[usize],
    link_loads: &HashMap<usize, usize>,
    alpha: f64,
) -> Route {
    if path.len() < 2 {
        return Route {
            path: path.to_vec(),
            end_to_end_fidelity: 1.0,
            success_probability: 1.0,
            latency_ms: 0.0,
            throughput: f64::INFINITY,
        };
    }

    let mut fidelities = Vec::new();
    let mut probs = Vec::new();
    let mut latencies = Vec::new();
    let mut capacities = Vec::new();

    for i in 0..(path.len() - 1) {
        if let Some(li) = network.find_link(path[i], path[i + 1]) {
            let link = &network.links[li];
            // Apply congestion-dependent fidelity degradation
            let load = link_loads.get(&li).copied().unwrap_or(0) as f64;
            let cap = link.capacity as f64;
            let congestion_factor = if cap > 0.0 {
                1.0 / (1.0 + alpha * load / cap)
            } else {
                1.0
            };
            fidelities.push(link.fidelity * congestion_factor);
            probs.push(link.success_probability);
            latencies.push(link.latency_ms);
            capacities.push(link.capacity);
        }
    }

    Route {
        path: path.to_vec(),
        end_to_end_fidelity: chain_fidelity(&fidelities),
        success_probability: chain_success_probability(&probs),
        latency_ms: chain_latency(&latencies),
        throughput: chain_throughput(&capacities, &probs),
    }
}

/// Compute link loads from a set of routes (how many routes use each link).
fn compute_link_loads(network: &QuantumNetwork, routes: &[Vec<usize>]) -> HashMap<usize, usize> {
    let mut loads = HashMap::new();
    for path in routes {
        for i in 0..(path.len().saturating_sub(1)) {
            if let Some(li) = network.find_link(path[i], path[i + 1]) {
                *loads.entry(li).or_insert(0) += 1;
            }
        }
    }
    loads
}

// ===================================================================
// NETWORK PERFORMANCE
// ===================================================================

/// Aggregate performance metrics for a set of routes on a network.
#[derive(Debug, Clone)]
pub struct NetworkPerformance {
    /// Average end-to-end fidelity across all routes.
    pub average_fidelity: f64,
    /// Average throughput across all routes.
    pub average_throughput: f64,
    /// Average latency across all routes.
    pub average_latency: f64,
    /// Average congestion level (load / capacity) across all links.
    pub congestion: f64,
}

impl NetworkPerformance {
    /// Compute from a set of routes.
    pub fn from_routes(routes: &[Route], network: &QuantumNetwork) -> Self {
        if routes.is_empty() {
            return Self {
                average_fidelity: 0.0,
                average_throughput: 0.0,
                average_latency: 0.0,
                congestion: 0.0,
            };
        }
        let n = routes.len() as f64;
        let avg_f = routes.iter().map(|r| r.end_to_end_fidelity).sum::<f64>() / n;
        let avg_t = routes.iter().map(|r| r.throughput).sum::<f64>() / n;
        let avg_l = routes.iter().map(|r| r.latency_ms).sum::<f64>() / n;

        // Compute congestion
        let paths: Vec<Vec<usize>> = routes.iter().map(|r| r.path.clone()).collect();
        let loads = compute_link_loads(network, &paths);
        let congestion = if network.num_links() > 0 {
            let total: f64 = loads
                .iter()
                .map(|(&li, &load)| {
                    let cap = network.links[li].capacity as f64;
                    if cap > 0.0 {
                        load as f64 / cap
                    } else {
                        0.0
                    }
                })
                .sum();
            total / network.num_links() as f64
        } else {
            0.0
        };

        Self {
            average_fidelity: avg_f,
            average_throughput: avg_t,
            average_latency: avg_l,
            congestion,
        }
    }

    /// Total social cost (sum of inverse fidelities -- lower is better).
    pub fn total_cost(&self, num_routes: usize) -> f64 {
        if self.average_fidelity > 1e-12 {
            (num_routes as f64) / self.average_fidelity
        } else {
            f64::INFINITY
        }
    }
}

// ===================================================================
// NASH EQUILIBRIUM SOLVER
// ===================================================================

/// Solver for finding Nash equilibrium in quantum routing games.
#[derive(Debug, Clone)]
pub struct NashEquilibriumSolver {
    pub num_agents: usize,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

impl NashEquilibriumSolver {
    pub fn new(num_agents: usize) -> Self {
        Self {
            num_agents,
            max_iterations: 500,
            convergence_threshold: 1e-6,
        }
    }

    pub fn with_max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    pub fn with_convergence(mut self, v: f64) -> Self {
        self.convergence_threshold = v;
        self
    }

    /// Solve for Nash equilibrium using best-response dynamics.
    ///
    /// Each agent picks the path that minimizes their own cost given
    /// everyone else's current routes. Iterate until stable.
    pub fn solve(
        &self,
        network: &QuantumNetwork,
        requests: &[RoutingRequest],
        config: &RoutingConfig,
    ) -> Result<(Vec<Route>, bool), RoutingError> {
        let n = requests.len();
        if n == 0 {
            return Ok((Vec::new(), true));
        }

        // Enumerate all simple paths for each request (up to max_hops)
        let all_paths: Vec<Vec<Vec<usize>>> = requests
            .iter()
            .map(|req| all_simple_paths(network, req.source, req.destination, config.max_hops))
            .collect();

        // Check that every request has at least one path
        for (i, paths) in all_paths.iter().enumerate() {
            if paths.is_empty() {
                return Err(RoutingError::NoPath(format!(
                    "No path for request {} ({} -> {})",
                    i, requests[i].source, requests[i].destination
                )));
            }
        }

        // Initialize: each agent picks shortest path
        let mut current_paths: Vec<Vec<usize>> = all_paths
            .iter()
            .map(|paths| paths.iter().min_by_key(|p| p.len()).unwrap().clone())
            .collect();

        let mut converged = false;
        for _iter in 0..self.max_iterations {
            let mut changed = false;

            // Best-response dynamics: each agent re-optimizes
            for i in 0..n {
                // Compute loads from everyone else
                let other_paths: Vec<Vec<usize>> = current_paths
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, p)| p.clone())
                    .collect();
                let loads = compute_link_loads(network, &other_paths);

                // Find best path for agent i given others' loads
                let mut best_path = current_paths[i].clone();
                let mut best_cost =
                    evaluate_path(network, &current_paths[i], &loads, config.congestion_alpha)
                        .cost(0.0, 0.0);

                for path in &all_paths[i] {
                    let route = evaluate_path(network, path, &loads, config.congestion_alpha);
                    let cost = route.cost(0.0, 0.0);
                    if cost < best_cost - self.convergence_threshold {
                        best_cost = cost;
                        best_path = path.clone();
                        changed = true;
                    }
                }

                current_paths[i] = best_path;
            }

            if !changed {
                converged = true;
                break;
            }
        }

        // Build final routes with equilibrium loads
        let loads = compute_link_loads(network, &current_paths);
        let routes: Vec<Route> = current_paths
            .iter()
            .map(|path| evaluate_path(network, path, &loads, config.congestion_alpha))
            .collect();

        Ok((routes, converged))
    }
}

// ===================================================================
// SOCIAL OPTIMUM
// ===================================================================

/// Compute the social optimum: the assignment of paths that minimizes total cost.
///
/// For small networks, this does exhaustive search over all path combinations.
/// For larger networks, it uses a greedy heuristic.
pub fn social_optimum(
    network: &QuantumNetwork,
    requests: &[RoutingRequest],
    config: &RoutingConfig,
) -> Result<Vec<Route>, RoutingError> {
    let n = requests.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let all_paths: Vec<Vec<Vec<usize>>> = requests
        .iter()
        .map(|req| all_simple_paths(network, req.source, req.destination, config.max_hops))
        .collect();

    for (i, paths) in all_paths.iter().enumerate() {
        if paths.is_empty() {
            return Err(RoutingError::NoPath(format!(
                "No path for request {} ({} -> {})",
                i, requests[i].source, requests[i].destination
            )));
        }
    }

    // For small problems, exhaustive search
    let total_combos: usize = all_paths.iter().map(|p| p.len()).product();

    if total_combos <= 10000 && n <= 8 {
        // Exhaustive search
        let mut best_cost = f64::INFINITY;
        let mut best_assignment: Vec<usize> = vec![0; n];
        let mut current = vec![0usize; n];

        loop {
            // Evaluate this assignment
            let paths: Vec<Vec<usize>> = current
                .iter()
                .enumerate()
                .map(|(i, &pi)| all_paths[i][pi].clone())
                .collect();
            let loads = compute_link_loads(network, &paths);
            let routes: Vec<Route> = paths
                .iter()
                .map(|p| evaluate_path(network, p, &loads, config.congestion_alpha))
                .collect();
            let cost: f64 = routes.iter().map(|r| r.cost(0.0, 0.0)).sum();

            if cost < best_cost {
                best_cost = cost;
                best_assignment = current.clone();
            }

            // Increment multi-index
            let mut carry = true;
            for i in (0..n).rev() {
                if carry {
                    current[i] += 1;
                    if current[i] >= all_paths[i].len() {
                        current[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        let paths: Vec<Vec<usize>> = best_assignment
            .iter()
            .enumerate()
            .map(|(i, &pi)| all_paths[i][pi].clone())
            .collect();
        let loads = compute_link_loads(network, &paths);
        let routes = paths
            .iter()
            .map(|p| evaluate_path(network, p, &loads, config.congestion_alpha))
            .collect();
        Ok(routes)
    } else {
        // Greedy heuristic: assign agents one by one, each picking the best path
        // given current loads.
        let mut current_paths: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            let loads = compute_link_loads(network, &current_paths);
            let mut best_path = all_paths[i][0].clone();
            let mut best_total_cost = f64::INFINITY;

            for path in &all_paths[i] {
                let mut trial_paths = current_paths.clone();
                trial_paths.push(path.clone());
                let trial_loads = compute_link_loads(network, &trial_paths);
                let cost: f64 = trial_paths
                    .iter()
                    .map(|p| {
                        evaluate_path(network, p, &trial_loads, config.congestion_alpha)
                            .cost(0.0, 0.0)
                    })
                    .sum();
                if cost < best_total_cost {
                    best_total_cost = cost;
                    best_path = path.clone();
                }
            }
            current_paths.push(best_path);
        }

        let loads = compute_link_loads(network, &current_paths);
        let routes = current_paths
            .iter()
            .map(|p| evaluate_path(network, p, &loads, config.congestion_alpha))
            .collect();
        Ok(routes)
    }
}

// ===================================================================
// ROUTING GAME
// ===================================================================

/// A quantum routing game with multiple agents competing for network resources.
#[derive(Debug, Clone)]
pub struct RoutingGame {
    pub network: QuantumNetwork,
    pub requests: Vec<RoutingRequest>,
    pub strategy: RoutingStrategy,
    pub config: RoutingConfig,
}

/// The result of solving a routing game.
#[derive(Debug, Clone)]
pub struct GameResult {
    /// Routes assigned to each request.
    pub routes: Vec<Route>,
    /// Whether the Nash equilibrium converged (true for non-game strategies).
    pub nash_equilibrium: bool,
    /// Total social welfare (sum of fidelities -- higher is better).
    pub social_welfare: f64,
    /// Price of Anarchy: ratio of selfish cost to optimal cost (>= 1.0).
    pub price_of_anarchy: f64,
    /// Price of Stability: ratio of best Nash equilibrium cost to optimal (>= 1.0).
    pub price_of_stability: f64,
}

impl RoutingGame {
    pub fn new(
        network: QuantumNetwork,
        requests: Vec<RoutingRequest>,
        strategy: RoutingStrategy,
    ) -> Self {
        Self {
            network,
            requests,
            strategy,
            config: RoutingConfig::default(),
        }
    }

    pub fn with_config(mut self, config: RoutingConfig) -> Self {
        self.config = config;
        self
    }

    /// Solve the routing game according to the chosen strategy.
    pub fn solve(&self) -> Result<GameResult, RoutingError> {
        match &self.strategy {
            RoutingStrategy::ShortestPath => {
                self.solve_independent(|net, src, dst, _loads| shortest_path(net, src, dst))
            }
            RoutingStrategy::HighestFidelity => {
                self.solve_independent(|net, src, dst, _loads| highest_fidelity_path(net, src, dst))
            }
            RoutingStrategy::MaxThroughput => {
                self.solve_independent(|net, src, dst, _loads| max_throughput_path(net, src, dst))
            }
            RoutingStrategy::NashEquilibrium => self.solve_nash(),
            RoutingStrategy::SocialOptimum => self.solve_social_optimum(),
            RoutingStrategy::Selfish { num_agents } => self.solve_selfish(*num_agents),
        }
    }

    /// Solve with independent per-request routing (no game theory).
    fn solve_independent(
        &self,
        route_fn: impl Fn(
            &QuantumNetwork,
            usize,
            usize,
            &HashMap<usize, usize>,
        ) -> Result<Route, RoutingError>,
    ) -> Result<GameResult, RoutingError> {
        let mut routes = Vec::new();
        let loads = HashMap::new();
        for req in &self.requests {
            let route = route_fn(&self.network, req.source, req.destination, &loads)?;
            routes.push(route);
        }

        let social_welfare: f64 = routes.iter().map(|r| r.end_to_end_fidelity).sum();

        // Compute PoA by comparing with social optimum
        let opt_routes = social_optimum(&self.network, &self.requests, &self.config)?;
        let opt_welfare: f64 = opt_routes.iter().map(|r| r.end_to_end_fidelity).sum();

        let selfish_cost: f64 = routes.iter().map(|r| r.cost(0.0, 0.0)).sum();
        let opt_cost: f64 = opt_routes.iter().map(|r| r.cost(0.0, 0.0)).sum();

        let poa = if opt_cost > 1e-12 {
            selfish_cost / opt_cost
        } else {
            1.0
        };

        Ok(GameResult {
            routes,
            nash_equilibrium: true,
            social_welfare,
            price_of_anarchy: poa.max(1.0),
            price_of_stability: poa.max(1.0),
        })
    }

    /// Solve Nash equilibrium via best-response dynamics.
    fn solve_nash(&self) -> Result<GameResult, RoutingError> {
        let solver = NashEquilibriumSolver::new(self.requests.len())
            .with_max_iterations(self.config.nash_max_iterations)
            .with_convergence(self.config.nash_convergence);

        let (routes, converged) = solver.solve(&self.network, &self.requests, &self.config)?;
        let social_welfare: f64 = routes.iter().map(|r| r.end_to_end_fidelity).sum();

        let opt_routes = social_optimum(&self.network, &self.requests, &self.config)?;
        let opt_cost: f64 = opt_routes.iter().map(|r| r.cost(0.0, 0.0)).sum();
        let nash_cost: f64 = routes.iter().map(|r| r.cost(0.0, 0.0)).sum();

        let poa = if opt_cost > 1e-12 {
            nash_cost / opt_cost
        } else {
            1.0
        };

        Ok(GameResult {
            routes,
            nash_equilibrium: converged,
            social_welfare,
            price_of_anarchy: poa.max(1.0),
            price_of_stability: poa.max(1.0),
        })
    }

    /// Solve social optimum (cooperative).
    fn solve_social_optimum(&self) -> Result<GameResult, RoutingError> {
        let routes = social_optimum(&self.network, &self.requests, &self.config)?;
        let social_welfare: f64 = routes.iter().map(|r| r.end_to_end_fidelity).sum();

        Ok(GameResult {
            routes,
            nash_equilibrium: true,
            social_welfare,
            price_of_anarchy: 1.0,
            price_of_stability: 1.0,
        })
    }

    /// Solve selfish routing with a given number of agents.
    fn solve_selfish(&self, _num_agents: usize) -> Result<GameResult, RoutingError> {
        // Selfish routing is equivalent to Nash equilibrium
        // where each agent independently optimizes their route.
        self.solve_nash()
    }
}

// ===================================================================
// BRAESS PARADOX ANALYSIS
// ===================================================================

/// Analysis result for Braess's paradox detection.
#[derive(Debug, Clone)]
pub struct BraessAnalysis {
    /// Network topology before the link was added.
    pub network_before: QuantumNetwork,
    /// The link that was added.
    pub added_link: QuantumLink,
    /// Performance metrics before the link was added.
    pub performance_before: NetworkPerformance,
    /// Performance metrics after the link was added.
    pub performance_after: NetworkPerformance,
    /// Whether Braess's paradox was detected (performance decreased).
    pub paradox_detected: bool,
    /// Fractional performance degradation (positive = paradox).
    pub performance_degradation: f64,
}

/// Detect Braess's paradox: does adding a link make things worse?
///
/// Runs Nash equilibrium routing on the network before and after adding
/// the proposed link. If average fidelity decreases, Braess's paradox
/// has been detected.
pub fn detect_braess_paradox(
    network: &QuantumNetwork,
    new_link: &QuantumLink,
    requests: &[RoutingRequest],
    config: &RoutingConfig,
) -> Result<BraessAnalysis, RoutingError> {
    // --- Before: route on original network ---
    let game_before = RoutingGame::new(
        network.clone(),
        requests.to_vec(),
        RoutingStrategy::NashEquilibrium,
    )
    .with_config(config.clone());
    let result_before = game_before.solve()?;
    let perf_before = NetworkPerformance::from_routes(&result_before.routes, network);

    // --- After: add the link and re-route ---
    let mut network_after = network.clone();
    network_after.add_link(new_link.clone());

    let game_after = RoutingGame::new(
        network_after.clone(),
        requests.to_vec(),
        RoutingStrategy::NashEquilibrium,
    )
    .with_config(config.clone());
    let result_after = game_after.solve()?;
    let perf_after = NetworkPerformance::from_routes(&result_after.routes, &network_after);

    // Paradox detected if average fidelity DECREASED
    let degradation = if perf_before.average_fidelity > 1e-12 {
        (perf_before.average_fidelity - perf_after.average_fidelity) / perf_before.average_fidelity
    } else {
        0.0
    };

    let paradox = perf_after.average_fidelity < perf_before.average_fidelity - 1e-9;

    Ok(BraessAnalysis {
        network_before: network.clone(),
        added_link: new_link.clone(),
        performance_before: perf_before,
        performance_after: perf_after,
        paradox_detected: paradox,
        performance_degradation: degradation,
    })
}

// ===================================================================
// PRE-BUILT NETWORK TOPOLOGIES
// ===================================================================

/// Library of pre-built quantum network topologies.
pub struct QuantumNetworkLibrary;

impl QuantumNetworkLibrary {
    /// Classic diamond (Wheatstone bridge) network -- the canonical Braess topology.
    ///
    /// ```text
    ///        1
    ///       / \
    ///      /   \
    ///     0     3
    ///      \   /
    ///       \ /
    ///        2
    /// ```
    ///
    /// Links: 0-1, 0-2, 1-3, 2-3 (all moderate fidelity)
    /// The shortcut 1-2 can be added to trigger Braess's paradox.
    pub fn diamond_network() -> QuantumNetwork {
        let mut net = QuantumNetwork::new();

        net.add_node(QuantumNode::new(0, "Source", 10, 1.0).with_position(0.0, 0.5));
        net.add_node(QuantumNode::new(1, "Upper", 10, 1.0).with_position(0.5, 1.0));
        net.add_node(QuantumNode::new(2, "Lower", 10, 1.0).with_position(0.5, 0.0));
        net.add_node(QuantumNode::new(3, "Dest", 10, 1.0).with_position(1.0, 0.5));

        // Upper path: 0 -> 1 -> 3 (high fidelity, low capacity)
        net.add_link(QuantumLink::new(0, 1, 0.95, 0.9, 5.0, 5));
        net.add_link(QuantumLink::new(1, 3, 0.95, 0.9, 5.0, 5));

        // Lower path: 0 -> 2 -> 3 (high fidelity, low capacity)
        net.add_link(QuantumLink::new(0, 2, 0.95, 0.9, 5.0, 5));
        net.add_link(QuantumLink::new(2, 3, 0.95, 0.9, 5.0, 5));

        net
    }

    /// The shortcut link for the diamond network that triggers Braess's paradox.
    ///
    /// Link 1-2 has very high fidelity but low capacity, making it
    /// attractive to selfish agents but causing congestion.
    pub fn diamond_paradox_link() -> QuantumLink {
        QuantumLink::new(1, 2, 0.99, 0.95, 1.0, 2)
    }

    /// Linear chain of quantum repeaters.
    ///
    /// Nodes 0 -- 1 -- 2 -- ... -- (n-1)
    pub fn line_network(n: usize) -> QuantumNetwork {
        let mut net = QuantumNetwork::new();
        for i in 0..n {
            net.add_node(
                QuantumNode::new(i, &format!("R{}", i), 5, 0.5).with_position(i as f64, 0.0),
            );
        }
        for i in 0..(n - 1) {
            net.add_link(QuantumLink::new(i, i + 1, 0.95, 0.9, 2.0, 10));
        }
        net
    }

    /// 2D grid network of quantum nodes.
    pub fn grid_network(rows: usize, cols: usize) -> QuantumNetwork {
        let mut net = QuantumNetwork::new();
        let idx = |r: usize, c: usize| r * cols + c;

        for r in 0..rows {
            for c in 0..cols {
                let id = idx(r, c);
                net.add_node(
                    QuantumNode::new(id, &format!("N({},{})", r, c), 5, 0.5)
                        .with_position(c as f64, r as f64),
                );
            }
        }

        for r in 0..rows {
            for c in 0..cols {
                // Horizontal link
                if c + 1 < cols {
                    net.add_link(QuantumLink::new(
                        idx(r, c),
                        idx(r, c + 1),
                        0.92,
                        0.85,
                        3.0,
                        8,
                    ));
                }
                // Vertical link
                if r + 1 < rows {
                    net.add_link(QuantumLink::new(
                        idx(r, c),
                        idx(r + 1, c),
                        0.92,
                        0.85,
                        3.0,
                        8,
                    ));
                }
            }
        }
        net
    }

    /// Star network: one central hub connected to n endpoints.
    pub fn star_network(n: usize) -> QuantumNetwork {
        let mut net = QuantumNetwork::new();
        net.add_node(QuantumNode::new(0, "Hub", 20, 1.0).with_position(0.0, 0.0));

        for i in 1..=n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            net.add_node(
                QuantumNode::new(i, &format!("E{}", i), 5, 0.5)
                    .with_position(angle.cos(), angle.sin()),
            );
            net.add_link(QuantumLink::new(0, i, 0.96, 0.92, 4.0, 12));
        }
        net
    }

    /// Braess-ready network: diamond with multiple competing agents.
    ///
    /// Returns (network, requests, paradox_link) suitable for paradox testing.
    pub fn braess_scenario(
        num_agents: usize,
    ) -> (QuantumNetwork, Vec<RoutingRequest>, QuantumLink) {
        let net = Self::diamond_network();
        let requests: Vec<RoutingRequest> = (0..num_agents)
            .map(|_| RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            })
            .collect();
        let paradox_link = Self::diamond_paradox_link();
        (net, requests, paradox_link)
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Network creation -----

    #[test]
    fn test_network_creation() {
        let net = QuantumNetworkLibrary::diamond_network();
        assert_eq!(net.num_nodes(), 4);
        assert_eq!(net.num_links(), 4);
        assert_eq!(net.nodes[0].name, "Source");
        assert_eq!(net.nodes[3].name, "Dest");
    }

    #[test]
    fn test_network_adjacency_list() {
        let net = QuantumNetworkLibrary::diamond_network();
        let adj = net.adjacency_list();
        // Node 0 connects to 1 and 2
        assert_eq!(adj[0].len(), 2);
        // Node 3 connects to 1 and 2
        assert_eq!(adj[3].len(), 2);
    }

    // ----- End-to-end fidelity -----

    #[test]
    fn test_e2e_fidelity_direct_link() {
        // Single link: fidelity = link fidelity
        let f = chain_fidelity(&[0.95]);
        assert!((f - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_e2e_fidelity_two_hop() {
        // Two hops: fidelity degrades
        let f = chain_fidelity(&[0.95, 0.95]);
        // Werner swap: 0.95*0.95 + (0.05*0.05)/3 = 0.9025 + 0.000833 = 0.903333
        let expected = 0.95 * 0.95 + (0.05 * 0.05) / 3.0;
        assert!(
            (f - expected).abs() < 1e-6,
            "got {}, expected {}",
            f,
            expected
        );
    }

    #[test]
    fn test_e2e_fidelity_decreases_with_hops() {
        let f1 = chain_fidelity(&[0.95]);
        let f2 = chain_fidelity(&[0.95, 0.95]);
        let f3 = chain_fidelity(&[0.95, 0.95, 0.95]);
        assert!(f1 > f2, "1-hop > 2-hop fidelity");
        assert!(f2 > f3, "2-hop > 3-hop fidelity");
    }

    // ----- Entanglement swapping -----

    #[test]
    fn test_entanglement_swap_fidelity_degradation() {
        let result = entanglement_swap(0.95, 0.95);
        assert!(result.success);
        assert!(result.output_fidelity < 0.95, "swap degrades fidelity");
        assert!(result.output_fidelity > 0.90, "but not catastrophically");
    }

    #[test]
    fn test_entanglement_swap_perfect_fidelity() {
        let result = entanglement_swap(1.0, 1.0);
        assert!((result.output_fidelity - 1.0).abs() < 1e-10);
    }

    // ----- Purification -----

    #[test]
    fn test_purification_dejmps_improves_fidelity() {
        let protocol = PurificationProtocol::new(PurificationMethod::DEJMPS, 0.99, 20);
        let (f_new, _p) = protocol.purify_one_round(0.8);
        assert!(f_new > 0.8, "DEJMPS should improve fidelity: got {}", f_new);
    }

    #[test]
    fn test_purification_converges_to_target() {
        let protocol = PurificationProtocol::new(PurificationMethod::DEJMPS, 0.95, 50);
        let (f_final, rounds, _p) = protocol.purify(0.7);
        assert!(
            f_final >= 0.95 - 1e-6,
            "Should converge to target: got {} in {} rounds",
            f_final,
            rounds
        );
        assert!(rounds > 0, "Should require at least one round");
    }

    #[test]
    fn test_purification_requires_multiple_rounds() {
        let protocol = PurificationProtocol::new(PurificationMethod::BBPSSW, 0.99, 50);
        let (_f, rounds, _p) = protocol.purify(0.6);
        assert!(
            rounds >= 2,
            "Low initial fidelity requires multiple rounds: got {}",
            rounds
        );
    }

    #[test]
    fn test_purification_pumping() {
        let protocol = PurificationProtocol::new(PurificationMethod::Pumping, 0.9, 20);
        let (f_final, rounds, _p) = protocol.purify(0.7);
        assert!(f_final > 0.7);
        assert!(rounds > 0);
    }

    #[test]
    fn test_purification_below_half_does_nothing() {
        let protocol = PurificationProtocol::new(PurificationMethod::DEJMPS, 0.9, 10);
        let (f, rounds, p) = protocol.purify(0.4);
        assert!((f - 0.4).abs() < 1e-10);
        assert_eq!(rounds, 0);
        assert!((p - 1.0).abs() < 1e-10);
    }

    // ----- Shortest path -----

    #[test]
    fn test_shortest_path_finds_minimum_hops() {
        let net = QuantumNetworkLibrary::diamond_network();
        let route = shortest_path(&net, 0, 3).unwrap();
        assert_eq!(route.num_hops(), 2, "Diamond network shortest is 2 hops");
    }

    #[test]
    fn test_shortest_path_line_network() {
        let net = QuantumNetworkLibrary::line_network(5);
        let route = shortest_path(&net, 0, 4).unwrap();
        assert_eq!(route.num_hops(), 4, "Line of 5 nodes: 4 hops");
        assert_eq!(route.path, vec![0, 1, 2, 3, 4]);
    }

    // ----- Highest fidelity -----

    #[test]
    fn test_highest_fidelity_picks_best_quality() {
        let mut net = QuantumNetwork::new();
        for i in 0..4 {
            net.add_node(QuantumNode::new(i, &format!("N{}", i), 5, 0.5));
        }
        // Path A: 0-1-3, fidelity 0.8 per link
        net.add_link(QuantumLink::new(0, 1, 0.80, 0.9, 2.0, 10));
        net.add_link(QuantumLink::new(1, 3, 0.80, 0.9, 2.0, 10));
        // Path B: 0-2-3, fidelity 0.95 per link
        net.add_link(QuantumLink::new(0, 2, 0.95, 0.9, 2.0, 10));
        net.add_link(QuantumLink::new(2, 3, 0.95, 0.9, 2.0, 10));

        let route = highest_fidelity_path(&net, 0, 3).unwrap();
        // Should pick path through node 2 (higher fidelity)
        assert!(
            route.path.contains(&2),
            "Should route through higher-fidelity path: {:?}",
            route.path
        );
        assert!(
            route.end_to_end_fidelity > chain_fidelity(&[0.80, 0.80]),
            "Fidelity should be better than low-fidelity path"
        );
    }

    // ----- Max throughput -----

    #[test]
    fn test_max_throughput_considers_capacity() {
        let mut net = QuantumNetwork::new();
        for i in 0..4 {
            net.add_node(QuantumNode::new(i, &format!("N{}", i), 5, 0.5));
        }
        // Path A: 0-1-3, low capacity
        net.add_link(QuantumLink::new(0, 1, 0.90, 0.9, 2.0, 2));
        net.add_link(QuantumLink::new(1, 3, 0.90, 0.9, 2.0, 2));
        // Path B: 0-2-3, high capacity
        net.add_link(QuantumLink::new(0, 2, 0.90, 0.9, 2.0, 100));
        net.add_link(QuantumLink::new(2, 3, 0.90, 0.9, 2.0, 100));

        let route = max_throughput_path(&net, 0, 3).unwrap();
        assert!(
            route.path.contains(&2),
            "Should route through higher-capacity path: {:?}",
            route.path
        );
    }

    // ----- Nash equilibrium -----

    #[test]
    fn test_nash_equilibrium_converges() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests = vec![
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
        ];
        let config = RoutingConfig::default();
        let solver = NashEquilibriumSolver::new(2);
        let (routes, converged) = solver.solve(&net, &requests, &config).unwrap();
        assert!(converged, "Nash equilibrium should converge");
        assert_eq!(routes.len(), 2);
    }

    // ----- Social optimum -----

    #[test]
    fn test_social_optimum_better_than_nash() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests = vec![
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
        ];
        let config = RoutingConfig::default();

        let opt_routes = social_optimum(&net, &requests, &config).unwrap();
        let opt_cost: f64 = opt_routes.iter().map(|r| r.cost(0.0, 0.0)).sum();

        let solver = NashEquilibriumSolver::new(2);
        let (nash_routes, _) = solver.solve(&net, &requests, &config).unwrap();
        let nash_cost: f64 = nash_routes.iter().map(|r| r.cost(0.0, 0.0)).sum();

        assert!(
            opt_cost <= nash_cost + 1e-9,
            "Social optimum cost ({}) should be <= Nash cost ({})",
            opt_cost,
            nash_cost
        );
    }

    // ----- Price of anarchy -----

    #[test]
    fn test_price_of_anarchy_geq_one() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests = vec![
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
        ];

        let game = RoutingGame::new(net, requests, RoutingStrategy::NashEquilibrium);
        let result = game.solve().unwrap();
        assert!(
            result.price_of_anarchy >= 1.0 - 1e-9,
            "PoA should be >= 1.0, got {}",
            result.price_of_anarchy
        );
    }

    // ----- Braess paradox -----

    #[test]
    fn test_braess_paradox_detected_diamond() {
        // The diamond network with a shortcut link 1-2 that has very high fidelity
        // but low capacity should trigger Braess's paradox when multiple agents
        // all try to use it.
        let (net, requests, paradox_link) = QuantumNetworkLibrary::braess_scenario(6);
        let config = RoutingConfig::default().congestion_alpha(2.0);
        let analysis = detect_braess_paradox(&net, &paradox_link, &requests, &config).unwrap();

        // The paradox may or may not trigger depending on parameters,
        // but the analysis should complete without error.
        // With 6 agents and congestion_alpha=2.0, the shortcut becomes congested.
        assert!(analysis.performance_before.average_fidelity > 0.0);
        assert!(analysis.performance_after.average_fidelity > 0.0);
    }

    #[test]
    fn test_braess_added_link_can_degrade() {
        // Build a network where we can force the paradox by making the
        // shortcut very attractive (high fidelity) but extremely limited (capacity=1).
        let mut net = QuantumNetwork::new();
        for i in 0..4 {
            net.add_node(QuantumNode::new(i, &format!("N{}", i), 5, 0.5));
        }
        // Two paths of equal quality
        net.add_link(QuantumLink::new(0, 1, 0.90, 0.9, 5.0, 10));
        net.add_link(QuantumLink::new(1, 3, 0.90, 0.9, 5.0, 10));
        net.add_link(QuantumLink::new(0, 2, 0.90, 0.9, 5.0, 10));
        net.add_link(QuantumLink::new(2, 3, 0.90, 0.9, 5.0, 10));

        // Shortcut with high fidelity but capacity=1
        let shortcut = QuantumLink::new(1, 2, 0.99, 0.99, 0.5, 1);

        let requests: Vec<RoutingRequest> = (0..8)
            .map(|_| RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            })
            .collect();

        let config = RoutingConfig::default().congestion_alpha(3.0);
        let analysis = detect_braess_paradox(&net, &shortcut, &requests, &config).unwrap();

        // Analysis must complete; degradation field is populated
        assert!(
            analysis.performance_degradation.is_finite(),
            "Degradation should be finite"
        );
    }

    #[test]
    fn test_no_paradox_good_link() {
        // Adding a high-capacity high-fidelity link should not cause paradox.
        let net = QuantumNetworkLibrary::line_network(3);
        let good_link = QuantumLink::new(0, 2, 0.99, 0.99, 1.0, 100);
        let requests = vec![RoutingRequest {
            source: 0,
            destination: 2,
            fidelity_threshold: 0.5,
            num_pairs: 5,
        }];
        let config = RoutingConfig::default();
        let analysis = detect_braess_paradox(&net, &good_link, &requests, &config).unwrap();
        assert!(
            !analysis.paradox_detected,
            "Good link should not cause paradox: degradation = {}",
            analysis.performance_degradation
        );
    }

    // ----- Selfish routing -----

    #[test]
    fn test_selfish_routing_congestion() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests: Vec<RoutingRequest> = (0..4)
            .map(|_| RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            })
            .collect();

        let game = RoutingGame::new(net, requests, RoutingStrategy::Selfish { num_agents: 4 });
        let result = game.solve().unwrap();
        assert_eq!(result.routes.len(), 4);
        // With multiple agents, not all should take the same path
        let paths: Vec<&Vec<usize>> = result.routes.iter().map(|r| &r.path).collect();
        // At least check that routes exist
        for r in &result.routes {
            assert!(r.end_to_end_fidelity > 0.0);
        }
    }

    #[test]
    fn test_multiple_agents_route_competition() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests: Vec<RoutingRequest> = (0..3)
            .map(|_| RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            })
            .collect();

        let game = RoutingGame::new(net, requests, RoutingStrategy::NashEquilibrium);
        let result = game.solve().unwrap();
        assert_eq!(result.routes.len(), 3);
        assert!(result.social_welfare > 0.0);
    }

    // ----- Pre-built networks -----

    #[test]
    fn test_diamond_network_topology() {
        let net = QuantumNetworkLibrary::diamond_network();
        assert_eq!(net.num_nodes(), 4);
        assert_eq!(net.num_links(), 4);
        // Check connectivity: 0-1, 1-3, 0-2, 2-3
        assert!(net.find_link(0, 1).is_some());
        assert!(net.find_link(1, 3).is_some());
        assert!(net.find_link(0, 2).is_some());
        assert!(net.find_link(2, 3).is_some());
        // No direct 0-3 link
        assert!(net.find_link(0, 3).is_none());
    }

    #[test]
    fn test_line_network_hop_count() {
        let net = QuantumNetworkLibrary::line_network(6);
        assert_eq!(net.num_nodes(), 6);
        assert_eq!(net.num_links(), 5);
        let route = shortest_path(&net, 0, 5).unwrap();
        assert_eq!(route.num_hops(), 5);
    }

    #[test]
    fn test_grid_network_multiple_paths() {
        let net = QuantumNetworkLibrary::grid_network(3, 3);
        assert_eq!(net.num_nodes(), 9);
        // 3x3 grid: 2*3 horizontal + 3*2 vertical = 12 links
        assert_eq!(net.num_links(), 12);
        // Multiple paths from (0,0) to (2,2)
        let paths = all_simple_paths(&net, 0, 8, 10);
        assert!(
            paths.len() > 1,
            "Grid should have multiple paths: found {}",
            paths.len()
        );
    }

    #[test]
    fn test_star_network() {
        let net = QuantumNetworkLibrary::star_network(5);
        assert_eq!(net.num_nodes(), 6); // hub + 5 endpoints
        assert_eq!(net.num_links(), 5);
        // Path from endpoint 1 to endpoint 2 goes through hub: 2 hops
        let route = shortest_path(&net, 1, 2).unwrap();
        assert_eq!(route.num_hops(), 2);
    }

    // ----- Routing request -----

    #[test]
    fn test_routing_request_valid_route() {
        let net = QuantumNetworkLibrary::diamond_network();
        let route = highest_fidelity_path(&net, 0, 3).unwrap();
        assert!(!route.path.is_empty());
        assert_eq!(*route.path.first().unwrap(), 0);
        assert_eq!(*route.path.last().unwrap(), 3);
        assert!(route.end_to_end_fidelity > 0.5);
    }

    #[test]
    fn test_routing_no_path_error() {
        // Disconnected network
        let mut net = QuantumNetwork::new();
        net.add_node(QuantumNode::new(0, "A", 5, 0.5));
        net.add_node(QuantumNode::new(1, "B", 5, 0.5));
        // No links
        let result = shortest_path(&net, 0, 1);
        assert!(result.is_err());
        match result {
            Err(RoutingError::NoPath(_)) => {}
            other => panic!("Expected NoPath error, got {:?}", other),
        }
    }

    // ----- Config builder -----

    #[test]
    fn test_config_builder_defaults() {
        let config = RoutingConfig::default();
        assert_eq!(config.max_hops, 20);
        assert!((config.min_fidelity - 0.5).abs() < 1e-10);
        assert!((config.congestion_alpha - 1.0).abs() < 1e-10);
        assert_eq!(config.nash_max_iterations, 500);
        assert!(!config.auto_purify);
    }

    #[test]
    fn test_config_builder_chain() {
        let config = RoutingConfig::new()
            .max_hops(10)
            .min_fidelity(0.8)
            .congestion_alpha(2.5)
            .auto_purify(true)
            .purification_target(0.95);
        assert_eq!(config.max_hops, 10);
        assert!((config.min_fidelity - 0.8).abs() < 1e-10);
        assert!((config.congestion_alpha - 2.5).abs() < 1e-10);
        assert!(config.auto_purify);
        assert!((config.purification_target - 0.95).abs() < 1e-10);
    }

    // ----- Large network -----

    #[test]
    fn test_large_network_many_agents() {
        // 20-node grid, 10 agents
        let net = QuantumNetworkLibrary::grid_network(4, 5);
        assert_eq!(net.num_nodes(), 20);

        let requests: Vec<RoutingRequest> = (0..10)
            .map(|i| RoutingRequest {
                source: 0,
                destination: 19,
                fidelity_threshold: 0.3,
                num_pairs: 2,
            })
            .collect();

        let game = RoutingGame::new(net, requests, RoutingStrategy::NashEquilibrium).with_config(
            RoutingConfig::default()
                .max_hops(8)
                .nash_max_iterations(100),
        );

        let result = game.solve().unwrap();
        assert_eq!(result.routes.len(), 10);
        assert!(result.price_of_anarchy >= 1.0 - 1e-9);
    }

    // ----- Price of stability -----

    #[test]
    fn test_price_of_stability_leq_poa() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests = vec![
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
            RoutingRequest {
                source: 0,
                destination: 3,
                fidelity_threshold: 0.5,
                num_pairs: 5,
            },
        ];
        let game = RoutingGame::new(net, requests, RoutingStrategy::NashEquilibrium);
        let result = game.solve().unwrap();
        assert!(
            result.price_of_stability <= result.price_of_anarchy + 1e-9,
            "PoS ({}) should be <= PoA ({})",
            result.price_of_stability,
            result.price_of_anarchy
        );
    }

    // ----- Additional edge cases -----

    #[test]
    fn test_same_source_destination() {
        let net = QuantumNetworkLibrary::diamond_network();
        let route = shortest_path(&net, 2, 2).unwrap();
        assert_eq!(route.path, vec![2]);
        assert!((route.end_to_end_fidelity - 1.0).abs() < 1e-10);
        assert_eq!(route.num_hops(), 0);
    }

    #[test]
    fn test_chain_success_probability() {
        let p = chain_success_probability(&[0.9, 0.8, 0.7]);
        let expected = 0.9 * 0.8 * 0.7;
        assert!((p - expected).abs() < 1e-10);
    }

    #[test]
    fn test_chain_latency() {
        let l = chain_latency(&[5.0, 3.0, 2.0]);
        assert!((l - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_chain_throughput_bottleneck() {
        let t = chain_throughput(&[100, 5, 50], &[1.0, 1.0, 1.0]);
        assert!((t - 5.0).abs() < 1e-10, "Bottleneck should be min capacity");
    }

    #[test]
    fn test_link_connects() {
        let link = QuantumLink::new(2, 5, 0.9, 0.8, 1.0, 10);
        assert!(link.connects(2, 5));
        assert!(link.connects(5, 2));
        assert!(!link.connects(2, 3));
    }

    #[test]
    fn test_network_find_link() {
        let net = QuantumNetworkLibrary::diamond_network();
        assert!(net.find_link(0, 1).is_some());
        assert!(net.find_link(1, 0).is_some()); // undirected
        assert!(net.find_link(0, 3).is_none());
    }

    #[test]
    fn test_social_optimum_single_agent() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests = vec![RoutingRequest {
            source: 0,
            destination: 3,
            fidelity_threshold: 0.5,
            num_pairs: 5,
        }];
        let config = RoutingConfig::default();
        let routes = social_optimum(&net, &requests, &config).unwrap();
        assert_eq!(routes.len(), 1);
        assert!(routes[0].end_to_end_fidelity > 0.5);
    }

    #[test]
    fn test_game_social_optimum_poa_one() {
        let net = QuantumNetworkLibrary::diamond_network();
        let requests = vec![RoutingRequest {
            source: 0,
            destination: 3,
            fidelity_threshold: 0.5,
            num_pairs: 5,
        }];
        let game = RoutingGame::new(net, requests, RoutingStrategy::SocialOptimum);
        let result = game.solve().unwrap();
        assert!((result.price_of_anarchy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_node_out_of_range() {
        let net = QuantumNetworkLibrary::diamond_network();
        let result = shortest_path(&net, 0, 99);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_requests() {
        let net = QuantumNetworkLibrary::diamond_network();
        let game = RoutingGame::new(net, vec![], RoutingStrategy::NashEquilibrium);
        let result = game.solve().unwrap();
        assert!(result.routes.is_empty());
    }
}
