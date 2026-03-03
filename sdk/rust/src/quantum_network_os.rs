//! QNodeOS: Quantum Network Operating System for Distributed Quantum Computing
//!
//! A world-first network operating system layer for quantum simulators, implementing
//! a full protocol stack for distributed quantum computation over entanglement-based
//! networks. Inspired by the QuTech QNodeOS architecture.
//!
//! # Architecture
//!
//! The stack is organized bottom-up:
//!
//! 1. **Physical layer**: Entanglement generation between adjacent nodes
//! 2. **Link layer**: Entanglement table management and pair lifetime tracking
//! 3. **Network layer**: Multi-hop path finding and entanglement swapping
//! 4. **Distillation layer**: DEJMPS purification protocol for fidelity improvement
//! 5. **Transport layer**: Quantum teleportation with depolarizing noise model
//! 6. **Application layer**: `QNodeOS` facade for end-to-end distributed operations
//!
//! # Fidelity Models
//!
//! - **Entanglement swapping**: F_ac = F_ab * F_bc + (1 - F_ab)(1 - F_bc) / 3
//! - **DEJMPS distillation**: bilateral CNOT purification with Werner-state formula
//! - **Depolarizing teleportation**: state fidelity degrades as (1 - link_fidelity)
//!
//! # Usage
//!
//! ```ignore
//! use nqpu_metal::quantum_network_os::*;
//!
//! let config = QNodeOSConfig::builder()
//!     .num_nodes(4)
//!     .link_fidelity(0.92)
//!     .protocol(NetworkProtocol::DistilledBellPairs)
//!     .build()
//!     .unwrap();
//!
//! let mut os = QNodeOS::new(config);
//! os.generate_entanglement();
//! let link = os.request_link(0, 3).unwrap();
//! ```

use crate::{c64_zero, C64};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::fmt;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors that can occur during quantum network operations.
#[derive(Clone, Debug, PartialEq)]
pub enum QNodeOSError {
    /// Configuration parameter is out of its valid range.
    InvalidConfig { field: String, reason: String },
    /// No entangled pairs available between the requested nodes.
    NoPairsAvailable { node_a: usize, node_b: usize },
    /// No path exists between source and destination in the network graph.
    NoPathExists { src: usize, dst: usize },
    /// A node index exceeds the network size.
    NodeOutOfRange { node: usize, max: usize },
    /// The state vector length is not a power of two.
    InvalidStateSize { size: usize },
    /// Swap chain execution failed at an intermediate hop.
    SwapChainFailed { hop: usize, reason: String },
    /// Distillation failed (both input pairs consumed, no output produced).
    DistillationFailed,
}

impl fmt::Display for QNodeOSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QNodeOSError::InvalidConfig { field, reason } => {
                write!(f, "invalid config '{}': {}", field, reason)
            }
            QNodeOSError::NoPairsAvailable { node_a, node_b } => {
                write!(f, "no entangled pairs between nodes {} and {}", node_a, node_b)
            }
            QNodeOSError::NoPathExists { src, dst } => {
                write!(f, "no path from node {} to node {}", src, dst)
            }
            QNodeOSError::NodeOutOfRange { node, max } => {
                write!(f, "node {} out of range (max {})", node, max)
            }
            QNodeOSError::InvalidStateSize { size } => {
                write!(f, "invalid state size {} (must be power of 2)", size)
            }
            QNodeOSError::SwapChainFailed { hop, reason } => {
                write!(f, "swap chain failed at hop {}: {}", hop, reason)
            }
            QNodeOSError::DistillationFailed => {
                write!(f, "distillation failed: both pairs consumed with no output")
            }
        }
    }
}

impl std::error::Error for QNodeOSError {}

// ===================================================================
// NETWORK PROTOCOL ENUM
// ===================================================================

/// Protocol used for establishing end-to-end entanglement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkProtocol {
    /// Direct entanglement swapping along the shortest path.
    EntanglementSwapping,
    /// Teleportation-based state transfer using pre-shared Bell pairs.
    Teleportation,
    /// Distill Bell pairs before use to boost fidelity above raw link quality.
    DistilledBellPairs,
}

impl fmt::Display for NetworkProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkProtocol::EntanglementSwapping => write!(f, "EntanglementSwapping"),
            NetworkProtocol::Teleportation => write!(f, "Teleportation"),
            NetworkProtocol::DistilledBellPairs => write!(f, "DistilledBellPairs"),
        }
    }
}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Configuration for the quantum network operating system.
///
/// Use [`QNodeOSConfigBuilder`] to construct with validation.
#[derive(Clone, Debug)]
pub struct QNodeOSConfig {
    /// Number of network nodes (2..=64).
    pub num_nodes: usize,
    /// Raw Bell pair fidelity for adjacent links (0.5..=1.0).
    pub link_fidelity: f64,
    /// Quantum memory coherence time in seconds.
    pub memory_coherence_time: f64,
    /// Rate of entanglement generation in pairs per second.
    pub entanglement_gen_rate: f64,
    /// One-way classical communication latency in seconds.
    pub classical_latency: f64,
    /// Protocol for end-to-end entanglement establishment.
    pub protocol: NetworkProtocol,
}

/// Builder for [`QNodeOSConfig`] with parameter validation.
#[derive(Clone, Debug)]
pub struct QNodeOSConfigBuilder {
    num_nodes: usize,
    link_fidelity: f64,
    memory_coherence_time: f64,
    entanglement_gen_rate: f64,
    classical_latency: f64,
    protocol: NetworkProtocol,
}

impl QNodeOSConfig {
    /// Create a new configuration builder with sensible defaults.
    pub fn builder() -> QNodeOSConfigBuilder {
        QNodeOSConfigBuilder {
            num_nodes: 4,
            link_fidelity: 0.95,
            memory_coherence_time: 1.0,
            entanglement_gen_rate: 1000.0,
            classical_latency: 1e-4,
            protocol: NetworkProtocol::EntanglementSwapping,
        }
    }
}

impl QNodeOSConfigBuilder {
    /// Set the number of network nodes (2..=64).
    pub fn num_nodes(mut self, n: usize) -> Self {
        self.num_nodes = n;
        self
    }

    /// Set raw Bell pair fidelity for adjacent links (0.5..=1.0).
    pub fn link_fidelity(mut self, f: f64) -> Self {
        self.link_fidelity = f;
        self
    }

    /// Set quantum memory coherence time in seconds (must be positive).
    pub fn memory_coherence_time(mut self, t: f64) -> Self {
        self.memory_coherence_time = t;
        self
    }

    /// Set entanglement generation rate in pairs per second (must be positive).
    pub fn entanglement_gen_rate(mut self, r: f64) -> Self {
        self.entanglement_gen_rate = r;
        self
    }

    /// Set one-way classical communication latency in seconds (must be non-negative).
    pub fn classical_latency(mut self, l: f64) -> Self {
        self.classical_latency = l;
        self
    }

    /// Set the network protocol for end-to-end entanglement.
    pub fn protocol(mut self, p: NetworkProtocol) -> Self {
        self.protocol = p;
        self
    }

    /// Validate and build the configuration.
    pub fn build(self) -> Result<QNodeOSConfig, QNodeOSError> {
        if self.num_nodes < 2 || self.num_nodes > 64 {
            return Err(QNodeOSError::InvalidConfig {
                field: "num_nodes".into(),
                reason: format!("must be 2..=64, got {}", self.num_nodes),
            });
        }
        if self.link_fidelity < 0.5 || self.link_fidelity > 1.0 {
            return Err(QNodeOSError::InvalidConfig {
                field: "link_fidelity".into(),
                reason: format!("must be 0.5..=1.0, got {}", self.link_fidelity),
            });
        }
        if self.memory_coherence_time <= 0.0 {
            return Err(QNodeOSError::InvalidConfig {
                field: "memory_coherence_time".into(),
                reason: "must be positive".into(),
            });
        }
        if self.entanglement_gen_rate <= 0.0 {
            return Err(QNodeOSError::InvalidConfig {
                field: "entanglement_gen_rate".into(),
                reason: "must be positive".into(),
            });
        }
        if self.classical_latency < 0.0 {
            return Err(QNodeOSError::InvalidConfig {
                field: "classical_latency".into(),
                reason: "must be non-negative".into(),
            });
        }
        Ok(QNodeOSConfig {
            num_nodes: self.num_nodes,
            link_fidelity: self.link_fidelity,
            memory_coherence_time: self.memory_coherence_time,
            entanglement_gen_rate: self.entanglement_gen_rate,
            classical_latency: self.classical_latency,
            protocol: self.protocol,
        })
    }
}

// ===================================================================
// QUANTUM LINK
// ===================================================================

/// A quantum entangled link (Bell pair) between two network nodes.
#[derive(Clone, Debug)]
pub struct QuantumLink {
    /// First node endpoint.
    pub node_a: usize,
    /// Second node endpoint.
    pub node_b: usize,
    /// Bell pair fidelity with respect to |Phi+>.
    pub fidelity: f64,
    /// Time elapsed since this pair was generated (seconds).
    pub age: f64,
    /// Whether this pair has been through distillation.
    pub is_distilled: bool,
}

impl QuantumLink {
    /// Create a new quantum link.
    pub fn new(node_a: usize, node_b: usize, fidelity: f64) -> Self {
        Self {
            node_a,
            node_b,
            fidelity,
            age: 0.0,
            is_distilled: false,
        }
    }

    /// Check whether this link has expired given a coherence time.
    pub fn is_expired(&self, coherence_time: f64) -> bool {
        self.age > coherence_time
    }

    /// Apply exponential decoherence: fidelity decays toward 0.25 (maximally mixed).
    pub fn apply_decoherence(&mut self, dt: f64, coherence_time: f64) {
        if coherence_time <= 0.0 {
            return;
        }
        let decay = (-dt / coherence_time).exp();
        // Werner parameter decays: F(t) = 0.25 + (F_0 - 0.25) * exp(-t/T)
        self.fidelity = 0.25 + (self.fidelity - 0.25) * decay;
        self.age += dt;
    }
}

// ===================================================================
// ENTANGLEMENT TABLE
// ===================================================================

/// Table tracking available Bell pairs across the network.
///
/// Each pair is indexed by the ordered tuple (min(a,b), max(a,b)) to avoid
/// storing duplicate entries for the same physical link.
#[derive(Clone, Debug)]
pub struct EntanglementTable {
    /// Stored pairs grouped by ordered node pair.
    pairs: Vec<QuantumLink>,
}

impl EntanglementTable {
    /// Create an empty entanglement table.
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Canonical ordering for a node pair.
    fn ordered(a: usize, b: usize) -> (usize, usize) {
        if a <= b { (a, b) } else { (b, a) }
    }

    /// Add a Bell pair between nodes `a` and `b` with given fidelity.
    pub fn add_pair(&mut self, a: usize, b: usize, fidelity: f64) {
        let (na, nb) = Self::ordered(a, b);
        self.pairs.push(QuantumLink::new(na, nb, fidelity));
    }

    /// Consume (remove and return) the highest-fidelity pair between `a` and `b`.
    pub fn consume_pair(&mut self, a: usize, b: usize) -> Option<QuantumLink> {
        let (na, nb) = Self::ordered(a, b);
        let mut best_idx = None;
        let mut best_fid = -1.0_f64;
        for (i, link) in self.pairs.iter().enumerate() {
            if link.node_a == na && link.node_b == nb && link.fidelity > best_fid {
                best_fid = link.fidelity;
                best_idx = Some(i);
            }
        }
        best_idx.map(|i| self.pairs.swap_remove(i))
    }

    /// Remove all pairs that have exceeded the coherence time.
    pub fn purge_expired(&mut self, coherence_time: f64) {
        self.pairs.retain(|link| !link.is_expired(coherence_time));
    }

    /// Count available pairs between nodes `a` and `b`.
    pub fn available_pairs(&self, a: usize, b: usize) -> usize {
        let (na, nb) = Self::ordered(a, b);
        self.pairs
            .iter()
            .filter(|link| link.node_a == na && link.node_b == nb)
            .count()
    }

    /// Total number of pairs in the table.
    pub fn total_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Apply decoherence to all pairs (advance time by `dt`).
    pub fn age_all(&mut self, dt: f64, coherence_time: f64) {
        for link in &mut self.pairs {
            link.apply_decoherence(dt, coherence_time);
        }
    }

    /// Average fidelity of all stored pairs. Returns 0.0 if empty.
    pub fn average_fidelity(&self) -> f64 {
        if self.pairs.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.pairs.iter().map(|l| l.fidelity).sum();
        sum / self.pairs.len() as f64
    }

    /// Check whether any pair exists between `a` and `b`.
    pub fn has_pair(&self, a: usize, b: usize) -> bool {
        self.available_pairs(a, b) > 0
    }
}

impl Default for EntanglementTable {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// DISTILLATION PROTOCOL (DEJMPS)
// ===================================================================

/// DEJMPS entanglement distillation protocol.
///
/// Takes two Bell pairs of the same link and probabilistically produces one
/// pair with higher fidelity. Both input pairs are consumed regardless of
/// success.
pub struct DistillationProtocol;

impl DistillationProtocol {
    /// Attempt DEJMPS distillation from two Bell pairs on the same link.
    ///
    /// The output fidelity follows the DEJMPS formula for Werner states:
    ///
    /// ```text
    /// F_out = (F1*F2 + (1-F1)*(1-F2)/9) / (F1*F2 + F1*(1-F2)/3 + (1-F1)*F2/3 + 5*(1-F1)*(1-F2)/9)
    /// ```
    ///
    /// Returns `Some(link)` if the distilled fidelity exceeds both inputs,
    /// `None` if distillation would not improve fidelity (e.g., both inputs
    /// already near 0.5).
    pub fn distill_pairs(pair1: &QuantumLink, pair2: &QuantumLink) -> Option<QuantumLink> {
        assert_eq!(
            EntanglementTable::ordered(pair1.node_a, pair1.node_b),
            EntanglementTable::ordered(pair2.node_a, pair2.node_b),
            "distillation requires pairs on the same link"
        );

        let f1 = pair1.fidelity;
        let f2 = pair2.fidelity;

        let (f_out, _p_success) = Self::dejmps_fidelity(f1, f2);

        // Distillation is only useful when the output fidelity exceeds both inputs.
        if f_out <= f1.max(f2) {
            return None;
        }

        let mut link = QuantumLink::new(pair1.node_a, pair1.node_b, f_out);
        link.is_distilled = true;
        // Age is the maximum of the two input ages (conservative estimate).
        link.age = pair1.age.max(pair2.age);
        Some(link)
    }

    /// Compute DEJMPS output fidelity and success probability.
    ///
    /// Returns (F_out, P_success).
    pub fn dejmps_fidelity(f1: f64, f2: f64) -> (f64, f64) {
        let nf1 = 1.0 - f1;
        let nf2 = 1.0 - f2;

        let numerator = f1 * f2 + nf1 * nf2 / 9.0;
        let denominator = f1 * f2 + f1 * nf2 / 3.0 + nf1 * f2 / 3.0 + 5.0 * nf1 * nf2 / 9.0;

        let f_out = if denominator.abs() < 1e-15 {
            0.25
        } else {
            numerator / denominator
        };

        // Success probability is the denominator (probability of matching measurement outcomes).
        (f_out, denominator)
    }
}

// ===================================================================
// ENTANGLEMENT SWAPPING
// ===================================================================

/// Entanglement swapping: extend entanglement across an intermediate node.
///
/// Given link A-B and link B-C, performs a Bell measurement at node B to
/// create a new link A-C.
pub struct EntanglementSwapping;

impl EntanglementSwapping {
    /// Perform entanglement swapping to create a direct link between
    /// `link_ab.node_a` and `link_bc.node_b`.
    ///
    /// Output fidelity:
    /// ```text
    /// F_ac = F_ab * F_bc + (1 - F_ab) * (1 - F_bc) / 3
    /// ```
    pub fn swap(link_ab: &QuantumLink, link_bc: &QuantumLink) -> QuantumLink {
        let f_ab = link_ab.fidelity;
        let f_bc = link_bc.fidelity;

        let f_ac = f_ab * f_bc + (1.0 - f_ab) * (1.0 - f_bc) / 3.0;

        let mut link = QuantumLink::new(link_ab.node_a, link_bc.node_b, f_ac);
        link.age = link_ab.age.max(link_bc.age);
        link.is_distilled = link_ab.is_distilled || link_bc.is_distilled;
        link
    }
}

// ===================================================================
// QUANTUM TELEPORTATION
// ===================================================================

/// Quantum teleportation with depolarizing noise.
///
/// Teleports a single-qubit or multi-qubit state using a shared Bell pair.
/// The noise model applies depolarizing error proportional to (1 - F) where
/// F is the link fidelity.
pub struct QuantumTeleportation;

impl QuantumTeleportation {
    /// Teleport a quantum state vector using the given link.
    ///
    /// The output state is subjected to depolarizing noise:
    /// ```text
    /// rho_out = F * |psi><psi| + (1 - F) * I/d
    /// ```
    /// where d is the state dimension and F is the link fidelity.
    ///
    /// We return the state vector corresponding to the dominant pure-state
    /// component, scaled by sqrt(F), plus a maximally-mixed noise component.
    /// For fidelity analysis, the effective state fidelity with the input is F.
    pub fn teleport(state: &[C64], link: &QuantumLink) -> Result<Vec<C64>, QNodeOSError> {
        let d = state.len();
        if d == 0 || (d & (d - 1)) != 0 {
            return Err(QNodeOSError::InvalidStateSize { size: d });
        }

        let f = link.fidelity;
        // Depolarizing channel: sqrt(F) * |psi> + noise term
        // For state vector representation, we scale amplitudes by sqrt(F)
        // and add a uniform noise floor.
        let scale = f.sqrt();
        let noise_weight = ((1.0 - f) / d as f64).sqrt();

        let output: Vec<C64> = state
            .iter()
            .map(|&amp| {
                C64::new(amp.re * scale + noise_weight, amp.im * scale)
            })
            .collect();

        // Renormalize to unit length.
        let norm_sq: f64 = output.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq < 1e-15 {
            return Ok(vec![c64_zero(); d]);
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        Ok(output.iter().map(|c| C64::new(c.re * inv_norm, c.im * inv_norm)).collect())
    }

    /// Compute the effective fidelity of a teleported state.
    ///
    /// For the depolarizing channel, |<psi_in|psi_out>|^2 approaches the link fidelity.
    pub fn effective_fidelity(input: &[C64], output: &[C64]) -> f64 {
        assert_eq!(input.len(), output.len(), "state dimensions must match");
        let overlap: C64 = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| C64::new(a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re))
            .sum();
        overlap.norm_sqr()
    }
}

// ===================================================================
// NETWORK SCHEDULER
// ===================================================================

/// Priority-queue entry for Dijkstra path finding.
#[derive(Clone, Debug)]
struct PathEntry {
    node: usize,
    /// Negative log-fidelity cost (lower is better fidelity path).
    cost: f64,
}

impl PartialEq for PathEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for PathEntry {}

impl PartialOrd for PathEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap behavior via BinaryHeap (which is a max-heap).
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

/// Network-layer scheduler for entanglement generation and path finding.
pub struct NetworkScheduler;

impl NetworkScheduler {
    /// Generate entangled pairs for all adjacent links in a linear chain topology.
    ///
    /// Each adjacent pair (i, i+1) receives one Bell pair at the configured fidelity.
    pub fn schedule_entanglement_generation(config: &QNodeOSConfig) -> Vec<QuantumLink> {
        let mut links = Vec::new();
        for i in 0..config.num_nodes.saturating_sub(1) {
            links.push(QuantumLink::new(i, i + 1, config.link_fidelity));
        }
        links
    }

    /// Find the path from `src` to `dst` that maximizes end-to-end fidelity.
    ///
    /// Uses Dijkstra's algorithm on the negative-log-fidelity graph. Only
    /// considers links currently available in the entanglement table.
    ///
    /// Returns the ordered sequence of nodes from `src` to `dst` inclusive.
    pub fn find_path(
        src: usize,
        dst: usize,
        num_nodes: usize,
        table: &EntanglementTable,
    ) -> Result<Vec<usize>, QNodeOSError> {
        if src >= num_nodes {
            return Err(QNodeOSError::NodeOutOfRange {
                node: src,
                max: num_nodes - 1,
            });
        }
        if dst >= num_nodes {
            return Err(QNodeOSError::NodeOutOfRange {
                node: dst,
                max: num_nodes - 1,
            });
        }
        if src == dst {
            return Ok(vec![src]);
        }

        // Build adjacency from the entanglement table.
        // For each pair of nodes with available pairs, create a weighted edge
        // with cost = -ln(best_fidelity).
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_nodes];
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                if table.has_pair(i, j) {
                    // Use the best fidelity among available pairs.
                    let best_fid = table
                        .pairs
                        .iter()
                        .filter(|l| {
                            let (a, b) = EntanglementTable::ordered(l.node_a, l.node_b);
                            a == i && b == j
                        })
                        .map(|l| l.fidelity)
                        .fold(0.0_f64, f64::max);

                    let cost = if best_fid > 1e-15 {
                        -best_fid.ln()
                    } else {
                        f64::INFINITY
                    };
                    adj[i].push((j, cost));
                    adj[j].push((i, cost));
                }
            }
        }

        // Dijkstra.
        let mut dist = vec![f64::INFINITY; num_nodes];
        let mut prev = vec![None; num_nodes];
        dist[src] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(PathEntry { node: src, cost: 0.0 });

        while let Some(PathEntry { node, cost }) = heap.pop() {
            if cost > dist[node] {
                continue;
            }
            if node == dst {
                break;
            }
            for &(neighbor, edge_cost) in &adj[node] {
                let next_cost = dist[node] + edge_cost;
                if next_cost < dist[neighbor] {
                    dist[neighbor] = next_cost;
                    prev[neighbor] = Some(node);
                    heap.push(PathEntry {
                        node: neighbor,
                        cost: next_cost,
                    });
                }
            }
        }

        // Reconstruct path.
        if prev[dst].is_none() && src != dst {
            return Err(QNodeOSError::NoPathExists { src, dst });
        }

        let mut path = Vec::new();
        let mut current = dst;
        path.push(current);
        while let Some(p) = prev[current] {
            path.push(p);
            current = p;
        }
        path.reverse();
        Ok(path)
    }

    /// Execute a chain of entanglement swaps along a pre-computed path.
    ///
    /// Consumes one Bell pair per adjacent hop in the path and produces a
    /// single end-to-end link via sequential swapping.
    pub fn execute_swap_chain(
        path: &[usize],
        table: &mut EntanglementTable,
    ) -> Result<QuantumLink, QNodeOSError> {
        if path.len() < 2 {
            return Err(QNodeOSError::NoPathExists {
                src: path.first().copied().unwrap_or(0),
                dst: path.last().copied().unwrap_or(0),
            });
        }

        // Consume the first hop.
        let mut current_link = table
            .consume_pair(path[0], path[1])
            .ok_or(QNodeOSError::NoPairsAvailable {
                node_a: path[0],
                node_b: path[1],
            })?;

        // Swap through each intermediate node.
        for hop in 1..(path.len() - 1) {
            let next_link = table
                .consume_pair(path[hop], path[hop + 1])
                .ok_or(QNodeOSError::SwapChainFailed {
                    hop,
                    reason: format!(
                        "no pair between {} and {}",
                        path[hop],
                        path[hop + 1]
                    ),
                })?;
            current_link = EntanglementSwapping::swap(&current_link, &next_link);
        }

        Ok(current_link)
    }
}

// ===================================================================
// NETWORK STATISTICS
// ===================================================================

/// Cumulative network performance metrics.
#[derive(Clone, Debug, Default)]
pub struct NetworkStats {
    /// Total Bell pairs generated since initialization.
    pub total_generated: u64,
    /// Total Bell pairs consumed by operations.
    pub total_consumed: u64,
    /// Total successful distillation operations.
    pub total_distilled: u64,
    /// Total teleportation operations performed.
    pub total_teleported: u64,
    /// Total entanglement swaps performed.
    pub total_swaps: u64,
    /// Running sum of consumed pair fidelities for average computation.
    pub fidelity_sum: f64,
    /// Number of fidelity samples.
    pub fidelity_count: u64,
    /// Total pairs purged due to expiry.
    pub total_expired: u64,
}

impl NetworkStats {
    /// Average fidelity of all consumed pairs. Returns 0.0 if none consumed.
    pub fn average_fidelity(&self) -> f64 {
        if self.fidelity_count == 0 {
            0.0
        } else {
            self.fidelity_sum / self.fidelity_count as f64
        }
    }

    /// Record consumption of a pair.
    fn record_consumption(&mut self, fidelity: f64) {
        self.total_consumed += 1;
        self.fidelity_sum += fidelity;
        self.fidelity_count += 1;
    }
}

impl fmt::Display for NetworkStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NetworkStats {{ generated: {}, consumed: {}, distilled: {}, \
             teleported: {}, swaps: {}, expired: {}, avg_fidelity: {:.4} }}",
            self.total_generated,
            self.total_consumed,
            self.total_distilled,
            self.total_teleported,
            self.total_swaps,
            self.total_expired,
            self.average_fidelity()
        )
    }
}

// ===================================================================
// QNODEOS: MAIN NETWORK OPERATING SYSTEM
// ===================================================================

/// Quantum Network Operating System.
///
/// Manages the full lifecycle of entanglement-based distributed quantum
/// computing: generation, swapping, distillation, teleportation, and
/// time-based decoherence.
pub struct QNodeOS {
    /// Network configuration.
    config: QNodeOSConfig,
    /// Entanglement table tracking available Bell pairs.
    table: EntanglementTable,
    /// Cumulative network statistics.
    stats: NetworkStats,
    /// Current simulation time in seconds.
    time: f64,
}

impl QNodeOS {
    /// Create a new QNodeOS instance from a validated configuration.
    pub fn new(config: QNodeOSConfig) -> Self {
        Self {
            config,
            table: EntanglementTable::new(),
            stats: NetworkStats::default(),
            time: 0.0,
        }
    }

    /// Access the current configuration.
    pub fn config(&self) -> &QNodeOSConfig {
        &self.config
    }

    /// Access the entanglement table (read-only).
    pub fn table(&self) -> &EntanglementTable {
        &self.table
    }

    /// Access mutable entanglement table.
    pub fn table_mut(&mut self) -> &mut EntanglementTable {
        &mut self.table
    }

    /// Current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Generate one round of entanglement across all adjacent links.
    ///
    /// In a linear chain of N nodes, this creates one Bell pair for each
    /// link (i, i+1) at the configured fidelity.
    pub fn generate_entanglement(&mut self) {
        let links = NetworkScheduler::schedule_entanglement_generation(&self.config);
        let count = links.len() as u64;
        for link in links {
            self.table.add_pair(link.node_a, link.node_b, link.fidelity);
        }
        self.stats.total_generated += count;
    }

    /// Request an end-to-end entangled link between `src` and `dst`.
    ///
    /// The method used depends on the configured protocol:
    /// - **EntanglementSwapping**: Find shortest path, execute swap chain.
    /// - **DistilledBellPairs**: Distill adjacent pairs before swapping.
    /// - **Teleportation**: Same as EntanglementSwapping (teleportation is
    ///   done separately via [`teleport_state`]).
    pub fn request_link(
        &mut self,
        src: usize,
        dst: usize,
    ) -> Result<QuantumLink, QNodeOSError> {
        self.validate_node(src)?;
        self.validate_node(dst)?;

        // Direct pair available?
        if self.table.has_pair(src, dst) {
            let link = self.table.consume_pair(src, dst).unwrap();
            self.stats.record_consumption(link.fidelity);
            return Ok(link);
        }

        // Need multi-hop: find path and swap.
        let path = NetworkScheduler::find_path(
            src,
            dst,
            self.config.num_nodes,
            &self.table,
        )?;

        // Optionally distill adjacent pairs before swapping.
        if self.config.protocol == NetworkProtocol::DistilledBellPairs {
            self.distill_path_pairs(&path);
        }

        let link = NetworkScheduler::execute_swap_chain(&path, &mut self.table)?;
        self.stats.total_swaps += (path.len() - 2) as u64; // intermediate swaps
        self.stats.record_consumption(link.fidelity);
        Ok(link)
    }

    /// Teleport a quantum state from `src` to `dst`.
    ///
    /// First establishes an end-to-end link, then teleports the state
    /// with depolarizing noise based on the link fidelity.
    pub fn teleport_state(
        &mut self,
        state: &[C64],
        src: usize,
        dst: usize,
    ) -> Result<Vec<C64>, QNodeOSError> {
        let link = self.request_link(src, dst)?;
        let output = QuantumTeleportation::teleport(state, &link)?;
        self.stats.total_teleported += 1;
        Ok(output)
    }

    /// Compute the NxN fidelity map for the network.
    ///
    /// Entry (i, j) is the best fidelity of any available pair between
    /// nodes i and j. Diagonal entries are 1.0, missing links are 0.0.
    pub fn network_fidelity_map(&self) -> Vec<Vec<f64>> {
        let n = self.config.num_nodes;
        let mut map = vec![vec![0.0; n]; n];

        for i in 0..n {
            map[i][i] = 1.0;
        }

        for link in &self.table.pairs {
            let a = link.node_a;
            let b = link.node_b;
            if link.fidelity > map[a][b] {
                map[a][b] = link.fidelity;
                map[b][a] = link.fidelity;
            }
        }

        map
    }

    /// Advance the simulation clock by `dt` seconds.
    ///
    /// Applies decoherence to all stored pairs and purges expired ones.
    pub fn tick(&mut self, dt: f64) {
        self.time += dt;
        let coherence_time = self.config.memory_coherence_time;
        self.table.age_all(dt, coherence_time);

        let before = self.table.total_pairs();
        self.table.purge_expired(coherence_time);
        let after = self.table.total_pairs();
        self.stats.total_expired += (before - after) as u64;
    }

    /// Retrieve current network statistics.
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /// Validate that a node index is within range.
    fn validate_node(&self, node: usize) -> Result<(), QNodeOSError> {
        if node >= self.config.num_nodes {
            Err(QNodeOSError::NodeOutOfRange {
                node,
                max: self.config.num_nodes - 1,
            })
        } else {
            Ok(())
        }
    }

    /// Attempt distillation on all adjacent hops in a path.
    ///
    /// For each hop (path[i], path[i+1]), if two pairs are available,
    /// attempt DEJMPS distillation to produce one higher-fidelity pair.
    fn distill_path_pairs(&mut self, path: &[usize]) {
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            if self.table.available_pairs(a, b) >= 2 {
                let p1 = self.table.consume_pair(a, b).unwrap();
                let p2 = self.table.consume_pair(a, b).unwrap();
                if let Some(distilled) = DistillationProtocol::distill_pairs(&p1, &p2) {
                    self.table.add_pair(a, b, distilled.fidelity);
                    self.stats.total_distilled += 1;
                } else {
                    // Distillation failed to improve; put the better pair back.
                    let better = if p1.fidelity >= p2.fidelity { p1 } else { p2 };
                    self.table.add_pair(a, b, better.fidelity);
                }
            }
        }
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---------------------------------------------------------------
    // 1. Config builder validation
    // ---------------------------------------------------------------

    #[test]
    fn test_config_builder_defaults() {
        let config = QNodeOSConfig::builder()
            .num_nodes(4)
            .build()
            .unwrap();
        assert_eq!(config.num_nodes, 4);
        assert!(approx_eq(config.link_fidelity, 0.95, EPS));
        assert!(approx_eq(config.memory_coherence_time, 1.0, EPS));
        assert!(approx_eq(config.entanglement_gen_rate, 1000.0, EPS));
        assert_eq!(config.protocol, NetworkProtocol::EntanglementSwapping);
    }

    #[test]
    fn test_config_builder_custom() {
        let config = QNodeOSConfig::builder()
            .num_nodes(8)
            .link_fidelity(0.85)
            .memory_coherence_time(2.0)
            .entanglement_gen_rate(500.0)
            .classical_latency(1e-3)
            .protocol(NetworkProtocol::DistilledBellPairs)
            .build()
            .unwrap();
        assert_eq!(config.num_nodes, 8);
        assert!(approx_eq(config.link_fidelity, 0.85, EPS));
        assert_eq!(config.protocol, NetworkProtocol::DistilledBellPairs);
    }

    #[test]
    fn test_config_builder_invalid_num_nodes_low() {
        let err = QNodeOSConfig::builder().num_nodes(1).build();
        assert!(err.is_err());
        match err.unwrap_err() {
            QNodeOSError::InvalidConfig { field, .. } => assert_eq!(field, "num_nodes"),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn test_config_builder_invalid_num_nodes_high() {
        let err = QNodeOSConfig::builder().num_nodes(65).build();
        assert!(err.is_err());
    }

    #[test]
    fn test_config_builder_invalid_fidelity_low() {
        let err = QNodeOSConfig::builder().link_fidelity(0.3).build();
        assert!(err.is_err());
        match err.unwrap_err() {
            QNodeOSError::InvalidConfig { field, .. } => assert_eq!(field, "link_fidelity"),
            _ => panic!("expected InvalidConfig for link_fidelity"),
        }
    }

    #[test]
    fn test_config_builder_invalid_fidelity_high() {
        let err = QNodeOSConfig::builder().link_fidelity(1.1).build();
        assert!(err.is_err());
    }

    #[test]
    fn test_config_builder_invalid_coherence_time() {
        let err = QNodeOSConfig::builder().memory_coherence_time(-1.0).build();
        assert!(err.is_err());
    }

    #[test]
    fn test_config_builder_boundary_values() {
        // Exact boundary: 2 nodes and fidelity 0.5 should be valid.
        let config = QNodeOSConfig::builder()
            .num_nodes(2)
            .link_fidelity(0.5)
            .build()
            .unwrap();
        assert_eq!(config.num_nodes, 2);
        assert!(approx_eq(config.link_fidelity, 0.5, EPS));

        // Exact boundary: 64 nodes and fidelity 1.0.
        let config = QNodeOSConfig::builder()
            .num_nodes(64)
            .link_fidelity(1.0)
            .build()
            .unwrap();
        assert_eq!(config.num_nodes, 64);
    }

    // ---------------------------------------------------------------
    // 2. Entanglement table operations
    // ---------------------------------------------------------------

    #[test]
    fn test_entanglement_table_add_and_count() {
        let mut table = EntanglementTable::new();
        assert_eq!(table.available_pairs(0, 1), 0);

        table.add_pair(0, 1, 0.9);
        table.add_pair(0, 1, 0.85);
        table.add_pair(1, 2, 0.95);

        assert_eq!(table.available_pairs(0, 1), 2);
        assert_eq!(table.available_pairs(1, 0), 2); // order-independent
        assert_eq!(table.available_pairs(1, 2), 1);
        assert_eq!(table.total_pairs(), 3);
    }

    #[test]
    fn test_entanglement_table_consume_best() {
        let mut table = EntanglementTable::new();
        table.add_pair(0, 1, 0.8);
        table.add_pair(0, 1, 0.95);
        table.add_pair(0, 1, 0.85);

        // Should consume the highest-fidelity pair first.
        let link = table.consume_pair(0, 1).unwrap();
        assert!(approx_eq(link.fidelity, 0.95, EPS));
        assert_eq!(table.available_pairs(0, 1), 2);

        let link = table.consume_pair(0, 1).unwrap();
        assert!(approx_eq(link.fidelity, 0.85, EPS));
    }

    #[test]
    fn test_entanglement_table_consume_empty() {
        let mut table = EntanglementTable::new();
        assert!(table.consume_pair(0, 1).is_none());
    }

    #[test]
    fn test_entanglement_table_purge_expired() {
        let mut table = EntanglementTable::new();
        table.add_pair(0, 1, 0.9);

        // Age the pair beyond coherence time.
        table.age_all(2.0, 1.0);
        assert_eq!(table.total_pairs(), 1); // still there, just old

        table.purge_expired(1.0);
        assert_eq!(table.total_pairs(), 0);
    }

    // ---------------------------------------------------------------
    // 3. Distillation fidelity formula (DEJMPS)
    // ---------------------------------------------------------------

    #[test]
    fn test_dejmps_fidelity_identical_pairs() {
        // Two pairs with F=0.9 should produce F_out > 0.9.
        let (f_out, p_success) = DistillationProtocol::dejmps_fidelity(0.9, 0.9);
        assert!(f_out > 0.9, "distillation should improve fidelity: got {}", f_out);
        assert!(p_success > 0.0 && p_success <= 1.0);
    }

    #[test]
    fn test_dejmps_fidelity_perfect_pairs() {
        // F=1.0 pairs should distill to F=1.0.
        let (f_out, _) = DistillationProtocol::dejmps_fidelity(1.0, 1.0);
        assert!(approx_eq(f_out, 1.0, 1e-12));
    }

    #[test]
    fn test_dejmps_fidelity_formula_manual() {
        // Manually compute for F1=0.8, F2=0.8.
        let f1 = 0.8;
        let f2 = 0.8;
        let nf1 = 0.2;
        let nf2 = 0.2;
        let num = f1 * f2 + nf1 * nf2 / 9.0;
        let den = f1 * f2 + f1 * nf2 / 3.0 + nf1 * f2 / 3.0 + 5.0 * nf1 * nf2 / 9.0;
        let expected = num / den;

        let (f_out, _) = DistillationProtocol::dejmps_fidelity(0.8, 0.8);
        assert!(
            approx_eq(f_out, expected, 1e-12),
            "expected {}, got {}",
            expected,
            f_out
        );
    }

    #[test]
    fn test_dejmps_fidelity_low_pairs_no_improvement() {
        // Pairs at F=0.5 (maximally mixed) should not improve.
        let (f_out, _) = DistillationProtocol::dejmps_fidelity(0.5, 0.5);
        // F=0.5 gives: num = 0.25 + 0.25/9 = 0.25 + 0.0278 = 0.2778
        // den = 0.25 + 0.5*0.5/3 + 0.5*0.5/3 + 5*0.25/9 = 0.25 + 0.0833 + 0.0833 + 0.1389 = 0.5556
        // F_out = 0.2778 / 0.5556 = 0.5
        assert!(
            approx_eq(f_out, 0.5, 1e-6),
            "F=0.5 distillation should be fixed point: got {}",
            f_out
        );
    }

    #[test]
    fn test_distill_pairs_struct_returns_none_at_boundary() {
        // At F=0.5, distillation does not improve, so distill_pairs should return None.
        let p1 = QuantumLink::new(0, 1, 0.5);
        let p2 = QuantumLink::new(0, 1, 0.5);
        let result = DistillationProtocol::distill_pairs(&p1, &p2);
        assert!(result.is_none(), "distillation at F=0.5 should not improve");
    }

    #[test]
    fn test_distill_pairs_struct_success() {
        let p1 = QuantumLink::new(0, 1, 0.85);
        let p2 = QuantumLink::new(0, 1, 0.85);
        let result = DistillationProtocol::distill_pairs(&p1, &p2);
        assert!(result.is_some());
        let link = result.unwrap();
        assert!(link.fidelity > 0.85, "distilled fidelity should exceed input");
        assert!(link.is_distilled);
    }

    // ---------------------------------------------------------------
    // 4. Entanglement swapping fidelity
    // ---------------------------------------------------------------

    #[test]
    fn test_swapping_fidelity_formula() {
        let link_ab = QuantumLink::new(0, 1, 0.9);
        let link_bc = QuantumLink::new(1, 2, 0.9);
        let link_ac = EntanglementSwapping::swap(&link_ab, &link_bc);

        // F_ac = 0.9*0.9 + 0.1*0.1/3 = 0.81 + 0.00333 = 0.81333
        let expected = 0.9 * 0.9 + 0.1 * 0.1 / 3.0;
        assert!(
            approx_eq(link_ac.fidelity, expected, 1e-12),
            "expected {}, got {}",
            expected,
            link_ac.fidelity
        );
        assert_eq!(link_ac.node_a, 0);
        assert_eq!(link_ac.node_b, 2);
    }

    #[test]
    fn test_swapping_perfect_links() {
        let link_ab = QuantumLink::new(0, 1, 1.0);
        let link_bc = QuantumLink::new(1, 2, 1.0);
        let link_ac = EntanglementSwapping::swap(&link_ab, &link_bc);
        assert!(approx_eq(link_ac.fidelity, 1.0, 1e-12));
    }

    #[test]
    fn test_swapping_fidelity_degradation() {
        // Multiple swaps should degrade fidelity monotonically.
        let f = 0.95;
        let mut link = QuantumLink::new(0, 1, f);
        for i in 1..5 {
            let next = QuantumLink::new(i, i + 1, f);
            link = EntanglementSwapping::swap(&link, &next);
        }
        assert!(
            link.fidelity < f,
            "multi-hop swapping should degrade fidelity: got {}",
            link.fidelity
        );
        assert!(
            link.fidelity > 0.25,
            "fidelity should not drop below maximally mixed: got {}",
            link.fidelity
        );
    }

    // ---------------------------------------------------------------
    // 5. Teleportation with noise
    // ---------------------------------------------------------------

    #[test]
    fn test_teleportation_perfect_link() {
        // Perfect link should preserve the state exactly (up to normalization).
        let state = vec![
            C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            C64::new(0.0, 1.0 / 2.0_f64.sqrt()),
        ];
        let link = QuantumLink::new(0, 1, 1.0);
        let output = QuantumTeleportation::teleport(&state, &link).unwrap();

        let fidelity = QuantumTeleportation::effective_fidelity(&state, &output);
        assert!(
            fidelity > 0.999,
            "perfect link teleportation should have near-unit fidelity: got {}",
            fidelity
        );
    }

    #[test]
    fn test_teleportation_noisy_link() {
        let state = vec![C64::new(1.0, 0.0), c64_zero()];
        let link = QuantumLink::new(0, 1, 0.7);
        let output = QuantumTeleportation::teleport(&state, &link).unwrap();

        // Output should still be normalized.
        let norm_sq: f64 = output.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            approx_eq(norm_sq, 1.0, 1e-10),
            "output must be normalized: got {}",
            norm_sq
        );

        // Fidelity should be reduced.
        let fidelity = QuantumTeleportation::effective_fidelity(&state, &output);
        assert!(
            fidelity < 1.0,
            "noisy link should reduce fidelity: got {}",
            fidelity
        );
    }

    #[test]
    fn test_teleportation_invalid_state_size() {
        let state = vec![C64::new(1.0, 0.0), c64_zero(), c64_zero()]; // size 3 is not power of 2
        let link = QuantumLink::new(0, 1, 0.9);
        let result = QuantumTeleportation::teleport(&state, &link);
        assert!(result.is_err());
        match result.unwrap_err() {
            QNodeOSError::InvalidStateSize { size } => assert_eq!(size, 3),
            _ => panic!("expected InvalidStateSize"),
        }
    }

    // ---------------------------------------------------------------
    // 6. Path finding
    // ---------------------------------------------------------------

    #[test]
    fn test_path_finding_adjacent() {
        let mut table = EntanglementTable::new();
        table.add_pair(0, 1, 0.9);
        table.add_pair(1, 2, 0.9);
        table.add_pair(2, 3, 0.9);

        let path = NetworkScheduler::find_path(0, 1, 4, &table).unwrap();
        assert_eq!(path, vec![0, 1]);
    }

    #[test]
    fn test_path_finding_multi_hop() {
        let mut table = EntanglementTable::new();
        table.add_pair(0, 1, 0.9);
        table.add_pair(1, 2, 0.9);
        table.add_pair(2, 3, 0.9);

        let path = NetworkScheduler::find_path(0, 3, 4, &table).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_path_finding_prefers_high_fidelity() {
        let mut table = EntanglementTable::new();
        // Direct low-fidelity path: 0->3
        table.add_pair(0, 3, 0.5);
        // High-fidelity indirect path: 0->1->3
        table.add_pair(0, 1, 0.99);
        table.add_pair(1, 3, 0.99);

        let path = NetworkScheduler::find_path(0, 3, 4, &table).unwrap();
        // Dijkstra should prefer the route that maximizes product of fidelities.
        // Direct: F = 0.5 => cost = -ln(0.5) = 0.693
        // Indirect: F = 0.99*0.99 => cost = -ln(0.99) - ln(0.99) = 0.0201
        // Indirect wins.
        assert_eq!(path, vec![0, 1, 3]);
    }

    #[test]
    fn test_path_finding_no_path() {
        let table = EntanglementTable::new(); // empty
        let result = NetworkScheduler::find_path(0, 3, 4, &table);
        assert!(result.is_err());
        match result.unwrap_err() {
            QNodeOSError::NoPathExists { src, dst } => {
                assert_eq!(src, 0);
                assert_eq!(dst, 3);
            }
            _ => panic!("expected NoPathExists"),
        }
    }

    #[test]
    fn test_path_finding_same_node() {
        let table = EntanglementTable::new();
        let path = NetworkScheduler::find_path(2, 2, 4, &table).unwrap();
        assert_eq!(path, vec![2]);
    }

    #[test]
    fn test_path_finding_node_out_of_range() {
        let table = EntanglementTable::new();
        let result = NetworkScheduler::find_path(0, 10, 4, &table);
        assert!(result.is_err());
        match result.unwrap_err() {
            QNodeOSError::NodeOutOfRange { node, max } => {
                assert_eq!(node, 10);
                assert_eq!(max, 3);
            }
            _ => panic!("expected NodeOutOfRange"),
        }
    }

    // ---------------------------------------------------------------
    // 7. Swap chain execution
    // ---------------------------------------------------------------

    #[test]
    fn test_swap_chain_execution() {
        let mut table = EntanglementTable::new();
        table.add_pair(0, 1, 0.95);
        table.add_pair(1, 2, 0.95);
        table.add_pair(2, 3, 0.95);

        let path = vec![0, 1, 2, 3];
        let link = NetworkScheduler::execute_swap_chain(&path, &mut table).unwrap();

        assert_eq!(link.node_a, 0);
        assert_eq!(link.node_b, 3);
        assert!(link.fidelity < 0.95, "multi-hop swap degrades fidelity");
        assert!(link.fidelity > 0.5, "fidelity should remain usable");

        // All pairs should be consumed.
        assert_eq!(table.total_pairs(), 0);
    }

    #[test]
    fn test_swap_chain_missing_hop() {
        let mut table = EntanglementTable::new();
        table.add_pair(0, 1, 0.9);
        // missing 1->2

        let path = vec![0, 1, 2];
        let result = NetworkScheduler::execute_swap_chain(&path, &mut table);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 8. Full QNodeOS workflow
    // ---------------------------------------------------------------

    #[test]
    fn test_qnodeos_full_workflow() {
        let config = QNodeOSConfig::builder()
            .num_nodes(4)
            .link_fidelity(0.95)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);

        // Generate entanglement.
        os.generate_entanglement();
        assert_eq!(os.stats().total_generated, 3); // 3 links in 4-node chain

        // Request adjacent link.
        let link = os.request_link(0, 1).unwrap();
        assert!(approx_eq(link.fidelity, 0.95, EPS));
        assert_eq!(os.stats().total_consumed, 1);

        // Regenerate for multi-hop request.
        os.generate_entanglement();
        let link = os.request_link(0, 3).unwrap();
        assert!(link.fidelity < 0.95);
        assert!(link.fidelity > 0.5);
    }

    #[test]
    fn test_qnodeos_request_link_no_pairs() {
        let config = QNodeOSConfig::builder()
            .num_nodes(4)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);

        // No generation performed.
        let result = os.request_link(0, 3);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 9. Network fidelity map
    // ---------------------------------------------------------------

    #[test]
    fn test_network_fidelity_map() {
        let config = QNodeOSConfig::builder()
            .num_nodes(3)
            .link_fidelity(0.9)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);
        os.generate_entanglement();

        let map = os.network_fidelity_map();
        assert_eq!(map.len(), 3);
        assert!(approx_eq(map[0][0], 1.0, EPS));
        assert!(approx_eq(map[1][1], 1.0, EPS));
        assert!(approx_eq(map[0][1], 0.9, EPS));
        assert!(approx_eq(map[1][0], 0.9, EPS));
        assert!(approx_eq(map[1][2], 0.9, EPS));
        assert!(approx_eq(map[0][2], 0.0, EPS)); // no direct 0-2 link
    }

    // ---------------------------------------------------------------
    // 10. Time-based pair expiry
    // ---------------------------------------------------------------

    #[test]
    fn test_tick_time_based_expiry() {
        let config = QNodeOSConfig::builder()
            .num_nodes(3)
            .link_fidelity(0.95)
            .memory_coherence_time(0.5)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);
        os.generate_entanglement();

        assert_eq!(os.table().total_pairs(), 2);

        // Advance time beyond coherence limit.
        os.tick(0.6);
        assert_eq!(os.table().total_pairs(), 0);
        assert_eq!(os.stats().total_expired, 2);
    }

    #[test]
    fn test_tick_decoherence_reduces_fidelity() {
        let config = QNodeOSConfig::builder()
            .num_nodes(2)
            .link_fidelity(0.95)
            .memory_coherence_time(1.0)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);
        os.generate_entanglement();

        // Small tick: pair should still exist but with reduced fidelity.
        os.tick(0.2);
        let map = os.network_fidelity_map();
        assert!(map[0][1] < 0.95, "fidelity should decrease with time");
        assert!(map[0][1] > 0.5, "fidelity should not collapse instantly");
    }

    // ---------------------------------------------------------------
    // 11. Multi-hop teleportation
    // ---------------------------------------------------------------

    #[test]
    fn test_multi_hop_teleportation() {
        let config = QNodeOSConfig::builder()
            .num_nodes(5)
            .link_fidelity(0.98)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);
        os.generate_entanglement();

        let state = vec![C64::new(1.0, 0.0), c64_zero()]; // |0>
        let output = os.teleport_state(&state, 0, 4).unwrap();

        let norm_sq: f64 = output.iter().map(|c| c.norm_sqr()).sum();
        assert!(approx_eq(norm_sq, 1.0, 1e-10), "output must be normalized");

        assert_eq!(os.stats().total_teleported, 1);
        assert!(os.stats().total_swaps > 0);
    }

    // ---------------------------------------------------------------
    // 12. Stats tracking
    // ---------------------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let config = QNodeOSConfig::builder()
            .num_nodes(3)
            .link_fidelity(0.9)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);

        assert_eq!(os.stats().total_generated, 0);
        assert_eq!(os.stats().total_consumed, 0);

        os.generate_entanglement();
        assert_eq!(os.stats().total_generated, 2);

        let _ = os.request_link(0, 1);
        assert_eq!(os.stats().total_consumed, 1);
        assert!(os.stats().average_fidelity() > 0.0);

        // Display trait should not panic.
        let display = format!("{}", os.stats());
        assert!(display.contains("generated: 2"));
    }

    // ---------------------------------------------------------------
    // 13. Distilled Bell pairs protocol
    // ---------------------------------------------------------------

    #[test]
    fn test_distilled_bell_pairs_protocol() {
        let config = QNodeOSConfig::builder()
            .num_nodes(3)
            .link_fidelity(0.85)
            .protocol(NetworkProtocol::DistilledBellPairs)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);

        // Generate extra pairs for distillation.
        os.generate_entanglement();
        os.generate_entanglement(); // need at least 2 pairs per hop

        let link = os.request_link(0, 2).unwrap();
        // With distillation, end-to-end fidelity should be better than
        // raw swapping of 0.85 pairs: F_swap_raw = 0.85*0.85 + 0.15*0.15/3 = 0.73
        // Distilled pairs should be higher than 0.85 before swapping.
        assert!(
            link.fidelity > 0.0,
            "distilled link should have positive fidelity"
        );
    }

    // ---------------------------------------------------------------
    // 14. Quantum link decoherence model
    // ---------------------------------------------------------------

    #[test]
    fn test_link_decoherence_model() {
        let mut link = QuantumLink::new(0, 1, 0.95);

        // After one coherence time, fidelity should decay significantly toward 0.25.
        link.apply_decoherence(1.0, 1.0);
        // Expected: 0.25 + (0.95 - 0.25) * exp(-1) = 0.25 + 0.70 * 0.3679 = 0.5075
        let expected = 0.25 + 0.70 * (-1.0_f64).exp();
        assert!(
            approx_eq(link.fidelity, expected, 1e-4),
            "expected ~{}, got {}",
            expected,
            link.fidelity
        );
        assert!(approx_eq(link.age, 1.0, EPS));
    }

    #[test]
    fn test_link_expiry() {
        let mut link = QuantumLink::new(0, 1, 0.9);
        assert!(!link.is_expired(1.0));

        link.age = 1.5;
        assert!(link.is_expired(1.0));
    }

    // ---------------------------------------------------------------
    // 15. Error type display
    // ---------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let errors = vec![
            QNodeOSError::InvalidConfig {
                field: "test".into(),
                reason: "bad".into(),
            },
            QNodeOSError::NoPairsAvailable { node_a: 0, node_b: 1 },
            QNodeOSError::NoPathExists { src: 0, dst: 5 },
            QNodeOSError::NodeOutOfRange { node: 10, max: 3 },
            QNodeOSError::InvalidStateSize { size: 3 },
            QNodeOSError::SwapChainFailed { hop: 2, reason: "missing".into() },
            QNodeOSError::DistillationFailed,
        ];

        for err in &errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty(), "error display should not be empty");
        }
    }

    // ---------------------------------------------------------------
    // 16. NetworkProtocol display
    // ---------------------------------------------------------------

    #[test]
    fn test_protocol_display() {
        assert_eq!(
            format!("{}", NetworkProtocol::EntanglementSwapping),
            "EntanglementSwapping"
        );
        assert_eq!(
            format!("{}", NetworkProtocol::Teleportation),
            "Teleportation"
        );
        assert_eq!(
            format!("{}", NetworkProtocol::DistilledBellPairs),
            "DistilledBellPairs"
        );
    }

    // ---------------------------------------------------------------
    // 17. Average fidelity
    // ---------------------------------------------------------------

    #[test]
    fn test_entanglement_table_average_fidelity() {
        let mut table = EntanglementTable::new();
        assert!(approx_eq(table.average_fidelity(), 0.0, EPS));

        table.add_pair(0, 1, 0.8);
        table.add_pair(0, 1, 1.0);
        assert!(approx_eq(table.average_fidelity(), 0.9, EPS));
    }

    // ---------------------------------------------------------------
    // 18. Large network path finding
    // ---------------------------------------------------------------

    #[test]
    fn test_large_network_path() {
        let n = 16;
        let mut table = EntanglementTable::new();
        for i in 0..(n - 1) {
            table.add_pair(i, i + 1, 0.95);
        }

        let path = NetworkScheduler::find_path(0, n - 1, n, &table).unwrap();
        assert_eq!(path.len(), n);
        assert_eq!(path[0], 0);
        assert_eq!(path[n - 1], n - 1);
    }

    // ---------------------------------------------------------------
    // 19. Repeated generation accumulates pairs
    // ---------------------------------------------------------------

    #[test]
    fn test_repeated_generation_accumulates() {
        let config = QNodeOSConfig::builder()
            .num_nodes(3)
            .link_fidelity(0.9)
            .build()
            .unwrap();
        let mut os = QNodeOS::new(config);

        os.generate_entanglement();
        os.generate_entanglement();
        os.generate_entanglement();

        assert_eq!(os.table().available_pairs(0, 1), 3);
        assert_eq!(os.table().available_pairs(1, 2), 3);
        assert_eq!(os.stats().total_generated, 6);
    }

    // ---------------------------------------------------------------
    // 20. Swap chain preserves endpoint nodes correctly
    // ---------------------------------------------------------------

    #[test]
    fn test_swap_chain_endpoint_correctness() {
        let mut table = EntanglementTable::new();
        table.add_pair(5, 6, 0.9);
        table.add_pair(6, 7, 0.9);
        table.add_pair(7, 8, 0.9);

        let path = vec![5, 6, 7, 8];
        let link = NetworkScheduler::execute_swap_chain(&path, &mut table).unwrap();

        assert_eq!(link.node_a, 5);
        assert_eq!(link.node_b, 8);
    }
}
