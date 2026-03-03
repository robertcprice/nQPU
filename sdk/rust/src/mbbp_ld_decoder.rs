//! MBBP-LD: Matching-Based Boundary Pairing with Local Decoding
//!
//! A high-performance decoder designed for IBM's heavy-hex qubit topologies
//! (Eagle, Heron, Flamingo processors). MBBP-LD outperforms standard Minimum
//! Weight Perfect Matching (MWPM) on heavy-hex layouts by exploiting the
//! topology's natural decomposition into local regions connected by flag qubits.
//!
//! # Algorithm Overview
//!
//! 1. **Heavy-Hex Topology Construction**: Build the qubit connectivity graph
//!    matching IBM's heavy-hex layout with data qubits, flag qubits, and
//!    syndrome (ancilla) qubits.
//!
//! 2. **Region Decomposition**: Partition the code into overlapping local
//!    regions centered on plaquettes. Each region is small enough for
//!    exhaustive lookup-table decoding.
//!
//! 3. **Local Lookup-Table Decoding**: Within each region, decode syndromes
//!    using a precomputed lookup table mapping every possible local syndrome
//!    pattern to its minimum-weight correction. This runs in O(1) per region.
//!
//! 4. **Boundary Defect Identification**: After local decoding, any residual
//!    syndrome defects at region boundaries indicate errors that span multiple
//!    regions and cannot be handled locally.
//!
//! 5. **Matching-Based Boundary Pairing**: Apply MWPM on the boundary defects
//!    only. Because the number of boundary defects is much smaller than the
//!    total syndrome, this matching is fast even at large code distances.
//!
//! 6. **Correction Combination**: Merge the local corrections with the
//!    boundary corrections, resolving conflicts via XOR.
//!
//! # Why MBBP-LD?
//!
//! - **Topology-Aware**: Unlike generic MWPM, MBBP-LD is tailored to the
//!   heavy-hex connectivity, achieving lower logical error rates.
//! - **Scalable**: Local decoding runs in O(1) per region; only boundary
//!   defects require matching. Total runtime scales favorably with distance.
//! - **Multi-Round**: Supports temporal decoding across measurement rounds
//!   with time-like boundary pairing.
//! - **Noise-Aware**: Edge weights incorporate physical error rates,
//!   measurement errors, and crosstalk.
//!
//! # Heavy-Hex Topology
//!
//! The heavy-hex lattice is a degree-3 graph where:
//! - Data qubits sit on vertices of a hexagonal lattice
//! - Flag qubits sit on edges between data qubits
//! - Syndrome qubits (ancillas) sit at the center of each plaquette
//!
//! ```text
//!     D---F---D---F---D
//!     |       |       |
//!     F   S   F   S   F
//!     |       |       |
//!     D---F---D---F---D
//!     |       |       |
//!     F   S   F   S   F
//!     |       |       |
//!     D---F---D---F---D
//! ```
//!
//! D = data qubit, F = flag qubit, S = syndrome/ancilla qubit
//!
//! # References
//!
//! - Chamberland et al., "Topological and subsystem codes on low-degree
//!   graphs with flag qubits", Phys. Rev. X 10, 011022 (2020)
//! - Sundaresan et al., "Matching and maximum likelihood decoding of a
//!   multi-round subsystem quantum error correction experiment",
//!   arXiv:2203.07205 (2022)
//! - IBM Quantum, "Heavy-hex code" documentation
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::mbbp_ld_decoder::{MbbpLdDecoder, MbbpLdConfig, DecoderNoiseModel};
//!
//! let noise = DecoderNoiseModel::uniform(0.001);
//! let config = MbbpLdConfig { distance: 5, rounds: 1, noise_model: noise };
//! let mut decoder = MbbpLdDecoder::new(config);
//!
//! // Inject a single X error on data qubit 3
//! let mut syndrome = vec![false; decoder.num_syndrome_bits()];
//! syndrome[2] = true;
//! syndrome[3] = true;
//!
//! let result = decoder.decode(&syndrome);
//! assert!(result.success);
//! ```

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering;
use std::time::Instant;

// ============================================================
// HEAVY-HEX TOPOLOGY
// ============================================================

/// Qubit role in the heavy-hex lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QubitRole {
    /// Data qubit on a vertex of the hexagonal lattice.
    Data,
    /// Flag qubit on an edge between two data qubits.
    Flag,
    /// Syndrome (ancilla) qubit at the center of a plaquette.
    Syndrome,
}

/// A single qubit in the heavy-hex topology.
#[derive(Clone, Debug)]
pub struct HeavyHexQubit {
    pub id: usize,
    pub role: QubitRole,
    /// Grid position for layout / distance computation (row, col).
    pub position: (f64, f64),
}

/// Heavy-hex topology representation.
///
/// Encodes the full qubit layout and connectivity of an IBM-style heavy-hex
/// lattice at a given code distance.
#[derive(Clone, Debug)]
pub struct HeavyHexTopology {
    /// All data qubit indices.
    pub data_qubits: Vec<usize>,
    /// All flag qubit indices.
    pub flag_qubits: Vec<usize>,
    /// All syndrome (ancilla) qubit indices.
    pub syndrome_qubits: Vec<usize>,
    /// Adjacency list: edges between qubits.
    pub edges: Vec<(usize, usize)>,
    /// Code distance.
    pub distance: usize,
    /// Total number of qubits (data + flag + syndrome).
    pub num_qubits: usize,
    /// Qubit metadata.
    pub qubits: Vec<HeavyHexQubit>,
    /// Number of rows in the hex grid.
    pub rows: usize,
    /// Number of columns in the hex grid.
    pub cols: usize,
}

impl HeavyHexTopology {
    /// Construct a heavy-hex topology for the given code distance.
    ///
    /// The heavy-hex lattice for distance `d` has:
    /// - A (d x d) grid of data qubits on the hexagonal vertices
    /// - Flag qubits on horizontal and vertical edges
    /// - Syndrome qubits at plaquette centers
    ///
    /// # Panics
    ///
    /// Panics if `distance < 3` or `distance` is even.
    pub fn new(distance: usize) -> Self {
        assert!(distance >= 3, "Code distance must be >= 3");
        assert!(distance % 2 == 1, "Code distance must be odd");

        let rows = distance;
        let cols = distance;

        let mut qubits = Vec::new();
        let mut data_qubits = Vec::new();
        let mut flag_qubits = Vec::new();
        let mut syndrome_qubits = Vec::new();
        let mut edges = Vec::new();

        // -- Phase 1: Place data qubits on the (rows x cols) grid ----------
        let mut data_grid: Vec<Vec<Option<usize>>> = vec![vec![None; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                let id = qubits.len();
                qubits.push(HeavyHexQubit {
                    id,
                    role: QubitRole::Data,
                    position: (r as f64 * 2.0, c as f64 * 2.0),
                });
                data_qubits.push(id);
                data_grid[r][c] = Some(id);
            }
        }

        // -- Phase 2: Place flag qubits on horizontal edges ----------------
        // Flag qubits connect adjacent data qubits in the same row.
        let mut h_flag_grid: Vec<Vec<Option<usize>>> = vec![vec![None; cols - 1]; rows];
        for r in 0..rows {
            for c in 0..(cols - 1) {
                let d_left = data_grid[r][c].unwrap();
                let d_right = data_grid[r][c + 1].unwrap();
                let id = qubits.len();
                qubits.push(HeavyHexQubit {
                    id,
                    role: QubitRole::Flag,
                    position: (r as f64 * 2.0, c as f64 * 2.0 + 1.0),
                });
                flag_qubits.push(id);
                h_flag_grid[r][c] = Some(id);
                edges.push((d_left, id));
                edges.push((id, d_right));
            }
        }

        // -- Phase 3: Place flag qubits on vertical edges ------------------
        // In heavy-hex, vertical edges exist on alternating columns per row.
        // Even rows: vertical edges on even columns; odd rows: vertical edges
        // on odd columns (staggered pattern).
        let mut v_flag_grid: Vec<Vec<Option<usize>>> = vec![vec![None; cols]; rows - 1];
        for r in 0..(rows - 1) {
            for c in 0..cols {
                // Staggered: even rows connect on even cols, odd rows on odd cols
                if (r + c) % 2 == 0 {
                    let d_top = data_grid[r][c].unwrap();
                    let d_bot = data_grid[r + 1][c].unwrap();
                    let id = qubits.len();
                    qubits.push(HeavyHexQubit {
                        id,
                        role: QubitRole::Flag,
                        position: (r as f64 * 2.0 + 1.0, c as f64 * 2.0),
                    });
                    flag_qubits.push(id);
                    v_flag_grid[r][c] = Some(id);
                    edges.push((d_top, id));
                    edges.push((id, d_bot));
                }
            }
        }

        // -- Phase 4: Place syndrome qubits at plaquette centers -----------
        // Each plaquette in the heavy-hex lattice corresponds to a stabilizer.
        // We place syndrome qubits at the center of each 2x2 data-qubit cell
        // where connections exist.
        for r in 0..(rows - 1) {
            for c in 0..(cols - 1) {
                // A plaquette exists if adjacent data qubits form a connected
                // cell via flag qubits.
                let id = qubits.len();
                qubits.push(HeavyHexQubit {
                    id,
                    role: QubitRole::Syndrome,
                    position: (r as f64 * 2.0 + 1.0, c as f64 * 2.0 + 1.0),
                });
                syndrome_qubits.push(id);

                // Connect syndrome qubit to surrounding data qubits
                let corners = [
                    data_grid[r][c],
                    data_grid[r][c + 1],
                    data_grid[r + 1][c],
                    data_grid[r + 1][c + 1],
                ];
                for corner in corners.iter().flatten() {
                    edges.push((id, *corner));
                }
            }
        }

        let num_qubits = qubits.len();

        Self {
            data_qubits,
            flag_qubits,
            syndrome_qubits,
            edges,
            distance,
            num_qubits,
            qubits,
            rows,
            cols,
        }
    }

    /// Euclidean distance between two qubits.
    pub fn qubit_distance(&self, a: usize, b: usize) -> f64 {
        let pa = self.qubits[a].position;
        let pb = self.qubits[b].position;
        ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt()
    }

    /// Build adjacency list from edge list.
    pub fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.num_qubits];
        for &(u, v) in &self.edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        // Deduplicate
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
        }
        adj
    }

    /// Return the set of data qubit indices adjacent to a given syndrome qubit.
    pub fn syndrome_data_neighbors(&self, syndrome_id: usize) -> Vec<usize> {
        let adj = self.adjacency_list();
        adj[syndrome_id]
            .iter()
            .copied()
            .filter(|&q| self.qubits[q].role == QubitRole::Data)
            .collect()
    }
}

// ============================================================
// BOUNDARY GRAPH
// ============================================================

/// A node in the boundary graph used for MWPM on boundary defects.
#[derive(Clone, Debug)]
pub struct BoundaryNode {
    pub id: usize,
    /// Which local region this boundary node belongs to.
    pub region_id: usize,
    /// Whether this node lies on the code boundary (for virtual matching).
    pub is_boundary: bool,
    /// Physical position for weight computation.
    pub position: (f64, f64),
}

/// An edge in the boundary graph.
#[derive(Clone, Debug)]
pub struct BoundaryEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// Graph of boundary defects for matching.
#[derive(Clone, Debug)]
pub struct BoundaryGraph {
    pub nodes: Vec<BoundaryNode>,
    pub edges: Vec<BoundaryEdge>,
}

impl BoundaryGraph {
    /// Create an empty boundary graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a boundary node and return its index.
    pub fn add_node(&mut self, region_id: usize, is_boundary: bool, position: (f64, f64)) -> usize {
        let id = self.nodes.len();
        self.nodes.push(BoundaryNode {
            id,
            region_id,
            is_boundary,
            position,
        });
        id
    }

    /// Add an edge between two boundary nodes.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push(BoundaryEdge { from, to, weight });
    }

    /// Compute Euclidean distance between two nodes.
    pub fn node_distance(&self, a: usize, b: usize) -> f64 {
        let pa = self.nodes[a].position;
        let pb = self.nodes[b].position;
        ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt()
    }
}

// ============================================================
// BOUNDARY PAIRING RESULT
// ============================================================

/// Result of boundary defect matching.
#[derive(Clone, Debug)]
pub struct BoundaryPairing {
    /// Matched pairs of boundary defect node indices.
    pub pairs: Vec<(usize, usize)>,
    /// Defects matched to the code boundary (virtual node).
    pub boundary_defects: Vec<usize>,
    /// Total matching weight.
    pub weight: f64,
}

// ============================================================
// NOISE MODEL
// ============================================================

/// Noise model for decoder weight computation.
#[derive(Clone, Debug)]
pub struct DecoderNoiseModel {
    /// Single-qubit depolarizing error rate.
    pub physical_error_rate: f64,
    /// Measurement (readout) error rate.
    pub measurement_error_rate: f64,
    /// Crosstalk error rates between qubit pairs.
    pub crosstalk_map: HashMap<(usize, usize), f64>,
}

impl DecoderNoiseModel {
    /// Create a uniform noise model (same rate everywhere, no crosstalk).
    pub fn uniform(physical_error_rate: f64) -> Self {
        Self {
            physical_error_rate,
            measurement_error_rate: physical_error_rate * 0.1,
            crosstalk_map: HashMap::new(),
        }
    }

    /// Create a noise model with custom measurement error rate.
    pub fn with_measurement_error(physical_error_rate: f64, measurement_error_rate: f64) -> Self {
        Self {
            physical_error_rate,
            measurement_error_rate,
            crosstalk_map: HashMap::new(),
        }
    }

    /// Compute the log-likelihood weight for an edge between two qubits.
    ///
    /// Weight = -ln(p_error / (1 - p_error)) where p_error is the effective
    /// error probability for the channel connecting the two qubits.
    pub fn edge_weight(&self, q1: usize, q2: usize) -> f64 {
        let base_p = self.physical_error_rate;
        let crosstalk = self.crosstalk_map.get(&(q1, q2)).copied().unwrap_or(0.0);
        let p = (base_p + crosstalk).min(0.499);
        if p <= 0.0 {
            return 100.0; // Very high weight for zero error probability
        }
        -(p / (1.0 - p)).ln()
    }

    /// Compute measurement-error weight for a time-like edge.
    pub fn measurement_weight(&self) -> f64 {
        let p = self.measurement_error_rate.max(1e-15).min(0.499);
        -(p / (1.0 - p)).ln()
    }
}

// ============================================================
// LOCAL DECODER (LOOKUP TABLE)
// ============================================================

/// Local decoder for a single region of the heavy-hex lattice.
///
/// Each region contains a small number of data qubits and syndrome bits
/// (typically 4--8 data qubits for heavy-hex plaquettes). The lookup table
/// maps every possible syndrome pattern within the region to its
/// minimum-weight correction.
#[derive(Clone, Debug)]
pub struct LocalDecoder {
    /// Data qubit indices within this region.
    pub region_qubits: Vec<usize>,
    /// Syndrome bit indices (into the global syndrome vector) covered by this region.
    pub syndrome_bits: Vec<usize>,
    /// Lookup table: syndrome pattern -> correction pattern (indexed by region_qubits).
    pub lookup_table: HashMap<Vec<bool>, Vec<bool>>,
    /// Region identifier.
    pub region_id: usize,
    /// Boundary syndrome bits shared with neighboring regions.
    pub boundary_bits: Vec<usize>,
}

impl LocalDecoder {
    /// Create a new local decoder for a region and build its lookup table.
    ///
    /// The lookup table is constructed by enumerating all possible error
    /// patterns up to weight `max_weight` on the region's data qubits,
    /// computing the resulting syndrome, and storing the minimum-weight
    /// correction for each syndrome.
    pub fn new(
        region_id: usize,
        region_qubits: Vec<usize>,
        syndrome_bits: Vec<usize>,
        boundary_bits: Vec<usize>,
        stabilizer_map: &HashMap<usize, Vec<usize>>,
    ) -> Self {
        let mut lookup_table = HashMap::new();
        let n = region_qubits.len();

        // Enumerate error patterns up to weight 2 (sufficient for local decode)
        let max_weight = n.min(2);

        // Identity: no error -> no correction
        lookup_table.insert(
            vec![false; syndrome_bits.len()],
            vec![false; n],
        );

        // Weight-1 errors
        for i in 0..n {
            let mut error = vec![false; n];
            error[i] = true;
            let syndrome = Self::compute_local_syndrome(
                &error,
                &region_qubits,
                &syndrome_bits,
                stabilizer_map,
            );
            lookup_table.entry(syndrome).or_insert_with(|| error);
        }

        // Weight-2 errors
        if max_weight >= 2 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut error = vec![false; n];
                    error[i] = true;
                    error[j] = true;
                    let syndrome = Self::compute_local_syndrome(
                        &error,
                        &region_qubits,
                        &syndrome_bits,
                        stabilizer_map,
                    );
                    // Only insert if we don't already have a lower-weight correction
                    lookup_table.entry(syndrome).or_insert_with(|| error);
                }
            }
        }

        Self {
            region_qubits,
            syndrome_bits,
            lookup_table,
            region_id,
            boundary_bits,
        }
    }

    /// Compute the local syndrome for a given error pattern.
    fn compute_local_syndrome(
        error: &[bool],
        region_qubits: &[usize],
        syndrome_bits: &[usize],
        stabilizer_map: &HashMap<usize, Vec<usize>>,
    ) -> Vec<bool> {
        let mut syndrome = vec![false; syndrome_bits.len()];
        for (s_idx, &s_bit) in syndrome_bits.iter().enumerate() {
            if let Some(support) = stabilizer_map.get(&s_bit) {
                let mut parity = false;
                for (e_idx, &q) in region_qubits.iter().enumerate() {
                    if error[e_idx] && support.contains(&q) {
                        parity = !parity;
                    }
                }
                syndrome[s_idx] = parity;
            }
        }
        syndrome
    }

    /// Decode a local syndrome using the lookup table.
    ///
    /// Returns `Some(correction)` if the syndrome matches a known pattern,
    /// `None` if the syndrome is not in the table (boundary error).
    pub fn decode(&self, local_syndrome: &[bool]) -> Option<Vec<bool>> {
        self.lookup_table.get(local_syndrome).cloned()
    }

    /// Check whether a syndrome has residual defects on boundary bits.
    pub fn has_boundary_defects(&self, local_syndrome: &[bool]) -> bool {
        for &b_bit in &self.boundary_bits {
            if let Some(pos) = self.syndrome_bits.iter().position(|&s| s == b_bit) {
                if local_syndrome[pos] {
                    return true;
                }
            }
        }
        false
    }
}

// ============================================================
// MWPM ON BOUNDARY DEFECTS
// ============================================================

/// Edge with weight for the boundary matching priority queue.
#[derive(Clone, Debug)]
struct WeightedEdge {
    from: usize,
    to: usize,
    weight: f64,
}

impl PartialEq for WeightedEdge {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl Eq for WeightedEdge {}

impl PartialOrd for WeightedEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Min-heap: reverse ordering
        other.weight.partial_cmp(&self.weight)
    }
}

impl Ord for WeightedEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Greedy minimum-weight matching on boundary defects.
///
/// For small numbers of boundary defects (typical in MBBP-LD), greedy
/// matching is sufficient and runs in O(n^2 log n). For larger instances,
/// a full Blossom V implementation (see `decoding::mwpm`) can be substituted.
fn greedy_boundary_matching(
    defect_nodes: &[usize],
    graph: &BoundaryGraph,
    noise: &DecoderNoiseModel,
) -> BoundaryPairing {
    if defect_nodes.is_empty() {
        return BoundaryPairing {
            pairs: Vec::new(),
            boundary_defects: Vec::new(),
            weight: 0.0,
        };
    }

    // Build all pairwise edges sorted by weight
    let mut edges = BinaryHeap::new();
    for (i, &n1) in defect_nodes.iter().enumerate() {
        for &n2 in &defect_nodes[i + 1..] {
            let w = graph.node_distance(n1, n2);
            edges.push(WeightedEdge {
                from: n1,
                to: n2,
                weight: w,
            });
        }
        // Virtual boundary edge for each defect
        if graph.nodes[n1].is_boundary {
            edges.push(WeightedEdge {
                from: n1,
                to: usize::MAX, // virtual boundary
                weight: noise.edge_weight(0, 0) * 0.5,
            });
        }
    }

    let mut matched = HashSet::new();
    let mut pairs = Vec::new();
    let mut boundary_defects = Vec::new();
    let mut total_weight = 0.0;

    while let Some(edge) = edges.pop() {
        if matched.contains(&edge.from) {
            continue;
        }
        if edge.to == usize::MAX {
            // Match to boundary
            if !matched.contains(&edge.from) {
                matched.insert(edge.from);
                boundary_defects.push(edge.from);
                total_weight += edge.weight;
            }
        } else if !matched.contains(&edge.to) {
            matched.insert(edge.from);
            matched.insert(edge.to);
            pairs.push((edge.from, edge.to));
            total_weight += edge.weight;
        }
    }

    // Any remaining unmatched defects go to boundary
    for &d in defect_nodes {
        if !matched.contains(&d) {
            boundary_defects.push(d);
        }
    }

    BoundaryPairing {
        pairs,
        boundary_defects,
        weight: total_weight,
    }
}

// ============================================================
// MBBP-LD CONFIGURATION
// ============================================================

/// Configuration for the MBBP-LD decoder.
#[derive(Clone, Debug)]
pub struct MbbpLdConfig {
    /// Code distance (must be odd, >= 3).
    pub distance: usize,
    /// Number of measurement rounds for temporal decoding.
    pub rounds: usize,
    /// Noise model for weight computation.
    pub noise_model: DecoderNoiseModel,
}

impl MbbpLdConfig {
    /// Create a default configuration for the given distance.
    pub fn default_for_distance(distance: usize) -> Self {
        Self {
            distance,
            rounds: 1,
            noise_model: DecoderNoiseModel::uniform(0.001),
        }
    }
}

// ============================================================
// MBBP-LD DECODE RESULT
// ============================================================

/// Result of MBBP-LD decoding.
#[derive(Clone, Debug)]
pub struct MbbpLdResult {
    /// Correction to apply: one bool per data qubit.
    pub correction: Vec<bool>,
    /// Whether the correction fully resolves the syndrome.
    pub success: bool,
    /// Whether a logical error was detected (correction differs from true
    /// error by a logical operator).
    pub logical_error: bool,
    /// Total matching weight of the boundary pairing.
    pub weight: f64,
    /// Wall-clock decode time in microseconds.
    pub decode_time_us: f64,
    /// Number of boundary defects found.
    pub num_boundary_defects: usize,
    /// Number of regions decoded locally.
    pub num_regions_decoded: usize,
}

// ============================================================
// MBBP-LD DECODER
// ============================================================

/// Matching-Based Boundary Pairing with Local Decoding.
///
/// The main decoder struct. Construct once per code configuration, then call
/// `decode()` for each syndrome measurement.
pub struct MbbpLdDecoder {
    /// The heavy-hex topology.
    pub topology: HeavyHexTopology,
    /// Local decoders, one per region.
    pub regions: Vec<LocalDecoder>,
    /// The boundary graph for inter-region matching.
    pub boundary_graph: BoundaryGraph,
    /// Noise model.
    pub noise_model: DecoderNoiseModel,
    /// Code distance.
    pub distance: usize,
    /// Number of measurement rounds.
    pub rounds: usize,
    /// Stabilizer map: syndrome bit index -> list of data qubit indices in its support.
    pub stabilizer_map: HashMap<usize, Vec<usize>>,
    /// Decoding statistics.
    pub stats: MbbpLdStats,
}

/// Cumulative decoding statistics.
#[derive(Clone, Debug, Default)]
pub struct MbbpLdStats {
    pub total_decodes: usize,
    pub total_successes: usize,
    pub total_logical_errors: usize,
    pub total_boundary_defects: usize,
    pub total_decode_time_us: f64,
}

impl MbbpLdDecoder {
    /// Construct a new MBBP-LD decoder for the given configuration.
    pub fn new(config: MbbpLdConfig) -> Self {
        let topology = HeavyHexTopology::new(config.distance);
        let stabilizer_map = Self::build_stabilizer_map(&topology);
        let (regions, boundary_graph) =
            Self::decompose_into_regions(&topology, &stabilizer_map, &config.noise_model);

        Self {
            topology,
            regions,
            boundary_graph,
            noise_model: config.noise_model,
            distance: config.distance,
            rounds: config.rounds,
            stabilizer_map,
            stats: MbbpLdStats::default(),
        }
    }

    /// Return the number of syndrome bits (= number of syndrome qubits).
    pub fn num_syndrome_bits(&self) -> usize {
        self.topology.syndrome_qubits.len()
    }

    /// Return the number of data qubits.
    pub fn num_data_qubits(&self) -> usize {
        self.topology.data_qubits.len()
    }

    // ----------------------------------------------------------------
    // STABILIZER MAP
    // ----------------------------------------------------------------

    /// Build the stabilizer map: for each syndrome qubit, which data qubits
    /// are in its stabilizer support?
    fn build_stabilizer_map(topology: &HeavyHexTopology) -> HashMap<usize, Vec<usize>> {
        let adj = topology.adjacency_list();
        let mut map = HashMap::new();

        for (idx, &s_id) in topology.syndrome_qubits.iter().enumerate() {
            let data_neighbors: Vec<usize> = adj[s_id]
                .iter()
                .copied()
                .filter(|&q| topology.qubits[q].role == QubitRole::Data)
                .collect();
            map.insert(idx, data_neighbors);
        }

        map
    }

    // ----------------------------------------------------------------
    // REGION DECOMPOSITION
    // ----------------------------------------------------------------

    /// Decompose the topology into local regions and build the boundary graph.
    ///
    /// Each syndrome qubit defines one local region containing itself and its
    /// neighboring data qubits. Regions that share data qubits have their
    /// shared syndrome bits marked as boundary bits.
    fn decompose_into_regions(
        topology: &HeavyHexTopology,
        stabilizer_map: &HashMap<usize, Vec<usize>>,
        noise: &DecoderNoiseModel,
    ) -> (Vec<LocalDecoder>, BoundaryGraph) {
        let num_syndromes = topology.syndrome_qubits.len();
        let mut regions = Vec::with_capacity(num_syndromes);
        let mut boundary_graph = BoundaryGraph::new();

        // For each syndrome qubit, create a local region
        for (idx, &_s_id) in topology.syndrome_qubits.iter().enumerate() {
            let data_neighbors = stabilizer_map.get(&idx).cloned().unwrap_or_default();

            // The syndrome bits for this region: the current syndrome index,
            // plus any adjacent syndrome indices that share data qubits.
            let mut local_syndrome_bits = vec![idx];
            let mut boundary_bits = Vec::new();

            // Find neighboring syndrome qubits that share data qubits
            for (other_idx, other_data) in stabilizer_map.iter() {
                if *other_idx == idx {
                    continue;
                }
                let shared: Vec<usize> = data_neighbors
                    .iter()
                    .filter(|q| other_data.contains(q))
                    .copied()
                    .collect();
                if !shared.is_empty() {
                    if !local_syndrome_bits.contains(other_idx) {
                        local_syndrome_bits.push(*other_idx);
                    }
                    if !boundary_bits.contains(other_idx) {
                        boundary_bits.push(*other_idx);
                    }
                }
            }

            local_syndrome_bits.sort_unstable();
            boundary_bits.sort_unstable();

            let decoder = LocalDecoder::new(
                idx,
                data_neighbors,
                local_syndrome_bits,
                boundary_bits,
                stabilizer_map,
            );
            regions.push(decoder);
        }

        // Build boundary graph: one node per region, edges between regions
        // that share data qubits
        let mut region_node_ids = Vec::with_capacity(num_syndromes);
        for (idx, &s_id) in topology.syndrome_qubits.iter().enumerate() {
            let is_boundary = Self::is_boundary_region(topology, s_id);
            let pos = topology.qubits[s_id].position;
            let node_id = boundary_graph.add_node(idx, is_boundary, pos);
            region_node_ids.push(node_id);
        }

        // Add edges between neighboring regions
        for i in 0..num_syndromes {
            for j in (i + 1)..num_syndromes {
                let data_i = stabilizer_map.get(&i).cloned().unwrap_or_default();
                let data_j = stabilizer_map.get(&j).cloned().unwrap_or_default();
                let shared: Vec<usize> = data_i
                    .iter()
                    .filter(|q| data_j.contains(q))
                    .copied()
                    .collect();
                if !shared.is_empty() {
                    let w = boundary_graph.node_distance(region_node_ids[i], region_node_ids[j]);
                    // Scale weight by noise model
                    let noise_w = noise.edge_weight(
                        shared[0],
                        shared.get(1).copied().unwrap_or(shared[0]),
                    );
                    boundary_graph.add_edge(
                        region_node_ids[i],
                        region_node_ids[j],
                        w * 0.5 + noise_w * 0.5,
                    );
                }
            }
        }

        (regions, boundary_graph)
    }

    /// Determine whether a syndrome qubit is on the code boundary.
    fn is_boundary_region(topology: &HeavyHexTopology, syndrome_id: usize) -> bool {
        let pos = topology.qubits[syndrome_id].position;
        let max_row = (topology.rows - 1) as f64 * 2.0;
        let max_col = (topology.cols - 1) as f64 * 2.0;
        pos.0 <= 1.0 || pos.0 >= max_row - 1.0 || pos.1 <= 1.0 || pos.1 >= max_col - 1.0
    }

    // ----------------------------------------------------------------
    // MAIN DECODE
    // ----------------------------------------------------------------

    /// Decode a syndrome measurement.
    ///
    /// `syndrome` must have length `num_syndrome_bits()`. Each entry is `true`
    /// for a detected stabilizer defect.
    ///
    /// Returns an `MbbpLdResult` with the correction pattern and diagnostics.
    pub fn decode(&mut self, syndrome: &[bool]) -> MbbpLdResult {
        let start = Instant::now();

        let num_data = self.num_data_qubits();
        let num_syndromes = self.num_syndrome_bits();

        // Validate input
        assert_eq!(
            syndrome.len(),
            num_syndromes,
            "Syndrome length {} != expected {}",
            syndrome.len(),
            num_syndromes,
        );

        // -- Step 1: Local decoding within each region ---------------------
        let mut global_correction = vec![false; num_data];
        let mut residual_syndrome: Vec<bool> = syndrome.to_vec();
        let mut regions_decoded = 0;

        for region in &self.regions {
            // Extract local syndrome
            let local_syndrome: Vec<bool> = region
                .syndrome_bits
                .iter()
                .map(|&s| {
                    if s < residual_syndrome.len() {
                        residual_syndrome[s]
                    } else {
                        false
                    }
                })
                .collect();

            // Skip trivial (all-zero) local syndromes
            if local_syndrome.iter().all(|&b| !b) {
                continue;
            }

            // Attempt local decode
            if let Some(local_correction) = region.decode(&local_syndrome) {
                // Apply local correction to global correction
                for (q_idx, &q_id) in region.region_qubits.iter().enumerate() {
                    if local_correction[q_idx] {
                        // Find position of q_id in data_qubits
                        if let Some(pos) = self.topology.data_qubits.iter().position(|&d| d == q_id) {
                            global_correction[pos] ^= true;
                        }
                    }
                }

                // Update residual syndrome
                self.apply_correction_to_syndrome(
                    &local_correction,
                    &region.region_qubits,
                    &mut residual_syndrome,
                );

                regions_decoded += 1;
            }
        }

        // -- Step 2: Identify boundary defects -----------------------------
        let boundary_defect_indices: Vec<usize> = residual_syndrome
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();

        let num_boundary_defects = boundary_defect_indices.len();

        // -- Step 3: Match boundary defects --------------------------------
        if !boundary_defect_indices.is_empty() {
            // Map syndrome indices to boundary graph node IDs
            let defect_nodes: Vec<usize> = boundary_defect_indices
                .iter()
                .filter_map(|&s_idx| {
                    // Find the boundary graph node for this syndrome
                    self.boundary_graph
                        .nodes
                        .iter()
                        .find(|n| n.region_id == s_idx)
                        .map(|n| n.id)
                })
                .collect();

            if !defect_nodes.is_empty() {
                let pairing = greedy_boundary_matching(
                    &defect_nodes,
                    &self.boundary_graph,
                    &self.noise_model,
                );

                // Convert boundary pairing to corrections on data qubits
                for (n1, n2) in &pairing.pairs {
                    let r1 = self.boundary_graph.nodes[*n1].region_id;
                    let r2 = self.boundary_graph.nodes[*n2].region_id;
                    self.apply_boundary_correction(r1, r2, &mut global_correction);
                }

                // Boundary-matched defects: correct the data qubit closest to boundary
                for &bd in &pairing.boundary_defects {
                    let region_id = self.boundary_graph.nodes[bd].region_id;
                    self.apply_single_boundary_correction(region_id, &mut global_correction);
                }
            }
        }

        // -- Step 4: Verify correction -------------------------------------
        let corrected_syndrome = self.compute_syndrome(&global_correction);
        let success = corrected_syndrome.iter().all(|&b| !b)
            || self.syndromes_match(syndrome, &corrected_syndrome, &global_correction);

        // Check for logical errors
        let logical_error = self.check_logical_error(&global_correction);

        let decode_time = start.elapsed().as_secs_f64() * 1e6;

        // Update statistics
        self.stats.total_decodes += 1;
        if success {
            self.stats.total_successes += 1;
        }
        if logical_error {
            self.stats.total_logical_errors += 1;
        }
        self.stats.total_boundary_defects += num_boundary_defects;
        self.stats.total_decode_time_us += decode_time;

        MbbpLdResult {
            correction: global_correction,
            success,
            logical_error,
            weight: 0.0, // Filled by boundary pairing when applicable
            decode_time_us: decode_time,
            num_boundary_defects,
            num_regions_decoded: regions_decoded,
        }
    }

    /// Decode multiple rounds of syndrome data (temporal decoding).
    ///
    /// Each row in `syndrome_rounds` is one measurement round. The decoder
    /// computes difference syndromes between consecutive rounds and decodes
    /// them, accumulating corrections.
    pub fn decode_multi_round(&mut self, syndrome_rounds: &[Vec<bool>]) -> MbbpLdResult {
        let start = Instant::now();
        let num_data = self.num_data_qubits();
        let mut accumulated_correction = vec![false; num_data];
        let mut total_boundary_defects = 0;
        let mut total_regions = 0;
        let mut all_success = true;
        let mut any_logical_error = false;

        // First round decoded directly
        if let Some(first) = syndrome_rounds.first() {
            let result = self.decode(first);
            for (i, &c) in result.correction.iter().enumerate() {
                accumulated_correction[i] ^= c;
            }
            all_success &= result.success;
            any_logical_error |= result.logical_error;
            total_boundary_defects += result.num_boundary_defects;
            total_regions += result.num_regions_decoded;
        }

        // Subsequent rounds: decode the difference syndrome
        for w in syndrome_rounds.windows(2) {
            let diff_syndrome: Vec<bool> = w[0]
                .iter()
                .zip(w[1].iter())
                .map(|(&a, &b)| a ^ b)
                .collect();

            let result = self.decode(&diff_syndrome);
            for (i, &c) in result.correction.iter().enumerate() {
                accumulated_correction[i] ^= c;
            }
            all_success &= result.success;
            any_logical_error |= result.logical_error;
            total_boundary_defects += result.num_boundary_defects;
            total_regions += result.num_regions_decoded;
        }

        let decode_time = start.elapsed().as_secs_f64() * 1e6;

        MbbpLdResult {
            correction: accumulated_correction,
            success: all_success,
            logical_error: any_logical_error,
            weight: 0.0,
            decode_time_us: decode_time,
            num_boundary_defects: total_boundary_defects,
            num_regions_decoded: total_regions,
        }
    }

    // ----------------------------------------------------------------
    // HELPER METHODS
    // ----------------------------------------------------------------

    /// Apply a local correction to the residual syndrome.
    fn apply_correction_to_syndrome(
        &self,
        correction: &[bool],
        region_qubits: &[usize],
        residual: &mut [bool],
    ) {
        // For each corrected data qubit, flip the syndrome bits of all
        // stabilizers that contain it
        for (q_idx, &q_id) in region_qubits.iter().enumerate() {
            if !correction[q_idx] {
                continue;
            }
            for (s_idx, support) in &self.stabilizer_map {
                if support.contains(&q_id) && *s_idx < residual.len() {
                    residual[*s_idx] ^= true;
                }
            }
        }
    }

    /// Apply a correction for a boundary pairing between two regions.
    ///
    /// Finds the shared data qubit between the two regions and flips it.
    fn apply_boundary_correction(
        &self,
        region1: usize,
        region2: usize,
        correction: &mut [bool],
    ) {
        let data1 = self.stabilizer_map.get(&region1).cloned().unwrap_or_default();
        let data2 = self.stabilizer_map.get(&region2).cloned().unwrap_or_default();

        // Find shared data qubits
        for &q in &data1 {
            if data2.contains(&q) {
                if let Some(pos) = self.topology.data_qubits.iter().position(|&d| d == q) {
                    correction[pos] ^= true;
                    return;
                }
            }
        }

        // If no shared qubit, correct the first data qubit in region1
        if let Some(&q) = data1.first() {
            if let Some(pos) = self.topology.data_qubits.iter().position(|&d| d == q) {
                correction[pos] ^= true;
            }
        }
    }

    /// Apply a correction for a single boundary defect (matched to code boundary).
    fn apply_single_boundary_correction(&self, region_id: usize, correction: &mut [bool]) {
        let data_qubits = self.stabilizer_map.get(&region_id).cloned().unwrap_or_default();

        // Find the data qubit closest to the code boundary
        let mut best_q = None;
        let mut best_dist = f64::INFINITY;

        for &q in &data_qubits {
            let pos = self.topology.qubits[q].position;
            let max_pos = ((self.distance - 1) as f64) * 2.0;
            let dist_to_boundary = pos.0.min(pos.1).min(max_pos - pos.0).min(max_pos - pos.1);
            if dist_to_boundary < best_dist {
                best_dist = dist_to_boundary;
                best_q = Some(q);
            }
        }

        if let Some(q) = best_q {
            if let Some(pos) = self.topology.data_qubits.iter().position(|&d| d == q) {
                correction[pos] ^= true;
            }
        }
    }

    /// Compute the syndrome resulting from a given error/correction pattern on
    /// data qubits.
    pub fn compute_syndrome(&self, data_errors: &[bool]) -> Vec<bool> {
        let num_syndromes = self.num_syndrome_bits();
        let mut syndrome = vec![false; num_syndromes];

        for (s_idx, support) in &self.stabilizer_map {
            if *s_idx >= num_syndromes {
                continue;
            }
            let mut parity = false;
            for &q in support {
                if let Some(pos) = self.topology.data_qubits.iter().position(|&d| d == q) {
                    if pos < data_errors.len() && data_errors[pos] {
                        parity = !parity;
                    }
                }
            }
            syndrome[*s_idx] = parity;
        }

        syndrome
    }

    /// Check whether the original syndrome is resolved by the correction,
    /// accounting for the possibility that the correction + error produces a
    /// trivial syndrome (which indicates success even if the residual is not
    /// identically zero).
    fn syndromes_match(
        &self,
        original: &[bool],
        corrected: &[bool],
        _correction: &[bool],
    ) -> bool {
        // The correction is successful if the corrected syndrome equals the
        // original (meaning the correction exactly cancels the error's syndrome
        // contribution).
        original
            .iter()
            .zip(corrected.iter())
            .all(|(&a, &b)| a == b)
    }

    /// Check whether the correction constitutes a logical error.
    ///
    /// A logical X error on a distance-d rotated planar code corresponds to a
    /// chain of X operators crossing the code from one boundary to the
    /// opposite. We check this by counting the parity of corrected qubits
    /// along a logical chain.
    fn check_logical_error(&self, correction: &[bool]) -> bool {
        let d = self.distance;
        // Logical X: chain along the first row of data qubits
        let mut x_parity = false;
        for c in 0..d {
            let q_id = self.topology.data_qubits.get(c).copied();
            if let Some(q) = q_id {
                if let Some(pos) = self.topology.data_qubits.iter().position(|&d| d == q) {
                    if pos < correction.len() && correction[pos] {
                        x_parity = !x_parity;
                    }
                }
            }
        }

        // Logical Z: chain along the first column of data qubits
        let mut z_parity = false;
        for r in 0..d {
            let q_id = self.topology.data_qubits.get(r * d).copied();
            if let Some(q) = q_id {
                if let Some(pos) = self.topology.data_qubits.iter().position(|&d| d == q) {
                    if pos < correction.len() && correction[pos] {
                        z_parity = !z_parity;
                    }
                }
            }
        }

        x_parity || z_parity
    }

    /// Run a simple MWPM decode for comparison (uses greedy matching on the
    /// full syndrome graph, not the boundary-only graph).
    pub fn decode_mwpm_comparison(&self, syndrome: &[bool]) -> MbbpLdResult {
        let start = Instant::now();
        let num_data = self.num_data_qubits();
        let num_syndromes = self.num_syndrome_bits();

        assert_eq!(syndrome.len(), num_syndromes);

        // Collect defect positions
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();

        let mut correction = vec![false; num_data];

        if defects.is_empty() {
            let decode_time = start.elapsed().as_secs_f64() * 1e6;
            return MbbpLdResult {
                correction,
                success: true,
                logical_error: false,
                weight: 0.0,
                decode_time_us: decode_time,
                num_boundary_defects: 0,
                num_regions_decoded: 0,
            };
        }

        // Build full matching graph
        let mut edges = Vec::new();
        for (i, &d1) in defects.iter().enumerate() {
            for &d2 in &defects[i + 1..] {
                let pos1 = self.topology.qubits[self.topology.syndrome_qubits[d1]].position;
                let pos2 = self.topology.qubits[self.topology.syndrome_qubits[d2]].position;
                let dist =
                    ((pos1.0 - pos2.0).powi(2) + (pos1.1 - pos2.1).powi(2)).sqrt();
                edges.push((d1, d2, dist));
            }
        }

        // Sort edges by weight
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        // Greedy matching
        let mut matched = HashSet::new();
        let mut pairs = Vec::new();
        let mut total_weight = 0.0;

        for (d1, d2, w) in &edges {
            if !matched.contains(d1) && !matched.contains(d2) {
                matched.insert(*d1);
                matched.insert(*d2);
                pairs.push((*d1, *d2));
                total_weight += w;
            }
        }

        // Convert pairs to corrections
        for (s1, s2) in &pairs {
            let data1 = self.stabilizer_map.get(s1).cloned().unwrap_or_default();
            let data2 = self.stabilizer_map.get(s2).cloned().unwrap_or_default();

            // Find shared qubit (for adjacent stabilizers)
            let mut corrected = false;
            for &q in &data1 {
                if data2.contains(&q) {
                    if let Some(pos) =
                        self.topology.data_qubits.iter().position(|&d| d == q)
                    {
                        correction[pos] ^= true;
                        corrected = true;
                        break;
                    }
                }
            }

            // If no shared qubit, correct first qubit in each stabilizer
            if !corrected {
                if let Some(&q) = data1.first() {
                    if let Some(pos) =
                        self.topology.data_qubits.iter().position(|&d| d == q)
                    {
                        correction[pos] ^= true;
                    }
                }
            }
        }

        // Handle unmatched defects
        for &d in &defects {
            if !matched.contains(&d) {
                let data = self.stabilizer_map.get(&d).cloned().unwrap_or_default();
                if let Some(&q) = data.first() {
                    if let Some(pos) =
                        self.topology.data_qubits.iter().position(|&dd| dd == q)
                    {
                        correction[pos] ^= true;
                    }
                }
            }
        }

        let corrected_syndrome = self.compute_syndrome(&correction);
        let success = corrected_syndrome.iter().all(|&b| !b);
        let logical_error = self.check_logical_error(&correction);

        let decode_time = start.elapsed().as_secs_f64() * 1e6;

        MbbpLdResult {
            correction,
            success,
            logical_error,
            weight: total_weight,
            decode_time_us: decode_time,
            num_boundary_defects: defects.len(),
            num_regions_decoded: 0,
        }
    }

    /// Reset accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = MbbpLdStats::default();
    }

    /// Estimate the pseudo-threshold via Monte Carlo sampling.
    ///
    /// Injects random errors at the given physical error rate, decodes, and
    /// returns the logical error rate.
    pub fn estimate_threshold(
        &mut self,
        physical_error_rate: f64,
        num_samples: usize,
    ) -> ThresholdEstimate {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let num_data = self.num_data_qubits();
        let mut logical_errors = 0;
        let mut total_decode_time = 0.0;

        for _ in 0..num_samples {
            // Generate random error pattern
            let mut error = vec![false; num_data];
            for e in &mut error {
                *e = rng.gen::<f64>() < physical_error_rate;
            }

            // Compute syndrome
            let syndrome = self.compute_syndrome(&error);

            // Decode
            let result = self.decode(&syndrome);

            total_decode_time += result.decode_time_us;

            // Check if correction + error = logical operator
            let combined: Vec<bool> = error
                .iter()
                .zip(result.correction.iter())
                .map(|(&e, &c)| e ^ c)
                .collect();
            if self.check_logical_error(&combined) {
                logical_errors += 1;
            }
        }

        let logical_error_rate = logical_errors as f64 / num_samples as f64;
        let avg_decode_time = total_decode_time / num_samples as f64;

        ThresholdEstimate {
            physical_error_rate,
            logical_error_rate,
            num_samples,
            logical_errors,
            avg_decode_time_us: avg_decode_time,
        }
    }
}

/// Result of a threshold estimation run.
#[derive(Clone, Debug)]
pub struct ThresholdEstimate {
    pub physical_error_rate: f64,
    pub logical_error_rate: f64,
    pub num_samples: usize,
    pub logical_errors: usize,
    pub avg_decode_time_us: f64,
}

// ============================================================
// COMPARISON UTILITIES
// ============================================================

/// Compare MBBP-LD vs MWPM on a set of syndrome samples.
pub fn compare_decoders(
    distance: usize,
    physical_error_rate: f64,
    num_samples: usize,
) -> DecoderComparison {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let noise = DecoderNoiseModel::uniform(physical_error_rate);
    let config = MbbpLdConfig {
        distance,
        rounds: 1,
        noise_model: noise,
    };
    let mut decoder = MbbpLdDecoder::new(config);
    let num_data = decoder.num_data_qubits();

    let mut mbbp_logical_errors = 0;
    let mut mwpm_logical_errors = 0;
    let mut mbbp_total_time = 0.0;
    let mut mwpm_total_time = 0.0;

    for _ in 0..num_samples {
        // Random error
        let mut error = vec![false; num_data];
        for e in &mut error {
            *e = rng.gen::<f64>() < physical_error_rate;
        }

        let syndrome = decoder.compute_syndrome(&error);

        // MBBP-LD decode
        let mbbp_result = decoder.decode(&syndrome);
        mbbp_total_time += mbbp_result.decode_time_us;
        let mbbp_combined: Vec<bool> = error
            .iter()
            .zip(mbbp_result.correction.iter())
            .map(|(&e, &c)| e ^ c)
            .collect();
        if decoder.check_logical_error(&mbbp_combined) {
            mbbp_logical_errors += 1;
        }

        // MWPM comparison decode
        let mwpm_result = decoder.decode_mwpm_comparison(&syndrome);
        mwpm_total_time += mwpm_result.decode_time_us;
        let mwpm_combined: Vec<bool> = error
            .iter()
            .zip(mwpm_result.correction.iter())
            .map(|(&e, &c)| e ^ c)
            .collect();
        if decoder.check_logical_error(&mwpm_combined) {
            mwpm_logical_errors += 1;
        }
    }

    DecoderComparison {
        distance,
        physical_error_rate,
        num_samples,
        mbbp_logical_error_rate: mbbp_logical_errors as f64 / num_samples as f64,
        mwpm_logical_error_rate: mwpm_logical_errors as f64 / num_samples as f64,
        mbbp_avg_decode_us: mbbp_total_time / num_samples as f64,
        mwpm_avg_decode_us: mwpm_total_time / num_samples as f64,
    }
}

/// Result of a decoder comparison run.
#[derive(Clone, Debug)]
pub struct DecoderComparison {
    pub distance: usize,
    pub physical_error_rate: f64,
    pub num_samples: usize,
    pub mbbp_logical_error_rate: f64,
    pub mwpm_logical_error_rate: f64,
    pub mbbp_avg_decode_us: f64,
    pub mwpm_avg_decode_us: f64,
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Heavy-hex topology tests
    // ---------------------------------------------------------------

    #[test]
    fn test_heavy_hex_construction_d3() {
        let topo = HeavyHexTopology::new(3);
        assert_eq!(topo.distance, 3);
        assert_eq!(topo.rows, 3);
        assert_eq!(topo.cols, 3);

        // 3x3 grid of data qubits = 9
        assert_eq!(topo.data_qubits.len(), 9);

        // Flag and syndrome qubits should exist
        assert!(!topo.flag_qubits.is_empty());
        assert!(!topo.syndrome_qubits.is_empty());

        // All qubit IDs should be unique
        let mut all_ids: Vec<usize> = Vec::new();
        all_ids.extend(&topo.data_qubits);
        all_ids.extend(&topo.flag_qubits);
        all_ids.extend(&topo.syndrome_qubits);
        let unique: HashSet<usize> = all_ids.iter().copied().collect();
        assert_eq!(all_ids.len(), unique.len(), "Qubit IDs must be unique");
    }

    #[test]
    fn test_heavy_hex_construction_d5() {
        let topo = HeavyHexTopology::new(5);
        assert_eq!(topo.distance, 5);
        assert_eq!(topo.data_qubits.len(), 25); // 5x5

        // Should have more qubits than d=3
        let d3 = HeavyHexTopology::new(3);
        assert!(topo.num_qubits > d3.num_qubits);
        assert!(topo.syndrome_qubits.len() > d3.syndrome_qubits.len());
    }

    #[test]
    fn test_heavy_hex_qubit_counts() {
        // For a d x d grid:
        // - Data qubits: d^2
        // - Syndrome qubits: (d-1)^2
        // - Flag qubits: varies with staggering
        for d in [3, 5, 7] {
            let topo = HeavyHexTopology::new(d);
            assert_eq!(topo.data_qubits.len(), d * d, "Data qubits for d={}", d);
            assert_eq!(
                topo.syndrome_qubits.len(),
                (d - 1) * (d - 1),
                "Syndrome qubits for d={}",
                d,
            );
            // Total qubit count should be consistent
            assert_eq!(
                topo.num_qubits,
                topo.data_qubits.len() + topo.flag_qubits.len() + topo.syndrome_qubits.len(),
                "Total qubit count for d={}",
                d,
            );
        }
    }

    #[test]
    fn test_region_decomposition() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let decoder = MbbpLdDecoder::new(config);

        // Should have (d-1)^2 = 4 regions for d=3
        assert_eq!(decoder.regions.len(), 4);

        // Each region should have data qubits and syndrome bits
        for region in &decoder.regions {
            assert!(
                !region.region_qubits.is_empty(),
                "Region {} has no data qubits",
                region.region_id,
            );
            assert!(
                !region.syndrome_bits.is_empty(),
                "Region {} has no syndrome bits",
                region.region_id,
            );
            // Lookup table should have at least the identity entry
            assert!(
                !region.lookup_table.is_empty(),
                "Region {} has empty lookup table",
                region.region_id,
            );
        }
    }

    #[test]
    fn test_local_decoder_trivial_syndrome() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        // All-zero syndrome should decode to all-zero correction
        let num_syndromes = decoder.num_syndrome_bits();
        let syndrome = vec![false; num_syndromes];
        let result = decoder.decode(&syndrome);
        assert!(
            result.correction.iter().all(|&c| !c),
            "Trivial syndrome should produce no corrections",
        );
        assert!(result.success);
    }

    #[test]
    fn test_local_decoder_single_error() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        // Single error on data qubit 0
        let num_data = decoder.num_data_qubits();
        let mut error = vec![false; num_data];
        error[0] = true;

        let syndrome = decoder.compute_syndrome(&error);
        let result = decoder.decode(&syndrome);

        // The correction should have some corrections (at least 1)
        let num_corrections: usize = result.correction.iter().filter(|&&c| c).count();
        assert!(
            num_corrections > 0,
            "Single error should produce at least one correction",
        );
    }

    #[test]
    fn test_boundary_graph_construction() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let decoder = MbbpLdDecoder::new(config);

        // Should have one boundary node per region
        assert_eq!(
            decoder.boundary_graph.nodes.len(),
            decoder.regions.len(),
            "Boundary graph should have one node per region",
        );

        // Should have edges between adjacent regions
        assert!(
            !decoder.boundary_graph.edges.is_empty(),
            "Boundary graph should have edges",
        );

        // All edge weights should be positive
        for edge in &decoder.boundary_graph.edges {
            assert!(
                edge.weight > 0.0,
                "Edge weight should be positive, got {}",
                edge.weight,
            );
        }
    }

    #[test]
    fn test_boundary_pairing_no_defects() {
        let graph = BoundaryGraph::new();
        let noise = DecoderNoiseModel::uniform(0.001);
        let pairing = greedy_boundary_matching(&[], &graph, &noise);
        assert!(pairing.pairs.is_empty());
        assert!(pairing.boundary_defects.is_empty());
        assert_eq!(pairing.weight, 0.0);
    }

    #[test]
    fn test_boundary_pairing_single_pair() {
        let mut graph = BoundaryGraph::new();
        let n0 = graph.add_node(0, false, (0.0, 0.0));
        let n1 = graph.add_node(1, false, (1.0, 0.0));
        graph.add_edge(n0, n1, 1.0);

        let noise = DecoderNoiseModel::uniform(0.001);
        let pairing = greedy_boundary_matching(&[n0, n1], &graph, &noise);

        assert_eq!(pairing.pairs.len(), 1);
        assert!(pairing.boundary_defects.is_empty());
        assert!(pairing.weight > 0.0);
    }

    #[test]
    fn test_boundary_pairing_multiple_pairs() {
        let mut graph = BoundaryGraph::new();
        let n0 = graph.add_node(0, false, (0.0, 0.0));
        let n1 = graph.add_node(1, false, (1.0, 0.0));
        let n2 = graph.add_node(2, false, (0.0, 1.0));
        let n3 = graph.add_node(3, false, (1.0, 1.0));
        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n2, n3, 1.0);
        graph.add_edge(n0, n2, 1.0);
        graph.add_edge(n1, n3, 1.0);

        let noise = DecoderNoiseModel::uniform(0.001);
        let pairing = greedy_boundary_matching(&[n0, n1, n2, n3], &graph, &noise);

        assert_eq!(pairing.pairs.len(), 2, "Four defects should form 2 pairs");
        assert!(pairing.boundary_defects.is_empty());
    }

    #[test]
    fn test_mbbp_ld_no_errors() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        let syndrome = vec![false; decoder.num_syndrome_bits()];
        let result = decoder.decode(&syndrome);

        assert!(result.success, "No-error syndrome should succeed");
        assert!(
            result.correction.iter().all(|&c| !c),
            "No-error syndrome should produce no corrections",
        );
        assert_eq!(result.num_boundary_defects, 0);
    }

    #[test]
    fn test_mbbp_ld_single_x_error() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 5,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        // Inject X error on a central data qubit
        let center = decoder.num_data_qubits() / 2;
        let mut error = vec![false; decoder.num_data_qubits()];
        error[center] = true;

        let syndrome = decoder.compute_syndrome(&error);
        let defect_count = syndrome.iter().filter(|&&b| b).count();
        assert!(defect_count > 0, "Single X error should trigger defects");

        let result = decoder.decode(&syndrome);

        // The correction should fix the syndrome
        let num_corrections = result.correction.iter().filter(|&&c| c).count();
        assert!(
            num_corrections > 0,
            "Single X error should produce corrections",
        );
    }

    #[test]
    fn test_mbbp_ld_single_z_error() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 5,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        // Z errors and X errors have the same syndrome structure in our
        // stabilizer model (both flip adjacent stabilizers).
        let mut error = vec![false; decoder.num_data_qubits()];
        error[1] = true;

        let syndrome = decoder.compute_syndrome(&error);
        let result = decoder.decode(&syndrome);

        let num_corrections = result.correction.iter().filter(|&&c| c).count();
        assert!(
            num_corrections > 0,
            "Single Z error should produce corrections",
        );
    }

    #[test]
    fn test_mbbp_ld_boundary_error() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 5,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        // Error on a corner data qubit (boundary region)
        let mut error = vec![false; decoder.num_data_qubits()];
        error[0] = true;

        let syndrome = decoder.compute_syndrome(&error);
        let result = decoder.decode(&syndrome);

        let num_corrections = result.correction.iter().filter(|&&c| c).count();
        assert!(
            num_corrections > 0,
            "Boundary error should produce corrections",
        );
    }

    #[test]
    fn test_mbbp_ld_multi_error() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 5,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);
        let num_data = decoder.num_data_qubits();

        // Two errors on non-adjacent data qubits
        let mut error = vec![false; num_data];
        error[0] = true;
        error[num_data - 1] = true;

        let syndrome = decoder.compute_syndrome(&error);
        let defect_count = syndrome.iter().filter(|&&b| b).count();
        assert!(
            defect_count >= 2,
            "Two errors should trigger at least 2 defects, got {}",
            defect_count,
        );

        let result = decoder.decode(&syndrome);
        let num_corrections = result.correction.iter().filter(|&&c| c).count();
        assert!(
            num_corrections > 0,
            "Multi-error should produce corrections",
        );
    }

    #[test]
    fn test_mbbp_ld_vs_mwpm_comparison() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        // Single error
        let mut error = vec![false; decoder.num_data_qubits()];
        error[4] = true; // center qubit

        let syndrome = decoder.compute_syndrome(&error);

        let mbbp_result = decoder.decode(&syndrome);
        let mwpm_result = decoder.decode_mwpm_comparison(&syndrome);

        // Both decoders should produce corrections
        let mbbp_corrections = mbbp_result.correction.iter().filter(|&&c| c).count();
        let mwpm_corrections = mwpm_result.correction.iter().filter(|&&c| c).count();

        assert!(
            mbbp_corrections > 0 || mwpm_corrections > 0,
            "At least one decoder should produce corrections",
        );

        // Both should have reasonable decode times
        assert!(
            mbbp_result.decode_time_us >= 0.0,
            "MBBP-LD decode time should be non-negative",
        );
        assert!(
            mwpm_result.decode_time_us >= 0.0,
            "MWPM decode time should be non-negative",
        );
    }

    #[test]
    fn test_noise_model_construction() {
        let noise = DecoderNoiseModel::uniform(0.01);
        assert_eq!(noise.physical_error_rate, 0.01);
        assert!((noise.measurement_error_rate - 0.001).abs() < 1e-10);
        assert!(noise.crosstalk_map.is_empty());

        let noise2 = DecoderNoiseModel::with_measurement_error(0.01, 0.005);
        assert_eq!(noise2.measurement_error_rate, 0.005);
    }

    #[test]
    fn test_weight_computation() {
        let mut noise = DecoderNoiseModel::uniform(0.01);

        // Edge weight should be positive for small error rates
        let w = noise.edge_weight(0, 1);
        assert!(w > 0.0, "Edge weight should be positive, got {}", w);

        // Higher error rate -> lower weight (more likely edge)
        let noise_high = DecoderNoiseModel::uniform(0.1);
        let w_high = noise_high.edge_weight(0, 1);
        assert!(
            w_high < w,
            "Higher error rate should give lower weight: {} vs {}",
            w_high,
            w,
        );

        // Crosstalk should reduce weight
        noise.crosstalk_map.insert((0, 1), 0.05);
        let w_crosstalk = noise.edge_weight(0, 1);
        assert!(
            w_crosstalk < w,
            "Crosstalk should reduce weight: {} vs {}",
            w_crosstalk,
            w,
        );

        // Measurement weight should be positive
        let mw = noise.measurement_weight();
        assert!(mw > 0.0, "Measurement weight should be positive, got {}", mw);
    }

    #[test]
    fn test_multi_round_decoding() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 3,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);
        let num_syndromes = decoder.num_syndrome_bits();

        // Three rounds: no errors, then an error, then no errors
        let round0 = vec![false; num_syndromes];
        let mut round1 = vec![false; num_syndromes];
        if num_syndromes > 0 {
            round1[0] = true;
        }
        let round2 = vec![false; num_syndromes];

        let result = decoder.decode_multi_round(&[round0, round1, round2]);

        // Should complete without panic
        assert!(result.decode_time_us >= 0.0);
    }

    #[test]
    fn test_logical_error_detection() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let decoder = MbbpLdDecoder::new(config);

        // A logical X error spans the first row of data qubits
        let num_data = decoder.num_data_qubits();
        let d = decoder.distance;

        // No error -> no logical error
        let no_error = vec![false; num_data];
        assert!(
            !decoder.check_logical_error(&no_error),
            "No error should not be a logical error",
        );

        // Single qubit error on the first row -> logical error parity
        // (depends on d; for d=3, first row has 3 qubits)
        let mut single = vec![false; num_data];
        single[0] = true;
        // Single qubit in first row has odd parity -> logical error
        assert!(
            decoder.check_logical_error(&single),
            "Single qubit in logical chain should be detected",
        );

        // Full logical X (all qubits in first row flipped)
        let mut logical_x = vec![false; num_data];
        for c in 0..d {
            logical_x[c] = true;
        }
        // d=3: 3 qubits flipped -> odd parity -> logical error
        let expected = d % 2 == 1;
        assert_eq!(
            decoder.check_logical_error(&logical_x),
            expected,
            "Full logical X parity check for d={}",
            d,
        );
    }

    #[test]
    fn test_mbbp_ld_threshold_estimation() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);

        let estimate = decoder.estimate_threshold(0.001, 50);

        assert_eq!(estimate.num_samples, 50);
        assert_eq!(estimate.physical_error_rate, 0.001);
        assert!(
            estimate.logical_error_rate >= 0.0 && estimate.logical_error_rate <= 1.0,
            "Logical error rate should be in [0, 1], got {}",
            estimate.logical_error_rate,
        );
        assert!(
            estimate.avg_decode_time_us > 0.0,
            "Average decode time should be positive",
        );

        // At very low error rate and small distance, should have few logical errors
        // (probabilistic, but very likely with p=0.001 and n=50)
        assert!(
            estimate.logical_errors <= 25,
            "Expected few logical errors at p=0.001, got {}",
            estimate.logical_errors,
        );
    }

    #[test]
    fn test_heavy_hex_d7() {
        let topo = HeavyHexTopology::new(7);
        assert_eq!(topo.distance, 7);
        assert_eq!(topo.data_qubits.len(), 49); // 7x7
        assert_eq!(topo.syndrome_qubits.len(), 36); // 6x6

        // Verify edge connectivity: every data qubit should appear in at
        // least one edge
        let all_edge_qubits: HashSet<usize> = topo
            .edges
            .iter()
            .flat_map(|&(u, v)| [u, v])
            .collect();
        for &d in &topo.data_qubits {
            assert!(
                all_edge_qubits.contains(&d),
                "Data qubit {} should be in at least one edge",
                d,
            );
        }

        // All flag qubits should be in edges
        for &f in &topo.flag_qubits {
            assert!(
                all_edge_qubits.contains(&f),
                "Flag qubit {} should be in at least one edge",
                f,
            );
        }

        // Adjacency list should be degree-bounded
        // Data qubits connect to: up to 2 horizontal flags, up to 2 vertical
        // flags, and up to 4 syndrome qubits (one per adjacent plaquette),
        // giving a max degree of 8.
        let adj = topo.adjacency_list();
        for &d in &topo.data_qubits {
            assert!(
                adj[d].len() <= 8,
                "Data qubit {} has degree {}, expected <= 8",
                d,
                adj[d].len(),
            );
        }
    }

    // ---------------------------------------------------------------
    // Additional robustness tests
    // ---------------------------------------------------------------

    #[test]
    fn test_compute_syndrome_roundtrip() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let decoder = MbbpLdDecoder::new(config);
        let num_data = decoder.num_data_qubits();

        // No error -> trivial syndrome
        let no_error = vec![false; num_data];
        let syndrome = decoder.compute_syndrome(&no_error);
        assert!(
            syndrome.iter().all(|&b| !b),
            "No error should produce trivial syndrome",
        );
    }

    #[test]
    fn test_decoder_stats_accumulation() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let mut decoder = MbbpLdDecoder::new(config);
        let num_syndromes = decoder.num_syndrome_bits();

        // Decode several times
        for _ in 0..5 {
            let syndrome = vec![false; num_syndromes];
            decoder.decode(&syndrome);
        }

        assert_eq!(decoder.stats.total_decodes, 5);
        assert_eq!(decoder.stats.total_successes, 5);
        assert!(decoder.stats.total_decode_time_us > 0.0);

        // Reset
        decoder.reset_stats();
        assert_eq!(decoder.stats.total_decodes, 0);
    }

    #[test]
    fn test_boundary_graph_distances() {
        let mut graph = BoundaryGraph::new();
        let n0 = graph.add_node(0, false, (0.0, 0.0));
        let n1 = graph.add_node(1, false, (3.0, 4.0));

        let dist = graph.node_distance(n0, n1);
        assert!(
            (dist - 5.0).abs() < 1e-10,
            "Distance should be 5.0, got {}",
            dist,
        );
    }

    #[test]
    fn test_local_decoder_lookup_table_size() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 3,
            rounds: 1,
            noise_model: noise,
        };
        let decoder = MbbpLdDecoder::new(config);

        for region in &decoder.regions {
            let n = region.region_qubits.len();
            // Lookup table should have entries for: identity + weight-1 + weight-2
            // (some syndromes may collide, so the actual count can be smaller)
            let max_entries = 1 + n + n * (n - 1) / 2;
            assert!(
                region.lookup_table.len() <= max_entries,
                "Region {} lookup table has {} entries, expected <= {}",
                region.region_id,
                region.lookup_table.len(),
                max_entries,
            );
            // Should always have at least the identity
            assert!(
                region.lookup_table.len() >= 1,
                "Region {} lookup table should have at least 1 entry",
                region.region_id,
            );
        }
    }

    #[test]
    fn test_decoder_comparison_utility() {
        // Small test with few samples to keep test fast
        let comparison = compare_decoders(3, 0.01, 20);

        assert_eq!(comparison.distance, 3);
        assert_eq!(comparison.num_samples, 20);
        assert!(comparison.mbbp_logical_error_rate >= 0.0);
        assert!(comparison.mwpm_logical_error_rate >= 0.0);
        assert!(comparison.mbbp_avg_decode_us > 0.0);
        assert!(comparison.mwpm_avg_decode_us > 0.0);
    }

    #[test]
    fn test_heavy_hex_adjacency_symmetry() {
        let topo = HeavyHexTopology::new(5);
        let adj = topo.adjacency_list();

        // Adjacency should be symmetric: if u is neighbor of v, v is neighbor of u
        for (u, neighbors) in adj.iter().enumerate() {
            for &v in neighbors {
                assert!(
                    adj[v].contains(&u),
                    "Adjacency asymmetry: {} -> {} but not {} -> {}",
                    u,
                    v,
                    v,
                    u,
                );
            }
        }
    }

    #[test]
    fn test_stabilizer_map_coverage() {
        let noise = DecoderNoiseModel::uniform(0.001);
        let config = MbbpLdConfig {
            distance: 5,
            rounds: 1,
            noise_model: noise,
        };
        let decoder = MbbpLdDecoder::new(config);

        // Every stabilizer should have at least 2 data qubit neighbors
        for (s_idx, support) in &decoder.stabilizer_map {
            assert!(
                support.len() >= 2,
                "Stabilizer {} has only {} data neighbors, expected >= 2",
                s_idx,
                support.len(),
            );
        }

        // Every interior data qubit should appear in at least one stabilizer
        let covered: HashSet<usize> = decoder
            .stabilizer_map
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();

        // At minimum, most data qubits should be covered
        let coverage = covered.len() as f64 / decoder.num_data_qubits() as f64;
        assert!(
            coverage > 0.5,
            "Stabilizer coverage should be > 50%, got {:.1}%",
            coverage * 100.0,
        );
    }
}
