//! Sliding Window QEC Decoder
//!
//! Processes quantum error correction syndromes incrementally using a sliding
//! window of `w` measurement rounds, advancing by `s` rounds each step. This
//! architecture is essential for real-time fault-tolerant quantum computing where
//! syndrome data arrives continuously and must be decoded with bounded latency.
//!
//! # Background
//!
//! In a fault-tolerant quantum memory experiment the stabilizer measurements are
//! repeated every code cycle (typically microseconds for superconducting qubits).
//! A monolithic decoder that waits for all rounds before decoding introduces
//! unbounded latency. The sliding window approach bounds latency to O(w) rounds
//! while only sacrificing a small amount of decoding accuracy near window edges.
//!
//! Reference: Terhal, "Quantum error correction for quantum memories",
//! Rev. Mod. Phys. 87, 307 (2015).
//!
//! # Algorithm
//!
//! ```text
//!  time ──────────────────────────────────────────►
//!  rounds:  0  1  2  3  4  5  6  7  8  9  ...
//!
//!  window 1: [0  1  2  3  4]          w=5, s=2
//!  commit:    ^  ^                     commit rounds 0,1
//!
//!  window 2:       [2  3  4  5  6]
//!  commit:          ^  ^               commit rounds 2,3
//!
//!  window 3:             [4  5  6  7  8]
//!  commit:                ^  ^         commit rounds 4,5
//! ```
//!
//! Each window is decoded independently by an inner decoder (greedy matching or
//! union-find). Only the oldest `s` rounds of corrections are committed; the
//! remaining rounds act as look-ahead context that improves accuracy.
//!
//! # Inner Decoders
//!
//! - **Greedy**: Iteratively pair the two closest defects by Manhattan distance.
//!   Simple, O(d^2 * w) per window. Good baseline.
//! - **Union-Find**: Grow clusters from each defect with weighted quick-union and
//!   path compression. Nearly-linear O(n alpha(n)) per window. Superior threshold
//!   and speed for larger codes.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::sliding_window_decoder::{
//!     SlidingWindowDecoder, SyndromeRound, WindowInnerDecoder,
//! };
//!
//! let mut decoder = SlidingWindowDecoder::new(5, 2, 3, WindowInnerDecoder::UnionFind);
//!
//! for round_id in 0..20 {
//!     let syndrome = vec![false; 8]; // from stabilizer measurements
//!     decoder.push_round(SyndromeRound {
//!         round_id,
//!         syndrome,
//!         timestamp: round_id as f64 * 1.0e-6,
//!     });
//!
//!     if decoder.ready() {
//!         let result = decoder.decode_window();
//!         println!(
//!             "Committed rounds {:?}, {} defects, {:.1} us",
//!             result.committed_rounds, result.defects_in_window, result.decode_time_us,
//!         );
//!     }
//! }
//!
//! // Flush any remaining buffered rounds at the end of the experiment
//! let final_results = decoder.flush();
//! ```

use std::collections::VecDeque;
use std::time::Instant;

// ============================================================
// PUBLIC TYPES
// ============================================================

/// A single round of stabilizer measurement outcomes.
///
/// Each element of `syndrome` corresponds to a detector (ancilla qubit)
/// outcome. A value of `true` indicates a detection event (the eigenvalue
/// flipped relative to the previous round).
#[derive(Debug, Clone)]
pub struct SyndromeRound {
    /// Sequential round identifier, monotonically increasing.
    pub round_id: usize,
    /// Detector outcomes for this round. Length must equal the number of
    /// detectors in the code (typically `d^2 - 1` for a distance-d surface
    /// code with one stabilizer type).
    pub syndrome: Vec<bool>,
    /// Wall-clock timestamp in seconds (informational; not used by decoder).
    pub timestamp: f64,
}

/// Choice of inner decoder applied to each window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowInnerDecoder {
    /// Greedy closest-first defect matching.
    ///
    /// Iteratively finds the two closest defects (by space-time Manhattan
    /// distance), pairs them, and records a correction along a shortest
    /// path. Complexity: O(k^2) per window where k is the defect count.
    Greedy,

    /// Weighted union-find with path compression.
    ///
    /// Grows clusters from each defect until clusters merge, recording
    /// corrections along fusion paths. Nearly-linear O(n alpha(n)) per
    /// window. Generally superior threshold to greedy.
    UnionFind,
}

/// Outcome of decoding a single window.
#[derive(Debug, Clone)]
pub struct WindowResult {
    /// Round IDs whose corrections are now committed (the oldest `slide_step`
    /// rounds in the window, or all remaining rounds during a flush).
    pub committed_rounds: Vec<usize>,
    /// Correction bit-vectors, one per committed round. Each vector has the
    /// same length as the corresponding syndrome and indicates which data
    /// qubits should receive a correction flip.
    pub corrections: Vec<Vec<bool>>,
    /// Wall-clock time spent decoding this window, in microseconds.
    pub decode_time_us: f64,
    /// Total number of defects (detection events) in the decoded window.
    pub defects_in_window: usize,
    /// Number of defect pairs (or boundary matches) found by the inner decoder.
    pub matches_found: usize,
}

/// Sliding window QEC decoder.
///
/// Buffers incoming syndrome rounds and, once `window_size` rounds are
/// available, decodes the window and commits corrections for the oldest
/// `slide_step` rounds.
pub struct SlidingWindowDecoder {
    /// Number of measurement rounds in each decoding window.
    window_size: usize,
    /// Number of rounds to slide forward after each decode (number of rounds
    /// whose corrections are committed per window).
    slide_step: usize,
    /// QEC code distance. Used for correction vector sizing and weight
    /// calculations.
    code_distance: usize,
    /// Inner decoder variant applied to each window.
    decoder: WindowInnerDecoder,
    /// Corrections that have been permanently committed. Indexed by
    /// sequential decode order.
    committed_corrections: Vec<Vec<bool>>,
    /// Buffered syndrome rounds awaiting decoding.
    buffer: VecDeque<SyndromeRound>,
}
// ============================================================
// DEFECT GRAPH
// ============================================================

/// Sentinel index representing the code boundary as a virtual matching target.
///
/// In a surface code, unpaired defects can be matched to the nearest code
/// boundary rather than to another defect. This is essential when there is an
/// odd number of defects (at least one must match to boundary) and when a
/// defect is closer to the boundary than to any other defect (matching to
/// boundary reduces the correction weight and improves the decoder threshold).
const BOUNDARY_NODE: usize = usize::MAX;

/// A defect is a detection event at a specific (round, detector) coordinate
/// in the space-time syndrome volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Defect {
    round: usize,
    detector: usize,
}

/// Edge in the defect graph connecting two defects (or a defect to the
/// boundary). Weight is the Manhattan distance in space-time.
#[derive(Debug, Clone)]
struct DefectEdge {
    a: usize,
    b: usize,
    weight: f64,
}

/// Graph of defects extracted from a window of syndrome rounds.
///
/// Defects are identified by differencing consecutive syndrome rounds:
/// a detection event at (round r, detector j) means syndrome[r][j] differs
/// from syndrome[r-1][j] (or from the all-zero initial state for r=0).
///
/// Each defect also has a computed distance to the nearest code boundary.
/// For a distance-d surface code, the spatial boundary distance of detector j
/// is `min(j, d-1-j)` (the minimum number of edges to reach either spatial
/// boundary in the 1D detector chain). Boundary edges are included in the
/// edge list with `b = BOUNDARY_NODE`.
struct DefectGraph {
    /// List of defect locations.
    defects: Vec<Defect>,
    /// Fully-connected edge list with Manhattan weights, including edges to
    /// the virtual boundary node.
    edges: Vec<DefectEdge>,
    /// Distance from each defect to the nearest code boundary. Indexed by
    /// defect index. For a distance-d code, this is `min(j, d-1-j)` where
    /// j is the detector index within the spatial dimension.
    boundary_distances: Vec<f64>,
    /// Code distance used for boundary distance computation.
    code_distance: usize,
}

impl DefectGraph {
    /// Build a defect graph from a slice of syndrome rounds.
    ///
    /// Detection events are computed as the XOR between consecutive rounds
    /// (the first round is XOR-ed against an implicit all-zero round). The
    /// resulting defect pairs are connected by edges weighted with space-time
    /// Manhattan distance `|dr| + |dd|` where dr is round separation and dd
    /// is detector index separation.
    ///
    /// Boundary edges are added for each defect with weight equal to the
    /// defect's spatial distance to the nearest code boundary:
    /// `min(detector, code_distance - 1 - detector)`, clamped to a minimum
    /// of 1.0 so that the boundary is always reachable.
    fn from_rounds(rounds: &[SyndromeRound], code_distance: usize) -> Self {
        let mut defects = Vec::new();

        // Identify defects via syndrome differencing.
        let mut prev: Option<&Vec<bool>> = None;
        for round in rounds {
            let syndrome = &round.syndrome;
            for (det_idx, &val) in syndrome.iter().enumerate() {
                let prev_val = prev.map_or(false, |p| {
                    p.get(det_idx).copied().unwrap_or(false)
                });
                // A defect occurs where the syndrome changed.
                if val != prev_val {
                    defects.push(Defect {
                        round: round.round_id,
                        detector: det_idx,
                    });
                }
            }
            prev = Some(syndrome);
        }

        // Compute boundary distances for each defect.
        // For a distance-d surface code, the spatial boundary distance is
        // min(j, d-1-j) where j is the detector index. We clamp to at least
        // 1.0 so that boundary matching is always possible (distance 0 would
        // mean the defect IS the boundary, but we still need a non-zero
        // weight for the matching algorithm).
        let boundary_distances: Vec<f64> = defects
            .iter()
            .map(|d| {
                if code_distance <= 1 {
                    1.0
                } else {
                    let j = d.detector;
                    let dist_to_left = j;
                    let dist_to_right = if code_distance > 1 + j {
                        code_distance - 1 - j
                    } else {
                        0
                    };
                    let min_dist = dist_to_left.min(dist_to_right);
                    // Clamp to at least 1 so boundary is always reachable.
                    (min_dist.max(1)) as f64
                }
            })
            .collect();

        // Build fully-connected edge list with Manhattan distance weights.
        let mut edges = Vec::new();
        for i in 0..defects.len() {
            for j in (i + 1)..defects.len() {
                let dr = (defects[i].round as isize - defects[j].round as isize).unsigned_abs();
                let dd =
                    (defects[i].detector as isize - defects[j].detector as isize).unsigned_abs();
                let weight = (dr + dd) as f64;
                edges.push(DefectEdge {
                    a: i,
                    b: j,
                    weight,
                });
            }
        }

        // Add boundary edges: each defect has an edge to the virtual boundary
        // node with weight equal to its boundary distance.
        for (i, &bdist) in boundary_distances.iter().enumerate() {
            edges.push(DefectEdge {
                a: i,
                b: BOUNDARY_NODE,
                weight: bdist,
            });
        }

        // Sort edges by weight for both decoders (greedy needs sorted order,
        // union-find benefits from processing lighter edges first).
        edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal));

        DefectGraph {
            defects,
            edges,
            boundary_distances,
            code_distance,
        }
    }

    /// Number of defects (excludes the virtual boundary node).
    fn num_defects(&self) -> usize {
        self.defects.len()
    }
}

// ============================================================
// UNION-FIND DATA STRUCTURE
// ============================================================

/// Weighted quick-union with path compression.
///
/// Classic data structure for the union-find QEC decoder. Each defect starts
/// as its own cluster. Clusters grow by processing edges in weight order.
/// When two clusters merge the edge is recorded as part of the correction
/// path.
struct WeightedUnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl WeightedUnionFind {
    fn new(n: usize) -> Self {
        WeightedUnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    /// Find the root of the component containing `x`, with path compression.
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the components containing `a` and `b`. Returns `true` if they
    /// were in different components (a merge happened).
    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        // Union by rank.
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
            self.size[rb] += self.size[ra];
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
        } else {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
            self.rank[ra] += 1;
        }
        true
    }

    /// Check whether `a` and `b` are in the same component.
    fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}

// ============================================================
// INNER DECODER IMPLEMENTATIONS
// ============================================================
/// Result of running the inner decoder on a defect graph: a set of matched
/// defect-index pairs and boundary-matched defects.
struct MatchResult {
    /// Pairs of defect indices that were matched to each other.
    pairs: Vec<(usize, usize)>,
    /// Defect indices that were matched to the code boundary rather than to
    /// another defect. In a surface code, this happens when a defect is closer
    /// to the boundary than to any other unmatched defect, or when there is an
    /// odd number of defects (at least one must go to boundary).
    boundary_matches: Vec<usize>,
}

impl MatchResult {
    /// Total number of matches (pairs + boundary matches).
    fn total_matches(&self) -> usize {
        self.pairs.len() + self.boundary_matches.len()
    }
}

/// Greedy closest-first matching with boundary support.
///
/// Process edges in ascending weight order. When both endpoints are unmatched,
/// pair them. Boundary edges (where one endpoint is `BOUNDARY_NODE`) match a
/// single defect to the boundary. After processing all edges, any remaining
/// unmatched defects are force-matched to the boundary.
///
/// This is an approximation to MWPM but runs in O(k^2 log k) where k is the
/// defect count.
fn decode_greedy(graph: &DefectGraph) -> MatchResult {
    let n = graph.num_defects();
    let mut matched = vec![false; n];
    let mut pairs = Vec::new();
    let mut boundary_matches = Vec::new();

    // Edges are already sorted by weight in DefectGraph::from_rounds.
    // This includes boundary edges (where b == BOUNDARY_NODE).
    for edge in &graph.edges {
        if edge.b == BOUNDARY_NODE {
            // Boundary edge: match defect a to boundary if unmatched.
            if !matched[edge.a] {
                matched[edge.a] = true;
                boundary_matches.push(edge.a);
            }
        } else if edge.a == BOUNDARY_NODE {
            // Boundary edge (reversed): match defect b to boundary if unmatched.
            if !matched[edge.b] {
                matched[edge.b] = true;
                boundary_matches.push(edge.b);
            }
        } else {
            // Normal defect-defect edge.
            if !matched[edge.a] && !matched[edge.b] {
                matched[edge.a] = true;
                matched[edge.b] = true;
                pairs.push((edge.a, edge.b));
            }
        }
    }

    // Safety: force-match any remaining unmatched defects to boundary.
    // This should not happen if boundary edges are included, but provides
    // robustness.
    for i in 0..n {
        if !matched[i] {
            boundary_matches.push(i);
        }
    }

    MatchResult {
        pairs,
        boundary_matches,
    }
}

/// Union-find cluster growth decoder with boundary support.
///
/// Processes edges in weight order, merging clusters. A match is recorded when
/// two odd-parity clusters merge (each starting with parity 1 from a single
/// defect). The boundary is represented as an additional node (index `n`) in
/// the union-find structure with even parity (parity = false). When an
/// odd-parity cluster merges with the boundary node's component, a boundary
/// match is recorded.
///
/// This produces a valid correction because every defect ends up in an
/// even-parity cluster (either by pairing with another defect or by merging
/// with the boundary).
fn decode_union_find(graph: &DefectGraph) -> MatchResult {
    let n = graph.num_defects();
    if n == 0 {
        return MatchResult {
            pairs: Vec::new(),
            boundary_matches: Vec::new(),
        };
    }

    // Create union-find with n+1 nodes: indices 0..n-1 are defects,
    // index n is the virtual boundary node.
    let boundary_idx = n;
    let mut uf = WeightedUnionFind::new(n + 1);

    // Track the parity (number of defects mod 2) of each cluster root.
    // Initially each defect cluster has parity true (odd, one defect).
    // The boundary node starts with parity false (even, zero defects).
    let mut cluster_parity = vec![true; n + 1];
    cluster_parity[boundary_idx] = false;

    let mut pairs = Vec::new();
    let mut boundary_matches = Vec::new();

    // Edges are pre-sorted by weight.
    for edge in &graph.edges {
        // Map BOUNDARY_NODE sentinel to our local boundary index.
        let a = if edge.a == BOUNDARY_NODE {
            boundary_idx
        } else {
            edge.a
        };
        let b = if edge.b == BOUNDARY_NODE {
            boundary_idx
        } else {
            edge.b
        };

        let ra = uf.find(a);
        let rb = uf.find(b);
        if ra == rb {
            continue; // Same cluster, skip.
        }

        let pa = cluster_parity[ra];
        let pb = cluster_parity[rb];

        // Merge the two clusters.
        uf.union(a, b);
        let new_root = uf.find(a);

        // New parity is XOR of the two cluster parities.
        cluster_parity[new_root] = pa ^ pb;

        // Record matches based on parity neutralisation.
        if pa && pb {
            // Both clusters were odd. Their merge neutralises them.
            // The new cluster parity is already set to false (even) above.
            if a == boundary_idx || b == boundary_idx {
                // The boundary cluster became odd from a previous boundary
                // merge. Now another odd defect merges in, neutralising both.
                if a == boundary_idx {
                    boundary_matches.push(edge.b);
                } else {
                    boundary_matches.push(edge.a);
                }
            } else {
                // Two defect clusters merged: record as a pair.
                pairs.push((edge.a, edge.b));
            }
        } else if pa && !pb && b == boundary_idx {
            // Odd-parity defect cluster merged with the even-parity boundary.
            // The defect is neutralised via boundary matching. The boundary
            // acts as a virtual partner. We force the resulting cluster to
            // even parity since the match has been recorded.
            boundary_matches.push(edge.a);
            cluster_parity[new_root] = false;
        } else if pb && !pa && a == boundary_idx {
            // Symmetric case: boundary on the left side.
            boundary_matches.push(edge.b);
            cluster_parity[new_root] = false;
        }
    }

    // After processing all edges, any remaining odd-parity cluster that
    // has not merged with the boundary should be force-matched to boundary.
    for i in 0..n {
        let root = uf.find(i);
        if cluster_parity[root] {
            // This cluster is still odd-parity. Merge with boundary.
            let br = uf.find(boundary_idx);
            if root != br {
                uf.union(i, boundary_idx);
                let new_root = uf.find(i);
                cluster_parity[new_root] = cluster_parity[root] ^ cluster_parity[br];
            }
            boundary_matches.push(i);
        }
    }

    MatchResult {
        pairs,
        boundary_matches,
    }
}

// ============================================================
// CORRECTION GENERATION
// ============================================================

/// Convert matched defect pairs and boundary matches into per-round correction
/// bit-vectors.
///
/// For each matched pair, a correction is applied along a shortest path in
/// the space-time lattice. In this simplified model, the correction is placed
/// on the detectors at each round between (and including) the two defect
/// endpoints.
///
/// For boundary matches, a correction is applied along the shortest path from
/// the defect to the nearest spatial boundary (detector index 0 or the maximum
/// detector index, whichever is closer).
fn generate_corrections(
    graph: &DefectGraph,
    matches: &MatchResult,
    rounds: &[SyndromeRound],
    num_detectors: usize,
) -> Vec<Vec<bool>> {
    // Build a round_id -> index mapping.
    let mut round_index: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (idx, r) in rounds.iter().enumerate() {
        round_index.insert(r.round_id, idx);
    }

    let num_rounds = rounds.len();
    let mut corrections = vec![vec![false; num_detectors]; num_rounds];

    // Process defect-defect pair matches.
    for &(ai, bi) in &matches.pairs {
        let da = &graph.defects[ai];
        let db = &graph.defects[bi];

        // Determine the space-time path: walk from da to db.
        let (r_start, r_end) = if da.round <= db.round {
            (da.round, db.round)
        } else {
            (db.round, da.round)
        };

        let (d_start, d_end) = if da.detector <= db.detector {
            (da.detector, db.detector)
        } else {
            (db.detector, da.detector)
        };

        // Apply correction along the temporal direction first: flip at the
        // starting detector for each round in [r_start, r_end).
        let start_det = da.detector.min(db.detector);
        for rid in r_start..r_end {
            if let Some(&idx) = round_index.get(&rid) {
                if start_det < num_detectors {
                    corrections[idx][start_det] ^= true;
                }
            }
        }

        // Then apply correction along the spatial direction at r_end.
        if let Some(&idx) = round_index.get(&r_end) {
            for det in d_start..d_end {
                if det < num_detectors {
                    corrections[idx][det] ^= true;
                }
            }
        }
    }

    // Process boundary matches: apply corrections from the defect to the
    // nearest spatial boundary.
    for &di in &matches.boundary_matches {
        let defect = &graph.defects[di];

        if let Some(&round_idx) = round_index.get(&defect.round) {
            // Determine which boundary is closer: left (detector 0) or right.
            let det = defect.detector;
            let dist_to_left = det;
            let dist_to_right = if num_detectors > det {
                num_detectors - 1 - det
            } else {
                0
            };

            if dist_to_left <= dist_to_right {
                // Correct toward the left boundary (detector 0).
                for d in 0..det {
                    if d < num_detectors {
                        corrections[round_idx][d] ^= true;
                    }
                }
            } else {
                // Correct toward the right boundary (last detector).
                for d in det..num_detectors {
                    if d < num_detectors {
                        corrections[round_idx][d] ^= true;
                    }
                }
            }
        }
    }

    corrections
}

// ============================================================
// SLIDING WINDOW DECODER
// ============================================================

impl SlidingWindowDecoder {
    /// Create a new sliding window decoder.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of measurement rounds per decoding window.
    ///   Must be >= 2. Larger windows give better accuracy at higher latency.
    /// * `slide_step` - Number of rounds to commit per window and advance.
    ///   Must be >= 1 and <= `window_size`.
    /// * `code_distance` - Distance of the QEC code. Used for sizing
    ///   correction vectors.
    /// * `decoder` - Inner decoder variant to use.
    ///
    /// # Panics
    ///
    /// Panics if `window_size < 2`, `slide_step < 1`, `slide_step > window_size`,
    /// or `code_distance < 1`.
    pub fn new(
        window_size: usize,
        slide_step: usize,
        code_distance: usize,
        decoder: WindowInnerDecoder,
    ) -> Self {
        assert!(window_size >= 2, "window_size must be >= 2");
        assert!(slide_step >= 1, "slide_step must be >= 1");
        assert!(
            slide_step <= window_size,
            "slide_step must be <= window_size"
        );
        assert!(code_distance >= 1, "code_distance must be >= 1");

        SlidingWindowDecoder {
            window_size,
            slide_step,
            code_distance,
            decoder,
            committed_corrections: Vec::new(),
            buffer: VecDeque::new(),
        }
    }

    /// Feed a new syndrome measurement round into the buffer.
    ///
    /// Rounds should be pushed in order of increasing `round_id`. The decoder
    /// does not enforce strict ordering but results are only meaningful if
    /// rounds arrive sequentially.
    pub fn push_round(&mut self, round: SyndromeRound) {
        self.buffer.push_back(round);
    }

    /// Check whether enough rounds have been buffered to perform a window decode.
    ///
    /// Returns `true` when `buffer.len() >= window_size`.
    pub fn ready(&self) -> bool {
        self.buffer.len() >= self.window_size
    }

    /// Decode the current window and commit corrections for the oldest
    /// `slide_step` rounds.
    ///
    /// The window comprises the first `window_size` rounds in the buffer.
    /// After decoding, the oldest `slide_step` rounds are drained from the
    /// buffer and their corrections are appended to `committed_corrections`.
    ///
    /// # Panics
    ///
    /// Panics if `ready()` is false (not enough rounds buffered).
    pub fn decode_window(&mut self) -> WindowResult {
        assert!(
            self.ready(),
            "Not enough rounds buffered (have {}, need {})",
            self.buffer.len(),
            self.window_size
        );

        let start = Instant::now();

        // Extract the window of rounds (borrow-friendly: collect to Vec).
        let window_rounds: Vec<SyndromeRound> =
            self.buffer.iter().take(self.window_size).cloned().collect();

        // Determine detector count from the first round.
        let num_detectors = window_rounds
            .first()
            .map(|r| r.syndrome.len())
            .unwrap_or(0);

        // Build defect graph.
        let graph = DefectGraph::from_rounds(&window_rounds, self.code_distance);
        let defects_in_window = graph.num_defects();

        // Run inner decoder.
        let matches = match self.decoder {
            WindowInnerDecoder::Greedy => decode_greedy(&graph),
            WindowInnerDecoder::UnionFind => decode_union_find(&graph),
        };
        let matches_found = matches.total_matches();

        // Generate per-round correction vectors.
        let all_corrections =
            generate_corrections(&graph, &matches, &window_rounds, num_detectors);

        // Commit only the oldest `slide_step` rounds.
        let committed_rounds: Vec<usize> = window_rounds
            .iter()
            .take(self.slide_step)
            .map(|r| r.round_id)
            .collect();

        let committed_corr: Vec<Vec<bool>> = all_corrections
            .into_iter()
            .take(self.slide_step)
            .collect();

        // Store committed corrections.
        self.committed_corrections.extend(committed_corr.clone());

        // Drain the oldest `slide_step` rounds from the buffer.
        for _ in 0..self.slide_step {
            self.buffer.pop_front();
        }

        let elapsed = start.elapsed();

        WindowResult {
            committed_rounds,
            corrections: committed_corr,
            decode_time_us: elapsed.as_secs_f64() * 1_000_000.0,
            defects_in_window,
            matches_found,
        }
    }

    /// Return a reference to all corrections committed so far, in the order
    /// they were decoded.
    pub fn committed(&self) -> &[Vec<bool>] {
        &self.committed_corrections
    }

    /// Decode all remaining buffered rounds by shrinking the window until
    /// the buffer is fully consumed.
    ///
    /// This is used at the end of an experiment to flush any trailing rounds
    /// that have not yet been decoded. The final window may be smaller than
    /// `window_size`.
    ///
    /// Returns a vector of `WindowResult`s, one per flush step.
    pub fn flush(&mut self) -> Vec<WindowResult> {
        let mut results = Vec::new();

        // First, decode any full windows remaining.
        while self.ready() {
            results.push(self.decode_window());
        }

        // Then decode the remaining partial window if any rounds remain.
        if !self.buffer.is_empty() {
            let start = Instant::now();

            let remaining: Vec<SyndromeRound> = self.buffer.drain(..).collect();
            let num_remaining = remaining.len();

            let num_detectors = remaining
                .first()
                .map(|r| r.syndrome.len())
                .unwrap_or(0);

            let graph = DefectGraph::from_rounds(&remaining, self.code_distance);
            let defects_in_window = graph.num_defects();

            let matches = match self.decoder {
                WindowInnerDecoder::Greedy => decode_greedy(&graph),
                WindowInnerDecoder::UnionFind => decode_union_find(&graph),
            };
            let matches_found = matches.total_matches();

            let all_corrections =
                generate_corrections(&graph, &matches, &remaining, num_detectors);

            let committed_rounds: Vec<usize> = remaining.iter().map(|r| r.round_id).collect();

            let committed_corr: Vec<Vec<bool>> =
                all_corrections.into_iter().take(num_remaining).collect();

            self.committed_corrections.extend(committed_corr.clone());

            let elapsed = start.elapsed();

            results.push(WindowResult {
                committed_rounds,
                corrections: committed_corr,
                decode_time_us: elapsed.as_secs_f64() * 1_000_000.0,
                defects_in_window,
                matches_found,
            });
        }

        results
    }

    /// Number of rounds currently buffered.
    pub fn buffered_rounds(&self) -> usize {
        self.buffer.len()
    }

    /// The configured window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// The configured slide step.
    pub fn slide_step(&self) -> usize {
        self.slide_step
    }

    /// The configured code distance.
    pub fn code_distance(&self) -> usize {
        self.code_distance
    }

    /// The configured inner decoder variant.
    pub fn inner_decoder(&self) -> WindowInnerDecoder {
        self.decoder
    }

    /// Total number of rounds committed so far.
    pub fn total_committed_rounds(&self) -> usize {
        self.committed_corrections.len()
    }
}

// ============================================================
// DISPLAY IMPLEMENTATIONS
// ============================================================

impl std::fmt::Display for WindowInnerDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowInnerDecoder::Greedy => write!(f, "Greedy"),
            WindowInnerDecoder::UnionFind => write!(f, "UnionFind"),
        }
    }
}

impl std::fmt::Display for WindowResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WindowResult(committed={:?}, defects={}, matches={}, time={:.1}us)",
            self.committed_rounds, self.defects_in_window, self.matches_found, self.decode_time_us,
        )
    }
}

// ============================================================
// StreamingDecoder TRAIT IMPLEMENTATION
// ============================================================

use crate::traits::StreamingDecoder;

impl StreamingDecoder for SlidingWindowDecoder {
    type WindowResult = WindowResult;

    fn push_syndrome(&mut self, round_id: usize, syndrome: Vec<bool>, timestamp: f64) {
        self.push_round(SyndromeRound {
            round_id,
            syndrome,
            timestamp,
        });
    }

    fn ready(&self) -> bool {
        SlidingWindowDecoder::ready(self)
    }

    fn decode_window(&mut self) -> WindowResult {
        SlidingWindowDecoder::decode_window(self)
    }

    fn flush(&mut self) -> Vec<WindowResult> {
        SlidingWindowDecoder::flush(self)
    }

    fn name(&self) -> &str {
        match self.decoder {
            WindowInnerDecoder::Greedy => "SlidingWindow(Greedy)",
            WindowInnerDecoder::UnionFind => "SlidingWindow(UnionFind)",
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------

    /// Create a syndrome round with the given detector values.
    fn make_round(round_id: usize, syndrome: Vec<bool>) -> SyndromeRound {
        SyndromeRound {
            round_id,
            syndrome,
            timestamp: round_id as f64 * 1.0e-6,
        }
    }

    /// Create an all-zero (no-error) syndrome round of the given width.
    fn clean_round(round_id: usize, num_detectors: usize) -> SyndromeRound {
        make_round(round_id, vec![false; num_detectors])
    }

    // ----------------------------------------------------------
    // Construction
    // ----------------------------------------------------------

    #[test]
    fn test_new_decoder() {
        let dec = SlidingWindowDecoder::new(5, 2, 3, WindowInnerDecoder::Greedy);
        assert_eq!(dec.window_size(), 5);
        assert_eq!(dec.slide_step(), 2);
        assert_eq!(dec.code_distance(), 3);
        assert_eq!(dec.inner_decoder(), WindowInnerDecoder::Greedy);
        assert_eq!(dec.buffered_rounds(), 0);
        assert!(dec.committed().is_empty());
    }

    #[test]
    #[should_panic(expected = "window_size must be >= 2")]
    fn test_new_decoder_invalid_window() {
        SlidingWindowDecoder::new(1, 1, 3, WindowInnerDecoder::Greedy);
    }

    #[test]
    #[should_panic(expected = "slide_step must be <= window_size")]
    fn test_new_decoder_invalid_step() {
        SlidingWindowDecoder::new(3, 5, 3, WindowInnerDecoder::Greedy);
    }

    // ----------------------------------------------------------
    // Push and buffering
    // ----------------------------------------------------------

    #[test]
    fn test_push_round() {
        let mut dec = SlidingWindowDecoder::new(4, 2, 3, WindowInnerDecoder::UnionFind);
        assert_eq!(dec.buffered_rounds(), 0);

        dec.push_round(clean_round(0, 8));
        assert_eq!(dec.buffered_rounds(), 1);

        dec.push_round(clean_round(1, 8));
        dec.push_round(clean_round(2, 8));
        assert_eq!(dec.buffered_rounds(), 3);
    }

    // ----------------------------------------------------------
    // Readiness
    // ----------------------------------------------------------

    #[test]
    fn test_ready_check() {
        let mut dec = SlidingWindowDecoder::new(3, 1, 3, WindowInnerDecoder::Greedy);
        assert!(!dec.ready());

        dec.push_round(clean_round(0, 4));
        assert!(!dec.ready());

        dec.push_round(clean_round(1, 4));
        assert!(!dec.ready());

        dec.push_round(clean_round(2, 4));
        assert!(dec.ready());
    }

    // ----------------------------------------------------------
    // Single window decode
    // ----------------------------------------------------------

    #[test]
    fn test_decode_single_window() {
        let mut dec = SlidingWindowDecoder::new(3, 2, 3, WindowInnerDecoder::Greedy);

        // Push 3 rounds: detectors 0,1,2,3 with a defect pair in round 1.
        dec.push_round(clean_round(0, 4));
        dec.push_round(make_round(1, vec![true, false, false, false]));
        dec.push_round(make_round(2, vec![true, false, false, false])); // defect cancels

        assert!(dec.ready());
        let result = dec.decode_window();

        // Should have committed 2 rounds (slide_step=2).
        assert_eq!(result.committed_rounds.len(), 2);
        assert_eq!(result.committed_rounds[0], 0);
        assert_eq!(result.committed_rounds[1], 1);
        assert_eq!(result.corrections.len(), 2);

        // Buffer should now have 1 round left.
        assert_eq!(dec.buffered_rounds(), 1);
    }

    // ----------------------------------------------------------
    // Multiple sliding windows
    // ----------------------------------------------------------

    #[test]
    fn test_sliding_multiple_windows() {
        let mut dec = SlidingWindowDecoder::new(4, 2, 3, WindowInnerDecoder::UnionFind);
        let num_det = 8;

        // Push 8 rounds.
        for i in 0..8 {
            dec.push_round(clean_round(i, num_det));
        }

        // Should be able to decode multiple windows.
        let mut total_committed = 0;
        let mut decode_count = 0;
        while dec.ready() {
            let result = dec.decode_window();
            total_committed += result.committed_rounds.len();
            decode_count += 1;
        }

        // With w=4, s=2 and 8 rounds: windows at [0..3], [2..5], [4..7] = 3 windows.
        // Each commits 2 rounds, so 6 total committed. 2 rounds remain in buffer.
        assert_eq!(decode_count, 3);
        assert_eq!(total_committed, 6);
        assert_eq!(dec.buffered_rounds(), 2);
        assert_eq!(dec.committed().len(), 6);
    }

    // ----------------------------------------------------------
    // Flush remaining
    // ----------------------------------------------------------

    #[test]
    fn test_flush_remaining() {
        let mut dec = SlidingWindowDecoder::new(4, 2, 3, WindowInnerDecoder::Greedy);
        let num_det = 4;

        // Push 7 rounds: 1 full window (4 rounds, commit 2), then 5 remain,
        // 1 more full window (commit 2), 3 remain, etc.
        for i in 0..7 {
            dec.push_round(clean_round(i, num_det));
        }

        let results = dec.flush();

        // Flush should have processed everything.
        assert_eq!(dec.buffered_rounds(), 0);

        // Total committed rounds should equal 7.
        let total: usize = results.iter().map(|r| r.committed_rounds.len()).sum();
        assert_eq!(total, 7);
        assert_eq!(dec.committed().len(), 7);
    }

    // ----------------------------------------------------------
    // Clean syndrome (no errors)
    // ----------------------------------------------------------

    #[test]
    fn test_no_errors_clean_syndrome() {
        let mut dec = SlidingWindowDecoder::new(3, 1, 5, WindowInnerDecoder::UnionFind);

        for i in 0..5 {
            dec.push_round(clean_round(i, 12));
        }

        while dec.ready() {
            let result = dec.decode_window();
            // No defects should be found.
            assert_eq!(result.defects_in_window, 0);
            assert_eq!(result.matches_found, 0);
            // Corrections should all be false.
            for corr in &result.corrections {
                assert!(corr.iter().all(|&b| !b));
            }
        }
    }

    // ----------------------------------------------------------
    // Single defect pair
    // ----------------------------------------------------------

    #[test]
    fn test_single_defect_pair() {
        // Two rounds with a single defect that appears and then disappears,
        // creating a matched pair.
        let mut dec = SlidingWindowDecoder::new(3, 1, 3, WindowInnerDecoder::Greedy);

        dec.push_round(clean_round(0, 4));
        // Defect appears at detector 2.
        dec.push_round(make_round(1, vec![false, false, true, false]));
        // Defect disappears (syndrome flips back).
        dec.push_round(clean_round(2, 4));

        let result = dec.decode_window();
        // Should find exactly 2 defects (appear and disappear) and 1 match.
        assert_eq!(result.defects_in_window, 2);
        assert_eq!(result.matches_found, 1);
    }

    // ----------------------------------------------------------
    // Union-find basic
    // ----------------------------------------------------------

    #[test]
    fn test_union_find_basic() {
        let mut uf = WeightedUnionFind::new(6);

        // Initially all separate.
        assert!(!uf.connected(0, 1));
        assert!(!uf.connected(2, 3));

        // Union some elements.
        assert!(uf.union(0, 1));
        assert!(uf.connected(0, 1));

        assert!(uf.union(2, 3));
        assert!(uf.connected(2, 3));
        assert!(!uf.connected(0, 2));

        // Merge the two clusters.
        assert!(uf.union(1, 3));
        assert!(uf.connected(0, 3));
        assert!(uf.connected(1, 2));

        // Duplicate union returns false.
        assert!(!uf.union(0, 3));
    }

    // ----------------------------------------------------------
    // Greedy decoder basic
    // ----------------------------------------------------------

    #[test]
    fn test_greedy_basic() {
        // Build a small defect graph with 4 defects.
        let defects = vec![
            Defect { round: 0, detector: 0 },
            Defect { round: 0, detector: 3 },
            Defect { round: 1, detector: 0 },
            Defect { round: 1, detector: 3 },
        ];

        // Fully connected edges sorted by weight.
        let mut edges = Vec::new();
        for i in 0..defects.len() {
            for j in (i + 1)..defects.len() {
                let dr = (defects[i].round as isize - defects[j].round as isize).unsigned_abs();
                let dd =
                    (defects[i].detector as isize - defects[j].detector as isize).unsigned_abs();
                edges.push(DefectEdge {
                    a: i,
                    b: j,
                    weight: (dr + dd) as f64,
                });
            }
        }
        edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal));

        let boundary_distances = vec![100.0; defects.len()];
        let graph = DefectGraph { defects, edges, boundary_distances, code_distance: 100 };

        let result = decode_greedy(&graph);
        // 4 defects should yield exactly 2 pairs.
        assert_eq!(result.pairs.len(), 2);
    }

    // ----------------------------------------------------------
    // Committed corrections accumulate
    // ----------------------------------------------------------

    #[test]
    fn test_committed_corrections_accumulate() {
        let mut dec = SlidingWindowDecoder::new(3, 1, 3, WindowInnerDecoder::Greedy);

        for i in 0..6 {
            dec.push_round(clean_round(i, 4));
        }

        let mut count = 0;
        while dec.ready() {
            dec.decode_window();
            count += 1;
            assert_eq!(dec.committed().len(), count);
        }

        assert_eq!(count, 4); // 6 rounds, w=3, s=1 => windows at 0..2, 1..3, 2..4, 3..5
    }

    // ----------------------------------------------------------
    // Decode timing
    // ----------------------------------------------------------

    #[test]
    fn test_decode_timing() {
        let mut dec = SlidingWindowDecoder::new(5, 2, 3, WindowInnerDecoder::UnionFind);

        for i in 0..5 {
            dec.push_round(clean_round(i, 8));
        }

        let result = dec.decode_window();
        // The decode_time_us should be a non-negative real number.
        assert!(result.decode_time_us >= 0.0);
        assert!(result.decode_time_us.is_finite());
    }

    // ----------------------------------------------------------
    // Surface code distance-3 scenario
    // ----------------------------------------------------------

    #[test]
    fn test_surface_code_d3() {
        // Simulate a distance-3 surface code with 8 detectors (4 X + 4 Z
        // for a d=3 rotated surface code). Inject a single X error that
        // triggers two X-type detectors.
        let d = 3;
        let num_det = (d * d - 1) / 2 * 2; // 8 detectors
        let window = 5;
        let step = 2;

        let mut dec = SlidingWindowDecoder::new(window, step, d, WindowInnerDecoder::UnionFind);

        // Rounds 0-1: no errors.
        dec.push_round(clean_round(0, num_det));
        dec.push_round(clean_round(1, num_det));

        // Round 2: single X error triggers detectors 0 and 1.
        let mut synd = vec![false; num_det];
        synd[0] = true;
        synd[1] = true;
        dec.push_round(make_round(2, synd));

        // Round 3: error persists, so syndrome is same => no new defects
        // relative to round 2.
        let mut synd3 = vec![false; num_det];
        synd3[0] = true;
        synd3[1] = true;
        dec.push_round(make_round(3, synd3));

        // Round 4: error is corrected, syndromes flip back.
        dec.push_round(clean_round(4, num_det));

        assert!(dec.ready());
        let result = dec.decode_window();

        // Should have found defects (from rounds 2 and 4 where the syndrome
        // changed) and produced matches.
        assert!(result.defects_in_window >= 2);
        assert!(result.matches_found >= 1);

        // Committed 2 rounds.
        assert_eq!(result.committed_rounds.len(), 2);
        assert_eq!(dec.committed().len(), 2);
    }

    // ----------------------------------------------------------
    // Defect graph construction
    // ----------------------------------------------------------

    #[test]
    fn test_defect_graph_construction() {
        let rounds = vec![
            clean_round(0, 4),
            make_round(1, vec![true, false, true, false]),
            clean_round(2, 4),
        ];

        let graph = DefectGraph::from_rounds(&rounds, 100);

        // Round 0->1: detectors 0 and 2 flip => 2 defects.
        // Round 1->2: detectors 0 and 2 flip back => 2 more defects.
        assert_eq!(graph.num_defects(), 4);

        // 4 defects => C(4,2) = 6 defect-defect edges + 4 boundary edges = 10.
        assert_eq!(graph.edges.len(), 10);

        // Edges should be sorted by weight.
        for w in graph.edges.windows(2) {
            assert!(w[0].weight <= w[1].weight);
        }
    }

    // ----------------------------------------------------------
    // Union-find decoder on defect graph
    // ----------------------------------------------------------

    #[test]
    fn test_union_find_decoder_on_graph() {
        let rounds = vec![
            clean_round(0, 4),
            make_round(1, vec![false, true, false, true]),
            clean_round(2, 4),
        ];

        let graph = DefectGraph::from_rounds(&rounds, 100);
        assert_eq!(graph.num_defects(), 4);

        let result = decode_union_find(&graph);
        // 4 defects should all be matched (via pairs and/or boundary matches).
        assert_eq!(result.pairs.len() * 2 + result.boundary_matches.len(), 4);
    }

    // ----------------------------------------------------------
    // Both decoders agree on simple cases
    // ----------------------------------------------------------

    #[test]
    fn test_greedy_and_union_find_agree_on_simple() {
        let rounds = vec![
            clean_round(0, 6),
            make_round(1, vec![true, false, false, false, false, true]),
            clean_round(2, 6),
        ];

        let graph = DefectGraph::from_rounds(&rounds, 100);

        let greedy = decode_greedy(&graph);
        let uf = decode_union_find(&graph);

        // Both should match all 4 defects (via pairs and/or boundary).
        assert_eq!(greedy.pairs.len() * 2 + greedy.boundary_matches.len(), 4);
        assert_eq!(uf.pairs.len() * 2 + uf.boundary_matches.len(), 4);
    }

    // ----------------------------------------------------------
    // Display implementations
    // ----------------------------------------------------------

    #[test]
    fn test_display_implementations() {
        assert_eq!(format!("{}", WindowInnerDecoder::Greedy), "Greedy");
        assert_eq!(format!("{}", WindowInnerDecoder::UnionFind), "UnionFind");

        let result = WindowResult {
            committed_rounds: vec![0, 1],
            corrections: vec![vec![false; 4], vec![false; 4]],
            decode_time_us: 42.5,
            defects_in_window: 3,
            matches_found: 1,
        };
        let s = format!("{}", result);
        assert!(s.contains("defects=3"));
        assert!(s.contains("matches=1"));
    }

    // ----------------------------------------------------------
    // Boundary matching: single defect must match boundary
    // ----------------------------------------------------------

    #[test]
    fn test_single_defect_boundary_match() {
        // A single defect cannot pair with another defect because there is
        // only one. It must be matched to the boundary. This tests the
        // fundamental odd-parity case.

        // Use a distance-5 code with 8 detectors.
        let mut dec_greedy =
            SlidingWindowDecoder::new(3, 1, 5, WindowInnerDecoder::Greedy);
        let mut dec_uf =
            SlidingWindowDecoder::new(3, 1, 5, WindowInnerDecoder::UnionFind);

        // Round 0: clean
        // Round 1: single defect at detector 3
        // Round 2: clean (defect disappears, but since round 1 had a defect
        //   and round 2 is clean, there is a second defect at round 2 as well
        //   due to syndrome differencing)
        // Actually, with syndrome differencing: round 0->1 sees det 3 flip = defect.
        // Round 1->2 sees det 3 flip back = second defect. That's 2 defects, even.
        //
        // To get a truly single defect, we need a persistent syndrome change.
        // Round 0: clean, Round 1: det 3 on, Round 2: det 3 still on
        // => defect only at round 0->1 transition.

        dec_greedy.push_round(clean_round(0, 8));
        dec_greedy.push_round(make_round(1, vec![false, false, false, true, false, false, false, false]));
        // Round 2: same syndrome as round 1 => no new defects
        dec_greedy.push_round(make_round(2, vec![false, false, false, true, false, false, false, false]));

        dec_uf.push_round(clean_round(0, 8));
        dec_uf.push_round(make_round(1, vec![false, false, false, true, false, false, false, false]));
        dec_uf.push_round(make_round(2, vec![false, false, false, true, false, false, false, false]));

        let result_greedy = dec_greedy.decode_window();
        let result_uf = dec_uf.decode_window();

        // Should find exactly 1 defect.
        assert_eq!(result_greedy.defects_in_window, 1);
        assert_eq!(result_uf.defects_in_window, 1);

        // The single defect must be matched (to boundary since no partner exists).
        assert_eq!(result_greedy.matches_found, 1);
        assert_eq!(result_uf.matches_found, 1);
    }

    // ----------------------------------------------------------
    // Boundary matching: odd number of defects
    // ----------------------------------------------------------

    #[test]
    fn test_odd_defects_boundary() {
        // 3 defects: one pair should match each other, the remaining one
        // should match to boundary. Tests the odd-parity boundary requirement.
        //
        // Use code_distance=20 and place defects at detectors 8, 9, 10
        // (interior of the code). Boundary distances will be ~8-10, while
        // defect-defect distances are 1-2. So the closest pair (8,9) with
        // distance 1 will match first, leaving defect 10 to go to boundary.
        let rounds = vec![
            clean_round(0, 16),
            // Defects at detectors 8, 9, 10 (3 defects from round 0->1).
            make_round(1, vec![
                false, false, false, false, false, false, false, false,
                true, true, true, false, false, false, false, false,
            ]),
            // Same syndrome persists => no new defects from round 1->2.
            make_round(2, vec![
                false, false, false, false, false, false, false, false,
                true, true, true, false, false, false, false, false,
            ]),
        ];

        let graph = DefectGraph::from_rounds(&rounds, 20);
        assert_eq!(graph.num_defects(), 3);

        // Boundary distances for d=20:
        // det 8: min(8, 11) = 8
        // det 9: min(9, 10) = 9
        // det 10: min(10, 9) = 9
        // Defect-defect distances: (8,9)=1, (9,10)=1, (8,10)=2
        // The closest edges are defect pairs at distance 1.

        // Greedy decoder
        let greedy = decode_greedy(&graph);
        // With 3 defects: 1 pair + 1 boundary match = all 3 defects handled.
        assert_eq!(greedy.pairs.len() * 2 + greedy.boundary_matches.len(), 3);
        assert_eq!(greedy.pairs.len(), 1,
            "Greedy should pair the two closest defects");
        assert_eq!(greedy.boundary_matches.len(), 1,
            "Greedy should boundary-match the remaining defect");

        // Union-find decoder
        let uf = decode_union_find(&graph);
        assert_eq!(uf.pairs.len() * 2 + uf.boundary_matches.len(), 3);
        assert!(uf.boundary_matches.len() >= 1,
            "Union-find should have at least 1 boundary match for 3 defects");
    }

    // ----------------------------------------------------------
    // Boundary distance calculation
    // ----------------------------------------------------------

    #[test]
    fn test_boundary_distance_calculation() {
        // Verify boundary distances are computed correctly for a distance-5
        // code. For detector j, boundary distance = min(j, d-1-j) clamped
        // to at least 1.
        let d = 5;

        // Create defects at specific detector positions to test distances.
        let rounds = vec![
            clean_round(0, 8),
            // Defects at detectors 0, 1, 2, 3, 4
            make_round(1, vec![true, true, true, true, true, false, false, false]),
            make_round(2, vec![true, true, true, true, true, false, false, false]),
        ];

        let graph = DefectGraph::from_rounds(&rounds, d);
        assert_eq!(graph.num_defects(), 5);

        // Expected boundary distances for d=5:
        // det 0: min(0, 4) = 0 => clamped to 1.0
        // det 1: min(1, 3) = 1 => max(1,1) = 1.0
        // det 2: min(2, 2) = 2 => 2.0
        // det 3: min(3, 1) = 1 => 1.0
        // det 4: min(4, 0) = 0 => clamped to 1.0
        let expected = vec![1.0, 1.0, 2.0, 1.0, 1.0];
        assert_eq!(graph.boundary_distances, expected);
    }

    // ----------------------------------------------------------
    // Boundary closer than pair: defect near edge prefers boundary
    // ----------------------------------------------------------

    #[test]
    fn test_boundary_closer_than_pair() {
        // Place two defects: one at detector 0 (boundary distance 1) and
        // one at detector 6 (boundary distance 1 for d=8). Their mutual
        // Manhattan distance is 6, which is much larger than either's
        // boundary distance. Both should prefer boundary matching.
        let d = 8;
        let rounds = vec![
            clean_round(0, 10),
            // Defects at detectors 0 and 6: far apart, both near boundaries.
            make_round(1, vec![true, false, false, false, false, false, true, false, false, false]),
            // Same syndrome persists => no new defects.
            make_round(2, vec![true, false, false, false, false, false, true, false, false, false]),
        ];

        let graph = DefectGraph::from_rounds(&rounds, d);
        assert_eq!(graph.num_defects(), 2);

        // Boundary distances:
        // det 0: min(0, 7) = 0 => clamped to 1
        // det 6: min(6, 1) = 1
        // Defect-defect distance: |0 - 6| = 6 (same round)
        // Both defects are much closer to boundary (1) than to each other (6).
        assert_eq!(graph.boundary_distances[0], 1.0);
        assert_eq!(graph.boundary_distances[1], 1.0);

        // Greedy: should match both to boundary since boundary edges (weight 1)
        // are lighter than the defect-defect edge (weight 6).
        let greedy = decode_greedy(&graph);
        assert_eq!(greedy.boundary_matches.len(), 2,
            "Both defects should match to boundary (distance 1) rather than each other (distance 6)");
        assert_eq!(greedy.pairs.len(), 0,
            "No pair matching should occur when boundary is closer");

        // Union-find: same behavior expected.
        let uf = decode_union_find(&graph);
        assert_eq!(uf.boundary_matches.len(), 2,
            "UF: Both defects should match to boundary");
        assert_eq!(uf.pairs.len(), 0,
            "UF: No pair matching when boundary is closer");
    }
}
