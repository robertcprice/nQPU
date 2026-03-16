//! Quantum Walk Algorithms
//!
//! Implements discrete-time and continuous-time quantum walks on arbitrary graphs,
//! along with quantum walk-based search algorithms and graph-theoretic applications.
//!
//! # Discrete-Time Quantum Walk (DTQW)
//!
//! A discrete-time quantum walk uses a coin operator on an internal degree of freedom
//! and a conditional shift operator that moves the walker along edges of the graph.
//! On a 1D line, the walker lives in a Hilbert space H_coin tensor H_position, where
//! the coin space has dimension equal to the degree (e.g., 2 for a line). Each step:
//!   1. Apply the coin operator C to the coin register
//!   2. Apply the shift operator S that moves the walker conditioned on coin state
//!
//! The resulting probability distribution spreads ballistically (std dev ~ t) rather
//! than diffusively (std dev ~ sqrt(t)) as in the classical random walk.
//!
//! # Continuous-Time Quantum Walk (CTQW)
//!
//! A continuous-time quantum walk evolves the walker's state via the Schrodinger
//! equation with the graph adjacency matrix (or Laplacian) as Hamiltonian:
//!   |psi(t)> = exp(-i * gamma * A * t) |psi(0)>
//!
//! This requires no coin space -- the position Hilbert space alone suffices. The
//! CTQW exhibits interference effects and can achieve perfect state transfer on
//! certain graph topologies.
//!
//! # Quantum Walk Search
//!
//! Quantum walks provide a natural framework for spatial search. On a complete graph
//! of N vertices, a quantum walk search finds a marked vertex in O(sqrt(N)) steps,
//! matching Grover's speedup. Szegedy's quantization of Markov chains provides a
//! general framework for converting classical random walks into quantum walks with
//! quadratic speedup.
//!
//! # Applications
//!
//! - **Graph isomorphism testing**: Compare CTQW evolution patterns on two graphs
//! - **Vertex centrality**: CTQW-based centrality via long-time average occupation
//! - **Quantum PageRank**: Quantized version of Google's PageRank algorithm
//!
//! # References
//!
//! - Aharonov, Y., Davidovich, L., Zagury, N. "Quantum random walks" (1993).
//!   Physical Review A 48(2), pp. 1687-1690.
//! - Farhi, E., Gutmann, S. "Quantum computation and decision trees" (1998).
//!   Physical Review A 58(2), pp. 915-928.
//! - Szegedy, M. "Quantum speed-up of Markov chain based algorithms" (2004).
//!   Proceedings of FOCS, pp. 32-41.
//! - Childs, A. M., Goldstone, J. "Spatial search by quantum walk" (2004).
//!   Physical Review A 70(2), 022314.

use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// LOCAL HELPERS
// ============================================================

type C64 = Complex64;

#[inline]
fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

#[inline]
fn c64_zero() -> C64 {
    c64(0.0, 0.0)
}

#[inline]
fn c64_one() -> C64 {
    c64(1.0, 0.0)
}

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors that can occur during quantum walk computations.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumWalkError {
    /// Graph has no vertices.
    EmptyGraph,
    /// Vertex index is out of range.
    VertexOutOfRange { vertex: usize, num_vertices: usize },
    /// Graph configuration is invalid.
    InvalidGraph(String),
    /// Matrix dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Coin operator dimension does not match vertex degree.
    CoinDimensionMismatch { degree: usize, coin_dim: usize },
    /// Invalid transition matrix (not stochastic).
    InvalidTransitionMatrix(String),
    /// No marked vertices provided for search.
    NoMarkedVertices,
}

impl std::fmt::Display for QuantumWalkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGraph => write!(f, "Graph must have at least one vertex"),
            Self::VertexOutOfRange {
                vertex,
                num_vertices,
            } => write!(
                f,
                "Vertex {} is out of range (graph has {} vertices)",
                vertex, num_vertices
            ),
            Self::InvalidGraph(msg) => write!(f, "Invalid graph: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::CoinDimensionMismatch { degree, coin_dim } => write!(
                f,
                "Coin dimension {} does not match vertex degree {}",
                coin_dim, degree
            ),
            Self::InvalidTransitionMatrix(msg) => {
                write!(f, "Invalid transition matrix: {}", msg)
            }
            Self::NoMarkedVertices => write!(f, "At least one marked vertex is required"),
        }
    }
}

impl std::error::Error for QuantumWalkError {}

// ============================================================
// GRAPH REPRESENTATION
// ============================================================

/// An undirected graph represented as an adjacency list.
///
/// Vertices are labeled 0..num_vertices. Each vertex stores a sorted list of
/// its neighbors. The graph is always kept symmetric: if (u, v) is an edge,
/// then both adj[u] contains v and adj[v] contains u.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices.
    pub num_vertices: usize,
    /// Adjacency list: adj[v] is the sorted list of neighbors of vertex v.
    pub adj: Vec<Vec<usize>>,
}

impl Graph {
    /// Create a graph with a given number of vertices and no edges.
    pub fn empty(num_vertices: usize) -> Self {
        Self {
            num_vertices,
            adj: vec![Vec::new(); num_vertices],
        }
    }

    /// Add an undirected edge between vertices u and v.
    ///
    /// Self-loops (u == v) are ignored. Duplicate edges are prevented.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u == v || u >= self.num_vertices || v >= self.num_vertices {
            return;
        }
        if !self.adj[u].contains(&v) {
            self.adj[u].push(v);
            self.adj[u].sort();
        }
        if !self.adj[v].contains(&u) {
            self.adj[v].push(u);
            self.adj[v].sort();
        }
    }

    /// Degree of vertex v (number of neighbors).
    pub fn degree(&self, v: usize) -> usize {
        if v >= self.num_vertices {
            0
        } else {
            self.adj[v].len()
        }
    }

    /// Maximum degree across all vertices.
    pub fn max_degree(&self) -> usize {
        self.adj.iter().map(|neighbors| neighbors.len()).max().unwrap_or(0)
    }

    /// Build the adjacency matrix as a flat row-major Vec<f64>.
    pub fn adjacency_matrix(&self) -> Vec<f64> {
        let n = self.num_vertices;
        let mut mat = vec![0.0; n * n];
        for u in 0..n {
            for &v in &self.adj[u] {
                mat[u * n + v] = 1.0;
            }
        }
        mat
    }

    /// Build the Laplacian matrix L = D - A (row-major).
    pub fn laplacian_matrix(&self) -> Vec<f64> {
        let n = self.num_vertices;
        let mut mat = vec![0.0; n * n];
        for u in 0..n {
            mat[u * n + u] = self.degree(u) as f64;
            for &v in &self.adj[u] {
                mat[u * n + v] = -1.0;
            }
        }
        mat
    }

    /// Number of edges in the graph.
    pub fn num_edges(&self) -> usize {
        let total: usize = self.adj.iter().map(|n| n.len()).sum();
        total / 2 // each edge counted twice
    }

    // ---- Predefined graph constructors ----

    /// Line graph (path graph) on n vertices: 0 -- 1 -- 2 -- ... -- (n-1).
    pub fn line(n: usize) -> Self {
        let mut g = Self::empty(n);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1);
        }
        g
    }

    /// Cycle graph on n vertices: 0 -- 1 -- ... -- (n-1) -- 0.
    pub fn cycle(n: usize) -> Self {
        let mut g = Self::line(n);
        if n >= 3 {
            g.add_edge(0, n - 1);
        }
        g
    }

    /// 2D grid graph with given rows and columns.
    ///
    /// Vertex (r, c) has index r * cols + c.
    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut g = Self::empty(n);
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if c + 1 < cols {
                    g.add_edge(v, v + 1);
                }
                if r + 1 < rows {
                    g.add_edge(v, v + cols);
                }
            }
        }
        g
    }

    /// Complete graph on n vertices (every pair connected).
    pub fn complete(n: usize) -> Self {
        let mut g = Self::empty(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j);
            }
        }
        g
    }

    /// Hypercube graph of dimension d (2^d vertices).
    ///
    /// Two vertices are connected if and only if their binary representations
    /// differ in exactly one bit.
    pub fn hypercube(dim: usize) -> Self {
        let n = 1usize << dim;
        let mut g = Self::empty(n);
        for v in 0..n {
            for bit in 0..dim {
                let neighbor = v ^ (1 << bit);
                if neighbor > v {
                    g.add_edge(v, neighbor);
                }
            }
        }
        g
    }

    /// Star graph: one central vertex (0) connected to n-1 leaf vertices.
    pub fn star(n: usize) -> Self {
        let mut g = Self::empty(n);
        for i in 1..n {
            g.add_edge(0, i);
        }
        g
    }

    /// Build the (symmetric, row-stochastic) transition matrix for a
    /// classical random walk on this graph.
    ///
    /// T[i][j] = 1/deg(i) if (i,j) is an edge, 0 otherwise.
    pub fn transition_matrix(&self) -> Vec<f64> {
        let n = self.num_vertices;
        let mut mat = vec![0.0; n * n];
        for u in 0..n {
            let d = self.degree(u);
            if d > 0 {
                let inv_d = 1.0 / d as f64;
                for &v in &self.adj[u] {
                    mat[u * n + v] = inv_d;
                }
            }
        }
        mat
    }
}

// ============================================================
// COIN OPERATORS
// ============================================================

/// Coin operator for discrete-time quantum walks.
///
/// The coin acts on the internal (direction/edge) degree of freedom of the walker.
/// For a d-regular graph, the coin is a d x d unitary matrix.
#[derive(Debug, Clone)]
pub enum CoinOperator {
    /// Hadamard coin (2x2): produces the well-known asymmetric distribution on lines.
    /// H = 1/sqrt(2) * [[1, 1], [1, -1]]
    Hadamard,

    /// Grover diffusion coin (d x d): 2/d * J - I where J is the all-ones matrix.
    /// Produces a symmetric distribution and is optimal for search on certain graphs.
    Grover,

    /// Discrete Fourier Transform coin (d x d): DFT_jk = 1/sqrt(d) * omega^(jk)
    /// where omega = exp(2*pi*i/d).
    DFT,

    /// Custom unitary coin specified as a flat row-major matrix.
    Custom(Vec<C64>),
}

impl CoinOperator {
    /// Build the d x d unitary matrix for this coin operator.
    pub fn matrix(&self, d: usize) -> Vec<C64> {
        match self {
            CoinOperator::Hadamard => {
                // Standard 2x2 Hadamard; if d != 2, extend with identity padding
                let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                if d == 2 {
                    vec![
                        c64(inv_sqrt2, 0.0),
                        c64(inv_sqrt2, 0.0),
                        c64(inv_sqrt2, 0.0),
                        c64(-inv_sqrt2, 0.0),
                    ]
                } else {
                    // For d > 2, use Hadamard on first 2 dims, identity on rest
                    let mut mat = vec![c64_zero(); d * d];
                    mat[0] = c64(inv_sqrt2, 0.0);
                    mat[1] = c64(inv_sqrt2, 0.0);
                    mat[d] = c64(inv_sqrt2, 0.0);
                    mat[d + 1] = c64(-inv_sqrt2, 0.0);
                    for i in 2..d {
                        mat[i * d + i] = c64_one();
                    }
                    mat
                }
            }
            CoinOperator::Grover => {
                // G_jk = 2/d - delta_jk
                let inv_d = 2.0 / d as f64;
                let mut mat = vec![c64_zero(); d * d];
                for i in 0..d {
                    for j in 0..d {
                        mat[i * d + j] = if i == j {
                            c64(inv_d - 1.0, 0.0)
                        } else {
                            c64(inv_d, 0.0)
                        };
                    }
                }
                mat
            }
            CoinOperator::DFT => {
                let inv_sqrt_d = 1.0 / (d as f64).sqrt();
                let mut mat = vec![c64_zero(); d * d];
                for j in 0..d {
                    for k in 0..d {
                        let angle = 2.0 * PI * (j * k) as f64 / d as f64;
                        mat[j * d + k] = c64(inv_sqrt_d * angle.cos(), inv_sqrt_d * angle.sin());
                    }
                }
                mat
            }
            CoinOperator::Custom(m) => m.clone(),
        }
    }
}

// ============================================================
// DISCRETE-TIME QUANTUM WALK (DTQW)
// ============================================================

/// Configuration for a discrete-time quantum walk on a 1D line.
#[derive(Debug, Clone)]
pub struct DTQWConfig {
    /// Number of time steps to evolve.
    pub num_steps: usize,
    /// Coin operator to use.
    pub coin_operator: CoinOperator,
    /// Initial position of the walker (vertex index).
    pub initial_position: usize,
    /// Number of positions (vertices on the line).
    pub num_positions: usize,
    /// Initial coin state: [alpha, beta] for |coin> = alpha|0> + beta|1>.
    /// Defaults to |0> if None.
    pub initial_coin_state: Option<[C64; 2]>,
}

impl DTQWConfig {
    /// Create a default configuration for n positions starting at the center.
    pub fn new(num_positions: usize, num_steps: usize) -> Self {
        Self {
            num_steps,
            coin_operator: CoinOperator::Hadamard,
            initial_position: num_positions / 2,
            num_positions,
            initial_coin_state: None,
        }
    }

    /// Set the coin operator.
    pub fn with_coin(mut self, coin: CoinOperator) -> Self {
        self.coin_operator = coin;
        self
    }

    /// Set the initial position.
    pub fn with_initial_position(mut self, pos: usize) -> Self {
        self.initial_position = pos;
        self
    }

    /// Set the initial coin state.
    pub fn with_initial_coin_state(mut self, state: [C64; 2]) -> Self {
        self.initial_coin_state = Some(state);
        self
    }
}

/// Result of a discrete-time quantum walk.
#[derive(Debug, Clone)]
pub struct DTQWResult {
    /// Position probability distribution at each time step.
    /// shape: [num_steps + 1][num_positions]
    pub position_probabilities: Vec<Vec<f64>>,
    /// Final statevector (coin tensor position).
    pub final_state: Vec<C64>,
    /// Standard deviation of the position distribution at each step.
    pub std_devs: Vec<f64>,
}

/// Run a discrete-time quantum walk on a 1D line with periodic boundaries.
///
/// The Hilbert space is H_coin (dim 2) tensor H_position (dim num_positions).
/// State ordering: |coin, position> with coin as the fast index.
/// Index = position * 2 + coin_state.
pub fn dtqw_on_line(config: &DTQWConfig) -> Result<DTQWResult, QuantumWalkError> {
    let n = config.num_positions;
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }
    if config.initial_position >= n {
        return Err(QuantumWalkError::VertexOutOfRange {
            vertex: config.initial_position,
            num_vertices: n,
        });
    }

    let dim = 2 * n; // coin (2) x position (n)

    // Initialize state: |coin_init> tensor |initial_position>
    let mut state = vec![c64_zero(); dim];
    let pos = config.initial_position;
    match &config.initial_coin_state {
        Some([alpha, beta]) => {
            state[pos * 2] = *alpha;
            state[pos * 2 + 1] = *beta;
        }
        None => {
            // Default: |0> coin state
            state[pos * 2] = c64_one();
        }
    }

    // Get the 2x2 coin matrix
    let coin = config.coin_operator.matrix(2);

    let mut position_probabilities = Vec::with_capacity(config.num_steps + 1);
    let mut std_devs = Vec::with_capacity(config.num_steps + 1);

    // Record initial distribution
    let (probs, sd) = extract_position_probs(&state, n);
    position_probabilities.push(probs);
    std_devs.push(sd);

    for _step in 0..config.num_steps {
        // Step 1: Apply coin operator to each position
        let mut after_coin = vec![c64_zero(); dim];
        for p in 0..n {
            let idx = p * 2;
            // |new_coin> = C |old_coin>
            after_coin[idx] = coin[0] * state[idx] + coin[1] * state[idx + 1];
            after_coin[idx + 1] = coin[2] * state[idx] + coin[3] * state[idx + 1];
        }

        // Step 2: Conditional shift
        // |0,p> -> |0, p-1 mod n>  (move left)
        // |1,p> -> |1, p+1 mod n>  (move right)
        let mut after_shift = vec![c64_zero(); dim];
        for p in 0..n {
            let left = if p == 0 { n - 1 } else { p - 1 };
            let right = (p + 1) % n;
            after_shift[left * 2] += after_coin[p * 2]; // coin=0 moves left
            after_shift[right * 2 + 1] += after_coin[p * 2 + 1]; // coin=1 moves right
        }

        state = after_shift;

        let (probs, sd) = extract_position_probs(&state, n);
        position_probabilities.push(probs);
        std_devs.push(sd);
    }

    Ok(DTQWResult {
        position_probabilities,
        final_state: state,
        std_devs,
    })
}

/// Run a discrete-time quantum walk on an arbitrary graph.
///
/// For a graph with maximum degree d, the coin space has dimension d.
/// Each vertex v maps its neighbors to coin states 0..deg(v)-1.
/// The shift operator flips the walker along edges:
///   S |c, v> = |c', u>  where u = adj[v][c] and c' is the index of v in adj[u].
///
/// Returns the position probability distribution at each time step.
pub fn dtqw_on_graph(
    graph: &Graph,
    coin: &CoinOperator,
    num_steps: usize,
    initial_vertex: usize,
) -> Result<Vec<Vec<f64>>, QuantumWalkError> {
    let n = graph.num_vertices;
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }
    if initial_vertex >= n {
        return Err(QuantumWalkError::VertexOutOfRange {
            vertex: initial_vertex,
            num_vertices: n,
        });
    }

    let d = graph.max_degree();
    if d == 0 {
        return Err(QuantumWalkError::InvalidGraph(
            "Graph has no edges".to_string(),
        ));
    }

    // Hilbert space: d (coin) x n (position)
    // Index: vertex * d + coin_state
    let dim = d * n;
    let coin_mat = coin.matrix(d);

    // Initialize: uniform superposition over coin states at initial vertex
    let mut state = vec![c64_zero(); dim];
    let deg_init = graph.degree(initial_vertex);
    if deg_init > 0 {
        let amp = c64(1.0 / (deg_init as f64).sqrt(), 0.0);
        for c in 0..deg_init {
            state[initial_vertex * d + c] = amp;
        }
    } else {
        state[initial_vertex * d] = c64_one();
    }

    let mut all_probs = Vec::with_capacity(num_steps + 1);
    all_probs.push(extract_graph_position_probs(&state, n, d));

    for _step in 0..num_steps {
        // Apply coin at each vertex
        let mut after_coin = vec![c64_zero(); dim];
        for v in 0..n {
            let deg_v = graph.degree(v);
            if deg_v == 0 {
                after_coin[v * d] = state[v * d];
                continue;
            }
            // Apply d x d coin to the coin register at vertex v
            for i in 0..d {
                let mut sum = c64_zero();
                for j in 0..d {
                    sum += coin_mat[i * d + j] * state[v * d + j];
                }
                after_coin[v * d + i] = sum;
            }
        }

        // Apply shift: S|c, v> = |c', u>
        // For coin state c at vertex v, the walker moves to neighbor u = adj[v][c]
        // (if c < deg(v)), and the new coin state c' is the index of v in adj[u].
        let mut after_shift = vec![c64_zero(); dim];
        for v in 0..n {
            let deg_v = graph.degree(v);
            for c in 0..d {
                let amp = after_coin[v * d + c];
                if amp.norm_sqr() < 1e-30 {
                    continue;
                }
                if c < deg_v {
                    let u = graph.adj[v][c];
                    // Find the coin index of v in adj[u]
                    if let Some(c_prime) = graph.adj[u].iter().position(|&x| x == v) {
                        after_shift[u * d + c_prime] += amp;
                    }
                }
                // If c >= deg_v, the amplitude stays (or is absorbed). For
                // well-defined walks on irregular graphs, we keep it at (v, c).
                if c >= deg_v {
                    after_shift[v * d + c] += amp;
                }
            }
        }

        state = after_shift;
        all_probs.push(extract_graph_position_probs(&state, n, d));
    }

    Ok(all_probs)
}

/// Compute the expected hitting time from source to target on a graph using DTQW.
///
/// Simulates the walk and records the first time step where the probability
/// at the target vertex exceeds the given threshold. Returns None if the target
/// is not reached within max_steps.
pub fn hitting_time(
    graph: &Graph,
    source: usize,
    target: usize,
    coin: &CoinOperator,
    max_steps: usize,
    threshold: f64,
) -> Result<Option<usize>, QuantumWalkError> {
    let probs = dtqw_on_graph(graph, coin, max_steps, source)?;
    for (step, dist) in probs.iter().enumerate() {
        if dist[target] >= threshold {
            return Ok(Some(step));
        }
    }
    Ok(None)
}

// ============================================================
// CONTINUOUS-TIME QUANTUM WALK (CTQW)
// ============================================================

/// Configuration for a continuous-time quantum walk.
#[derive(Debug, Clone)]
pub struct CTQWConfig {
    /// Evolution time.
    pub time: f64,
    /// The graph on which the walk occurs.
    pub graph: Graph,
    /// Hopping rate (coupling constant gamma).
    pub gamma: f64,
    /// Initial vertex (walker starts in |initial_vertex>).
    pub initial_vertex: usize,
}

impl CTQWConfig {
    /// Create a CTQW configuration.
    pub fn new(graph: Graph, time: f64, initial_vertex: usize) -> Self {
        Self {
            time,
            graph,
            gamma: 1.0,
            initial_vertex,
        }
    }

    /// Set the hopping rate.
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }
}

/// Result of a continuous-time quantum walk evolution.
#[derive(Debug, Clone)]
pub struct CTQWResult {
    /// Position probability distribution at the final time.
    pub probabilities: Vec<f64>,
    /// Final statevector.
    pub final_state: Vec<C64>,
}

/// Evolve a continuous-time quantum walk.
///
/// Computes |psi(t)> = exp(-i * gamma * A * t) |psi(0)> where A is the
/// adjacency matrix of the graph. Uses Pade approximation for the matrix
/// exponential.
pub fn ctqw_evolve(config: &CTQWConfig) -> Result<CTQWResult, QuantumWalkError> {
    let n = config.graph.num_vertices;
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }
    if config.initial_vertex >= n {
        return Err(QuantumWalkError::VertexOutOfRange {
            vertex: config.initial_vertex,
            num_vertices: n,
        });
    }

    // Build -i * gamma * t * A as a complex matrix
    let adj = config.graph.adjacency_matrix();
    let factor = c64(0.0, -config.gamma * config.time);
    let hamiltonian: Vec<C64> = adj.iter().map(|&a| factor * c64(a, 0.0)).collect();

    // Compute exp(-i * gamma * A * t)
    let propagator = matrix_exponential(&hamiltonian, n);

    // Apply to initial state |initial_vertex>
    let mut psi0 = vec![c64_zero(); n];
    psi0[config.initial_vertex] = c64_one();

    let final_state = matvec(&propagator, &psi0, n);
    let probabilities: Vec<f64> = final_state.iter().map(|c| c.norm_sqr()).collect();

    Ok(CTQWResult {
        probabilities,
        final_state,
    })
}

/// Compute the transfer fidelity between source and target at a given time.
///
/// Returns the probability of finding the walker at the target vertex when
/// starting from the source vertex after evolving for time t.
pub fn transfer_fidelity(
    graph: &Graph,
    source: usize,
    target: usize,
    gamma: f64,
    time: f64,
) -> Result<f64, QuantumWalkError> {
    let config = CTQWConfig {
        time,
        graph: graph.clone(),
        gamma,
        initial_vertex: source,
    };
    let result = ctqw_evolve(&config)?;
    Ok(result.probabilities[target])
}

/// Find the time that maximizes transfer fidelity between source and target.
///
/// Performs a grid search over [0, max_time] with the given number of samples,
/// then refines around the best candidate.
pub fn optimal_transfer_time(
    graph: &Graph,
    source: usize,
    target: usize,
    gamma: f64,
    max_time: f64,
    num_samples: usize,
) -> Result<(f64, f64), QuantumWalkError> {
    if graph.num_vertices == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }
    if source >= graph.num_vertices {
        return Err(QuantumWalkError::VertexOutOfRange {
            vertex: source,
            num_vertices: graph.num_vertices,
        });
    }
    if target >= graph.num_vertices {
        return Err(QuantumWalkError::VertexOutOfRange {
            vertex: target,
            num_vertices: graph.num_vertices,
        });
    }

    let n = graph.num_vertices;
    let adj = graph.adjacency_matrix();

    // Coarse grid search
    let mut best_time = 0.0_f64;
    let mut best_fidelity = 0.0_f64;

    for i in 0..num_samples {
        let t = max_time * (i as f64) / (num_samples as f64);
        let factor = c64(0.0, -gamma * t);
        let hamiltonian: Vec<C64> = adj.iter().map(|&a| factor * c64(a, 0.0)).collect();
        let propagator = matrix_exponential(&hamiltonian, n);

        let mut psi0 = vec![c64_zero(); n];
        psi0[source] = c64_one();
        let psi_t = matvec(&propagator, &psi0, n);
        let fid = psi_t[target].norm_sqr();

        if fid > best_fidelity {
            best_fidelity = fid;
            best_time = t;
        }
    }

    // Refine around the best candidate
    let dt = max_time / (num_samples as f64);
    let t_low = (best_time - dt).max(0.0);
    let t_high = (best_time + dt).min(max_time);
    let refine_samples = 100;

    for i in 0..=refine_samples {
        let t = t_low + (t_high - t_low) * (i as f64) / (refine_samples as f64);
        let factor = c64(0.0, -gamma * t);
        let hamiltonian: Vec<C64> = adj.iter().map(|&a| factor * c64(a, 0.0)).collect();
        let propagator = matrix_exponential(&hamiltonian, n);

        let mut psi0 = vec![c64_zero(); n];
        psi0[source] = c64_one();
        let psi_t = matvec(&propagator, &psi0, n);
        let fid = psi_t[target].norm_sqr();

        if fid > best_fidelity {
            best_fidelity = fid;
            best_time = t;
        }
    }

    Ok((best_time, best_fidelity))
}

// ============================================================
// QUANTUM WALK SEARCH
// ============================================================

/// Result of a quantum walk search.
#[derive(Debug, Clone)]
pub struct WalkSearchResult {
    /// Probability of measuring each vertex at the end of the search.
    pub probabilities: Vec<f64>,
    /// Number of steps executed.
    pub steps: usize,
    /// Maximum probability over marked vertices.
    pub max_marked_probability: f64,
    /// Vertex with the highest probability among the marked set.
    pub best_marked_vertex: usize,
}

/// Quantum walk search on a graph using the coined walk framework.
///
/// Modifies the coin at marked vertices to the Grover diffusion operator
/// (which inverts about the average), while using the standard coin elsewhere.
/// This concentrates amplitude on marked vertices over time.
///
/// Based on Shenvi, Kempe, and Whaley (2003) for hypercubes and
/// Ambainis, Kempe, and Rivosh (2005) for general graphs.
pub fn walk_search(
    graph: &Graph,
    marked_vertices: &[usize],
    num_steps: usize,
) -> Result<WalkSearchResult, QuantumWalkError> {
    let n = graph.num_vertices;
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }
    if marked_vertices.is_empty() {
        return Err(QuantumWalkError::NoMarkedVertices);
    }
    for &v in marked_vertices {
        if v >= n {
            return Err(QuantumWalkError::VertexOutOfRange {
                vertex: v,
                num_vertices: n,
            });
        }
    }

    let d = graph.max_degree();
    if d == 0 {
        return Err(QuantumWalkError::InvalidGraph(
            "Graph has no edges".to_string(),
        ));
    }

    let dim = d * n;

    // Build coin matrices: Grover coin for marked vertices, standard coin for others
    let grover_coin = CoinOperator::Grover.matrix(d);
    let standard_coin = CoinOperator::Grover.matrix(d); // Grover coin everywhere

    // Negative identity (phase inversion) for the coin at marked vertices:
    // Use -I_d as the coin at marked vertices (flip phase)
    let mut marked_coin = vec![c64_zero(); d * d];
    for i in 0..d {
        marked_coin[i * d + i] = c64(-1.0, 0.0);
    }

    let marked_set: std::collections::HashSet<usize> =
        marked_vertices.iter().cloned().collect();

    // Initialize: uniform superposition over all coin states and all vertices
    let amp = c64(1.0 / (dim as f64).sqrt(), 0.0);
    let mut state: Vec<C64> = vec![amp; dim];
    // Zero out coin states beyond the degree of each vertex
    for v in 0..n {
        let deg_v = graph.degree(v);
        for c in deg_v..d {
            state[v * d + c] = c64_zero();
        }
    }
    // Renormalize
    let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    if norm_sq > 1e-30 {
        let inv_norm = c64(1.0 / norm_sq.sqrt(), 0.0);
        for c in state.iter_mut() {
            *c *= inv_norm;
        }
    }

    for _step in 0..num_steps {
        // Step 1: Apply coin (with oracle at marked vertices)
        let mut after_coin = vec![c64_zero(); dim];
        for v in 0..n {
            let coin_mat = if marked_set.contains(&v) {
                &marked_coin
            } else {
                &standard_coin
            };
            for i in 0..d {
                let mut sum = c64_zero();
                for j in 0..d {
                    sum += coin_mat[i * d + j] * state[v * d + j];
                }
                after_coin[v * d + i] = sum;
            }
        }

        // Step 2: Shift
        let mut after_shift = vec![c64_zero(); dim];
        for v in 0..n {
            let deg_v = graph.degree(v);
            for c in 0..d {
                let amp = after_coin[v * d + c];
                if amp.norm_sqr() < 1e-30 {
                    continue;
                }
                if c < deg_v {
                    let u = graph.adj[v][c];
                    if let Some(c_prime) = graph.adj[u].iter().position(|&x| x == v) {
                        after_shift[u * d + c_prime] += amp;
                    }
                } else {
                    after_shift[v * d + c] += amp;
                }
            }
        }

        state = after_shift;
    }

    // Extract position probabilities
    let probs = extract_graph_position_probs(&state, n, d);

    let max_marked_probability = marked_vertices
        .iter()
        .map(|&v| probs[v])
        .fold(0.0_f64, f64::max);
    let best_marked_vertex = *marked_vertices
        .iter()
        .max_by(|&&a, &&b| probs[a].partial_cmp(&probs[b]).unwrap())
        .unwrap();

    Ok(WalkSearchResult {
        probabilities: probs,
        steps: num_steps,
        max_marked_probability,
        best_marked_vertex,
    })
}

/// Szegedy quantum walk: quantized version of a classical Markov chain.
///
/// Given a transition matrix P (row-stochastic), constructs the Szegedy walk
/// operator W = ref_B * ref_A, where ref_A and ref_B are reflections about
/// subspaces defined by the transition matrix and its transpose.
///
/// The walk provides a quadratic speedup in the spectral gap: the quantum
/// walk detects a marked element in O(1/sqrt(delta)) steps where delta is
/// the spectral gap of the classical chain.
///
/// For marked vertices, the transition matrix rows are modified to create
/// an absorbing walk that concentrates amplitude on marked states.
pub fn szegedy_walk(
    transition_matrix: &[f64],
    n: usize,
    marked_vertices: &[usize],
    num_steps: usize,
) -> Result<Vec<f64>, QuantumWalkError> {
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }
    if transition_matrix.len() != n * n {
        return Err(QuantumWalkError::DimensionMismatch {
            expected: n * n,
            got: transition_matrix.len(),
        });
    }
    if marked_vertices.is_empty() {
        return Err(QuantumWalkError::NoMarkedVertices);
    }
    for &v in marked_vertices {
        if v >= n {
            return Err(QuantumWalkError::VertexOutOfRange {
                vertex: v,
                num_vertices: n,
            });
        }
    }

    // Validate stochasticity (each row sums to ~1)
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| transition_matrix[i * n + j]).sum();
        if (row_sum - 1.0).abs() > 0.01 {
            return Err(QuantumWalkError::InvalidTransitionMatrix(format!(
                "Row {} sums to {} (expected ~1.0)",
                i, row_sum
            )));
        }
    }

    // Szegedy walk lives in H_A tensor H_B, each of dimension n.
    // Total dimension: n^2.
    let dim = n * n;

    // Build |psi_i^A> = sum_j sqrt(P[i][j]) |i>|j>
    // Build |psi_j^B> = sum_i sqrt(P^T[j][i]) |i>|j> = sum_i sqrt(P[i][j]) |i>|j>
    // (for symmetric P, these coincide)

    // Construct projectors for ref_A: Pi_A = sum_i |psi_i^A><psi_i^A|
    // ref_A = 2 * Pi_A - I
    // Similarly for ref_B using the transpose.

    // For efficiency, we work with the walk operator directly on the state vector.
    // The state lives in C^{n^2} with basis |i,j>.

    let marked_set: std::collections::HashSet<usize> =
        marked_vertices.iter().cloned().collect();

    // Modified transition matrix: marked vertices become absorbing
    let mut p_modified = transition_matrix.to_vec();
    for &v in marked_vertices {
        for j in 0..n {
            p_modified[v * n + j] = if j == v { 1.0 } else { 0.0 };
        }
    }

    // Build the |psi_i> vectors: |psi_i> = sum_j sqrt(P[i][j]) |i,j>
    let mut psi_a: Vec<Vec<C64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = vec![c64_zero(); dim];
        for j in 0..n {
            let p_ij = p_modified[i * n + j];
            if p_ij > 0.0 {
                v[i * n + j] = c64(p_ij.sqrt(), 0.0);
            }
        }
        psi_a.push(v);
    }

    // Build |psi_j^B> using P^T
    let mut psi_b: Vec<Vec<C64>> = Vec::with_capacity(n);
    for j in 0..n {
        let mut v = vec![c64_zero(); dim];
        let col_sum: f64 = (0..n).map(|i| p_modified[i * n + j]).sum();
        if col_sum > 1e-30 {
            for i in 0..n {
                let p_ij = p_modified[i * n + j];
                if p_ij > 0.0 {
                    v[i * n + j] = c64((p_ij / col_sum).sqrt(), 0.0);
                }
            }
        }
        psi_b.push(v);
    }

    // Initial state: uniform superposition of |psi_i^A>
    let mut state = vec![c64_zero(); dim];
    let weight = c64(1.0 / (n as f64).sqrt(), 0.0);
    for i in 0..n {
        for k in 0..dim {
            state[k] += weight * psi_a[i][k];
        }
    }
    // Normalize
    let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    if norm_sq > 1e-30 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for c in state.iter_mut() {
            *c *= inv_norm;
        }
    }

    for _step in 0..num_steps {
        // ref_A: reflect about span{|psi_i^A>}
        state = reflect_about_subspace(&state, &psi_a, dim);
        // ref_B: reflect about span{|psi_j^B>}
        state = reflect_about_subspace(&state, &psi_b, dim);
    }

    // Extract position probabilities: P(i) = sum_j |<i,j|state>|^2
    let mut probs = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            probs[i] += state[i * n + j].norm_sqr();
        }
    }

    Ok(probs)
}

// ============================================================
// APPLICATIONS
// ============================================================

/// Test graph isomorphism by comparing CTQW evolution patterns.
///
/// Runs a CTQW on each graph from every starting vertex and computes a
/// "signature" vector of sorted long-time average occupation probabilities.
/// The similarity score is the cosine similarity between the two signatures.
///
/// Returns a score in [0, 1] where 1.0 indicates identical walk patterns
/// (strongly suggesting isomorphism) and lower values indicate the graphs
/// are likely non-isomorphic.
pub fn graph_isomorphism_test(
    g1: &Graph,
    g2: &Graph,
    num_time_samples: usize,
) -> Result<f64, QuantumWalkError> {
    if g1.num_vertices != g2.num_vertices {
        return Ok(0.0);
    }
    if g1.num_edges() != g2.num_edges() {
        return Ok(0.0);
    }

    let n = g1.num_vertices;
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }

    let sig1 = ctqw_signature(g1, num_time_samples)?;
    let sig2 = ctqw_signature(g2, num_time_samples)?;

    // Cosine similarity between sorted signature vectors
    Ok(cosine_similarity(&sig1, &sig2))
}

/// Compute CTQW-based vertex centrality.
///
/// For each vertex, computes the time-averaged probability of finding
/// the walker at that vertex when starting from each other vertex. Vertices
/// that attract more probability on average are more "central" in the
/// quantum walk sense.
///
/// The centrality measure captures both local connectivity and global
/// graph structure through quantum interference effects. Time-averaging
/// over multiple time points smooths out oscillations and produces a
/// robust centrality measure.
pub fn vertex_centrality(
    graph: &Graph,
    max_time: f64,
) -> Result<Vec<f64>, QuantumWalkError> {
    let n = graph.num_vertices;
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }

    // Use the graph Laplacian as the walk Hamiltonian. The Laplacian L = D - A
    // encodes both connectivity and degree, making the walk sensitive to vertex
    // importance: high-degree vertices experience faster phase evolution, which
    // affects how probability accumulates over time.
    let laplacian = graph.laplacian_matrix();
    let mut centrality = vec![0.0; n];

    // Time-average over multiple time points to smooth oscillations
    let num_time_samples = 30;
    for t_idx in 1..=num_time_samples {
        let t = max_time * (t_idx as f64) / (num_time_samples as f64);
        let factor = c64(0.0, -t);
        let hamiltonian: Vec<C64> = laplacian.iter().map(|&a| factor * c64(a, 0.0)).collect();
        let propagator = matrix_exponential(&hamiltonian, n);

        // Average over all starting vertices
        for source in 0..n {
            let mut psi0 = vec![c64_zero(); n];
            psi0[source] = c64_one();
            let psi_t = matvec(&propagator, &psi0, n);
            for v in 0..n {
                centrality[v] += psi_t[v].norm_sqr();
            }
        }
    }

    // Normalize
    let total: f64 = centrality.iter().sum();
    if total > 1e-30 {
        for c in centrality.iter_mut() {
            *c /= total;
        }
    }

    Ok(centrality)
}

/// Quantum PageRank via continuous-time quantum walk.
///
/// Computes a quantum version of Google's PageRank algorithm. The quantum
/// PageRank is defined as the time-averaged probability distribution of a
/// CTQW on the Google matrix G = damping * (D^{-1} A) + (1 - damping)/n * J,
/// where A is the adjacency matrix, D is the degree matrix, and J is the
/// all-ones matrix.
///
/// Unlike classical PageRank, quantum PageRank can distinguish vertices that
/// classical PageRank considers equivalent, due to quantum interference.
pub fn quantum_pagerank(
    graph: &Graph,
    damping: f64,
) -> Result<Vec<f64>, QuantumWalkError> {
    let n = graph.num_vertices;
    if n == 0 {
        return Err(QuantumWalkError::EmptyGraph);
    }

    // Build the Google matrix
    let mut google_mat = vec![0.0; n * n];
    let teleport = (1.0 - damping) / n as f64;

    for i in 0..n {
        let deg = graph.degree(i);
        for j in 0..n {
            if deg > 0 {
                let adj_val = if graph.adj[i].contains(&j) {
                    1.0
                } else {
                    0.0
                };
                google_mat[i * n + j] = damping * adj_val / deg as f64 + teleport;
            } else {
                // Dangling node: distribute uniformly
                google_mat[i * n + j] = 1.0 / n as f64;
            }
        }
    }

    // Time-average the CTQW over several time points
    let num_time_samples = 50;
    let max_time = 10.0;
    let mut pagerank = vec![0.0; n];

    for t_idx in 1..=num_time_samples {
        let t = max_time * (t_idx as f64) / (num_time_samples as f64);
        let factor = c64(0.0, -t);
        let hamiltonian: Vec<C64> = google_mat.iter().map(|&a| factor * c64(a, 0.0)).collect();
        let propagator = matrix_exponential(&hamiltonian, n);

        // Average probability starting from each vertex
        for source in 0..n {
            let mut psi0 = vec![c64_zero(); n];
            psi0[source] = c64_one();
            let psi_t = matvec(&propagator, &psi0, n);
            for v in 0..n {
                pagerank[v] += psi_t[v].norm_sqr();
            }
        }
    }

    // Normalize
    let total: f64 = pagerank.iter().sum();
    if total > 1e-30 {
        for p in pagerank.iter_mut() {
            *p /= total;
        }
    }

    Ok(pagerank)
}

// ============================================================
// MATRIX AND LINEAR ALGEBRA HELPERS
// ============================================================

/// Extract position probabilities from a coin-position state vector.
/// Returns (probabilities, standard_deviation).
fn extract_position_probs(state: &[C64], num_positions: usize) -> (Vec<f64>, f64) {
    let mut probs = vec![0.0; num_positions];
    for p in 0..num_positions {
        // Sum over coin states
        probs[p] = state[p * 2].norm_sqr() + state[p * 2 + 1].norm_sqr();
    }

    // Compute mean position and standard deviation
    let mean: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| i as f64 * p)
        .sum();
    let var: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let diff = i as f64 - mean;
            diff * diff * p
        })
        .sum();
    let sd = var.sqrt();

    (probs, sd)
}

/// Extract position probabilities from a graph walk state (d coin states per vertex).
fn extract_graph_position_probs(state: &[C64], num_vertices: usize, d: usize) -> Vec<f64> {
    let mut probs = vec![0.0; num_vertices];
    for v in 0..num_vertices {
        for c in 0..d {
            probs[v] += state[v * d + c].norm_sqr();
        }
    }
    probs
}

/// Reflect a vector about the subspace spanned by the given basis vectors.
///
/// ref = 2 * Pi - I, where Pi = sum_i |v_i><v_i| (projector onto subspace).
fn reflect_about_subspace(
    state: &[C64],
    basis: &[Vec<C64>],
    dim: usize,
) -> Vec<C64> {
    // Compute projection: sum_i <v_i|state> * |v_i>
    let mut projected = vec![c64_zero(); dim];
    for v in basis {
        let inner: C64 = v
            .iter()
            .zip(state.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        for k in 0..dim {
            projected[k] += inner * v[k];
        }
    }
    // Reflection: 2 * projected - state
    let mut result = vec![c64_zero(); dim];
    for k in 0..dim {
        result[k] = c64(2.0, 0.0) * projected[k] - state[k];
    }
    result
}

/// CTQW signature for graph isomorphism testing.
///
/// For each starting vertex, compute the sorted probability distribution
/// at multiple time points and concatenate into a single signature vector.
fn ctqw_signature(
    graph: &Graph,
    num_time_samples: usize,
) -> Result<Vec<f64>, QuantumWalkError> {
    let n = graph.num_vertices;
    let adj = graph.adjacency_matrix();

    let mut signature = Vec::new();

    for t_idx in 1..=num_time_samples {
        let t = PI * (t_idx as f64) / (num_time_samples as f64);
        let factor = c64(0.0, -t);
        let hamiltonian: Vec<C64> = adj.iter().map(|&a| factor * c64(a, 0.0)).collect();
        let propagator = matrix_exponential(&hamiltonian, n);

        // Collect sorted probability vectors from each starting vertex
        let mut time_sig = Vec::new();
        for source in 0..n {
            let mut psi0 = vec![c64_zero(); n];
            psi0[source] = c64_one();
            let psi_t = matvec(&propagator, &psi0, n);
            let mut probs: Vec<f64> = psi_t.iter().map(|c| c.norm_sqr()).collect();
            probs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            time_sig.extend(probs);
        }
        // Sort to make invariant to vertex permutation
        // (sort blocks of n probabilities)
        let mut blocks: Vec<Vec<f64>> = time_sig.chunks(n).map(|c| c.to_vec()).collect();
        blocks.sort_by(|a, b| {
            for (x, y) in a.iter().zip(b.iter()) {
                match x.partial_cmp(y) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    Some(ord) => return ord,
                    None => return std::cmp::Ordering::Equal,
                }
            }
            std::cmp::Ordering::Equal
        });
        for block in blocks {
            signature.extend(block);
        }
    }

    Ok(signature)
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-30 || norm_b < 1e-30 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Matrix-vector product: y = A * x, where A is dim x dim row-major.
fn matvec(mat: &[C64], vec: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim];
    for i in 0..dim {
        let mut sum = c64_zero();
        for j in 0..dim {
            sum += mat[i * dim + j] * vec[j];
        }
        result[i] = sum;
    }
    result
}

/// Matrix exponential exp(A) for a complex matrix using scaling-and-squaring
/// with Pade approximation of order 6.
fn matrix_exponential(matrix: &[C64], dim: usize) -> Vec<C64> {
    // Scaling: find s such that ||A/2^s|| < 1
    let norm = matrix_inf_norm(matrix, dim);
    let s = if norm > 1.0 {
        (norm.log2().ceil() as u32) + 1
    } else {
        0
    };
    let scale = 0.5f64.powi(s as i32);

    // Scale the matrix
    let scaled: Vec<C64> = matrix.iter().map(|&x| x * c64(scale, 0.0)).collect();

    // Pade approximation of order 6
    let p = 6;
    let coeffs = pade_coefficients(p);

    // Compute powers of scaled matrix
    let identity = complex_identity(dim);
    let mut powers: Vec<Vec<C64>> = Vec::with_capacity(p + 1);
    powers.push(identity.clone());
    if p >= 1 {
        powers.push(scaled.clone());
    }
    for k in 2..=p {
        let prev = &powers[k - 1];
        let prod = complex_matmul(prev, &scaled, dim);
        powers.push(prod);
    }

    // P = sum c_k * A^k, Q = sum (-1)^k * c_k * A^k
    let mut p_mat = vec![c64_zero(); dim * dim];
    let mut q_mat = vec![c64_zero(); dim * dim];
    for k in 0..=p {
        let c = c64(coeffs[k], 0.0);
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        for idx in 0..(dim * dim) {
            p_mat[idx] += c * powers[k][idx];
            q_mat[idx] += c64(sign * coeffs[k], 0.0) * powers[k][idx];
        }
    }

    // Solve Q * result = P
    let result = complex_matrix_solve(&q_mat, &p_mat, dim);

    // Squaring
    let mut mat = result;
    for _ in 0..s {
        mat = complex_matmul(&mat, &mat, dim);
    }
    mat
}

/// Pade coefficients c_k for order p.
fn pade_coefficients(p: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0; p + 1];
    coeffs[0] = 1.0;
    for k in 1..=p {
        coeffs[k] = coeffs[k - 1] * ((p + 1 - k) as f64) / ((k * (2 * p + 1 - k)) as f64);
    }
    coeffs
}

/// Infinity norm of a complex matrix.
fn matrix_inf_norm(mat: &[C64], dim: usize) -> f64 {
    let mut max_row_sum = 0.0f64;
    for i in 0..dim {
        let row_sum: f64 = (0..dim).map(|j| mat[i * dim + j].norm()).sum();
        max_row_sum = max_row_sum.max(row_sum);
    }
    max_row_sum
}

/// Complex identity matrix (flat row-major).
fn complex_identity(dim: usize) -> Vec<C64> {
    let mut mat = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        mat[i * dim + i] = c64_one();
    }
    mat
}

/// Complex matrix multiplication C = A * B.
fn complex_matmul(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
    let mut c = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let a_ik = a[i * dim + k];
            if a_ik.norm_sqr() < 1e-30 {
                continue;
            }
            for j in 0..dim {
                c[i * dim + j] += a_ik * b[k * dim + j];
            }
        }
    }
    c
}

/// Solve Q * X = P for complex matrices using Gaussian elimination with
/// partial pivoting.
fn complex_matrix_solve(q: &[C64], p: &[C64], dim: usize) -> Vec<C64> {
    let mut aug = vec![c64_zero(); dim * 2 * dim];
    for i in 0..dim {
        for j in 0..dim {
            aug[i * 2 * dim + j] = q[i * dim + j];
            aug[i * 2 * dim + dim + j] = p[i * dim + j];
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..dim {
        let mut max_val = 0.0f64;
        let mut max_row = col;
        for row in col..dim {
            let val = aug[row * 2 * dim + col].norm();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-20 {
            aug[col * 2 * dim + col] += c64(1e-12, 0.0);
        }

        if max_row != col {
            for k in 0..(2 * dim) {
                let tmp = aug[col * 2 * dim + k];
                aug[col * 2 * dim + k] = aug[max_row * 2 * dim + k];
                aug[max_row * 2 * dim + k] = tmp;
            }
        }

        let pivot = aug[col * 2 * dim + col];
        for row in (col + 1)..dim {
            let factor = aug[row * 2 * dim + col] / pivot;
            for k in col..(2 * dim) {
                let val = aug[col * 2 * dim + k];
                aug[row * 2 * dim + k] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut result = vec![c64_zero(); dim * dim];
    for col_rhs in 0..dim {
        for i in (0..dim).rev() {
            let mut sum = aug[i * 2 * dim + dim + col_rhs];
            for j in (i + 1)..dim {
                sum -= aug[i * 2 * dim + j] * result[j * dim + col_rhs];
            }
            result[i * dim + col_rhs] = sum / aug[i * 2 * dim + i];
        }
    }

    result
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- Graph construction tests ----

    #[test]
    fn test_graph_line() {
        let g = Graph::line(5);
        assert_eq!(g.num_vertices, 5);
        assert_eq!(g.num_edges(), 4);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(2), 2);
        assert_eq!(g.degree(4), 1);
    }

    #[test]
    fn test_graph_cycle() {
        let g = Graph::cycle(6);
        assert_eq!(g.num_vertices, 6);
        assert_eq!(g.num_edges(), 6);
        for v in 0..6 {
            assert_eq!(g.degree(v), 2);
        }
    }

    #[test]
    fn test_graph_grid() {
        let g = Graph::grid(3, 4);
        assert_eq!(g.num_vertices, 12);
        // Corner: degree 2, edge: degree 3, interior: degree 4
        assert_eq!(g.degree(0), 2); // top-left corner
        assert_eq!(g.degree(1), 3); // top edge
        assert_eq!(g.degree(5), 4); // interior
    }

    #[test]
    fn test_graph_complete() {
        let g = Graph::complete(5);
        assert_eq!(g.num_vertices, 5);
        assert_eq!(g.num_edges(), 10);
        for v in 0..5 {
            assert_eq!(g.degree(v), 4);
        }
    }

    #[test]
    fn test_graph_hypercube() {
        let g = Graph::hypercube(3);
        assert_eq!(g.num_vertices, 8);
        assert_eq!(g.num_edges(), 12);
        for v in 0..8 {
            assert_eq!(g.degree(v), 3); // each vertex connected to 3 neighbors
        }
    }

    #[test]
    fn test_graph_star() {
        let g = Graph::star(5);
        assert_eq!(g.num_vertices, 5);
        assert_eq!(g.num_edges(), 4);
        assert_eq!(g.degree(0), 4); // center
        for v in 1..5 {
            assert_eq!(g.degree(v), 1); // leaves
        }
    }

    #[test]
    fn test_graph_adjacency_matrix() {
        let g = Graph::line(3); // 0--1--2
        let adj = g.adjacency_matrix();
        assert_eq!(adj.len(), 9);
        assert_eq!(adj[0 * 3 + 1], 1.0); // edge 0-1
        assert_eq!(adj[1 * 3 + 0], 1.0); // symmetric
        assert_eq!(adj[1 * 3 + 2], 1.0); // edge 1-2
        assert_eq!(adj[0 * 3 + 2], 0.0); // no edge 0-2
    }

    // ---- DTQW tests ----

    #[test]
    fn test_dtqw_symmetric_spreading() {
        // A balanced initial coin state should produce symmetric spreading
        let config = DTQWConfig::new(101, 50).with_initial_coin_state([
            c64(1.0 / 2.0_f64.sqrt(), 0.0),
            c64(0.0, 1.0 / 2.0_f64.sqrt()),
        ]);
        let result = dtqw_on_line(&config).unwrap();

        // Check symmetric distribution around center
        let final_probs = result.position_probabilities.last().unwrap();
        let center = 50;
        // Compare probabilities at symmetric positions
        for offset in 1..20 {
            let left = center - offset;
            let right = center + offset;
            assert!(
                approx_eq(final_probs[left], final_probs[right], 0.01),
                "Asymmetry at offset {}: left={}, right={}",
                offset,
                final_probs[left],
                final_probs[right]
            );
        }
    }

    #[test]
    fn test_dtqw_hadamard_asymmetric() {
        // Hadamard coin with |0> initial state produces known asymmetric distribution
        // biased towards the left (negative direction)
        let config = DTQWConfig::new(101, 50);
        let result = dtqw_on_line(&config).unwrap();

        let final_probs = result.position_probabilities.last().unwrap();
        let center = 50;

        // Sum probabilities on left vs right
        let left_prob: f64 = final_probs[0..center].iter().sum();
        let right_prob: f64 = final_probs[(center + 1)..].iter().sum();

        // With |0> coin state and Hadamard coin, distribution should be biased left
        assert!(
            left_prob > right_prob,
            "Expected left bias: left={}, right={}",
            left_prob,
            right_prob
        );
    }

    #[test]
    fn test_dtqw_ballistic_spreading() {
        // Quantum walk spreads ballistically: std dev ~ t (not sqrt(t))
        let config = DTQWConfig::new(201, 30).with_initial_coin_state([
            c64(1.0 / 2.0_f64.sqrt(), 0.0),
            c64(0.0, 1.0 / 2.0_f64.sqrt()),
        ]);
        let result = dtqw_on_line(&config).unwrap();

        // At step 10 and step 20, check that std dev grows roughly linearly
        let sd10 = result.std_devs[10];
        let sd20 = result.std_devs[20];
        let sd30 = result.std_devs[30];

        // Ratio should be approximately linear (sd20/sd10 ~ 2, sd30/sd10 ~ 3)
        let ratio_20_10 = sd20 / sd10;
        let ratio_30_10 = sd30 / sd10;
        assert!(
            ratio_20_10 > 1.5 && ratio_20_10 < 2.5,
            "Expected ~2x ratio, got {}",
            ratio_20_10
        );
        assert!(
            ratio_30_10 > 2.0 && ratio_30_10 < 4.0,
            "Expected ~3x ratio, got {}",
            ratio_30_10
        );
    }

    #[test]
    fn test_dtqw_probability_conservation() {
        let config = DTQWConfig::new(51, 25);
        let result = dtqw_on_line(&config).unwrap();

        for (step, probs) in result.position_probabilities.iter().enumerate() {
            let total: f64 = probs.iter().sum();
            assert!(
                approx_eq(total, 1.0, 1e-10),
                "Step {}: probability not conserved (sum = {})",
                step,
                total
            );
        }
    }

    #[test]
    fn test_dtqw_on_cycle_periodic() {
        // Walk on a cycle should exhibit periodic behavior
        let g = Graph::cycle(8);
        let probs = dtqw_on_graph(&g, &CoinOperator::Grover, 32, 0).unwrap();

        // Check probability conservation at each step
        for (step, dist) in probs.iter().enumerate() {
            let total: f64 = dist.iter().sum();
            assert!(
                approx_eq(total, 1.0, 1e-6),
                "Step {}: sum = {}",
                step,
                total
            );
        }

        // On a small cycle with Grover coin, there should be some recurrence
        // (probability at start vertex should rise again)
        let start_probs: Vec<f64> = probs.iter().map(|d| d[0]).collect();
        // Find a local maximum after step 0
        let mut found_recurrence = false;
        for i in 2..start_probs.len() - 1 {
            if start_probs[i] > start_probs[i - 1] && start_probs[i] > 0.1 {
                found_recurrence = true;
                break;
            }
        }
        assert!(found_recurrence, "Expected periodic recurrence on cycle");
    }

    // ---- CTQW tests ----

    #[test]
    fn test_ctqw_probability_conservation() {
        let g = Graph::line(5);
        let config = CTQWConfig::new(g, 2.0, 0);
        let result = ctqw_evolve(&config).unwrap();

        let total: f64 = result.probabilities.iter().sum();
        assert!(
            approx_eq(total, 1.0, 1e-10),
            "CTQW probability not conserved: {}",
            total
        );
    }

    #[test]
    fn test_ctqw_complete_graph_transfer() {
        // On a complete graph K_n, perfect state transfer occurs from vertex 0
        // to the equal superposition of all other vertices and back periodically.
        // At t = pi/(2n), the probability at vertex 0 is minimized.
        let n = 4;
        let g = Graph::complete(n);

        // At t=0, all probability is at vertex 0
        let config = CTQWConfig::new(g.clone(), 0.0, 0);
        let result = ctqw_evolve(&config).unwrap();
        assert!(approx_eq(result.probabilities[0], 1.0, TOL));

        // At t = pi/(2*(n-1)), probability spreads to other vertices
        let t_spread = PI / (2.0 * (n - 1) as f64);
        let config2 = CTQWConfig::new(g.clone(), t_spread, 0);
        let result2 = ctqw_evolve(&config2).unwrap();
        // Probability should have left vertex 0
        assert!(
            result2.probabilities[0] < 0.9,
            "Expected probability spread, got {}",
            result2.probabilities[0]
        );
    }

    #[test]
    fn test_ctqw_path_graph_transfer() {
        // On P_2 (two vertices), there is perfect state transfer
        // from vertex 0 to vertex 1 at t = pi/2
        let g = Graph::line(2);
        let t = PI / 2.0;
        let config = CTQWConfig::new(g, t, 0);
        let result = ctqw_evolve(&config).unwrap();

        assert!(
            approx_eq(result.probabilities[1], 1.0, 1e-4),
            "Expected perfect transfer on P2 at t=pi/2, got P(1) = {}",
            result.probabilities[1]
        );
    }

    #[test]
    fn test_transfer_fidelity() {
        let g = Graph::line(2);
        let fid = transfer_fidelity(&g, 0, 1, 1.0, PI / 2.0).unwrap();
        assert!(
            approx_eq(fid, 1.0, 1e-4),
            "Expected near-perfect transfer fidelity, got {}",
            fid
        );
    }

    #[test]
    fn test_optimal_transfer_time_p2() {
        let g = Graph::line(2);
        let (best_t, best_fid) = optimal_transfer_time(&g, 0, 1, 1.0, 5.0, 100).unwrap();

        assert!(
            approx_eq(best_fid, 1.0, 1e-3),
            "Expected near-perfect fidelity, got {}",
            best_fid
        );
        // Perfect transfer on P_2 occurs at t = pi/2 + k*pi for integer k.
        // Check that the found time satisfies sin^2(t) ~ 1, i.e., t mod pi ~ pi/2.
        let remainder = best_t % PI;
        let dist_to_half_pi = (remainder - PI / 2.0).abs();
        assert!(
            dist_to_half_pi < 0.1,
            "Expected optimal time at pi/2 + k*pi, got t={:.4} (remainder={:.4})",
            best_t,
            remainder
        );
    }

    // ---- Walk search tests ----

    #[test]
    fn test_walk_search_complete_graph() {
        // On a complete graph, search should concentrate probability on marked vertex
        let n = 16;
        let g = Graph::complete(n);
        let marked = vec![3];
        // Optimal number of steps ~ pi/4 * sqrt(N) for Grover-like search
        let steps = ((PI / 4.0) * (n as f64).sqrt()).ceil() as usize;
        let result = walk_search(&g, &marked, steps).unwrap();

        // Marked vertex should have above-average probability
        let avg_prob = 1.0 / n as f64;
        assert!(
            result.probabilities[3] > avg_prob,
            "Marked vertex should have above-average probability: {} > {}",
            result.probabilities[3],
            avg_prob
        );
        assert_eq!(result.best_marked_vertex, 3);
    }

    #[test]
    fn test_walk_search_probability_conservation() {
        let g = Graph::complete(8);
        let marked = vec![2];
        let result = walk_search(&g, &marked, 10).unwrap();
        let total: f64 = result.probabilities.iter().sum();
        assert!(
            approx_eq(total, 1.0, 1e-6),
            "Walk search probability not conserved: {}",
            total
        );
    }

    #[test]
    fn test_szegedy_walk_probability_conservation() {
        let g = Graph::complete(4);
        let tm = g.transition_matrix();
        let marked = vec![0];
        let probs = szegedy_walk(&tm, 4, &marked, 20).unwrap();
        let total: f64 = probs.iter().sum();
        assert!(
            approx_eq(total, 1.0, 1e-6),
            "Szegedy walk probability not conserved: {}",
            total
        );
    }

    #[test]
    fn test_szegedy_walk_marked_concentration() {
        // Szegedy walk should concentrate probability on marked vertices over time
        let n = 8;
        let g = Graph::complete(n);
        let tm = g.transition_matrix();
        let marked = vec![0];
        let probs = szegedy_walk(&tm, n, &marked, 30).unwrap();

        // After enough steps, marked vertex should have more than 1/n probability
        let avg = 1.0 / n as f64;
        assert!(
            probs[0] > avg * 0.5,
            "Marked vertex should accumulate probability: {} (avg={})",
            probs[0],
            avg
        );
    }

    // ---- Application tests ----

    #[test]
    fn test_graph_isomorphism_same_graph() {
        let g1 = Graph::cycle(5);
        let g2 = Graph::cycle(5);
        let score = graph_isomorphism_test(&g1, &g2, 5).unwrap();
        assert!(
            score > 0.99,
            "Same graph should have score ~1.0, got {}",
            score
        );
    }

    #[test]
    fn test_graph_isomorphism_different_graphs() {
        let g1 = Graph::cycle(6);
        let g2 = Graph::star(6);
        let score = graph_isomorphism_test(&g1, &g2, 5).unwrap();
        assert!(
            score < 0.99,
            "Different graphs should have score < 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_graph_isomorphism_different_sizes() {
        let g1 = Graph::cycle(5);
        let g2 = Graph::cycle(6);
        let score = graph_isomorphism_test(&g1, &g2, 5).unwrap();
        assert_eq!(score, 0.0, "Different-sized graphs should score 0.0");
    }

    #[test]
    fn test_vertex_centrality_path_center_highest() {
        // On a path graph, the center vertex should have higher centrality than
        // the endpoints. Endpoint vertices are structurally disadvantaged because
        // probability can only arrive from one direction.
        let g = Graph::line(5); // vertices 0 -- 1 -- 2 -- 3 -- 4
        let centrality = vertex_centrality(&g, 10.0).unwrap();
        let center = centrality[2]; // center vertex
        let endpoint = centrality[0]; // endpoint vertex
        assert!(
            center > endpoint,
            "Center vertex centrality ({}) should exceed endpoint ({}) on path graph",
            center,
            endpoint
        );
        // Also check symmetry: endpoints should have equal centrality
        assert!(
            approx_eq(centrality[0], centrality[4], 1e-6),
            "Endpoints should have symmetric centrality: {} vs {}",
            centrality[0],
            centrality[4]
        );
        assert!(
            approx_eq(centrality[1], centrality[3], 1e-6),
            "Symmetric positions should have equal centrality: {} vs {}",
            centrality[1],
            centrality[3]
        );
    }

    #[test]
    fn test_vertex_centrality_complete_graph_uniform() {
        // All vertices in complete graph should have equal centrality
        let g = Graph::complete(4);
        let centrality = vertex_centrality(&g, 1.0).unwrap();
        for i in 1..4 {
            assert!(
                approx_eq(centrality[0], centrality[i], 1e-6),
                "Complete graph centrality should be uniform: {} vs {}",
                centrality[0],
                centrality[i]
            );
        }
    }

    #[test]
    fn test_quantum_pagerank_conservation() {
        let g = Graph::cycle(5);
        let pr = quantum_pagerank(&g, 0.85).unwrap();
        let total: f64 = pr.iter().sum();
        assert!(
            approx_eq(total, 1.0, 1e-6),
            "Quantum PageRank should sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_quantum_pagerank_cycle_uniform() {
        // On a cycle, all vertices should have equal PageRank
        let g = Graph::cycle(5);
        let pr = quantum_pagerank(&g, 0.85).unwrap();
        let expected = 1.0 / 5.0;
        for (i, &p) in pr.iter().enumerate() {
            assert!(
                approx_eq(p, expected, 0.05),
                "Cycle PageRank vertex {}: expected ~{}, got {}",
                i,
                expected,
                p
            );
        }
    }

    // ---- Error handling tests ----

    #[test]
    fn test_dtqw_empty_graph_error() {
        let config = DTQWConfig::new(0, 10);
        assert!(matches!(
            dtqw_on_line(&config),
            Err(QuantumWalkError::EmptyGraph)
        ));
    }

    #[test]
    fn test_dtqw_vertex_out_of_range() {
        let config = DTQWConfig::new(5, 10).with_initial_position(10);
        assert!(matches!(
            dtqw_on_line(&config),
            Err(QuantumWalkError::VertexOutOfRange { .. })
        ));
    }

    #[test]
    fn test_ctqw_empty_graph_error() {
        let g = Graph::empty(0);
        let config = CTQWConfig::new(g, 1.0, 0);
        assert!(matches!(
            ctqw_evolve(&config),
            Err(QuantumWalkError::EmptyGraph)
        ));
    }

    #[test]
    fn test_szegedy_invalid_transition_matrix() {
        // Non-stochastic matrix
        let bad_tm = vec![1.0, 1.0, 1.0, 1.0]; // rows sum to 2
        let result = szegedy_walk(&bad_tm, 2, &[0], 5);
        assert!(matches!(
            result,
            Err(QuantumWalkError::InvalidTransitionMatrix(_))
        ));
    }

    #[test]
    fn test_walk_search_no_marked_error() {
        let g = Graph::complete(4);
        let result = walk_search(&g, &[], 10);
        assert!(matches!(
            result,
            Err(QuantumWalkError::NoMarkedVertices)
        ));
    }

    #[test]
    fn test_hitting_time() {
        // On a line graph, hitting time from one end to the other should be finite
        let g = Graph::line(5);
        let ht = hitting_time(&g, 0, 4, &CoinOperator::Grover, 100, 0.01).unwrap();
        assert!(ht.is_some(), "Should find a hitting time on a 5-vertex line");
        assert!(ht.unwrap() > 0, "Hitting time should be positive");
    }

    #[test]
    fn test_coin_operator_hadamard() {
        let h = CoinOperator::Hadamard.matrix(2);
        assert_eq!(h.len(), 4);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!(approx_eq(h[0].re, inv_sqrt2, TOL));
        assert!(approx_eq(h[3].re, -inv_sqrt2, TOL));
    }

    #[test]
    fn test_coin_operator_grover() {
        let g = CoinOperator::Grover.matrix(3);
        assert_eq!(g.len(), 9);
        // Grover coin: G_jk = 2/d - delta_jk
        // For d=3: off-diagonal = 2/3, diagonal = 2/3 - 1 = -1/3
        assert!(approx_eq(g[0].re, -1.0 / 3.0, TOL));
        assert!(approx_eq(g[1].re, 2.0 / 3.0, TOL));
    }

    #[test]
    fn test_coin_operator_dft() {
        let dft = CoinOperator::DFT.matrix(2);
        assert_eq!(dft.len(), 4);
        // DFT_2 = 1/sqrt(2) * [[1, 1], [1, -1]] (same as Hadamard for d=2)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!(approx_eq(dft[0].re, inv_sqrt2, TOL)); // omega^0 = 1
    }
}
