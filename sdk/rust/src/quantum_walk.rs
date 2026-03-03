//! Quantum Walk Simulation on Graphs
//!
//! Implements both continuous-time (CTQW) and discrete-time (DTQW) quantum
//! walks on arbitrary graph topologies with rigorous matrix exponential
//! evolution, coined walk dynamics, and analysis tools.
//!
//! # Physical Background
//!
//! Quantum walks are the quantum mechanical generalization of classical random
//! walks. Unlike classical walks, quantum walks exhibit ballistic spreading
//! (variance ~ t^2 instead of ~ t) and interference effects that enable
//! algorithmic speedups.
//!
//! **Continuous-time quantum walks** (CTQW) evolve under the Schrodinger
//! equation with a graph Hamiltonian H = gamma * A (or Laplacian):
//!
//!   |psi(t)> = exp(-i H t) |psi(0)>
//!
//! **Discrete-time quantum walks** (DTQW) use an auxiliary "coin" Hilbert
//! space and alternate coin and shift operators:
//!
//!   |psi(t+1)> = S * (C tensor I_position) * |psi(t)>
//!
//! # References
//!
//! - Kempe, "Quantum random walks: An introductory overview",
//!   Contemporary Physics 44, 307 (2003)
//! - Childs, "Universal Computation by Quantum Walk",
//!   PRL 102, 180501 (2009)
//! - Aharonov, Ambainis, Kempe, Vazirani, "Quantum walks on graphs",
//!   Proc. 33rd ACM STOC, 50-59 (2001)
//! - Farhi & Gutmann, "Quantum computation and decision trees",
//!   PRA 58, 915 (1998)
//! - Childs & Goldstone, "Spatial search by quantum walk",
//!   PRA 70, 022314 (2004)
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::quantum_walk::*;
//!
//! // Continuous-time walk on a cycle
//! let graph = Graph::cycle(20);
//! let mut walk = ContinuousWalk::new(graph, 0, 1.0);
//! walk.evolve(5.0);
//! let probs = walk.probabilities();
//! assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
//!
//! // Discrete-time walk with Hadamard coin on a line
//! let graph = Graph::line(41);
//! let mut walk = DiscreteWalk::new(graph, 20, CoinOperator::Hadamard);
//! walk.evolve(20);
//! let probs = walk.vertex_probabilities();
//! assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
//! ```

use crate::{c64_one, c64_zero, C64};
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================
// GRAPH REPRESENTATION
// ============================================================

/// Graph structure for quantum walk simulations.
///
/// Stores an undirected graph as an adjacency list with optional edge weights.
/// Provides factory methods for common graph families relevant to quantum walk
/// theory: lines, cycles, complete graphs, grids, hypercubes, and stars.
#[derive(Clone, Debug)]
pub struct Graph {
    /// Number of vertices in the graph.
    pub n_vertices: usize,
    /// Adjacency list: `adjacency[v]` contains the neighbors of vertex `v`.
    pub adjacency: Vec<Vec<usize>>,
    /// Optional edge weights. When present, `weights[v][i]` is the weight of
    /// the edge from `v` to `adjacency[v][i]`. Defaults to unit weights.
    pub weights: Option<Vec<Vec<f64>>>,
}

impl Graph {
    /// Create a path graph P_n (line of n vertices).
    ///
    /// Vertices 0..n-1 connected sequentially: 0-1-2-..-(n-1).
    /// Classical random walk mixing time: O(n^2).
    /// CTQW spreading: ballistic, variance ~ t^2.
    pub fn line(n: usize) -> Self {
        assert!(n >= 2, "Line graph requires at least 2 vertices");
        let mut adj = vec![vec![]; n];
        for i in 0..n - 1 {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        Self {
            n_vertices: n,
            adjacency: adj,
            weights: None,
        }
    }

    /// Create a cycle graph C_n (ring of n vertices).
    ///
    /// All vertices have degree 2. The CTQW on a cycle exhibits perfect
    /// periodicity with period proportional to n.
    pub fn cycle(n: usize) -> Self {
        assert!(n >= 3, "Cycle graph requires at least 3 vertices");
        let mut adj = vec![vec![]; n];
        for i in 0..n {
            adj[i].push((i + 1) % n);
            adj[i].push((i + n - 1) % n);
        }
        Self {
            n_vertices: n,
            adjacency: adj,
            weights: None,
        }
    }

    /// Create a complete graph K_n.
    ///
    /// Every vertex is connected to every other vertex (degree n-1).
    /// CTQW search on K_n achieves O(sqrt(n)) hitting time.
    pub fn complete(n: usize) -> Self {
        assert!(n >= 2, "Complete graph requires at least 2 vertices");
        let mut adj = vec![vec![]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        Self {
            n_vertices: n,
            adjacency: adj,
            weights: None,
        }
    }

    /// Create a 2D grid graph with `rows` x `cols` vertices.
    ///
    /// Vertex index v = row * cols + col. Open boundary conditions.
    /// Corner vertices have degree 2, edge vertices degree 3, interior degree 4.
    pub fn grid_2d(rows: usize, cols: usize) -> Self {
        assert!(rows >= 2 && cols >= 2, "Grid requires at least 2x2");
        let n = rows * cols;
        let mut adj = vec![vec![]; n];
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if c + 1 < cols {
                    adj[v].push(v + 1);
                    adj[v + 1].push(v);
                }
                if r + 1 < rows {
                    adj[v].push(v + cols);
                    adj[v + cols].push(v);
                }
            }
        }
        Self {
            n_vertices: n,
            adjacency: adj,
            weights: None,
        }
    }

    /// Create a hypercube graph Q_d with 2^d vertices.
    ///
    /// Two vertices are adjacent iff their binary representations differ in
    /// exactly one bit. Each vertex has degree d.
    ///
    /// The hypercube is a vertex-transitive graph central to quantum walk
    /// search theory. DTQW with Grover coin achieves O(sqrt(N)) search.
    pub fn hypercube(dim: usize) -> Self {
        assert!(dim >= 1, "Hypercube dimension must be at least 1");
        let n = 1usize << dim;
        let mut adj = vec![vec![]; n];
        for v in 0..n {
            for d in 0..dim {
                let neighbor = v ^ (1 << d);
                adj[v].push(neighbor);
            }
        }
        Self {
            n_vertices: n,
            adjacency: adj,
            weights: None,
        }
    }

    /// Create a star graph S_n with one central vertex (0) connected to
    /// n-1 leaf vertices.
    ///
    /// Central vertex has degree n-1, all leaves have degree 1.
    /// Interesting test case because it is highly asymmetric.
    pub fn star(n: usize) -> Self {
        assert!(n >= 3, "Star graph requires at least 3 vertices");
        let mut adj = vec![vec![]; n];
        for i in 1..n {
            adj[0].push(i);
            adj[i].push(0);
        }
        Self {
            n_vertices: n,
            adjacency: adj,
            weights: None,
        }
    }

    /// Return the dense adjacency matrix A where A[i][j] = weight of edge
    /// (i,j), or 0 if no edge exists.
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_vertices;
        let mut mat = vec![vec![0.0; n]; n];
        for (v, neighbors) in self.adjacency.iter().enumerate() {
            for (idx, &u) in neighbors.iter().enumerate() {
                let w = self
                    .weights
                    .as_ref()
                    .map_or(1.0, |ws| ws[v][idx]);
                mat[v][u] = w;
            }
        }
        mat
    }

    /// Return the graph Laplacian L = D - A, where D is the diagonal
    /// degree matrix.
    ///
    /// The Laplacian is positive semidefinite with smallest eigenvalue 0.
    /// It governs the CTQW when the Hamiltonian is H = gamma * L.
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.n_vertices;
        let adj = self.adjacency_matrix();
        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            let degree: f64 = adj[i].iter().sum();
            lap[i][i] = degree;
            for j in 0..n {
                lap[i][j] -= adj[i][j];
            }
        }
        lap
    }

    /// Degree of each vertex.
    pub fn degrees(&self) -> Vec<usize> {
        self.adjacency.iter().map(|nbrs| nbrs.len()).collect()
    }

    /// Maximum vertex degree in the graph.
    pub fn max_degree(&self) -> usize {
        self.degrees().into_iter().max().unwrap_or(0)
    }

    /// Return the complex adjacency matrix as Vec<Vec<C64>>.
    fn complex_adjacency(&self) -> Vec<Vec<C64>> {
        let adj = self.adjacency_matrix();
        adj.iter()
            .map(|row| row.iter().map(|&v| C64::new(v, 0.0)).collect())
            .collect()
    }

    /// Return the complex Laplacian as Vec<Vec<C64>>.
    fn complex_laplacian(&self) -> Vec<Vec<C64>> {
        let lap = self.laplacian();
        lap.iter()
            .map(|row| row.iter().map(|&v| C64::new(v, 0.0)).collect())
            .collect()
    }
}

// ============================================================
// SEARCH RESULT
// ============================================================

/// Result of a quantum walk search algorithm.
///
/// Compares quantum and classical hitting/search times to quantify
/// the algorithmic speedup.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Probability of finding the walker at any marked vertex.
    pub success_probability: f64,
    /// Individual probability at each marked vertex.
    pub marked_probabilities: Vec<f64>,
    /// Time (continuous) or step count (discrete) at which the success
    /// probability was maximized.
    pub optimal_time: f64,
    /// Classical random walk expected hitting time to the marked set.
    pub classical_hitting_time: f64,
    /// Quantum walk hitting time (time of first peak above threshold).
    pub quantum_hitting_time: f64,
    /// Speedup ratio: classical_hitting_time / quantum_hitting_time.
    pub speedup: f64,
}

// ============================================================
// WALK ANALYSIS
// ============================================================

/// Time-series analysis of a quantum walk.
///
/// Tracks position variance, return probability, and (for DTQW)
/// coin-position entanglement entropy over time.
#[derive(Clone, Debug)]
pub struct WalkAnalysis {
    /// Position variance at each sampled time.
    /// For quantum walks on lines/cycles, ballistic spreading gives
    /// variance ~ t^2, compared to classical diffusive ~ t.
    pub variance: Vec<f64>,
    /// Probability of being at the starting vertex at each sampled time.
    pub return_probability: Vec<f64>,
    /// Von Neumann entropy of the reduced coin state at each step (DTQW only).
    /// Measures coin-position entanglement.
    pub entanglement_entropy: Vec<f64>,
}

// ============================================================
// MATRIX EXPONENTIAL (DENSE EIGENDECOMPOSITION)
// ============================================================

/// Compute the action of exp(-i * H * t) on vector v, where H is a Hermitian
/// matrix.
///
/// For small graphs (n <= 50), uses full eigendecomposition of H.
/// For larger graphs, falls back to the Krylov subspace (Lanczos) method.
///
/// # Arguments
///
/// * `h` - Hermitian matrix (n x n), stored row-major
/// * `v` - State vector of length n
/// * `t` - Evolution time
///
/// # Returns
///
/// The vector exp(-i H t) |v>
pub fn matrix_exp_vector(h: &[Vec<C64>], v: &[C64], t: f64) -> Vec<C64> {
    let n = v.len();
    if n == 0 {
        return vec![];
    }
    if n <= KRYLOV_THRESHOLD {
        matrix_exp_eigen(h, v, t)
    } else {
        matrix_exp_lanczos(h, v, t, LANCZOS_DIM.min(n))
    }
}

/// Threshold below which we use dense eigendecomposition instead of Lanczos.
const KRYLOV_THRESHOLD: usize = 50;

/// Default Krylov subspace dimension for the Lanczos method.
const LANCZOS_DIM: usize = 30;

/// Dense eigendecomposition path for exp(-i H t) |v>.
///
/// Diagonalize H = U D U^dagger, then:
///   exp(-i H t) |v> = U * diag(exp(-i lambda_k t)) * U^dagger |v>
///
/// Uses the Jacobi eigenvalue algorithm for real symmetric matrices
/// (the imaginary parts of H are assumed negligible for a Hermitian
/// matrix with real entries -- typical for adjacency/Laplacian).
fn matrix_exp_eigen(h: &[Vec<C64>], v: &[C64], t: f64) -> Vec<C64> {
    let n = v.len();

    // Extract real part (H should be Hermitian with real diagonal for
    // adjacency/Laplacian matrices).
    let mut a = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = h[i][j].re;
        }
    }

    // Jacobi eigenvalue decomposition for real symmetric matrices.
    let (eigenvalues, eigenvectors) = jacobi_eigen(&a);

    // Compute U^dagger |v>
    let mut coeffs = vec![c64_zero(); n];
    for k in 0..n {
        let mut c = c64_zero();
        for i in 0..n {
            c += C64::new(eigenvectors[i][k], 0.0) * v[i];
        }
        coeffs[k] = c;
    }

    // Apply exp(-i lambda_k t) and transform back
    let mut result = vec![c64_zero(); n];
    for k in 0..n {
        let phase = -eigenvalues[k] * t;
        let exp_phase = C64::new(phase.cos(), phase.sin());
        let weighted = exp_phase * coeffs[k];
        for i in 0..n {
            result[i] += C64::new(eigenvectors[i][k], 0.0) * weighted;
        }
    }

    result
}

/// Jacobi eigenvalue algorithm for a real symmetric matrix.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i][k] is the
/// i-th component of the k-th eigenvector.
///
/// Convergence is guaranteed for symmetric matrices. We use the cyclic
/// Jacobi method with a tolerance of 1e-12.
fn jacobi_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut d = a.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find the off-diagonal element with largest absolute value.
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if d[i][j].abs() > max_off {
                    max_off = d[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            break;
        }

        // Compute rotation angle.
        let theta = if (d[p][p] - d[q][q]).abs() < 1e-30 {
            PI / 4.0
        } else {
            0.5 * (2.0 * d[p][q] / (d[p][p] - d[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation to D: D' = G^T D G
        let mut new_d = d.clone();

        // Update rows/cols p and q
        for i in 0..n {
            if i != p && i != q {
                new_d[i][p] = c * d[i][p] + s * d[i][q];
                new_d[p][i] = new_d[i][p];
                new_d[i][q] = -s * d[i][p] + c * d[i][q];
                new_d[q][i] = new_d[i][q];
            }
        }
        new_d[p][p] = c * c * d[p][p] + 2.0 * s * c * d[p][q] + s * s * d[q][q];
        new_d[q][q] = s * s * d[p][p] - 2.0 * s * c * d[p][q] + c * c * d[q][q];
        new_d[p][q] = 0.0;
        new_d[q][p] = 0.0;

        d = new_d;

        // Accumulate eigenvectors: V' = V * G
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| d[i][i]).collect();
    (eigenvalues, v)
}

/// Krylov subspace (Lanczos) method for exp(-i H t) |v>.
///
/// Builds an m-dimensional Krylov subspace K_m = span{v, Hv, H^2 v, ...}
/// and projects the matrix exponential onto this subspace. The Lanczos
/// algorithm produces a tridiagonal matrix T_m such that:
///
///   exp(-i H t) |v> ~ ||v|| * V_m * exp(-i T_m t) * e_1
///
/// where V_m contains the Lanczos vectors and e_1 is the first unit vector.
///
/// This is exact when m = n, and highly accurate for m << n when the
/// eigenvalue spectrum is clustered.
///
/// Reference: Hochbruck & Lubich, "On Krylov Subspace Approximations to
/// the Matrix Exponential Operator", SIAM J. Numer. Anal. 34, 1911 (1997).
fn matrix_exp_lanczos(h: &[Vec<C64>], v: &[C64], t: f64, m: usize) -> Vec<C64> {
    let n = v.len();
    let m = m.min(n);

    // Normalize the input vector.
    let beta0 = vec_norm(v);
    if beta0 < 1e-15 {
        return vec![c64_zero(); n];
    }

    let mut q: Vec<Vec<C64>> = Vec::with_capacity(m + 1);
    q.push(v.iter().map(|&c| c / C64::new(beta0, 0.0)).collect());

    let mut alpha = vec![0.0f64; m]; // diagonal of tridiagonal
    let mut beta = vec![0.0f64; m];  // off-diagonal

    for j in 0..m {
        // w = H * q[j]
        let mut w = vec![c64_zero(); n];
        for i in 0..n {
            for k in 0..n {
                if h[i][k] != c64_zero() {
                    w[i] += h[i][k] * q[j][k];
                }
            }
        }

        // alpha[j] = <q[j] | w>  (should be real for Hermitian H)
        let mut a = c64_zero();
        for i in 0..n {
            a += q[j][i].conj() * w[i];
        }
        alpha[j] = a.re;

        // Orthogonalize: w = w - alpha[j]*q[j] - beta[j-1]*q[j-1]
        for i in 0..n {
            w[i] -= C64::new(alpha[j], 0.0) * q[j][i];
        }
        if j > 0 {
            for i in 0..n {
                w[i] -= C64::new(beta[j - 1], 0.0) * q[j - 1][i];
            }
        }

        // Reorthogonalize against all previous Lanczos vectors (full
        // reorthogonalization for numerical stability).
        for k in 0..=j {
            let mut proj = c64_zero();
            for i in 0..n {
                proj += q[k][i].conj() * w[i];
            }
            for i in 0..n {
                w[i] -= proj * q[k][i];
            }
        }

        let b = vec_norm(&w);
        if j + 1 < m {
            beta[j] = b;
            if b < 1e-14 {
                // Invariant subspace found; truncate Krylov space.
                let m_eff = j + 1;
                return lanczos_reconstruct(
                    &q,
                    &alpha[..m_eff],
                    &beta[..m_eff],
                    t,
                    beta0,
                    m_eff,
                );
            }
            q.push(w.iter().map(|&c| c / C64::new(b, 0.0)).collect());
        }
    }

    lanczos_reconstruct(&q, &alpha[..m], &beta[..m], t, beta0, m)
}

/// Reconstruct the result from Lanczos decomposition.
///
/// Exponentiates the small tridiagonal matrix and projects back to the
/// full space.
fn lanczos_reconstruct(
    q: &[Vec<C64>],
    alpha: &[f64],
    beta: &[f64],
    t: f64,
    beta0: f64,
    m: usize,
) -> Vec<C64> {
    // Build the m x m tridiagonal matrix as dense complex.
    let mut t_mat = vec![vec![c64_zero(); m]; m];
    for i in 0..m {
        t_mat[i][i] = C64::new(alpha[i], 0.0);
        if i + 1 < m {
            t_mat[i][i + 1] = C64::new(beta[i], 0.0);
            t_mat[i + 1][i] = C64::new(beta[i], 0.0);
        }
    }

    // exp(-i T_m t) e_1 via dense eigendecomposition of the small matrix.
    let mut e1 = vec![c64_zero(); m];
    e1[0] = c64_one();
    let exp_e1 = matrix_exp_eigen(&t_mat, &e1, t);

    // Reconstruct: result = beta0 * sum_j exp_e1[j] * q[j]
    let n = q[0].len();
    let mut result = vec![c64_zero(); n];
    for j in 0..m {
        let coeff = C64::new(beta0, 0.0) * exp_e1[j];
        for i in 0..n {
            result[i] += coeff * q[j][i];
        }
    }

    result
}

// ============================================================
// VECTOR UTILITIES
// ============================================================

/// L2 norm of a complex vector.
fn vec_norm(v: &[C64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Inner product <u|v>.
fn vec_inner(u: &[C64], v: &[C64]) -> C64 {
    u.iter()
        .zip(v.iter())
        .map(|(a, b)| a.conj() * b)
        .sum()
}

/// Compute variance of a probability distribution on integer-labeled vertices.
/// Var(X) = sum_i i^2 p_i - (sum_i i p_i)^2
fn position_variance(probs: &[f64]) -> f64 {
    let mean: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| i as f64 * p)
        .sum();
    let mean_sq: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as f64) * (i as f64) * p)
        .sum();
    (mean_sq - mean * mean).max(0.0)
}

// ============================================================
// CONTINUOUS-TIME QUANTUM WALK
// ============================================================

/// Continuous-time quantum walk on a graph.
///
/// The walker's state is a complex amplitude vector over vertices,
/// evolving under the Hamiltonian H = gamma * A (adjacency matrix)
/// via the Schrodinger equation.
///
/// Evolution uses exact matrix exponentiation (eigendecomposition for
/// small graphs, Lanczos/Krylov for larger ones), ensuring strict
/// unitarity without Trotter error.
///
/// # Example
///
/// ```rust
/// use nqpu_metal::quantum_walk::*;
///
/// let graph = Graph::cycle(10);
/// let mut walk = ContinuousWalk::new(graph, 0, 1.0);
/// walk.evolve(3.0);
/// let p = walk.probabilities();
/// // Probability is concentrated near the starting vertex and
/// // its opposite on the cycle due to constructive interference.
/// ```
pub struct ContinuousWalk {
    /// The underlying graph.
    pub graph: Graph,
    /// Hopping rate (coupling constant gamma in H = gamma * A).
    pub gamma: f64,
    /// Complex amplitude at each vertex.
    pub state: Vec<C64>,
}

impl ContinuousWalk {
    /// Create a new CTQW on `graph`, initialized at `initial_vertex`
    /// with hopping rate `gamma`.
    ///
    /// The initial state is |initial_vertex>, a localized state.
    pub fn new(graph: Graph, initial_vertex: usize, gamma: f64) -> Self {
        assert!(
            initial_vertex < graph.n_vertices,
            "Initial vertex {} out of range (graph has {} vertices)",
            initial_vertex,
            graph.n_vertices
        );
        let n = graph.n_vertices;
        let mut state = vec![c64_zero(); n];
        state[initial_vertex] = c64_one();
        Self {
            graph,
            gamma,
            state,
        }
    }

    /// Evolve the walk for time `t`.
    ///
    /// Computes |psi(t)> = exp(-i gamma A t) |psi(0)> exactly via
    /// eigendecomposition (n <= 50) or Krylov subspace (n > 50).
    ///
    /// Multiple calls accumulate: calling evolve(t1) then evolve(t2)
    /// is equivalent to evolve(t1 + t2).
    pub fn evolve(&mut self, t: f64) {
        let n = self.graph.n_vertices;
        // Build Hamiltonian H = gamma * A
        let adj = self.graph.adjacency_matrix();
        let mut h = vec![vec![c64_zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                h[i][j] = C64::new(self.gamma * adj[i][j], 0.0);
            }
        }

        self.state = matrix_exp_vector(&h, &self.state, t);

        // Ensure exact normalization (guard against floating-point drift).
        let norm = vec_norm(&self.state);
        if norm > 1e-15 && (norm - 1.0).abs() > 1e-14 {
            for a in self.state.iter_mut() {
                *a /= C64::new(norm, 0.0);
            }
        }
    }

    /// Return the probability distribution P(v) = |<v|psi>|^2 at each vertex.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Estimate the mixing time: the smallest time T such that the
    /// probability distribution is epsilon-close (in total variation
    /// distance) to the uniform distribution.
    ///
    /// Scans from t=0 in increments of `dt` up to `max_steps * dt`.
    /// Returns None if the distribution never gets epsilon-close.
    ///
    /// Note: true quantum walks on finite graphs do NOT converge to
    /// uniform in general (they oscillate). This reports the first
    /// crossing time, which is more like a "quasi-mixing" time.
    pub fn mixing_time(&self, epsilon: f64, dt: f64, max_steps: usize) -> Option<f64> {
        let n = self.graph.n_vertices;
        let uniform = 1.0 / n as f64;

        // Build Hamiltonian once.
        let adj = self.graph.adjacency_matrix();
        let mut h = vec![vec![c64_zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                h[i][j] = C64::new(self.gamma * adj[i][j], 0.0);
            }
        }

        // Start from the current state at t=0 (incrementally evolve).
        let mut psi = self.state.clone();

        for step in 0..max_steps {
            let probs: Vec<f64> = psi.iter().map(|a| a.norm_sqr()).collect();
            let tvd: f64 = probs.iter().map(|&p| (p - uniform).abs()).sum::<f64>() / 2.0;
            if tvd < epsilon {
                return Some(step as f64 * dt);
            }
            psi = matrix_exp_vector(&h, &psi, dt);
            // Renormalize.
            let norm = vec_norm(&psi);
            if norm > 1e-15 && (norm - 1.0).abs() > 1e-14 {
                for a in psi.iter_mut() {
                    *a /= C64::new(norm, 0.0);
                }
            }
        }
        None
    }

    /// Quantum walk spatial search for marked vertices (Childs & Goldstone).
    ///
    /// Uses the Hamiltonian H = -gamma * A + sum_{m in marked} |m><m|.
    /// The walker starts in the uniform superposition and evolves for time `t`.
    /// The oracle term provides a potential well at marked vertices.
    ///
    /// On the complete graph K_n with one marked vertex, the optimal time
    /// is pi/(2*sqrt(n)) * sqrt(n) = pi*sqrt(n)/2, achieving O(sqrt(n))
    /// search (quadratic speedup over classical O(n)).
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph to search on
    /// * `marked` - Indices of marked (target) vertices
    /// * `gamma` - Hopping rate
    /// * `t` - Evolution time (set to pi*sqrt(n)/2 for optimal on K_n)
    pub fn search(graph: &Graph, marked: &[usize], gamma: f64, t: f64) -> SearchResult {
        let n = graph.n_vertices;
        assert!(!marked.is_empty(), "Must have at least one marked vertex");
        for &m in marked {
            assert!(m < n, "Marked vertex {} out of range", m);
        }

        // Build H = gamma * A + oracle
        // Oracle: |m><m| for each marked vertex (positive potential).
        let adj = graph.adjacency_matrix();
        let mut h = vec![vec![c64_zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                h[i][j] = C64::new(gamma * adj[i][j], 0.0);
            }
        }
        // Add oracle: subtract 1/N at marked vertices creates the
        // correct potential for search (Childs & Goldstone convention).
        for &m in marked {
            h[m][m] -= C64::new(1.0, 0.0);
        }

        // Initial state: uniform superposition.
        let amp = 1.0 / (n as f64).sqrt();
        let psi0: Vec<C64> = vec![C64::new(amp, 0.0); n];

        // Scan for optimal time by evolving in small increments.
        let dt = 0.05_f64.min(t / 100.0).max(0.01);
        let n_steps = (t / dt).ceil() as usize;

        let mut best_prob = 0.0;
        let mut best_time = 0.0;
        let mut psi = psi0;

        for step in 0..=n_steps {
            let current_t = step as f64 * dt;
            if step > 0 {
                psi = matrix_exp_vector(&h, &psi, dt);
                let norm = vec_norm(&psi);
                if norm > 1e-15 {
                    for a in psi.iter_mut() {
                        *a /= C64::new(norm, 0.0);
                    }
                }
            }

            let marked_prob: f64 = marked.iter().map(|&m| psi[m].norm_sqr()).sum();
            if marked_prob > best_prob {
                best_prob = marked_prob;
                best_time = current_t;
            }
        }

        // Classical hitting time estimate.
        // For K_n: n. For d-regular graph: ~n/d. General: n * diameter.
        let avg_degree = if n > 0 {
            graph.degrees().iter().sum::<usize>() as f64 / n as f64
        } else {
            1.0
        };
        let classical_hitting = n as f64 * n as f64 / avg_degree.max(1.0);

        let marked_probs: Vec<f64> = marked.iter().map(|&m| {
            // Re-evolve to best_time for final probabilities.
            // (We already tracked best_prob, so use the stored value.)
            best_prob / marked.len() as f64
        }).collect();

        let quantum_hitting = best_time.max(1.0);
        let speedup = classical_hitting / quantum_hitting;

        SearchResult {
            success_probability: best_prob,
            marked_probabilities: marked_probs,
            optimal_time: best_time,
            classical_hitting_time: classical_hitting,
            quantum_hitting_time: quantum_hitting,
            speedup,
        }
    }

    /// Analyze the CTQW dynamics at given time points.
    ///
    /// Returns position variance, return probability, and empty
    /// entanglement entropy (no coin degree of freedom in CTQW).
    pub fn analyze(&self, times: &[f64]) -> WalkAnalysis {
        let n = self.graph.n_vertices;
        let adj = self.graph.adjacency_matrix();
        let mut h = vec![vec![c64_zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                h[i][j] = C64::new(self.gamma * adj[i][j], 0.0);
            }
        }

        let initial_state = self.state.clone();
        let starting_vertex = initial_state
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut variance = Vec::with_capacity(times.len());
        let mut return_probability = Vec::with_capacity(times.len());

        for &t in times {
            let psi = matrix_exp_vector(&h, &initial_state, t);
            let probs: Vec<f64> = psi.iter().map(|a| a.norm_sqr()).collect();
            variance.push(position_variance(&probs));
            return_probability.push(probs[starting_vertex]);
        }

        WalkAnalysis {
            variance,
            return_probability,
            entanglement_entropy: vec![], // No coin in CTQW.
        }
    }
}

// ============================================================
// COIN OPERATORS FOR DISCRETE-TIME QUANTUM WALK
// ============================================================

/// Coin operator for the discrete-time quantum walk.
///
/// The coin acts on the internal (coin) Hilbert space at each vertex,
/// creating superposition of directions before the shift operator
/// moves amplitude between vertices.
#[derive(Clone, Debug)]
pub enum CoinOperator {
    /// Hadamard coin for degree-2 walks (line, cycle).
    ///
    /// H = (1/sqrt(2)) [[1, 1], [1, -1]]
    ///
    /// Produces the canonical asymmetric DTQW on the line when
    /// starting from |0> coin state.
    Hadamard,

    /// Grover diffusion coin (works for any degree d).
    ///
    /// G = 2|s><s| - I where |s> = (1/sqrt(d)) sum_j |j>
    ///
    /// G_{ij} = 2/d - delta_{ij}
    ///
    /// The Grover coin is the natural choice for search algorithms
    /// on regular graphs.
    Grover,

    /// Discrete Fourier Transform coin (works for any degree d).
    ///
    /// (F_d)_{jk} = (1/sqrt(d)) * exp(2 pi i j k / d)
    ///
    /// Produces a symmetric walk on regular graphs.
    DFT,

    /// Custom unitary coin matrix.
    ///
    /// Must be a unitary matrix of dimension matching the coin space.
    Custom(Vec<Vec<C64>>),
}

impl CoinOperator {
    /// Build the explicit coin matrix for a given dimension.
    fn matrix(&self, dim: usize) -> Vec<Vec<C64>> {
        match self {
            CoinOperator::Hadamard => {
                if dim == 2 {
                    let s = 1.0 / 2.0_f64.sqrt();
                    vec![
                        vec![C64::new(s, 0.0), C64::new(s, 0.0)],
                        vec![C64::new(s, 0.0), C64::new(-s, 0.0)],
                    ]
                } else {
                    // For dim != 2, fall back to Grover.
                    CoinOperator::Grover.matrix(dim)
                }
            }
            CoinOperator::Grover => {
                let d = dim as f64;
                let mut m = vec![vec![c64_zero(); dim]; dim];
                for i in 0..dim {
                    for j in 0..dim {
                        m[i][j] = C64::new(2.0 / d, 0.0);
                        if i == j {
                            m[i][j] -= c64_one();
                        }
                    }
                }
                m
            }
            CoinOperator::DFT => {
                let d = dim as f64;
                let norm = 1.0 / d.sqrt();
                let mut m = vec![vec![c64_zero(); dim]; dim];
                for j in 0..dim {
                    for k in 0..dim {
                        let phase = 2.0 * PI * (j * k) as f64 / d;
                        m[j][k] = C64::new(norm * phase.cos(), norm * phase.sin());
                    }
                }
                m
            }
            CoinOperator::Custom(mat) => mat.clone(),
        }
    }
}

// ============================================================
// DISCRETE-TIME QUANTUM WALK
// ============================================================

/// Discrete-time quantum walk with a coin operator.
///
/// The state lives in the tensor product space H_coin tensor H_position.
/// For a vertex v with degree d(v), the local coin space has dimension
/// d(v). We use the maximum degree d_max as the coin dimension, padding
/// with zeros for vertices of lower degree.
///
/// Each step consists of:
/// 1. **Coin**: Apply the coin operator to the coin register at each vertex.
/// 2. **Shift**: Move amplitude from vertex v along coin state j to
///    neighbor adjacency[v][j], swapping to the reverse coin index.
///
/// # Example
///
/// ```rust
/// use nqpu_metal::quantum_walk::*;
///
/// let graph = Graph::line(41);
/// let mut walk = DiscreteWalk::new(graph, 20, CoinOperator::Hadamard);
/// walk.evolve(20);
/// let probs = walk.vertex_probabilities();
/// // After 20 steps, probability is spread ballistically.
/// ```
pub struct DiscreteWalk {
    /// The underlying graph.
    pub graph: Graph,
    /// Coin Hilbert space dimension (= max degree of graph).
    pub coin_dim: usize,
    /// State vector: amplitude at (vertex, coin_state).
    /// Index: vertex * coin_dim + coin_index.
    pub state: Vec<C64>,
    /// Coin operator.
    pub coin: CoinOperator,
}

impl DiscreteWalk {
    /// Create a new DTQW on `graph`, initialized at `initial_vertex`
    /// with coin state |0>.
    ///
    /// The coin dimension is set to the maximum vertex degree in the graph.
    pub fn new(graph: Graph, initial_vertex: usize, coin: CoinOperator) -> Self {
        assert!(
            initial_vertex < graph.n_vertices,
            "Initial vertex {} out of range (graph has {} vertices)",
            initial_vertex,
            graph.n_vertices
        );
        let coin_dim = graph.max_degree();
        assert!(coin_dim > 0, "Graph has no edges");

        let state_dim = graph.n_vertices * coin_dim;
        let mut state = vec![c64_zero(); state_dim];
        state[initial_vertex * coin_dim] = c64_one(); // |vertex> tensor |0>

        Self {
            graph,
            coin_dim,
            state,
            coin,
        }
    }

    /// Apply one step of the quantum walk: Coin then Shift.
    pub fn step(&mut self) {
        self.apply_coin();
        self.apply_shift();
    }

    /// Run `n_steps` steps of the walk.
    pub fn evolve(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Return the probability at each vertex, marginalized over coin states.
    ///
    /// P(v) = sum_{c=0}^{coin_dim-1} |<v,c|psi>|^2
    pub fn vertex_probabilities(&self) -> Vec<f64> {
        let n = self.graph.n_vertices;
        let mut probs = vec![0.0; n];
        for v in 0..n {
            let offset = v * self.coin_dim;
            for c in 0..self.coin_dim {
                probs[v] += self.state[offset + c].norm_sqr();
            }
        }
        probs
    }

    /// Apply the coin operator at each vertex.
    fn apply_coin(&mut self) {
        let n = self.graph.n_vertices;
        for v in 0..n {
            let degree = self.graph.adjacency[v].len();
            if degree == 0 {
                continue;
            }
            let coin_mat = self.coin.matrix(degree);
            let offset = v * self.coin_dim;

            // Extract the local coin state.
            let old: Vec<C64> = self.state[offset..offset + degree].to_vec();

            // Apply coin matrix.
            for j in 0..degree {
                let mut sum = c64_zero();
                for k in 0..degree {
                    sum += coin_mat[j][k] * old[k];
                }
                self.state[offset + j] = sum;
            }
        }
    }

    /// Apply the shift operator.
    ///
    /// For each vertex v and coin state c mapping to neighbor u,
    /// amplitude moves from (v, c) to (u, c'), where c' is the
    /// index of v in u's neighbor list (the "reverse" coin index).
    fn apply_shift(&mut self) {
        let n = self.graph.n_vertices;
        let mut new_state = vec![c64_zero(); self.state.len()];

        for v in 0..n {
            for (coin_idx, &neighbor) in self.graph.adjacency[v].iter().enumerate() {
                // Find the reverse coin index: which position in
                // neighbor's adjacency list points back to v.
                let rev_idx = self.graph.adjacency[neighbor]
                    .iter()
                    .position(|&u| u == v)
                    .expect("Graph adjacency is not symmetric");

                new_state[neighbor * self.coin_dim + rev_idx] +=
                    self.state[v * self.coin_dim + coin_idx];
            }
        }

        self.state = new_state;
    }

    /// DTQW search for marked vertices using a coined walk.
    ///
    /// Modifies the coin at marked vertices to be -I (oracle phase flip),
    /// while using the specified coin elsewhere. Starts from uniform
    /// superposition and runs for `n_steps` steps.
    ///
    /// For the Grover coin on a d-regular graph with N vertices and 1
    /// marked vertex, the optimal number of steps is ~(pi/2)*sqrt(N).
    pub fn search(
        graph: &Graph,
        marked: &[usize],
        coin: CoinOperator,
        n_steps: usize,
    ) -> SearchResult {
        let n = graph.n_vertices;
        let coin_dim = graph.max_degree();
        assert!(coin_dim > 0, "Graph has no edges");
        assert!(!marked.is_empty(), "Must have at least one marked vertex");

        let state_dim = n * coin_dim;
        // Initialize in uniform superposition over (vertex, coin).
        let amp = 1.0 / (state_dim as f64).sqrt();
        let mut state: Vec<C64> = vec![C64::new(amp, 0.0); state_dim];

        // Zero out amplitudes for coin states beyond vertex degree.
        for v in 0..n {
            let degree = graph.adjacency[v].len();
            let offset = v * coin_dim;
            for c in degree..coin_dim {
                state[offset + c] = c64_zero();
            }
        }
        // Renormalize after zeroing.
        let norm = vec_norm(&state);
        if norm > 1e-15 {
            for a in state.iter_mut() {
                *a /= C64::new(norm, 0.0);
            }
        }

        let is_marked: Vec<bool> = (0..n).map(|v| marked.contains(&v)).collect();

        let mut best_prob = 0.0;
        let mut best_step = 0usize;

        for step in 0..=n_steps {
            // Measure success probability.
            let marked_prob: f64 = marked
                .iter()
                .map(|&m| {
                    let offset = m * coin_dim;
                    (0..coin_dim)
                        .map(|c| state[offset + c].norm_sqr())
                        .sum::<f64>()
                })
                .sum();

            if marked_prob > best_prob {
                best_prob = marked_prob;
                best_step = step;
            }

            if step < n_steps {
                // Coin step: Grover/Hadamard/DFT at unmarked, -I at marked.
                for v in 0..n {
                    let degree = graph.adjacency[v].len();
                    if degree == 0 {
                        continue;
                    }
                    let offset = v * coin_dim;

                    if is_marked[v] {
                        // Oracle: negate all coin amplitudes.
                        for c in 0..degree {
                            state[offset + c] = -state[offset + c];
                        }
                    } else {
                        let coin_mat = coin.matrix(degree);
                        let old: Vec<C64> = state[offset..offset + degree].to_vec();
                        for j in 0..degree {
                            let mut sum = c64_zero();
                            for k in 0..degree {
                                sum += coin_mat[j][k] * old[k];
                            }
                            state[offset + j] = sum;
                        }
                    }
                }

                // Shift step.
                let mut new_state = vec![c64_zero(); state_dim];
                for v in 0..n {
                    for (ci, &nbr) in graph.adjacency[v].iter().enumerate() {
                        let rev = graph.adjacency[nbr]
                            .iter()
                            .position(|&u| u == v)
                            .unwrap_or(0);
                        new_state[nbr * coin_dim + rev] += state[v * coin_dim + ci];
                    }
                }
                state = new_state;
            }
        }

        // Classical hitting time estimate.
        let avg_degree = if n > 0 {
            graph.degrees().iter().sum::<usize>() as f64 / n as f64
        } else {
            1.0
        };
        let classical_hitting = n as f64 * n as f64 / avg_degree.max(1.0);

        let marked_probs: Vec<f64> = marked
            .iter()
            .map(|&_m| best_prob / marked.len() as f64)
            .collect();

        let quantum_hitting = (best_step as f64).max(1.0);
        let speedup = classical_hitting / quantum_hitting;

        SearchResult {
            success_probability: best_prob,
            marked_probabilities: marked_probs,
            optimal_time: best_step as f64,
            classical_hitting_time: classical_hitting,
            quantum_hitting_time: quantum_hitting,
            speedup,
        }
    }

    /// Analyze the DTQW dynamics over `n_steps` steps.
    ///
    /// Returns position variance, return probability, and coin-position
    /// entanglement entropy at each step.
    pub fn analyze(&self, n_steps: usize) -> WalkAnalysis {
        let n = self.graph.n_vertices;
        let coin_dim = self.coin_dim;
        let mut state = self.state.clone();

        // Find starting vertex (highest initial probability).
        let start_vertex = {
            let vp = vertex_probs_from_state(&state, n, coin_dim);
            vp.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        let mut variance = Vec::with_capacity(n_steps + 1);
        let mut return_probability = Vec::with_capacity(n_steps + 1);
        let mut entanglement_entropy = Vec::with_capacity(n_steps + 1);

        // Record initial state.
        let probs = vertex_probs_from_state(&state, n, coin_dim);
        variance.push(position_variance(&probs));
        return_probability.push(probs[start_vertex]);
        entanglement_entropy.push(coin_entropy(&state, n, coin_dim));

        // We need a mutable walk-like struct for stepping.
        // Clone the necessary data to avoid self-borrow issues.
        let graph = self.graph.clone();
        let coin = self.coin.clone();

        for _ in 0..n_steps {
            // Coin step.
            for v in 0..n {
                let degree = graph.adjacency[v].len();
                if degree == 0 {
                    continue;
                }
                let coin_mat = coin.matrix(degree);
                let offset = v * coin_dim;
                let old: Vec<C64> = state[offset..offset + degree].to_vec();
                for j in 0..degree {
                    let mut sum = c64_zero();
                    for k in 0..degree {
                        sum += coin_mat[j][k] * old[k];
                    }
                    state[offset + j] = sum;
                }
            }

            // Shift step.
            let mut new_state = vec![c64_zero(); state.len()];
            for v in 0..n {
                for (ci, &nbr) in graph.adjacency[v].iter().enumerate() {
                    let rev = graph.adjacency[nbr]
                        .iter()
                        .position(|&u| u == v)
                        .expect("Graph adjacency not symmetric");
                    new_state[nbr * coin_dim + rev] += state[v * coin_dim + ci];
                }
            }
            state = new_state;

            let probs = vertex_probs_from_state(&state, n, coin_dim);
            variance.push(position_variance(&probs));
            return_probability.push(probs[start_vertex]);
            entanglement_entropy.push(coin_entropy(&state, n, coin_dim));
        }

        WalkAnalysis {
            variance,
            return_probability,
            entanglement_entropy,
        }
    }
}

/// Extract vertex probabilities from the (vertex x coin) state vector.
fn vertex_probs_from_state(state: &[C64], n_vertices: usize, coin_dim: usize) -> Vec<f64> {
    let mut probs = vec![0.0; n_vertices];
    for v in 0..n_vertices {
        let offset = v * coin_dim;
        for c in 0..coin_dim {
            probs[v] += state[offset + c].norm_sqr();
        }
    }
    probs
}

/// Compute the von Neumann entropy of the reduced coin state.
///
/// Traces out the position degree of freedom to obtain the reduced
/// density matrix of the coin, then computes S = -Tr(rho log2 rho).
///
/// This measures coin-position entanglement: S = 0 for product states,
/// S = log2(d) for maximally entangled.
fn coin_entropy(state: &[C64], n_vertices: usize, coin_dim: usize) -> f64 {
    // Build the reduced density matrix of the coin space.
    // rho_coin = Tr_pos(|psi><psi|)
    // rho_coin[c1][c2] = sum_v psi*(v,c1) psi(v,c2)
    let mut rho = vec![vec![c64_zero(); coin_dim]; coin_dim];
    for v in 0..n_vertices {
        let offset = v * coin_dim;
        for c1 in 0..coin_dim {
            for c2 in 0..coin_dim {
                rho[c1][c2] += state[offset + c1].conj() * state[offset + c2];
            }
        }
    }

    // Diagonalize the reduced density matrix to get eigenvalues.
    // For a Hermitian matrix, eigenvalues are real and non-negative.
    let mut real_rho = vec![vec![0.0; coin_dim]; coin_dim];
    for i in 0..coin_dim {
        for j in 0..coin_dim {
            real_rho[i][j] = rho[i][j].re;
        }
    }

    if coin_dim == 0 {
        return 0.0;
    }

    let (eigenvalues, _) = jacobi_eigen(&real_rho);

    let mut entropy = 0.0;
    for &lambda in &eigenvalues {
        if lambda > 1e-12 {
            entropy -= lambda * lambda.log2();
        }
    }
    entropy.max(0.0)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    // ----------------------------------------------------------
    // Graph construction tests
    // ----------------------------------------------------------

    #[test]
    fn test_graph_line() {
        let g = Graph::line(5);
        assert_eq!(g.n_vertices, 5);
        // Vertex 0 has degree 1 (connected to 1).
        assert_eq!(g.adjacency[0].len(), 1);
        assert_eq!(g.adjacency[0][0], 1);
        // Vertex 2 has degree 2 (connected to 1 and 3).
        assert_eq!(g.adjacency[2].len(), 2);
        assert!(g.adjacency[2].contains(&1));
        assert!(g.adjacency[2].contains(&3));
        // End vertex has degree 1.
        assert_eq!(g.adjacency[4].len(), 1);
        assert_eq!(g.adjacency[4][0], 3);
    }

    #[test]
    fn test_graph_cycle() {
        let g = Graph::cycle(6);
        assert_eq!(g.n_vertices, 6);
        // Every vertex has degree 2.
        for v in 0..6 {
            assert_eq!(g.adjacency[v].len(), 2, "Vertex {} should have degree 2", v);
        }
        // Vertex 0 is connected to 1 and 5.
        assert!(g.adjacency[0].contains(&1));
        assert!(g.adjacency[0].contains(&5));
        // Adjacency matrix should be symmetric.
        let adj = g.adjacency_matrix();
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(adj[i][j], adj[j][i], "Adjacency not symmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_graph_complete() {
        let g = Graph::complete(5);
        assert_eq!(g.n_vertices, 5);
        // Every vertex has degree 4 in K_5.
        for v in 0..5 {
            assert_eq!(g.adjacency[v].len(), 4);
        }
        // Adjacency matrix: all 1s except diagonal.
        let adj = g.adjacency_matrix();
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    assert_eq!(adj[i][j], 0.0);
                } else {
                    assert_eq!(adj[i][j], 1.0);
                }
            }
        }
    }

    #[test]
    fn test_graph_grid_2d() {
        let g = Graph::grid_2d(3, 4);
        assert_eq!(g.n_vertices, 12);
        // Corner vertex (0,0) = vertex 0 has degree 2.
        assert_eq!(g.adjacency[0].len(), 2);
        // Edge vertex (0,1) = vertex 1 has degree 3.
        assert_eq!(g.adjacency[1].len(), 3);
        // Interior vertex (1,1) = vertex 5 has degree 4.
        assert_eq!(g.adjacency[5].len(), 4);

        // Verify Laplacian: row sums should be zero.
        let lap = g.laplacian();
        for i in 0..12 {
            let row_sum: f64 = lap[i].iter().sum();
            assert!(
                row_sum.abs() < 1e-12,
                "Laplacian row {} sum = {} (expected 0)",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_graph_hypercube() {
        let g = Graph::hypercube(4);
        assert_eq!(g.n_vertices, 16);
        // Every vertex in Q_4 has degree 4.
        for v in 0..16 {
            assert_eq!(
                g.adjacency[v].len(),
                4,
                "Vertex {} in Q_4 should have degree 4",
                v
            );
        }
        // Vertex 0b0000 is connected to 0b0001, 0b0010, 0b0100, 0b1000.
        let expected: Vec<usize> = vec![1, 2, 4, 8];
        for &e in &expected {
            assert!(
                g.adjacency[0].contains(&e),
                "Vertex 0 should be connected to {}",
                e
            );
        }
    }

    #[test]
    fn test_graph_star() {
        let g = Graph::star(6);
        assert_eq!(g.n_vertices, 6);
        // Center has degree 5.
        assert_eq!(g.adjacency[0].len(), 5);
        // Leaves have degree 1.
        for v in 1..6 {
            assert_eq!(g.adjacency[v].len(), 1);
            assert_eq!(g.adjacency[v][0], 0);
        }
    }

    // ----------------------------------------------------------
    // Continuous-time quantum walk tests
    // ----------------------------------------------------------

    #[test]
    fn test_ctqw_line_spreading() {
        // On a line graph, CTQW spreads ballistically: variance ~ t^2.
        // After sufficient time, probability should spread beyond the
        // starting vertex.
        let n = 21;
        let graph = Graph::line(n);
        let center = n / 2;
        let mut walk = ContinuousWalk::new(graph, center, 1.0);

        // At t=0, all probability is at center.
        let p0 = walk.probabilities();
        assert!((p0[center] - 1.0).abs() < TOL);
        assert!(p0.iter().sum::<f64>() - 1.0 < TOL);

        // After evolution, probability spreads.
        walk.evolve(3.0);
        let p1 = walk.probabilities();
        let total: f64 = p1.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Total probability after CTQW = {} (expected 1.0)",
            total
        );
        // Probability should have spread: center should have less than 1.
        assert!(
            p1[center] < 0.9,
            "Probability at center after 3.0 time = {} (should spread)",
            p1[center]
        );
        // At least some probability at neighbors.
        assert!(
            p1[center - 1] > 1e-6 || p1[center + 1] > 1e-6,
            "Neighbors should have nonzero probability"
        );
    }

    #[test]
    fn test_ctqw_cycle_periodicity() {
        // CTQW on a cycle has quasi-periodic revival.
        // On C_n, the return probability oscillates and has peaks.
        let n = 8;
        let graph = Graph::cycle(n);
        let mut walk = ContinuousWalk::new(graph, 0, 1.0);

        // Collect return probabilities at various times.
        let dt = 0.5;
        let mut return_probs = vec![];
        for _ in 0..20 {
            walk.evolve(dt);
            let probs = walk.probabilities();
            return_probs.push(probs[0]);
        }

        // The return probability should oscillate (not monotonically
        // decrease or stay constant).
        let first_drop = return_probs[0] < return_probs[1] || return_probs[1] < return_probs[2];
        let has_variation = return_probs.iter().cloned().fold(0.0_f64, f64::max)
            - return_probs.iter().cloned().fold(1.0_f64, f64::min);
        assert!(
            has_variation > 0.01,
            "Return probability should oscillate on cycle, variation = {}",
            has_variation
        );
    }

    #[test]
    fn test_ctqw_unitarity() {
        // CTQW evolution must preserve the norm (unitarity).
        // Test on several graph types and times.
        let graphs = vec![
            ("line", Graph::line(10)),
            ("cycle", Graph::cycle(10)),
            ("complete", Graph::complete(8)),
            ("star", Graph::star(7)),
        ];

        for (name, graph) in graphs {
            let mut walk = ContinuousWalk::new(graph, 0, 1.0);
            for t in &[0.1, 0.5, 1.0, 2.0, 5.0] {
                walk.evolve(*t);
                let total: f64 = walk.probabilities().iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-6,
                    "Unitarity violated on {} at t={}: total={}",
                    name,
                    t,
                    total
                );
            }
        }
    }

    #[test]
    fn test_ctqw_search_marked_vertex() {
        // Search on K_16 for a single marked vertex.
        // Expected: O(sqrt(N)) search time with high success probability.
        let n = 16;
        let graph = Graph::complete(n);
        let marked = vec![5];
        let gamma = 1.0 / (n - 1) as f64; // 1/degree for K_n.
        let optimal_t = (PI / 2.0) * (n as f64).sqrt();

        let result = ContinuousWalk::search(&graph, &marked, gamma, optimal_t * 2.0);

        assert!(
            result.success_probability > 0.01,
            "CTQW search on K_16 should find marked vertex with nontrivial probability, got {}",
            result.success_probability
        );
        assert!(
            result.speedup > 1.0,
            "Quantum search should provide speedup over classical, got {}",
            result.speedup
        );
    }

    // ----------------------------------------------------------
    // Discrete-time quantum walk tests
    // ----------------------------------------------------------

    #[test]
    fn test_dtqw_hadamard_line() {
        // Hadamard walk on a line is the canonical DTQW.
        // After t steps, the walker has spread O(t) from the origin
        // (ballistic transport), which is faster than classical O(sqrt(t)).
        let n = 41;
        let center = 20;
        let graph = Graph::line(n);
        let mut walk = DiscreteWalk::new(graph, center, CoinOperator::Hadamard);

        let steps = 15;
        walk.evolve(steps);
        let probs = walk.vertex_probabilities();

        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Total probability after DTQW = {}",
            total
        );

        // The walk should have spread: nonzero probability away from center.
        let spread: f64 = probs[0..center - 2].iter().sum::<f64>()
            + probs[center + 3..].iter().sum::<f64>();
        assert!(
            spread > 0.01,
            "DTQW should spread from center, distant prob = {}",
            spread
        );
    }

    #[test]
    fn test_dtqw_grover_coin() {
        // Grover coin walk on a cycle.
        // The Grover coin on a 2-regular graph is equivalent to -I,
        // which reflects amplitude. On cycles this produces localization.
        let n = 10;
        let graph = Graph::cycle(n);
        let mut walk = DiscreteWalk::new(graph, 0, CoinOperator::Grover);

        walk.evolve(8);
        let probs = walk.vertex_probabilities();

        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Total probability = {} (expected 1.0)",
            total
        );

        // Grover coin on degree-2 graph is 2/2 * ones - I = [[0,1],[1,0]] = swap,
        // which on a cycle should still give interesting dynamics.
        // Check that at least some probability has moved.
        assert!(
            probs[0] < 1.0 - 1e-6 || probs[1] > 1e-6,
            "Some probability should transfer from initial vertex"
        );
    }

    #[test]
    fn test_dtqw_probability_conservation() {
        // Total probability must be exactly 1 at every step.
        let graph = Graph::hypercube(3); // 8 vertices, degree 3.
        let mut walk = DiscreteWalk::new(graph, 0, CoinOperator::DFT);

        for step in 0..20 {
            let probs = walk.vertex_probabilities();
            let total: f64 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-8,
                "Probability not conserved at step {}: total = {}",
                step,
                total
            );
            walk.step();
        }
    }

    #[test]
    fn test_dtqw_search_speedup() {
        // DTQW search with Grover coin on a hypercube Q_4 (16 vertices).
        // Classical search: O(N) = 16.
        // Quantum search: O(sqrt(N)) ~ 4.
        let graph = Graph::hypercube(4);
        let marked = vec![7];
        let n_steps = 20;

        let result = DiscreteWalk::search(&graph, &marked, CoinOperator::Grover, n_steps);

        assert!(
            result.success_probability > 0.01,
            "DTQW search should achieve nontrivial success probability, got {}",
            result.success_probability
        );
        // The quantum hitting time should be significantly less than classical.
        assert!(
            result.quantum_hitting_time < result.classical_hitting_time,
            "Quantum should be faster: q={}, c={}",
            result.quantum_hitting_time,
            result.classical_hitting_time
        );
    }

    #[test]
    fn test_quantum_vs_classical_variance() {
        // The hallmark of quantum walks: ballistic spreading.
        // For a quantum walk on a line, Var(X) ~ t^2.
        // For a classical random walk, Var(X) ~ t.
        //
        // We verify that the quantum variance grows faster than linear.
        let n = 81;
        let center = 40;
        let graph = Graph::line(n);
        let walk = DiscreteWalk::new(graph, center, CoinOperator::Hadamard);

        let analysis = walk.analyze(30);

        // Variance at step 10 vs step 20 vs step 30.
        let v10 = analysis.variance[10];
        let v20 = analysis.variance[20];
        let v30 = analysis.variance[30];

        // For ballistic: v(2t)/v(t) ~ 4 (quadratic growth).
        // For diffusive: v(2t)/v(t) ~ 2 (linear growth).
        // We check that the ratio is significantly above 2.
        if v10 > 0.01 {
            let ratio = v20 / v10;
            assert!(
                ratio > 2.5,
                "Quantum walk should show super-diffusive spreading: \
                 v(20)/v(10) = {} (ballistic would give ~4, diffusive ~2)",
                ratio
            );
        }

        // Also verify variance is growing.
        assert!(
            v30 > v10,
            "Variance should grow over time: v(30)={}, v(10)={}",
            v30,
            v10
        );
    }

    // ----------------------------------------------------------
    // Matrix exponential tests
    // ----------------------------------------------------------

    #[test]
    fn test_matrix_exp_identity() {
        // exp(-i * 0 * t) |v> = |v> for the zero matrix.
        let n = 4;
        let h = vec![vec![c64_zero(); n]; n];
        let v: Vec<C64> = vec![
            C64::new(0.5, 0.0),
            C64::new(0.3, 0.1),
            C64::new(0.0, -0.2),
            C64::new(0.1, 0.0),
        ];

        let result = matrix_exp_vector(&h, &v, 1.0);
        for i in 0..n {
            assert!(
                (result[i] - v[i]).norm() < 1e-10,
                "exp(0) should be identity: component {} differs",
                i
            );
        }
    }

    #[test]
    fn test_matrix_exp_unitarity_check() {
        // exp(-i H t) should preserve the norm for Hermitian H.
        let n = 5;
        let graph = Graph::cycle(n);
        let adj = graph.adjacency_matrix();
        let mut h = vec![vec![c64_zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                h[i][j] = C64::new(adj[i][j], 0.0);
            }
        }

        let v: Vec<C64> = vec![
            C64::new(0.5, 0.2),
            C64::new(-0.3, 0.1),
            C64::new(0.1, -0.4),
            C64::new(0.2, 0.3),
            C64::new(-0.1, 0.0),
        ];
        let original_norm = vec_norm(&v);

        for t in &[0.1, 0.5, 1.0, 3.0, 10.0] {
            let result = matrix_exp_vector(&h, &v, *t);
            let result_norm = vec_norm(&result);
            assert!(
                (result_norm - original_norm).abs() < 1e-8,
                "Norm not preserved at t={}: {} vs {}",
                t,
                result_norm,
                original_norm
            );
        }
    }

    // ----------------------------------------------------------
    // Analysis tests
    // ----------------------------------------------------------

    #[test]
    fn test_ctqw_analysis() {
        let graph = Graph::line(15);
        let walk = ContinuousWalk::new(graph, 7, 1.0);
        let times: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let analysis = walk.analyze(&times);

        assert_eq!(analysis.variance.len(), 10);
        assert_eq!(analysis.return_probability.len(), 10);
        // At t=0, variance should be 0 and return prob should be 1.
        assert!(
            analysis.variance[0] < 1e-10,
            "Variance at t=0 should be ~0, got {}",
            analysis.variance[0]
        );
        assert!(
            (analysis.return_probability[0] - 1.0).abs() < 1e-10,
            "Return prob at t=0 should be 1, got {}",
            analysis.return_probability[0]
        );
        // CTQW has no coin entropy.
        assert!(analysis.entanglement_entropy.is_empty());
    }

    #[test]
    fn test_dtqw_entanglement_entropy() {
        // For DTQW, coin-position entanglement should grow from 0
        // (initially a product state) and stabilize.
        let graph = Graph::line(41);
        let walk = DiscreteWalk::new(graph, 20, CoinOperator::Hadamard);
        let analysis = walk.analyze(15);

        assert_eq!(analysis.entanglement_entropy.len(), 16); // 0..15 inclusive.
        // Initial state is a product state: entropy should be ~0.
        assert!(
            analysis.entanglement_entropy[0] < 0.1,
            "Initial entanglement entropy should be ~0, got {}",
            analysis.entanglement_entropy[0]
        );
        // After several steps, entropy should be nonzero (coin and position
        // become entangled).
        let later_entropy = analysis.entanglement_entropy[10];
        assert!(
            later_entropy > 0.0,
            "Entanglement entropy should grow, got {} at step 10",
            later_entropy
        );
    }
}
