//! QUBO/Ising Encoder for Combinatorial Optimization
//!
//! Encodes combinatorial optimization problems as Quadratic Unconstrained Binary
//! Optimization (QUBO) matrices or Ising models, suitable for quantum annealing
//! and QAOA execution on nQPU-Metal.
//!
//! # Supported Problems
//!
//! - **Max-Cut**: Graph partitioning to maximize cut weight
//! - **Traveling Salesman (TSP)**: Shortest Hamiltonian cycle
//! - **Graph Coloring**: Chromatic assignment with minimum colors
//! - **Portfolio Optimization**: Risk-return tradeoff with budget constraint
//! - **Number Partitioning**: Equal-sum subset division
//! - **Minimum Vertex Cover**: Smallest set covering all edges
//! - **Maximum Independent Set**: Largest set with no adjacent vertices
//!
//! # QAOA Integration
//!
//! QUBO and Ising models can be converted to Pauli Hamiltonians for direct
//! use with QAOA circuits via [`qubo_to_qaoa_hamiltonian`] and
//! [`ising_to_qaoa_hamiltonian`].

use std::collections::HashMap;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from QUBO/Ising encoding and conversion.
#[derive(Debug, Clone)]
pub enum QuboError {
    /// Problem size is invalid (e.g., zero or mismatched dimensions).
    InvalidSize(String),
    /// A constraint cannot be satisfied by any binary assignment.
    InfeasibleConstraint(String),
    /// Conversion between QUBO and Ising representations failed.
    ConversionFailed(String),
}

impl fmt::Display for QuboError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuboError::InvalidSize(msg) => write!(f, "InvalidSize: {}", msg),
            QuboError::InfeasibleConstraint(msg) => write!(f, "InfeasibleConstraint: {}", msg),
            QuboError::ConversionFailed(msg) => write!(f, "ConversionFailed: {}", msg),
        }
    }
}

impl std::error::Error for QuboError {}

// ============================================================
// QUBO MATRIX
// ============================================================

/// Upper-triangular QUBO matrix representation.
///
/// Stores Q_{ij} with i <= j. Diagonal entries Q_{ii} are linear terms;
/// off-diagonal entries Q_{ij} (i < j) are quadratic interaction terms.
/// The objective is to minimize x^T Q x + offset for x ∈ {0,1}^n.
#[derive(Debug, Clone)]
pub struct QuboMatrix {
    /// Number of binary variables.
    pub size: usize,
    /// Sparse upper-triangular entries: (i, j) -> coefficient, where i <= j.
    pub entries: HashMap<(usize, usize), f64>,
    /// Constant energy offset.
    pub offset: f64,
}

impl QuboMatrix {
    /// Create a new empty QUBO matrix for `size` binary variables.
    pub fn new(size: usize) -> Self {
        QuboMatrix {
            size,
            entries: HashMap::new(),
            offset: 0.0,
        }
    }

    /// Add a linear term: Q_{i,i} += coeff.
    pub fn add_linear(&mut self, i: usize, coeff: f64) {
        assert!(
            i < self.size,
            "Variable index {} out of bounds (size {})",
            i,
            self.size
        );
        *self.entries.entry((i, i)).or_insert(0.0) += coeff;
    }

    /// Add a quadratic term: Q_{min(i,j), max(i,j)} += coeff.
    ///
    /// Automatically normalizes to upper-triangular form (i <= j).
    /// If i == j, this is equivalent to `add_linear`.
    pub fn add_quadratic(&mut self, i: usize, j: usize, coeff: f64) {
        assert!(
            i < self.size,
            "Variable index {} out of bounds (size {})",
            i,
            self.size
        );
        assert!(
            j < self.size,
            "Variable index {} out of bounds (size {})",
            j,
            self.size
        );
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        *self.entries.entry((lo, hi)).or_insert(0.0) += coeff;
    }

    /// Return the number of binary variables.
    pub fn num_variables(&self) -> usize {
        self.size
    }

    /// Convert to a dense n x n matrix (upper triangular).
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let n = self.size;
        let mut mat = vec![vec![0.0; n]; n];
        for (&(i, j), &val) in &self.entries {
            mat[i][j] = val;
        }
        mat
    }
}

// ============================================================
// ISING MODEL
// ============================================================

/// Ising model representation with linear fields h and quadratic couplings J.
///
/// Energy: E(s) = Σ_i h_i s_i + Σ_{i<j} J_{ij} s_i s_j + offset
/// where s_i ∈ {-1, +1}.
#[derive(Debug, Clone)]
pub struct IsingModel {
    /// Number of spin variables.
    pub num_spins: usize,
    /// Linear fields h_i for each spin.
    pub h: Vec<f64>,
    /// Quadratic couplings J_{ij} with i < j.
    pub j: HashMap<(usize, usize), f64>,
    /// Constant energy offset.
    pub offset: f64,
}

// ============================================================
// QUBO SOLUTION
// ============================================================

/// Solution to a QUBO problem.
#[derive(Debug, Clone)]
pub struct QuboSolution {
    /// Binary assignment: each entry is 0 or 1.
    pub bitstring: Vec<u8>,
    /// Objective value (energy) of this solution.
    pub energy: f64,
    /// Whether the solution satisfies all problem constraints.
    pub feasible: bool,
}

// ============================================================
// QUBO <-> ISING CONVERSION
// ============================================================

/// Convert a QUBO matrix to an Ising model.
///
/// Uses the substitution x_i = (1 + s_i) / 2 to transform
/// binary variables x_i ∈ {0,1} to spin variables s_i ∈ {-1,+1}.
///
/// The QUBO energy x^T Q x transforms as:
///   x_i * x_j = (1+s_i)(1+s_j)/4
///   x_i^2 = x_i = (1+s_i)/2
pub fn qubo_to_ising(qubo: &QuboMatrix) -> IsingModel {
    let n = qubo.size;
    let mut h = vec![0.0; n];
    let mut j_couplings: HashMap<(usize, usize), f64> = HashMap::new();
    let mut offset = qubo.offset;

    for (&(i, j), &q_val) in &qubo.entries {
        if q_val == 0.0 {
            continue;
        }
        if i == j {
            // Diagonal (linear) term: Q_ii * x_i = Q_ii * (1 + s_i)/2
            //   = Q_ii/2 + Q_ii/2 * s_i
            offset += q_val / 2.0;
            h[i] += q_val / 2.0;
        } else {
            // Off-diagonal (quadratic) term: Q_ij * x_i * x_j
            //   = Q_ij * (1+s_i)(1+s_j)/4
            //   = Q_ij/4 + Q_ij/4 * s_i + Q_ij/4 * s_j + Q_ij/4 * s_i*s_j
            offset += q_val / 4.0;
            h[i] += q_val / 4.0;
            h[j] += q_val / 4.0;
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            *j_couplings.entry((lo, hi)).or_insert(0.0) += q_val / 4.0;
        }
    }

    IsingModel {
        num_spins: n,
        h,
        j: j_couplings,
        offset,
    }
}

/// Convert an Ising model to a QUBO matrix.
///
/// Uses the substitution s_i = 2*x_i - 1 to transform
/// spin variables s_i ∈ {-1,+1} to binary variables x_i ∈ {0,1}.
///
/// Ising energy: h_i * s_i = h_i * (2x_i - 1) = 2*h_i*x_i - h_i
/// J_ij * s_i * s_j = J_ij * (2x_i-1)(2x_j-1) = 4*J_ij*x_i*x_j - 2*J_ij*x_i - 2*J_ij*x_j + J_ij
pub fn ising_to_qubo(ising: &IsingModel) -> QuboMatrix {
    let n = ising.num_spins;
    let mut qubo = QuboMatrix::new(n);
    qubo.offset = ising.offset;

    // Linear fields: h_i * s_i = h_i*(2x_i - 1) = 2*h_i*x_i - h_i
    for i in 0..n {
        if ising.h[i] != 0.0 {
            qubo.add_linear(i, 2.0 * ising.h[i]);
            qubo.offset -= ising.h[i];
        }
    }

    // Quadratic couplings: J_ij * s_i * s_j = J_ij*(2x_i-1)*(2x_j-1)
    //   = 4*J_ij*x_i*x_j - 2*J_ij*x_i - 2*J_ij*x_j + J_ij
    for (&(i, j), &j_val) in &ising.j {
        if j_val == 0.0 {
            continue;
        }
        qubo.add_quadratic(i, j, 4.0 * j_val);
        qubo.add_linear(i, -2.0 * j_val);
        qubo.add_linear(j, -2.0 * j_val);
        qubo.offset += j_val;
    }

    qubo
}

// ============================================================
// SOLUTION EVALUATION
// ============================================================

/// Evaluate the QUBO objective for a given binary assignment.
///
/// Computes x^T Q x + offset where x ∈ {0,1}^n.
pub fn evaluate_qubo(qubo: &QuboMatrix, solution: &[u8]) -> f64 {
    assert_eq!(
        solution.len(),
        qubo.size,
        "Solution length must match QUBO size"
    );
    let mut energy = qubo.offset;
    for (&(i, j), &val) in &qubo.entries {
        if i == j {
            energy += val * solution[i] as f64;
        } else {
            energy += val * solution[i] as f64 * solution[j] as f64;
        }
    }
    energy
}

/// Evaluate the Ising energy for a given spin configuration.
///
/// Computes Σ h_i s_i + Σ J_{ij} s_i s_j + offset where s_i ∈ {-1,+1}.
pub fn evaluate_ising(ising: &IsingModel, spins: &[i8]) -> f64 {
    assert_eq!(
        spins.len(),
        ising.num_spins,
        "Spins length must match Ising size"
    );
    let mut energy = ising.offset;
    for (i, &h_i) in ising.h.iter().enumerate() {
        energy += h_i * spins[i] as f64;
    }
    for (&(i, j), &j_val) in &ising.j {
        energy += j_val * spins[i] as f64 * spins[j] as f64;
    }
    energy
}

/// Brute-force solve a QUBO by enumerating all 2^n solutions.
///
/// Feasible for n <= 20. Returns the solution with minimum energy.
pub fn brute_force_solve(qubo: &QuboMatrix) -> QuboSolution {
    let n = qubo.size;
    assert!(
        n <= 20,
        "Brute force is only feasible for n <= 20, got n = {}",
        n
    );

    let mut best_energy = f64::INFINITY;
    let mut best_bitstring = vec![0u8; n];

    let total = 1u64 << n;
    for bits in 0..total {
        let solution: Vec<u8> = (0..n).map(|k| ((bits >> k) & 1) as u8).collect();
        let energy = evaluate_qubo(qubo, &solution);
        if energy < best_energy {
            best_energy = energy;
            best_bitstring = solution;
        }
    }

    QuboSolution {
        bitstring: best_bitstring,
        energy: best_energy,
        feasible: true,
    }
}

// ============================================================
// PROBLEM ENCODERS
// ============================================================

/// Encode a Max-Cut problem as a QUBO matrix.
///
/// Given a weighted graph, maximize the total weight of edges crossing
/// the partition defined by binary variables x_i ∈ {0,1}.
///
/// Objective: maximize Σ w_{ij} (x_i + x_j - 2 x_i x_j)
/// Equivalently minimize: Σ w_{ij} (2 x_i x_j - x_i - x_j)
///
/// # Arguments
/// - `adjacency`: Edge list as (node_i, node_j, weight).
/// - `num_nodes`: Total number of nodes in the graph.
pub fn encode_max_cut(adjacency: &[(usize, usize, f64)], num_nodes: usize) -> QuboMatrix {
    let mut qubo = QuboMatrix::new(num_nodes);

    for &(i, j, w) in adjacency {
        // Minimize -(x_i + x_j - 2*x_i*x_j) * w = -w*x_i - w*x_j + 2*w*x_i*x_j
        qubo.add_linear(i, -w);
        qubo.add_linear(j, -w);
        qubo.add_quadratic(i, j, 2.0 * w);
    }

    qubo
}

/// Encode a Traveling Salesman Problem (TSP) as a QUBO matrix.
///
/// Uses n^2 binary variables x_{i,t} indicating that city i is visited at step t.
/// Variable index for city i at time t: i * n + t.
///
/// # Arguments
/// - `distances`: n x n distance matrix between cities.
/// - `penalty`: Lagrange multiplier for constraint enforcement.
pub fn encode_tsp(distances: &Vec<Vec<f64>>, penalty: f64) -> QuboMatrix {
    let n = distances.len();
    let num_vars = n * n;
    let mut qubo = QuboMatrix::new(num_vars);

    // Helper: variable index for city i at time step t
    let var = |city: usize, time: usize| -> usize { city * n + time };

    // Objective: minimize total tour distance
    // For consecutive time steps t and t+1 (cyclically), add distance if city i at t and city j at t+1
    for t in 0..n {
        let t_next = (t + 1) % n;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let v_it = var(i, t);
                    let v_jt1 = var(j, t_next);
                    qubo.add_quadratic(v_it, v_jt1, distances[i][j]);
                }
            }
        }
    }

    // Constraint 1: Each city is visited exactly once
    // For each city i: (Σ_t x_{i,t} - 1)^2 * penalty
    // = penalty * (Σ_t x_{i,t}^2 - 2*Σ_t x_{i,t} + 1 + 2*Σ_{t<t'} x_{i,t}*x_{i,t'})
    // = penalty * (Σ_t x_{i,t} - 2*Σ_t x_{i,t} + 1 + 2*Σ_{t<t'} x_{i,t}*x_{i,t'})
    // = penalty * (-Σ_t x_{i,t} + 1 + 2*Σ_{t<t'} x_{i,t}*x_{i,t'})
    for i in 0..n {
        for t in 0..n {
            qubo.add_linear(var(i, t), -penalty);
        }
        for t1 in 0..n {
            for t2 in (t1 + 1)..n {
                qubo.add_quadratic(var(i, t1), var(i, t2), 2.0 * penalty);
            }
        }
        qubo.offset += penalty;
    }

    // Constraint 2: Each time step has exactly one city
    // For each time t: (Σ_i x_{i,t} - 1)^2 * penalty
    for t in 0..n {
        for i in 0..n {
            qubo.add_linear(var(i, t), -penalty);
        }
        for i1 in 0..n {
            for i2 in (i1 + 1)..n {
                qubo.add_quadratic(var(i1, t), var(i2, t), 2.0 * penalty);
            }
        }
        qubo.offset += penalty;
    }

    qubo
}

/// Encode a Graph Coloring problem as a QUBO matrix.
///
/// Uses n * num_colors binary variables: x_{i,c} = 1 if node i has color c.
///
/// # Arguments
/// - `edges`: Edge list as (node_i, node_j).
/// - `num_nodes`: Total number of nodes.
/// - `num_colors`: Number of available colors.
/// - `penalty`: Lagrange multiplier for constraint enforcement.
pub fn encode_graph_coloring(
    edges: &[(usize, usize)],
    num_nodes: usize,
    num_colors: usize,
    penalty: f64,
) -> QuboMatrix {
    let num_vars = num_nodes * num_colors;
    let mut qubo = QuboMatrix::new(num_vars);

    // Variable index: node i, color c -> i * num_colors + c
    let var = |node: usize, color: usize| -> usize { node * num_colors + color };

    // Constraint 1: Each node has exactly one color
    // For each node i: (Σ_c x_{i,c} - 1)^2 * penalty
    for i in 0..num_nodes {
        for c in 0..num_colors {
            qubo.add_linear(var(i, c), -penalty);
        }
        for c1 in 0..num_colors {
            for c2 in (c1 + 1)..num_colors {
                qubo.add_quadratic(var(i, c1), var(i, c2), 2.0 * penalty);
            }
        }
        qubo.offset += penalty;
    }

    // Constraint 2: Adjacent nodes must have different colors
    // For each edge (i,j) and each color c: x_{i,c} * x_{j,c} * penalty
    for &(i, j) in edges {
        for c in 0..num_colors {
            qubo.add_quadratic(var(i, c), var(j, c), penalty);
        }
    }

    qubo
}

/// Encode a Portfolio Optimization problem as a QUBO matrix.
///
/// Maximize expected returns minus risk, subject to a budget constraint.
///
/// Objective: maximize Σ r_i x_i - risk_factor * Σ_{i,j} C_{ij} x_i x_j
/// Constraint: Σ x_i = budget
///
/// # Arguments
/// - `expected_returns`: Expected return for each asset.
/// - `covariance`: Covariance matrix between assets.
/// - `risk_factor`: Weight given to risk (higher = more risk-averse).
/// - `budget`: Exact number of assets to select.
/// - `penalty`: Lagrange multiplier for budget constraint.
pub fn encode_portfolio(
    expected_returns: &[f64],
    covariance: &Vec<Vec<f64>>,
    risk_factor: f64,
    budget: usize,
    penalty: f64,
) -> QuboMatrix {
    let n = expected_returns.len();
    let mut qubo = QuboMatrix::new(n);

    // Objective: maximize returns, minimize risk
    // Minimize: -Σ r_i x_i + risk_factor * Σ_{i,j} C_{ij} x_i x_j
    for i in 0..n {
        qubo.add_linear(i, -expected_returns[i]);
    }

    for i in 0..n {
        for j in i..n {
            if i == j {
                qubo.add_linear(i, risk_factor * covariance[i][j]);
            } else {
                // C_{ij} + C_{ji} combined into upper triangle
                qubo.add_quadratic(i, j, risk_factor * (covariance[i][j] + covariance[j][i]));
            }
        }
    }

    // Budget constraint: (Σ x_i - budget)^2 * penalty
    // = penalty * (Σ x_i^2 - 2*budget*Σ x_i + budget^2 + 2*Σ_{i<j} x_i*x_j)
    // = penalty * ((1-2*budget)*Σ x_i + budget^2 + 2*Σ_{i<j} x_i*x_j)
    let b = budget as f64;
    for i in 0..n {
        qubo.add_linear(i, penalty * (1.0 - 2.0 * b));
    }
    for i in 0..n {
        for j in (i + 1)..n {
            qubo.add_quadratic(i, j, 2.0 * penalty);
        }
    }
    qubo.offset += penalty * b * b;

    qubo
}

/// Encode a Number Partitioning problem as a QUBO matrix.
///
/// Partition a set of integers into two subsets with equal (or near-equal) sum.
///
/// Minimize (Σ n_i (2x_i - 1))^2 where x_i = 1 means number i is in subset A.
///
/// # Arguments
/// - `numbers`: The set of integers to partition.
pub fn encode_number_partition(numbers: &[i64]) -> QuboMatrix {
    let n = numbers.len();
    let mut qubo = QuboMatrix::new(n);

    // (Σ n_i (2x_i - 1))^2
    // = (Σ 2*n_i*x_i - Σ n_i)^2
    // = 4*(Σ n_i*x_i)^2 - 4*(Σ n_i)*(Σ n_i*x_i) + (Σ n_i)^2
    //
    // (Σ n_i*x_i)^2 = Σ n_i^2 * x_i + 2*Σ_{i<j} n_i*n_j*x_i*x_j  (using x_i^2 = x_i)
    //
    // So the QUBO is:
    //   4*n_i^2*x_i + 8*n_i*n_j*x_i*x_j - 4*S*n_i*x_i + S^2
    // where S = Σ n_i

    let s: f64 = numbers.iter().map(|&x| x as f64).sum();

    for i in 0..n {
        let ni = numbers[i] as f64;
        qubo.add_linear(i, 4.0 * ni * ni - 4.0 * s * ni);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let ni = numbers[i] as f64;
            let nj = numbers[j] as f64;
            qubo.add_quadratic(i, j, 8.0 * ni * nj);
        }
    }

    qubo.offset = s * s;

    qubo
}

/// Encode a Minimum Vertex Cover problem as a QUBO matrix.
///
/// Find the smallest set of vertices such that every edge has at least
/// one endpoint in the set.
///
/// Objective: minimize Σ x_i
/// Constraint: for each edge (i,j), x_i + x_j >= 1
///   Enforced as penalty * (1 - x_i - x_j + x_i*x_j) for violation
///
/// # Arguments
/// - `edges`: Edge list as (node_i, node_j).
/// - `num_nodes`: Total number of nodes.
/// - `penalty`: Lagrange multiplier for edge coverage constraint.
pub fn encode_vertex_cover(edges: &[(usize, usize)], num_nodes: usize, penalty: f64) -> QuboMatrix {
    let mut qubo = QuboMatrix::new(num_nodes);

    // Objective: minimize number of selected vertices
    for i in 0..num_nodes {
        qubo.add_linear(i, 1.0);
    }

    // Constraint: for each edge (i,j), x_i + x_j >= 1
    // Penalize the case x_i = 0, x_j = 0:
    // penalty * (1 - x_i)(1 - x_j) = penalty * (1 - x_i - x_j + x_i*x_j)
    for &(i, j) in edges {
        qubo.add_linear(i, -penalty);
        qubo.add_linear(j, -penalty);
        qubo.add_quadratic(i, j, penalty);
        qubo.offset += penalty;
    }

    qubo
}

/// Encode a Maximum Independent Set problem as a QUBO matrix.
///
/// Find the largest set of vertices such that no two are adjacent.
///
/// Objective: maximize Σ x_i (equivalently minimize -Σ x_i)
/// Constraint: for each edge (i,j), x_i * x_j = 0
///
/// # Arguments
/// - `edges`: Edge list as (node_i, node_j).
/// - `num_nodes`: Total number of nodes.
/// - `penalty`: Lagrange multiplier for independence constraint.
pub fn encode_independent_set(
    edges: &[(usize, usize)],
    num_nodes: usize,
    penalty: f64,
) -> QuboMatrix {
    let mut qubo = QuboMatrix::new(num_nodes);

    // Objective: maximize Σ x_i → minimize -Σ x_i
    for i in 0..num_nodes {
        qubo.add_linear(i, -1.0);
    }

    // Constraint: for each edge (i,j), penalize x_i * x_j = 1
    for &(i, j) in edges {
        qubo.add_quadratic(i, j, penalty);
    }

    qubo
}

// ============================================================
// QAOA INTEGRATION
// ============================================================

/// Convert a QUBO matrix to a Pauli Hamiltonian for QAOA.
///
/// Returns a list of (Pauli term, coefficient) pairs where each Pauli term
/// is a list of (qubit_index, pauli_char) with pauli_char ∈ {'Z', 'I'}.
///
/// The mapping uses x_i = (I - Z_i) / 2:
///   - Linear Q_{ii}: coefficient on Z_i
///   - Quadratic Q_{ij}: coefficient on Z_i Z_j
///   - Constant offset from the identity terms
pub fn qubo_to_qaoa_hamiltonian(qubo: &QuboMatrix) -> Vec<(Vec<(usize, char)>, f64)> {
    // Convert through Ising first, then to Pauli Hamiltonian
    let ising = qubo_to_ising(qubo);
    ising_to_qaoa_hamiltonian(&ising)
}

/// Convert an Ising model to a Pauli Hamiltonian for QAOA.
///
/// Direct mapping: s_i → Z_i
///   - h_i * s_i → h_i * Z_i
///   - J_{ij} * s_i * s_j → J_{ij} * Z_i Z_j
///   - offset → offset * I
pub fn ising_to_qaoa_hamiltonian(ising: &IsingModel) -> Vec<(Vec<(usize, char)>, f64)> {
    let mut terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();

    // Identity (constant offset)
    if ising.offset.abs() > 1e-15 {
        terms.push((vec![], ising.offset));
    }

    // Single-qubit Z terms from linear fields
    for (i, &h_i) in ising.h.iter().enumerate() {
        if h_i.abs() > 1e-15 {
            terms.push((vec![(i, 'Z')], h_i));
        }
    }

    // Two-qubit ZZ terms from quadratic couplings
    for (&(i, j), &j_val) in &ising.j {
        if j_val.abs() > 1e-15 {
            terms.push((vec![(i, 'Z'), (j, 'Z')], j_val));
        }
    }

    terms
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: convert bitstring to spins for Ising evaluation.
    fn bitstring_to_spins(bits: &[u8]) -> Vec<i8> {
        bits.iter().map(|&b| if b == 0 { -1 } else { 1 }).collect()
    }

    // ----- Test 1: QUBO to Ising round-trip preserves energy -----
    #[test]
    fn test_qubo_ising_round_trip_energy() {
        let mut qubo = QuboMatrix::new(3);
        qubo.add_linear(0, 1.0);
        qubo.add_linear(1, -2.0);
        qubo.add_linear(2, 0.5);
        qubo.add_quadratic(0, 1, 3.0);
        qubo.add_quadratic(1, 2, -1.5);
        qubo.offset = 2.0;

        let ising = qubo_to_ising(&qubo);

        // For every possible bitstring, energy should be preserved
        for bits in 0u8..8 {
            let solution: Vec<u8> = (0..3).map(|k| (bits >> k) & 1).collect();
            let spins = bitstring_to_spins(&solution);
            let qubo_energy = evaluate_qubo(&qubo, &solution);
            let ising_energy = evaluate_ising(&ising, &spins);
            assert!(
                (qubo_energy - ising_energy).abs() < 1e-10,
                "Energy mismatch for {:?}: QUBO={} Ising={}",
                solution,
                qubo_energy,
                ising_energy
            );
        }
    }

    // ----- Test 2: Ising to QUBO round-trip preserves energy -----
    #[test]
    fn test_ising_qubo_round_trip_energy() {
        let ising = IsingModel {
            num_spins: 3,
            h: vec![0.5, -1.0, 0.25],
            j: HashMap::from([((0, 1), 1.5), ((1, 2), -0.75)]),
            offset: 1.0,
        };

        let qubo = ising_to_qubo(&ising);
        let ising2 = qubo_to_ising(&qubo);

        // Verify round-trip through all spin configs
        for bits in 0u8..8 {
            let spins: Vec<i8> = (0..3)
                .map(|k| if (bits >> k) & 1 == 0 { -1 } else { 1 })
                .collect();
            let e1 = evaluate_ising(&ising, &spins);
            let e2 = evaluate_ising(&ising2, &spins);
            assert!(
                (e1 - e2).abs() < 1e-10,
                "Round-trip energy mismatch for {:?}: {} vs {}",
                spins,
                e1,
                e2
            );
        }
    }

    // ----- Test 3: Max-Cut encoding for triangle graph -----
    #[test]
    fn test_max_cut_triangle() {
        // Triangle graph: 3 nodes, 3 edges, all weight 1.0
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let qubo = encode_max_cut(&edges, 3);

        // In a triangle, max cut = 2 (any partition of 1 vs 2 nodes)
        let sol = brute_force_solve(&qubo);
        // Energy should be -2.0 (we minimize the negated cut value)
        assert!(
            (sol.energy - (-2.0)).abs() < 1e-10,
            "Max-cut of triangle should be -2.0, got {}",
            sol.energy
        );
    }

    // ----- Test 4: Max-Cut brute force matches for 4-node graph -----
    #[test]
    fn test_max_cut_4_nodes() {
        // Square graph with diagonal
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 0, 1.0),
            (0, 2, 1.0), // diagonal
        ];
        let qubo = encode_max_cut(&edges, 4);
        let sol = brute_force_solve(&qubo);

        // Verify the brute-force result: the best solution energy must match
        // re-evaluated energy
        let recalc = evaluate_qubo(&qubo, &sol.bitstring);
        assert!(
            (sol.energy - recalc).abs() < 1e-10,
            "Brute force energy doesn't match recalculation"
        );

        // Max cut of a square+diagonal with all unit weights = 4
        // (partition {0,2} vs {1,3} cuts edges 0-1, 1-2, 2-3, 3-0 = 4 edges)
        assert!(
            (sol.energy - (-4.0)).abs() < 1e-10,
            "Expected max cut value -4.0, got {}",
            sol.energy
        );
    }

    // ----- Test 5: TSP encoding size = n^2 variables -----
    #[test]
    fn test_tsp_encoding_size() {
        let n = 4;
        let distances = vec![vec![0.0; n]; n];
        let qubo = encode_tsp(&distances, 100.0);
        assert_eq!(qubo.num_variables(), n * n, "TSP should have n^2 variables");
    }

    // ----- Test 6: TSP 3 cities brute force finds optimal tour -----
    #[test]
    fn test_tsp_3_cities_brute_force() {
        let distances = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.5],
            vec![2.0, 1.5, 0.0],
        ];
        let penalty = 100.0;
        let qubo = encode_tsp(&distances, penalty);
        let sol = brute_force_solve(&qubo);

        // Verify the solution is a valid tour (each city once, each time once)
        let n = 3;
        let mut assignment = vec![vec![0u8; n]; n];
        for i in 0..n {
            for t in 0..n {
                assignment[i][t] = sol.bitstring[i * n + t];
            }
        }

        // Each city should appear exactly once
        for i in 0..n {
            let city_sum: u8 = assignment[i].iter().sum();
            assert_eq!(
                city_sum, 1,
                "City {} should be visited exactly once, got {}",
                i, city_sum
            );
        }

        // Each time step should have exactly one city
        for t in 0..n {
            let time_sum: u8 = (0..n).map(|i| assignment[i][t]).sum();
            assert_eq!(
                time_sum, 1,
                "Time {} should have exactly one city, got {}",
                t, time_sum
            );
        }
    }

    // ----- Test 7: Graph coloring valid solution -----
    #[test]
    fn test_graph_coloring_valid() {
        // Triangle graph with 3 colors (chromatic number = 3)
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let num_nodes = 3;
        let num_colors = 3;
        let penalty = 100.0;

        let qubo = encode_graph_coloring(&edges, num_nodes, num_colors, penalty);
        let sol = brute_force_solve(&qubo);

        // Extract coloring
        let mut colors = vec![None; num_nodes];
        for i in 0..num_nodes {
            for c in 0..num_colors {
                if sol.bitstring[i * num_colors + c] == 1 {
                    assert!(colors[i].is_none(), "Node {} has multiple colors", i);
                    colors[i] = Some(c);
                }
            }
        }

        // Each node should have exactly one color
        for i in 0..num_nodes {
            assert!(colors[i].is_some(), "Node {} has no color", i);
        }

        // No adjacent nodes should share a color
        for &(i, j) in &edges {
            assert_ne!(
                colors[i], colors[j],
                "Adjacent nodes {} and {} have the same color {:?}",
                i, j, colors[i]
            );
        }
    }

    // ----- Test 8: Portfolio budget constraint satisfied -----
    #[test]
    fn test_portfolio_budget_constraint() {
        let returns = vec![0.10, 0.15, 0.08, 0.12];
        let cov = vec![
            vec![0.04, 0.01, 0.00, 0.02],
            vec![0.01, 0.09, 0.01, 0.01],
            vec![0.00, 0.01, 0.02, 0.00],
            vec![0.02, 0.01, 0.00, 0.06],
        ];
        let budget = 2;
        let penalty = 100.0;
        let risk_factor = 1.0;

        let qubo = encode_portfolio(&returns, &cov, risk_factor, budget, penalty);
        let sol = brute_force_solve(&qubo);

        let selected: usize = sol.bitstring.iter().map(|&b| b as usize).sum();
        assert_eq!(
            selected, budget,
            "Portfolio should select exactly {} assets, got {}",
            budget, selected
        );
    }

    // ----- Test 9: Number partition of [1,2,3] finds equal partition -----
    #[test]
    fn test_number_partition() {
        let numbers = vec![1, 2, 3];
        // Sum = 6, equal partition means each subset sums to 3
        // Possible: {3} vs {1,2}, i.e., x = [0,0,1] or [1,1,0]
        let qubo = encode_number_partition(&numbers);
        let sol = brute_force_solve(&qubo);

        // The minimum energy should be 0 (perfect partition exists)
        // Subset A sum = Σ n_i * x_i, Subset B sum = Σ n_i * (1-x_i)
        // Difference = Σ n_i * (2x_i - 1)
        let diff: i64 = numbers
            .iter()
            .zip(sol.bitstring.iter())
            .map(|(&n, &x)| n * (2 * x as i64 - 1))
            .sum();
        assert_eq!(diff, 0, "Partition difference should be 0, got {}", diff);
    }

    // ----- Test 10: Vertex cover covers all edges -----
    #[test]
    fn test_vertex_cover() {
        // Path graph: 0-1-2
        let edges = vec![(0, 1), (1, 2)];
        let num_nodes = 3;
        let penalty = 10.0;

        let qubo = encode_vertex_cover(&edges, num_nodes, penalty);
        let sol = brute_force_solve(&qubo);

        // Every edge should have at least one endpoint selected
        for &(i, j) in &edges {
            assert!(
                sol.bitstring[i] == 1 || sol.bitstring[j] == 1,
                "Edge ({}, {}) not covered",
                i,
                j
            );
        }

        // Optimal: select node 1 covers both edges (cover size = 1)
        let cover_size: u8 = sol.bitstring.iter().sum();
        assert_eq!(
            cover_size, 1,
            "Minimum vertex cover of path 0-1-2 is 1, got {}",
            cover_size
        );
    }

    // ----- Test 11: Independent set has no adjacent selected nodes -----
    #[test]
    fn test_independent_set() {
        // Triangle graph
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let num_nodes = 3;
        let penalty = 10.0;

        let qubo = encode_independent_set(&edges, num_nodes, penalty);
        let sol = brute_force_solve(&qubo);

        // No two adjacent nodes should both be selected
        for &(i, j) in &edges {
            assert!(
                !(sol.bitstring[i] == 1 && sol.bitstring[j] == 1),
                "Adjacent nodes {} and {} both selected",
                i,
                j
            );
        }

        // For a triangle, max independent set = 1
        let set_size: u8 = sol.bitstring.iter().sum();
        assert_eq!(
            set_size, 1,
            "Max independent set of triangle is 1, got {}",
            set_size
        );
    }

    // ----- Test 12: QUBO evaluation matches manual computation -----
    #[test]
    fn test_qubo_evaluation_manual() {
        // Q = [[1, 2], [0, -1]], offset = 0.5
        // For x = [1, 1]: E = 1*1 + 2*1*1 + (-1)*1 + 0.5 = 2.5
        let mut qubo = QuboMatrix::new(2);
        qubo.add_linear(0, 1.0);
        qubo.add_quadratic(0, 1, 2.0);
        qubo.add_linear(1, -1.0);
        qubo.offset = 0.5;

        let e00 = evaluate_qubo(&qubo, &[0, 0]);
        let e01 = evaluate_qubo(&qubo, &[0, 1]);
        let e10 = evaluate_qubo(&qubo, &[1, 0]);
        let e11 = evaluate_qubo(&qubo, &[1, 1]);

        assert!((e00 - 0.5).abs() < 1e-10, "E(0,0) = 0.5, got {}", e00);
        assert!((e01 - (-0.5)).abs() < 1e-10, "E(0,1) = -0.5, got {}", e01);
        assert!((e10 - 1.5).abs() < 1e-10, "E(1,0) = 1.5, got {}", e10);
        assert!((e11 - 2.5).abs() < 1e-10, "E(1,1) = 2.5, got {}", e11);
    }

    // ----- Test 13: Ising evaluation matches manual computation -----
    #[test]
    fn test_ising_evaluation_manual() {
        // h = [0.5, -1.0], J = {(0,1): 0.25}, offset = 1.0
        // For s = [+1, -1]: E = 0.5*1 + (-1)*(-1) + 0.25*1*(-1) + 1.0 = 2.25
        let ising = IsingModel {
            num_spins: 2,
            h: vec![0.5, -1.0],
            j: HashMap::from([((0, 1), 0.25)]),
            offset: 1.0,
        };

        let e_pp = evaluate_ising(&ising, &[1, 1]); // 0.5 + (-1.0) + 0.25 + 1.0 = 0.75
        let e_pm = evaluate_ising(&ising, &[1, -1]); // 0.5 + 1.0 + (-0.25) + 1.0 = 2.25
        let e_mp = evaluate_ising(&ising, &[-1, 1]); // -0.5 + (-1.0) + (-0.25) + 1.0 = -0.75
        let e_mm = evaluate_ising(&ising, &[-1, -1]); // -0.5 + 1.0 + 0.25 + 1.0 = 1.75

        assert!((e_pp - 0.75).abs() < 1e-10, "E(+1,+1) = 0.75, got {}", e_pp);
        assert!((e_pm - 2.25).abs() < 1e-10, "E(+1,-1) = 2.25, got {}", e_pm);
        assert!(
            (e_mp - (-0.75)).abs() < 1e-10,
            "E(-1,+1) = -0.75, got {}",
            e_mp
        );
        assert!((e_mm - 1.75).abs() < 1e-10, "E(-1,-1) = 1.75, got {}", e_mm);
    }

    // ----- Test 14: QAOA Hamiltonian has correct number of terms -----
    #[test]
    fn test_qaoa_hamiltonian_term_count() {
        let mut qubo = QuboMatrix::new(3);
        qubo.add_linear(0, 1.0);
        qubo.add_linear(1, 2.0);
        qubo.add_linear(2, -1.0);
        qubo.add_quadratic(0, 1, 0.5);
        qubo.add_quadratic(1, 2, 0.3);
        qubo.offset = 1.0;

        let hamiltonian = qubo_to_qaoa_hamiltonian(&qubo);

        // Count term types
        let identity_terms = hamiltonian.iter().filter(|(ops, _)| ops.is_empty()).count();
        let single_z = hamiltonian.iter().filter(|(ops, _)| ops.len() == 1).count();
        let zz_terms = hamiltonian.iter().filter(|(ops, _)| ops.len() == 2).count();

        // Should have: 1 identity (offset), 3 single-Z (linear), 2 ZZ (quadratic)
        assert_eq!(identity_terms, 1, "Should have 1 identity term");
        assert_eq!(single_z, 3, "Should have 3 single-Z terms");
        assert_eq!(zz_terms, 2, "Should have 2 ZZ terms");
        assert_eq!(hamiltonian.len(), 6, "Total terms should be 6");

        // Verify all operators are 'Z'
        for (ops, _) in &hamiltonian {
            for &(_, pauli) in ops {
                assert_eq!(pauli, 'Z', "All Pauli operators should be Z");
            }
        }
    }

    // ----- Test 15: Brute force finds global minimum -----
    #[test]
    fn test_brute_force_global_minimum() {
        // Known minimum: x = [1, 0, 1] gives energy = -5 + offset
        let mut qubo = QuboMatrix::new(3);
        qubo.add_linear(0, -3.0); // x_0 = 1 contributes -3
        qubo.add_linear(1, 5.0); // x_1 = 1 costs +5, so avoid
        qubo.add_linear(2, -2.0); // x_2 = 1 contributes -2
        qubo.add_quadratic(0, 1, 10.0); // penalty for both x_0 and x_1 on
        qubo.add_quadratic(0, 2, -1.0); // bonus for x_0 and x_2 both on

        let sol = brute_force_solve(&qubo);

        // x = [1, 0, 1]: E = -3 + 0 + (-2) + 0 + (-1) = -6
        assert_eq!(
            sol.bitstring,
            vec![1, 0, 1],
            "Optimal solution should be [1, 0, 1]"
        );
        assert!(
            (sol.energy - (-6.0)).abs() < 1e-10,
            "Minimum energy should be -6.0, got {}",
            sol.energy
        );

        // Verify no other solution has lower energy
        for bits in 0u8..8 {
            let x: Vec<u8> = (0..3).map(|k| (bits >> k) & 1).collect();
            let e = evaluate_qubo(&qubo, &x);
            assert!(
                e >= sol.energy - 1e-10,
                "Solution {:?} has energy {} which is lower than brute-force minimum {}",
                x,
                e,
                sol.energy
            );
        }
    }

    // ----- Test 16: QUBO to_dense matches entries -----
    #[test]
    fn test_qubo_to_dense() {
        let mut qubo = QuboMatrix::new(3);
        qubo.add_linear(0, 2.0);
        qubo.add_linear(2, -1.0);
        qubo.add_quadratic(0, 1, 3.0);
        qubo.add_quadratic(1, 2, 0.5);

        let dense = qubo.to_dense();
        assert_eq!(dense[0][0], 2.0);
        assert_eq!(dense[0][1], 3.0);
        assert_eq!(dense[1][2], 0.5);
        assert_eq!(dense[2][2], -1.0);
        // Lower triangle should be zero
        assert_eq!(dense[1][0], 0.0);
        assert_eq!(dense[2][0], 0.0);
        assert_eq!(dense[2][1], 0.0);
    }

    // ----- Test 17: QAOA Hamiltonian energy matches QUBO -----
    #[test]
    fn test_qaoa_hamiltonian_energy_consistency() {
        let mut qubo = QuboMatrix::new(3);
        qubo.add_linear(0, 1.5);
        qubo.add_linear(1, -0.5);
        qubo.add_quadratic(0, 1, 2.0);
        qubo.add_quadratic(0, 2, -1.0);
        qubo.offset = 0.5;

        let hamiltonian = qubo_to_qaoa_hamiltonian(&qubo);

        // For each bitstring, the QAOA Hamiltonian evaluated with Z eigenvalues
        // should match the QUBO energy
        for bits in 0u8..8 {
            let solution: Vec<u8> = (0..3).map(|k| (bits >> k) & 1).collect();
            let spins: Vec<i8> = solution
                .iter()
                .map(|&b| if b == 0 { -1i8 } else { 1i8 })
                .collect();

            let qubo_energy = evaluate_qubo(&qubo, &solution);

            // Evaluate Hamiltonian: each Z_i has eigenvalue s_i
            let mut ham_energy = 0.0;
            for (ops, coeff) in &hamiltonian {
                let mut term_val = 1.0;
                for &(qubit, _pauli) in ops {
                    term_val *= spins[qubit] as f64;
                }
                ham_energy += coeff * term_val;
            }

            assert!(
                (qubo_energy - ham_energy).abs() < 1e-10,
                "Hamiltonian energy mismatch for {:?}: QUBO={} HAM={}",
                solution,
                qubo_energy,
                ham_energy
            );
        }
    }
}
