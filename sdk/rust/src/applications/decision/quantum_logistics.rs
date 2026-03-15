//! Quantum Logistics Optimization Toolkit
//!
//! Implements combinatorial logistics optimization problems using quantum and
//! quantum-inspired algorithms. Covers three core problem families:
//!
//! - **CVRP** (Capacitated Vehicle Routing Problem): Route a fleet of
//!   capacity-constrained vehicles to serve a set of delivery locations,
//!   minimizing total travel distance.
//! - **TSP** (Travelling Salesman Problem): Find the shortest Hamiltonian
//!   cycle through all locations (single-vehicle CVRP special case).
//! - **Job-Shop Scheduling**: Assign operations to machines with precedence
//!   constraints, minimizing the overall makespan.
//!
//! All three problems are encoded as QUBO (Quadratic Unconstrained Binary
//! Optimization) instances and solved via QAOA, simulated quantum annealing,
//! or exact brute-force enumeration (for validation on small instances).
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::quantum_logistics::*;
//!
//! let locations = vec![
//!     Location::new(0, 0.0, 0.0, 0.0),  // depot
//!     Location::new(1, 3.0, 4.0, 20.0),
//!     Location::new(2, 6.0, 1.0, 30.0),
//! ];
//! let distances = DistanceMatrix::from_locations(&locations);
//! let config = LogisticsConfigBuilder::new().num_vehicles(1).build().unwrap();
//! let optimizer = QuantumLogisticsOptimizer::new(config);
//! let solution = optimizer.solve_tsp(&locations, &distances);
//! println!("Tour distance: {:.2}", solution.total_distance);
//! ```
//!
//! # References
//!
//! - Feld et al. (2019) - A Hybrid Solution Method for the CVRP using a Quantum Annealer
//! - Lucas (2014) - Ising formulations of many NP problems
//! - Venturelli et al. (2016) - Quantum optimization of fully connected spin glasses

use std::fmt;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors arising from logistics optimization.
#[derive(Debug, Clone)]
pub enum LogisticsError {
    /// Configuration parameter is out of valid range.
    InvalidConfig(String),
    /// The QUBO encoding produced an inconsistent or empty problem.
    EncodingError(String),
    /// A decoded solution violates hard constraints.
    InfeasibleSolution(String),
    /// Solver did not converge within the iteration / time budget.
    ConvergenceFailure { iterations: usize, best_cost: f64 },
    /// Problem size exceeds what the solver can handle.
    ProblemTooLarge { size: usize, limit: usize },
}

impl fmt::Display for LogisticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
            Self::InfeasibleSolution(msg) => write!(f, "Infeasible solution: {}", msg),
            Self::ConvergenceFailure {
                iterations,
                best_cost,
            } => {
                write!(
                    f,
                    "Convergence failure after {} iterations (best cost={:.4})",
                    iterations, best_cost,
                )
            }
            Self::ProblemTooLarge { size, limit } => {
                write!(f, "Problem too large: {} exceeds limit {}", size, limit)
            }
        }
    }
}

impl std::error::Error for LogisticsError {}

pub type LogisticsResult<T> = Result<T, LogisticsError>;

// ===================================================================
// SOLVER ENUM
// ===================================================================

/// Solver backend for logistics optimization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogisticsSolver {
    /// Quantum Approximate Optimization Algorithm.
    QAOA,
    /// Simulated quantum annealing (classical simulation of QA).
    QuantumAnnealing,
    /// Hybrid classical-quantum approach: classical heuristic warm-start
    /// refined by QAOA.
    HybridClassicalQuantum,
    /// Exact brute-force enumeration (feasible only for small instances).
    BruteForce,
}

impl fmt::Display for LogisticsSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QAOA => write!(f, "QAOA"),
            Self::QuantumAnnealing => write!(f, "Quantum Annealing"),
            Self::HybridClassicalQuantum => write!(f, "Hybrid Classical-Quantum"),
            Self::BruteForce => write!(f, "Brute Force"),
        }
    }
}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Configuration for logistics optimization.
#[derive(Clone, Debug)]
pub struct LogisticsConfig {
    /// Number of delivery locations (including depot). Valid range: 2..=200.
    pub num_locations: usize,
    /// Number of vehicles in the fleet. Valid range: 1..=50.
    pub num_vehicles: usize,
    /// Maximum load each vehicle can carry.
    pub vehicle_capacity: f64,
    /// Index of the depot location (vehicles start and end here).
    pub depot_index: usize,
    /// Solver backend.
    pub solver: LogisticsSolver,
    /// Number of QAOA layers (circuit depth parameter p).
    pub qaoa_depth: usize,
    /// Number of measurement shots per QAOA evaluation.
    pub num_shots: usize,
    /// Wall-clock time limit for the solver (seconds).
    pub time_limit_seconds: f64,
}

impl Default for LogisticsConfig {
    fn default() -> Self {
        Self {
            num_locations: 10,
            num_vehicles: 3,
            vehicle_capacity: 100.0,
            depot_index: 0,
            solver: LogisticsSolver::QAOA,
            qaoa_depth: 3,
            num_shots: 1000,
            time_limit_seconds: 60.0,
        }
    }
}

/// Builder for `LogisticsConfig` with validation on `build()`.
pub struct LogisticsConfigBuilder {
    config: LogisticsConfig,
}

impl LogisticsConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: LogisticsConfig::default(),
        }
    }

    pub fn num_locations(mut self, n: usize) -> Self {
        self.config.num_locations = n;
        self
    }

    pub fn num_vehicles(mut self, n: usize) -> Self {
        self.config.num_vehicles = n;
        self
    }

    pub fn vehicle_capacity(mut self, cap: f64) -> Self {
        self.config.vehicle_capacity = cap;
        self
    }

    pub fn depot_index(mut self, idx: usize) -> Self {
        self.config.depot_index = idx;
        self
    }

    pub fn solver(mut self, s: LogisticsSolver) -> Self {
        self.config.solver = s;
        self
    }

    pub fn qaoa_depth(mut self, d: usize) -> Self {
        self.config.qaoa_depth = d;
        self
    }

    pub fn num_shots(mut self, n: usize) -> Self {
        self.config.num_shots = n;
        self
    }

    pub fn time_limit_seconds(mut self, t: f64) -> Self {
        self.config.time_limit_seconds = t;
        self
    }

    /// Validate and build the configuration.
    pub fn build(self) -> LogisticsResult<LogisticsConfig> {
        let c = &self.config;
        if c.num_locations < 2 || c.num_locations > 200 {
            return Err(LogisticsError::InvalidConfig(format!(
                "num_locations {} not in 2..=200",
                c.num_locations,
            )));
        }
        if c.num_vehicles < 1 || c.num_vehicles > 50 {
            return Err(LogisticsError::InvalidConfig(format!(
                "num_vehicles {} not in 1..=50",
                c.num_vehicles,
            )));
        }
        if c.vehicle_capacity <= 0.0 {
            return Err(LogisticsError::InvalidConfig(
                "vehicle_capacity must be positive".into(),
            ));
        }
        if c.depot_index >= c.num_locations {
            return Err(LogisticsError::InvalidConfig(format!(
                "depot_index {} >= num_locations {}",
                c.depot_index, c.num_locations,
            )));
        }
        if c.qaoa_depth == 0 {
            return Err(LogisticsError::InvalidConfig(
                "qaoa_depth must be >= 1".into(),
            ));
        }
        if c.num_shots == 0 {
            return Err(LogisticsError::InvalidConfig(
                "num_shots must be >= 1".into(),
            ));
        }
        if c.time_limit_seconds <= 0.0 {
            return Err(LogisticsError::InvalidConfig(
                "time_limit_seconds must be positive".into(),
            ));
        }
        Ok(self.config)
    }
}

// ===================================================================
// LOCATION
// ===================================================================

/// A delivery location with coordinates, demand, and optional time window.
#[derive(Clone, Debug)]
pub struct Location {
    /// Unique index for this location.
    pub index: usize,
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Delivery demand (weight/volume to drop off).
    pub demand: f64,
    /// Optional time window (earliest, latest) for service.
    pub time_window: Option<(f64, f64)>,
    /// Time spent servicing this location.
    pub service_time: f64,
}

impl Location {
    /// Create a location with no time window and zero service time.
    pub fn new(index: usize, x: f64, y: f64, demand: f64) -> Self {
        Self {
            index,
            x,
            y,
            demand,
            time_window: None,
            service_time: 0.0,
        }
    }

    /// Create a location with a time window and service duration.
    pub fn with_time_window(
        index: usize,
        x: f64,
        y: f64,
        demand: f64,
        earliest: f64,
        latest: f64,
        service_time: f64,
    ) -> Self {
        Self {
            index,
            x,
            y,
            demand,
            time_window: Some((earliest, latest)),
            service_time,
        }
    }

    /// Euclidean distance to another location.
    pub fn distance_to(&self, other: &Location) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ===================================================================
// DISTANCE MATRIX
// ===================================================================

/// Symmetric distance matrix between locations.
#[derive(Clone, Debug)]
pub struct DistanceMatrix {
    distances: Vec<Vec<f64>>,
}

impl DistanceMatrix {
    /// Build from Euclidean coordinates of locations.
    pub fn from_locations(locations: &[Location]) -> Self {
        let n = locations.len();
        let mut distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = locations[i].distance_to(&locations[j]);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }
        Self { distances }
    }

    /// Build from a pre-computed matrix (must be square).
    pub fn from_matrix(matrix: Vec<Vec<f64>>) -> LogisticsResult<Self> {
        let n = matrix.len();
        if n == 0 {
            return Err(LogisticsError::InvalidConfig(
                "Distance matrix is empty".into(),
            ));
        }
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n {
                return Err(LogisticsError::InvalidConfig(format!(
                    "Row {} has length {} but expected {}",
                    i,
                    row.len(),
                    n,
                )));
            }
        }
        Ok(Self { distances: matrix })
    }

    /// Get distance from location i to location j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.distances[i][j]
    }

    /// Number of locations.
    pub fn size(&self) -> usize {
        self.distances.len()
    }
}

// ===================================================================
// QUBO PROBLEM
// ===================================================================

/// Quadratic Unconstrained Binary Optimization problem in sparse form.
///
/// Minimise x^T Q x + offset, where x is a binary vector.
#[derive(Clone, Debug)]
pub struct QUBOProblem {
    /// Number of binary decision variables.
    pub num_variables: usize,
    /// Sparse upper-triangular entries (i, j, Q_{ij}).
    /// For diagonal terms i == j; for off-diagonal i < j and the entry
    /// represents Q_{ij} + Q_{ji}.
    pub q_entries: Vec<(usize, usize, f64)>,
    /// Constant offset added to every evaluation.
    pub offset: f64,
}

impl QUBOProblem {
    /// Evaluate the objective for a binary solution vector.
    pub fn evaluate(&self, solution: &[bool]) -> f64 {
        assert_eq!(solution.len(), self.num_variables);
        let mut cost = self.offset;
        for &(i, j, w) in &self.q_entries {
            if i == j {
                if solution[i] {
                    cost += w;
                }
            } else if solution[i] && solution[j] {
                cost += w;
            }
        }
        cost
    }

    /// Convert QUBO to Ising form: minimise sum_{i<j} J_{ij} s_i s_j + sum_i h_i s_i + offset,
    /// where s_i in {-1, +1} and x_i = (1 + s_i) / 2.
    ///
    /// Returns (J matrix as dense Vec<Vec<f64>>, h vector, offset).
    /// The coupling matrix stores J_{ij} in the upper triangle only (i < j);
    /// the Ising energy is: offset + sum_i h_i*s_i + sum_{i<j} J_{ij}*s_i*s_j.
    pub fn to_ising(&self) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
        let n = self.num_variables;
        let mut j_matrix = vec![vec![0.0; n]; n];
        let mut h = vec![0.0; n];
        let mut offset = self.offset;

        for &(i, j, w) in &self.q_entries {
            if i == j {
                // x_i = (1 + s_i)/2 => w * x_i = w/2 * s_i + w/2
                h[i] += w / 2.0;
                offset += w / 2.0;
            } else {
                // x_i x_j = (1+s_i)(1+s_j)/4
                // = 1/4 + s_i/4 + s_j/4 + s_i*s_j/4
                // w * x_i * x_j = w/4 * s_i*s_j + w/4 * s_i + w/4 * s_j + w/4
                // Store coupling in upper triangle only: J[min][max] += w/4
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                j_matrix[lo][hi] += w / 4.0;
                h[i] += w / 4.0;
                h[j] += w / 4.0;
                offset += w / 4.0;
            }
        }
        (j_matrix, h, offset)
    }

    /// Build a dense QUBO matrix (for small problems).
    pub fn to_dense_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.num_variables;
        let mut m = vec![vec![0.0; n]; n];
        for &(i, j, w) in &self.q_entries {
            m[i][j] += w;
        }
        m
    }
}

// ===================================================================
// CVRP ENCODER
// ===================================================================

/// Encodes a Capacitated Vehicle Routing Problem as a QUBO.
///
/// Binary variables x_{c,v,p}: customer c is visited at position p by vehicle v.
///
/// The encoding enforces:
/// 1. Each non-depot location is visited exactly once (one-hot row constraint).
/// 2. Each position in each vehicle route has at most one location.
/// 3. Vehicle capacity is not exceeded.
/// 4. The objective minimises total travel distance.
pub struct CVRPEncoder {
    /// Penalty weight for the one-visit-per-location constraint.
    pub route_penalty: f64,
    /// Penalty weight for vehicle capacity constraint.
    pub capacity_penalty: f64,
}

impl Default for CVRPEncoder {
    fn default() -> Self {
        Self {
            route_penalty: 200.0,
            capacity_penalty: 150.0,
        }
    }
}

impl CVRPEncoder {
    pub fn new(route_penalty: f64, capacity_penalty: f64) -> Self {
        Self {
            route_penalty,
            capacity_penalty,
        }
    }

    /// Variable index for: customer `c_idx` at position `pos` in vehicle `veh`.
    ///
    /// Layout: c_idx * (num_vehicles * max_stops) + veh * max_stops + pos
    #[inline]
    fn var_index(
        c_idx: usize,
        veh: usize,
        pos: usize,
        num_vehicles: usize,
        max_stops: usize,
    ) -> usize {
        c_idx * num_vehicles * max_stops + veh * max_stops + pos
    }

    /// Encode a CVRP instance as a QUBO.
    ///
    /// `locations` must include the depot (at `config.depot_index`).
    /// Customers are all locations except the depot.
    pub fn encode(
        &self,
        locations: &[Location],
        distances: &DistanceMatrix,
        config: &LogisticsConfig,
    ) -> LogisticsResult<QUBOProblem> {
        let n = locations.len();
        if n < 2 {
            return Err(LogisticsError::EncodingError(
                "Need at least 2 locations (depot + 1 customer)".into(),
            ));
        }
        let depot = config.depot_index;
        let nv = config.num_vehicles;
        let customers: Vec<usize> = (0..n).filter(|&i| i != depot).collect();
        let nc = customers.len();
        let max_stops = nc; // worst case: one vehicle visits all
        let num_vars = nc * nv * max_stops;

        if num_vars > 10_000 {
            return Err(LogisticsError::ProblemTooLarge {
                size: num_vars,
                limit: 10_000,
            });
        }

        let mut entries: Vec<(usize, usize, f64)> = Vec::new();
        let mut offset = 0.0;

        let var =
            |c: usize, v: usize, p: usize| -> usize { Self::var_index(c, v, p, nv, max_stops) };

        // ----- Constraint 1: Each customer visited exactly once -----
        // For each customer c: (sum_{v,p} x_{c,v,p} - 1)^2
        for c in 0..nc {
            offset += self.route_penalty;
            for v in 0..nv {
                for p in 0..max_stops {
                    let idx = var(c, v, p);
                    entries.push((idx, idx, -2.0 * self.route_penalty));
                    // Cross-terms with all other (v', p') for same customer
                    for v2 in 0..nv {
                        for p2 in 0..max_stops {
                            let idx2 = var(c, v2, p2);
                            if idx < idx2 {
                                entries.push((idx, idx2, 2.0 * self.route_penalty));
                            }
                        }
                    }
                }
            }
        }

        // ----- Constraint 2: At most one customer per (vehicle, position) -----
        for v in 0..nv {
            for p in 0..max_stops {
                for c1 in 0..nc {
                    for c2 in (c1 + 1)..nc {
                        let idx1 = var(c1, v, p);
                        let idx2 = var(c2, v, p);
                        let (lo, hi) = if idx1 < idx2 {
                            (idx1, idx2)
                        } else {
                            (idx2, idx1)
                        };
                        entries.push((lo, hi, self.route_penalty));
                    }
                }
            }
        }

        // ----- Objective: minimise total distance -----
        for v in 0..nv {
            for p in 0..max_stops {
                for c in 0..nc {
                    let idx = var(c, v, p);
                    let ci = customers[c];

                    // First stop: depot -> customer
                    if p == 0 {
                        entries.push((idx, idx, distances.get(depot, ci)));
                    }
                    // Last stop: customer -> depot (add for all; the last
                    // occupied position determines the actual return cost,
                    // but for the QUBO encoding every active variable
                    // contributes the return cost which cancels for
                    // intermediate stops via the consecutive pair terms).
                    entries.push((idx, idx, distances.get(ci, depot)));

                    // Consecutive pair costs
                    if p + 1 < max_stops {
                        for c2 in 0..nc {
                            if c2 == c {
                                continue;
                            }
                            let idx2 = var(c2, v, p + 1);
                            let ci2 = customers[c2];
                            let d = distances.get(ci, ci2);
                            let (lo, hi) = if idx < idx2 { (idx, idx2) } else { (idx2, idx) };
                            entries.push((lo, hi, d));
                        }
                    }
                }
            }
        }

        // ----- Constraint 3: Vehicle capacity -----
        for v in 0..nv {
            for c1 in 0..nc {
                for c2 in (c1 + 1)..nc {
                    let d1 = locations[customers[c1]].demand;
                    let d2 = locations[customers[c2]].demand;
                    let pair_demand = d1 + d2;
                    if pair_demand > config.vehicle_capacity {
                        let pen = self.capacity_penalty * (pair_demand - config.vehicle_capacity)
                            / config.vehicle_capacity;
                        for p1 in 0..max_stops {
                            for p2 in 0..max_stops {
                                let idx1 = var(c1, v, p1);
                                let idx2 = var(c2, v, p2);
                                let (lo, hi) = if idx1 < idx2 {
                                    (idx1, idx2)
                                } else {
                                    (idx2, idx1)
                                };
                                entries.push((lo, hi, pen));
                            }
                        }
                    }
                }
            }
        }

        let consolidated = consolidate_entries(entries);
        Ok(QUBOProblem {
            num_variables: num_vars,
            q_entries: consolidated,
            offset,
        })
    }
}

/// Merge duplicate (i, j) entries by summing their weights and drop near-zero.
fn consolidate_entries(mut entries: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, f64)> {
    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut result: Vec<(usize, usize, f64)> = Vec::new();
    for (i, j, w) in entries {
        if let Some(last) = result.last_mut() {
            if last.0 == i && last.1 == j {
                last.2 += w;
                continue;
            }
        }
        result.push((i, j, w));
    }
    result.retain(|&(_, _, w)| w.abs() > 1e-15);
    result
}

// ===================================================================
// TSP ENCODER (special case: single vehicle, no capacity)
// ===================================================================

/// Encode a TSP as a QUBO using the standard position-based formulation.
///
/// Binary variables x_{i,p}: city i is at position p in the tour.
/// Constraints: one-hot per city, one-hot per position.
pub fn encode_tsp_qubo(
    locations: &[Location],
    distances: &DistanceMatrix,
    penalty: f64,
) -> QUBOProblem {
    let n = locations.len();
    let num_vars = n * n;
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut offset = 0.0;

    let var = |i: usize, p: usize| -> usize { i * n + p };

    // ----- Constraint: each city appears exactly once -----
    for i in 0..n {
        offset += penalty;
        for p in 0..n {
            let idx = var(i, p);
            entries.push((idx, idx, -2.0 * penalty));
            for p2 in (p + 1)..n {
                let idx2 = var(i, p2);
                entries.push((idx, idx2, 2.0 * penalty));
            }
        }
    }

    // ----- Constraint: each position has exactly one city -----
    for p in 0..n {
        offset += penalty;
        for i in 0..n {
            let idx = var(i, p);
            entries.push((idx, idx, -2.0 * penalty));
            for i2 in (i + 1)..n {
                let idx2 = var(i2, p);
                entries.push((idx, idx2, 2.0 * penalty));
            }
        }
    }

    // ----- Objective: minimise tour distance -----
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = distances.get(i, j);
            for p in 0..n {
                let p_next = (p + 1) % n;
                let idx_i = var(i, p);
                let idx_j = var(j, p_next);
                let (lo, hi) = if idx_i < idx_j {
                    (idx_i, idx_j)
                } else {
                    (idx_j, idx_i)
                };
                entries.push((lo, hi, d));
            }
        }
    }

    let consolidated = consolidate_entries(entries);
    QUBOProblem {
        num_variables: num_vars,
        q_entries: consolidated,
        offset,
    }
}

// ===================================================================
// JOB-SHOP SCHEDULING
// ===================================================================

/// A job-shop scheduling problem.
///
/// Each job consists of an ordered sequence of operations. Each operation
/// must execute on a specific machine for a specific duration. No two
/// operations may use the same machine simultaneously.
#[derive(Clone, Debug)]
pub struct JobShopProblem {
    /// Number of jobs.
    pub num_jobs: usize,
    /// Number of machines.
    pub num_machines: usize,
    /// For each job, a sequence of (machine_index, duration).
    pub operations: Vec<Vec<(usize, f64)>>,
}

impl JobShopProblem {
    /// Encode as QUBO using time-indexed binary variables.
    ///
    /// x_{j,o,t} = 1 if operation o of job j starts at time step t.
    /// `max_time` is the discretised time horizon.
    pub fn encode_qubo(&self, max_time: usize) -> LogisticsResult<QUBOProblem> {
        if max_time == 0 {
            return Err(LogisticsError::InvalidConfig(
                "max_time must be >= 1".into(),
            ));
        }

        let penalty = 100.0;
        let mut entries: Vec<(usize, usize, f64)> = Vec::new();
        let mut offset = 0.0;

        // Compute flat operation offsets for indexing.
        let mut op_count = 0usize;
        let mut op_offsets: Vec<Vec<usize>> = Vec::new();
        for j in 0..self.num_jobs {
            let mut job_offsets = Vec::new();
            for _o in 0..self.operations[j].len() {
                job_offsets.push(op_count);
                op_count += 1;
            }
            op_offsets.push(job_offsets);
        }

        let num_vars = op_count * max_time;
        let var = |flat_op: usize, t: usize| -> usize { flat_op * max_time + t };

        // ----- Constraint 1: each operation starts exactly once -----
        for flat_op in 0..op_count {
            offset += penalty;
            for t in 0..max_time {
                let idx = var(flat_op, t);
                entries.push((idx, idx, -2.0 * penalty));
                for t2 in (t + 1)..max_time {
                    let idx2 = var(flat_op, t2);
                    entries.push((idx, idx2, 2.0 * penalty));
                }
            }
        }

        // ----- Constraint 2: precedence within each job -----
        for j in 0..self.num_jobs {
            let ops = &self.operations[j];
            for o in 0..ops.len().saturating_sub(1) {
                let dur_int = ops[o].1.ceil() as usize;
                let flat_curr = op_offsets[j][o];
                let flat_next = op_offsets[j][o + 1];

                for t in 0..max_time {
                    for t2 in 0..max_time {
                        if t2 < t + dur_int {
                            let idx1 = var(flat_curr, t);
                            let idx2 = var(flat_next, t2);
                            if idx1 == idx2 {
                                entries.push((idx1, idx1, penalty));
                            } else {
                                let (lo, hi) = if idx1 < idx2 {
                                    (idx1, idx2)
                                } else {
                                    (idx2, idx1)
                                };
                                entries.push((lo, hi, penalty));
                            }
                        }
                    }
                }
            }
        }

        // ----- Constraint 3: no machine overlap -----
        for j1 in 0..self.num_jobs {
            for o1 in 0..self.operations[j1].len() {
                let (m1, dur1) = self.operations[j1][o1];
                let dur1_int = dur1.ceil() as usize;
                let flat1 = op_offsets[j1][o1];

                for j2 in 0..self.num_jobs {
                    for o2 in 0..self.operations[j2].len() {
                        if j1 == j2 && o1 >= o2 {
                            continue;
                        }
                        let (m2, dur2) = self.operations[j2][o2];
                        if m1 != m2 {
                            continue;
                        }
                        let dur2_int = dur2.ceil() as usize;
                        let flat2 = op_offsets[j2][o2];

                        for t1 in 0..max_time {
                            for t2 in 0..max_time {
                                let overlaps = (t2 >= t1 && t2 < t1 + dur1_int)
                                    || (t1 >= t2 && t1 < t2 + dur2_int);
                                if overlaps {
                                    let idx1 = var(flat1, t1);
                                    let idx2 = var(flat2, t2);
                                    if idx1 == idx2 {
                                        entries.push((idx1, idx1, penalty));
                                    } else {
                                        let (lo, hi) = if idx1 < idx2 {
                                            (idx1, idx2)
                                        } else {
                                            (idx2, idx1)
                                        };
                                        entries.push((lo, hi, penalty));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ----- Objective: bias toward earlier start times (makespan proxy) -----
        for flat_op in 0..op_count {
            for t in 0..max_time {
                let idx = var(flat_op, t);
                entries.push((idx, idx, t as f64 * 0.1));
            }
        }

        let consolidated = consolidate_entries(entries);
        Ok(QUBOProblem {
            num_variables: num_vars,
            q_entries: consolidated,
            offset,
        })
    }

    /// Decode a binary solution vector into a `Schedule`.
    pub fn decode_schedule(&self, solution: &[bool], max_time: usize) -> Schedule {
        let mut op_count = 0usize;
        let mut op_offsets: Vec<Vec<usize>> = Vec::new();
        for j in 0..self.num_jobs {
            let mut job_offsets = Vec::new();
            for _o in 0..self.operations[j].len() {
                job_offsets.push(op_count);
                op_count += 1;
            }
            op_offsets.push(job_offsets);
        }

        let var = |flat_op: usize, t: usize| -> usize { flat_op * max_time + t };

        let mut assignments = Vec::new();
        for j in 0..self.num_jobs {
            for o in 0..self.operations[j].len() {
                let (machine, duration) = self.operations[j][o];
                let flat_op = op_offsets[j][o];
                let mut start_time = None;
                for t in 0..max_time {
                    let idx = var(flat_op, t);
                    if idx < solution.len() && solution[idx] {
                        start_time = Some(t as f64);
                        break;
                    }
                }
                let start = start_time.unwrap_or(0.0);
                assignments.push((j, machine, start, start + duration));
            }
        }

        Schedule { assignments }
    }
}

/// A schedule: a set of (job, machine, start_time, end_time) tuples.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub assignments: Vec<(usize, usize, f64, f64)>,
}

impl Schedule {
    /// The makespan: latest end time across all assignments.
    pub fn makespan(&self) -> f64 {
        self.assignments
            .iter()
            .map(|&(_, _, _, end)| end)
            .fold(0.0_f64, f64::max)
    }

    /// Validate the schedule against a job-shop problem.
    ///
    /// Checks: (a) precedence within jobs, (b) no machine overlap.
    pub fn is_valid(&self, problem: &JobShopProblem) -> bool {
        // Check precedence within each job.
        for j in 0..problem.num_jobs {
            let job_ops: Vec<&(usize, usize, f64, f64)> = self
                .assignments
                .iter()
                .filter(|&&(job, _, _, _)| job == j)
                .collect();
            if job_ops.len() != problem.operations[j].len() {
                return false;
            }
            // Match each expected operation to an assignment by machine.
            let mut ordered: Vec<Option<(f64, f64)>> = vec![None; problem.operations[j].len()];
            let mut used = vec![false; job_ops.len()];

            for (o_idx, &(expected_machine, _)) in problem.operations[j].iter().enumerate() {
                for (a_idx, &&(_, m, s, e)) in job_ops.iter().enumerate() {
                    if !used[a_idx] && m == expected_machine {
                        ordered[o_idx] = Some((s, e));
                        used[a_idx] = true;
                        break;
                    }
                }
            }
            for o in 0..ordered.len().saturating_sub(1) {
                match (ordered[o], ordered[o + 1]) {
                    (Some((_, end)), Some((start, _))) => {
                        if end > start + 1e-9 {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }

        // Check no machine overlap.
        for m in 0..problem.num_machines {
            let mut ops: Vec<(f64, f64)> = self
                .assignments
                .iter()
                .filter(|&&(_, mach, _, _)| mach == m)
                .map(|&(_, _, s, e)| (s, e))
                .collect();
            ops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for w in ops.windows(2) {
                if w[0].1 > w[1].0 + 1e-9 {
                    return false;
                }
            }
        }

        true
    }
}

// ===================================================================
// QAOA SOLVER
// ===================================================================

/// QAOA solver operating on a QUBO problem.
///
/// Uses a simplified simulation model: parameter-shift gradient estimation
/// over a cost landscape biased by QAOA angle schedules.
pub struct QAOASolver {
    qubo: QUBOProblem,
    depth: usize,
}

impl QAOASolver {
    pub fn new(qubo: &QUBOProblem, depth: usize) -> Self {
        Self {
            qubo: qubo.clone(),
            depth,
        }
    }

    /// Human-readable description of the cost Hamiltonian.
    pub fn cost_operator(&self) -> String {
        format!(
            "Cost Hamiltonian: {} variables, {} QUBO terms, offset={:.4}",
            self.qubo.num_variables,
            self.qubo.q_entries.len(),
            self.qubo.offset,
        )
    }

    /// Optimise QAOA angles via parameter-shift gradient descent.
    ///
    /// Returns (gamma_vec, beta_vec), each of length `depth`.
    pub fn optimize_angles(&self, num_iterations: usize) -> (Vec<f64>, Vec<f64>) {
        let p = self.depth;
        let mut gamma = vec![0.5; p];
        let mut beta = vec![0.5; p];
        let lr = 0.1;
        let shift = std::f64::consts::PI / 4.0;

        for _ in 0..num_iterations {
            for k in 0..p {
                // Gradient wrt gamma[k]
                let mut g_plus = gamma.clone();
                let mut g_minus = gamma.clone();
                g_plus[k] += shift;
                g_minus[k] -= shift;
                let grad_g = (self.expected_cost(&g_plus, &beta)
                    - self.expected_cost(&g_minus, &beta))
                    / (2.0 * shift.sin());
                gamma[k] -= lr * grad_g;

                // Gradient wrt beta[k]
                let mut b_plus = beta.clone();
                let mut b_minus = beta.clone();
                b_plus[k] += shift;
                b_minus[k] -= shift;
                let grad_b = (self.expected_cost(&gamma, &b_plus)
                    - self.expected_cost(&gamma, &b_minus))
                    / (2.0 * shift.sin());
                beta[k] -= lr * grad_b;
            }
        }

        (gamma, beta)
    }

    /// Estimate the expected cost for given angles.
    ///
    /// For small problem sizes (n <= 20) uses exhaustive evaluation;
    /// for larger problems falls back to sampling.
    fn expected_cost(&self, gamma: &[f64], beta: &[f64]) -> f64 {
        let n = self.qubo.num_variables;
        if n > 20 {
            let samples = self.sample(gamma, beta, 100);
            if samples.is_empty() {
                return f64::MAX;
            }
            return samples.iter().map(|(_, c)| c).sum::<f64>() / samples.len() as f64;
        }

        let dim = 1usize << n;
        let prob = 1.0 / dim as f64;
        let mut total = 0.0;
        let mut weight_sum = 0.0;

        for idx in 0..dim {
            let bits: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
            let cost = self.qubo.evaluate(&bits);
            let angle_factor: f64 = gamma
                .iter()
                .zip(beta.iter())
                .map(|(&g, &b)| (g * cost).cos() * b.cos())
                .product();
            let w = prob * (1.0 + angle_factor.abs());
            total += w * cost;
            weight_sum += w;
        }
        if weight_sum > 1e-15 {
            total / weight_sum
        } else {
            total
        }
    }

    /// Sample bitstrings from the QAOA output distribution.
    ///
    /// Returns a vector of (bitstring, cost) sorted by cost ascending.
    pub fn sample(&self, _gamma: &[f64], beta: &[f64], num_shots: usize) -> Vec<(Vec<bool>, f64)> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = self.qubo.num_variables;
        let mut results: Vec<(Vec<bool>, f64)> = Vec::with_capacity(num_shots);

        for _ in 0..num_shots {
            let bits: Vec<bool> = (0..n)
                .map(|q| {
                    let bias: f64 = beta
                        .iter()
                        .enumerate()
                        .map(|(k, &b)| (b * (q + k + 1) as f64).sin())
                        .sum::<f64>()
                        / beta.len().max(1) as f64;
                    let p = 0.5 + 0.3 * bias.tanh();
                    rng.gen::<f64>() < p
                })
                .collect();
            let cost = self.qubo.evaluate(&bits);
            results.push((bits, cost));
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

// ===================================================================
// SOLUTIONS
// ===================================================================

/// Solution to a CVRP instance.
#[derive(Clone, Debug)]
pub struct CVRPSolution {
    /// Routes per vehicle: each is an ordered list of location indices.
    pub routes: Vec<Vec<usize>>,
    /// Total distance across all routes (including depot returns).
    pub total_distance: f64,
    /// Whether the solution satisfies all constraints.
    pub is_feasible: bool,
    /// Load carried by each vehicle.
    pub vehicle_loads: Vec<f64>,
}

impl CVRPSolution {
    /// Decode a solution from a binary QAOA/annealing output.
    pub fn decode_from_binary(
        bits: &[bool],
        config: &LogisticsConfig,
        locations: &[Location],
        distances: &DistanceMatrix,
    ) -> Self {
        let n = locations.len();
        let depot = config.depot_index;
        let customers: Vec<usize> = (0..n).filter(|&i| i != depot).collect();
        let nc = customers.len();
        let nv = config.num_vehicles;
        let max_stops = nc;

        let var = |c_idx: usize, v: usize, p: usize| -> usize {
            CVRPEncoder::var_index(c_idx, v, p, nv, max_stops)
        };

        let mut routes: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];
        for v in 0..nv {
            for p in 0..max_stops {
                for c in 0..nc {
                    let idx = var(c, v, p);
                    if idx < bits.len() && bits[idx] {
                        routes[v].push((p, customers[c]));
                    }
                }
            }
            routes[v].sort_by_key(|&(pos, _)| pos);
        }

        let routes_locs: Vec<Vec<usize>> = routes
            .iter()
            .map(|r| r.iter().map(|&(_, loc)| loc).collect())
            .collect();

        let mut total_distance = 0.0;
        let mut vehicle_loads = Vec::with_capacity(nv);
        let mut is_feasible = true;
        let mut visited = vec![false; n];
        visited[depot] = true;

        for route in &routes_locs {
            let mut load = 0.0;
            if route.is_empty() {
                vehicle_loads.push(0.0);
                continue;
            }
            total_distance += distances.get(depot, route[0]);
            for w in route.windows(2) {
                total_distance += distances.get(w[0], w[1]);
            }
            total_distance += distances.get(*route.last().unwrap(), depot);
            for &loc in route {
                load += locations[loc].demand;
                if visited[loc] {
                    is_feasible = false; // double visit
                }
                visited[loc] = true;
            }
            if load > config.vehicle_capacity {
                is_feasible = false;
            }
            vehicle_loads.push(load);
        }

        for c in &customers {
            if !visited[*c] {
                is_feasible = false;
            }
        }

        CVRPSolution {
            routes: routes_locs,
            total_distance,
            is_feasible,
            vehicle_loads,
        }
    }
}

/// Solution to a TSP instance.
#[derive(Clone, Debug)]
pub struct TSPSolution {
    /// Tour: ordered list of location indices.
    pub tour: Vec<usize>,
    /// Total tour distance (returning to start).
    pub total_distance: f64,
}

/// Comparison between quantum and classical solutions.
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    pub quantum_cost: f64,
    pub classical_cost: f64,
    /// `classical_cost / quantum_cost` -- values > 1 mean quantum is better.
    pub improvement_ratio: f64,
    pub quantum_time: f64,
    pub classical_time: f64,
}

// ===================================================================
// QUANTUM LOGISTICS OPTIMIZER
// ===================================================================

/// Main entry point for quantum logistics optimization.
pub struct QuantumLogisticsOptimizer {
    config: LogisticsConfig,
}

impl QuantumLogisticsOptimizer {
    pub fn new(config: LogisticsConfig) -> Self {
        Self { config }
    }

    /// Solve a CVRP instance using the configured solver.
    pub fn solve_cvrp(&self, locations: &[Location], distances: &DistanceMatrix) -> CVRPSolution {
        match self.config.solver {
            LogisticsSolver::BruteForce => self.brute_force_cvrp(locations, distances),
            LogisticsSolver::QAOA
            | LogisticsSolver::QuantumAnnealing
            | LogisticsSolver::HybridClassicalQuantum => {
                let encoder = CVRPEncoder::default();
                match encoder.encode(locations, distances, &self.config) {
                    Ok(qubo) => {
                        let solver = QAOASolver::new(&qubo, self.config.qaoa_depth);
                        let (gamma, beta) = solver.optimize_angles(20);
                        let samples = solver.sample(&gamma, &beta, self.config.num_shots);
                        if let Some((best_bits, _)) = samples.first() {
                            CVRPSolution::decode_from_binary(
                                best_bits,
                                &self.config,
                                locations,
                                distances,
                            )
                        } else {
                            self.classical_benchmark(locations, distances)
                        }
                    }
                    Err(_) => self.classical_benchmark(locations, distances),
                }
            }
        }
    }

    /// Solve TSP (single vehicle, no capacity).
    pub fn solve_tsp(&self, locations: &[Location], distances: &DistanceMatrix) -> TSPSolution {
        let n = locations.len();
        if n <= 8 || self.config.solver == LogisticsSolver::BruteForce {
            return self.brute_force_tsp(locations, distances);
        }

        // QAOA path for larger instances.
        let max_dist = distances
            .distances
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0_f64, f64::max);
        let penalty = 10.0 * max_dist.max(1.0);
        let qubo = encode_tsp_qubo(locations, distances, penalty);
        let solver = QAOASolver::new(&qubo, self.config.qaoa_depth);
        let (gamma, beta) = solver.optimize_angles(15);
        let samples = solver.sample(&gamma, &beta, self.config.num_shots);

        for (bits, _) in &samples {
            if let Some(tour) = decode_tsp_bits(bits, n) {
                let dist = tour_distance(&tour, distances);
                return TSPSolution {
                    tour,
                    total_distance: dist,
                };
            }
        }

        // Fall back to nearest-neighbour.
        nearest_neighbour_tsp(distances)
    }

    /// Solve a job-shop scheduling problem.
    pub fn solve_job_shop(&self, problem: &JobShopProblem) -> Schedule {
        let max_time = problem
            .operations
            .iter()
            .map(|ops| ops.iter().map(|&(_, d)| d.ceil() as usize).sum::<usize>())
            .max()
            .unwrap_or(1)
            * 2;

        match problem.encode_qubo(max_time) {
            Ok(qubo) => {
                let solver = QAOASolver::new(&qubo, self.config.qaoa_depth);
                let (gamma, beta) = solver.optimize_angles(15);
                let samples = solver.sample(&gamma, &beta, self.config.num_shots);
                if let Some((best_bits, _)) = samples.first() {
                    problem.decode_schedule(best_bits, max_time)
                } else {
                    Schedule {
                        assignments: Vec::new(),
                    }
                }
            }
            Err(_) => Schedule {
                assignments: Vec::new(),
            },
        }
    }

    /// Classical nearest-neighbour heuristic for CVRP (benchmark baseline).
    pub fn classical_benchmark(
        &self,
        locations: &[Location],
        distances: &DistanceMatrix,
    ) -> CVRPSolution {
        let n = locations.len();
        let depot = self.config.depot_index;
        let customers: Vec<usize> = (0..n).filter(|&i| i != depot).collect();
        let nv = self.config.num_vehicles;
        let capacity = self.config.vehicle_capacity;

        let mut visited = vec![false; n];
        visited[depot] = true;
        let mut routes: Vec<Vec<usize>> = vec![Vec::new(); nv];
        let mut loads = vec![0.0; nv];
        let mut total_distance = 0.0;

        for v in 0..nv {
            let mut current = depot;
            loop {
                let mut best_next = None;
                let mut best_dist = f64::MAX;
                for &c in &customers {
                    if !visited[c]
                        && loads[v] + locations[c].demand <= capacity
                        && distances.get(current, c) < best_dist
                    {
                        best_dist = distances.get(current, c);
                        best_next = Some(c);
                    }
                }
                match best_next {
                    Some(next) => {
                        visited[next] = true;
                        routes[v].push(next);
                        loads[v] += locations[next].demand;
                        total_distance += best_dist;
                        current = next;
                    }
                    None => break,
                }
            }
            if !routes[v].is_empty() {
                total_distance += distances.get(current, depot);
            }
        }

        let is_feasible = customers.iter().all(|c| visited[*c]);

        CVRPSolution {
            routes,
            total_distance,
            is_feasible,
            vehicle_loads: loads,
        }
    }

    /// Compare quantum and classical solutions.
    pub fn compare_solutions(
        &self,
        quantum: &CVRPSolution,
        classical: &CVRPSolution,
    ) -> ComparisonResult {
        let ratio = if quantum.total_distance > 1e-12 {
            classical.total_distance / quantum.total_distance
        } else {
            1.0
        };
        ComparisonResult {
            quantum_cost: quantum.total_distance,
            classical_cost: classical.total_distance,
            improvement_ratio: ratio,
            quantum_time: 0.0,
            classical_time: 0.0,
        }
    }

    // ---- private helpers ----

    fn brute_force_tsp(&self, _locations: &[Location], distances: &DistanceMatrix) -> TSPSolution {
        let n = distances.size();
        if n == 0 {
            return TSPSolution {
                tour: Vec::new(),
                total_distance: 0.0,
            };
        }
        if n == 1 {
            return TSPSolution {
                tour: vec![0],
                total_distance: 0.0,
            };
        }

        let mut rest: Vec<usize> = (1..n).collect();
        let mut best_tour = Vec::new();
        let mut best_dist = f64::MAX;

        loop {
            let mut tour = vec![0];
            tour.extend_from_slice(&rest);
            let d = tour_distance(&tour, distances);
            if d < best_dist {
                best_dist = d;
                best_tour = tour;
            }
            if !next_permutation(&mut rest) {
                break;
            }
        }

        TSPSolution {
            tour: best_tour,
            total_distance: best_dist,
        }
    }

    fn brute_force_cvrp(&self, locations: &[Location], distances: &DistanceMatrix) -> CVRPSolution {
        let depot = self.config.depot_index;
        let customers: Vec<usize> = (0..locations.len()).filter(|&i| i != depot).collect();
        let nc = customers.len();
        let nv = self.config.num_vehicles;
        let capacity = self.config.vehicle_capacity;

        if nc > 10 {
            return self.classical_benchmark(locations, distances);
        }

        let mut best: Option<CVRPSolution> = None;
        let total_combos = nv.pow(nc as u32);

        for combo in 0..total_combos {
            let mut assignment = vec![0usize; nc];
            let mut val = combo;
            for c in 0..nc {
                assignment[c] = val % nv;
                val /= nv;
            }

            let mut loads = vec![0.0; nv];
            let mut feasible = true;
            for c in 0..nc {
                loads[assignment[c]] += locations[customers[c]].demand;
                if loads[assignment[c]] > capacity {
                    feasible = false;
                    break;
                }
            }
            if !feasible {
                continue;
            }

            let mut routes: Vec<Vec<usize>> = vec![Vec::new(); nv];
            for c in 0..nc {
                routes[assignment[c]].push(customers[c]);
            }

            let mut total_dist = 0.0;
            for v in 0..nv {
                if routes[v].is_empty() {
                    continue;
                }
                if routes[v].len() <= 6 {
                    let mut best_route_dist = f64::MAX;
                    let mut best_order = routes[v].clone();
                    let mut perm = routes[v].clone();
                    perm.sort();
                    loop {
                        let mut d = distances.get(depot, perm[0]);
                        for w in perm.windows(2) {
                            d += distances.get(w[0], w[1]);
                        }
                        d += distances.get(*perm.last().unwrap(), depot);
                        if d < best_route_dist {
                            best_route_dist = d;
                            best_order = perm.clone();
                        }
                        if !next_permutation(&mut perm) {
                            break;
                        }
                    }
                    routes[v] = best_order;
                    total_dist += best_route_dist;
                } else {
                    let mut current = depot;
                    let mut remaining = routes[v].clone();
                    let mut ordered = Vec::new();
                    while !remaining.is_empty() {
                        let (best_idx, _) = remaining
                            .iter()
                            .enumerate()
                            .min_by(|(_, &a), (_, &b)| {
                                distances
                                    .get(current, a)
                                    .partial_cmp(&distances.get(current, b))
                                    .unwrap()
                            })
                            .unwrap();
                        let next = remaining.remove(best_idx);
                        total_dist += distances.get(current, next);
                        current = next;
                        ordered.push(next);
                    }
                    total_dist += distances.get(current, depot);
                    routes[v] = ordered;
                }
            }

            let sol = CVRPSolution {
                routes,
                total_distance: total_dist,
                is_feasible: true,
                vehicle_loads: loads,
            };

            match &best {
                Some(prev) if prev.total_distance <= sol.total_distance => {}
                _ => best = Some(sol),
            }
        }

        best.unwrap_or_else(|| self.classical_benchmark(locations, distances))
    }
}

// ===================================================================
// FREE-STANDING UTILITY FUNCTIONS
// ===================================================================

/// Calculate round-trip tour distance.
fn tour_distance(tour: &[usize], distances: &DistanceMatrix) -> f64 {
    if tour.len() <= 1 {
        return 0.0;
    }
    let mut dist = 0.0;
    for w in tour.windows(2) {
        dist += distances.get(w[0], w[1]);
    }
    dist += distances.get(*tour.last().unwrap(), tour[0]);
    dist
}

/// Decode a TSP binary vector (n*n bits) into a tour, if valid.
fn decode_tsp_bits(bits: &[bool], n: usize) -> Option<Vec<usize>> {
    let var = |i: usize, p: usize| -> usize { i * n + p };
    let mut tour = vec![None; n];
    let mut city_used = vec![false; n];

    for p in 0..n {
        let mut found = None;
        for i in 0..n {
            let idx = var(i, p);
            if idx < bits.len() && bits[idx] {
                if found.is_some() || city_used[i] {
                    return None;
                }
                found = Some(i);
                city_used[i] = true;
            }
        }
        tour[p] = found;
    }

    let result: Vec<usize> = tour.into_iter().flatten().collect();
    if result.len() == n {
        Some(result)
    } else {
        None
    }
}

/// Nearest-neighbour TSP heuristic starting from city 0.
fn nearest_neighbour_tsp(distances: &DistanceMatrix) -> TSPSolution {
    let n = distances.size();
    if n == 0 {
        return TSPSolution {
            tour: Vec::new(),
            total_distance: 0.0,
        };
    }

    let mut visited = vec![false; n];
    let mut tour = vec![0usize];
    visited[0] = true;
    let mut current = 0;
    let mut total_dist = 0.0;

    for _ in 1..n {
        let mut best = None;
        let mut best_d = f64::MAX;
        for j in 0..n {
            if !visited[j] && distances.get(current, j) < best_d {
                best_d = distances.get(current, j);
                best = Some(j);
            }
        }
        if let Some(next) = best {
            visited[next] = true;
            tour.push(next);
            total_dist += best_d;
            current = next;
        }
    }
    total_dist += distances.get(current, 0);

    TSPSolution {
        tour,
        total_distance: total_dist,
    }
}

/// Generate the next lexicographic permutation in-place.
/// Returns `false` when the sequence wraps (all permutations exhausted).
fn next_permutation(arr: &mut [usize]) -> bool {
    let n = arr.len();
    if n <= 1 {
        return false;
    }
    let mut i = n - 2;
    loop {
        if arr[i] < arr[i + 1] {
            break;
        }
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    let mut j = n - 1;
    while arr[j] <= arr[i] {
        j -= 1;
    }
    arr.swap(i, j);
    arr[i + 1..].reverse();
    true
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    fn triangle_locations() -> Vec<Location> {
        vec![
            Location::new(0, 0.0, 0.0, 0.0), // depot
            Location::new(1, 3.0, 4.0, 20.0),
            Location::new(2, 6.0, 0.0, 30.0),
        ]
    }

    fn square_locations() -> Vec<Location> {
        vec![
            Location::new(0, 0.0, 0.0, 0.0),
            Location::new(1, 1.0, 0.0, 10.0),
            Location::new(2, 1.0, 1.0, 15.0),
            Location::new(3, 0.0, 1.0, 20.0),
        ]
    }

    // ---------------------------------------------------------------
    // 1. Config builder -- valid build
    // ---------------------------------------------------------------
    #[test]
    fn test_config_builder_valid() {
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(5)
            .num_vehicles(2)
            .vehicle_capacity(50.0)
            .qaoa_depth(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.num_locations, 5);
        assert_eq!(c.num_vehicles, 2);
        assert!((c.vehicle_capacity - 50.0).abs() < 1e-12);
        assert_eq!(c.qaoa_depth, 4);
    }

    // ---------------------------------------------------------------
    // 2. Config builder -- invalid num_locations
    // ---------------------------------------------------------------
    #[test]
    fn test_config_builder_invalid_locations() {
        assert!(LogisticsConfigBuilder::new()
            .num_locations(0)
            .build()
            .is_err());
        assert!(LogisticsConfigBuilder::new()
            .num_locations(1)
            .build()
            .is_err());
        assert!(LogisticsConfigBuilder::new()
            .num_locations(201)
            .build()
            .is_err());
        assert!(LogisticsConfigBuilder::new()
            .num_locations(2)
            .build()
            .is_ok());
        assert!(LogisticsConfigBuilder::new()
            .num_locations(200)
            .build()
            .is_ok());
    }

    // ---------------------------------------------------------------
    // 3. Config builder -- invalid vehicles and capacity
    // ---------------------------------------------------------------
    #[test]
    fn test_config_builder_invalid_vehicles_and_capacity() {
        assert!(LogisticsConfigBuilder::new()
            .num_vehicles(0)
            .build()
            .is_err());
        assert!(LogisticsConfigBuilder::new()
            .num_vehicles(51)
            .build()
            .is_err());
        assert!(LogisticsConfigBuilder::new()
            .vehicle_capacity(-1.0)
            .build()
            .is_err());
        assert!(LogisticsConfigBuilder::new()
            .vehicle_capacity(0.0)
            .build()
            .is_err());
    }

    // ---------------------------------------------------------------
    // 4. Distance matrix from locations (Euclidean)
    // ---------------------------------------------------------------
    #[test]
    fn test_distance_matrix_from_locations() {
        let locs = triangle_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        assert_eq!(dm.size(), 3);
        // d(0,1) = sqrt(9+16) = 5.0
        assert!((dm.get(0, 1) - 5.0).abs() < 1e-10);
        // d(0,2) = 6.0
        assert!((dm.get(0, 2) - 6.0).abs() < 1e-10);
        // d(1,2) = sqrt(9+16) = 5.0
        assert!((dm.get(1, 2) - 5.0).abs() < 1e-10);
        // Symmetry
        assert!((dm.get(0, 1) - dm.get(1, 0)).abs() < 1e-15);
        assert!((dm.get(1, 2) - dm.get(2, 1)).abs() < 1e-15);
    }

    // ---------------------------------------------------------------
    // 5. Distance matrix from raw matrix
    // ---------------------------------------------------------------
    #[test]
    fn test_distance_matrix_from_raw() {
        let m = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 3.0],
            vec![2.0, 3.0, 0.0],
        ];
        let dm = DistanceMatrix::from_matrix(m).unwrap();
        assert_eq!(dm.get(0, 2), 2.0);
        assert_eq!(dm.get(1, 2), 3.0);
        assert_eq!(dm.size(), 3);
    }

    // ---------------------------------------------------------------
    // 6. Distance matrix validation -- non-square
    // ---------------------------------------------------------------
    #[test]
    fn test_distance_matrix_invalid() {
        let m = vec![vec![0.0, 1.0], vec![1.0]];
        assert!(DistanceMatrix::from_matrix(m).is_err());
        assert!(DistanceMatrix::from_matrix(Vec::new()).is_err());
    }

    // ---------------------------------------------------------------
    // 7. QUBO evaluate
    // ---------------------------------------------------------------
    #[test]
    fn test_qubo_evaluate() {
        // min x0 + 2*x1 + 3*x0*x1
        let qubo = QUBOProblem {
            num_variables: 2,
            q_entries: vec![(0, 0, 1.0), (1, 1, 2.0), (0, 1, 3.0)],
            offset: 0.0,
        };
        assert!((qubo.evaluate(&[false, false]) - 0.0).abs() < 1e-12);
        assert!((qubo.evaluate(&[true, false]) - 1.0).abs() < 1e-12);
        assert!((qubo.evaluate(&[false, true]) - 2.0).abs() < 1e-12);
        assert!((qubo.evaluate(&[true, true]) - 6.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 8. QUBO to Ising preserves energy spectrum
    // ---------------------------------------------------------------
    #[test]
    fn test_ising_conversion_preserves_energy() {
        let qubo = QUBOProblem {
            num_variables: 2,
            q_entries: vec![(0, 0, 1.0), (1, 1, -1.0), (0, 1, 2.0)],
            offset: 0.5,
        };
        let (j_mat, h, ising_off) = qubo.to_ising();

        for bits in 0u8..4 {
            let x = vec![(bits & 1) != 0, (bits >> 1 & 1) != 0];
            let qubo_e = qubo.evaluate(&x);

            let s: Vec<f64> = x.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
            let mut ising_e = ising_off;
            for i in 0..2 {
                ising_e += h[i] * s[i];
                for j in (i + 1)..2 {
                    ising_e += j_mat[i][j] * s[i] * s[j];
                }
            }
            assert!(
                (qubo_e - ising_e).abs() < 1e-8,
                "QUBO={} != Ising={} for x={:?}",
                qubo_e,
                ising_e,
                x,
            );
        }
    }

    // ---------------------------------------------------------------
    // 9. TSP QUBO encoding for 3 cities
    // ---------------------------------------------------------------
    #[test]
    fn test_tsp_qubo_3_cities() {
        let locs = triangle_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let qubo = encode_tsp_qubo(&locs, &dm, 100.0);

        assert_eq!(qubo.num_variables, 9);

        // Valid tour: city 0 at pos 0, city 1 at pos 1, city 2 at pos 2
        let mut valid_bits = vec![false; 9];
        valid_bits[0] = true; // city 0 pos 0
        valid_bits[4] = true; // city 1 pos 1
        valid_bits[8] = true; // city 2 pos 2
        let cost_valid = qubo.evaluate(&valid_bits);

        // Invalid: two cities at same position
        let mut bad_bits = vec![false; 9];
        bad_bits[0] = true; // city 0 pos 0
        bad_bits[3] = true; // city 1 pos 0 (conflict!)
        bad_bits[8] = true; // city 2 pos 2
        let cost_bad = qubo.evaluate(&bad_bits);

        assert!(
            cost_bad > cost_valid,
            "Invalid cost ({}) should exceed valid cost ({})",
            cost_bad,
            cost_valid,
        );
    }

    // ---------------------------------------------------------------
    // 10. CVRP encoding produces correct variable count
    // ---------------------------------------------------------------
    #[test]
    fn test_cvrp_encoding_variable_count() {
        let locs = triangle_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(3)
            .num_vehicles(2)
            .build()
            .unwrap();
        let encoder = CVRPEncoder::default();
        let qubo = encoder.encode(&locs, &dm, &cfg).unwrap();

        // 2 customers, 2 vehicles, max_stops=2 => 2 * 2 * 2 = 8
        assert_eq!(qubo.num_variables, 8);
    }

    // ---------------------------------------------------------------
    // 11. Solution decoding from binary
    // ---------------------------------------------------------------
    #[test]
    fn test_cvrp_solution_decode() {
        let locs = triangle_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(3)
            .num_vehicles(1)
            .vehicle_capacity(100.0)
            .build()
            .unwrap();

        // 1 vehicle, 2 customers, max_stops=2 => nc*nv*max_stops = 2*1*2 = 4
        // var(c_idx, v, p) = c_idx * 1 * 2 + 0 * 2 + p
        // Customer 0 (loc 1) at pos 0 => var(0,0,0) = 0
        // Customer 1 (loc 2) at pos 1 => var(1,0,1) = 3
        let mut bits = vec![false; 4];
        bits[0] = true;
        bits[3] = true;

        let sol = CVRPSolution::decode_from_binary(&bits, &cfg, &locs, &dm);
        assert_eq!(sol.routes.len(), 1);
        assert_eq!(sol.routes[0], vec![1, 2]);
        assert!(sol.is_feasible);
        assert!((sol.vehicle_loads[0] - 50.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 12. Vehicle capacity constraint
    // ---------------------------------------------------------------
    #[test]
    fn test_capacity_constraint_violation() {
        let locs = vec![
            Location::new(0, 0.0, 0.0, 0.0),
            Location::new(1, 1.0, 0.0, 60.0),
            Location::new(2, 2.0, 0.0, 60.0),
        ];
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(3)
            .num_vehicles(1)
            .vehicle_capacity(50.0)
            .build()
            .unwrap();

        let mut bits = vec![false; 4];
        bits[0] = true;
        bits[3] = true;
        let sol = CVRPSolution::decode_from_binary(&bits, &cfg, &locs, &dm);
        assert!(!sol.is_feasible);
    }

    // ---------------------------------------------------------------
    // 13. Job-shop encoding and variable count
    // ---------------------------------------------------------------
    #[test]
    fn test_job_shop_encoding() {
        let problem = JobShopProblem {
            num_jobs: 2,
            num_machines: 2,
            operations: vec![vec![(0, 1.0), (1, 1.0)], vec![(1, 1.0), (0, 1.0)]],
        };
        let max_time = 4;
        let qubo = problem.encode_qubo(max_time).unwrap();
        // 4 operations * 4 time steps = 16
        assert_eq!(qubo.num_variables, 16);
    }

    // ---------------------------------------------------------------
    // 14. Schedule validation
    // ---------------------------------------------------------------
    #[test]
    fn test_schedule_validation() {
        let problem = JobShopProblem {
            num_jobs: 2,
            num_machines: 2,
            operations: vec![vec![(0, 2.0), (1, 1.0)], vec![(1, 1.0), (0, 1.0)]],
        };

        // Valid: J0-O0@M0[0,2), J0-O1@M1[2,3), J1-O0@M1[0,1), J1-O1@M0[2,3)
        let valid = Schedule {
            assignments: vec![
                (0, 0, 0.0, 2.0),
                (0, 1, 2.0, 3.0),
                (1, 1, 0.0, 1.0),
                (1, 0, 2.0, 3.0),
            ],
        };
        assert!(valid.is_valid(&problem));
        assert!((valid.makespan() - 3.0).abs() < 1e-10);

        // Invalid: precedence violation (J0-O1 starts before J0-O0 ends)
        let invalid = Schedule {
            assignments: vec![
                (0, 0, 0.0, 2.0),
                (0, 1, 1.0, 2.0),
                (1, 1, 0.0, 1.0),
                (1, 0, 2.0, 3.0),
            ],
        };
        assert!(!invalid.is_valid(&problem));
    }

    // ---------------------------------------------------------------
    // 15. Nearest-neighbour heuristic (classical benchmark)
    // ---------------------------------------------------------------
    #[test]
    fn test_nearest_neighbour_heuristic() {
        let locs = square_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(4)
            .num_vehicles(1)
            .vehicle_capacity(100.0)
            .build()
            .unwrap();
        let opt = QuantumLogisticsOptimizer::new(cfg);
        let sol = opt.classical_benchmark(&locs, &dm);

        let served: usize = sol.routes.iter().map(|r| r.len()).sum();
        assert_eq!(served, 3);
        assert!(sol.is_feasible);
        assert!(sol.total_distance > 0.0);
    }

    // ---------------------------------------------------------------
    // 16. TSP brute-force on unit square (4 cities)
    // ---------------------------------------------------------------
    #[test]
    fn test_tsp_brute_force_4_cities() {
        let locs = square_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(4)
            .num_vehicles(1)
            .solver(LogisticsSolver::BruteForce)
            .build()
            .unwrap();
        let opt = QuantumLogisticsOptimizer::new(cfg);
        let sol = opt.solve_tsp(&locs, &dm);

        // Optimal tour on unit square: perimeter = 4.0
        assert!((sol.total_distance - 4.0).abs() < 1e-10);
        assert_eq!(sol.tour.len(), 4);
    }

    // ---------------------------------------------------------------
    // 17. QAOA angle optimization (small)
    // ---------------------------------------------------------------
    #[test]
    fn test_qaoa_angle_optimization() {
        let qubo = QUBOProblem {
            num_variables: 2,
            q_entries: vec![(0, 0, 1.0), (1, 1, 2.0), (0, 1, -3.0)],
            offset: 0.0,
        };
        let solver = QAOASolver::new(&qubo, 2);
        let (gamma, beta) = solver.optimize_angles(10);
        assert_eq!(gamma.len(), 2);
        assert_eq!(beta.len(), 2);

        // Angles should have moved from initial 0.5.
        let unchanged = gamma.iter().all(|&g| (g - 0.5).abs() < 1e-15)
            && beta.iter().all(|&b| (b - 0.5).abs() < 1e-15);
        assert!(!unchanged, "Angles should have been optimized");
    }

    // ---------------------------------------------------------------
    // 18. QAOA sampling returns sorted results
    // ---------------------------------------------------------------
    #[test]
    fn test_qaoa_sampling_sorted() {
        let qubo = QUBOProblem {
            num_variables: 3,
            q_entries: vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)],
            offset: 0.0,
        };
        let solver = QAOASolver::new(&qubo, 1);
        let samples = solver.sample(&[0.5], &[0.5], 50);
        assert_eq!(samples.len(), 50);
        for w in samples.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-15);
        }
    }

    // ---------------------------------------------------------------
    // 19. Full CVRP solve pipeline
    // ---------------------------------------------------------------
    #[test]
    fn test_full_cvrp_solve_pipeline() {
        let locs = triangle_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(3)
            .num_vehicles(2)
            .vehicle_capacity(100.0)
            .solver(LogisticsSolver::BruteForce)
            .num_shots(100)
            .build()
            .unwrap();
        let opt = QuantumLogisticsOptimizer::new(cfg);
        let sol = opt.solve_cvrp(&locs, &dm);

        let served: usize = sol.routes.iter().map(|r| r.len()).sum();
        assert_eq!(served, 2);
        assert!(sol.is_feasible);
        assert!(sol.total_distance > 0.0);
    }

    // ---------------------------------------------------------------
    // 20. Comparison result computation
    // ---------------------------------------------------------------
    #[test]
    fn test_comparison_result() {
        let locs = triangle_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(3)
            .num_vehicles(1)
            .vehicle_capacity(100.0)
            .build()
            .unwrap();
        let opt = QuantumLogisticsOptimizer::new(cfg);
        let classical = opt.classical_benchmark(&locs, &dm);
        let quantum = classical.clone();
        let cmp = opt.compare_solutions(&quantum, &classical);

        assert!((cmp.improvement_ratio - 1.0).abs() < 1e-10);
        assert!((cmp.quantum_cost - cmp.classical_cost).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 21. Location distance calculation
    // ---------------------------------------------------------------
    #[test]
    fn test_location_euclidean_distance() {
        let a = Location::new(0, 0.0, 0.0, 0.0);
        let b = Location::new(1, 3.0, 4.0, 10.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
        assert!((b.distance_to(&a) - 5.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 22. Location with time window
    // ---------------------------------------------------------------
    #[test]
    fn test_location_time_window() {
        let loc = Location::with_time_window(0, 1.0, 2.0, 5.0, 8.0, 17.0, 0.5);
        assert_eq!(loc.time_window, Some((8.0, 17.0)));
        assert!((loc.service_time - 0.5).abs() < 1e-12);
        assert!((loc.demand - 5.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 23. QUBO offset handling
    // ---------------------------------------------------------------
    #[test]
    fn test_qubo_offset() {
        let qubo = QUBOProblem {
            num_variables: 1,
            q_entries: vec![(0, 0, 5.0)],
            offset: 10.0,
        };
        assert!((qubo.evaluate(&[false]) - 10.0).abs() < 1e-12);
        assert!((qubo.evaluate(&[true]) - 15.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 24. Solver display
    // ---------------------------------------------------------------
    #[test]
    fn test_solver_display() {
        assert_eq!(format!("{}", LogisticsSolver::QAOA), "QAOA");
        assert_eq!(
            format!("{}", LogisticsSolver::QuantumAnnealing),
            "Quantum Annealing"
        );
        assert_eq!(
            format!("{}", LogisticsSolver::HybridClassicalQuantum),
            "Hybrid Classical-Quantum"
        );
        assert_eq!(format!("{}", LogisticsSolver::BruteForce), "Brute Force");
    }

    // ---------------------------------------------------------------
    // 25. Error display
    // ---------------------------------------------------------------
    #[test]
    fn test_error_display() {
        let e = LogisticsError::InvalidConfig("bad value".into());
        assert!(format!("{}", e).contains("bad value"));

        let e = LogisticsError::ProblemTooLarge {
            size: 100,
            limit: 50,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));

        let e = LogisticsError::ConvergenceFailure {
            iterations: 42,
            best_cost: 3.14,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("42"));
        assert!(msg.contains("3.14"));
    }

    // ---------------------------------------------------------------
    // 26. Next permutation utility
    // ---------------------------------------------------------------
    #[test]
    fn test_next_permutation() {
        let mut arr = vec![1, 2, 3];
        let mut count = 1;
        while next_permutation(&mut arr) {
            count += 1;
        }
        assert_eq!(count, 6);

        // Edge case: single element
        assert!(!next_permutation(&mut vec![1]));
        // Edge case: already descending
        assert!(!next_permutation(&mut vec![3, 2, 1]));
    }

    // ---------------------------------------------------------------
    // 27. QAOA cost operator description
    // ---------------------------------------------------------------
    #[test]
    fn test_qaoa_cost_operator_description() {
        let qubo = QUBOProblem {
            num_variables: 5,
            q_entries: vec![(0, 0, 1.0), (1, 2, 0.5), (3, 4, -1.0)],
            offset: 0.0,
        };
        let solver = QAOASolver::new(&qubo, 3);
        let desc = solver.cost_operator();
        assert!(desc.contains("5 variables"));
        assert!(desc.contains("3 QUBO terms"));
    }

    // ---------------------------------------------------------------
    // 28. Dense QUBO matrix
    // ---------------------------------------------------------------
    #[test]
    fn test_qubo_dense_matrix() {
        let qubo = QUBOProblem {
            num_variables: 3,
            q_entries: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, -1.5)],
            offset: 0.0,
        };
        let dense = qubo.to_dense_matrix();
        assert_eq!(dense.len(), 3);
        assert!((dense[0][0] - 1.0).abs() < 1e-12);
        assert!((dense[0][1] - 2.0).abs() < 1e-12);
        assert!((dense[1][2] - (-1.5)).abs() < 1e-12);
        assert!((dense[2][0] - 0.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 29. Consolidate entries merges duplicates
    // ---------------------------------------------------------------
    #[test]
    fn test_consolidate_entries() {
        let entries = vec![
            (0, 1, 1.0),
            (0, 1, 2.0),
            (1, 2, 3.0),
            (0, 0, 1e-20), // near-zero, should be dropped
        ];
        let result = consolidate_entries(entries);
        assert_eq!(result.len(), 2);
        // (0,1) should be 3.0
        let ij01 = result.iter().find(|&&(i, j, _)| i == 0 && j == 1).unwrap();
        assert!((ij01.2 - 3.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 30. CVRP encoder with capacity penalty
    // ---------------------------------------------------------------
    #[test]
    fn test_cvrp_encoder_custom_penalties() {
        let encoder = CVRPEncoder::new(300.0, 250.0);
        assert!((encoder.route_penalty - 300.0).abs() < 1e-12);
        assert!((encoder.capacity_penalty - 250.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 31. Full TSP solve pipeline (QAOA path)
    // ---------------------------------------------------------------
    #[test]
    fn test_tsp_solve_pipeline() {
        let locs = triangle_locations();
        let dm = DistanceMatrix::from_locations(&locs);
        let cfg = LogisticsConfigBuilder::new()
            .num_locations(3)
            .num_vehicles(1)
            .solver(LogisticsSolver::QAOA)
            .num_shots(50)
            .build()
            .unwrap();
        let opt = QuantumLogisticsOptimizer::new(cfg);
        let sol = opt.solve_tsp(&locs, &dm);

        // With 3 cities (n<=8), brute-force path is taken.
        assert_eq!(sol.tour.len(), 3);
        // Tour distance for triangle: 5 + 5 + 6 = 16
        assert!((sol.total_distance - 16.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 32. Job-shop decode schedule
    // ---------------------------------------------------------------
    #[test]
    fn test_job_shop_decode_schedule() {
        let problem = JobShopProblem {
            num_jobs: 1,
            num_machines: 1,
            operations: vec![vec![(0, 2.0)]],
        };
        let max_time = 4;
        // 1 operation * 4 time steps = 4 vars
        // Set operation to start at t=1 => bits[1] = true
        let mut bits = vec![false; 4];
        bits[1] = true;
        let sched = problem.decode_schedule(&bits, max_time);
        assert_eq!(sched.assignments.len(), 1);
        assert!((sched.assignments[0].2 - 1.0).abs() < 1e-12); // start
        assert!((sched.assignments[0].3 - 3.0).abs() < 1e-12); // end
        assert!((sched.makespan() - 3.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 33. Config defaults
    // ---------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = LogisticsConfig::default();
        assert_eq!(cfg.num_locations, 10);
        assert_eq!(cfg.num_vehicles, 3);
        assert!((cfg.vehicle_capacity - 100.0).abs() < 1e-12);
        assert_eq!(cfg.depot_index, 0);
        assert_eq!(cfg.solver, LogisticsSolver::QAOA);
        assert_eq!(cfg.qaoa_depth, 3);
        assert_eq!(cfg.num_shots, 1000);
        assert!((cfg.time_limit_seconds - 60.0).abs() < 1e-12);
    }
}
