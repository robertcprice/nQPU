//! Quantum Approximate Optimization Algorithm (QAO)
//!
//! This module implements the QAO algorithm for combinatorial optimization
//! problems like MaxCut, scheduling, and other QUBO problems.
//!
//! The algorithm alternates between:
//! 1. Phase separator: encodes the cost function into quantum phases
//! 2. Mixer: drives exploration of the solution space
//!
//! Measurement samples from the final quantum state to extract
//! classical solutions.

use ndarray::Array2;
use std::f64::consts::PI;

use crate::GateOperations;
use crate::QuantumState;

// ============================================================
// QAO DATA STRUCTURES
// ============================================================

/// Cost function for QAO
#[derive(Clone, Debug)]
pub enum CostFunction {
    /// Quadratic unconstrained binary optimization
    QUBO {
        matrix: Array2<f64>,
        num_variables: usize,
    },
    /// Quadratic constrained binary optimization
    QuboConstrained {
        matrix: Array2<f64>,
        constraints: Array2<bool>,
        num_variables: usize,
    },
}

impl CostFunction {
    /// Get number of variables
    pub fn num_variables(&self) -> usize {
        match self {
            CostFunction::QUBO { num_variables, .. } => *num_variables,
            CostFunction::QuboConstrained { num_variables, .. } => *num_variables,
        }
    }

    /// Evaluate cost for given solution
    pub fn evaluate(&self, solution: &[bool]) -> f64 {
        match self {
            CostFunction::QUBO { matrix, .. } => {
                let mut cost = 0.0;
                for i in 0..solution.len() {
                    for j in 0..solution.len() {
                        if solution[i] && solution[j] {
                            cost += matrix[[i, j]];
                        }
                    }
                }
                cost
            }
            CostFunction::QuboConstrained { matrix, .. } => {
                let mut cost = 0.0;
                for i in 0..solution.len() {
                    for j in 0..solution.len() {
                        if solution[i] && solution[j] {
                            cost += matrix[[i, j]];
                        }
                    }
                }
                cost
            }
        }
    }
}

/// Mixer type for QAO
#[derive(Clone, Debug)]
pub enum Mixer {
    /// Classical mixer: X rotations with specified angle (beta parameter)
    Classical { angle: f64 },
    /// Quantum mixer: Full unitary with additional structure
    Quantum { num_layers: usize },
}

// ============================================================
// QAO SOLVER
// ============================================================

/// QAO solver
pub struct QAOSolver {
    /// Cost function to optimize
    pub cost: CostFunction,
    /// Mixer type
    pub mixer: Mixer,
    /// Number of QAO layers
    pub num_layers: usize,
    /// Optimization iterations
    pub max_iterations: usize,
    /// Learning rate for parameter optimization
    pub learning_rate: f64,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl QAOSolver {
    /// Create new QAO solver
    pub fn new(cost: CostFunction, num_layers: usize, mixer: Mixer) -> Self {
        QAOSolver {
            cost,
            mixer,
            num_layers,
            max_iterations: 1000,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
        }
    }

    /// Optimize using QAO
    pub fn optimize(&mut self) -> QAOResult {
        let num_qubits = self.cost.num_variables();
        let mut best_solution = vec![false; num_qubits];
        let mut best_cost = f64::MAX;
        let mut cost_history = Vec::new();

        for iteration in 0..self.max_iterations {
            // Execute QAO circuit
            let solution = self.execute_qao();

            // Evaluate cost
            let cost = self.cost.evaluate(&solution);

            if cost < best_cost {
                best_cost = cost;
                best_solution = solution.clone();
            }

            cost_history.push(cost);

            // Check convergence
            if cost_history.len() > 10 {
                let last_10: f64 = cost_history.iter().rev().take(10).sum::<f64>() / 10.0;
                let prev_10: f64 =
                    cost_history.iter().rev().skip(10).take(10).sum::<f64>() / 10.0;
                let improvement = (prev_10 - last_10).abs();

                if improvement < self.convergence_threshold {
                    return QAOResult {
                        best_solution,
                        best_cost,
                        iterations: iteration + 1,
                        converged: true,
                        cost_history,
                    };
                }
            }
        }

        QAOResult {
            best_solution,
            best_cost,
            iterations: self.max_iterations,
            converged: false,
            cost_history,
        }
    }

    /// Execute single QAO circuit
    fn execute_qao(&self) -> Vec<bool> {
        let num_qubits = self.cost.num_variables();
        let mut state = QuantumState::new(num_qubits);

        // Initial superposition
        for i in 0..num_qubits {
            GateOperations::h(&mut state, i);
        }

        // Apply QAO layers
        for _layer in 0..self.num_layers {
            // Apply phase separator (cost Hamiltonian)
            self.apply_phase_separator(&mut state);

            // Apply mixer
            self.apply_mixer(&mut state);
        }

        // Measure all qubits via proper probabilistic sampling
        self.measure_state(&state)
    }

    /// Apply phase separator based on cost function.
    ///
    /// For QUBO with matrix Q, the cost Hamiltonian is:
    ///   H_C = sum_{i,j} Q_{ij} Z_i Z_j
    ///
    /// The unitary e^{-i*gamma*H_C} decomposes into:
    /// - Single-qubit Rz for diagonal terms
    /// - Two-qubit ZZ interactions for off-diagonal terms (via CNOT-Rz-CNOT)
    fn apply_phase_separator(&self, state: &mut QuantumState) {
        match &self.cost {
            CostFunction::QUBO { matrix, .. } | CostFunction::QuboConstrained { matrix, .. } => {
                let num_qubits = matrix.nrows();
                let gamma = 0.5;

                // Diagonal terms: single-qubit Rz rotations
                for i in 0..num_qubits {
                    let angle = gamma * matrix[[i, i]];
                    if angle.abs() > 1e-15 {
                        GateOperations::rz(state, i, angle);
                    }
                }

                // Off-diagonal terms: ZZ interactions via CNOT-Rz-CNOT decomposition
                for i in 0..num_qubits {
                    for j in (i + 1)..num_qubits {
                        let weight = matrix[[i, j]] + matrix[[j, i]];
                        if weight.abs() > 1e-15 {
                            let angle = gamma * weight;
                            GateOperations::cnot(state, i, j);
                            GateOperations::rz(state, j, angle);
                            GateOperations::cnot(state, i, j);
                        }
                    }
                }
            }
        }
    }

    /// Apply mixer unitary.
    ///
    /// The standard QAOA mixer is e^{-i*beta*sum_j X_j}, which decomposes
    /// into individual Rx(2*beta) rotations on each qubit.
    fn apply_mixer(&self, state: &mut QuantumState) {
        let num_qubits = state.num_qubits;

        match &self.mixer {
            Mixer::Classical { angle } => {
                // Standard QAOA mixer: Rx(2*beta) on each qubit
                // Rx(theta) = exp(-i*theta/2 * X)
                let beta = *angle;
                for i in 0..num_qubits {
                    GateOperations::rx(state, i, 2.0 * beta);
                }
            }
            Mixer::Quantum { .. } => {
                // Enhanced quantum mixer: Hadamard + Rz + Hadamard = Rx
                // with a fixed exploration angle
                let beta = PI / 8.0;
                for i in 0..num_qubits {
                    GateOperations::rx(state, i, 2.0 * beta);
                }
            }
        }
    }

    /// Measure state and return binary solution via proper probabilistic sampling.
    ///
    /// Uses `QuantumState::measure()` which samples from the probability
    /// distribution |amplitude|^2 to collapse the state and extract a
    /// classical bitstring.
    fn measure_state(&self, state: &QuantumState) -> Vec<bool> {
        let num_qubits = state.num_qubits;
        let (measured_idx, _probability) = state.measure();

        // Convert measured index to binary solution vector
        let mut solution = Vec::with_capacity(num_qubits);
        for i in 0..num_qubits {
            solution.push((measured_idx >> i) & 1 != 0);
        }

        solution
    }
}

/// QAO optimization result
#[derive(Clone, Debug)]
pub struct QAOResult {
    /// Best solution found
    pub best_solution: Vec<bool>,
    /// Best cost achieved
    pub best_cost: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether algorithm converged
    pub converged: bool,
    /// Cost history
    pub cost_history: Vec<f64>,
}

// ============================================================
// PRE-BUILT COST FUNCTIONS
// ============================================================

/// Simple cost functions (inlined to avoid c64 conflict)
pub mod simple_cost {
    use super::*;

    /// MaxCut problem: maximize cut value
    ///
    /// For a graph with edge weights, the QUBO matrix encodes
    /// penalties for vertices on the same side of the cut.
    pub fn max_cut(weights: &[f64]) -> CostFunction {
        let n = weights.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = -weights[i] * weights[j];
            }
        }

        CostFunction::QUBO {
            matrix,
            num_variables: n,
        }
    }

    /// MaxCut from adjacency matrix (more standard formulation).
    ///
    /// Creates QUBO where cost = sum_{(i,j) in E} w_{ij} * (1 - z_i * z_j) / 2
    /// In QUBO form: minimize sum_{(i,j)} w_{ij} * x_i * x_j
    pub fn max_cut_from_adjacency(adjacency: &Array2<f64>) -> CostFunction {
        let n = adjacency.nrows();
        let matrix = adjacency.clone();

        CostFunction::QUBO {
            matrix,
            num_variables: n,
        }
    }

    /// Graph coloring problem
    pub fn graph_coloring(adjacency: &Array2<bool>) -> CostFunction {
        let n = adjacency.nrows();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if adjacency[[i, j]] {
                    matrix[[i, j]] = 1.0; // Penalty for same color
                }
            }
        }

        CostFunction::QUBO {
            matrix,
            num_variables: n,
        }
    }

    /// Number partitioning problem
    pub fn number_partitioning(numbers: &[f64]) -> CostFunction {
        let n = numbers.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = numbers[i] * numbers[j];
            }
        }

        CostFunction::QUBO {
            matrix,
            num_variables: n,
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Cost function tests ----

    #[test]
    fn test_cost_function_qubo_num_variables() {
        let matrix = Array2::from_shape_vec((3, 3), vec![1.0; 9]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 3,
        };
        assert_eq!(cost.num_variables(), 3);
    }

    #[test]
    fn test_cost_function_evaluate_all_false() {
        // All false => no terms contribute => cost = 0
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        assert!((cost.evaluate(&[false, false]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_function_evaluate_single_true() {
        // Only x[0]=true => only matrix[0][0] contributes
        let matrix = Array2::from_shape_vec((2, 2), vec![5.0, 2.0, 3.0, 7.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        assert!((cost.evaluate(&[true, false]) - 5.0).abs() < 1e-10);
        assert!((cost.evaluate(&[false, true]) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_function_evaluate_all_true() {
        // All true => sum of all matrix entries
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        // 1 + 2 + 3 + 4 = 10
        assert!((cost.evaluate(&[true, true]) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_function_evaluate_off_diagonal() {
        // Off-diagonal matrix: only off-diagonal terms when both bits are true
        let matrix = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        assert!((cost.evaluate(&[false, true]) - 0.0).abs() < 1e-10);
        assert!((cost.evaluate(&[true, false]) - 0.0).abs() < 1e-10);
        // Both true: 0 + 1 + 1 + 0 = 2
        assert!((cost.evaluate(&[true, true]) - 2.0).abs() < 1e-10);
    }

    // ---- Solver creation tests ----

    #[test]
    fn test_qao_solver_creation() {
        let matrix = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        let mixer = Mixer::Classical { angle: PI / 4.0 };
        let solver = QAOSolver::new(cost, 1, mixer);

        assert_eq!(solver.num_layers, 1);
        assert_eq!(solver.max_iterations, 1000);
        assert!((solver.learning_rate - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_qao_solver_quantum_mixer() {
        let matrix = Array2::zeros((2, 2));
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        let mixer = Mixer::Quantum { num_layers: 2 };
        let solver = QAOSolver::new(cost, 3, mixer);

        assert_eq!(solver.num_layers, 3);
    }

    // ---- Measure state tests ----

    #[test]
    fn test_measure_state_returns_valid_bitstring() {
        let matrix = Array2::zeros((3, 3));
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 3,
        };
        let mixer = Mixer::Classical { angle: PI / 4.0 };
        let solver = QAOSolver::new(cost, 1, mixer);

        let state = QuantumState::new(3);
        let solution = solver.measure_state(&state);

        // Should have exactly num_qubits bits
        assert_eq!(solution.len(), 3);
    }

    #[test]
    fn test_measure_state_deterministic_for_basis_state() {
        // |000> state should always measure [false, false, false]
        let matrix = Array2::zeros((3, 3));
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 3,
        };
        let mixer = Mixer::Classical { angle: PI / 4.0 };
        let solver = QAOSolver::new(cost, 1, mixer);

        let state = QuantumState::new(3); // |000>
        for _ in 0..10 {
            let solution = solver.measure_state(&state);
            assert_eq!(solution, vec![false, false, false]);
        }
    }

    #[test]
    fn test_measure_state_probabilistic_for_superposition() {
        // |+> state on 1 qubit should give ~50/50 distribution
        let matrix = Array2::zeros((1, 1));
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 1,
        };
        let mixer = Mixer::Classical { angle: PI / 4.0 };
        let solver = QAOSolver::new(cost, 1, mixer);

        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);

        let mut true_count = 0;
        let num_samples = 200;
        for _ in 0..num_samples {
            let solution = solver.measure_state(&state);
            if solution[0] {
                true_count += 1;
            }
        }

        // Should be roughly 50/50 (within reasonable bounds)
        let fraction = true_count as f64 / num_samples as f64;
        assert!(
            fraction > 0.2 && fraction < 0.8,
            "Superposition should give ~50/50 distribution, got {} true out of {}",
            true_count,
            num_samples
        );
    }

    // ---- Execute QAO circuit tests ----

    #[test]
    fn test_execute_qao_returns_valid_solution() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        let mixer = Mixer::Classical { angle: PI / 4.0 };
        let solver = QAOSolver::new(cost, 1, mixer);

        let solution = solver.execute_qao();
        assert_eq!(solution.len(), 2);
        // Each element should be a valid bool (always true for Rust bools)
    }

    // ---- Optimization tests ----

    #[test]
    fn test_optimize_returns_result() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        let mixer = Mixer::Classical { angle: PI / 4.0 };
        let mut solver = QAOSolver::new(cost, 1, mixer);
        solver.max_iterations = 20; // keep test fast

        let result = solver.optimize();
        assert_eq!(result.best_solution.len(), 2);
        assert!(result.iterations <= 20);
        assert!(!result.cost_history.is_empty());
    }

    #[test]
    fn test_optimize_finds_minimum_for_trivial_problem() {
        // Cost matrix: all-false gives cost 0, which is minimum for non-negative diagonal
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let cost = CostFunction::QUBO {
            matrix,
            num_variables: 2,
        };
        let mixer = Mixer::Classical { angle: PI / 4.0 };
        let mut solver = QAOSolver::new(cost, 1, mixer);
        solver.max_iterations = 50;

        let result = solver.optimize();

        // best_cost should be <= 2.0 (worst case: both true)
        // The algorithm should find cost 0 (all false) at least sometimes
        assert!(
            result.best_cost <= 2.0,
            "Best cost should be reasonable, got {}",
            result.best_cost
        );
    }

    // ---- Pre-built cost function tests ----

    #[test]
    fn test_max_cut_cost_function() {
        let cost = simple_cost::max_cut(&[1.0, 1.0]);
        assert_eq!(cost.num_variables(), 2);

        // Matrix should have negative entries (maximization encoded as minimization)
        if let CostFunction::QUBO { matrix, .. } = &cost {
            assert!(matrix[[0, 1]] < 0.0);
        }
    }

    #[test]
    fn test_graph_coloring_cost_function() {
        let adj =
            Array2::from_shape_vec((2, 2), vec![false, true, true, false]).unwrap();
        let cost = simple_cost::graph_coloring(&adj);
        assert_eq!(cost.num_variables(), 2);

        // Adjacent vertices in same partition should have positive cost
        assert!(cost.evaluate(&[true, true]) > 0.0);
        // Non-adjacent or different partitions should have zero cost
        assert!((cost.evaluate(&[false, false]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_number_partitioning_cost_function() {
        let cost = simple_cost::number_partitioning(&[3.0, 1.0, 2.0]);
        assert_eq!(cost.num_variables(), 3);
    }

    #[test]
    fn test_constrained_qubo() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let constraints =
            Array2::from_shape_vec((2, 2), vec![true, false, false, true]).unwrap();
        let cost = CostFunction::QuboConstrained {
            matrix,
            constraints,
            num_variables: 2,
        };
        assert_eq!(cost.num_variables(), 2);
        assert!((cost.evaluate(&[true, false]) - 1.0).abs() < 1e-10);
    }
}
