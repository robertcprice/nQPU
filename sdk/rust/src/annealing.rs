//! Quantum Annealing
//!
//! This module implements quantum annealing for optimization
//! using simulated quantum tunneling to escape local minima.

use ndarray::Array2;
use rand::{rngs::ThreadRng, Rng};


// ============================================================
// ANNEALING DATA STRUCTURES
// ============================================================

/// Annealing schedule
#[derive(Clone, Debug)]
pub enum AnnealingSchedule {
    /// Exponential cooling: T(t) = T0 * α^t
    Exponential {
        initial_temp: f64,
        alpha: f64, // Cooling rate
    },
    /// Linear cooling: T(t) = T0 - α*t
    Linear {
        initial_temp: f64,
        cooling_rate: f64, // Temperature decrease per iteration
    },
}

impl Default for AnnealingSchedule {
    fn default() -> Self {
        Self::Exponential {
            initial_temp: 10.0,
            alpha: 0.99,
        }
    }
}

/// Annealing configuration
#[derive(Clone, Debug)]
pub struct AnnealingConfig {
    /// Cost function to minimize
    pub cost_matrix: Array2<f64>,
    /// Number of variables
    pub num_variables: usize,
    /// Annealing schedule
    pub schedule: AnnealingSchedule,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Quantum tunneling probability
    pub tunneling_prob: f64,
    /// Restart rate
    pub restart_rate: f64,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            cost_matrix: Array2::zeros((2, 2)),
            num_variables: 2,
            schedule: AnnealingSchedule::Exponential {
                initial_temp: 10.0,
                alpha: 0.99,
            },
            max_iterations: 1000,
            tunneling_prob: 0.1,
            restart_rate: 0.01,
        }
    }
}

// ============================================================
// QUANTUM ANNEALING SOLVER
// ============================================================

/// Quantum annealing solver
pub struct QuantumAnnealing {
    config: AnnealingConfig,
    rng: ThreadRng,
}

impl QuantumAnnealing {
    pub fn new(config: AnnealingConfig) -> Self {
        Self {
            config,
            rng: rand::rngs::ThreadRng::default(),
        }
    }

    /// Run quantum annealing optimization
    pub fn anneal(&mut self) -> AnnealingResult {
        let mut current_solution = self.random_solution();
        let mut current_cost = self.evaluate_cost(&current_solution);
        let mut best_solution = current_solution.clone();
        let mut best_cost = current_cost;
        let mut temperature = match &self.config.schedule {
            AnnealingSchedule::Exponential { initial_temp, .. } => *initial_temp,
            AnnealingSchedule::Linear { initial_temp, .. } => *initial_temp,
        };

        for iteration in 0..self.config.max_iterations {
            // Generate neighbor by flipping random bits
            let neighbor = self.generate_neighbor(&current_solution);

            // Calculate energy difference
            let neighbor_cost = self.evaluate_cost(&neighbor);
            let delta = neighbor_cost - current_cost;

            // Metropolis-Hastings acceptance criterion
            let accept_prob = if delta < 0.0 {
                1.0 // Always accept improvements
            } else {
                (-delta / temperature).exp()
            };

            // Quantum tunneling: occasionally accept worse solutions
            let tunnel = self.rng.gen::<f64>() < self.config.tunneling_prob;
            let accept = tunnel || self.rng.gen::<f64>() < accept_prob;

            if accept {
                current_solution = neighbor;
                current_cost = neighbor_cost;

                if current_cost < best_cost {
                    best_cost = current_cost;
                    best_solution = current_solution.clone();
                }
            }

            // Update temperature according to schedule
            temperature = match &self.config.schedule {
                AnnealingSchedule::Exponential {
                    initial_temp: _,
                    alpha,
                } => temperature * *alpha,
                AnnealingSchedule::Linear {
                    initial_temp: _,
                    cooling_rate,
                } => {
                    let new_temp = temperature - *cooling_rate;
                    if new_temp > 0.01 {
                        new_temp
                    } else {
                        0.01
                    }
                }
            };

            // Periodic restart
            if iteration % 100 == 0 && self.rng.gen::<f64>() < self.config.restart_rate {
                current_solution = self.random_solution();
                current_cost = self.evaluate_cost(&current_solution);
            }
        }

        AnnealingResult {
            best_solution,
            best_cost,
            iterations: self.config.max_iterations,
            final_temperature: temperature,
        }
    }

    /// Generate random initial solution
    fn random_solution(&mut self) -> Vec<bool> {
        (0..self.config.num_variables)
            .map(|_| self.rng.gen::<bool>())
            .collect()
    }

    /// Generate neighbor by flipping random bits
    fn generate_neighbor(&mut self, solution: &[bool]) -> Vec<bool> {
        let mut neighbor = solution.to_vec();
        let n = self.config.num_variables;

        // Flip 1-2 unique bits for local search
        let max_flips = if n < 2 { 1 } else { 2 };
        let num_flips = self.rng.gen_range(1..=max_flips);

        // Collect unique indices to flip
        let mut flipped = std::collections::HashSet::new();
        while flipped.len() < num_flips {
            flipped.insert(self.rng.gen_range(0..n));
        }
        for bit in flipped {
            neighbor[bit] = !neighbor[bit];
        }

        neighbor
    }

    /// Evaluate cost for binary solution
    fn evaluate_cost(&self, solution: &[bool]) -> f64 {
        let mut cost = 0.0;

        // Off-diagonal terms
        for i in 0..solution.len() {
            if solution[i] {
                cost += self.config.cost_matrix[[i, i]];
            }
        }

        // Interaction terms
        for i in 0..solution.len() {
            for j in (i + 1)..solution.len() {
                if solution[i] && solution[j] {
                    cost += self.config.cost_matrix[[i, j]];
                }
            }
        }

        cost
    }
}

/// Annealing optimization result
#[derive(Clone, Debug)]
pub struct AnnealingResult {
    /// Best solution found
    pub best_solution: Vec<bool>,
    /// Best cost achieved
    pub best_cost: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Final temperature
    pub final_temperature: f64,
}

// ============================================================
// PRE-BUILT COST FUNCTIONS
// ============================================================

/// Common cost functions for annealing
pub mod cost_functions {
    use super::*;

    /// MaxCut problem as annealing optimization
    pub fn max_cut_annealing(weights: &[f64]) -> AnnealingConfig {
        let n = weights.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = -weights[i] * weights[j];
            }
        }

        AnnealingConfig {
            cost_matrix: matrix,
            num_variables: n,
            ..Default::default()
        }
    }

    /// Number partitioning problem
    pub fn number_partitioning(numbers: &[f64]) -> AnnealingConfig {
        let n = numbers.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = (numbers[i] - numbers[j]).powi(2);
            }
        }

        AnnealingConfig {
            cost_matrix: matrix,
            num_variables: n,
            ..Default::default()
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annealing_basic() {
        let matrix = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let config = AnnealingConfig {
            cost_matrix: matrix,
            num_variables: 2,
            ..Default::default()
        };

        let mut solver = QuantumAnnealing::new(config);
        let result = solver.anneal();

        assert_eq!(result.best_solution.len(), 2);
    }

    #[test]
    fn test_neighbor_generation() {
        let matrix = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let config = AnnealingConfig {
            cost_matrix: matrix,
            num_variables: 2,
            ..Default::default()
        };

        let mut solver = QuantumAnnealing::new(config);
        let solution = vec![false, true];
        let neighbor = solver.generate_neighbor(&solution);

        // Should differ by 1-2 bits (local search flips 1-2 unique bits)
        let num_diff = solution
            .iter()
            .zip(neighbor.iter())
            .filter(|(a, b)| a != b)
            .count();

        assert!(
            num_diff >= 1 && num_diff <= 2,
            "expected 1-2 bit flips, got {}",
            num_diff
        );
    }
}
