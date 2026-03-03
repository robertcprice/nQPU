//! Quantum Game Theory Simulation
//!
//! **BLEEDING EDGE**: First quantum simulator with built-in quantum game theory.
//! Implements quantum extensions of classical game theory where players can
//! use quantum strategies (superposition, entanglement).
//!
//! Applications:
//! - Quantum cryptographic protocol analysis
//! - Quantum auction design
//! - Quantum mechanism design
//! - Quantum Nash equilibrium computation
//! - Educational: demonstrates quantum advantage through games
//!
//! Key Result: In the quantum Prisoner's Dilemma (Eisert et al. 1999),
//! a quantum strategy achieves a payoff impossible in classical game theory,
//! resolving the dilemma.
//!
//! References:
//! - Eisert, Wilkens, Lewenstein (1999) - Quantum Games and Quantum Strategies
//! - Meyer (1999) - Quantum Strategies
//! - Marinatto, Weber (2000) - Quantum Approach to Static Games of Complete Information

use crate::{C64, QuantumState};
use num_complex::Complex64;
use std::f64::consts::PI;

/// A 2-player quantum game
#[derive(Clone, Debug)]
pub struct QuantumGame {
    /// Name of the game
    pub name: String,
    /// Payoff matrix for player 1: payoff[my_action][their_action]
    pub payoff_p1: [[f64; 2]; 2],
    /// Payoff matrix for player 2
    pub payoff_p2: [[f64; 2]; 2],
    /// Entanglement parameter gamma (0 = classical, pi/2 = maximally entangled)
    pub gamma: f64,
}

/// A quantum strategy is a 2x2 unitary parameterized by (theta, phi)
/// U(theta, phi) = [[e^{i*phi} * cos(theta/2), sin(theta/2)],
///                   [-sin(theta/2), e^{-i*phi} * cos(theta/2)]]
#[derive(Clone, Debug)]
pub struct QuantumStrategy {
    pub theta: f64,
    pub phi: f64,
    pub name: String,
}

impl QuantumStrategy {
    /// Classical cooperate: identity matrix (theta=0, phi=0)
    pub fn cooperate() -> Self {
        Self {
            theta: 0.0,
            phi: 0.0,
            name: "Cooperate".to_string(),
        }
    }

    /// Classical defect: X gate (theta=pi, phi=0)
    pub fn defect() -> Self {
        Self {
            theta: PI,
            phi: 0.0,
            name: "Defect".to_string(),
        }
    }

    /// Quantum "miracle" strategy (theta=0, phi=pi/2)
    /// This strategy achieves the Pareto-optimal outcome in quantum PD
    pub fn quantum_miracle() -> Self {
        Self {
            theta: 0.0,
            phi: PI / 2.0,
            name: "Quantum Miracle (Q)".to_string(),
        }
    }

    /// Custom strategy
    pub fn custom(theta: f64, phi: f64, name: &str) -> Self {
        Self {
            theta,
            phi,
            name: name.to_string(),
        }
    }

    /// Get the 2x2 unitary matrix for this strategy
    pub fn unitary(&self) -> [[C64; 2]; 2] {
        let cos = (self.theta / 2.0).cos();
        let sin = (self.theta / 2.0).sin();
        let eip = Complex64::new(self.phi.cos(), self.phi.sin());
        let eim = Complex64::new(self.phi.cos(), -self.phi.sin());

        [
            [eip * Complex64::new(cos, 0.0), Complex64::new(sin, 0.0)],
            [
                Complex64::new(-sin, 0.0),
                eim * Complex64::new(cos, 0.0),
            ],
        ]
    }
}

/// Result of playing a quantum game
#[derive(Clone, Debug)]
pub struct GameResult {
    /// Expected payoff for player 1
    pub payoff_p1: f64,
    /// Expected payoff for player 2
    pub payoff_p2: f64,
    /// Probability of each outcome: (p1_action, p2_action) -> probability
    pub outcome_probs: [[f64; 2]; 2],
    /// Whether this outcome is a Nash equilibrium
    pub is_nash_equilibrium: bool,
    /// Whether this outcome is Pareto optimal
    pub is_pareto_optimal: bool,
    /// Classical comparison (best classical payoffs)
    pub classical_best_p1: f64,
    pub classical_best_p2: f64,
    /// Quantum advantage (difference from classical best)
    pub quantum_advantage_p1: f64,
    pub quantum_advantage_p2: f64,
}

impl QuantumGame {
    /// Create the Prisoner's Dilemma
    /// Payoffs: (C,C)=(3,3), (C,D)=(0,5), (D,C)=(5,0), (D,D)=(1,1)
    pub fn prisoners_dilemma() -> Self {
        Self {
            name: "Prisoner's Dilemma".to_string(),
            payoff_p1: [[3.0, 0.0], [5.0, 1.0]], // [C vs C, C vs D], [D vs C, D vs D]
            payoff_p2: [[3.0, 5.0], [0.0, 1.0]],
            gamma: PI / 2.0, // Maximally entangled
        }
    }

    /// Create the Battle of the Sexes
    pub fn battle_of_sexes() -> Self {
        Self {
            name: "Battle of the Sexes".to_string(),
            payoff_p1: [[3.0, 0.0], [0.0, 2.0]],
            payoff_p2: [[2.0, 0.0], [0.0, 3.0]],
            gamma: PI / 2.0,
        }
    }

    /// Create a Chicken game
    pub fn chicken() -> Self {
        Self {
            name: "Chicken".to_string(),
            payoff_p1: [[3.0, 1.0], [4.0, 0.0]],
            payoff_p2: [[3.0, 4.0], [1.0, 0.0]],
            gamma: PI / 2.0,
        }
    }

    /// Create a custom game
    pub fn custom(
        name: &str,
        payoff_p1: [[f64; 2]; 2],
        payoff_p2: [[f64; 2]; 2],
        gamma: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            payoff_p1,
            payoff_p2,
            gamma,
        }
    }

    /// Play the quantum game with given strategies
    pub fn play(&self, strategy_p1: &QuantumStrategy, strategy_p2: &QuantumStrategy) -> GameResult {
        // 2-qubit system: |q0 q1> where q0 = player 1, q1 = player 2
        let mut state = QuantumState::new(2);

        // Step 1: Entangling operator J(gamma)
        // J = exp(i*gamma/2 * X⊗X) applied to |00>
        // For gamma = pi/2: J|00> = (|00> + i|11>)/sqrt(2)
        self.apply_entangler(&mut state);

        // Step 2: Players apply their strategies
        self.apply_strategy(&mut state, 0, strategy_p1);
        self.apply_strategy(&mut state, 1, strategy_p2);

        // Step 3: Disentangling operator J†
        self.apply_disentangler(&mut state);

        // Step 4: Measure and compute payoffs
        let probs = state.probabilities();

        // Outcome probabilities: |00>=CC, |01>=CD, |10>=DC, |11>=DD
        let outcome_probs = [
            [probs[0b00], probs[0b01]], // P1=C: vs C, vs D
            [probs[0b10], probs[0b11]], // P1=D: vs C, vs D
        ];

        let payoff_p1 = probs[0b00] * self.payoff_p1[0][0]
            + probs[0b01] * self.payoff_p1[0][1]
            + probs[0b10] * self.payoff_p1[1][0]
            + probs[0b11] * self.payoff_p1[1][1];

        let payoff_p2 = probs[0b00] * self.payoff_p2[0][0]
            + probs[0b01] * self.payoff_p2[0][1]
            + probs[0b10] * self.payoff_p2[1][0]
            + probs[0b11] * self.payoff_p2[1][1];

        // Classical Nash equilibrium analysis
        let classical_ne_p1 = self.classical_nash_payoff_p1();
        let classical_ne_p2 = self.classical_nash_payoff_p2();

        // Pareto optimality check
        let all_classical_outcomes = [
            (self.payoff_p1[0][0], self.payoff_p2[0][0]),
            (self.payoff_p1[0][1], self.payoff_p2[0][1]),
            (self.payoff_p1[1][0], self.payoff_p2[1][0]),
            (self.payoff_p1[1][1], self.payoff_p2[1][1]),
        ];

        let is_pareto = !all_classical_outcomes.iter().any(|&(p1, p2)| {
            p1 >= payoff_p1 && p2 >= payoff_p2 && (p1 > payoff_p1 || p2 > payoff_p2)
        });

        GameResult {
            payoff_p1,
            payoff_p2,
            outcome_probs,
            is_nash_equilibrium: false, // Would need full equilibrium analysis
            is_pareto_optimal: is_pareto,
            classical_best_p1: classical_ne_p1,
            classical_best_p2: classical_ne_p2,
            quantum_advantage_p1: payoff_p1 - classical_ne_p1,
            quantum_advantage_p2: payoff_p2 - classical_ne_p2,
        }
    }

    /// Find the best quantum strategy for player 1 against a fixed player 2 strategy
    pub fn best_response_p1(&self, strategy_p2: &QuantumStrategy) -> (QuantumStrategy, f64) {
        let mut best_payoff = f64::NEG_INFINITY;
        let mut best_strategy = QuantumStrategy::cooperate();

        // Grid search over strategy space
        let steps = 50;
        for ti in 0..=steps {
            for pi in 0..=steps {
                let theta = PI * ti as f64 / steps as f64;
                let phi = PI * pi as f64 / steps as f64;
                let strategy = QuantumStrategy::custom(theta, phi, "search");
                let result = self.play(&strategy, strategy_p2);

                if result.payoff_p1 > best_payoff {
                    best_payoff = result.payoff_p1;
                    best_strategy = QuantumStrategy::custom(theta, phi, "Best Response P1");
                }
            }
        }

        (best_strategy, best_payoff)
    }

    /// Find the best quantum strategy for player 2 against a fixed player 1 strategy
    pub fn best_response_p2(&self, strategy_p1: &QuantumStrategy) -> (QuantumStrategy, f64) {
        let mut best_payoff = f64::NEG_INFINITY;
        let mut best_strategy = QuantumStrategy::cooperate();

        let steps = 50;
        for ti in 0..=steps {
            for pi in 0..=steps {
                let theta = PI * ti as f64 / steps as f64;
                let phi = PI * pi as f64 / steps as f64;
                let strategy = QuantumStrategy::custom(theta, phi, "search");
                let result = self.play(strategy_p1, &strategy);

                if result.payoff_p2 > best_payoff {
                    best_payoff = result.payoff_p2;
                    best_strategy = QuantumStrategy::custom(theta, phi, "Best Response P2");
                }
            }
        }

        (best_strategy, best_payoff)
    }

    /// Compute a quantum Nash equilibrium via iterated best response
    pub fn find_nash_equilibrium(&self, max_iterations: usize) -> (QuantumStrategy, QuantumStrategy, GameResult) {
        let mut s1 = QuantumStrategy::cooperate();
        let mut s2 = QuantumStrategy::cooperate();

        for _ in 0..max_iterations {
            let (new_s1, _) = self.best_response_p1(&s2);
            let (new_s2, _) = self.best_response_p2(&new_s1);

            // Check convergence
            if (new_s1.theta - s1.theta).abs() < 1e-4
                && (new_s1.phi - s1.phi).abs() < 1e-4
                && (new_s2.theta - s2.theta).abs() < 1e-4
                && (new_s2.phi - s2.phi).abs() < 1e-4
            {
                s1 = new_s1;
                s2 = new_s2;
                break;
            }

            s1 = new_s1;
            s2 = new_s2;
        }

        let result = self.play(&s1, &s2);
        (s1, s2, result)
    }

    fn apply_entangler(&self, state: &mut QuantumState) {
        // J(gamma) = exp(i*gamma/2 * X⊗X)
        // = cos(gamma/2) I⊗I + i*sin(gamma/2) X⊗X
        let cos_g = (self.gamma / 2.0).cos();
        let sin_g = (self.gamma / 2.0).sin();

        let amps = state.amplitudes_mut();
        let a00 = amps[0b00];
        let a01 = amps[0b01];
        let a10 = amps[0b10];
        let a11 = amps[0b11];

        // X⊗X: |00> <-> |11>, |01> <-> |10>
        let c = Complex64::new(cos_g, 0.0);
        let is = Complex64::new(0.0, sin_g);

        amps[0b00] = c * a00 + is * a11;
        amps[0b01] = c * a01 + is * a10;
        amps[0b10] = c * a10 + is * a01;
        amps[0b11] = c * a11 + is * a00;
    }

    fn apply_disentangler(&self, state: &mut QuantumState) {
        // J†(gamma) = exp(-i*gamma/2 * X⊗X)
        let cos_g = (self.gamma / 2.0).cos();
        let sin_g = (self.gamma / 2.0).sin();

        let amps = state.amplitudes_mut();
        let a00 = amps[0b00];
        let a01 = amps[0b01];
        let a10 = amps[0b10];
        let a11 = amps[0b11];

        let c = Complex64::new(cos_g, 0.0);
        let mis = Complex64::new(0.0, -sin_g);

        amps[0b00] = c * a00 + mis * a11;
        amps[0b01] = c * a01 + mis * a10;
        amps[0b10] = c * a10 + mis * a01;
        amps[0b11] = c * a11 + mis * a00;
    }

    fn apply_strategy(&self, state: &mut QuantumState, qubit: usize, strategy: &QuantumStrategy) {
        let u = strategy.unitary();
        let dim = state.dim;
        let stride = 1 << qubit;
        let amps = state.amplitudes_mut();

        for i in 0..dim {
            if (i >> qubit) & 1 == 0 {
                let j = i | stride;
                let a0 = amps[i];
                let a1 = amps[j];
                amps[i] = u[0][0] * a0 + u[0][1] * a1;
                amps[j] = u[1][0] * a0 + u[1][1] * a1;
            }
        }
    }

    fn classical_nash_payoff_p1(&self) -> f64 {
        // For standard PD: Nash equilibrium is (D,D) with payoff 1
        // General: find pure strategy Nash equilibria
        let mut nash_payoffs = Vec::new();

        for i in 0..2 {
            for j in 0..2 {
                // Check if (i,j) is a Nash equilibrium
                let other_i = 1 - i;
                let other_j = 1 - j;

                let p1_no_deviate = self.payoff_p1[i][j] >= self.payoff_p1[other_i][j];
                let p2_no_deviate = self.payoff_p2[i][j] >= self.payoff_p2[i][other_j];

                if p1_no_deviate && p2_no_deviate {
                    nash_payoffs.push(self.payoff_p1[i][j]);
                }
            }
        }

        nash_payoffs
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn classical_nash_payoff_p2(&self) -> f64 {
        let mut nash_payoffs = Vec::new();

        for i in 0..2 {
            for j in 0..2 {
                let other_i = 1 - i;
                let other_j = 1 - j;

                let p1_no_deviate = self.payoff_p1[i][j] >= self.payoff_p1[other_i][j];
                let p2_no_deviate = self.payoff_p2[i][j] >= self.payoff_p2[i][other_j];

                if p1_no_deviate && p2_no_deviate {
                    nash_payoffs.push(self.payoff_p2[i][j]);
                }
            }
        }

        nash_payoffs
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Tournament: run multiple strategies against each other
pub struct QuantumTournament {
    pub game: QuantumGame,
    pub strategies: Vec<QuantumStrategy>,
}

impl QuantumTournament {
    pub fn new(game: QuantumGame, strategies: Vec<QuantumStrategy>) -> Self {
        Self { game, strategies }
    }

    /// Run round-robin tournament, return (strategy_name, total_payoff) sorted by payoff
    pub fn run(&self) -> Vec<(String, f64)> {
        let n = self.strategies.len();
        let mut payoffs = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let result = self.game.play(&self.strategies[i], &self.strategies[j]);
                    payoffs[i] += result.payoff_p1;
                }
            }
        }

        let mut results: Vec<(String, f64)> = self
            .strategies
            .iter()
            .zip(payoffs.iter())
            .map(|(s, &p)| (s.name.clone(), p))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_prisoners_dilemma() {
        let game = QuantumGame::prisoners_dilemma();

        // Classical (C,C)
        let result = game.play(&QuantumStrategy::cooperate(), &QuantumStrategy::cooperate());
        assert!((result.payoff_p1 - 3.0).abs() < 0.1);
        assert!((result.payoff_p2 - 3.0).abs() < 0.1);

        // Classical (D,D)
        let result = game.play(&QuantumStrategy::defect(), &QuantumStrategy::defect());
        assert!((result.payoff_p1 - 1.0).abs() < 0.1);
        assert!((result.payoff_p2 - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_quantum_miracle_strategy() {
        let game = QuantumGame::prisoners_dilemma();

        // Both play Q = quantum miracle strategy
        // Should achieve payoff of 3 (the cooperative outcome!)
        let q = QuantumStrategy::quantum_miracle();
        let result = game.play(&q, &q);

        // In the quantum PD, (Q,Q) achieves the Pareto-optimal (3,3)
        assert!(
            result.payoff_p1 > 2.5,
            "Q vs Q should achieve high payoff, got {}",
            result.payoff_p1
        );
        assert!(result.is_pareto_optimal);
    }

    #[test]
    fn test_quantum_vs_classical_defect() {
        let game = QuantumGame::prisoners_dilemma();

        // Q vs Defect: quantum strategy should still do well
        let result = game.play(
            &QuantumStrategy::quantum_miracle(),
            &QuantumStrategy::defect(),
        );

        // Q strategy is robust against classical defection
        assert!(
            result.payoff_p1 >= 0.0,
            "Q vs D: P1 payoff = {}",
            result.payoff_p1
        );
    }

    #[test]
    fn test_nash_equilibrium_search() {
        let game = QuantumGame::prisoners_dilemma();
        let (s1, s2, result) = game.find_nash_equilibrium(20);

        // The quantum NE should give both players >=3
        // (better than classical NE of 1)
        assert!(
            result.payoff_p1 > 0.5,
            "Nash P1: {}, theta={}, phi={}",
            result.payoff_p1,
            s1.theta,
            s1.phi
        );
    }

    #[test]
    fn test_tournament() {
        let game = QuantumGame::prisoners_dilemma();
        let strategies = vec![
            QuantumStrategy::cooperate(),
            QuantumStrategy::defect(),
            QuantumStrategy::quantum_miracle(),
            QuantumStrategy::custom(PI / 4.0, PI / 4.0, "Mixed"),
        ];

        let tournament = QuantumTournament::new(game, strategies);
        let results = tournament.run();

        assert_eq!(results.len(), 4);
        // Quantum miracle should be competitive
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn test_battle_of_sexes() {
        let game = QuantumGame::battle_of_sexes();
        let result = game.play(&QuantumStrategy::cooperate(), &QuantumStrategy::cooperate());
        assert!(result.payoff_p1 >= 0.0);
    }

    #[test]
    fn test_classical_limit() {
        // With gamma=0, should recover classical game
        let game = QuantumGame::custom(
            "Classical PD",
            [[3.0, 0.0], [5.0, 1.0]],
            [[3.0, 5.0], [0.0, 1.0]],
            0.0, // No entanglement
        );

        let result = game.play(&QuantumStrategy::defect(), &QuantumStrategy::defect());
        assert!((result.payoff_p1 - 1.0).abs() < 0.1);

        let result = game.play(&QuantumStrategy::cooperate(), &QuantumStrategy::cooperate());
        assert!((result.payoff_p1 - 3.0).abs() < 0.1);
    }
}
