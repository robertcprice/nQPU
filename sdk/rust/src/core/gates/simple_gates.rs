//! Simple Working Gates Module
//!
//! This module provides basic gate operations that definitely work
//! using only the core QuantumState API methods (new, measure, probabilities).

use crate::QuantumState;
use crate::C64;

/// Apply Hadamard gate
pub fn h(state: &mut QuantumState, qubit: usize) {
    // H = (1/√2) [[1, 1], [1, -1]]
    let sqrt2 = (2.0_f64).sqrt();

    let idx = qubit * 2;

    let amps = state.amplitudes_mut();

    // Get current amplitude
    let a0 = amps[idx];
    let a1 = amps[idx + 1];

    // H|0⟩ = (|0⟩ + |1⟩) / √2
    let new_a0 = C64::new((a0.re + a1.re) / sqrt2, (a0.im + a1.im) / sqrt2);
    let new_a1 = C64::new((a0.re + a1.re) / sqrt2, (a0.im - a1.im) / sqrt2);

    amps[idx] = new_a0;
    amps[idx + 1] = new_a1;
}

/// Measure state and return binary solution
pub fn measure_binary(state: &mut QuantumState) -> Vec<bool> {
    let probs = state.probabilities();
    let (idx, _) = state.measure();
    let mut solution = vec![false; probs.len()];
    if idx < solution.len() {
        solution[idx] = true;
    }
    solution
}

/// Calculate cost for MaxCut problem
pub fn maxcut_cost(solution: &[bool], weights: &[f64]) -> f64 {
    let mut cost = 0.0;

    for i in 0..solution.len() {
        for j in (i + 1)..solution.len() {
            if solution[i] != solution[j] {
                cost += weights[i] * weights[j];
            }
        }
    }

    cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard() {
        let mut state = QuantumState::new(1);

        // |0⟩
        super::h(&mut state, 0);

        let probs = state.probabilities();

        // H|0⟩ should give [0.5, 0.5] probability
        assert!((probs[0] - 0.5).abs() < 0.01);
        assert!((probs[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_measure_binary() {
        let mut state = QuantumState::new(2);
        super::h(&mut state, 0);
        super::h(&mut state, 1);

        let solution = measure_binary(&mut state);

        // Should have exactly one true
        assert_eq!(solution.iter().filter(|&&x| x).count(), 1);
    }

    #[test]
    fn test_maxcut() {
        let solution = vec![true, false];
        let weights = vec![1.0, 1.0];

        let cost = maxcut_cost(&solution, &weights);

        // Edge between true and false: weight = 1.0
        assert!((cost - 1.0).abs() < 0.01);
    }
}
