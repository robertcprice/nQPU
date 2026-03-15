//! Hybrid Clifford+T simulator using stabilizer decomposition of T.

use crate::stabilizer::StabilizerState;
use num_complex::Complex64;
use rand::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct CliffordTBranch {
    pub weight: Complex64,
    pub state: StabilizerState,
}

#[derive(Clone, Debug)]
pub struct CliffordTHybrid {
    pub branches: Vec<CliffordTBranch>,
    pub max_branches: usize,
}

impl CliffordTHybrid {
    pub fn new(num_qubits: usize) -> Self {
        let state = StabilizerState::new(num_qubits);
        Self {
            branches: vec![CliffordTBranch {
                weight: Complex64::new(1.0, 0.0),
                state,
            }],
            max_branches: 1024,
        }
    }

    pub fn with_max_branches(mut self, max: usize) -> Self {
        self.max_branches = max.max(1);
        self
    }

    pub fn num_qubits(&self) -> usize {
        self.branches[0].state.num_qubits()
    }

    pub fn h(&mut self, qubit: usize) {
        for b in &mut self.branches {
            b.state.h(qubit);
        }
    }

    pub fn s(&mut self, qubit: usize) {
        for b in &mut self.branches {
            b.state.s(qubit);
        }
    }

    pub fn x(&mut self, qubit: usize) {
        for b in &mut self.branches {
            b.state.x(qubit);
        }
    }
    pub fn y(&mut self, qubit: usize) {
        for b in &mut self.branches {
            b.state.y(qubit);
        }
    }
    pub fn z(&mut self, qubit: usize) {
        for b in &mut self.branches {
            b.state.z(qubit);
        }
    }
    pub fn cnot(&mut self, c: usize, t: usize) {
        for b in &mut self.branches {
            b.state.cx(c, t);
        }
    }
    pub fn cz(&mut self, c: usize, t: usize) {
        for b in &mut self.branches {
            b.state.cz(c, t);
        }
    }
    pub fn swap(&mut self, a: usize, bq: usize) {
        for b in &mut self.branches {
            b.state.swap(a, bq);
        }
    }

    /// Apply T gate via stabilizer decomposition: T = a I + b Z.
    pub fn t(&mut self, qubit: usize) {
        let phase = std::f64::consts::FRAC_PI_4;
        let a = Complex64::new((1.0 + phase.cos()) / 2.0, phase.sin() / 2.0);
        let b = Complex64::new((1.0 - phase.cos()) / 2.0, -phase.sin() / 2.0);

        let mut new_branches = Vec::with_capacity(self.branches.len() * 2);
        for br in &self.branches {
            let s1 = br.state.clone();
            let mut s2 = br.state.clone();
            // branch 1: I
            new_branches.push(CliffordTBranch {
                weight: br.weight * a,
                state: s1,
            });
            // branch 2: Z
            s2.z(qubit);
            new_branches.push(CliffordTBranch {
                weight: br.weight * b,
                state: s2,
            });
        }

        // Truncate branches by weight magnitude if needed
        if new_branches.len() > self.max_branches {
            new_branches.sort_by(|a, b| {
                b.weight
                    .norm_sqr()
                    .partial_cmp(&a.weight.norm_sqr())
                    .unwrap()
            });
            new_branches.truncate(self.max_branches);
        }
        self.branches = new_branches;
    }

    /// Sample measurements from the mixture defined by branch weights.
    pub fn sample_measurements(&mut self, shots: usize) -> HashMap<usize, usize> {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = self.branches.iter().map(|b| b.weight.norm_sqr()).collect();
        let total: f64 = weights.iter().sum();
        let mut counts = HashMap::new();

        for _ in 0..shots {
            let mut r = rng.gen::<f64>() * total;
            let mut idx = 0;
            for (i, w) in weights.iter().enumerate() {
                r -= *w;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            let mut state = self.branches[idx].state.clone();
            let mut bitstring = 0usize;
            for q in 0..state.num_qubits() {
                let bit = state.measure(q) as usize;
                bitstring |= bit << q;
            }
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clifford_t_small() {
        let mut sim = CliffordTHybrid::new(2).with_max_branches(16);
        sim.h(0);
        sim.cnot(0, 1);
        sim.t(0);
        let counts = sim.sample_measurements(200);
        let total: usize = counts.values().sum();
        assert!(total > 0);
    }
}
