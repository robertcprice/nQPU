//! Classical Shadows for scalable tomography / observable estimation.

use crate::tensor_network::MPSSimulator;
use crate::{GateOperations, QuantumState};
use rand::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SingleQBasis {
    Z,
    X,
    Y,
}

#[derive(Clone, Debug)]
pub struct ShadowSample {
    pub bitstring: usize,
    pub bases: Vec<SingleQBasis>,
}

#[derive(Clone, Debug)]
pub struct ShadowDataset {
    pub num_qubits: usize,
    pub samples: Vec<ShadowSample>,
}

#[derive(Clone, Debug)]
pub struct PauliObservable {
    /// Pauli string over qubits, e.g. "IZX".
    pub paulis: Vec<char>,
}

impl PauliObservable {
    pub fn new(paulis: &str) -> Self {
        Self {
            paulis: paulis.chars().collect(),
        }
    }
}

fn apply_basis_statevector(state: &mut QuantumState, qubit: usize, basis: SingleQBasis) {
    match basis {
        SingleQBasis::Z => {}
        SingleQBasis::X => GateOperations::h(state, qubit),
        SingleQBasis::Y => {
            // Y basis: S^† then H. We don't have Sdg, so use S^3.
            GateOperations::s(state, qubit);
            GateOperations::s(state, qubit);
            GateOperations::s(state, qubit);
            GateOperations::h(state, qubit);
        }
    }
}

fn apply_basis_mps(state: &mut MPSSimulator, qubit: usize, basis: SingleQBasis) {
    match basis {
        SingleQBasis::Z => {}
        SingleQBasis::X => state.h(qubit),
        SingleQBasis::Y => {
            state.s(qubit);
            state.s(qubit);
            state.s(qubit);
            state.h(qubit);
        }
    }
}

fn random_bases(num_qubits: usize, rng: &mut impl Rng) -> Vec<SingleQBasis> {
    (0..num_qubits)
        .map(|_| match rng.gen_range(0..3) {
            0 => SingleQBasis::Z,
            1 => SingleQBasis::X,
            _ => SingleQBasis::Y,
        })
        .collect()
}

/// Generate classical shadows using state vector backend.
pub fn classical_shadows_state_vector<F>(
    num_qubits: usize,
    shots: usize,
    mut prepare: F,
) -> ShadowDataset
where
    F: FnMut() -> QuantumState,
{
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(shots);

    for _ in 0..shots {
        let bases = random_bases(num_qubits, &mut rng);
        let mut state = prepare();
        for (q, b) in bases.iter().enumerate() {
            apply_basis_statevector(&mut state, q, *b);
        }
        let (bitstring, _) = state.measure();
        samples.push(ShadowSample { bitstring, bases });
    }

    ShadowDataset {
        num_qubits,
        samples,
    }
}

/// Generate classical shadows using MPS backend.
pub fn classical_shadows_mps<F>(num_qubits: usize, shots: usize, mut prepare: F) -> ShadowDataset
where
    F: FnMut() -> MPSSimulator,
{
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(shots);

    for _ in 0..shots {
        let bases = random_bases(num_qubits, &mut rng);
        let mut state = prepare();
        for (q, b) in bases.iter().enumerate() {
            apply_basis_mps(&mut state, q, *b);
        }
        let bitstring = state.measure();
        samples.push(ShadowSample { bitstring, bases });
    }

    ShadowDataset {
        num_qubits,
        samples,
    }
}

/// Estimate expectation value of a Pauli observable from shadows.
pub fn estimate_pauli_observable(dataset: &ShadowDataset, obs: &PauliObservable) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;

    for sample in &dataset.samples {
        let mut weight = 1.0;
        for (q, p) in obs.paulis.iter().enumerate() {
            match p {
                'I' => {}
                'X' => {
                    if sample.bases[q] == SingleQBasis::X {
                        let bit = (sample.bitstring >> q) & 1;
                        let val = if bit == 0 { 1.0 } else { -1.0 };
                        weight *= 3.0 * val;
                    } else {
                        weight = 0.0;
                        break;
                    }
                }
                'Y' => {
                    if sample.bases[q] == SingleQBasis::Y {
                        let bit = (sample.bitstring >> q) & 1;
                        let val = if bit == 0 { 1.0 } else { -1.0 };
                        weight *= 3.0 * val;
                    } else {
                        weight = 0.0;
                        break;
                    }
                }
                'Z' => {
                    if sample.bases[q] == SingleQBasis::Z {
                        let bit = (sample.bitstring >> q) & 1;
                        let val = if bit == 0 { 1.0 } else { -1.0 };
                        weight *= 3.0 * val;
                    } else {
                        weight = 0.0;
                        break;
                    }
                }
                _ => {}
            }
        }
        sum += weight;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadows_bell_state() {
        // Prepare Bell state and estimate ZZ ~ 1
        let dataset = classical_shadows_state_vector(2, 2000, || {
            let mut s = QuantumState::new(2);
            GateOperations::h(&mut s, 0);
            GateOperations::cnot(&mut s, 0, 1);
            s
        });
        let zz = PauliObservable::new("ZZ");
        let est = estimate_pauli_observable(&dataset, &zz);
        assert!(est > 0.6); // noisy estimator
    }
}
