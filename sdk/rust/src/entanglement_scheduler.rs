//! Entanglement-aware scheduling utilities for MPS simulators.

use num_complex::Complex64;

use crate::tensor_network::MPSSimulator;

/// Edge between two qubits with an optional scheduling weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub q1: usize,
    pub q2: usize,
    pub weight: f64,
}

/// Alias for Edge - used by gate scheduling contexts.
pub type GateEdge = Edge;

impl Edge {
    pub fn new(q1: usize, q2: usize, weight: f64) -> Self {
        Self { q1, q2, weight }
    }

    pub fn distance(&self) -> usize {
        if self.q1 > self.q2 {
            self.q1 - self.q2
        } else {
            self.q2 - self.q1
        }
    }
}

/// Sort edges by increasing distance (shorter interactions first).
pub fn schedule_edges_by_distance(mut edges: Vec<Edge>) -> Vec<Edge> {
    edges.sort_by_key(|e| (e.distance(), e.q1.min(e.q2), e.q1.max(e.q2)));
    edges
}

/// Greedy layer packing: place disjoint edges into parallel layers.
pub fn greedy_layers(edges: &[Edge]) -> Vec<Vec<Edge>> {
    let mut layers: Vec<Vec<Edge>> = Vec::new();
    for edge in edges.iter().copied() {
        let mut placed = false;
        for layer in layers.iter_mut() {
            if !layer
                .iter()
                .any(|e| e.q1 == edge.q1 || e.q1 == edge.q2 || e.q2 == edge.q1 || e.q2 == edge.q2)
            {
                layer.push(edge);
                placed = true;
                break;
            }
        }
        if !placed {
            layers.push(vec![edge]);
        }
    }
    layers
}

/// Order edges using the entanglement profile when available.
pub fn order_edges_by_entanglement(sim: &MPSSimulator, edges: &[Edge]) -> Vec<Edge> {
    let Some(profile) = sim.entanglement_profile() else {
        return schedule_edges_by_distance(edges.to_vec());
    };

    let mut weighted: Vec<(Edge, f64)> = edges
        .iter()
        .copied()
        .map(|e| {
            let (a, b) = if e.q1 < e.q2 {
                (e.q1, e.q2)
            } else {
                (e.q2, e.q1)
            };
            let slice = &profile[a..b];
            let avg = if slice.is_empty() {
                0.0
            } else {
                slice.iter().sum::<f64>() / slice.len() as f64
            };
            (e, avg)
        })
        .collect();

    weighted.sort_by(|(ea, wa), (eb, wb)| {
        wb.partial_cmp(wa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ea.distance().cmp(&eb.distance()))
    });

    weighted.into_iter().map(|(e, _)| e).collect()
}

/// Apply a two-qubit gate using a SWAP network to make qubits adjacent.
pub fn apply_gate_with_swap_network(
    sim: &mut MPSSimulator,
    q1: usize,
    q2: usize,
    gate: &[[Complex64; 4]; 4],
) {
    if q1 == q2 {
        return;
    }
    let (a, b) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
    if b == a + 1 {
        sim.apply_two_qubit_gate_matrix(a, b, gate);
        return;
    }

    let swap = swap_gate();
    for i in a..b {
        sim.apply_two_qubit_gate_matrix(i, i + 1, &swap);
    }
    sim.apply_two_qubit_gate_matrix(b - 1, b, gate);
    for i in (a..b).rev() {
        sim.apply_two_qubit_gate_matrix(i, i + 1, &swap);
    }
}

/// Standard SWAP gate matrix.
pub fn swap_gate() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    ]
}

/// Standard CZ gate matrix.
pub fn cz_gate() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_layers_disjoint() {
        let edges = vec![
            Edge::new(0, 1, 1.0),
            Edge::new(2, 3, 1.0),
            Edge::new(1, 2, 1.0),
        ];
        let layers = greedy_layers(&edges);
        assert!(layers.len() >= 2);
        for layer in layers {
            for i in 0..layer.len() {
                for j in (i + 1)..layer.len() {
                    let a = layer[i];
                    let b = layer[j];
                    assert!(a.q1 != b.q1 && a.q1 != b.q2 && a.q2 != b.q1 && a.q2 != b.q2);
                }
            }
        }
    }

    #[test]
    fn test_swap_network_apply() {
        let mut sim = MPSSimulator::new(4, Some(4));
        let gate = cz_gate();
        apply_gate_with_swap_network(&mut sim, 0, 3, &gate);
        let _ = sim.measure();
    }
}
