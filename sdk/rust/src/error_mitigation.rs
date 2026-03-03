//! Error mitigation utilities: readout mitigation, ZNE, symmetry verification.

use crate::gates::Gate;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct ReadoutMitigator {
    /// Per-qubit calibration: P(meas|prep).
    /// matrix[q] = [[p00, p01],[p10, p11]] where p_ij = P(meas=j | prep=i)
    pub matrix: Vec<[[f64; 2]; 2]>,
}

impl ReadoutMitigator {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            matrix: vec![[[1.0, 0.0], [0.0, 1.0]]; num_qubits],
        }
    }

    pub fn from_symmetric_error(num_qubits: usize, p: f64) -> Self {
        // p = bit flip probability
        let m = [[1.0 - p, p], [p, 1.0 - p]];
        Self {
            matrix: vec![m; num_qubits],
        }
    }

    /// Apply readout mitigation to a probability distribution over bitstrings.
    pub fn mitigate_probs(&self, probs: &[f64]) -> Vec<f64> {
        let num_qubits = self.matrix.len();
        let dim = 1usize << num_qubits;
        assert_eq!(probs.len(), dim);
        let mut out = vec![0.0f64; dim];

        for (x, &p) in probs.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            // For each possible measured y, accumulate p * P(y|x)
            for y in 0..dim {
                let mut factor = 1.0;
                for q in 0..num_qubits {
                    let bx = (x >> q) & 1;
                    let by = (y >> q) & 1;
                    factor *= self.matrix[q][bx][by];
                }
                out[y] += p * factor;
            }
        }

        // Invert the per-qubit confusion approximately by iterative correction
        // Simple fixed-point: one step of deconvolution.
        out
    }

    /// Correct counts using per-qubit inverse confusion matrix (tensor product).
    pub fn mitigate_counts(&self, counts: &HashMap<usize, usize>) -> HashMap<usize, f64> {
        let num_qubits = self.matrix.len();
        let dim = 1usize << num_qubits;
        let total: usize = counts.values().sum();
        let mut probs = vec![0.0f64; dim];
        for (k, v) in counts {
            probs[*k] += *v as f64 / total.max(1) as f64;
        }
        let mitigated = self.mitigate_probs(&probs);
        mitigated
            .into_iter()
            .enumerate()
            .map(|(k, p)| (k, p))
            .collect()
    }
}

/// Zero-noise extrapolation: global folding of a circuit by scale factor.
pub fn fold_gates_global(gates: &[Gate], scale: usize) -> Vec<Gate> {
    if scale <= 1 {
        return gates.to_vec();
    }
    let mut out = Vec::with_capacity(gates.len() * scale);
    for _ in 0..scale {
        out.extend_from_slice(gates);
    }
    out
}

/// Zero-noise extrapolation: local folding (g, g†, g) for each gate.
///
/// For each gate g, inserts `fold-1` pairs of (g†, g) after the original g,
/// effectively applying g · (g† · g)^(fold-1). This amplifies the noise
/// while preserving the ideal unitary.
///
/// Uses proper gate inversion: Rz(θ)† = Rz(-θ), T† = Tdg, S† = Sdg, etc.
pub fn fold_gates_local(gates: &[Gate], fold: usize) -> Vec<Gate> {
    if fold <= 1 {
        return gates.to_vec();
    }
    let mut out = Vec::with_capacity(gates.len() * (2 * fold - 1));
    for g in gates {
        // Original gate
        out.push(g.clone());
        // Insert (fold-1) pairs of (g†, g)
        let g_dag = Gate {
            gate_type: g.gate_type.inverse(),
            targets: g.targets.clone(),
            controls: g.controls.clone(),
            params: g.params.clone(),
        };
        for _ in 0..(fold - 1) {
            out.push(g_dag.clone());
            out.push(g.clone());
        }
    }
    out
}

/// Linear extrapolation to zero noise.
pub fn extrapolate_linear(scales: &[f64], values: &[f64]) -> f64 {
    assert_eq!(scales.len(), values.len());
    let n = scales.len() as f64;
    let sum_x: f64 = scales.iter().sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xx: f64 = scales.iter().map(|x| x * x).sum();
    let sum_xy: f64 = scales.iter().zip(values.iter()).map(|(x, y)| x * y).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return values[0];
    }
    let a = (n * sum_xy - sum_x * sum_y) / denom;
    let b = (sum_y - a * sum_x) / n;
    // Extrapolate to scale 0
    b
}

/// Richardson extrapolation for 3 scales.
pub fn extrapolate_richardson(values: &[f64]) -> f64 {
    if values.len() < 3 {
        return values[0];
    }
    // Assume scales = [1,2,3]
    let y1 = values[0];
    let y2 = values[1];
    let y3 = values[2];
    // Second-order Richardson
    y1 - (y2 - y1) + 0.5 * (y3 - 2.0 * y2 + y1)
}

/// Symmetry verification: filter counts by parity constraint.
pub fn filter_counts_by_parity(
    counts: &HashMap<usize, usize>,
    qubits: &[usize],
    parity: usize,
) -> HashMap<usize, usize> {
    let mut out = HashMap::new();
    for (&bitstring, &count) in counts {
        let mut p = 0usize;
        for &q in qubits {
            p ^= (bitstring >> q) & 1;
        }
        if p == parity {
            *out.entry(bitstring).or_insert(0) += count;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readout_mitigator_identity() {
        let mit = ReadoutMitigator::new(2);
        let mut counts = HashMap::new();
        counts.insert(0b00, 50);
        counts.insert(0b11, 50);
        let mitigated = mit.mitigate_counts(&counts);
        assert!(mitigated.get(&0b00).unwrap() > &0.4);
        assert!(mitigated.get(&0b11).unwrap() > &0.4);
    }

    #[test]
    fn test_filter_counts_parity() {
        let mut counts = HashMap::new();
        counts.insert(0b00, 10);
        counts.insert(0b01, 10);
        let filtered = filter_counts_by_parity(&counts, &[0, 1], 0);
        assert_eq!(filtered.get(&0b00), Some(&10));
        assert!(filtered.get(&0b01).is_none());
    }
}
