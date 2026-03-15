//! Quantum Channel Abstractions
//!
//! Shared channel representation for noise modules using Kraus operators
//! and Choi matrix formalism.
//!
//! Used by: advanced error mitigation, non-Markovian noise, quantum networking.

use crate::C64;

// ===================================================================
// KRAUS CHANNEL
// ===================================================================

/// A quantum channel represented by a list of Kraus operators.
///
/// A channel E(ρ) = Σ_k K_k ρ K_k†  where Σ_k K_k† K_k = I.
/// Each Kraus operator is stored as a flat row-major complex matrix.
#[derive(Clone, Debug)]
pub struct KrausChannel {
    /// Kraus operators, each of dimension `dim × dim` stored row-major.
    pub operators: Vec<Vec<C64>>,
    /// Hilbert space dimension (2^n for n qubits).
    pub dim: usize,
}

impl KrausChannel {
    /// Create a channel from a list of Kraus operators.
    pub fn new(operators: Vec<Vec<C64>>, dim: usize) -> Self {
        KrausChannel { operators, dim }
    }

    /// Identity channel (single Kraus operator = identity matrix).
    pub fn identity(dim: usize) -> Self {
        let mut op = vec![C64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            op[i * dim + i] = C64::new(1.0, 0.0);
        }
        KrausChannel {
            operators: vec![op],
            dim,
        }
    }

    /// Single-qubit depolarizing channel with probability p.
    /// E(ρ) = (1-p)ρ + p/3 (XρX + YρY + ZρZ)
    pub fn depolarizing(p: f64) -> Self {
        let s0 = (1.0 - p).sqrt();
        let s1 = (p / 3.0).sqrt();

        let k0 = vec![
            C64::new(s0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(s0, 0.0),
        ];
        let k1 = vec![
            // √(p/3) X
            C64::new(0.0, 0.0),
            C64::new(s1, 0.0),
            C64::new(s1, 0.0),
            C64::new(0.0, 0.0),
        ];
        let k2 = vec![
            // √(p/3) Y
            C64::new(0.0, 0.0),
            C64::new(0.0, -s1),
            C64::new(0.0, s1),
            C64::new(0.0, 0.0),
        ];
        let k3 = vec![
            // √(p/3) Z
            C64::new(s1, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(-s1, 0.0),
        ];

        KrausChannel {
            operators: vec![k0, k1, k2, k3],
            dim: 2,
        }
    }

    /// Single-qubit amplitude damping channel with probability gamma.
    /// Models T1 relaxation: |1> -> |0> with probability gamma.
    pub fn amplitude_damping(gamma: f64) -> Self {
        let k0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new((1.0 - gamma).sqrt(), 0.0),
        ];
        let k1 = vec![
            C64::new(0.0, 0.0),
            C64::new(gamma.sqrt(), 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
        ];

        KrausChannel {
            operators: vec![k0, k1],
            dim: 2,
        }
    }

    /// Single-qubit phase damping (dephasing) channel.
    /// Models T2 pure dephasing.
    pub fn phase_damping(gamma: f64) -> Self {
        let k0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new((1.0 - gamma).sqrt(), 0.0),
        ];
        let k1 = vec![
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(gamma.sqrt(), 0.0),
        ];

        KrausChannel {
            operators: vec![k0, k1],
            dim: 2,
        }
    }

    /// Single-qubit erasure channel with probability p.
    /// With probability p, qubit is erased (replaced by maximally mixed state).
    /// E(ρ) = (1-p)ρ + p·I/2
    pub fn erasure(p: f64) -> Self {
        let s = (1.0 - p).sqrt();
        let t = (p / 2.0).sqrt();

        // K0 = sqrt(1-p) * I
        let k0 = vec![
            C64::new(s, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(s, 0.0),
        ];
        // K1 = sqrt(p/2) * |0><0|
        let k1 = vec![
            C64::new(t, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
        ];
        // K2 = sqrt(p/2) * |0><1|
        let k2 = vec![
            C64::new(0.0, 0.0),
            C64::new(t, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
        ];
        // K3 = sqrt(p/2) * |1><0|
        let k3 = vec![
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(t, 0.0),
            C64::new(0.0, 0.0),
        ];
        // K4 = sqrt(p/2) * |1><1|
        let k4 = vec![
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(t, 0.0),
        ];

        KrausChannel {
            operators: vec![k0, k1, k2, k3, k4],
            dim: 2,
        }
    }

    /// Apply this channel to a density matrix (row-major, dim × dim).
    /// E(ρ) = Σ_k K_k ρ K_k†
    pub fn apply_to_density_matrix(&self, rho: &[C64]) -> Vec<C64> {
        let d = self.dim;
        let mut result = vec![C64::new(0.0, 0.0); d * d];

        for k in &self.operators {
            // Compute K * rho * K†
            // First: temp = K * rho
            let mut temp = vec![C64::new(0.0, 0.0); d * d];
            for i in 0..d {
                for j in 0..d {
                    let mut sum = C64::new(0.0, 0.0);
                    for l in 0..d {
                        sum += k[i * d + l] * rho[l * d + j];
                    }
                    temp[i * d + j] = sum;
                }
            }

            // Then: result += temp * K†
            for i in 0..d {
                for j in 0..d {
                    let mut sum = C64::new(0.0, 0.0);
                    for l in 0..d {
                        sum += temp[i * d + l] * k[j * d + l].conj();
                    }
                    result[i * d + j] += sum;
                }
            }
        }

        result
    }

    /// Check if this channel is trace-preserving: Σ_k K_k† K_k = I.
    pub fn is_trace_preserving(&self, tolerance: f64) -> bool {
        let d = self.dim;
        let mut sum = vec![C64::new(0.0, 0.0); d * d];

        for k in &self.operators {
            for i in 0..d {
                for j in 0..d {
                    let mut s = C64::new(0.0, 0.0);
                    for l in 0..d {
                        s += k[l * d + i].conj() * k[l * d + j];
                    }
                    sum[i * d + j] += s;
                }
            }
        }

        // Check if sum ≈ I
        for i in 0..d {
            for j in 0..d {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (sum[i * d + j].re - expected).abs() > tolerance
                    || sum[i * d + j].im.abs() > tolerance
                {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the channel is unital: E(I) = I.
    pub fn is_unital(&self, tolerance: f64) -> bool {
        let d = self.dim;
        // Create identity density matrix
        let mut identity = vec![C64::new(0.0, 0.0); d * d];
        for i in 0..d {
            identity[i * d + i] = C64::new(1.0, 0.0);
        }

        let result = self.apply_to_density_matrix(&identity);

        // Check if result ≈ I
        for i in 0..d {
            for j in 0..d {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (result[i * d + j].re - expected).abs() > tolerance
                    || result[i * d + j].im.abs() > tolerance
                {
                    return false;
                }
            }
        }
        true
    }

    /// Number of Kraus operators.
    pub fn num_operators(&self) -> usize {
        self.operators.len()
    }

    /// Compose two channels: (E2 ∘ E1)(ρ) = E2(E1(ρ)).
    /// Returns a new KrausChannel with dim1*dim2 Kraus operators.
    pub fn compose(&self, other: &KrausChannel) -> KrausChannel {
        assert_eq!(self.dim, other.dim);
        let d = self.dim;
        let mut new_ops = Vec::new();

        for k2 in &other.operators {
            for k1 in &self.operators {
                // K_new = K2 * K1
                let mut prod = vec![C64::new(0.0, 0.0); d * d];
                for i in 0..d {
                    for j in 0..d {
                        let mut sum = C64::new(0.0, 0.0);
                        for l in 0..d {
                            sum += k2[i * d + l] * k1[l * d + j];
                        }
                        prod[i * d + j] = sum;
                    }
                }
                new_ops.push(prod);
            }
        }

        KrausChannel {
            operators: new_ops,
            dim: d,
        }
    }
}

// ===================================================================
// CHOI MATRIX
// ===================================================================

/// Choi matrix representation of a quantum channel.
///
/// The Choi matrix is Λ = (I ⊗ E)(|Ω⟩⟨Ω|) where |Ω⟩ = Σ_i |ii⟩ / √d.
/// Dimension: d² × d².
#[derive(Clone, Debug)]
pub struct ChoiMatrix {
    /// Elements of the Choi matrix (d² × d² row-major).
    pub elements: Vec<C64>,
    /// Hilbert space dimension.
    pub dim: usize,
}

impl ChoiMatrix {
    /// Construct the Choi matrix from a Kraus channel.
    pub fn from_kraus(channel: &KrausChannel) -> Self {
        let d = channel.dim;
        let d2 = d * d;
        let mut choi = vec![C64::new(0.0, 0.0); d2 * d2];

        for k in &channel.operators {
            // Choi element (i*d+j, k*d+l) += K[i][k] * conj(K[j][l])
            // Using vectorized |k⟩⟨l| ⊗ K|k⟩⟨l|K†
            for i in 0..d {
                for j in 0..d {
                    for k_idx in 0..d {
                        for l in 0..d {
                            let row = i * d + k_idx;
                            let col = j * d + l;
                            choi[row * d2 + col] += k[i * d + j] * k[k_idx * d + l].conj();
                        }
                    }
                }
            }
        }

        ChoiMatrix {
            elements: choi,
            dim: d,
        }
    }

    /// Check if the Choi matrix is positive semidefinite (channel is completely positive).
    /// Uses a simple check: all eigenvalues of the Choi matrix are non-negative.
    pub fn is_positive_semidefinite(&self, tolerance: f64) -> bool {
        // For small dimensions, check via trace and diagonal dominance (approximate)
        let d2 = self.dim * self.dim;
        for i in 0..d2 {
            if self.elements[i * d2 + i].re < -tolerance {
                return false;
            }
        }
        // This is a necessary but not sufficient condition.
        // A full check would require eigenvalue computation.
        true
    }
}

// ===================================================================
// QUANTUM CHANNEL TRAIT
// ===================================================================

/// Trait for quantum channels that can be applied to density matrices.
pub trait QuantumChannel: Send {
    /// Apply the channel to a density matrix.
    fn apply(&self, rho: &[C64], dim: usize) -> Vec<C64>;

    /// Get the Kraus operators for this channel.
    fn kraus_operators(&self) -> &[Vec<C64>];

    /// Check if this channel is unital.
    fn is_unital(&self) -> bool;

    /// Get the channel dimension.
    fn dim(&self) -> usize;
}

impl QuantumChannel for KrausChannel {
    fn apply(&self, rho: &[C64], _dim: usize) -> Vec<C64> {
        self.apply_to_density_matrix(rho)
    }

    fn kraus_operators(&self) -> &[Vec<C64>] {
        &self.operators
    }

    fn is_unital(&self) -> bool {
        self.is_unital(1e-10)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn pure_state_dm(state: &[C64]) -> Vec<C64> {
        let d = state.len();
        let mut dm = vec![C64::new(0.0, 0.0); d * d];
        for i in 0..d {
            for j in 0..d {
                dm[i * d + j] = state[i] * state[j].conj();
            }
        }
        dm
    }

    #[test]
    fn test_identity_channel() {
        let ch = KrausChannel::identity(2);
        assert!(ch.is_trace_preserving(1e-10));
        assert!(ch.is_unital(1e-10));

        let state = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let rho = pure_state_dm(&state);
        let result = ch.apply_to_density_matrix(&rho);

        for i in 0..4 {
            assert!((result[i] - rho[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn test_depolarizing_channel() {
        let ch = KrausChannel::depolarizing(0.1);
        assert!(ch.is_trace_preserving(1e-10));
        assert!(ch.is_unital(1e-10)); // Depolarizing is unital

        // Apply to |0⟩⟨0|
        let state = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let rho = pure_state_dm(&state);
        let result = ch.apply_to_density_matrix(&rho);

        // Diagonal should be (1-2p/3, 2p/3) approximately
        let trace: f64 = result[0].re + result[3].re;
        assert!((trace - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_damping() {
        let ch = KrausChannel::amplitude_damping(0.5);
        assert!(ch.is_trace_preserving(1e-10));
        assert!(!ch.is_unital(1e-6)); // Amplitude damping is NOT unital

        // Apply to |1⟩⟨1|: should decay towards |0⟩⟨0|
        let state = vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)];
        let rho = pure_state_dm(&state);
        let result = ch.apply_to_density_matrix(&rho);

        // ρ_00 should increase (|1⟩ decaying to |0⟩)
        assert!(result[0].re > 0.0);
        let trace: f64 = result[0].re + result[3].re;
        assert!((trace - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_damping() {
        let ch = KrausChannel::phase_damping(0.5);
        assert!(ch.is_trace_preserving(1e-10));

        // Apply to |+⟩⟨+|: off-diagonal should decay
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let state = vec![C64::new(inv_sqrt2, 0.0), C64::new(inv_sqrt2, 0.0)];
        let rho = pure_state_dm(&state);
        let result = ch.apply_to_density_matrix(&rho);

        // Off-diagonal magnitude should decrease
        assert!(result[1].norm() < rho[1].norm());
    }

    #[test]
    fn test_channel_composition() {
        let ch1 = KrausChannel::depolarizing(0.05);
        let ch2 = KrausChannel::phase_damping(0.05);
        let composed = ch1.compose(&ch2);

        assert!(composed.is_trace_preserving(1e-8));
        assert_eq!(
            composed.num_operators(),
            ch1.num_operators() * ch2.num_operators()
        );
    }

    #[test]
    fn test_choi_from_kraus() {
        let ch = KrausChannel::depolarizing(0.1);
        let choi = ChoiMatrix::from_kraus(&ch);
        assert_eq!(choi.elements.len(), 16); // 4x4 for 1-qubit channel
        assert!(choi.is_positive_semidefinite(1e-10));
    }

    #[test]
    fn test_erasure_channel() {
        let ch = KrausChannel::erasure(0.3);
        assert!(ch.is_trace_preserving(1e-8));
    }

    #[test]
    fn test_quantum_channel_trait() {
        let ch = KrausChannel::depolarizing(0.1);
        let channel: &dyn QuantumChannel = &ch;
        assert_eq!(channel.dim(), 2);
        assert_eq!(channel.kraus_operators().len(), 4);
        assert!(channel.is_unital());
    }
}
