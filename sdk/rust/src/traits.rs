//! Unified Trait Hierarchy for All Simulation Backends
//!
//! Provides a common interface (`QuantumBackend`) that all simulators implement,
//! enabling polymorphic dispatch, auto-routing, and backend-agnostic algorithms.

use crate::gates::Gate;
use crate::C64;
use std::collections::HashMap;
use std::fmt;

// ===================================================================
// BACKEND ERROR
// ===================================================================

/// Errors that can occur during backend simulation.
#[derive(Clone, Debug)]
pub enum BackendError {
    /// The gate type is not supported by this backend.
    UnsupportedGate(String),
    /// A qubit index is out of range.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// Internal simulation error.
    Internal(String),
    /// Backend not available (e.g. GPU not present).
    Unavailable(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::UnsupportedGate(g) => write!(f, "unsupported gate: {}", g),
            BackendError::QubitOutOfRange { qubit, num_qubits } => {
                write!(f, "qubit {} out of range ({})", qubit, num_qubits)
            }
            BackendError::Internal(msg) => write!(f, "internal error: {}", msg),
            BackendError::Unavailable(msg) => write!(f, "backend unavailable: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Result type for backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

// ===================================================================
// QUANTUM BACKEND TRAIT
// ===================================================================

/// Unified interface for all quantum simulation backends.
///
/// Every simulator (state-vector, MPS, stabilizer, density matrix, etc.)
/// implements this trait so that algorithms can be written generically.
pub trait QuantumBackend: Send {
    /// Return the number of qubits this backend is configured for.
    fn num_qubits(&self) -> usize;

    /// Apply a single gate to the state.
    fn apply_gate(&mut self, gate: &Gate) -> BackendResult<()>;

    /// Get the probability distribution over computational basis states.
    ///
    /// Returns a vector of length 2^n (or an approximation for backends
    /// that cannot produce the full distribution efficiently).
    fn probabilities(&self) -> BackendResult<Vec<f64>>;

    /// Sample `n_shots` measurement outcomes from the current state.
    ///
    /// Returns a histogram mapping outcome index to count.
    fn sample(&self, n_shots: usize) -> BackendResult<HashMap<usize, usize>>;

    /// Measure a single qubit, collapsing the state. Returns the outcome (0 or 1).
    fn measure_qubit(&mut self, qubit: usize) -> BackendResult<u8>;

    /// Reset the backend to the |0...0> state.
    fn reset(&mut self);

    /// Human-readable name for this backend.
    fn name(&self) -> &str;

    /// Apply a sequence of gates.
    fn apply_circuit(&mut self, gates: &[Gate]) -> BackendResult<()> {
        for gate in gates {
            self.apply_gate(gate)?;
        }
        Ok(())
    }

    /// Expectation value of a Pauli-Z operator on a single qubit.
    /// Default implementation: compute from probabilities.
    fn expectation_z(&self, qubit: usize) -> BackendResult<f64> {
        let probs = self.probabilities()?;
        let n = self.num_qubits();
        if qubit >= n {
            return Err(BackendError::QubitOutOfRange {
                qubit,
                num_qubits: n,
            });
        }
        let stride = 1 << qubit;
        let mut exp = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            if i & stride == 0 {
                exp += p;
            } else {
                exp -= p;
            }
        }
        Ok(exp)
    }
}

// ===================================================================
// ERROR MODEL TRAIT
// ===================================================================

/// Trait for noise / error models that can be layered onto backends.
pub trait ErrorModel: Send {
    /// Apply noise after a gate has been executed.
    fn apply_noise_after_gate(
        &self,
        gate: &Gate,
        state: &mut dyn QuantumBackend,
    ) -> BackendResult<()>;

    /// Apply idle noise to a qubit that was not operated on this cycle.
    fn apply_idle_noise(&self, qubit: usize, state: &mut dyn QuantumBackend)
        -> BackendResult<()>;

    /// Return the gate error rate for a given gate type.
    fn gate_error_rate(&self, gate: &Gate) -> f64;
}

// ===================================================================
// TENSOR CONTRACTOR TRAIT
// ===================================================================

/// Abstraction for tensor contraction engines (nalgebra, AMX, GPU, etc.).
pub trait TensorContractor: Send {
    /// Contract two matrices: C = A * B.
    fn contract(&self, a: &[C64], b: &[C64], m: usize, k: usize, n: usize) -> Vec<C64>;

    /// Compute the SVD of a matrix, returning (U, S, Vt).
    /// The matrix is `m × n` stored in row-major order.
    fn decompose_svd(
        &self,
        matrix: &[C64],
        m: usize,
        n: usize,
    ) -> (Vec<C64>, Vec<f64>, Vec<C64>);

    /// Truncate an SVD to keep at most `max_rank` singular values.
    /// Returns (U_trunc, S_trunc, Vt_trunc, truncation_error).
    fn truncate(
        &self,
        u: &[C64],
        s: &[f64],
        vt: &[C64],
        m: usize,
        n: usize,
        max_rank: usize,
    ) -> (Vec<C64>, Vec<f64>, Vec<C64>, f64) {
        let rank = s.len().min(max_rank);
        let trunc_error: f64 = s[rank..].iter().map(|x| x * x).sum::<f64>().sqrt();

        // Truncate U: m × rank
        let mut u_trunc = vec![C64::new(0.0, 0.0); m * rank];
        for i in 0..m {
            for j in 0..rank {
                u_trunc[i * rank + j] = u[i * s.len() + j];
            }
        }

        let s_trunc = s[..rank].to_vec();

        // Truncate Vt: rank × n
        let mut vt_trunc = vec![C64::new(0.0, 0.0); rank * n];
        for i in 0..rank {
            for j in 0..n {
                vt_trunc[i * n + j] = vt[i * n + j];
            }
        }

        (u_trunc, s_trunc, vt_trunc, trunc_error)
    }
}

// ===================================================================
// FERMION MAPPING TRAIT
// ===================================================================

/// Trait for mapping fermionic operators to qubit operators.
pub trait FermionMapping: Send {
    /// Number of qubits required for this mapping of `n_modes` fermionic modes.
    fn num_qubits(&self, n_modes: usize) -> usize;

    /// Return the qubit representation of the creation operator a†_j
    /// as a list of (coefficient, pauli_string) where pauli_string is
    /// a vector of (qubit_index, pauli_char).
    fn creation_operator(&self, mode: usize, n_modes: usize) -> Vec<(C64, Vec<(usize, char)>)>;

    /// Return the qubit representation of the annihilation operator a_j.
    fn annihilation_operator(
        &self,
        mode: usize,
        n_modes: usize,
    ) -> Vec<(C64, Vec<(usize, char)>)>;

    /// Return the qubit representation of the number operator n_j = a†_j a_j.
    fn number_operator(&self, mode: usize, n_modes: usize) -> Vec<(f64, Vec<(usize, char)>)>;
}

// ===================================================================
// QEC DECODER TRAITS
// ===================================================================

/// Generic syndrome type for QEC decoders.
///
/// Different codes use different syndrome representations, but all can be
/// converted to a common bit vector form for unified processing.
pub trait Syndrome {
    /// Convert to a flat bit vector.
    fn to_bits(&self) -> Vec<bool>;

    /// Number of syndrome bits.
    fn len(&self) -> usize;

    /// Check if syndrome is empty (no errors detected).
    fn is_empty(&self) -> bool {
        self.to_bits().iter().all(|&b| !b)
    }
}

/// Generic correction type for QEC decoders.
///
/// A correction represents the recovery operation to apply after decoding.
pub trait Correction {
    /// Convert to a list of qubit indices to apply X corrections.
    fn x_corrections(&self) -> Vec<usize>;

    /// Convert to a list of qubit indices to apply Z corrections.
    fn z_corrections(&self) -> Vec<usize>;

    /// Check if correction is trivial (no operations needed).
    fn is_trivial(&self) -> bool {
        self.x_corrections().is_empty() && self.z_corrections().is_empty()
    }
}

/// Unified trait for QEC decoders.
///
/// All decoders (MWPM, neural, BP, etc.) implement this trait for
/// polymorphic decoding across different code families.
///
/// # Type Parameters
/// - `S`: Syndrome type (must implement `Syndrome` trait)
/// - `C`: Correction type (must implement `Correction` trait)
pub trait QECDecoder<S: Syndrome, C: Correction>: Send {
    /// Decode a syndrome to produce a correction.
    fn decode(&self, syndrome: &S) -> C;

    /// Decode multiple syndromes in batch (for parallel processing).
    fn decode_batch(&self, syndromes: &[S]) -> Vec<C> {
        syndromes.iter().map(|s| self.decode(s)).collect()
    }

    /// Human-readable name for this decoder.
    fn name(&self) -> &str;
}

/// Trait for decoders that can be trained (neural, ML-based).
pub trait TrainableDecoder<S: Syndrome, C: Correction>: QECDecoder<S, C> {
    /// Training data type.
    type TrainingData;

    /// Train the decoder on a dataset.
    fn train(&mut self, data: &Self::TrainingData, epochs: usize) -> TrainingHistory;

    /// Get the current training loss.
    fn loss(&self) -> f64;
}

/// Training history for tracking convergence.
#[derive(Clone, Debug, Default)]
pub struct TrainingHistory {
    /// Loss at each epoch.
    pub epoch_losses: Vec<f64>,
    /// Accuracy at each epoch (if applicable).
    pub epoch_accuracy: Vec<f64>,
    /// Training duration in seconds.
    pub duration_secs: f64,
}

/// Trait for decoders that support soft decisions (probabilistic outputs).
pub trait SoftDecoder<S: Syndrome>: Send {
    /// Decode with soft output (probability of each correction).
    fn decode_soft(&self, syndrome: &S) -> Vec<(usize, f64)>;
}

// ===================================================================
// ERROR MITIGATION TRAITS
// ===================================================================

/// Unified trait for error mitigation techniques.
///
/// All error mitigation methods (ZNE, PEC, CDR, virtual distillation, etc.)
/// implement this trait for polymorphic mitigation.
pub trait ErrorMitigator: Send {
    /// Mitigation method name.
    fn name(&self) -> &str;

    /// Number of extra circuit evaluations required.
    /// Higher values indicate more expensive mitigation.
    fn overhead(&self) -> usize;

    /// Check if mitigation is applicable to the given circuit.
    fn is_applicable(&self, num_qubits: usize, gate_count: usize) -> bool;
}

/// Trait for zero-noise extrapolation methods.
pub trait ZNEMitigator: ErrorMitigator {
    /// Extrapolate to zero noise from measurements at different noise scales.
    ///
    /// # Arguments
    /// * `measurements` - (scale_factor, expectation_value) pairs
    ///
    /// # Returns
    /// Extrapolated expectation value at zero noise.
    fn extrapolate(&self, measurements: &[(f64, f64)]) -> f64;

    /// Get the noise scale factors to use.
    fn scale_factors(&self) -> Vec<f64>;
}

/// Trait for probabilistic error cancellation.
pub trait PECDiscriminator: ErrorMitigator {
    /// Generate the quasiprobability decomposition for a noisy operation.
    fn decompose(&self, error_rate: f64) -> Vec<(f64, String)>;
}

// ===================================================================
// STREAMING DECODER TRAIT
// ===================================================================

/// Trait for streaming QEC decoders that process syndrome rounds incrementally.
///
/// Unlike [`QECDecoder`] which takes a single syndrome and returns a single
/// correction (batch/one-shot model), streaming decoders maintain internal
/// state and process data as it arrives — essential for real-time fault-tolerant
/// quantum computing where syndrome data arrives every code cycle (~1 μs).
pub trait StreamingDecoder: Send {
    /// The result type produced by one decoding window.
    type WindowResult;

    /// Push a new syndrome round into the decoder's buffer.
    fn push_syndrome(&mut self, round_id: usize, syndrome: Vec<bool>, timestamp: f64);

    /// Check whether enough rounds have been buffered to decode a window.
    fn ready(&self) -> bool;

    /// Decode the current window and commit the oldest rounds.
    ///
    /// Panics or returns an error if `ready()` is false.
    fn decode_window(&mut self) -> Self::WindowResult;

    /// Flush all remaining buffered rounds (final decoding at end of experiment).
    fn flush(&mut self) -> Vec<Self::WindowResult>;

    /// Human-readable name for this decoder.
    fn name(&self) -> &str;
}

// ===================================================================
// DEFAULT IMPLEMENTATIONS
// ===================================================================

/// A default tensor contractor using nalgebra.
pub struct NalgebraTensorContractor;

impl TensorContractor for NalgebraTensorContractor {
    fn contract(&self, a: &[C64], b: &[C64], m: usize, k: usize, n: usize) -> Vec<C64> {
        let mut c = vec![C64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = C64::new(0.0, 0.0);
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn decompose_svd(
        &self,
        matrix: &[C64],
        m: usize,
        n: usize,
    ) -> (Vec<C64>, Vec<f64>, Vec<C64>) {
        use nalgebra::{DMatrix, Complex};

        let mat = DMatrix::from_fn(m, n, |i, j| {
            let c = matrix[i * n + j];
            Complex::new(c.re, c.im)
        });

        let svd = mat.svd(true, true);
        let u_mat = svd.u.unwrap();
        let vt_mat = svd.v_t.unwrap();
        let rank = svd.singular_values.len();

        let mut u = Vec::with_capacity(m * rank);
        for i in 0..m {
            for j in 0..rank {
                u.push(C64::new(u_mat[(i, j)].re, u_mat[(i, j)].im));
            }
        }

        let s: Vec<f64> = svd.singular_values.iter().copied().collect();

        let mut vt = Vec::with_capacity(rank * n);
        for i in 0..rank {
            for j in 0..n {
                vt.push(C64::new(vt_mat[(i, j)].re, vt_mat[(i, j)].im));
            }
        }

        (u, s, vt)
    }
}

// ===================================================================
// BACKEND IMPLEMENTATIONS FOR EXISTING SIMULATORS
// ===================================================================

/// Implementation of QuantumBackend for the main QuantumState + GateOperations.
pub struct StateVectorBackend {
    state: crate::QuantumState,
}

impl StateVectorBackend {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            state: crate::QuantumState::new(num_qubits),
        }
    }

    /// Get a reference to the underlying state.
    pub fn state(&self) -> &crate::QuantumState {
        &self.state
    }

    /// Get a mutable reference to the underlying state.
    pub fn state_mut(&mut self) -> &mut crate::QuantumState {
        &mut self.state
    }
}

impl QuantumBackend for StateVectorBackend {
    fn num_qubits(&self) -> usize {
        self.state.num_qubits
    }

    fn apply_gate(&mut self, gate: &Gate) -> BackendResult<()> {
        crate::ascii_viz::apply_gate_to_state(&mut self.state, gate);
        Ok(())
    }

    fn probabilities(&self) -> BackendResult<Vec<f64>> {
        Ok(self.state.probabilities())
    }

    fn sample(&self, n_shots: usize) -> BackendResult<HashMap<usize, usize>> {
        Ok(self.state.sample(n_shots))
    }

    fn measure_qubit(&mut self, qubit: usize) -> BackendResult<u8> {
        let (outcome, _) = self.state.measure();
        Ok(((outcome >> qubit) & 1) as u8)
    }

    fn reset(&mut self) {
        self.state = crate::QuantumState::new(self.state.num_qubits);
    }

    fn name(&self) -> &str {
        "StateVector"
    }

    fn expectation_z(&self, qubit: usize) -> BackendResult<f64> {
        Ok(self.state.expectation_z(qubit))
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};

    #[test]
    fn test_state_vector_backend_basic() {
        let mut backend = StateVectorBackend::new(2);
        assert_eq!(backend.num_qubits(), 2);
        assert_eq!(backend.name(), "StateVector");

        // Apply H to qubit 0
        let h_gate = Gate::single(GateType::H, 0);
        backend.apply_gate(&h_gate).unwrap();

        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_state_vector_backend_bell_state() {
        let mut backend = StateVectorBackend::new(2);

        // Create Bell state: H on q0, CNOT(q0, q1)
        backend
            .apply_gate(&Gate::single(GateType::H, 0))
            .unwrap();
        backend
            .apply_gate(&Gate::two(GateType::CNOT, 0, 1))
            .unwrap();

        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00>
        assert!(probs[1].abs() < 1e-10); // |01>
        assert!(probs[2].abs() < 1e-10); // |10>
        assert!((probs[3] - 0.5).abs() < 1e-10); // |11>
    }

    #[test]
    fn test_state_vector_backend_sample() {
        let mut backend = StateVectorBackend::new(1);
        backend
            .apply_gate(&Gate::single(GateType::X, 0))
            .unwrap();

        let counts = backend.sample(100).unwrap();
        // Should always measure |1>
        assert_eq!(*counts.get(&1).unwrap_or(&0), 100);
    }

    #[test]
    fn test_state_vector_backend_reset() {
        let mut backend = StateVectorBackend::new(2);
        backend
            .apply_gate(&Gate::single(GateType::X, 0))
            .unwrap();
        backend.reset();

        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_vector_backend_circuit() {
        let mut backend = StateVectorBackend::new(2);
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::two(GateType::CNOT, 0, 1),
        ];
        backend.apply_circuit(&circuit).unwrap();

        let probs = backend.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_nalgebra_contractor_basic() {
        let contractor = NalgebraTensorContractor;

        // 2x2 identity times [1, 0]
        let a = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ];
        let b = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let c = contractor.contract(&a, &b, 2, 2, 1);
        assert!((c[0].re - 1.0).abs() < 1e-10);
        assert!(c[1].re.abs() < 1e-10);
    }

    #[test]
    fn test_nalgebra_svd() {
        let contractor = NalgebraTensorContractor;

        // Simple 2x2 matrix
        let mat = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(2.0, 0.0),
        ];
        let (u, s, vt) = contractor.decompose_svd(&mat, 2, 2);
        assert_eq!(s.len(), 2);
        // Singular values should be 2.0 and 1.0 (sorted descending)
        let mut s_sorted = s.clone();
        s_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((s_sorted[0] - 2.0).abs() < 1e-10);
        assert!((s_sorted[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_backend_error_display() {
        let e = BackendError::UnsupportedGate("Toffoli".to_string());
        assert!(e.to_string().contains("Toffoli"));

        let e = BackendError::QubitOutOfRange {
            qubit: 5,
            num_qubits: 3,
        };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_truncation() {
        let contractor = NalgebraTensorContractor;
        let s = vec![3.0, 2.0, 1.0, 0.5, 0.1];
        let m = 5;
        let n = 5;
        let u = vec![C64::new(1.0, 0.0); m * s.len()];
        let vt = vec![C64::new(1.0, 0.0); s.len() * n];

        let (u_t, s_t, vt_t, err) = contractor.truncate(&u, &s, &vt, m, n, 3);
        assert_eq!(s_t.len(), 3);
        assert_eq!(u_t.len(), m * 3);
        assert_eq!(vt_t.len(), 3 * n);
        // Truncation error = sqrt(0.5^2 + 0.1^2)
        let expected_err = (0.25 + 0.01f64).sqrt();
        assert!((err - expected_err).abs() < 1e-10);
    }
}
