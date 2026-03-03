//! Advanced Error Mitigation Techniques
//!
//! Implements four complementary error mitigation strategies for noisy quantum
//! simulation, going beyond the basic ZNE and readout mitigation in
//! `crate::error_mitigation`:
//!
//! 1. **Probabilistic Error Cancellation (PEC)** -- decomposes the inverse of a
//!    noisy channel into a quasi-probability distribution over implementable Pauli
//!    corrections, then estimates noiseless expectation values via Monte Carlo
//!    sampling.
//!
//! 2. **Clifford Data Regression (CDR)** -- trains a linear model on near-Clifford
//!    circuits where both ideal and noisy values are cheaply computable, then
//!    applies the learned correction to arbitrary noisy results.
//!
//! 3. **Symmetry Verification** -- post-selects measurement outcomes that satisfy
//!    known symmetries (Z2 parity, particle conservation, custom predicates),
//!    discarding outcomes corrupted by errors.
//!
//! 4. **Virtual Distillation** -- uses M copies of a noisy state to estimate
//!    Tr(rho^M . O) / Tr(rho^M), exponentially suppressing incoherent errors
//!    in expectation values.

use crate::gates::{Gate, GateType};
use crate::{GateOperations, QuantumState, C64};
use std::collections::HashMap;

// ===================================================================
// PROBABILISTIC ERROR CANCELLATION (PEC)
// ===================================================================

/// A single correction operation in the PEC decomposition.
///
/// After executing the noisy gate, a correction gate is sampled from the
/// quasi-probability distribution and applied.  The sign of the coefficient
/// is tracked for importance-weighted averaging.
#[derive(Clone, Debug)]
pub struct PECOperation {
    /// Correction gate to apply after the noisy gate.
    pub correction: GateType,
    /// Quasi-probability coefficient (can be negative).
    pub coefficient: f64,
    /// Target qubit for the correction.
    pub qubit: usize,
}

/// Probabilistic Error Cancellation (PEC) decomposition.
///
/// Represents the inverse of a noisy channel as a quasi-probability
/// distribution over implementable Pauli corrections:
///
///   E^{-1} = sum_i q_i O_i
///
/// where O_i are Pauli gates (I, X, Y, Z) applied after each noisy gate,
/// and q_i are real coefficients that may be negative.  The one-norm
/// `gamma = sum_i |q_i|` determines the sampling overhead: O(gamma^{2L})
/// samples are needed for L gates.
#[derive(Clone, Debug)]
pub struct PECDecomposition {
    /// Operations in the quasi-probability distribution.
    pub operations: Vec<PECOperation>,
    /// One-norm of the quasi-probability distribution (sampling overhead).
    pub gamma: f64,
}

impl PECDecomposition {
    /// Decompose the inverse of a single-qubit depolarizing channel.
    ///
    /// For a depolarizing channel E(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ),
    /// the inverse channel is a Pauli channel:
    ///
    ///   E^{-1}(rho) = alpha * rho + beta * (X rho X + Y rho Y + Z rho Z)
    ///
    /// Derived from the Pauli transfer matrix (PTM) inverse:
    ///   PTM eigenvalue for Pauli components: lambda = 1 - 4p/3
    ///   Inverse eigenvalue: eta = 1 / (1 - 4p/3)
    ///   alpha = (1 + 3*eta) / 4
    ///   beta  = (1 - eta) / 4
    ///
    /// The coefficients satisfy alpha + 3*beta = 1 (trace-preserving).
    /// The one-norm gamma = |alpha| + 3*|beta| determines the sampling overhead.
    ///
    /// # Panics
    ///
    /// Panics if `error_rate >= 0.75` (channel becomes non-invertible).
    pub fn from_depolarizing(error_rate: f64, qubit: usize) -> Self {
        assert!(
            error_rate < 0.75,
            "Depolarizing error rate must be < 0.75 for invertibility"
        );
        assert!(error_rate >= 0.0, "Error rate must be non-negative");

        // For depolarizing channel parametrized as:
        //   E(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
        //
        // PTM eigenvalue for Pauli components: lambda = 1 - 4p/3
        // Inverse eigenvalue: eta = 1/lambda
        //
        // The inverse channel as a Pauli channel:
        //   E^{-1}(rho) = alpha * rho + beta * (X rho X + Y rho Y + Z rho Z)
        //
        // with constraints:
        //   alpha + 3*beta = 1          (trace-preserving)
        //   alpha - beta   = eta        (PTM inverse eigenvalue)
        //
        // Solution:
        //   alpha = (1 + 3*eta) / 4
        //   beta  = (1 - eta) / 4

        let p = error_rate;
        let lambda = 1.0 - 4.0 * p / 3.0;
        let eta = 1.0 / lambda;
        let alpha = (1.0 + 3.0 * eta) / 4.0;
        let beta = (1.0 - eta) / 4.0;

        let gamma = alpha.abs() + 3.0 * beta.abs();

        let operations = vec![
            PECOperation {
                correction: GateType::Rz(0.0), // Identity (Rz(0) = I)
                coefficient: alpha,
                qubit,
            },
            PECOperation {
                correction: GateType::X,
                coefficient: beta,
                qubit,
            },
            PECOperation {
                correction: GateType::Y,
                coefficient: beta,
                qubit,
            },
            PECOperation {
                correction: GateType::Z,
                coefficient: beta,
                qubit,
            },
        ];

        PECDecomposition { operations, gamma }
    }

    /// Randomly sample a correction operation from the quasi-probability
    /// distribution.
    ///
    /// Returns `(gate_type, sign)` where `sign` is +1.0 or -1.0, used for
    /// importance-weighted averaging.  The gate is sampled with probability
    /// proportional to `|coefficient|`, and the sign carries the original sign.
    pub fn sample_correction(&self) -> (GateType, f64) {
        let r: f64 = rand::random::<f64>() * self.gamma;
        let mut cumulative = 0.0;

        for op in &self.operations {
            cumulative += op.coefficient.abs();
            if r <= cumulative {
                let sign = op.coefficient.signum();
                return (op.correction.clone(), sign);
            }
        }

        // Fallback to last operation (rounding guard).
        let last = self.operations.last().unwrap();
        (last.correction.clone(), last.coefficient.signum())
    }

    /// Full PEC protocol for estimating <Z_q> on a given circuit.
    ///
    /// For each sample:
    /// 1. Execute the original (noisy) circuit on a fresh |0...0> state.
    /// 2. After each gate, sample a Pauli correction from the PEC decomposition
    ///    and apply it, accumulating the sign.
    /// 3. Measure <Z> on the observable qubit, weighted by gamma^L * sign_product.
    ///
    /// The average over all samples converges to the ideal (noiseless) expectation.
    ///
    /// # Arguments
    ///
    /// * `circuit`          - Sequence of gates defining the circuit.
    /// * `observable_qubit` - Qubit on which to estimate <Z>.
    /// * `error_rate`       - Per-gate depolarizing error rate.
    /// * `num_samples`      - Number of Monte Carlo samples.
    pub fn apply_pec_expectation(
        circuit: &[Gate],
        observable_qubit: usize,
        error_rate: f64,
        num_samples: usize,
    ) -> f64 {
        if num_samples == 0 {
            return 0.0;
        }

        // Determine the number of qubits from the circuit.
        let num_qubits = circuit
            .iter()
            .flat_map(|g| g.targets.iter().chain(g.controls.iter()))
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        // Build per-qubit PEC decompositions.
        let decompositions: Vec<PECDecomposition> = (0..num_qubits)
            .map(|q| PECDecomposition::from_depolarizing(error_rate, q))
            .collect();

        let gamma_per_gate = decompositions[0].gamma;
        let num_gates = circuit.len();
        let gamma_total = gamma_per_gate.powi(num_gates as i32);

        let mut sum = 0.0;

        for _ in 0..num_samples {
            let mut state = QuantumState::new(num_qubits);
            let mut sign_product = 1.0;

            for gate in circuit {
                // Apply the original gate.
                apply_gate_to_state(&mut state, gate);

                // Sample a PEC correction for the first target qubit.
                let target = gate.targets[0];
                let decomp = &decompositions[target];
                let (correction, sign) = decomp.sample_correction();
                sign_product *= sign;

                // Apply the correction gate.
                apply_gate_type(&mut state, &correction, target);
            }

            let expectation = state.expectation_z(observable_qubit);
            sum += gamma_total * sign_product * expectation;
        }

        sum / num_samples as f64
    }
}

// ===================================================================
// CLIFFORD DATA REGRESSION (CDR)
// ===================================================================

/// A single training data point for CDR.
#[derive(Clone, Debug)]
pub struct CDRTrainingPoint {
    /// Ideal (classically computed) expectation value.
    pub ideal_value: f64,
    /// Noisy (hardware or noisy-simulation) expectation value.
    pub noisy_value: f64,
}

/// Clifford Data Regression (CDR) model.
///
/// Learns a linear correction `y_mitigated = slope * y_noisy + intercept` from
/// near-Clifford training circuits where both ideal and noisy values are
/// available, then applies this correction to arbitrary noisy results.
#[derive(Clone, Debug)]
pub struct CDRModel {
    /// Learned linear slope:  y_mit = slope * y_noisy + intercept.
    pub slope: f64,
    /// Learned linear intercept.
    pub intercept: f64,
    /// Training data points used to fit the model.
    pub training_data: Vec<CDRTrainingPoint>,
    /// Number of training circuits used.
    pub num_training_circuits: usize,
}

impl CDRModel {
    /// Fit a linear CDR model from paired ideal/noisy values.
    ///
    /// Uses ordinary least-squares regression:
    ///   ideal = slope * noisy + intercept.
    ///
    /// # Panics
    ///
    /// Panics if `ideal_values` and `noisy_values` have different lengths or
    /// are empty.
    pub fn train(ideal_values: &[f64], noisy_values: &[f64]) -> Self {
        assert_eq!(
            ideal_values.len(),
            noisy_values.len(),
            "Training arrays must have equal length"
        );
        assert!(!ideal_values.is_empty(), "Training data must not be empty");

        let n = ideal_values.len() as f64;
        let sum_x: f64 = noisy_values.iter().sum();
        let sum_y: f64 = ideal_values.iter().sum();
        let sum_xx: f64 = noisy_values.iter().map(|x| x * x).sum();
        let sum_xy: f64 = noisy_values
            .iter()
            .zip(ideal_values.iter())
            .map(|(x, y)| x * y)
            .sum();

        let denom = n * sum_xx - sum_x * sum_x;
        let (slope, intercept) = if denom.abs() < 1e-15 {
            // All noisy values are identical -- fall back to mean correction.
            (1.0, (sum_y - sum_x) / n)
        } else {
            let a = (n * sum_xy - sum_x * sum_y) / denom;
            let b = (sum_y - a * sum_x) / n;
            (a, b)
        };

        let training_data: Vec<CDRTrainingPoint> = ideal_values
            .iter()
            .zip(noisy_values.iter())
            .map(|(&ideal_value, &noisy_value)| CDRTrainingPoint {
                ideal_value,
                noisy_value,
            })
            .collect();

        CDRModel {
            slope,
            intercept,
            training_data: training_data.clone(),
            num_training_circuits: training_data.len(),
        }
    }

    /// Apply the learned linear correction to a noisy observation.
    pub fn mitigate(&self, noisy_value: f64) -> f64 {
        self.slope * noisy_value + self.intercept
    }

    /// Coefficient of determination (R^2) measuring model quality.
    ///
    /// Returns a value in (-inf, 1].  A value of 1.0 means a perfect linear
    /// relationship between ideal and noisy values; values near 0 indicate
    /// that the linear model explains little variance.
    pub fn r_squared(&self) -> f64 {
        if self.training_data.is_empty() {
            return 0.0;
        }

        let n = self.training_data.len() as f64;
        let mean_ideal: f64 = self.training_data.iter().map(|p| p.ideal_value).sum::<f64>() / n;

        let ss_tot: f64 = self
            .training_data
            .iter()
            .map(|p| (p.ideal_value - mean_ideal).powi(2))
            .sum();

        if ss_tot < 1e-15 {
            return 1.0; // All ideal values identical -- trivially perfect.
        }

        let ss_res: f64 = self
            .training_data
            .iter()
            .map(|p| {
                let predicted = self.slope * p.noisy_value + self.intercept;
                (p.ideal_value - predicted).powi(2)
            })
            .sum();

        1.0 - ss_res / ss_tot
    }
}

// ===================================================================
// SYMMETRY VERIFICATION
// ===================================================================

/// A quantum symmetry constraint used for measurement post-selection.
pub enum Symmetry {
    /// Z2 parity symmetry: the parity of the specified qubits must equal
    /// `expected_parity` (0 = even, 1 = odd).
    Z2Parity {
        qubits: Vec<usize>,
        expected_parity: u8,
    },
    /// Particle number conservation: the total number of |1> among the
    /// specified qubits must equal `expected_particles`.
    ParticleConservation {
        qubits: Vec<usize>,
        expected_particles: usize,
    },
    /// Custom symmetry check function.
    ///
    /// Receives `(outcome, num_qubits)` and returns `true` if the outcome
    /// satisfies the symmetry.
    Custom {
        name: String,
        check_fn: Box<dyn Fn(usize, usize) -> bool + Send + Sync>,
    },
}

/// Symmetry verifier for post-selection of measurement outcomes.
///
/// Filters measurement results to keep only outcomes consistent with known
/// physical symmetries, discarding error-corrupted outcomes.
pub struct SymmetryVerifier {
    pub symmetries: Vec<Symmetry>,
}

impl SymmetryVerifier {
    /// Create an empty symmetry verifier.
    pub fn new() -> Self {
        SymmetryVerifier {
            symmetries: Vec::new(),
        }
    }

    /// Add a Z2 parity symmetry constraint.
    ///
    /// The parity (XOR) of the specified qubit values in the measurement
    /// outcome must equal `expected` (0 or 1).
    pub fn add_z2_parity(&mut self, qubits: Vec<usize>, expected: u8) -> &mut Self {
        self.symmetries.push(Symmetry::Z2Parity {
            qubits,
            expected_parity: expected,
        });
        self
    }

    /// Add a particle conservation constraint.
    ///
    /// The Hamming weight (number of 1-bits) among the specified qubits must
    /// equal `expected`.
    pub fn add_particle_conservation(
        &mut self,
        qubits: Vec<usize>,
        expected: usize,
    ) -> &mut Self {
        self.symmetries.push(Symmetry::ParticleConservation {
            qubits,
            expected_particles: expected,
        });
        self
    }

    /// Add a custom symmetry constraint.
    pub fn add_custom<F>(&mut self, name: &str, check_fn: F) -> &mut Self
    where
        F: Fn(usize, usize) -> bool + Send + Sync + 'static,
    {
        self.symmetries.push(Symmetry::Custom {
            name: name.to_string(),
            check_fn: Box::new(check_fn),
        });
        self
    }

    /// Check whether a single measurement outcome satisfies all symmetries.
    pub fn verify_outcome(&self, outcome: usize, num_qubits: usize) -> bool {
        for sym in &self.symmetries {
            match sym {
                Symmetry::Z2Parity {
                    qubits,
                    expected_parity,
                } => {
                    let mut parity: u8 = 0;
                    for &q in qubits {
                        parity ^= ((outcome >> q) & 1) as u8;
                    }
                    if parity != *expected_parity {
                        return false;
                    }
                }
                Symmetry::ParticleConservation {
                    qubits,
                    expected_particles,
                } => {
                    let count: usize = qubits.iter().map(|&q| (outcome >> q) & 1).sum();
                    if count != *expected_particles {
                        return false;
                    }
                }
                Symmetry::Custom { check_fn, .. } => {
                    if !check_fn(outcome, num_qubits) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Post-select measurement counts, keeping only outcomes that satisfy all
    /// symmetry constraints.
    pub fn filter_counts(
        &self,
        counts: &HashMap<usize, usize>,
        num_qubits: usize,
    ) -> HashMap<usize, usize> {
        counts
            .iter()
            .filter(|(&outcome, _)| self.verify_outcome(outcome, num_qubits))
            .map(|(&k, &v)| (k, v))
            .collect()
    }

    /// Compute the acceptance ratio: fraction of total counts that pass
    /// symmetry verification.
    pub fn acceptance_ratio(
        &self,
        counts: &HashMap<usize, usize>,
        num_qubits: usize,
    ) -> f64 {
        let total: usize = counts.values().sum();
        if total == 0 {
            return 0.0;
        }
        let accepted: usize = self
            .filter_counts(counts, num_qubits)
            .values()
            .sum();
        accepted as f64 / total as f64
    }
}

impl Default for SymmetryVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// VIRTUAL DISTILLATION
// ===================================================================

/// Virtual distillation for error suppression using M copies of a noisy state.
///
/// Estimates Tr(rho^M . O) / Tr(rho^M), which for a noisy state
/// rho = (1-eps)|psi><psi| + eps * sigma converges to the ideal <psi|O|psi>
/// as M increases, exponentially suppressing the incoherent error component.
pub struct VirtualDistillation {
    /// Number of state copies to use.
    pub num_copies: usize,
}

impl VirtualDistillation {
    /// Create a new virtual distillation estimator with `num_copies` copies.
    ///
    /// # Panics
    ///
    /// Panics if `num_copies < 2`.
    pub fn new(num_copies: usize) -> Self {
        assert!(
            num_copies >= 2,
            "Virtual distillation requires at least 2 copies"
        );
        VirtualDistillation { num_copies }
    }

    /// Estimate the mitigated expectation value <O> using virtual distillation.
    ///
    /// Computes Tr(rho^M . Z_q) / Tr(rho^M) by:
    /// 1. Constructing density matrices from the provided state vectors.
    /// 2. Computing rho^M via repeated matrix multiplication.
    /// 3. Taking the ratio of Tr(rho^M . Z_q) and Tr(rho^M).
    ///
    /// In practice, all `states` may be copies of the same noisy state; the
    /// math uses the average density matrix.
    ///
    /// # Arguments
    ///
    /// * `states` - Slice of references to QuantumState (one per copy).
    /// * `observable_qubit` - Qubit on which to estimate <Z>.
    pub fn estimate_expectation(
        &self,
        states: &[&QuantumState],
        observable_qubit: usize,
    ) -> f64 {
        assert!(
            !states.is_empty(),
            "At least one state must be provided"
        );

        let num_qubits = states[0].num_qubits;
        let dim = 1usize << num_qubits;

        // Construct the average density matrix from all provided copies.
        let mut rho = vec![C64::new(0.0, 0.0); dim * dim];
        for st in states {
            let amps = st.amplitudes_ref();
            for i in 0..dim {
                for j in 0..dim {
                    rho[i * dim + j] += amps[i] * amps[j].conj();
                }
            }
        }
        let inv_n = 1.0 / states.len() as f64;
        for elem in rho.iter_mut() {
            *elem = C64::new(elem.re * inv_n, elem.im * inv_n);
        }

        // Compute rho^M by repeated matrix multiplication.
        let mut rho_m = rho.clone();
        for _ in 1..self.num_copies {
            rho_m = matrix_multiply(&rho_m, &rho, dim);
        }

        // Tr(rho^M)
        let trace_rho_m: f64 = (0..dim).map(|i| rho_m[i * dim + i].re).sum();

        if trace_rho_m.abs() < 1e-15 {
            return 0.0;
        }

        // Construct Z_q operator and compute Tr(rho^M . Z_q).
        // Z_q is diagonal: +1 for |0> on qubit q, -1 for |1> on qubit q.
        let stride = 1usize << observable_qubit;
        let mut trace_rho_m_z: f64 = 0.0;
        for i in 0..dim {
            let z_eigenvalue = if i & stride == 0 { 1.0 } else { -1.0 };
            trace_rho_m_z += rho_m[i * dim + i].re * z_eigenvalue;
        }

        trace_rho_m_z / trace_rho_m
    }

    /// Estimate Tr(rho^M) directly -- useful for purity estimation (M=2).
    pub fn estimate_purity(&self, states: &[&QuantumState]) -> f64 {
        assert!(!states.is_empty());

        let num_qubits = states[0].num_qubits;
        let dim = 1usize << num_qubits;

        // Average density matrix.
        let mut rho = vec![C64::new(0.0, 0.0); dim * dim];
        for st in states {
            let amps = st.amplitudes_ref();
            for i in 0..dim {
                for j in 0..dim {
                    rho[i * dim + j] += amps[i] * amps[j].conj();
                }
            }
        }
        let inv_n = 1.0 / states.len() as f64;
        for elem in rho.iter_mut() {
            *elem = C64::new(elem.re * inv_n, elem.im * inv_n);
        }

        let mut rho_m = rho.clone();
        for _ in 1..self.num_copies {
            rho_m = matrix_multiply(&rho_m, &rho, dim);
        }

        (0..dim).map(|i| rho_m[i * dim + i].re).sum()
    }
}

// ===================================================================
// HELPER FUNCTIONS
// ===================================================================

/// Dense complex matrix multiplication: C = A * B, both dim x dim row-major.
fn matrix_multiply(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
    let mut c = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let a_ik = a[i * dim + k];
            if a_ik.re == 0.0 && a_ik.im == 0.0 {
                continue;
            }
            for j in 0..dim {
                c[i * dim + j] += a_ik * b[k * dim + j];
            }
        }
    }
    c
}

/// Apply a `Gate` to a `QuantumState` using the existing `GateOperations` API.
fn apply_gate_to_state(state: &mut QuantumState, gate: &Gate) {
    let target = gate.targets[0];
    match &gate.gate_type {
        GateType::H => GateOperations::h(state, target),
        GateType::X => GateOperations::x(state, target),
        GateType::Y => GateOperations::y(state, target),
        GateType::Z => GateOperations::z(state, target),
        GateType::CNOT => {
            let control = gate.controls[0];
            GateOperations::cnot(state, control, target);
        }
        GateType::Rx(theta) => GateOperations::rx(state, target, *theta),
        GateType::Ry(theta) => GateOperations::ry(state, target, *theta),
        GateType::Rz(theta) => GateOperations::rz(state, target, *theta),
        GateType::S => GateOperations::s(state, target),
        GateType::T => GateOperations::t(state, target),
        GateType::CZ => {
            let control = gate.controls[0];
            GateOperations::cz(state, control, target);
        }
        GateType::SWAP => {
            let target2 = gate.targets[1];
            GateOperations::swap(state, target, target2);
        }
        _ => {
            // Fallback: apply via the gate's unitary matrix.
            let matrix = gate.gate_type.matrix();
            apply_single_qubit_unitary(state, target, &matrix);
        }
    }
}

/// Apply a single `GateType` to a specific qubit on a `QuantumState`.
fn apply_gate_type(state: &mut QuantumState, gate_type: &GateType, qubit: usize) {
    match gate_type {
        GateType::X => GateOperations::x(state, qubit),
        GateType::Y => GateOperations::y(state, qubit),
        GateType::Z => GateOperations::z(state, qubit),
        GateType::H => GateOperations::h(state, qubit),
        GateType::Rz(theta) => {
            if theta.abs() < 1e-15 {
                // Identity -- no-op.
                return;
            }
            GateOperations::rz(state, qubit, *theta);
        }
        _ => {
            let matrix = gate_type.matrix();
            apply_single_qubit_unitary(state, qubit, &matrix);
        }
    }
}

/// Apply an arbitrary 2x2 unitary to a single qubit in the state vector.
fn apply_single_qubit_unitary(state: &mut QuantumState, qubit: usize, u: &[Vec<C64>]) {
    let stride = 1usize << qubit;
    let dim = state.dim;
    let amps = state.amplitudes_mut();

    let u00 = u[0][0];
    let u01 = u[0][1];
    let u10 = u[1][0];
    let u11 = u[1][1];

    let mut i = 0;
    while i < dim {
        for k in 0..stride {
            let idx0 = i + k;
            let idx1 = idx0 + stride;
            let a = amps[idx0];
            let b = amps[idx1];
            amps[idx0] = u00 * a + u01 * b;
            amps[idx1] = u10 * a + u11 * b;
        }
        i += stride << 1;
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};

    const TOLERANCE: f64 = 1e-6;

    // ----- PEC tests -----

    #[test]
    fn test_pec_gamma_depolarizing() {
        // For depolarizing channel E(rho) = (1-p)rho + (p/3)(X+Y+Z),
        // the inverse is a Pauli channel with coefficients derived from PTM inverse:
        //   eta = 1/(1 - 4p/3)
        //   alpha = (1 + 3*eta)/4,  beta = (1 - eta)/4
        //   gamma = |alpha| + 3*|beta|
        let p = 0.1;
        let decomp = PECDecomposition::from_depolarizing(p, 0);

        let lambda = 1.0 - 4.0 * p / 3.0;
        let eta = 1.0 / lambda;
        let alpha = (1.0 + 3.0 * eta) / 4.0;
        let beta = (1.0 - eta) / 4.0;
        let expected_gamma = alpha.abs() + 3.0 * beta.abs();

        assert!(
            (decomp.gamma - expected_gamma).abs() < TOLERANCE,
            "PEC gamma mismatch: got {}, expected {}",
            decomp.gamma,
            expected_gamma
        );
    }

    #[test]
    fn test_pec_gamma_zero_noise() {
        // At p=0, the inverse is the identity: gamma = 1.
        let decomp = PECDecomposition::from_depolarizing(0.0, 0);
        assert!(
            (decomp.gamma - 1.0).abs() < TOLERANCE,
            "PEC gamma at p=0 should be 1.0, got {}",
            decomp.gamma
        );
    }

    #[test]
    fn test_pec_coefficients_sum_to_one() {
        let p = 0.05;
        let decomp = PECDecomposition::from_depolarizing(p, 0);
        let sum: f64 = decomp.operations.iter().map(|op| op.coefficient).sum();
        assert!(
            (sum - 1.0).abs() < TOLERANCE,
            "PEC coefficients should sum to 1.0 (trace-preserving inverse), got {}",
            sum
        );
    }

    #[test]
    fn test_pec_sample_correction_returns_valid_gate() {
        let decomp = PECDecomposition::from_depolarizing(0.05, 0);
        for _ in 0..100 {
            let (gate, sign) = decomp.sample_correction();
            assert!(
                sign == 1.0 || sign == -1.0,
                "Sign must be +/- 1.0, got {}",
                sign
            );
            // Gate should be one of I (Rz(0)), X, Y, Z.
            match gate {
                GateType::Rz(_) | GateType::X | GateType::Y | GateType::Z => {}
                other => panic!("Unexpected correction gate: {:?}", other),
            }
        }
    }

    #[test]
    fn test_pec_expectation_identity_circuit() {
        // A circuit with no gates should return <Z> = 1.0 for |0>.
        let result = PECDecomposition::apply_pec_expectation(&[], 0, 0.01, 1000);
        // With empty circuit, state is |0>, so <Z> = 1.0.
        // PEC with no gates means no corrections, so gamma_total = 1.
        assert!(
            (result - 1.0).abs() < 0.1,
            "PEC on empty circuit should give <Z> ~ 1.0, got {}",
            result
        );
    }

    #[test]
    fn test_pec_expectation_x_gate() {
        // X|0> = |1>, so ideal <Z> = -1.
        // With very low noise, PEC should recover approximately -1.
        let circuit = vec![Gate::new(GateType::X, vec![0], vec![])];
        let result = PECDecomposition::apply_pec_expectation(&circuit, 0, 0.001, 5000);
        assert!(
            (result - (-1.0)).abs() < 0.15,
            "PEC on X gate should give <Z> ~ -1.0, got {}",
            result
        );
    }

    // ----- CDR tests -----

    #[test]
    fn test_cdr_perfect_linear() {
        // If ideal = 2 * noisy + 1, CDR should recover the exact relationship.
        let noisy = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ideal: Vec<f64> = noisy.iter().map(|x| 2.0 * x + 1.0).collect();

        let model = CDRModel::train(&ideal, &noisy);
        assert!(
            (model.slope - 2.0).abs() < TOLERANCE,
            "CDR slope should be 2.0, got {}",
            model.slope
        );
        assert!(
            (model.intercept - 1.0).abs() < TOLERANCE,
            "CDR intercept should be 1.0, got {}",
            model.intercept
        );
        assert!(
            (model.r_squared() - 1.0).abs() < TOLERANCE,
            "R^2 should be 1.0 for perfect linear data, got {}",
            model.r_squared()
        );
    }

    #[test]
    fn test_cdr_mitigation_reduces_error() {
        // Simulate a systematic bias: noisy = ideal * 0.8 (20% underestimate).
        let ideal = vec![1.0, 0.5, -0.3, -0.8, 0.0];
        let noisy: Vec<f64> = ideal.iter().map(|x| x * 0.8).collect();

        let model = CDRModel::train(&ideal, &noisy);

        // Mitigate a new noisy value.
        let test_noisy = 0.6 * 0.8; // True ideal = 0.6, noisy = 0.48
        let mitigated = model.mitigate(test_noisy);
        let error_before = (0.6 - test_noisy).abs();
        let error_after = (0.6 - mitigated).abs();

        assert!(
            error_after < error_before,
            "CDR should reduce error: before={}, after={}",
            error_before,
            error_after
        );
    }

    #[test]
    fn test_cdr_r_squared_range() {
        let ideal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let noisy = vec![1.1, 1.9, 3.2, 3.8, 5.1];
        let model = CDRModel::train(&ideal, &noisy);
        let r2 = model.r_squared();
        assert!(
            r2 > 0.9 && r2 <= 1.0,
            "R^2 should be high for near-linear data, got {}",
            r2
        );
    }

    #[test]
    fn test_cdr_training_data_stored() {
        let ideal = vec![1.0, 2.0, 3.0];
        let noisy = vec![0.9, 2.1, 2.8];
        let model = CDRModel::train(&ideal, &noisy);
        assert_eq!(model.num_training_circuits, 3);
        assert_eq!(model.training_data.len(), 3);
    }

    // ----- Symmetry verification tests -----

    #[test]
    fn test_z2_parity_even() {
        let mut verifier = SymmetryVerifier::new();
        verifier.add_z2_parity(vec![0, 1], 0); // Even parity on qubits 0,1.

        // outcome = 0b00 (0) -> parity = 0 -> pass
        assert!(verifier.verify_outcome(0b00, 2));
        // outcome = 0b11 (3) -> parity = 0 -> pass
        assert!(verifier.verify_outcome(0b11, 2));
        // outcome = 0b01 (1) -> parity = 1 -> fail
        assert!(!verifier.verify_outcome(0b01, 2));
        // outcome = 0b10 (2) -> parity = 1 -> fail
        assert!(!verifier.verify_outcome(0b10, 2));
    }

    #[test]
    fn test_z2_parity_odd() {
        let mut verifier = SymmetryVerifier::new();
        verifier.add_z2_parity(vec![0, 1], 1); // Odd parity.

        assert!(!verifier.verify_outcome(0b00, 2));
        assert!(!verifier.verify_outcome(0b11, 2));
        assert!(verifier.verify_outcome(0b01, 2));
        assert!(verifier.verify_outcome(0b10, 2));
    }

    #[test]
    fn test_particle_conservation() {
        let mut verifier = SymmetryVerifier::new();
        verifier.add_particle_conservation(vec![0, 1, 2], 2); // Exactly 2 particles.

        // 0b011 = 3 -> 2 particles -> pass
        assert!(verifier.verify_outcome(0b011, 3));
        // 0b101 = 5 -> 2 particles -> pass
        assert!(verifier.verify_outcome(0b101, 3));
        // 0b110 = 6 -> 2 particles -> pass
        assert!(verifier.verify_outcome(0b110, 3));
        // 0b111 = 7 -> 3 particles -> fail
        assert!(!verifier.verify_outcome(0b111, 3));
        // 0b001 = 1 -> 1 particle -> fail
        assert!(!verifier.verify_outcome(0b001, 3));
        // 0b000 = 0 -> 0 particles -> fail
        assert!(!verifier.verify_outcome(0b000, 3));
    }

    #[test]
    fn test_symmetry_filter_counts() {
        let mut verifier = SymmetryVerifier::new();
        verifier.add_z2_parity(vec![0, 1], 0); // Even parity.

        let mut counts = HashMap::new();
        counts.insert(0b00, 100); // Even -> keep
        counts.insert(0b01, 50); // Odd -> discard
        counts.insert(0b10, 30); // Odd -> discard
        counts.insert(0b11, 120); // Even -> keep

        let filtered = verifier.filter_counts(&counts, 2);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[&0b00], 100);
        assert_eq!(filtered[&0b11], 120);
    }

    #[test]
    fn test_symmetry_acceptance_ratio() {
        let mut verifier = SymmetryVerifier::new();
        verifier.add_particle_conservation(vec![0, 1], 1); // Exactly 1 particle in 2 qubits.

        let mut counts = HashMap::new();
        counts.insert(0b00, 25); // 0 particles -> discard
        counts.insert(0b01, 40); // 1 particle -> keep
        counts.insert(0b10, 35); // 1 particle -> keep
        counts.insert(0b11, 0); // 2 particles -> discard

        let ratio = verifier.acceptance_ratio(&counts, 2);
        // Accepted: 40 + 35 = 75 out of 100.
        assert!(
            (ratio - 0.75).abs() < TOLERANCE,
            "Acceptance ratio should be 0.75, got {}",
            ratio
        );
    }

    #[test]
    fn test_custom_symmetry() {
        let mut verifier = SymmetryVerifier::new();
        // Custom: outcome must be less than half the Hilbert space dimension.
        verifier.add_custom("lower_half", |outcome, num_qubits| {
            outcome < (1 << num_qubits) / 2
        });

        assert!(verifier.verify_outcome(0, 2)); // 0 < 2
        assert!(verifier.verify_outcome(1, 2)); // 1 < 2
        assert!(!verifier.verify_outcome(2, 2)); // 2 >= 2
        assert!(!verifier.verify_outcome(3, 2)); // 3 >= 2
    }

    #[test]
    fn test_multiple_symmetries() {
        let mut verifier = SymmetryVerifier::new();
        verifier.add_z2_parity(vec![0, 1, 2], 0); // Even parity overall.
        verifier.add_particle_conservation(vec![0, 1, 2], 2); // Exactly 2 particles.

        // 0b011 = 3 -> parity = 0, particles = 2 -> pass both
        assert!(verifier.verify_outcome(0b011, 3));
        // 0b101 = 5 -> parity = 0, particles = 2 -> pass both
        assert!(verifier.verify_outcome(0b101, 3));
        // 0b110 = 6 -> parity = 0, particles = 2 -> pass both
        assert!(verifier.verify_outcome(0b110, 3));
        // 0b111 = 7 -> parity = 1, particles = 3 -> fail parity
        assert!(!verifier.verify_outcome(0b111, 3));
        // 0b001 = 1 -> parity = 1, particles = 1 -> fail both
        assert!(!verifier.verify_outcome(0b001, 3));
    }

    // ----- Virtual distillation tests -----

    #[test]
    fn test_virtual_distillation_pure_state() {
        // For a pure state, Tr(rho^M) = 1 and <Z> is exact.
        let state = QuantumState::new(1); // |0>
        let vd = VirtualDistillation::new(2);
        let exp = vd.estimate_expectation(&[&state, &state], 0);
        assert!(
            (exp - 1.0).abs() < TOLERANCE,
            "Virtual distillation on |0> should give <Z>=1.0, got {}",
            exp
        );
    }

    #[test]
    fn test_virtual_distillation_x_state() {
        let mut state = QuantumState::new(1);
        GateOperations::x(&mut state, 0); // |1>
        let vd = VirtualDistillation::new(2);
        let exp = vd.estimate_expectation(&[&state, &state], 0);
        assert!(
            (exp - (-1.0)).abs() < TOLERANCE,
            "Virtual distillation on |1> should give <Z>=-1.0, got {}",
            exp
        );
    }

    #[test]
    fn test_virtual_distillation_purity_pure_state() {
        let state = QuantumState::new(2); // |00>
        let vd = VirtualDistillation::new(2);
        let purity = vd.estimate_purity(&[&state, &state]);
        assert!(
            (purity - 1.0).abs() < TOLERANCE,
            "Purity of pure state should be 1.0, got {}",
            purity
        );
    }

    #[test]
    fn test_virtual_distillation_suppresses_noise() {
        // Simulate a "noisy" scenario by creating two distinct states and
        // averaging their density matrices.  The distillation should suppress
        // the mixed component relative to the dominant pure state.
        let state0 = QuantumState::new(1); // |0>
        let mut state1 = QuantumState::new(1);
        GateOperations::x(&mut state1, 0); // |1>

        // Using two copies of |0> gives purity 1.0.  If one copy is |1>,
        // the averaged density matrix is mixed, and Tr(rho^2) < 1.
        let vd = VirtualDistillation::new(2);
        let purity_mixed = vd.estimate_purity(&[&state0, &state1]);
        assert!(
            purity_mixed < 1.0 - TOLERANCE,
            "Mixed state purity should be < 1.0, got {}",
            purity_mixed
        );

        // Pure reference.
        let purity_pure = vd.estimate_purity(&[&state0, &state0]);
        assert!(
            purity_pure > purity_mixed,
            "Pure purity ({}) should exceed mixed purity ({})",
            purity_pure,
            purity_mixed
        );
    }

    #[test]
    fn test_virtual_distillation_higher_copies() {
        // More copies should not change the result for a pure state.
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0); // |+>

        let vd2 = VirtualDistillation::new(2);
        let vd3 = VirtualDistillation::new(3);
        let vd4 = VirtualDistillation::new(4);

        let exp2 = vd2.estimate_expectation(&[&state, &state], 0);
        let exp3 = vd3.estimate_expectation(&[&state, &state, &state], 0);
        let exp4 = vd4.estimate_expectation(&[&state, &state, &state, &state], 0);

        // |+> gives <Z> = 0 for all M.
        assert!(exp2.abs() < TOLERANCE, "M=2: <Z> should be ~0, got {}", exp2);
        assert!(exp3.abs() < TOLERANCE, "M=3: <Z> should be ~0, got {}", exp3);
        assert!(exp4.abs() < TOLERANCE, "M=4: <Z> should be ~0, got {}", exp4);
    }

    // ----- Integration tests -----

    #[test]
    fn test_pec_with_symmetry_verification() {
        // Build a 2-qubit circuit that preserves even parity.
        // H on qubit 0, CNOT(0->1): produces Bell state (|00> + |11>)/sqrt(2).
        let circuit = vec![
            Gate::new(GateType::H, vec![0], vec![]),
            Gate::new(GateType::CNOT, vec![1], vec![0]),
        ];

        // Run the ideal circuit to get expected counts.
        let mut state = QuantumState::new(2);
        for gate in &circuit {
            apply_gate_to_state(&mut state, gate);
        }
        let counts = state.sample(10000);

        // Verify even parity post-selection on qubits 0,1.
        let mut verifier = SymmetryVerifier::new();
        verifier.add_z2_parity(vec![0, 1], 0);

        let filtered = verifier.filter_counts(&counts, 2);

        // Bell state only produces |00> and |11>, both even parity.
        // Acceptance ratio should be ~1.0.
        let ratio = verifier.acceptance_ratio(&counts, 2);
        assert!(
            ratio > 0.95,
            "Bell state should have near-perfect even parity acceptance, got {}",
            ratio
        );

        // Filtered counts should contain only 0b00 and 0b11.
        for (&outcome, _) in &filtered {
            assert!(
                outcome == 0b00 || outcome == 0b11,
                "Unexpected outcome in filtered Bell state: {}",
                outcome
            );
        }
    }

    #[test]
    fn test_cdr_with_virtual_distillation() {
        // Verify that CDR and virtual distillation can be combined.
        // Train CDR on known data, then check that the model applies.
        let ideal = vec![1.0, 0.0, -1.0];
        let noisy = vec![0.9, 0.05, -0.85];

        let model = CDRModel::train(&ideal, &noisy);

        // Virtual distillation on a pure state should give exact <Z>.
        let state = QuantumState::new(1);
        let vd = VirtualDistillation::new(2);
        let raw_exp = vd.estimate_expectation(&[&state, &state], 0);

        // CDR mitigation of the VD result (for a pure state, should stay ~1.0).
        let mitigated = model.mitigate(raw_exp);

        // Both raw and mitigated should be close to 1.0 for a pure |0> state.
        assert!(
            (raw_exp - 1.0).abs() < TOLERANCE,
            "VD raw expectation should be ~1.0, got {}",
            raw_exp
        );
        assert!(
            (mitigated - 1.0).abs() < 0.2,
            "CDR-mitigated VD result should be near 1.0, got {}",
            mitigated
        );
    }
}
