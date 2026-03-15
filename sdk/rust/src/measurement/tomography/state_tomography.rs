//! Quantum State Tomography
//!
//! This module implements quantum state tomography for reconstructing
//! an unknown quantum state from measurement statistics.

use crate::{QuantumState, C64};
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

/// Measurement basis for tomography
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeasurementBasis {
    X, // Pauli-X basis
    Y, // Pauli-Y basis
    Z, // Pauli-Z basis (computational)
}

impl MeasurementBasis {
    /// Get the rotation needed to measure in this basis
    pub fn rotation_to_z(&self) -> Option<(f64, f64)> {
        match self {
            MeasurementBasis::Z => None,
            MeasurementBasis::X => Some((FRAC_1_SQRT_2, FRAC_1_SQRT_2)), // H gate
            MeasurementBasis::Y => Some((FRAC_1_SQRT_2, -FRAC_1_SQRT_2)), // H·S†
        }
    }
}

/// Density matrix representation
pub type DensityMatrix = Vec<Vec<C64>>;

/// Tomography measurement settings
#[derive(Clone, Debug)]
pub struct TomographySettings {
    pub num_measurements: usize,
    pub bases: Vec<Vec<MeasurementBasis>>,
}

impl TomographySettings {
    /// Create settings for full state tomography
    pub fn full_state_tomography(num_qubits: usize, shots_per_setting: usize) -> Self {
        let mut bases = Vec::new();

        for setting in 0..(3_usize.pow(num_qubits as u32)) {
            let mut basis_for_setting = Vec::new();
            let mut temp = setting;

            for _ in 0..num_qubits {
                let digit = temp % 3;
                temp /= 3;

                basis_for_setting.push(match digit {
                    0 => MeasurementBasis::X,
                    1 => MeasurementBasis::Y,
                    2 => MeasurementBasis::Z,
                    _ => unreachable!(),
                });
            }

            bases.push(basis_for_setting);
        }

        TomographySettings {
            num_measurements: shots_per_setting,
            bases,
        }
    }

    /// Create settings for simplified tomography (fewer measurements)
    pub fn simplified_tomography(num_qubits: usize, shots_per_setting: usize) -> Self {
        let mut bases = Vec::new();

        for setting in 0..(2_usize.pow(num_qubits as u32)) {
            let mut basis_for_setting = Vec::new();
            let mut temp = setting;

            for _ in 0..num_qubits {
                let digit = temp % 2;
                temp /= 2;

                basis_for_setting.push(if digit == 0 {
                    MeasurementBasis::X
                } else {
                    MeasurementBasis::Z
                });
            }

            bases.push(basis_for_setting);
        }

        TomographySettings {
            num_measurements: shots_per_setting,
            bases,
        }
    }
}

/// Measurement result for tomography
#[derive(Clone, Debug)]
pub struct TomographyMeasurement {
    pub basis_setting: Vec<MeasurementBasis>,
    pub outcome: usize,
    pub counts: usize,
}

/// State tomography engine
pub struct StateTomography {
    num_qubits: usize,
}

impl StateTomography {
    pub fn new(num_qubits: usize) -> Self {
        StateTomography { num_qubits }
    }

    /// Perform state tomography from measurement data
    ///
    /// Reconstructs the density matrix from measurement statistics
    pub fn reconstruct_state(
        &self,
        measurements: &[TomographyMeasurement],
        settings: &TomographySettings,
    ) -> Result<DensityMatrix, String> {
        let dim = 1_usize << self.num_qubits;
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        // Use maximum likelihood estimation (simplified)
        for measurement in measurements {
            let basis = &measurement.basis_setting;

            let probability = measurement.counts as f64 / settings.num_measurements as f64;

            // Add contribution to density matrix
            let outcome_state = self.basis_state(basis, measurement.outcome);
            for i in 0..dim {
                for j in 0..dim {
                    rho[i][j] = rho[i][j] + outcome_state[i][j].scale(probability);
                }
            }
        }

        // Normalize trace to 1
        let trace: f64 = (0..dim).map(|i| rho[i][i].re).sum();
        if trace > 0.0 {
            for i in 0..dim {
                for j in 0..dim {
                    rho[i][j] = rho[i][j].scale(1.0 / trace);
                }
            }
        }

        Ok(rho)
    }

    /// Get the basis state for a given measurement
    fn basis_state(&self, bases: &[MeasurementBasis], outcome: usize) -> DensityMatrix {
        let dim = 1 << self.num_qubits;
        // Column vector |ψ⟩ with shape (dim x 1)
        let mut state = vec![vec![Complex64::new(0.0, 0.0); 1]; dim];

        // Create computational basis state for outcome
        state[outcome][0] = Complex64::new(1.0, 0.0);

        // Apply rotations to match measurement bases
        let mut rotated = state;
        for (qubit, basis) in bases.iter().enumerate() {
            if let Some((cos, sin)) = basis.rotation_to_z() {
                rotated = self.apply_single_qubit_rotation(&rotated, qubit, cos, sin);
            }
        }

        // Create density matrix |ψ⟩⟨ψ|
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let psi_i = rotated[i][0];
                let psi_j_conj = C64 {
                    re: rotated[j][0].re,
                    im: -rotated[j][0].im,
                };
                rho[i][j] = psi_i * psi_j_conj;
            }
        }

        rho
    }

    /// Apply single-qubit rotation to state vector
    fn apply_single_qubit_rotation(
        &self,
        state: &DensityMatrix,
        qubit: usize,
        cos: f64,
        sin: f64,
    ) -> DensityMatrix {
        let dim = state.len();
        let mut result = state.clone();

        let stride = 1 << qubit;
        for i in (0..dim).step_by(stride * 2) {
            for j in i..(i + stride) {
                let idx1 = j;
                let idx2 = j | stride;

                if idx2 < dim {
                    let a = state[idx1][0];
                    let b = state[idx2][0];

                    result[idx1][0] = C64 {
                        re: a.re * cos + b.re * sin,
                        im: a.im * cos + b.im * sin,
                    };
                    result[idx2][0] = C64 {
                        re: a.re * sin - b.re * cos,
                        im: a.im * sin - b.im * cos,
                    };
                }
            }
        }

        result
    }

    /// Simulate measurements on a given state for tomography
    pub fn simulate_measurements(
        &self,
        state: &QuantumState,
        settings: &TomographySettings,
    ) -> Vec<TomographyMeasurement> {
        let mut measurements = Vec::new();

        for basis_setting in &settings.bases {
            let mut outcome_counts = vec![0usize; 1 << self.num_qubits];

            for _ in 0..settings.num_measurements {
                let mut state_copy = state.clone();

                // Rotate to measurement basis
                for (qubit, basis) in basis_setting.iter().enumerate() {
                    match basis {
                        MeasurementBasis::Z => {}
                        MeasurementBasis::X => {
                            crate::GateOperations::h(&mut state_copy, qubit);
                        }
                        MeasurementBasis::Y => {
                            crate::GateOperations::h(&mut state_copy, qubit);
                            crate::GateOperations::s(&mut state_copy, qubit);
                            crate::GateOperations::s(&mut state_copy, qubit);
                        }
                    }
                }

                let (outcome, _) = state_copy.measure();
                outcome_counts[outcome] += 1;
            }

            for (outcome, &counts) in outcome_counts.iter().enumerate() {
                if counts > 0 {
                    measurements.push(TomographyMeasurement {
                        basis_setting: basis_setting.clone(),
                        outcome,
                        counts,
                    });
                }
            }
        }

        measurements
    }

    /// Compute fidelity between reconstructed and true state
    pub fn fidelity(&self, reconstructed: &DensityMatrix, true_state: &QuantumState) -> f64 {
        let dim = true_state.dim;
        let mut true_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        let amplitudes = true_state.amplitudes_ref();
        for i in 0..dim {
            for j in 0..dim {
                let a = amplitudes[i];
                let b_conj = C64 {
                    re: amplitudes[j].re,
                    im: -amplitudes[j].im,
                };
                true_rho[i][j] = a * b_conj;
            }
        }

        // Compute fidelity: F = Tr(√(√ρ σ √ρ))
        // Simplified: F = Tr(ρ σ) for pure states
        let mut fidelity = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                fidelity += (reconstructed[i][j] * true_rho[j][i]).re;
            }
        }

        fidelity.max(0.0).min(1.0)
    }
}

/// Process tomography for quantum channels
pub struct ProcessTomography {
    num_qubits: usize,
}

impl ProcessTomography {
    pub fn new(num_qubits: usize) -> Self {
        ProcessTomography { num_qubits }
    }

    /// Reconstruct a quantum process (channel) from input-output state pairs
    pub fn reconstruct_process(
        &self,
        input_states: &[QuantumState],
        output_states: &[QuantumState],
    ) -> Result<Vec<Vec<Vec<Vec<C64>>>>, String> {
        if input_states.len() != output_states.len() {
            return Err("Input and output state counts must match".to_string());
        }

        let dim = 1 << self.num_qubits;
        let mut chi = Vec::new();

        for _ in 0..dim {
            let mut chi_1 = Vec::new();
            for _ in 0..dim {
                let mut chi_2 = Vec::new();
                for _ in 0..dim {
                    let chi_3 = vec![Complex64::new(0.0, 0.0); dim];
                    chi_2.push(chi_3);
                }
                chi_1.push(chi_2);
            }
            chi.push(chi_1);
        }

        for (input, output) in input_states.iter().zip(output_states.iter()) {
            let rho_in = self.state_to_density_matrix(input);
            let rho_out = self.state_to_density_matrix(output);

            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        for l in 0..dim {
                            let rho_kl_conj = C64 {
                                re: rho_in[k][l].re,
                                im: -rho_in[k][l].im,
                            };
                            chi[i][j][k][l] = chi[i][j][k][l] + rho_out[i][j] * rho_kl_conj;
                        }
                    }
                }
            }
        }

        Ok(chi)
    }

    fn state_to_density_matrix(&self, state: &QuantumState) -> DensityMatrix {
        let dim = state.dim;
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        let amplitudes = state.amplitudes_ref();
        for i in 0..dim {
            for j in 0..dim {
                let a = amplitudes[i];
                let b_conj = C64 {
                    re: amplitudes[j].re,
                    im: -amplitudes[j].im,
                };
                rho[i][j] = a * b_conj;
            }
        }

        rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tomography_settings() {
        let settings = TomographySettings::full_state_tomography(2, 100);
        assert_eq!(settings.bases.len(), 9);

        let settings = TomographySettings::simplified_tomography(2, 100);
        assert_eq!(settings.bases.len(), 4);
    }

    #[test]
    fn test_simulate_measurements() {
        let tomography = StateTomography::new(2);
        let mut state = QuantumState::new(2);

        crate::GateOperations::h(&mut state, 0);
        crate::GateOperations::cnot(&mut state, 0, 1);

        let settings = TomographySettings::simplified_tomography(2, 100);
        let measurements = tomography.simulate_measurements(&state, &settings);

        assert!(!measurements.is_empty());
    }

    #[test]
    fn test_reconstruct_state() {
        let tomography = StateTomography::new(1);
        let mut state = QuantumState::new(1);
        crate::GateOperations::h(&mut state, 0);

        let settings = TomographySettings::simplified_tomography(1, 1000);
        let measurements = tomography.simulate_measurements(&state, &settings);

        let reconstructed = tomography.reconstruct_state(&measurements, &settings);
        assert!(reconstructed.is_ok());

        let rho = reconstructed.unwrap();
        let fidelity = tomography.fidelity(&rho, &state);
        assert!(fidelity > 0.5);
    }
}
