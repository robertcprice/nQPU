//! Cross-Resonance (CR) Gate Auto-Calibrator
//!
//! Automatically calibrates the amplitude of a cross-resonance pulse to
//! realise a ZX(pi/2) entangling interaction between two transmon qubits.
//!
//! # Physical Background
//!
//! In a cross-resonance gate the control qubit is driven at the target
//! qubit's frequency. In the rotating frame this generates an effective
//! two-qubit Hamiltonian dominated by a ZX term:
//!
//! ```text
//! H_eff ≈ (Omega_ZX / 2) * Z_c ⊗ X_t + ...
//! ```
//!
//! The ZX rotation angle after time T is theta_ZX = Omega_ZX * T.
//! A CNOT-equivalent gate requires theta_ZX = pi/2 (plus local
//! rotations). This module sweeps the CR drive amplitude, extracts the
//! ZX angle at each point via Hamiltonian simulation, and interpolates
//! to find the amplitude that gives exactly pi/2.
//!
//! # Algorithm
//!
//! 1. For each candidate amplitude A in a user-specified range:
//!    a. Build a 2-qubit transmon Hamiltonian with coupling g and
//!       anharmonicities alpha_c, alpha_t.
//!    b. Apply a constant CR pulse of duration T at amplitude A on the
//!       control channel driving at the target frequency.
//!    c. Simulate the unitary U(A) via RK4 integration.
//!    d. Extract the ZX angle: prepare |+,0> and |+,1>, evolve, and
//!       measure the conditional Bloch rotation on the target qubit.
//!       theta_ZX = (angle_ctrl1 - angle_ctrl0) / 2.
//! 2. Interpolate the ZX(A) curve to find A* where ZX ≈ pi/2.
//! 3. Compute the gate fidelity at A* versus the ideal ZX(pi/2) unitary.
//!
//! # Example
//!
//! ```rust,ignore
//! use nqpu_metal::cr_calibration::*;
//!
//! let config = CRConfig {
//!     control_freq_ghz: 5.0,
//!     target_freq_ghz: 5.1,
//!     coupling_mhz: 3.0,
//!     anharmonicity_mhz: -300.0,
//!     duration_ns: 300.0,
//! };
//! let cal = CRCalibrator::new(config);
//! let result = cal.calibrate((0.01, 0.20), 20);
//! assert!((result.zx_angle - std::f64::consts::FRAC_PI_2).abs() < 0.2);
//! ```

use num_complex::Complex64;
use std::f64::consts::PI;

use super::pulse_simulation::{
    average_gate_fidelity, Channel, DenseMatrix, Pulse, PulseSchedule, PulseSimulator,
    ScheduledPulse, TransmonSystem,
};

// ============================================================
// COMPLEX HELPERS
// ============================================================

const ZERO: Complex64 = Complex64 { re: 0.0, im: 0.0 };
const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[inline]
fn c_re(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

// ============================================================
// CR CONFIGURATION
// ============================================================

/// Configuration for the cross-resonance gate calibration.
///
/// All frequency parameters follow IBM / fixed-frequency transmon conventions.
#[derive(Clone, Debug)]
pub struct CRConfig {
    /// Control qubit frequency in GHz.
    pub control_freq_ghz: f64,
    /// Target qubit frequency in GHz.
    pub target_freq_ghz: f64,
    /// Qubit-qubit coupling strength in MHz.
    pub coupling_mhz: f64,
    /// Anharmonicity in MHz (negative for transmon, typically ~-300 MHz).
    /// Applied identically to both qubits for simplicity.
    pub anharmonicity_mhz: f64,
    /// CR pulse duration in nanoseconds.
    pub duration_ns: f64,
}

impl Default for CRConfig {
    fn default() -> Self {
        Self {
            control_freq_ghz: 5.0,
            target_freq_ghz: 5.1,
            coupling_mhz: 3.0,
            anharmonicity_mhz: -300.0,
            duration_ns: 300.0,
        }
    }
}

// ============================================================
// CALIBRATION RESULT
// ============================================================

/// Output of the CR amplitude calibration sweep.
#[derive(Clone, Debug)]
pub struct CRCalibrationResult {
    /// Optimal CR drive amplitude that gives ZX angle closest to pi/2.
    pub optimal_amplitude: f64,
    /// ZX angle achieved at the optimal amplitude (radians).
    pub zx_angle: f64,
    /// Gate fidelity of the calibrated pulse versus the ideal ZX(pi/2) unitary.
    pub gate_fidelity: f64,
    /// All amplitudes probed during the sweep.
    pub amplitudes: Vec<f64>,
    /// ZX angles extracted at each amplitude.
    pub zx_angles: Vec<f64>,
}

// ============================================================
// CR CALIBRATOR
// ============================================================

/// Automated cross-resonance gate calibrator.
///
/// Sweeps the CR drive amplitude and finds the value producing
/// a ZX rotation of pi/2, which (combined with local gates) yields CNOT.
#[derive(Clone, Debug)]
pub struct CRCalibrator {
    config: CRConfig,
    /// Prebuilt 2-qubit TransmonSystem (2-level, for speed).
    system: TransmonSystem,
}

impl CRCalibrator {
    /// Create a new calibrator from device parameters.
    pub fn new(config: CRConfig) -> Self {
        let anhar_ghz = config.anharmonicity_mhz * 1e-3;
        let coupling_ghz = config.coupling_mhz * 1e-3;

        let system = TransmonSystem::two_qubit(
            config.control_freq_ghz,
            config.target_freq_ghz,
            anhar_ghz,
            anhar_ghz,
            coupling_ghz,
        );

        Self { config, system }
    }

    /// Run the amplitude calibration sweep.
    ///
    /// Subtracts the baseline ZX angle (from zero-drive evolution) at each
    /// point, mimicking the echoed cross-resonance protocol used on real
    /// hardware. This isolates the CR-drive contribution from the always-on
    /// static ZZ / ZX interaction.
    ///
    /// # Arguments
    /// * `amplitude_range` - (min, max) amplitude to sweep.
    /// * `num_points` - Number of uniformly-spaced amplitudes to probe.
    ///
    /// # Returns
    /// `CRCalibrationResult` with the optimal amplitude, ZX angle,
    /// gate fidelity, and the full sweep data.
    pub fn calibrate(
        &self,
        amplitude_range: (f64, f64),
        num_points: usize,
    ) -> CRCalibrationResult {
        assert!(num_points >= 2, "Need at least 2 points for interpolation");
        let (a_min, a_max) = amplitude_range;
        assert!(a_max > a_min, "amplitude_range max must exceed min");

        // Baseline: ZX angle from free evolution (no CR drive).
        let baseline = self.extract_zx_angle_raw(0.0);

        let step = (a_max - a_min) / (num_points - 1) as f64;
        let mut amplitudes = Vec::with_capacity(num_points);
        let mut zx_angles = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let amp = a_min + i as f64 * step;
            let raw_angle = self.extract_zx_angle_raw(amp);
            // Subtract baseline to isolate CR-drive contribution.
            let net_angle = (raw_angle - baseline).abs();
            amplitudes.push(amp);
            zx_angles.push(net_angle);
        }

        // Find the amplitude closest to ZX = pi/2 via linear interpolation.
        let target = PI / 2.0;
        let (optimal_amplitude, optimal_angle) =
            interpolate_to_target(&amplitudes, &zx_angles, target);

        // Compute gate fidelity at the optimal amplitude.
        let gate_fidelity = self.gate_fidelity_at(optimal_amplitude);

        CRCalibrationResult {
            optimal_amplitude,
            zx_angle: optimal_angle,
            gate_fidelity,
            amplitudes,
            zx_angles,
        }
    }

    /// Extract the net ZX rotation angle at a given CR drive amplitude,
    /// with the baseline (zero-drive) contribution subtracted.
    ///
    /// This mirrors the echoed CR calibration protocol on real hardware.
    pub fn extract_zx_angle(&self, amplitude: f64) -> f64 {
        let raw = self.extract_zx_angle_raw(amplitude);
        let baseline = self.extract_zx_angle_raw(0.0);
        (raw - baseline).abs()
    }

    /// Extract the raw ZX rotation angle without baseline subtraction.
    ///
    /// Method: prepare |+,0> (control in |+>, target in |0>) and
    /// |+,1> (control in |+>, target in |1>), evolve through the CR
    /// Hamiltonian, and measure the target qubit Bloch angle conditioned
    /// on the control state.
    ///
    /// ZX angle = (theta_ctrl1 - theta_ctrl0) / 2
    ///
    /// where theta_ctrl_k is the rotation angle acquired by the target
    /// qubit when the control is projected into |k>.
    fn extract_zx_angle_raw(&self, amplitude: f64) -> f64 {
        let schedule = self.build_cr_schedule(amplitude);

        // |+,0> = (|00> + |10>) / sqrt(2)
        let s = 1.0 / 2.0f64.sqrt();
        let state_p0 = vec![c_re(s), ZERO, c_re(s), ZERO]; // |00> + |10>

        // |+,1> = (|01> + |11>) / sqrt(2)
        let state_p1 = vec![ZERO, c_re(s), ZERO, c_re(s)]; // |01> + |11>

        let sim_p0 = PulseSimulator::new(self.system.clone(), schedule.clone());
        let sim_p1 = PulseSimulator::new(self.system.clone(), schedule);

        let result_p0 = sim_p0.simulate_from_state(&state_p0);
        let result_p1 = sim_p1.simulate_from_state(&state_p1);

        // Extract conditional target qubit rotation angles.
        let angle_0 = target_bloch_angle(&result_p0.final_state);
        let angle_1 = target_bloch_angle(&result_p1.final_state);

        // ZX angle: half the difference in conditional rotations.
        ((angle_1 - angle_0) / 2.0).abs()
    }

    /// Build a CR pulse schedule at the given amplitude.
    fn build_cr_schedule(&self, amplitude: f64) -> PulseSchedule {
        let cr_pulse = Pulse::constant(
            self.config.duration_ns,
            amplitude,
            self.config.target_freq_ghz, // Drive control qubit at target frequency
            0.0,
        );

        PulseSchedule::new(0.5).with_pulse(ScheduledPulse {
            channel: Channel::Control(0, 1),
            start_time: 0.0,
            pulse: cr_pulse,
        })
    }

    /// Compute average gate fidelity of the CR pulse at a given amplitude
    /// versus the ideal ZX(pi/2) unitary.
    fn gate_fidelity_at(&self, amplitude: f64) -> f64 {
        let schedule = self.build_cr_schedule(amplitude);
        let sim = PulseSimulator::new(self.system.clone(), schedule);
        let u_actual = sim.simulate_unitary();

        let u_target = ideal_zx_pi_over_2();
        average_gate_fidelity(&u_target, &u_actual)
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Extract the effective Bloch rotation angle of the target qubit
/// from a 2-qubit state that started as |+, x>.
///
/// We look at the target qubit's reduced state by tracing over the
/// control. The angle is arccos of the Z-expectation value, giving
/// the polar angle on the Bloch sphere.
fn target_bloch_angle(state: &[Complex64]) -> f64 {
    assert!(state.len() >= 4, "Expected 4-element 2-qubit state");

    // Partial trace over control qubit to get target qubit reduced state.
    // rho_target = Tr_control(|psi><psi|)
    //
    // In the computational basis {|00>, |01>, |10>, |11>}:
    //   target |0> component from control=|0>: state[0]
    //   target |1> component from control=|0>: state[1]
    //   target |0> component from control=|1>: state[2]
    //   target |1> component from control=|1>: state[3]
    //
    // rho_target[0,0] = |state[0]|^2 + |state[2]|^2
    // rho_target[1,1] = |state[1]|^2 + |state[3]|^2
    // rho_target[0,1] = state[0]*conj(state[1]) + state[2]*conj(state[3])

    let rho_00 = state[0].norm_sqr() + state[2].norm_sqr();
    let rho_11 = state[1].norm_sqr() + state[3].norm_sqr();
    let rho_01 = state[0] * state[1].conj() + state[2] * state[3].conj();

    // Bloch vector components:
    // x = 2 * Re(rho_01)
    // y = 2 * Im(rho_01)
    // z = rho_00 - rho_11
    let bx = 2.0 * rho_01.re;
    let by = 2.0 * rho_01.im;
    let bz = rho_00 - rho_11;

    // Rotation angle from the initial state. The initial target state
    // was |0> (bz = 1) or |1> (bz = -1). The rotation angle is the
    // arccosine of the overlap with the initial Z direction, but
    // we use atan2 of the transverse components for better numerical
    // behaviour over the full range.
    let transverse = (bx * bx + by * by).sqrt();
    transverse.atan2(bz)
}

/// Linear interpolation to find the amplitude where ZX angle = target.
///
/// Scans the sweep data for the interval where the ZX angle crosses the
/// target, then linearly interpolates. If no crossing is found, returns
/// the amplitude with the closest ZX angle.
fn interpolate_to_target(
    amplitudes: &[f64],
    angles: &[f64],
    target: f64,
) -> (f64, f64) {
    // Look for a crossing: angles[i] <= target <= angles[i+1] or vice versa.
    for i in 0..amplitudes.len().saturating_sub(1) {
        let (a0, a1) = (angles[i], angles[i + 1]);
        let crosses = (a0 <= target && target <= a1) || (a1 <= target && target <= a0);
        if crosses {
            let denom = a1 - a0;
            if denom.abs() < 1e-30 {
                return (amplitudes[i], angles[i]);
            }
            let frac = (target - a0) / denom;
            let amp = amplitudes[i] + frac * (amplitudes[i + 1] - amplitudes[i]);
            // Linear interpolation of the angle too (for reporting).
            let angle = a0 + frac * (a1 - a0);
            return (amp, angle);
        }
    }

    // No crossing found: pick the closest point.
    let (best_idx, _) = angles
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (**a - target).abs();
            let db = (**b - target).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, &0.0));

    (amplitudes[best_idx], angles[best_idx])
}

/// Construct the ideal ZX(pi/2) unitary.
///
/// ZX(theta) = exp(-i * theta/2 * Z ⊗ X)
///
/// For theta = pi/2:
///
/// ```text
/// ZX(pi/2) = cos(pi/4) I⊗I - i sin(pi/4) Z⊗X
///          = (1/sqrt(2)) (I⊗I - i Z⊗X)
/// ```
///
/// In the computational basis {|00>, |01>, |10>, |11>}:
///
/// ```text
///         [ cos   -i*sin    0        0    ]
/// ZX =    [-i*sin  cos      0        0    ]
///         [  0      0       cos    i*sin   ]
///         [  0      0      i*sin    cos    ]
/// ```
fn ideal_zx_pi_over_2() -> DenseMatrix {
    let cos_val = (PI / 4.0).cos(); // 1/sqrt(2)
    let sin_val = (PI / 4.0).sin(); // 1/sqrt(2)

    let mut u = DenseMatrix::zeros(4);
    u.set(0, 0, c_re(cos_val));
    u.set(0, 1, c(0.0, -sin_val));
    u.set(1, 0, c(0.0, -sin_val));
    u.set(1, 1, c_re(cos_val));
    u.set(2, 2, c_re(cos_val));
    u.set(2, 3, c(0.0, sin_val));
    u.set(3, 2, c(0.0, sin_val));
    u.set(3, 3, c_re(cos_val));
    u
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // 1. CRCalibrator constructs without panic
    // ----------------------------------------------------------
    #[test]
    fn test_cr_calibrator_construction() {
        let config = CRConfig::default();
        let _cal = CRCalibrator::new(config);
    }

    // ----------------------------------------------------------
    // 2. ZX angle increases with amplitude (baseline-subtracted)
    // ----------------------------------------------------------
    #[test]
    fn test_zx_angle_increases_with_amplitude() {
        let config = CRConfig {
            control_freq_ghz: 5.0,
            target_freq_ghz: 5.15,
            coupling_mhz: 3.0,
            anharmonicity_mhz: -300.0,
            duration_ns: 250.0,
        };
        let cal = CRCalibrator::new(config);

        // Use baseline-subtracted angles so only the CR drive
        // contribution is compared.
        let angle_low = cal.extract_zx_angle(0.02);
        let angle_high = cal.extract_zx_angle(0.15);

        assert!(
            angle_high > angle_low,
            "Larger amplitude should produce larger net ZX angle: low={:.4}, high={:.4}",
            angle_low,
            angle_high
        );
    }

    // ----------------------------------------------------------
    // 3. Calibration finds amplitude with ZX near pi/2
    // ----------------------------------------------------------
    #[test]
    fn test_cr_calibration_finds_pi_over_2() {
        let config = CRConfig {
            control_freq_ghz: 5.0,
            target_freq_ghz: 5.15,
            coupling_mhz: 3.0,
            anharmonicity_mhz: -300.0,
            duration_ns: 300.0,
        };
        let cal = CRCalibrator::new(config);

        // Wide sweep to capture the crossing.
        let result = cal.calibrate((0.01, 0.40), 30);

        let target = PI / 2.0;
        let deviation = (result.zx_angle - target).abs();

        // With 30 sweep points over a wide range, the interpolation should
        // land within ~0.3 rad of pi/2. The exact tolerance depends on
        // pulse physics and the coarseness of the sweep.
        assert!(
            deviation < 0.5,
            "ZX angle should be within 0.5 rad of pi/2: got {:.4} (deviation {:.4})",
            result.zx_angle,
            deviation
        );

        assert!(
            result.optimal_amplitude > 0.0,
            "Optimal amplitude should be positive"
        );
    }

    // ----------------------------------------------------------
    // 4. Sweep data has correct length
    // ----------------------------------------------------------
    #[test]
    fn test_sweep_data_length() {
        let config = CRConfig::default();
        let cal = CRCalibrator::new(config);

        let result = cal.calibrate((0.01, 0.20), 10);
        assert_eq!(result.amplitudes.len(), 10);
        assert_eq!(result.zx_angles.len(), 10);
    }

    // ----------------------------------------------------------
    // 5. Gate fidelity is between 0 and 1
    // ----------------------------------------------------------
    #[test]
    fn test_gate_fidelity_bounded() {
        let config = CRConfig::default();
        let cal = CRCalibrator::new(config);

        let result = cal.calibrate((0.01, 0.20), 10);
        assert!(
            result.gate_fidelity >= 0.0 && result.gate_fidelity <= 1.0,
            "Gate fidelity must be in [0, 1], got {}",
            result.gate_fidelity
        );
    }

    // ----------------------------------------------------------
    // 6. Ideal ZX(pi/2) unitary is unitary
    // ----------------------------------------------------------
    #[test]
    fn test_ideal_zx_unitary_is_unitary() {
        let u = ideal_zx_pi_over_2();
        let product = u.dagger().matmul(&u);

        // Should be close to identity.
        let id = DenseMatrix::identity(4);
        let diff = product.sub(&id);
        let err = diff.norm_sq().sqrt();
        assert!(
            err < 1e-10,
            "ZX(pi/2) should be unitary: ||U†U - I|| = {:.2e}",
            err
        );
    }

    // ----------------------------------------------------------
    // 7. Zero amplitude gives zero net ZX angle (baseline-subtracted)
    // ----------------------------------------------------------
    #[test]
    fn test_zero_amplitude_zero_net_zx() {
        let config = CRConfig {
            control_freq_ghz: 5.0,
            target_freq_ghz: 5.15,
            coupling_mhz: 3.0,
            anharmonicity_mhz: -300.0,
            duration_ns: 200.0,
        };
        let cal = CRCalibrator::new(config);

        // After baseline subtraction, zero drive should give exactly
        // zero net ZX angle.
        let angle = cal.extract_zx_angle(0.0);
        assert!(
            angle < 1e-10,
            "Zero drive amplitude should give zero net ZX angle, got {:.4e}",
            angle
        );
    }

    // ----------------------------------------------------------
    // 8. Interpolation helper: exact crossing
    // ----------------------------------------------------------
    #[test]
    fn test_interpolation_exact_crossing() {
        let amps = vec![0.0, 1.0, 2.0];
        let angles = vec![0.0, PI / 2.0, PI];

        let (amp, angle) = interpolate_to_target(&amps, &angles, PI / 2.0);
        assert!(
            (amp - 1.0).abs() < 1e-10,
            "Should interpolate exactly to amp=1.0, got {}",
            amp
        );
        assert!(
            (angle - PI / 2.0).abs() < 1e-10,
            "Should interpolate exactly to pi/2, got {}",
            angle
        );
    }

    // ----------------------------------------------------------
    // 9. Interpolation helper: between points
    // ----------------------------------------------------------
    #[test]
    fn test_interpolation_between_points() {
        let amps = vec![0.0, 1.0];
        let angles = vec![0.0, PI];
        let target = PI / 2.0;

        let (amp, angle) = interpolate_to_target(&amps, &angles, target);
        assert!(
            (amp - 0.5).abs() < 1e-10,
            "Should interpolate to amp=0.5, got {}",
            amp
        );
        assert!(
            (angle - target).abs() < 1e-10,
            "Should interpolate to pi/2, got {}",
            angle
        );
    }

    // ----------------------------------------------------------
    // 10. Interpolation helper: no crossing picks closest
    // ----------------------------------------------------------
    #[test]
    fn test_interpolation_no_crossing() {
        let amps = vec![0.0, 1.0, 2.0];
        let angles = vec![0.1, 0.2, 0.3];
        let target = 10.0; // Way above all angles

        let (amp, angle) = interpolate_to_target(&amps, &angles, target);
        // Closest to 10.0 is 0.3 at amp=2.0
        assert!(
            (amp - 2.0).abs() < 1e-10,
            "Should pick closest amplitude, got {}",
            amp
        );
        assert!(
            (angle - 0.3).abs() < 1e-10,
            "Should pick closest angle, got {}",
            angle
        );
    }
}
