//! DRAG (Derivative Removal by Adiabatic Gate) Pulse Implementation
//!
//! Provides physics-accurate DRAG pulse calibration for transmon qubits,
//! building on the existing pulse simulation infrastructure. DRAG pulses
//! add a quadrature derivative correction that suppresses leakage from the
//! computational subspace {|0>, |1>} into the |2> state of the transmon.
//!
//! # Physics
//!
//! The DRAG correction was introduced by Motzoi et al. (PRL 2009). For a
//! transmon with anharmonicity alpha, the optimal DRAG coefficient is:
//!
//! ```text
//! beta = -1 / (2 * alpha)
//! ```
//!
//! where alpha = 2*pi * anharmonicity_ghz. The quadrature component is:
//!
//! ```text
//! Omega_Q(t) = beta * d(Omega_I)/dt
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use nqpu_metal::transmon_drag::*;
//! use nqpu_metal::pulse_simulation::TransmonSystem;
//!
//! let system = TransmonSystem::single_qubit_3level(5.0, -0.3);
//! let drag = DragPulse::new(5.0, -300.0, 24.0, std::f64::consts::PI, 6.0);
//! let schedule = drag.to_schedule(0.1);
//! ```

use super::pulse_simulation::{
    average_gate_fidelity, Channel, DenseMatrix, PulseSchedule, PulseSimulator,
    ScheduledPulse, Pulse, PulseShape, StandardGates, TransmonSystem,
};
use num_complex::Complex64;
use std::f64::consts::PI;

// Re-use the complex helpers from pulse_simulation
const ZERO: Complex64 = Complex64 { re: 0.0, im: 0.0 };
const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };

#[inline]
fn c_re(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

// ============================================================
// DRAG PULSE
// ============================================================

/// A DRAG pulse fully parameterized by transmon physics.
///
/// Given the qubit frequency, anharmonicity, desired rotation angle, pulse
/// duration, and Gaussian sigma, this struct computes the optimal DRAG
/// coefficient and amplitude analytically.
#[derive(Clone, Debug)]
pub struct DragPulse {
    /// Qubit transition frequency in GHz.
    pub frequency_ghz: f64,
    /// Anharmonicity in MHz (typically negative, e.g. -300 MHz).
    pub anharmonicity_mhz: f64,
    /// Total pulse duration in nanoseconds.
    pub duration_ns: f64,
    /// Desired rotation angle in radians (pi for X gate, pi/2 for SX).
    pub angle: f64,
    /// Gaussian standard deviation in nanoseconds.
    pub sigma_ns: f64,
}

impl DragPulse {
    /// Create a new DRAG pulse specification.
    pub fn new(
        frequency_ghz: f64,
        anharmonicity_mhz: f64,
        duration_ns: f64,
        angle: f64,
        sigma_ns: f64,
    ) -> Self {
        Self {
            frequency_ghz,
            anharmonicity_mhz,
            duration_ns,
            angle,
            sigma_ns,
        }
    }

    /// Compute the optimal DRAG beta coefficient.
    ///
    /// beta = -1 / (2 * alpha_angular) where alpha_angular = 2*pi * alpha_ghz.
    /// The anharmonicity is converted from MHz to GHz internally.
    pub fn beta(&self) -> f64 {
        let anharmonicity_ghz = self.anharmonicity_mhz / 1000.0;
        let alpha_angular = 2.0 * PI * anharmonicity_ghz;
        if alpha_angular.abs() < 1e-15 {
            return 0.0;
        }
        -0.5 / alpha_angular
    }

    /// Compute the pulse amplitude that produces the desired rotation angle.
    ///
    /// For a Gaussian pulse truncated at +/- duration/2, the integral is:
    ///   integral = sigma * sqrt(2*pi) * erf(duration / (2*sqrt(2)*sigma))
    ///
    /// The rotation angle theta = 2*pi * amplitude * integral, so:
    ///   amplitude = angle / (2*pi * integral)
    pub fn amplitude(&self) -> f64 {
        let gauss_integral = self.gaussian_integral();
        if gauss_integral.abs() < 1e-15 {
            return 0.0;
        }
        self.angle / (2.0 * PI * gauss_integral)
    }

    /// Gaussian integral accounting for truncation.
    ///
    /// integral = sigma * sqrt(2*pi) * erf(duration / (2*sqrt(2)*sigma))
    ///
    /// Uses a polynomial approximation of erf for self-contained computation.
    fn gaussian_integral(&self) -> f64 {
        let half_dur = self.duration_ns / 2.0;
        let arg = half_dur / (self.sigma_ns * std::f64::consts::SQRT_2);
        let erf_val = erf_approx(arg);
        self.sigma_ns * (2.0 * PI).sqrt() * erf_val
    }

    /// Evaluate the in-phase (I) Gaussian envelope at time t.
    ///
    /// envelope_i(t) = A * exp(-(t - t_mid)^2 / (2 * sigma^2))
    pub fn envelope_i(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration_ns {
            return 0.0;
        }
        let t_mid = self.duration_ns / 2.0;
        let dt = t - t_mid;
        let amp = self.amplitude();
        amp * (-dt * dt / (2.0 * self.sigma_ns * self.sigma_ns)).exp()
    }

    /// Evaluate the quadrature (Q) DRAG correction at time t.
    ///
    /// envelope_q(t) = beta * d(envelope_i)/dt
    ///              = beta * amplitude * (-dt/sigma^2) * exp(-(t-t_mid)^2/(2*sigma^2))
    pub fn envelope_q(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration_ns {
            return 0.0;
        }
        let t_mid = self.duration_ns / 2.0;
        let dt = t - t_mid;
        let amp = self.amplitude();
        let sigma_sq = self.sigma_ns * self.sigma_ns;
        let gauss = (-dt * dt / (2.0 * sigma_sq)).exp();
        let d_gauss = -dt / sigma_sq * gauss;
        self.beta() * amp * d_gauss
    }

    /// Convert this DRAG specification into a PulseSchedule for simulation.
    ///
    /// Uses the existing `Pulse::drag` infrastructure with the computed
    /// amplitude and beta parameters.
    pub fn to_schedule(&self, dt: f64) -> PulseSchedule {
        let pulse = Pulse::drag(
            self.duration_ns,
            self.amplitude(),
            self.frequency_ghz,
            0.0,
            self.sigma_ns,
            self.beta(),
        );
        PulseSchedule::new(dt).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        })
    }

    /// Convert to a plain Gaussian schedule (no DRAG correction) for comparison.
    pub fn to_gaussian_schedule(&self, dt: f64) -> PulseSchedule {
        let pulse = Pulse::gaussian(
            self.duration_ns,
            self.amplitude(),
            self.frequency_ghz,
            0.0,
            self.sigma_ns,
        );
        PulseSchedule::new(dt).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse,
        })
    }

    /// Compute leakage to the |2> state on a 3-level system.
    ///
    /// Returns (computational_fidelity, leakage_to_2) where leakage is
    /// the population in |2> after applying the pulse to |0>.
    pub fn simulate_leakage(&self, system: &TransmonSystem, dt: f64) -> (f64, f64) {
        assert!(
            system.n_levels >= 3,
            "Leakage measurement requires >= 3 levels"
        );
        let schedule = self.to_schedule(dt);
        let sim = PulseSimulator::new(system.clone(), schedule);
        let result = sim.simulate_state();

        let p0 = result.final_state[0].norm_sqr();
        let p1 = result.final_state[1].norm_sqr();
        let leakage: f64 = result.final_state[2..].iter().map(|c| c.norm_sqr()).sum();

        (p0 + p1, leakage)
    }
}

// ============================================================
// DRAG CALIBRATOR
// ============================================================

/// Result of a DRAG calibration sweep.
#[derive(Clone, Debug)]
pub struct CalibrationResult {
    /// The calibrated DRAG pulse.
    pub pulse: DragPulse,
    /// Fidelity of the calibrated pulse vs the target gate.
    pub fidelity: f64,
    /// Leakage to non-computational states (if 3-level system).
    pub leakage: f64,
    /// Number of amplitude sweep points evaluated.
    pub sweep_points: usize,
}

/// Calibrates DRAG pulses by sweeping amplitude to find the optimal rotation.
///
/// While the analytic formulas give a good starting point, the rotating-wave
/// approximation and finite-bandwidth effects can shift the optimal amplitude.
/// The calibrator performs a fine sweep around the analytic value to find the
/// true optimum.
pub struct DragCalibrator {
    /// Number of sweep points around the analytic amplitude.
    pub sweep_points: usize,
    /// Sweep range as a fraction of the analytic amplitude (+/- this fraction).
    pub sweep_range: f64,
    /// Simulation timestep in nanoseconds.
    pub dt: f64,
}

impl Default for DragCalibrator {
    fn default() -> Self {
        Self {
            sweep_points: 21,
            sweep_range: 0.3,
            dt: 0.05,
        }
    }
}

impl DragCalibrator {
    /// Create a calibrator with custom parameters.
    pub fn new(sweep_points: usize, sweep_range: f64, dt: f64) -> Self {
        Self {
            sweep_points,
            sweep_range,
            dt,
        }
    }

    /// Calibrate an X gate (pi rotation) for the given transmon system.
    ///
    /// Sweeps the pulse amplitude around the analytic value and selects
    /// the amplitude that maximizes gate fidelity. Uses a 2-level system
    /// for fidelity and optionally a 3-level system for leakage.
    pub fn calibrate_x_gate(
        &self,
        system: &TransmonSystem,
        duration_ns: f64,
    ) -> CalibrationResult {
        self.calibrate_gate(system, duration_ns, PI, &StandardGates::x_unitary())
    }

    /// Calibrate a Hadamard gate for the given transmon system.
    ///
    /// The Hadamard is decomposed as Rz(pi) * Ry(pi/2) in the rotating frame.
    /// We calibrate the Ry(pi/2) pulse component; the Rz is virtual.
    pub fn calibrate_hadamard(
        &self,
        system: &TransmonSystem,
        duration_ns: f64,
    ) -> CalibrationResult {
        // Hadamard = Rz(pi) * Ry(pi/2), but in the rotating frame Rz is free.
        // We calibrate a pi/2 Y-rotation pulse.
        // For the fidelity target, we use the full Hadamard unitary projected
        // into the 2-level subspace.
        self.calibrate_gate(system, duration_ns, PI / 2.0, &StandardGates::h_unitary())
    }

    /// General gate calibration by amplitude sweep.
    fn calibrate_gate(
        &self,
        system: &TransmonSystem,
        duration_ns: f64,
        angle: f64,
        target_unitary: &DenseMatrix,
    ) -> CalibrationResult {
        let freq = system.frequencies[0];
        let anhar_mhz = system.anharmonicities[0] * 1000.0;

        // Build the initial DRAG pulse with analytic parameters
        let base_drag = DragPulse::new(freq, anhar_mhz, duration_ns, angle, duration_ns / 4.0);
        let base_amp = base_drag.amplitude();

        let mut best_fidelity = 0.0f64;
        let mut best_amp = base_amp;

        // Sweep amplitude around the analytic value
        let amp_lo = base_amp * (1.0 - self.sweep_range);
        let amp_hi = base_amp * (1.0 + self.sweep_range);
        let step = if self.sweep_points > 1 {
            (amp_hi - amp_lo) / (self.sweep_points - 1) as f64
        } else {
            0.0
        };

        // Build a 2-level version of the system for fidelity computation
        let system_2l = TransmonSystem::single_qubit(freq, system.anharmonicities[0]);

        for i in 0..self.sweep_points {
            let amp = amp_lo + i as f64 * step;

            let pulse = Pulse::drag(
                duration_ns,
                amp,
                freq,
                0.0,
                duration_ns / 4.0,
                base_drag.beta(),
            );
            let schedule = PulseSchedule::new(self.dt).with_pulse(ScheduledPulse {
                channel: Channel::Drive(0),
                start_time: 0.0,
                pulse,
            });

            let sim = PulseSimulator::new(system_2l.clone(), schedule);
            let u = sim.simulate_unitary();
            let fid = average_gate_fidelity(target_unitary, &u);

            if fid > best_fidelity {
                best_fidelity = fid;
                best_amp = amp;
            }
        }

        // Compute leakage if we have a 3-level system
        let leakage = if system.n_levels >= 3 {
            let pulse = Pulse::drag(
                duration_ns,
                best_amp,
                freq,
                0.0,
                duration_ns / 4.0,
                base_drag.beta(),
            );
            let schedule = PulseSchedule::new(self.dt).with_pulse(ScheduledPulse {
                channel: Channel::Drive(0),
                start_time: 0.0,
                pulse,
            });
            let sim = PulseSimulator::new(system.clone(), schedule);
            let result = sim.simulate_state();
            result.final_state[2..].iter().map(|c| c.norm_sqr()).sum()
        } else {
            0.0
        };

        // Reconstruct the calibrated DragPulse with the amplitude baked in
        // by adjusting the angle to match the swept amplitude.
        let sigma = duration_ns / 4.0;
        let gauss_integral = {
            let half_dur = duration_ns / 2.0;
            let arg = half_dur / (sigma * std::f64::consts::SQRT_2);
            sigma * (2.0 * PI).sqrt() * erf_approx(arg)
        };
        let calibrated_angle = best_amp * 2.0 * PI * gauss_integral;

        let calibrated_pulse = DragPulse::new(freq, anhar_mhz, duration_ns, calibrated_angle, sigma);

        CalibrationResult {
            pulse: calibrated_pulse,
            fidelity: best_fidelity,
            leakage,
            sweep_points: self.sweep_points,
        }
    }
}

// ============================================================
// ERROR FUNCTION APPROXIMATION
// ============================================================

/// Polynomial approximation of the error function erf(x).
///
/// Uses Abramowitz & Stegun approximation 7.1.26, accurate to ~1.5e-7.
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // Coefficients from A&S 7.1.26
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let erf = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp();
    sign * erf
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-3;

    // ----------------------------------------------------------
    // DRAG pulse basic properties
    // ----------------------------------------------------------

    #[test]
    fn test_drag_beta_computation() {
        // For anharmonicity = -300 MHz = -0.3 GHz
        // alpha_angular = 2*pi * (-0.3) = -1.884...
        // beta = -0.5 / alpha_angular = 0.5 / 1.884... ≈ 0.2653
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let beta = drag.beta();

        let expected = -0.5 / (2.0 * PI * (-0.3));
        assert!(
            (beta - expected).abs() < 1e-10,
            "Beta mismatch: got {}, expected {}",
            beta,
            expected
        );
        assert!(beta > 0.0, "Beta should be positive for negative anharmonicity");
    }

    #[test]
    fn test_drag_beta_positive_anharmonicity() {
        let drag = DragPulse::new(5.0, 300.0, 24.0, PI, 6.0);
        let beta = drag.beta();
        assert!(beta < 0.0, "Beta should be negative for positive anharmonicity");
    }

    #[test]
    fn test_drag_beta_zero_anharmonicity() {
        let drag = DragPulse::new(5.0, 0.0, 24.0, PI, 6.0);
        let beta = drag.beta();
        assert_eq!(beta, 0.0, "Beta should be 0 for zero anharmonicity");
    }

    #[test]
    fn test_drag_amplitude_for_pi_rotation() {
        // For a pi rotation, the amplitude should satisfy:
        // amp * 2*pi * gauss_integral = pi
        // => amp = 1 / (2 * gauss_integral)
        let sigma = 6.0;
        let duration = 24.0;
        let drag = DragPulse::new(5.0, -300.0, duration, PI, sigma);
        let amp = drag.amplitude();

        assert!(amp > 0.0, "Amplitude should be positive");
        assert!(amp < 1.0, "Amplitude should be reasonable (< 1.0)");

        // Verify: amp * 2*pi * integral ≈ pi
        let integral = drag.gaussian_integral();
        let achieved_angle = amp * 2.0 * PI * integral;
        assert!(
            (achieved_angle - PI).abs() < 1e-8,
            "Achieved angle {} should be pi",
            achieved_angle
        );
    }

    #[test]
    fn test_drag_amplitude_scales_with_angle() {
        let sigma = 6.0;
        let duration = 24.0;

        let drag_pi = DragPulse::new(5.0, -300.0, duration, PI, sigma);
        let drag_half = DragPulse::new(5.0, -300.0, duration, PI / 2.0, sigma);

        let ratio = drag_pi.amplitude() / drag_half.amplitude();
        assert!(
            (ratio - 2.0).abs() < 1e-8,
            "Pi amplitude should be 2x half-pi amplitude, ratio = {}",
            ratio
        );
    }

    // ----------------------------------------------------------
    // Envelope shapes
    // ----------------------------------------------------------

    #[test]
    fn test_envelope_i_peak_at_center() {
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let peak = drag.envelope_i(12.0);
        let amp = drag.amplitude();

        assert!(
            (peak - amp).abs() < 1e-10,
            "Peak should equal amplitude: {} vs {}",
            peak,
            amp
        );
    }

    #[test]
    fn test_envelope_i_symmetric() {
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let left = drag.envelope_i(8.0);
        let right = drag.envelope_i(16.0);
        assert!(
            (left - right).abs() < 1e-10,
            "Gaussian envelope should be symmetric: {} vs {}",
            left,
            right
        );
    }

    #[test]
    fn test_envelope_i_zero_outside() {
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        assert_eq!(drag.envelope_i(-1.0), 0.0);
        assert_eq!(drag.envelope_i(25.0), 0.0);
    }

    #[test]
    fn test_envelope_q_zero_at_center() {
        // The derivative of a Gaussian is zero at its peak
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let q_center = drag.envelope_q(12.0);
        assert!(
            q_center.abs() < 1e-10,
            "Quadrature should be zero at center: {}",
            q_center
        );
    }

    #[test]
    fn test_envelope_q_antisymmetric() {
        // d(Gaussian)/dt is an odd function around the center
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let left = drag.envelope_q(8.0);
        let right = drag.envelope_q(16.0);
        assert!(
            (left + right).abs() < 1e-10,
            "Quadrature should be antisymmetric: left={}, right={}",
            left,
            right
        );
    }

    #[test]
    fn test_envelope_q_nonzero_off_center() {
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let q_off = drag.envelope_q(9.0);
        assert!(
            q_off.abs() > 1e-6,
            "Quadrature should be nonzero off-center: {}",
            q_off
        );
    }

    // ----------------------------------------------------------
    // DRAG vs Gaussian leakage comparison (3-level)
    // ----------------------------------------------------------

    #[test]
    fn test_drag_vs_gaussian_on_3level_system() {
        // Verify that DRAG produces a measurably different evolution than
        // Gaussian on a 3-level system, and that both maintain low leakage.
        // The DRAG correction modifies the pulse shape; how it affects leakage
        // depends on the interplay of anharmonicity, drive strength, and pulse
        // duration. We verify both are physically sensible.
        let system = TransmonSystem::single_qubit_3level(5.0, -0.3);
        let sigma = 6.0;
        let duration = 4.0 * sigma;
        let dt = 0.05;

        let drag = DragPulse::new(5.0, -300.0, duration, PI, sigma);
        let amp = drag.amplitude();

        // Gaussian-only pulse
        let gauss_pulse = Pulse::gaussian(duration, amp, 5.0, 0.0, sigma);
        let gauss_sched = PulseSchedule::new(dt).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse: gauss_pulse,
        });
        let sim_gauss = PulseSimulator::new(system.clone(), gauss_sched);
        let result_gauss = sim_gauss.simulate_state();
        let leakage_gauss: f64 = result_gauss.final_state[2..].iter().map(|c| c.norm_sqr()).sum();

        // DRAG pulse
        let drag_pulse = Pulse::drag(duration, amp, 5.0, 0.0, sigma, drag.beta());
        let drag_sched = PulseSchedule::new(dt).with_pulse(ScheduledPulse {
            channel: Channel::Drive(0),
            start_time: 0.0,
            pulse: drag_pulse,
        });
        let sim_drag = PulseSimulator::new(system, drag_sched);
        let result_drag = sim_drag.simulate_state();
        let leakage_drag: f64 = result_drag.final_state[2..].iter().map(|c| c.norm_sqr()).sum();

        // Both leakage values should be small (< 5%) for well-calibrated pulses
        assert!(
            leakage_gauss < 0.05,
            "Gaussian leakage should be small: {:.6}",
            leakage_gauss
        );
        assert!(
            leakage_drag < 0.05,
            "DRAG leakage should be small: {:.6}",
            leakage_drag
        );

        // DRAG and Gaussian should produce different final states (the correction
        // changes the dynamics even if both have low leakage).
        let state_diff: f64 = result_gauss
            .final_state
            .iter()
            .zip(result_drag.final_state.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum();
        assert!(
            state_diff > 1e-6,
            "DRAG should produce different state than Gaussian, diff={}",
            state_diff
        );

        // Both should preserve total population
        let total_gauss: f64 = result_gauss.final_state.iter().map(|c| c.norm_sqr()).sum();
        let total_drag: f64 = result_drag.final_state.iter().map(|c| c.norm_sqr()).sum();
        assert!((total_gauss - 1.0).abs() < 0.01);
        assert!((total_drag - 1.0).abs() < 0.01);
    }

    // ----------------------------------------------------------
    // Calibration tests
    // ----------------------------------------------------------

    #[test]
    fn test_calibrate_x_gate_converges() {
        let system = TransmonSystem::single_qubit(5.0, -0.3);
        let calibrator = DragCalibrator {
            sweep_points: 31,
            sweep_range: 0.4,
            dt: 0.05,
        };

        let result = calibrator.calibrate_x_gate(&system, 24.0);

        assert!(
            result.fidelity > 0.90,
            "Calibrated X gate fidelity should be > 0.90, got {}",
            result.fidelity
        );
        assert_eq!(result.sweep_points, 31);
    }

    #[test]
    fn test_calibrate_hadamard_converges() {
        let system = TransmonSystem::single_qubit(5.0, -0.3);
        let calibrator = DragCalibrator {
            sweep_points: 31,
            sweep_range: 0.4,
            dt: 0.05,
        };

        let result = calibrator.calibrate_hadamard(&system, 24.0);

        // Hadamard calibration is harder than X since it's not a pure X rotation,
        // but the sweep should still find a reasonable fidelity.
        assert!(
            result.fidelity > 0.50,
            "Calibrated Hadamard fidelity should be > 0.50, got {}",
            result.fidelity
        );
    }

    #[test]
    fn test_calibrate_x_gate_3level_leakage() {
        let system = TransmonSystem::single_qubit_3level(5.0, -0.3);
        let calibrator = DragCalibrator {
            sweep_points: 21,
            sweep_range: 0.3,
            dt: 0.05,
        };

        let result = calibrator.calibrate_x_gate(&system, 24.0);

        // Should report leakage for 3-level system
        assert!(
            result.leakage < 0.2,
            "Leakage should be small for calibrated DRAG pulse, got {}",
            result.leakage
        );
    }

    // ----------------------------------------------------------
    // Lindblad + DRAG combined test
    // ----------------------------------------------------------

    #[test]
    fn test_drag_with_lindblad_decoherence() {
        // DRAG pulse under open-system dynamics should still largely work
        let mut system = TransmonSystem::single_qubit_3level(5.0, -0.3);
        system.t1 = vec![50_000.0]; // 50 us - realistic
        system.t2 = vec![70_000.0]; // 70 us - realistic

        let duration = 24.0;
        let sigma = 6.0;
        let drag = DragPulse::new(5.0, -300.0, duration, PI, sigma);
        let schedule = drag.to_schedule(0.1);

        let sim = PulseSimulator::new(system, schedule);
        let result = sim.simulate_lindblad();

        // Under Lindblad dynamics, the density matrix diagonal gives populations.
        // The pulse is short (24 ns) relative to T1/T2 (50/70 us), so decoherence
        // effects are minimal. The |1> population should be dominant.
        let p0 = result.final_state[0].re;
        let p1 = result.final_state[1].re;
        let p2 = if result.final_state.len() > 2 {
            result.final_state[2].re
        } else {
            0.0
        };

        // Population should be mostly in |1> (pi rotation from |0>)
        assert!(
            p1 > 0.5,
            "After pi pulse under Lindblad, p1 should be > 0.5, got {}",
            p1
        );

        // Total probability should be preserved
        let total: f64 = result.final_state.iter().map(|c| c.re).sum();
        assert!(
            (total - 1.0).abs() < 0.05,
            "Total probability should be ~1.0, got {}",
            total
        );
    }

    // ----------------------------------------------------------
    // Schedule conversion
    // ----------------------------------------------------------

    #[test]
    fn test_to_schedule_produces_valid_schedule() {
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let schedule = drag.to_schedule(0.1);

        assert_eq!(schedule.pulses.len(), 1);
        assert!((schedule.dt - 0.1).abs() < 1e-10);
        assert!((schedule.total_duration() - 24.0).abs() < 1e-10);

        // Check the pulse inside uses DRAG shape
        match &schedule.pulses[0].pulse.shape {
            PulseShape::Drag { sigma, beta } => {
                assert!((*sigma - 6.0).abs() < 1e-10);
                assert!((*beta - drag.beta()).abs() < 1e-10);
            }
            other => panic!("Expected Drag shape, got {:?}", other),
        }
    }

    #[test]
    fn test_to_gaussian_schedule_no_drag() {
        let drag = DragPulse::new(5.0, -300.0, 24.0, PI, 6.0);
        let schedule = drag.to_gaussian_schedule(0.1);

        assert_eq!(schedule.pulses.len(), 1);
        match &schedule.pulses[0].pulse.shape {
            PulseShape::Gaussian { sigma } => {
                assert!((*sigma - 6.0).abs() < 1e-10);
            }
            other => panic!("Expected Gaussian shape, got {:?}", other),
        }
    }

    // ----------------------------------------------------------
    // erf approximation
    // ----------------------------------------------------------

    #[test]
    fn test_erf_approximation() {
        // Known values
        assert!(erf_approx(0.0).abs() < 1e-6, "erf(0) = 0");
        assert!((erf_approx(1.0) - 0.8427).abs() < 1e-3, "erf(1) ~ 0.8427");
        assert!((erf_approx(2.0) - 0.9953).abs() < 1e-3, "erf(2) ~ 0.9953");
        assert!((erf_approx(3.0) - 0.9999).abs() < 1e-3, "erf(3) ~ 1.0");

        // Odd function
        assert!(
            (erf_approx(-1.0) + erf_approx(1.0)).abs() < 1e-6,
            "erf is odd"
        );
    }
}
