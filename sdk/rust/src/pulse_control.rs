//! Pulse-Level Control for Quantum Hardware
//!
//! Provides low-level microwave pulse control for quantum gates.
//! This allows direct manipulation of control pulses instead of
//! abstract gate operations.
//!
//! # Why Pulse Control?
//!
//! Real quantum hardware uses microwave pulses to manipulate qubits.
//! Gate-level operations are abstractions. Pulse-level control enables:
//!
//! - **Custom gates**: Implement gates not in standard library
//! - **Error mitigation**: Shape pulses to reduce errors
//! - **Calibration**: Fine-tune gate parameters
//! - **Cross-talk reduction**: Optimize pulse timing
//! - **Shorter circuits**: Combine operations into single pulses
//!
//! # Example
//!
//! ```rust,ignore
//! use nqpu_metal::pulse_control::{PulseBuilder, PulseShape, DriveChannel};
//!
//! // Create a custom X gate pulse
//! let mut builder = PulseBuilder::new();
//!
//! // Add a Gaussian pulse on drive channel 0
//! builder.add_pulse(
//!     DriveChannel(0),
//!     PulseShape::Gaussian {
//!         duration: 160,  // 160 dt (dt ~0.222ns on IBM)
//!         sigma: 40,
//!         amplitude: 0.2,
//!     }
//! );
//!
//! // Convert to schedule
//! let schedule = builder.build();
//!
//! // Simulate pulse execution
//! let result = schedule.simulate();
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PULSE TYPES
// ---------------------------------------------------------------------------

/// Drive channel identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DriveChannel(pub usize);

/// Measure channel identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeasureChannel(pub usize);

/// Acquire channel for readout
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AcquireChannel(pub usize);

/// Pulse shape definition
#[derive(Clone, Debug)]
pub enum PulseShape {
    /// Constant amplitude pulse
    Constant {
        duration: usize,
        amplitude: f64,
    },
    /// Gaussian pulse
    Gaussian {
        duration: usize,
        sigma: f64,
        amplitude: f64,
    },
    /// Gaussian with DRAG correction
    GaussianSquare {
        duration: usize,
        sigma: f64,
        width: f64,
        amplitude: f64,
    },
    /// DRAG pulse (error-corrected Gaussian)
    Drag {
        duration: usize,
        sigma: f64,
        amplitude: f64,
        beta: f64,  // DRAG parameter
    },
    /// Custom waveform
    Custom {
        samples: Vec<f64>,
    },
}

impl PulseShape {
    /// Generate waveform samples
    pub fn samples(&self, _dt: f64) -> Vec<f64> {
        match self {
            PulseShape::Constant { duration, amplitude } => {
                vec![*amplitude; *duration]
            }
            PulseShape::Gaussian { duration, sigma, amplitude } => {
                let center = (*duration as f64 - 1.0) / 2.0;
                (0..*duration)
                    .map(|t| {
                        let x = t as f64 - center;
                        amplitude * (-x * x / (2.0 * sigma * sigma)).exp()
                    })
                    .collect()
            }
            PulseShape::Drag { duration, sigma, amplitude, beta } => {
                let center = (*duration as f64 - 1.0) / 2.0;
                (0..*duration)
                    .map(|t| {
                        let x = t as f64 - center;
                        let gauss = (-x * x / (2.0 * sigma * sigma)).exp();
                        // DRAG correction: add derivative of Gaussian
                        amplitude * gauss + beta * x / (sigma * sigma) * amplitude * gauss
                    })
                    .collect()
            }
            PulseShape::GaussianSquare { duration, sigma, width, amplitude } => {
                let center = (*duration as f64 - 1.0) / 2.0;
                let half_width = width / 2.0;
                (0..*duration)
                    .map(|t| {
                        let x = t as f64 - center;
                        if x.abs() <= half_width {
                            *amplitude
                        } else {
                            let edge = x.signum() * half_width;
                            let gauss_x = x - edge;
                            amplitude * (-gauss_x * gauss_x / (2.0 * sigma * sigma)).exp()
                        }
                    })
                    .collect()
            }
            PulseShape::Custom { samples } => samples.clone(),
        }
    }

    /// Get duration in dt units
    pub fn duration(&self) -> usize {
        match self {
            PulseShape::Constant { duration, .. } => *duration,
            PulseShape::Gaussian { duration, .. } => *duration,
            PulseShape::GaussianSquare { duration, .. } => *duration,
            PulseShape::Drag { duration, .. } => *duration,
            PulseShape::Custom { samples } => samples.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// PULSE INSTRUCTION
// ---------------------------------------------------------------------------

/// A single pulse instruction
#[derive(Clone, Debug)]
pub struct PulseInstruction {
    /// Channel to apply pulse to
    pub channel: Channel,
    /// Pulse shape
    pub shape: PulseShape,
    /// Start time in dt
    pub start_time: usize,
    /// Phase (for frame tracking)
    pub phase: f64,
    /// Frequency (MHz)
    pub frequency: f64,
}

/// Channel type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    Drive(DriveChannel),
    Measure(MeasureChannel),
    Acquire(AcquireChannel),
}

// ---------------------------------------------------------------------------
// PULSE SCHEDULE
// ---------------------------------------------------------------------------

/// Complete pulse schedule
#[derive(Clone, Debug)]
pub struct PulseSchedule {
    /// Instructions sorted by time
    instructions: Vec<PulseInstruction>,
    /// Total duration in dt
    duration: usize,
    /// Number of qubits
    n_qubits: usize,
    /// dt in nanoseconds (typically ~0.222ns for IBM)
    dt_ns: f64,
}

impl PulseSchedule {
    /// Create empty schedule
    pub fn new(n_qubits: usize) -> Self {
        Self {
            instructions: Vec::new(),
            duration: 0,
            n_qubits,
            dt_ns: 0.222,  // Default IBM dt
        }
    }

    /// Add instruction
    pub fn add(&mut self, instruction: PulseInstruction) {
        let end_time = instruction.start_time + instruction.shape.duration();
        self.duration = self.duration.max(end_time);
        self.instructions.push(instruction);
    }

    /// Get total duration
    pub fn duration(&self) -> usize {
        self.duration
    }

    /// Get duration in nanoseconds
    pub fn duration_ns(&self) -> f64 {
        self.duration as f64 * self.dt_ns
    }

    /// Get instructions in time order
    pub fn instructions(&self) -> &[PulseInstruction] {
        &self.instructions
    }

    /// Simulate pulse execution
    pub fn simulate(&self) -> PulseSimulationResult {
        let mut state = vec![0.0f64; self.duration];
        let mut qubit_state = QubitState::new(self.n_qubits);

        // Apply each pulse
        for inst in &self.instructions {
            if let Channel::Drive(DriveChannel(q)) = inst.channel {
                let samples = inst.shape.samples(self.dt_ns);
                for (t, amp) in samples.iter().enumerate() {
                    let time = inst.start_time + t;
                    if time < state.len() {
                        state[time] += amp * inst.phase.cos();
                    }
                }

                // Update qubit state (simplified)
                let total_rotation: f64 = samples.iter().sum();
                qubit_state.apply_rotation(q, total_rotation * std::f64::consts::PI);
            }
        }

        PulseSimulationResult {
            waveform: state,
            final_state: qubit_state,
            duration_ns: self.duration_ns(),
        }
    }
}

/// Result of pulse simulation
#[derive(Clone, Debug)]
pub struct PulseSimulationResult {
    /// Combined waveform
    pub waveform: Vec<f64>,
    /// Final qubit state
    pub final_state: QubitState,
    /// Total duration in nanoseconds
    pub duration_ns: f64,
}

/// Simple qubit state tracker
#[derive(Clone, Debug)]
pub struct QubitState {
    n_qubits: usize,
    // Bloch sphere angles
    thetas: Vec<f64>,  // Polar angles
    phis: Vec<f64>,    // Azimuthal angles
}

impl QubitState {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            thetas: vec![0.0; n_qubits],  // All in |0⟩ state
            phis: vec![0.0; n_qubits],
        }
    }

    pub fn apply_rotation(&mut self, qubit: usize, angle: f64) {
        if qubit < self.n_qubits {
            self.thetas[qubit] = (self.thetas[qubit] + angle).clamp(0.0, std::f64::consts::PI);
        }
    }

    pub fn get_bloch_coords(&self, qubit: usize) -> Option<(f64, f64, f64)> {
        if qubit >= self.n_qubits {
            return None;
        }
        let theta = self.thetas[qubit];
        let phi = self.phis[qubit];
        Some((
            theta.sin() * phi.cos(),  // x
            theta.sin() * phi.sin(),  // y
            theta.cos(),               // z
        ))
    }

    pub fn probability_one(&self, qubit: usize) -> f64 {
        if qubit >= self.n_qubits {
            return 0.0;
        }
        (self.thetas[qubit] / 2.0).sin().powi(2)
    }
}

// ---------------------------------------------------------------------------
// PULSE BUILDER
// ---------------------------------------------------------------------------

/// Builder for creating pulse schedules
pub struct PulseBuilder {
    n_qubits: usize,
    current_time: usize,
    instructions: Vec<PulseInstruction>,
    phases: HashMap<usize, f64>,
    frequencies: HashMap<usize, f64>,
}

impl PulseBuilder {
    /// Create new builder
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            current_time: 0,
            instructions: Vec::new(),
            phases: HashMap::new(),
            frequencies: HashMap::new(),
        }
    }

    /// Set qubit frequency
    pub fn set_frequency(mut self, qubit: usize, freq_mhz: f64) -> Self {
        self.frequencies.insert(qubit, freq_mhz);
        self
    }

    /// Add pulse on drive channel
    pub fn add_pulse(mut self, channel: DriveChannel, shape: PulseShape) -> Self {
        let qubit = channel.0;
        let instruction = PulseInstruction {
            channel: Channel::Drive(channel),
            shape,
            start_time: self.current_time,
            phase: self.phases.get(&qubit).copied().unwrap_or(0.0),
            frequency: self.frequencies.get(&qubit).copied().unwrap_or(5000.0),
        };
        self.current_time += instruction.shape.duration();
        self.instructions.push(instruction);
        self
    }

    /// Add delay
    pub fn delay(mut self, duration: usize) -> Self {
        self.current_time += duration;
        self
    }

    /// Shift phase
    pub fn shift_phase(mut self, qubit: usize, phase: f64) -> Self {
        *self.phases.entry(qubit).or_insert(0.0) += phase;
        self
    }

    /// Set phase
    pub fn set_phase(mut self, qubit: usize, phase: f64) -> Self {
        self.phases.insert(qubit, phase);
        self
    }

    /// Build final schedule
    pub fn build(self) -> PulseSchedule {
        let mut schedule = PulseSchedule::new(self.n_qubits);
        for inst in self.instructions {
            schedule.add(inst);
        }
        schedule
    }
}

// ---------------------------------------------------------------------------
// STANDARD PULSE GATES
// ---------------------------------------------------------------------------

/// Create X gate pulse (π rotation)
pub fn x_pulse(qubit: usize, duration: usize, amplitude: f64) -> PulseInstruction {
    PulseInstruction {
        channel: Channel::Drive(DriveChannel(qubit)),
        shape: PulseShape::Gaussian {
            duration,
            sigma: duration as f64 / 4.0,
            amplitude,
        },
        start_time: 0,
        phase: 0.0,
        frequency: 5000.0,
    }
}

/// Create Y gate pulse (π rotation with phase)
pub fn y_pulse(qubit: usize, duration: usize, amplitude: f64) -> PulseInstruction {
    PulseInstruction {
        channel: Channel::Drive(DriveChannel(qubit)),
        shape: PulseShape::Gaussian {
            duration,
            sigma: duration as f64 / 4.0,
            amplitude,
        },
        start_time: 0,
        phase: std::f64::consts::PI / 2.0,
        frequency: 5000.0,
    }
}

/// Create rotation pulse
pub fn rotation_pulse(qubit: usize, angle: f64, duration: usize) -> PulseInstruction {
    // Amplitude calibrated for π rotation = 0.2
    let amplitude = 0.2 * angle / std::f64::consts::PI;
    PulseInstruction {
        channel: Channel::Drive(DriveChannel(qubit)),
        shape: PulseShape::Drag {
            duration,
            sigma: duration as f64 / 4.0,
            amplitude,
            beta: 1.0,  // DRAG parameter
        },
        start_time: 0,
        phase: 0.0,
        frequency: 5000.0,
    }
}

/// Create cross-resonance pulse for CX gate
pub fn cx_pulse(control: usize, target: usize, duration: usize) -> Vec<PulseInstruction> {
    vec![
        // CR pulse on target
        PulseInstruction {
            channel: Channel::Drive(DriveChannel(target)),
            shape: PulseShape::GaussianSquare {
                duration,
                sigma: duration as f64 / 8.0,
                width: duration as f64 / 2.0,
                amplitude: 0.1,
            },
            start_time: 0,
            phase: 0.0,
            frequency: 5000.0,
        },
        // Echo pulse on control
        PulseInstruction {
            channel: Channel::Drive(DriveChannel(control)),
            shape: PulseShape::Gaussian {
                duration: duration / 4,
                sigma: duration as f64 / 16.0,
                amplitude: 0.2,
            },
            start_time: duration / 4,
            phase: 0.0,
            frequency: 5000.0,
        },
    ]
}

// ---------------------------------------------------------------------------
// PULSE CALIBRATION
// ---------------------------------------------------------------------------

/// Calibration data for a qubit
#[derive(Clone, Debug)]
pub struct QubitCalibration {
    /// Qubit index
    pub qubit: usize,
    /// Resonant frequency (MHz)
    pub frequency: f64,
    /// Anharmonicity (MHz)
    pub anharmonicity: f64,
    /// π pulse amplitude
    pub pi_pulse_amplitude: f64,
    /// π pulse duration (dt)
    pub pi_pulse_duration: usize,
    /// T1 time (μs)
    pub t1: f64,
    /// T2 time (μs)
    pub t2: f64,
    /// Readout frequency (MHz)
    pub readout_frequency: f64,
    /// Readout amplitude
    pub readout_amplitude: f64,
}

impl QubitCalibration {
    /// Create default calibration
    pub fn new(qubit: usize) -> Self {
        Self {
            qubit,
            frequency: 5000.0,
            anharmonicity: -340.0,
            pi_pulse_amplitude: 0.2,
            pi_pulse_duration: 160,
            t1: 100.0,
            t2: 80.0,
            readout_frequency: 7000.0,
            readout_amplitude: 0.1,
        }
    }

    /// Generate calibrated X pulse
    pub fn x_pulse(&self) -> PulseInstruction {
        PulseInstruction {
            channel: Channel::Drive(DriveChannel(self.qubit)),
            shape: PulseShape::Drag {
                duration: self.pi_pulse_duration,
                sigma: self.pi_pulse_duration as f64 / 4.0,
                amplitude: self.pi_pulse_amplitude,
                beta: 1.0,
            },
            start_time: 0,
            phase: 0.0,
            frequency: self.frequency,
        }
    }

    /// Generate calibrated rotation pulse
    pub fn rotation_pulse(&self, angle: f64) -> PulseInstruction {
        let amplitude = self.pi_pulse_amplitude * angle / std::f64::consts::PI;
        PulseInstruction {
            channel: Channel::Drive(DriveChannel(self.qubit)),
            shape: PulseShape::Drag {
                duration: self.pi_pulse_duration,
                sigma: self.pi_pulse_duration as f64 / 4.0,
                amplitude,
                beta: 1.0,
            },
            start_time: 0,
            phase: 0.0,
            frequency: self.frequency,
        }
    }
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_shape_constant() {
        let shape = PulseShape::Constant { duration: 10, amplitude: 0.5 };
        let samples = shape.samples(0.222);
        assert_eq!(samples.len(), 10);
        assert!((samples[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_pulse_shape_gaussian() {
        let shape = PulseShape::Gaussian {
            duration: 50,
            sigma: 10.0,
            amplitude: 0.2,
        };
        let samples = shape.samples(0.222);
        assert_eq!(samples.len(), 50);
        // Peak should be in the middle
        let mid = samples[25];
        assert!(mid > samples[0]);
    }

    #[test]
    fn test_pulse_schedule() {
        let mut schedule = PulseSchedule::new(2);
        schedule.add(x_pulse(0, 160, 0.2));

        assert!(schedule.duration() > 0);
    }

    #[test]
    fn test_pulse_builder() {
        let schedule = PulseBuilder::new(2)
            .add_pulse(DriveChannel(0), PulseShape::Constant { duration: 100, amplitude: 0.2 })
            .delay(50)
            .add_pulse(DriveChannel(1), PulseShape::Constant { duration: 100, amplitude: 0.2 })
            .build();

        assert_eq!(schedule.duration(), 250);
        assert_eq!(schedule.instructions().len(), 2);
    }

    #[test]
    fn test_qubit_state() {
        let mut state = QubitState::new(2);
        assert!((state.probability_one(0) - 0.0).abs() < 0.001);

        state.apply_rotation(0, std::f64::consts::PI);
        assert!((state.probability_one(0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calibration() {
        let cal = QubitCalibration::new(0);
        let pulse = cal.x_pulse();

        assert_eq!(pulse.shape.duration(), cal.pi_pulse_duration);
    }

    #[test]
    fn test_drag_pulse() {
        let shape = PulseShape::Drag {
            duration: 100,
            sigma: 20.0,
            amplitude: 0.2,
            beta: 1.0,
        };
        let samples = shape.samples(0.222);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_cx_pulse() {
        let pulses = cx_pulse(0, 1, 400);
        assert_eq!(pulses.len(), 2);
    }

    #[test]
    fn test_pulse_simulation() {
        let schedule = PulseBuilder::new(1)
            .add_pulse(DriveChannel(0), PulseShape::Constant { duration: 100, amplitude: 0.5 })
            .build();

        let result = schedule.simulate();
        assert!(!result.waveform.is_empty());
    }
}
