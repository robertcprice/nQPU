//! Device-specific noise models from real quantum hardware calibration data.
//!
//! Provides realistic noise profiles based on actual device parameters from
//! IBM, Google, IonQ, and Rigetti quantum processors. Supports construction
//! of noise models from calibration data, hardware presets, noise application
//! to quantum states, and calibration import/export.

use num_complex::Complex64;
use rand::Rng;
use serde_json::Value;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during device noise model construction.
#[derive(Debug, Clone)]
pub enum DeviceNoiseError {
    /// Calibration data is invalid (e.g., negative T1, T2 > 2*T1).
    InvalidCalibration(String),
    /// Referenced qubit index is out of range for the device.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// Required gate calibration data is missing.
    MissingGateData(String),
}

impl fmt::Display for DeviceNoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceNoiseError::InvalidCalibration(msg) => {
                write!(f, "Invalid calibration: {}", msg)
            }
            DeviceNoiseError::QubitOutOfRange { qubit, num_qubits } => {
                write!(
                    f,
                    "Qubit {} out of range for {}-qubit device",
                    qubit, num_qubits
                )
            }
            DeviceNoiseError::MissingGateData(msg) => {
                write!(f, "Missing gate data: {}", msg)
            }
        }
    }
}

impl std::error::Error for DeviceNoiseError {}

// ============================================================
// CALIBRATION DATA TYPES
// ============================================================

/// Physical properties of a single qubit from device calibration.
#[derive(Debug, Clone)]
pub struct QubitProperties {
    /// T1 relaxation time in microseconds.
    pub t1_us: f64,
    /// T2 dephasing time in microseconds.
    pub t2_us: f64,
    /// Qubit frequency in GHz.
    pub frequency_ghz: f64,
    /// Anharmonicity in GHz (typically negative for transmons).
    pub anharmonicity_ghz: f64,
}

/// Calibrated properties of a quantum gate.
#[derive(Debug, Clone)]
pub struct GateProperties {
    /// Name of the gate (e.g., "cx", "sx", "rz", "syc").
    pub gate_name: String,
    /// Qubit indices this gate acts on.
    pub qubits: Vec<usize>,
    /// Gate error rate from randomized benchmarking.
    pub error_rate: f64,
    /// Gate execution time in nanoseconds.
    pub gate_time_ns: f64,
}

/// Readout error characterization for a single qubit.
#[derive(Debug, Clone)]
pub struct ReadoutError {
    /// Qubit index.
    pub qubit: usize,
    /// Probability of measuring 0 given the true state is |0>.
    pub p0_given_0: f64,
    /// Probability of measuring 1 given the true state is |1>.
    pub p1_given_1: f64,
}

/// Complete calibration data for a quantum device.
#[derive(Debug, Clone)]
pub struct DeviceCalibration {
    /// Device name (e.g., "ibm_brisbane").
    pub name: String,
    /// Number of qubits on the device.
    pub num_qubits: usize,
    /// Per-qubit physical properties.
    pub qubit_properties: Vec<QubitProperties>,
    /// Gate calibration data.
    pub gate_properties: Vec<GateProperties>,
    /// Readout error data.
    pub readout_errors: Vec<ReadoutError>,
    /// Calibration timestamp (Unix epoch seconds).
    pub timestamp: u64,
}

// ============================================================
// NOISE MODEL TYPES
// ============================================================

/// Single-qubit noise channel parameters.
#[derive(Debug, Clone)]
pub struct SingleQubitNoise {
    /// Qubit index.
    pub qubit: usize,
    /// Depolarizing error rate.
    pub depolarizing_rate: f64,
    /// Dephasing error rate.
    pub dephasing_rate: f64,
}

/// Two-qubit noise channel parameters.
#[derive(Debug, Clone)]
pub struct TwoQubitNoise {
    /// Pair of qubit indices.
    pub qubits: (usize, usize),
    /// Two-qubit depolarizing error rate.
    pub depolarizing_rate: f64,
    /// Crosstalk error rate.
    pub crosstalk_rate: f64,
}

/// Readout noise channel with confusion matrix.
#[derive(Debug, Clone)]
pub struct ReadoutNoiseChannel {
    /// Qubit index.
    pub qubit: usize,
    /// 2x2 confusion matrix: `[[P(0|0), P(1|0)], [P(0|1), P(1|1)]]`.
    pub confusion_matrix: [[f64; 2]; 2],
}

/// Thermal relaxation parameters for a qubit.
#[derive(Debug, Clone)]
pub struct ThermalRelaxation {
    /// Qubit index.
    pub qubit: usize,
    /// T1 relaxation time (same units as gate_time).
    pub t1: f64,
    /// T2 dephasing time (same units as gate_time).
    pub t2: f64,
    /// Gate time duration.
    pub gate_time: f64,
    /// Excited state population at thermal equilibrium.
    pub excited_state_population: f64,
}

/// Complete noise model derived from device calibration.
#[derive(Debug, Clone)]
pub struct DeviceNoiseModel {
    /// Single-qubit noise channels.
    pub single_qubit_errors: Vec<SingleQubitNoise>,
    /// Two-qubit noise channels.
    pub two_qubit_errors: Vec<TwoQubitNoise>,
    /// Readout noise channels.
    pub readout_errors: Vec<ReadoutNoiseChannel>,
    /// Thermal relaxation channels.
    pub thermal_relaxation: Vec<ThermalRelaxation>,
}

// ============================================================
// NOISE MODEL CONSTRUCTION
// ============================================================

/// Build a complete noise model from device calibration data.
///
/// For each qubit, computes depolarizing and dephasing rates from the
/// gate error rates and T1/T2 times. For two-qubit gates, extracts
/// depolarizing rates. Readout errors are converted to confusion matrices.
pub fn build_noise_model(
    calibration: &DeviceCalibration,
) -> Result<DeviceNoiseModel, DeviceNoiseError> {
    // Validate calibration data
    validate_calibration(calibration)?;

    let mut single_qubit_errors = Vec::new();
    let mut two_qubit_errors = Vec::new();
    let mut readout_channels = Vec::new();
    let mut thermal_channels = Vec::new();

    // Build per-qubit noise from single-qubit gate errors
    for qubit_idx in 0..calibration.num_qubits {
        // Find best single-qubit gate error for this qubit
        let sq_error = calibration
            .gate_properties
            .iter()
            .filter(|g| g.qubits.len() == 1 && g.qubits[0] == qubit_idx)
            .map(|g| g.error_rate)
            .fold(f64::MAX, f64::min);

        let sq_error = if sq_error == f64::MAX { 0.0 } else { sq_error };

        // Find gate time for thermal relaxation
        let gate_time_ns = calibration
            .gate_properties
            .iter()
            .filter(|g| g.qubits.len() == 1 && g.qubits[0] == qubit_idx)
            .map(|g| g.gate_time_ns)
            .fold(f64::MAX, f64::min);
        let gate_time_ns = if gate_time_ns == f64::MAX {
            35.0
        } else {
            gate_time_ns
        };

        // Depolarizing rate from gate error: p_depol ≈ (4/3) * error_rate for single qubit
        let depolarizing_rate = (4.0 / 3.0) * sq_error;

        // Dephasing rate from T2: additional pure dephasing beyond T1 contribution
        let dephasing_rate = if qubit_idx < calibration.qubit_properties.len() {
            let qp = &calibration.qubit_properties[qubit_idx];
            let t1_ns = qp.t1_us * 1000.0;
            let t2_ns = qp.t2_us * 1000.0;
            if t2_ns > 0.0 && t1_ns > 0.0 {
                // Pure dephasing rate: 1/T_phi = 1/T2 - 1/(2*T1)
                let rate_phi = (1.0 / t2_ns) - 1.0 / (2.0 * t1_ns);
                (rate_phi * gate_time_ns).max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        single_qubit_errors.push(SingleQubitNoise {
            qubit: qubit_idx,
            depolarizing_rate,
            dephasing_rate,
        });

        // Build thermal relaxation channel
        if qubit_idx < calibration.qubit_properties.len() {
            let qp = &calibration.qubit_properties[qubit_idx];
            let tr = thermal_relaxation_channel(
                qp.t1_us * 1000.0, // convert to ns
                qp.t2_us * 1000.0,
                gate_time_ns,
            );
            thermal_channels.push(ThermalRelaxation {
                qubit: qubit_idx,
                t1: tr.t1,
                t2: tr.t2,
                gate_time: tr.gate_time,
                excited_state_population: tr.excited_state_population,
            });
        }
    }

    // Build two-qubit noise from two-qubit gate errors
    for gate in &calibration.gate_properties {
        if gate.qubits.len() == 2 {
            let q0 = gate.qubits[0];
            let q1 = gate.qubits[1];
            // Two-qubit depolarizing rate: p_depol ≈ (16/15) * error_rate
            let depolarizing_rate = (16.0 / 15.0) * gate.error_rate;
            // Crosstalk is estimated as a fraction of the gate error
            let crosstalk_rate = gate.error_rate * 0.1;

            two_qubit_errors.push(TwoQubitNoise {
                qubits: (q0, q1),
                depolarizing_rate,
                crosstalk_rate,
            });
        }
    }

    // Build readout noise channels
    for re in &calibration.readout_errors {
        let p0_given_0 = re.p0_given_0;
        let p1_given_1 = re.p1_given_1;
        // Confusion matrix: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
        let confusion_matrix = [
            [p0_given_0, 1.0 - p0_given_0],
            [1.0 - p1_given_1, p1_given_1],
        ];

        readout_channels.push(ReadoutNoiseChannel {
            qubit: re.qubit,
            confusion_matrix,
        });
    }

    Ok(DeviceNoiseModel {
        single_qubit_errors,
        two_qubit_errors,
        readout_errors: readout_channels,
        thermal_relaxation: thermal_channels,
    })
}

/// Validate calibration data for physical consistency.
fn validate_calibration(cal: &DeviceCalibration) -> Result<(), DeviceNoiseError> {
    if cal.num_qubits == 0 {
        return Err(DeviceNoiseError::InvalidCalibration(
            "Device must have at least 1 qubit".to_string(),
        ));
    }

    for (i, qp) in cal.qubit_properties.iter().enumerate() {
        if qp.t1_us < 0.0 {
            return Err(DeviceNoiseError::InvalidCalibration(format!(
                "Qubit {} has negative T1: {}",
                i, qp.t1_us
            )));
        }
        if qp.t2_us < 0.0 {
            return Err(DeviceNoiseError::InvalidCalibration(format!(
                "Qubit {} has negative T2: {}",
                i, qp.t2_us
            )));
        }
        // T2 <= 2*T1 is a fundamental physical constraint
        if qp.t1_us > 0.0 && qp.t2_us > 2.0 * qp.t1_us + 1e-10 {
            return Err(DeviceNoiseError::InvalidCalibration(format!(
                "Qubit {} violates T2 <= 2*T1: T2={}, T1={}",
                i, qp.t2_us, qp.t1_us
            )));
        }
    }

    for gate in &cal.gate_properties {
        for &q in &gate.qubits {
            if q >= cal.num_qubits {
                return Err(DeviceNoiseError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: cal.num_qubits,
                });
            }
        }
        if gate.error_rate < 0.0 || gate.error_rate > 1.0 {
            return Err(DeviceNoiseError::InvalidCalibration(format!(
                "Gate {} on qubits {:?} has invalid error rate: {}",
                gate.gate_name, gate.qubits, gate.error_rate
            )));
        }
    }

    for re in &cal.readout_errors {
        if re.qubit >= cal.num_qubits {
            return Err(DeviceNoiseError::QubitOutOfRange {
                qubit: re.qubit,
                num_qubits: cal.num_qubits,
            });
        }
    }

    Ok(())
}

/// Compute thermal relaxation channel parameters.
///
/// - Amplitude damping: gamma1 = 1 - exp(-gate_time / T1)
/// - Phase damping: gamma2 = 1 - exp(-gate_time / T2), with T2 clamped to 2*T1
pub fn thermal_relaxation_channel(t1: f64, t2: f64, gate_time: f64) -> ThermalRelaxation {
    // Enforce T2 <= 2*T1 physical constraint
    let t2_clamped = if t1 > 0.0 { t2.min(2.0 * t1) } else { t2 };

    let _gamma1 = if t1 > 0.0 {
        1.0 - (-gate_time / t1).exp()
    } else {
        0.0
    };

    let _gamma2 = if t2_clamped > 0.0 {
        1.0 - (-gate_time / t2_clamped).exp()
    } else {
        0.0
    };

    // Excited state population at thermal equilibrium (assume ~50 mK for superconducting)
    // For typical transmon at 5 GHz, n_th ≈ 0.01
    let excited_state_population = 0.01;

    ThermalRelaxation {
        qubit: 0, // Caller sets the actual qubit index
        t1,
        t2: t2_clamped,
        gate_time,
        excited_state_population,
    }
}

// ============================================================
// HARDWARE PRESETS
// ============================================================

impl DeviceCalibration {
    /// IBM Brisbane: 127-qubit Eagle r3 processor with heavy-hex connectivity.
    ///
    /// Approximate calibration: CX error ~0.01, T1 ~300 us, T2 ~200 us, readout ~0.02.
    pub fn ibm_brisbane() -> DeviceCalibration {
        let num_qubits = 127;

        let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
            .map(|i| {
                // Slight variation per qubit to mimic real calibration
                let variation = 1.0 + 0.1 * ((i as f64 * 0.7).sin());
                QubitProperties {
                    t1_us: 300.0 * variation,
                    t2_us: 200.0 * variation,
                    frequency_ghz: 5.0 + 0.2 * ((i as f64 * 0.3).sin()),
                    anharmonicity_ghz: -0.34,
                }
            })
            .collect();

        // Heavy-hex connectivity: each qubit connects to 2-3 neighbors
        let mut gate_properties = Vec::new();
        for i in 0..num_qubits {
            // Single-qubit gates (sx, rz)
            gate_properties.push(GateProperties {
                gate_name: "sx".to_string(),
                qubits: vec![i],
                error_rate: 0.0003,
                gate_time_ns: 35.0,
            });
            gate_properties.push(GateProperties {
                gate_name: "rz".to_string(),
                qubits: vec![i],
                error_rate: 0.0,
                gate_time_ns: 0.0, // Virtual gate
            });
        }
        // Two-qubit CX gates along heavy-hex edges (simplified)
        for i in 0..(num_qubits - 1) {
            if i % 4 != 3 {
                // Skip some to approximate heavy-hex
                let variation = 1.0 + 0.2 * ((i as f64 * 1.1).sin());
                gate_properties.push(GateProperties {
                    gate_name: "cx".to_string(),
                    qubits: vec![i, i + 1],
                    error_rate: 0.01 * variation,
                    gate_time_ns: 300.0,
                });
            }
        }

        let readout_errors: Vec<ReadoutError> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.1 * ((i as f64 * 0.5).sin());
                ReadoutError {
                    qubit: i,
                    p0_given_0: 1.0 - 0.02 * variation,
                    p1_given_1: 1.0 - 0.03 * variation,
                }
            })
            .collect();

        DeviceCalibration {
            name: "ibm_brisbane".to_string(),
            num_qubits,
            qubit_properties,
            gate_properties,
            readout_errors,
            timestamp: 1707900000, // Approximate
        }
    }

    /// IBM Sherbrooke: 127-qubit Eagle r3 processor.
    ///
    /// Similar to Brisbane but with different calibration values.
    pub fn ibm_sherbrooke() -> DeviceCalibration {
        let num_qubits = 127;

        let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.08 * ((i as f64 * 0.9).sin());
                QubitProperties {
                    t1_us: 280.0 * variation,
                    t2_us: 180.0 * variation,
                    frequency_ghz: 4.9 + 0.25 * ((i as f64 * 0.4).sin()),
                    anharmonicity_ghz: -0.33,
                }
            })
            .collect();

        let mut gate_properties = Vec::new();
        for i in 0..num_qubits {
            gate_properties.push(GateProperties {
                gate_name: "sx".to_string(),
                qubits: vec![i],
                error_rate: 0.00025,
                gate_time_ns: 35.0,
            });
            gate_properties.push(GateProperties {
                gate_name: "rz".to_string(),
                qubits: vec![i],
                error_rate: 0.0,
                gate_time_ns: 0.0,
            });
        }
        for i in 0..(num_qubits - 1) {
            if i % 4 != 3 {
                let variation = 1.0 + 0.15 * ((i as f64 * 1.3).sin());
                gate_properties.push(GateProperties {
                    gate_name: "ecr".to_string(),
                    qubits: vec![i, i + 1],
                    error_rate: 0.008 * variation,
                    gate_time_ns: 660.0,
                });
            }
        }

        let readout_errors: Vec<ReadoutError> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.1 * ((i as f64 * 0.6).sin());
                ReadoutError {
                    qubit: i,
                    p0_given_0: 1.0 - 0.015 * variation,
                    p1_given_1: 1.0 - 0.025 * variation,
                }
            })
            .collect();

        DeviceCalibration {
            name: "ibm_sherbrooke".to_string(),
            num_qubits,
            qubit_properties,
            gate_properties,
            readout_errors,
            timestamp: 1707900000,
        }
    }

    /// Google Sycamore: 53-qubit processor with grid connectivity.
    ///
    /// Uses SYC (Sycamore) two-qubit gate. T1 ~15 us, SYC error ~0.006.
    pub fn google_sycamore() -> DeviceCalibration {
        let num_qubits = 53;

        let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.05 * ((i as f64 * 1.2).sin());
                QubitProperties {
                    t1_us: 15.0 * variation,
                    t2_us: 10.0 * variation,
                    frequency_ghz: 6.0 + 0.3 * ((i as f64 * 0.5).sin()),
                    anharmonicity_ghz: -0.22,
                }
            })
            .collect();

        let mut gate_properties = Vec::new();
        for i in 0..num_qubits {
            gate_properties.push(GateProperties {
                gate_name: "phased_xz".to_string(),
                qubits: vec![i],
                error_rate: 0.001,
                gate_time_ns: 25.0,
            });
        }
        // Grid connectivity (simplified: connect adjacent in ~7x8 grid)
        let cols = 8;
        for i in 0..num_qubits {
            // Right neighbor
            if (i % cols) < (cols - 1) && (i + 1) < num_qubits {
                let variation = 1.0 + 0.1 * ((i as f64 * 0.8).sin());
                gate_properties.push(GateProperties {
                    gate_name: "syc".to_string(),
                    qubits: vec![i, i + 1],
                    error_rate: 0.006 * variation,
                    gate_time_ns: 12.0,
                });
            }
            // Down neighbor
            if (i + cols) < num_qubits {
                let variation = 1.0 + 0.1 * ((i as f64 * 1.0).sin());
                gate_properties.push(GateProperties {
                    gate_name: "syc".to_string(),
                    qubits: vec![i, i + cols],
                    error_rate: 0.006 * variation,
                    gate_time_ns: 12.0,
                });
            }
        }

        let readout_errors: Vec<ReadoutError> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.05 * ((i as f64 * 0.7).sin());
                ReadoutError {
                    qubit: i,
                    p0_given_0: 1.0 - 0.035 * variation,
                    p1_given_1: 1.0 - 0.06 * variation,
                }
            })
            .collect();

        DeviceCalibration {
            name: "google_sycamore".to_string(),
            num_qubits,
            qubit_properties,
            gate_properties,
            readout_errors,
            timestamp: 1707900000,
        }
    }

    /// IonQ Aria: 25-qubit trapped-ion processor with all-to-all connectivity.
    ///
    /// Uses MS (Molmer-Sorensen) two-qubit gate. T1 ~10 s, MS error ~0.005.
    pub fn ionq_aria() -> DeviceCalibration {
        let num_qubits = 25;

        let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.02 * ((i as f64 * 0.4).sin());
                QubitProperties {
                    t1_us: 10_000_000.0 * variation, // ~10 seconds in microseconds
                    t2_us: 1_000_000.0 * variation,  // ~1 second
                    frequency_ghz: 12.6,              // Hyperfine transition
                    anharmonicity_ghz: 0.0,           // Not applicable for trapped ions
                }
            })
            .collect();

        let mut gate_properties = Vec::new();
        for i in 0..num_qubits {
            gate_properties.push(GateProperties {
                gate_name: "gpi".to_string(),
                qubits: vec![i],
                error_rate: 0.0003,
                gate_time_ns: 135_000.0, // ~135 us
            });
        }
        // All-to-all connectivity
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                let variation = 1.0 + 0.05 * (((i + j) as f64 * 0.3).sin());
                gate_properties.push(GateProperties {
                    gate_name: "ms".to_string(),
                    qubits: vec![i, j],
                    error_rate: 0.005 * variation,
                    gate_time_ns: 600_000.0, // ~600 us
                });
            }
        }

        let readout_errors: Vec<ReadoutError> = (0..num_qubits)
            .map(|i| ReadoutError {
                qubit: i,
                p0_given_0: 0.997,
                p1_given_1: 0.995,
            })
            .collect();

        DeviceCalibration {
            name: "ionq_aria".to_string(),
            num_qubits,
            qubit_properties,
            gate_properties,
            readout_errors,
            timestamp: 1707900000,
        }
    }

    /// Rigetti Aspen-M3: 80-qubit superconducting processor.
    ///
    /// Uses CZ two-qubit gate. CZ error ~0.05, T1 ~20 us.
    pub fn rigetti_aspen_m3() -> DeviceCalibration {
        let num_qubits = 80;

        let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.1 * ((i as f64 * 0.6).sin());
                QubitProperties {
                    t1_us: 20.0 * variation,
                    t2_us: 12.0 * variation,
                    frequency_ghz: 5.5 + 0.3 * ((i as f64 * 0.4).sin()),
                    anharmonicity_ghz: -0.30,
                }
            })
            .collect();

        let mut gate_properties = Vec::new();
        for i in 0..num_qubits {
            gate_properties.push(GateProperties {
                gate_name: "rx".to_string(),
                qubits: vec![i],
                error_rate: 0.002,
                gate_time_ns: 40.0,
            });
        }
        // Octagonal ring connectivity (simplified)
        for i in 0..(num_qubits - 1) {
            if i % 8 != 7 {
                let variation = 1.0 + 0.2 * ((i as f64 * 0.9).sin());
                gate_properties.push(GateProperties {
                    gate_name: "cz".to_string(),
                    qubits: vec![i, i + 1],
                    error_rate: 0.05 * variation,
                    gate_time_ns: 200.0,
                });
            }
        }

        let readout_errors: Vec<ReadoutError> = (0..num_qubits)
            .map(|i| {
                let variation = 1.0 + 0.1 * ((i as f64 * 0.5).sin());
                ReadoutError {
                    qubit: i,
                    p0_given_0: 1.0 - 0.04 * variation,
                    p1_given_1: 1.0 - 0.05 * variation,
                }
            })
            .collect();

        DeviceCalibration {
            name: "rigetti_aspen_m3".to_string(),
            num_qubits,
            qubit_properties,
            gate_properties,
            readout_errors,
            timestamp: 1707900000,
        }
    }

    /// Ideal device: zero-error noise model for testing and comparison.
    pub fn ideal() -> DeviceCalibration {
        let num_qubits = 10;

        let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
            .map(|_| QubitProperties {
                t1_us: f64::INFINITY,
                t2_us: f64::INFINITY,
                frequency_ghz: 5.0,
                anharmonicity_ghz: -0.34,
            })
            .collect();

        let mut gate_properties = Vec::new();
        for i in 0..num_qubits {
            gate_properties.push(GateProperties {
                gate_name: "u3".to_string(),
                qubits: vec![i],
                error_rate: 0.0,
                gate_time_ns: 0.0,
            });
        }
        for i in 0..(num_qubits - 1) {
            gate_properties.push(GateProperties {
                gate_name: "cx".to_string(),
                qubits: vec![i, i + 1],
                error_rate: 0.0,
                gate_time_ns: 0.0,
            });
        }

        let readout_errors: Vec<ReadoutError> = (0..num_qubits)
            .map(|i| ReadoutError {
                qubit: i,
                p0_given_0: 1.0,
                p1_given_1: 1.0,
            })
            .collect();

        DeviceCalibration {
            name: "ideal".to_string(),
            num_qubits,
            qubit_properties,
            gate_properties,
            readout_errors,
            timestamp: 0,
        }
    }
}

// ============================================================
// NOISE APPLICATION
// ============================================================

/// Apply single-qubit depolarizing noise to a quantum state.
///
/// With probability `p = noise.depolarizing_rate`, applies a random Pauli
/// operator (X, Y, or Z) to the target qubit.
pub fn apply_single_qubit_noise(
    state: &mut Vec<Complex64>,
    noise: &SingleQubitNoise,
    rng: &mut impl Rng,
) {
    let p = noise.depolarizing_rate;
    if p <= 0.0 {
        return;
    }

    let n_qubits = (state.len() as f64).log2() as usize;
    let qubit = noise.qubit;
    if qubit >= n_qubits {
        return;
    }

    let r: f64 = rng.gen();
    if r >= p {
        return; // No error
    }

    // Choose random Pauli: X, Y, or Z with equal probability
    let pauli_choice: f64 = rng.gen();
    let dim = state.len();
    let mask = 1usize << qubit;

    if pauli_choice < 1.0 / 3.0 {
        // Apply X (bit flip)
        for i in 0..dim {
            if i & mask == 0 {
                let j = i | mask;
                state.swap(i, j);
            }
        }
    } else if pauli_choice < 2.0 / 3.0 {
        // Apply Y (bit+phase flip)
        for i in 0..dim {
            if i & mask == 0 {
                let j = i | mask;
                let a = state[i];
                let b = state[j];
                state[i] = Complex64::new(b.im, -b.re); // -i * b
                state[j] = Complex64::new(-a.im, a.re); // i * a
            }
        }
    } else {
        // Apply Z (phase flip)
        for i in 0..dim {
            if i & mask != 0 {
                state[i] = -state[i];
            }
        }
    }

    // Also apply dephasing (extra Z rotation with small probability)
    let dp = noise.dephasing_rate;
    if dp > 0.0 {
        let r2: f64 = rng.gen();
        if r2 < dp {
            for i in 0..dim {
                if i & mask != 0 {
                    state[i] = -state[i];
                }
            }
        }
    }
}

/// Apply two-qubit depolarizing noise to a quantum state.
///
/// With probability `p = noise.depolarizing_rate`, applies a random two-qubit
/// Pauli operator to the target qubit pair.
pub fn apply_two_qubit_noise(
    state: &mut Vec<Complex64>,
    noise: &TwoQubitNoise,
    rng: &mut impl Rng,
) {
    let p = noise.depolarizing_rate;
    if p <= 0.0 {
        return;
    }

    let n_qubits = (state.len() as f64).log2() as usize;
    let (q0, q1) = noise.qubits;
    if q0 >= n_qubits || q1 >= n_qubits {
        return;
    }

    let r: f64 = rng.gen();
    if r >= p {
        return;
    }

    // For simplicity, apply a random single-qubit Pauli to each qubit independently
    let mask0 = 1usize << q0;
    let mask1 = 1usize << q1;
    let dim = state.len();

    // Choose which Pauli on each qubit (0=I, 1=X, 2=Y, 3=Z)
    let p0: u8 = rng.gen_range(0..4);
    let p1: u8 = rng.gen_range(0..4);

    // Skip identity on both (II)
    if p0 == 0 && p1 == 0 {
        return;
    }

    // Apply Pauli to qubit 0
    apply_pauli_to_qubit(state, dim, mask0, p0);
    // Apply Pauli to qubit 1
    apply_pauli_to_qubit(state, dim, mask1, p1);
}

/// Helper: apply a specific Pauli gate (0=I, 1=X, 2=Y, 3=Z) to a qubit.
fn apply_pauli_to_qubit(state: &mut Vec<Complex64>, dim: usize, mask: usize, pauli: u8) {
    match pauli {
        0 => {} // Identity
        1 => {
            // X (bit flip)
            for i in 0..dim {
                if i & mask == 0 {
                    let j = i | mask;
                    state.swap(i, j);
                }
            }
        }
        2 => {
            // Y
            for i in 0..dim {
                if i & mask == 0 {
                    let j = i | mask;
                    let a = state[i];
                    let b = state[j];
                    state[i] = Complex64::new(b.im, -b.re);
                    state[j] = Complex64::new(-a.im, a.re);
                }
            }
        }
        3 => {
            // Z (phase flip)
            for i in 0..dim {
                if i & mask != 0 {
                    state[i] = -state[i];
                }
            }
        }
        _ => {}
    }
}

/// Apply readout noise: probabilistically flip a measurement result.
///
/// Uses the confusion matrix to determine whether to flip the outcome.
pub fn apply_readout_noise(
    measurement: u8,
    noise: &ReadoutNoiseChannel,
    rng: &mut impl Rng,
) -> u8 {
    let r: f64 = rng.gen();
    match measurement {
        0 => {
            // P(flip to 1 | true state is 0) = confusion_matrix[0][1]
            if r < noise.confusion_matrix[0][1] {
                1
            } else {
                0
            }
        }
        1 => {
            // P(flip to 0 | true state is 1) = confusion_matrix[1][0]
            if r < noise.confusion_matrix[1][0] {
                0
            } else {
                1
            }
        }
        _ => measurement,
    }
}

/// Apply thermal relaxation to a quantum state.
///
/// Models amplitude damping (T1) and phase damping (T2) processes.
pub fn apply_thermal_relaxation(
    state: &mut Vec<Complex64>,
    tr: &ThermalRelaxation,
    qubit: usize,
    rng: &mut impl Rng,
) {
    let n_qubits = (state.len() as f64).log2() as usize;
    if qubit >= n_qubits {
        return;
    }

    // Compute damping probabilities
    let gamma1 = if tr.t1 > 0.0 && tr.t1.is_finite() {
        1.0 - (-tr.gate_time / tr.t1).exp()
    } else {
        0.0
    };

    let t2_eff = if tr.t1 > 0.0 && tr.t1.is_finite() {
        tr.t2.min(2.0 * tr.t1)
    } else {
        tr.t2
    };

    let gamma2 = if t2_eff > 0.0 && t2_eff.is_finite() {
        1.0 - (-tr.gate_time / t2_eff).exp()
    } else {
        0.0
    };

    if gamma1 <= 0.0 && gamma2 <= 0.0 {
        return;
    }

    let mask = 1usize << qubit;
    let dim = state.len();

    // Amplitude damping: |1> -> |0> with probability gamma1
    let r1: f64 = rng.gen();
    if r1 < gamma1 {
        let damping = (1.0 - gamma1).sqrt();
        for i in 0..dim {
            if i & mask != 0 {
                // State with qubit in |1>: dampen amplitude
                let j = i & !mask; // Corresponding |0> state
                let excited_amp = state[i];
                let transfer = excited_amp * (gamma1).sqrt();
                state[j] = state[j] + transfer;
                state[i] = state[i] * damping;
            }
        }
        // Re-normalize
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for amp in state.iter_mut() {
                *amp = *amp / norm;
            }
        }
    }

    // Phase damping: dephase |1> component with probability gamma2
    let r2: f64 = rng.gen();
    if r2 < gamma2 {
        // Apply random phase to |1> component
        let phase_angle: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
        let phase = Complex64::new(phase_angle.cos(), phase_angle.sin());
        for i in 0..dim {
            if i & mask != 0 {
                state[i] = state[i] * phase;
            }
        }
    }
}

// ============================================================
// CALIBRATION IMPORT / EXPORT (manual JSON, no serde)
// ============================================================

/// Serialize a DeviceCalibration to a JSON string (manual formatting).
pub fn calibration_to_json(cal: &DeviceCalibration) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("{\n");
    s.push_str(&format!("  \"name\": \"{}\",\n", cal.name));
    s.push_str(&format!("  \"num_qubits\": {},\n", cal.num_qubits));
    s.push_str(&format!("  \"timestamp\": {},\n", cal.timestamp));

    // Qubit properties
    s.push_str("  \"qubit_properties\": [\n");
    for (i, qp) in cal.qubit_properties.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"t1_us\": {},\n", format_f64(qp.t1_us)));
        s.push_str(&format!("      \"t2_us\": {},\n", format_f64(qp.t2_us)));
        s.push_str(&format!(
            "      \"frequency_ghz\": {},\n",
            format_f64(qp.frequency_ghz)
        ));
        s.push_str(&format!(
            "      \"anharmonicity_ghz\": {}\n",
            format_f64(qp.anharmonicity_ghz)
        ));
        s.push('}');
        if i + 1 < cal.qubit_properties.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ],\n");

    // Gate properties
    s.push_str("  \"gate_properties\": [\n");
    for (i, gp) in cal.gate_properties.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"gate_name\": \"{}\",\n", gp.gate_name));
        s.push_str(&format!(
            "      \"qubits\": [{}],\n",
            gp.qubits
                .iter()
                .map(|q| q.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        s.push_str(&format!(
            "      \"error_rate\": {},\n",
            format_f64(gp.error_rate)
        ));
        s.push_str(&format!(
            "      \"gate_time_ns\": {}\n",
            format_f64(gp.gate_time_ns)
        ));
        s.push('}');
        if i + 1 < cal.gate_properties.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ],\n");

    // Readout errors
    s.push_str("  \"readout_errors\": [\n");
    for (i, re) in cal.readout_errors.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"qubit\": {},\n", re.qubit));
        s.push_str(&format!(
            "      \"p0_given_0\": {},\n",
            format_f64(re.p0_given_0)
        ));
        s.push_str(&format!(
            "      \"p1_given_1\": {}\n",
            format_f64(re.p1_given_1)
        ));
        s.push('}');
        if i + 1 < cal.readout_errors.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");

    s.push('}');
    s
}

/// Format f64 for JSON, handling infinity and NaN.
fn format_f64(v: f64) -> String {
    if v.is_infinite() {
        if v > 0.0 {
            "1e308".to_string()
        } else {
            "-1e308".to_string()
        }
    } else if v.is_nan() {
        "null".to_string()
    } else {
        format!("{}", v)
    }
}

/// Parse a DeviceCalibration from a JSON string (manual parsing).
///
/// This is a simplified parser that handles the format produced by
/// `calibration_to_json`. It does not handle arbitrary JSON.
pub fn calibration_from_json(json: &str) -> Result<DeviceCalibration, DeviceNoiseError> {
    let json = json.trim();

    let name = extract_string_field(json, "name")
        .ok_or_else(|| DeviceNoiseError::InvalidCalibration("Missing 'name' field".to_string()))?;

    let num_qubits = extract_int_field(json, "num_qubits").ok_or_else(|| {
        DeviceNoiseError::InvalidCalibration("Missing 'num_qubits' field".to_string())
    })? as usize;

    let timestamp = extract_int_field(json, "timestamp").unwrap_or(0) as u64;

    // Parse qubit_properties array
    let qubit_properties = parse_qubit_properties_array(json)?;

    // Parse gate_properties array
    let gate_properties = parse_gate_properties_array(json)?;

    // Parse readout_errors array
    let readout_errors = parse_readout_errors_array(json)?;

    Ok(DeviceCalibration {
        name,
        num_qubits,
        qubit_properties,
        gate_properties,
        readout_errors,
        timestamp,
    })
}

// ============================================================
// VENDOR CALIBRATION IMPORTERS (E1)
// ============================================================

/// Import IBM BackendProperties-style JSON into `DeviceCalibration`.
///
/// Accepts both canonical IBM layouts and reduced snapshots as long as
/// `qubits` and `gates` are present in compatible form.
pub fn import_ibm_backend_properties_json(
    json: &str,
) -> Result<DeviceCalibration, DeviceNoiseError> {
    let root: Value = serde_json::from_str(json).map_err(|e| {
        DeviceNoiseError::InvalidCalibration(format!("IBM JSON parse failed: {}", e))
    })?;

    let name = string_from_paths(&root, &[&["backend_name"], &["name"], &["backend"]])
        .unwrap_or_else(|| "ibm_device".to_string());

    let qubits_arr = root
        .get("qubits")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            DeviceNoiseError::InvalidCalibration("IBM JSON missing 'qubits' array".to_string())
        })?;

    let mut qubit_properties = Vec::with_capacity(qubits_arr.len());
    let mut readout_errors = Vec::with_capacity(qubits_arr.len());

    for (qi, q_entry) in qubits_arr.iter().enumerate() {
        let q_params = q_entry.as_array().ok_or_else(|| {
            DeviceNoiseError::InvalidCalibration(
                "IBM 'qubits' entries must be arrays of named parameters".to_string(),
            )
        })?;

        let t1_us = named_param_value(q_params, &["T1"]).unwrap_or(100.0);
        let t2_us = named_param_value(q_params, &["T2"]).unwrap_or((2.0 * t1_us).min(100.0));
        let frequency_ghz = named_param_value(q_params, &["frequency"]).unwrap_or(5.0);
        let anharmonicity_ghz =
            named_param_value(q_params, &["anharmonicity"]).unwrap_or(-0.34);

        let readout_error = named_param_value(q_params, &["readout_error"]).unwrap_or(0.02);
        let p01 = named_param_value(q_params, &["prob_meas1_prep0"]).unwrap_or(readout_error);
        let p10 = named_param_value(q_params, &["prob_meas0_prep1"]).unwrap_or(readout_error);

        qubit_properties.push(QubitProperties {
            t1_us,
            t2_us,
            frequency_ghz,
            anharmonicity_ghz,
        });
        readout_errors.push(ReadoutError {
            qubit: qi,
            p0_given_0: (1.0 - p01).clamp(0.0, 1.0),
            p1_given_1: (1.0 - p10).clamp(0.0, 1.0),
        });
    }

    let gates_arr = root
        .get("gates")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            DeviceNoiseError::InvalidCalibration("IBM JSON missing 'gates' array".to_string())
        })?;

    let mut gate_properties = Vec::new();
    for g in gates_arr {
        let gate_name = string_from_paths(g, &[&["gate"], &["name"]])
            .unwrap_or_else(|| "unknown".to_string());
        let qubits: Vec<usize> = g
            .get("qubits")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(value_to_usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let params = g.get("parameters").and_then(Value::as_array);
        let error_rate = params
            .and_then(|p| named_param_value(p, &["gate_error", "error"]))
            .or_else(|| number_from_paths(g, &[&["gate_error"], &["error_rate"]]))
            .unwrap_or(0.0);
        let gate_time_ns = params
            .and_then(|p| named_param_value_with_units(p, &["gate_length", "gate_time"]))
            .or_else(|| number_from_paths(g, &[&["gate_length"], &["gate_time_ns"]]))
            .unwrap_or(if qubits.len() == 2 { 300.0 } else { 35.0 });

        if !qubits.is_empty() {
            gate_properties.push(GateProperties {
                gate_name,
                qubits,
                error_rate: error_rate.clamp(0.0, 1.0),
                gate_time_ns,
            });
        }
    }

    Ok(DeviceCalibration {
        name,
        num_qubits: qubit_properties.len(),
        qubit_properties,
        gate_properties,
        readout_errors,
        timestamp: root
            .get("timestamp")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Import IonQ characterization JSON into `DeviceCalibration`.
pub fn import_ionq_characterization_json(
    json: &str,
) -> Result<DeviceCalibration, DeviceNoiseError> {
    let root: Value = serde_json::from_str(json).map_err(|e| {
        DeviceNoiseError::InvalidCalibration(format!("IonQ JSON parse failed: {}", e))
    })?;

    let num_qubits = number_from_paths(
        &root,
        &[&["num_qubits"], &["qubits"], &["device", "num_qubits"]],
    )
    .map(|x| x as usize)
    .filter(|&x| x > 0)
    .ok_or_else(|| {
        DeviceNoiseError::InvalidCalibration("IonQ JSON missing qubit count".to_string())
    })?;

    let name = string_from_paths(&root, &[&["name"], &["device", "name"]])
        .unwrap_or_else(|| "ionq_device".to_string());

    let sq_fidelity = number_from_paths(
        &root,
        &[
            &["single_qubit", "fidelity"],
            &["single_qubit_fidelity"],
            &["fidelity", "1q"],
        ],
    )
    .unwrap_or(0.999);
    let tq_fidelity = number_from_paths(
        &root,
        &[
            &["two_qubit", "fidelity"],
            &["two_qubit_fidelity"],
            &["fidelity", "2q"],
        ],
    )
    .unwrap_or(0.995);
    let readout_fidelity = number_from_paths(
        &root,
        &[
            &["readout", "fidelity"],
            &["readout_fidelity"],
            &["fidelity", "readout"],
        ],
    )
    .unwrap_or(0.996);
    let sq_gate_ns = number_from_paths(
        &root,
        &[
            &["single_qubit", "duration_ns"],
            &["single_qubit", "gate_time_ns"],
            &["timing", "single_ns"],
        ],
    )
    .unwrap_or(135_000.0);
    let tq_gate_ns = number_from_paths(
        &root,
        &[
            &["two_qubit", "duration_ns"],
            &["two_qubit", "gate_time_ns"],
            &["timing", "two_ns"],
        ],
    )
    .unwrap_or(600_000.0);

    let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
        .map(|_| QubitProperties {
            t1_us: 10_000_000.0,
            t2_us: 1_000_000.0,
            frequency_ghz: 12.6,
            anharmonicity_ghz: 0.0,
        })
        .collect();

    let mut gate_properties = Vec::new();
    for q in 0..num_qubits {
        gate_properties.push(GateProperties {
            gate_name: "gpi".to_string(),
            qubits: vec![q],
            error_rate: (1.0 - sq_fidelity).clamp(0.0, 1.0),
            gate_time_ns: sq_gate_ns,
        });
    }
    for i in 0..num_qubits {
        for j in (i + 1)..num_qubits {
            gate_properties.push(GateProperties {
                gate_name: "ms".to_string(),
                qubits: vec![i, j],
                error_rate: (1.0 - tq_fidelity).clamp(0.0, 1.0),
                gate_time_ns: tq_gate_ns,
            });
        }
    }

    let readout_errors: Vec<ReadoutError> = (0..num_qubits)
        .map(|q| ReadoutError {
            qubit: q,
            p0_given_0: readout_fidelity.clamp(0.0, 1.0),
            p1_given_1: readout_fidelity.clamp(0.0, 1.0),
        })
        .collect();

    Ok(DeviceCalibration {
        name,
        num_qubits,
        qubit_properties,
        gate_properties,
        readout_errors,
        timestamp: root
            .get("timestamp")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Import Cirq/Google-style device specification JSON into `DeviceCalibration`.
pub fn import_cirq_device_spec_json(
    json: &str,
) -> Result<DeviceCalibration, DeviceNoiseError> {
    let root: Value = serde_json::from_str(json).map_err(|e| {
        DeviceNoiseError::InvalidCalibration(format!("Cirq JSON parse failed: {}", e))
    })?;

    let num_qubits = root
        .get("valid_qubits")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .or_else(|| root.get("qubits").and_then(Value::as_array).map(|a| a.len()))
        .or_else(|| number_from_paths(&root, &[&["num_qubits"]]).map(|x| x as usize))
        .filter(|&x| x > 0)
        .ok_or_else(|| {
            DeviceNoiseError::InvalidCalibration("Cirq JSON missing qubit count".to_string())
        })?;

    let name = string_from_paths(&root, &[&["name"], &["device_name"]])
        .unwrap_or_else(|| "cirq_device".to_string());
    let sq_error = number_from_paths(
        &root,
        &[
            &["single_qubit_gate_error"],
            &["errors", "single_qubit"],
            &["average_single_qubit_error"],
        ],
    )
    .unwrap_or(0.001);
    let tq_error = number_from_paths(
        &root,
        &[
            &["two_qubit_gate_error"],
            &["errors", "two_qubit"],
            &["average_two_qubit_error"],
        ],
    )
    .unwrap_or(0.006);
    let meas_error = number_from_paths(
        &root,
        &[
            &["measurement_error"],
            &["errors", "measurement"],
            &["readout_error"],
        ],
    )
    .unwrap_or(0.04);

    let edges: Vec<(usize, usize)> = root
        .get("connectivity")
        .and_then(Value::as_array)
        .map(|v| parse_edge_list(v))
        .unwrap_or_else(|| (0..num_qubits.saturating_sub(1)).map(|i| (i, i + 1)).collect());

    let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
        .map(|_| QubitProperties {
            t1_us: 15.0,
            t2_us: 10.0,
            frequency_ghz: 6.0,
            anharmonicity_ghz: -0.22,
        })
        .collect();

    let mut gate_properties = Vec::new();
    for q in 0..num_qubits {
        gate_properties.push(GateProperties {
            gate_name: "phased_xz".to_string(),
            qubits: vec![q],
            error_rate: sq_error.clamp(0.0, 1.0),
            gate_time_ns: 25.0,
        });
    }
    for (a, b) in edges {
        if a < num_qubits && b < num_qubits && a != b {
            gate_properties.push(GateProperties {
                gate_name: "syc".to_string(),
                qubits: vec![a, b],
                error_rate: tq_error.clamp(0.0, 1.0),
                gate_time_ns: 12.0,
            });
        }
    }

    let readout_errors: Vec<ReadoutError> = (0..num_qubits)
        .map(|q| ReadoutError {
            qubit: q,
            p0_given_0: (1.0 - meas_error).clamp(0.0, 1.0),
            p1_given_1: (1.0 - meas_error).clamp(0.0, 1.0),
        })
        .collect();

    Ok(DeviceCalibration {
        name,
        num_qubits,
        qubit_properties,
        gate_properties,
        readout_errors,
        timestamp: root
            .get("timestamp")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Import Rigetti ISA/noise summary JSON into `DeviceCalibration`.
pub fn import_rigetti_isa_noise_json(json: &str) -> Result<DeviceCalibration, DeviceNoiseError> {
    let root: Value = serde_json::from_str(json).map_err(|e| {
        DeviceNoiseError::InvalidCalibration(format!("Rigetti JSON parse failed: {}", e))
    })?;

    let num_qubits = number_from_paths(&root, &[&["num_qubits"], &["isa", "num_qubits"]])
        .map(|x| x as usize)
        .or_else(|| {
            root.get("qubits")
                .and_then(Value::as_array)
                .map(|a| a.len())
        })
        .filter(|&x| x > 0)
        .ok_or_else(|| {
            DeviceNoiseError::InvalidCalibration("Rigetti JSON missing qubit count".to_string())
        })?;

    let name = string_from_paths(&root, &[&["name"], &["device", "name"]])
        .unwrap_or_else(|| "rigetti_device".to_string());
    let sq_fidelity = number_from_paths(
        &root,
        &[
            &["one_qubit", "fidelity"],
            &["1Q", "fidelity"],
            &["fidelity", "1q"],
        ],
    )
    .unwrap_or(0.998);
    let tq_fidelity = number_from_paths(
        &root,
        &[
            &["two_qubit", "fidelity"],
            &["2Q", "fidelity"],
            &["fidelity", "2q"],
        ],
    )
    .unwrap_or(0.95);
    let readout_fidelity = number_from_paths(
        &root,
        &[
            &["readout", "fidelity"],
            &["fidelity", "readout"],
            &["measurement", "fidelity"],
        ],
    )
    .unwrap_or(0.95);

    let edges: Vec<(usize, usize)> = root
        .get("edges")
        .and_then(Value::as_array)
        .or_else(|| root.get("isa").and_then(|x| x.get("edges")).and_then(Value::as_array))
        .map(|v| parse_edge_list(v))
        .unwrap_or_else(|| (0..num_qubits.saturating_sub(1)).map(|i| (i, i + 1)).collect());

    let qubit_properties: Vec<QubitProperties> = (0..num_qubits)
        .map(|_| QubitProperties {
            t1_us: 20.0,
            t2_us: 12.0,
            frequency_ghz: 5.5,
            anharmonicity_ghz: -0.30,
        })
        .collect();

    let mut gate_properties = Vec::new();
    for q in 0..num_qubits {
        gate_properties.push(GateProperties {
            gate_name: "rx".to_string(),
            qubits: vec![q],
            error_rate: (1.0 - sq_fidelity).clamp(0.0, 1.0),
            gate_time_ns: 40.0,
        });
    }
    for (a, b) in edges {
        if a < num_qubits && b < num_qubits && a != b {
            gate_properties.push(GateProperties {
                gate_name: "cz".to_string(),
                qubits: vec![a, b],
                error_rate: (1.0 - tq_fidelity).clamp(0.0, 1.0),
                gate_time_ns: 200.0,
            });
        }
    }

    let readout_errors: Vec<ReadoutError> = (0..num_qubits)
        .map(|q| ReadoutError {
            qubit: q,
            p0_given_0: readout_fidelity.clamp(0.0, 1.0),
            p1_given_1: readout_fidelity.clamp(0.0, 1.0),
        })
        .collect();

    Ok(DeviceCalibration {
        name,
        num_qubits,
        qubit_properties,
        gate_properties,
        readout_errors,
        timestamp: root
            .get("timestamp")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Auto-detect vendor format and import a calibration.
pub fn import_vendor_calibration_json(json: &str) -> Result<DeviceCalibration, DeviceNoiseError> {
    let root: Value = serde_json::from_str(json).map_err(|e| {
        DeviceNoiseError::InvalidCalibration(format!("Calibration JSON parse failed: {}", e))
    })?;

    let name_lower = string_from_paths(&root, &[&["name"], &["backend_name"], &["provider"]])
        .unwrap_or_default()
        .to_lowercase();

    if root.get("backend_name").is_some() || root.get("gates").is_some() && root.get("qubits").is_some() {
        return import_ibm_backend_properties_json(json);
    }
    if name_lower.contains("ionq") || root.get("single_qubit").is_some() || root.get("two_qubit").is_some() {
        return import_ionq_characterization_json(json);
    }
    if root.get("isa").is_some() || root.get("edges").is_some() {
        return import_rigetti_isa_noise_json(json);
    }
    if root.get("valid_qubits").is_some() || name_lower.contains("sycamore") {
        return import_cirq_device_spec_json(json);
    }

    Err(DeviceNoiseError::InvalidCalibration(
        "Unknown vendor format: expected IBM/IonQ/Cirq/Rigetti-like JSON".to_string(),
    ))
}

fn value_to_usize(v: &Value) -> Option<usize> {
    v.as_u64().map(|x| x as usize).or_else(|| {
        v.as_str()
            .and_then(|s| s.parse::<usize>().ok())
    })
}

fn value_to_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|x| x as f64)).or_else(|| {
        v.as_str().and_then(|s| s.parse::<f64>().ok())
    })
}

fn value_at_path<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cur = v;
    for key in path {
        cur = cur.get(*key)?;
    }
    Some(cur)
}

fn number_from_paths(root: &Value, paths: &[&[&str]]) -> Option<f64> {
    for path in paths {
        if let Some(v) = value_at_path(root, path).and_then(value_to_f64) {
            return Some(v);
        }
    }
    None
}

fn string_from_paths(root: &Value, paths: &[&[&str]]) -> Option<String> {
    for path in paths {
        if let Some(s) = value_at_path(root, path).and_then(Value::as_str) {
            return Some(s.to_string());
        }
    }
    None
}

fn named_param_value(params: &[Value], accepted_names: &[&str]) -> Option<f64> {
    for p in params {
        let name = p.get("name").and_then(Value::as_str)?;
        if accepted_names.iter().any(|n| name.eq_ignore_ascii_case(n)) {
            return p.get("value").and_then(value_to_f64);
        }
    }
    None
}

fn named_param_value_with_units(params: &[Value], accepted_names: &[&str]) -> Option<f64> {
    for p in params {
        let name = p.get("name").and_then(Value::as_str)?;
        if accepted_names.iter().any(|n| name.eq_ignore_ascii_case(n)) {
            let value = p.get("value").and_then(value_to_f64)?;
            let unit = p.get("unit").and_then(Value::as_str).unwrap_or("ns");
            let ns = match unit {
                "s" => value * 1e9,
                "ms" => value * 1e6,
                "us" | "µs" => value * 1e3,
                _ => value,
            };
            return Some(ns);
        }
    }
    None
}

fn parse_edge_list(edges: &[Value]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for e in edges {
        if let Some(pair) = e.as_array() {
            if pair.len() >= 2 {
                if let (Some(a), Some(b)) = (value_to_usize(&pair[0]), value_to_usize(&pair[1])) {
                    out.push((a, b));
                }
            }
        } else if let (Some(a), Some(b)) = (
            e.get("a").and_then(value_to_usize),
            e.get("b").and_then(value_to_usize),
        ) {
            out.push((a, b));
        }
    }
    out
}

/// Extract a string field value from JSON text.
fn extract_string_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{}\":", field);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    if rest.starts_with('"') {
        let inner = &rest[1..];
        let end = inner.find('"')?;
        Some(inner[..end].to_string())
    } else {
        None
    }
}

/// Extract a numeric (integer) field value from JSON text.
fn extract_int_field(json: &str, field: &str) -> Option<i64> {
    let pattern = format!("\"{}\":", field);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extract a float field value from JSON text.
fn extract_float_field(json: &str, field: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", field);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.' && c != 'e' && c != 'E' && c != '+')
        .unwrap_or(rest.len());
    let val_str = &rest[..end];
    if val_str == "null" {
        Some(f64::NAN)
    } else {
        val_str.parse().ok()
    }
}

/// Parse the qubit_properties JSON array.
fn parse_qubit_properties_array(json: &str) -> Result<Vec<QubitProperties>, DeviceNoiseError> {
    let mut result = Vec::new();
    let array_start = json
        .find("\"qubit_properties\":")
        .ok_or_else(|| {
            DeviceNoiseError::InvalidCalibration("Missing qubit_properties".to_string())
        })?;
    let array_content = &json[array_start..];
    let bracket_start = array_content.find('[').ok_or_else(|| {
        DeviceNoiseError::InvalidCalibration("Malformed qubit_properties array".to_string())
    })?;

    // Find matching bracket
    let inner = &array_content[(bracket_start + 1)..];

    // Split by objects
    let mut depth = 0;
    let mut obj_start = None;
    for (i, c) in inner.chars().enumerate() {
        match c {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start {
                        let obj = &inner[start..=i];
                        let t1 = extract_float_field(obj, "t1_us").unwrap_or(0.0);
                        let t2 = extract_float_field(obj, "t2_us").unwrap_or(0.0);
                        let freq = extract_float_field(obj, "frequency_ghz").unwrap_or(5.0);
                        let anh = extract_float_field(obj, "anharmonicity_ghz").unwrap_or(-0.34);
                        result.push(QubitProperties {
                            t1_us: t1,
                            t2_us: t2,
                            frequency_ghz: freq,
                            anharmonicity_ghz: anh,
                        });
                    }
                    obj_start = None;
                }
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }

    Ok(result)
}

/// Parse the gate_properties JSON array.
fn parse_gate_properties_array(json: &str) -> Result<Vec<GateProperties>, DeviceNoiseError> {
    let mut result = Vec::new();
    let array_start = json.find("\"gate_properties\":").ok_or_else(|| {
        DeviceNoiseError::InvalidCalibration("Missing gate_properties".to_string())
    })?;
    let array_content = &json[array_start..];
    let bracket_start = array_content.find('[').ok_or_else(|| {
        DeviceNoiseError::InvalidCalibration("Malformed gate_properties array".to_string())
    })?;

    let inner = &array_content[(bracket_start + 1)..];
    let mut depth = 0;
    let mut obj_start = None;

    for (i, c) in inner.chars().enumerate() {
        match c {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start {
                        let obj = &inner[start..=i];
                        let gate_name =
                            extract_string_field(obj, "gate_name").unwrap_or_default();
                        let error_rate = extract_float_field(obj, "error_rate").unwrap_or(0.0);
                        let gate_time_ns =
                            extract_float_field(obj, "gate_time_ns").unwrap_or(0.0);

                        // Parse qubits array
                        let qubits = parse_int_array(obj, "qubits");

                        result.push(GateProperties {
                            gate_name,
                            qubits,
                            error_rate,
                            gate_time_ns,
                        });
                    }
                    obj_start = None;
                }
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }

    Ok(result)
}

/// Parse an integer array field from JSON text.
fn parse_int_array(json: &str, field: &str) -> Vec<usize> {
    let pattern = format!("\"{}\":", field);
    let start = match json.find(&pattern) {
        Some(s) => s + pattern.len(),
        None => return Vec::new(),
    };
    let rest = json[start..].trim_start();
    let bracket_start = match rest.find('[') {
        Some(s) => s,
        None => return Vec::new(),
    };
    let after_bracket = &rest[(bracket_start + 1)..];
    let bracket_end = match after_bracket.find(']') {
        Some(e) => e,
        None => return Vec::new(),
    };
    let content = &after_bracket[..bracket_end];
    content
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

/// Parse the readout_errors JSON array.
fn parse_readout_errors_array(json: &str) -> Result<Vec<ReadoutError>, DeviceNoiseError> {
    let mut result = Vec::new();
    let array_start = json.find("\"readout_errors\":").ok_or_else(|| {
        DeviceNoiseError::InvalidCalibration("Missing readout_errors".to_string())
    })?;
    let array_content = &json[array_start..];
    let bracket_start = array_content.find('[').ok_or_else(|| {
        DeviceNoiseError::InvalidCalibration("Malformed readout_errors array".to_string())
    })?;

    let inner = &array_content[(bracket_start + 1)..];
    let mut depth = 0;
    let mut obj_start = None;

    for (i, c) in inner.chars().enumerate() {
        match c {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start {
                        let obj = &inner[start..=i];
                        let qubit = extract_int_field(obj, "qubit").unwrap_or(0) as usize;
                        let p0 = extract_float_field(obj, "p0_given_0").unwrap_or(1.0);
                        let p1 = extract_float_field(obj, "p1_given_1").unwrap_or(1.0);
                        result.push(ReadoutError {
                            qubit,
                            p0_given_0: p0,
                            p1_given_1: p1,
                        });
                    }
                    obj_start = None;
                }
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }

    Ok(result)
}

// ============================================================
// COMPARISON & STATISTICS
// ============================================================

/// Compute the average single-qubit gate error rate across all single-qubit gates.
pub fn average_gate_error(cal: &DeviceCalibration) -> f64 {
    let sq_gates: Vec<&GateProperties> = cal
        .gate_properties
        .iter()
        .filter(|g| g.qubits.len() == 1 && g.error_rate > 0.0)
        .collect();

    if sq_gates.is_empty() {
        return 0.0;
    }

    let sum: f64 = sq_gates.iter().map(|g| g.error_rate).sum();
    sum / sq_gates.len() as f64
}

/// Compute the average T1 time across all qubits (in microseconds).
pub fn average_t1(cal: &DeviceCalibration) -> f64 {
    if cal.qubit_properties.is_empty() {
        return 0.0;
    }
    let sum: f64 = cal.qubit_properties.iter().map(|q| q.t1_us).sum();
    sum / cal.qubit_properties.len() as f64
}

/// Compute the average readout error rate: 1 - (p0_given_0 + p1_given_1) / 2.
pub fn average_readout_error(cal: &DeviceCalibration) -> f64 {
    if cal.readout_errors.is_empty() {
        return 0.0;
    }
    let sum: f64 = cal
        .readout_errors
        .iter()
        .map(|r| 1.0 - (r.p0_given_0 + r.p1_given_1) / 2.0)
        .sum();
    sum / cal.readout_errors.len() as f64
}

/// Compute the average two-qubit gate error rate.
pub fn average_two_qubit_gate_error(cal: &DeviceCalibration) -> f64 {
    let tq_gates: Vec<&GateProperties> = cal
        .gate_properties
        .iter()
        .filter(|g| g.qubits.len() == 2)
        .collect();

    if tq_gates.is_empty() {
        return 0.0;
    }

    let sum: f64 = tq_gates.iter().map(|g| g.error_rate).sum();
    sum / tq_gates.len() as f64
}

/// Generate a formatted comparison table for multiple devices.
pub fn compare_devices(devices: &[DeviceCalibration]) -> String {
    if devices.is_empty() {
        return String::new();
    }

    let mut table = String::with_capacity(2048);
    table.push_str(&format!(
        "{:<25} {:>8} {:>12} {:>12} {:>12} {:>14}\n",
        "Device", "Qubits", "Avg T1 (us)", "1Q Err", "2Q Err", "Readout Err"
    ));
    table.push_str(&"-".repeat(85));
    table.push('\n');

    for dev in devices {
        let avg_t1 = average_t1(dev);
        let avg_1q = average_gate_error(dev);
        let avg_2q = average_two_qubit_gate_error(dev);
        let avg_ro = average_readout_error(dev);

        let t1_str = if avg_t1 > 1e6 {
            format!("{:.0}s", avg_t1 / 1e6)
        } else if avg_t1 > 1000.0 {
            format!("{:.0}ms", avg_t1 / 1000.0)
        } else {
            format!("{:.1}us", avg_t1)
        };

        table.push_str(&format!(
            "{:<25} {:>8} {:>12} {:>12.6} {:>12.6} {:>14.6}\n",
            dev.name, dev.num_qubits, t1_str, avg_1q, avg_2q, avg_ro
        ));
    }

    table
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    /// Create a |0> state for n qubits.
    fn zero_state(n_qubits: usize) -> Vec<Complex64> {
        let dim = 1 << n_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        state
    }

    /// Compute state norm squared.
    fn norm_sq(state: &[Complex64]) -> f64 {
        state.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Compute fidelity between two states.
    fn fidelity(a: &[Complex64], b: &[Complex64]) -> f64 {
        let overlap: Complex64 = a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum();
        overlap.norm_sqr()
    }

    #[test]
    fn test_ibm_brisbane_127_qubits() {
        let cal = DeviceCalibration::ibm_brisbane();
        assert_eq!(cal.num_qubits, 127);
        assert_eq!(cal.qubit_properties.len(), 127);
        assert_eq!(cal.readout_errors.len(), 127);
        assert_eq!(cal.name, "ibm_brisbane");
    }

    #[test]
    fn test_google_sycamore_53_qubits() {
        let cal = DeviceCalibration::google_sycamore();
        assert_eq!(cal.num_qubits, 53);
        assert_eq!(cal.qubit_properties.len(), 53);
        assert_eq!(cal.name, "google_sycamore");
    }

    #[test]
    fn test_ionq_aria_all_to_all() {
        let cal = DeviceCalibration::ionq_aria();
        assert_eq!(cal.num_qubits, 25);

        // All-to-all: should have C(25,2) = 300 two-qubit gates
        let two_qubit_gates: Vec<_> = cal
            .gate_properties
            .iter()
            .filter(|g| g.qubits.len() == 2)
            .collect();
        assert_eq!(two_qubit_gates.len(), 300, "IonQ should have all-to-all MS gates");

        // Very long T1 (trapped ions)
        assert!(cal.qubit_properties[0].t1_us > 1_000_000.0);
    }

    #[test]
    fn test_ideal_device_zero_errors() {
        let cal = DeviceCalibration::ideal();
        let model = build_noise_model(&cal).expect("Should build ideal noise model");

        // All single-qubit depolarizing rates should be zero
        for sq in &model.single_qubit_errors {
            assert_eq!(sq.depolarizing_rate, 0.0, "Ideal device has zero depolarizing");
        }

        // All two-qubit depolarizing rates should be zero
        for tq in &model.two_qubit_errors {
            assert_eq!(tq.depolarizing_rate, 0.0, "Ideal device has zero 2Q depolarizing");
        }

        // Readout confusion matrices should be identity
        for ro in &model.readout_errors {
            assert!(
                (ro.confusion_matrix[0][0] - 1.0).abs() < 1e-12,
                "Ideal P(0|0) = 1"
            );
            assert!(
                (ro.confusion_matrix[1][1] - 1.0).abs() < 1e-12,
                "Ideal P(1|1) = 1"
            );
            assert!(
                ro.confusion_matrix[0][1].abs() < 1e-12,
                "Ideal P(1|0) = 0"
            );
            assert!(
                ro.confusion_matrix[1][0].abs() < 1e-12,
                "Ideal P(0|1) = 0"
            );
        }
    }

    #[test]
    fn test_t2_leq_2t1_constraint() {
        // Valid: T2 = 2*T1
        let cal_valid = DeviceCalibration {
            name: "test".to_string(),
            num_qubits: 1,
            qubit_properties: vec![QubitProperties {
                t1_us: 100.0,
                t2_us: 200.0,
                frequency_ghz: 5.0,
                anharmonicity_ghz: -0.34,
            }],
            gate_properties: vec![GateProperties {
                gate_name: "sx".to_string(),
                qubits: vec![0],
                error_rate: 0.001,
                gate_time_ns: 35.0,
            }],
            readout_errors: vec![ReadoutError {
                qubit: 0,
                p0_given_0: 0.98,
                p1_given_1: 0.97,
            }],
            timestamp: 0,
        };
        assert!(build_noise_model(&cal_valid).is_ok());

        // Invalid: T2 > 2*T1
        let cal_invalid = DeviceCalibration {
            name: "test".to_string(),
            num_qubits: 1,
            qubit_properties: vec![QubitProperties {
                t1_us: 100.0,
                t2_us: 250.0, // > 2 * 100 = 200
                frequency_ghz: 5.0,
                anharmonicity_ghz: -0.34,
            }],
            gate_properties: vec![GateProperties {
                gate_name: "sx".to_string(),
                qubits: vec![0],
                error_rate: 0.001,
                gate_time_ns: 35.0,
            }],
            readout_errors: vec![ReadoutError {
                qubit: 0,
                p0_given_0: 0.98,
                p1_given_1: 0.97,
            }],
            timestamp: 0,
        };
        assert!(build_noise_model(&cal_invalid).is_err());
    }

    #[test]
    fn test_thermal_relaxation_positive_rates() {
        let tr = thermal_relaxation_channel(300_000.0, 200_000.0, 35.0);
        // Rates should be non-negative
        let gamma1 = 1.0 - (-tr.gate_time / tr.t1).exp();
        let gamma2 = 1.0 - (-tr.gate_time / tr.t2).exp();
        assert!(gamma1 >= 0.0, "gamma1 should be non-negative");
        assert!(gamma2 >= 0.0, "gamma2 should be non-negative");
        assert!(gamma1 < 1.0, "gamma1 should be < 1 for reasonable T1");
        assert!(gamma2 < 1.0, "gamma2 should be < 1 for reasonable T2");
        // T2 should be clamped
        assert!(tr.t2 <= 2.0 * tr.t1);
    }

    #[test]
    fn test_readout_confusion_matrix_rows_sum_to_one() {
        let cal = DeviceCalibration::ibm_brisbane();
        let model = build_noise_model(&cal).expect("Should build model");

        for ro in &model.readout_errors {
            let row0_sum: f64 = ro.confusion_matrix[0].iter().sum();
            let row1_sum: f64 = ro.confusion_matrix[1].iter().sum();
            assert!(
                (row0_sum - 1.0).abs() < 1e-12,
                "Row 0 should sum to 1, got {}",
                row0_sum
            );
            assert!(
                (row1_sum - 1.0).abs() < 1e-12,
                "Row 1 should sum to 1, got {}",
                row1_sum
            );
        }
    }

    #[test]
    fn test_noise_preserves_normalization() {
        let mut rng = seeded_rng();
        let mut state = zero_state(2);

        let noise = SingleQubitNoise {
            qubit: 0,
            depolarizing_rate: 0.5,
            dephasing_rate: 0.1,
        };

        // Apply noise many times
        for _ in 0..100 {
            apply_single_qubit_noise(&mut state, &noise, &mut rng);
            let n = norm_sq(&state);
            assert!(
                (n - 1.0).abs() < 1e-10,
                "State should remain normalized after noise, got norm^2 = {}",
                n
            );
        }
    }

    #[test]
    fn test_high_error_produces_state_change() {
        let mut rng = seeded_rng();
        let original = zero_state(1);
        let mut state = original.clone();

        let noise = SingleQubitNoise {
            qubit: 0,
            depolarizing_rate: 1.0, // 100% error rate
            dephasing_rate: 0.0,
        };

        // With p=1.0, every application changes the state
        apply_single_qubit_noise(&mut state, &noise, &mut rng);

        let fid = fidelity(&original, &state);
        // After a random Pauli on |0>, fidelity should not be 1
        // (X|0> = |1>, Y|0> = i|1>, Z|0> = |0>; only Z preserves,
        //  so P(unchanged) = 1/3)
        // We just check it changed at least once over multiple trials
        let mut ever_changed = fid < 0.999;
        for _ in 0..20 {
            let mut s = original.clone();
            apply_single_qubit_noise(&mut s, &noise, &mut rng);
            if fidelity(&original, &s) < 0.999 {
                ever_changed = true;
                break;
            }
        }
        assert!(ever_changed, "High error rate should produce state changes");
    }

    #[test]
    fn test_zero_error_no_state_change() {
        let mut rng = seeded_rng();
        let original = zero_state(2);
        let mut state = original.clone();

        let noise = SingleQubitNoise {
            qubit: 0,
            depolarizing_rate: 0.0,
            dephasing_rate: 0.0,
        };

        for _ in 0..100 {
            apply_single_qubit_noise(&mut state, &noise, &mut rng);
        }

        let fid = fidelity(&original, &state);
        assert!(
            (fid - 1.0).abs() < 1e-12,
            "Zero error should produce no state change, fidelity = {}",
            fid
        );
    }

    #[test]
    fn test_average_gate_error() {
        let cal = DeviceCalibration::ibm_brisbane();
        let avg = average_gate_error(&cal);
        // IBM Brisbane sx error is ~0.0003
        assert!(avg > 0.0001, "Average gate error should be > 0.0001");
        assert!(avg < 0.01, "Average gate error should be < 0.01");

        let ideal = DeviceCalibration::ideal();
        let avg_ideal = average_gate_error(&ideal);
        assert_eq!(avg_ideal, 0.0, "Ideal device has zero gate error");
    }

    #[test]
    fn test_compare_devices_nonempty() {
        let devices = vec![
            DeviceCalibration::ibm_brisbane(),
            DeviceCalibration::google_sycamore(),
            DeviceCalibration::ionq_aria(),
            DeviceCalibration::rigetti_aspen_m3(),
            DeviceCalibration::ideal(),
        ];

        let table = compare_devices(&devices);
        assert!(!table.is_empty(), "Comparison table should not be empty");
        assert!(
            table.contains("ibm_brisbane"),
            "Table should contain ibm_brisbane"
        );
        assert!(
            table.contains("google_sycamore"),
            "Table should contain google_sycamore"
        );
        assert!(
            table.contains("ionq_aria"),
            "Table should contain ionq_aria"
        );
        assert!(
            table.contains("rigetti_aspen_m3"),
            "Table should contain rigetti_aspen_m3"
        );
        assert!(table.contains("ideal"), "Table should contain ideal");
    }

    #[test]
    fn test_json_roundtrip() {
        // Use a small calibration for reliable roundtrip
        let cal = DeviceCalibration::ideal();
        let json = calibration_to_json(&cal);
        let parsed = calibration_from_json(&json).expect("Should parse JSON");

        assert_eq!(parsed.name, cal.name);
        assert_eq!(parsed.num_qubits, cal.num_qubits);
        assert_eq!(parsed.qubit_properties.len(), cal.qubit_properties.len());
        assert_eq!(parsed.gate_properties.len(), cal.gate_properties.len());
        assert_eq!(parsed.readout_errors.len(), cal.readout_errors.len());
    }

    #[test]
    fn test_import_ibm_backend_properties_json() {
        let sample = r#"{
          "backend_name":"ibm_sample",
          "qubits":[
            [
              {"name":"T1","value":300.0},
              {"name":"T2","value":200.0},
              {"name":"frequency","value":5.1},
              {"name":"anharmonicity","value":-0.34},
              {"name":"readout_error","value":0.02}
            ],
            [
              {"name":"T1","value":280.0},
              {"name":"T2","value":190.0},
              {"name":"frequency","value":5.0},
              {"name":"anharmonicity","value":-0.33},
              {"name":"readout_error","value":0.03}
            ]
          ],
          "gates":[
            {"gate":"sx","qubits":[0],"parameters":[{"name":"gate_error","value":0.0003},{"name":"gate_length","value":35,"unit":"ns"}]},
            {"gate":"cx","qubits":[0,1],"parameters":[{"name":"gate_error","value":0.01},{"name":"gate_length","value":300,"unit":"ns"}]}
          ]
        }"#;

        let cal = import_ibm_backend_properties_json(sample).expect("ibm import");
        assert_eq!(cal.name, "ibm_sample");
        assert_eq!(cal.num_qubits, 2);
        assert_eq!(cal.qubit_properties.len(), 2);
        assert!(cal.gate_properties.iter().any(|g| g.gate_name == "cx"));
        build_noise_model(&cal).expect("noise model from ibm import");
    }

    #[test]
    fn test_import_ionq_characterization_json() {
        let sample = r#"{
          "name":"ionq_aria_mock",
          "num_qubits":4,
          "single_qubit":{"fidelity":0.9997,"duration_ns":135000},
          "two_qubit":{"fidelity":0.995,"duration_ns":600000},
          "readout":{"fidelity":0.996}
        }"#;

        let cal = import_ionq_characterization_json(sample).expect("ionq import");
        assert_eq!(cal.num_qubits, 4);
        let two_q = cal
            .gate_properties
            .iter()
            .filter(|g| g.qubits.len() == 2)
            .count();
        assert_eq!(two_q, 6, "all-to-all for 4 qubits = C(4,2)=6");
        build_noise_model(&cal).expect("noise model from ionq import");
    }

    #[test]
    fn test_import_cirq_device_spec_json() {
        let sample = r#"{
          "name":"sycamore_mock",
          "valid_qubits":[0,1,2,3],
          "single_qubit_gate_error":0.001,
          "two_qubit_gate_error":0.006,
          "measurement_error":0.04,
          "connectivity":[[0,1],[1,2],[2,3]]
        }"#;

        let cal = import_cirq_device_spec_json(sample).expect("cirq import");
        assert_eq!(cal.num_qubits, 4);
        assert!(cal
            .gate_properties
            .iter()
            .any(|g| g.gate_name == "syc" && g.qubits == vec![1, 2]));
        build_noise_model(&cal).expect("noise model from cirq import");
    }

    #[test]
    fn test_import_rigetti_isa_noise_json() {
        let sample = r#"{
          "name":"rigetti_mock",
          "num_qubits":5,
          "one_qubit":{"fidelity":0.998},
          "two_qubit":{"fidelity":0.95},
          "readout":{"fidelity":0.95},
          "edges":[[0,1],[1,2],[2,3],[3,4]]
        }"#;

        let cal = import_rigetti_isa_noise_json(sample).expect("rigetti import");
        assert_eq!(cal.num_qubits, 5);
        assert!(cal.gate_properties.iter().any(|g| g.gate_name == "cz"));
        build_noise_model(&cal).expect("noise model from rigetti import");
    }

    #[test]
    fn test_import_vendor_auto_detect() {
        let ibm_like = r#"{
          "backend_name":"ibm_auto",
          "qubits":[[{"name":"T1","value":300.0},{"name":"T2","value":200.0},{"name":"frequency","value":5.0},{"name":"anharmonicity","value":-0.34}]],
          "gates":[{"gate":"sx","qubits":[0],"parameters":[{"name":"gate_error","value":0.0003},{"name":"gate_length","value":35,"unit":"ns"}]}]
        }"#;
        let cal = import_vendor_calibration_json(ibm_like).expect("auto import");
        assert_eq!(cal.name, "ibm_auto");
        assert_eq!(cal.num_qubits, 1);
    }

    #[test]
    fn test_readout_noise_application() {
        let mut rng = seeded_rng();
        let noise = ReadoutNoiseChannel {
            qubit: 0,
            confusion_matrix: [[0.98, 0.02], [0.03, 0.97]],
        };

        // Over many trials, readout noise should sometimes flip
        let mut flips_0 = 0;
        let mut flips_1 = 0;
        let trials = 10_000;
        for _ in 0..trials {
            if apply_readout_noise(0, &noise, &mut rng) == 1 {
                flips_0 += 1;
            }
            if apply_readout_noise(1, &noise, &mut rng) == 0 {
                flips_1 += 1;
            }
        }

        // Should be approximately 2% and 3%
        let rate_0 = flips_0 as f64 / trials as f64;
        let rate_1 = flips_1 as f64 / trials as f64;
        assert!(
            (rate_0 - 0.02).abs() < 0.01,
            "P(1|0) should be ~0.02, got {}",
            rate_0
        );
        assert!(
            (rate_1 - 0.03).abs() < 0.01,
            "P(0|1) should be ~0.03, got {}",
            rate_1
        );
    }

    #[test]
    fn test_ibm_sherbrooke_preset() {
        let cal = DeviceCalibration::ibm_sherbrooke();
        assert_eq!(cal.num_qubits, 127);
        assert_eq!(cal.name, "ibm_sherbrooke");
        let model = build_noise_model(&cal).expect("Should build Sherbrooke model");
        assert_eq!(model.single_qubit_errors.len(), 127);
    }

    #[test]
    fn test_rigetti_aspen_m3_preset() {
        let cal = DeviceCalibration::rigetti_aspen_m3();
        assert_eq!(cal.num_qubits, 80);
        assert_eq!(cal.name, "rigetti_aspen_m3");
        // Rigetti has higher 2Q error rates
        let avg_2q = average_two_qubit_gate_error(&cal);
        assert!(avg_2q > 0.01, "Rigetti 2Q error should be > 0.01");
    }

    #[test]
    fn test_two_qubit_noise_application() {
        let mut rng = seeded_rng();
        let original = zero_state(2);
        let mut state = original.clone();

        let noise = TwoQubitNoise {
            qubits: (0, 1),
            depolarizing_rate: 1.0,
            crosstalk_rate: 0.0,
        };

        let mut ever_changed = false;
        for _ in 0..20 {
            let mut s = original.clone();
            apply_two_qubit_noise(&mut s, &noise, &mut rng);
            if fidelity(&original, &s) < 0.999 {
                ever_changed = true;
                break;
            }
        }
        assert!(ever_changed, "Two-qubit noise with p=1.0 should change state");
    }

    #[test]
    fn test_thermal_relaxation_application() {
        let mut rng = seeded_rng();

        // Create |1> state (1 qubit)
        let mut state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        let tr = ThermalRelaxation {
            qubit: 0,
            t1: 100.0,    // Short T1
            t2: 100.0,    // Short T2
            gate_time: 50.0, // gate_time = T1/2, so significant damping
            excited_state_population: 0.01,
        };

        // Apply many times; |1> should decay toward |0>
        let mut total_ground_pop = 0.0;
        let trials = 1000;
        for _ in 0..trials {
            let mut s = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
            apply_thermal_relaxation(&mut s, &tr, 0, &mut rng);
            total_ground_pop += s[0].norm_sqr();
        }

        let avg_ground = total_ground_pop / trials as f64;
        // With T1 = gate_time * 2, about 39% of amplitude should decay
        assert!(
            avg_ground > 0.05,
            "Thermal relaxation should transfer some population to ground, got {}",
            avg_ground
        );
    }
}
