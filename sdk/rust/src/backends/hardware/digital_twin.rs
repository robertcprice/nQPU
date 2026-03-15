//! Digital Twin, Calibration Drift, and Compiler Benchmark System
//!
//! Provides a high-fidelity digital twin of superconducting quantum processors,
//! built from real calibration data (IBM/Google JSON formats) or published device
//! presets. The twin predicts circuit fidelity, models calibration drift over
//! time using stochastic processes (Wiener, Poisson), and benchmarks native-gate
//! compilation overhead across ECR, SqrtISWAP, and CZ gate families.
//!
//! # Architecture
//!
//! ```text
//! CalibrationData ──┬── from_ibm_json()    ──┐
//!                   └── from_google_json()  ──┤
//!                                             ▼
//!                                     DigitalTwin
//!                                       │
//!                           ┌───────────┼───────────┐
//!                           ▼           ▼           ▼
//!                       validate()  predict()  compare_presets()
//!                           │
//!                           ▼
//!                   ValidationReport
//!
//! CalibrationDrift ── at_time(hours) ──► CalibrationData (drifted)
//!                  └── stability_report() ──► StabilityReport
//!
//! CompilerBenchmark ── benchmark_all(n) ──► Vec<CompilationResult>
//! ```
//!
//! # Drift Physics
//!
//! - **T1**: Wiener process with 2% volatility/sqrt(hr), modeling TLS fluctuations
//! - **T2**: Correlated with T1 (T2 <= 2*T1), plus independent pure dephasing noise
//! - **Frequency**: Poisson TLS jumps (~100 kHz amplitude, rate 0.5/qubit/hr)
//! - **Gate fidelity**: Sinusoidal aging from parameter drift, period ~12 hours
//!
//! # References
//!
//! - Klimov et al., PRL 121 (2018) -- Fluctuations of energy-relaxation times
//! - Burnett et al., npj Quantum Info 5 (2019) -- TLS frequency fluctuations
//! - Schlör et al., PRL 123 (2019) -- Correlating decoherence with TLS

use std::collections::HashMap;
use std::f64::consts::PI;

use super::superconducting::{
    compile_to_native, ChipTopology, NativeGate, NativeGateFamily, TransmonProcessor,
    TransmonQubit,
};

use crate::gates::{Gate, GateType};

// ===================================================================
// ERROR TYPE
// ===================================================================

/// Errors from digital twin operations.
#[derive(Debug, Clone)]
pub enum DigitalTwinError {
    /// Calibration data is invalid or inconsistent.
    InvalidCalibration(String),
    /// JSON parsing failed.
    ParseError(String),
}

impl std::fmt::Display for DigitalTwinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCalibration(msg) => write!(f, "Invalid calibration: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for DigitalTwinError {}

pub type DigitalTwinResult<T> = Result<T, DigitalTwinError>;

// ===================================================================
// CALIBRATION DATA
// ===================================================================

/// Snapshot of a superconducting processor's calibration parameters.
///
/// This is the portable interchange format: it can be constructed from
/// vendor-specific JSON (IBM backend properties, Google Cirq device specs)
/// or from published device presets, then fed into a `DigitalTwin` for
/// simulation and fidelity prediction.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Qubit transition frequencies (0->1) in GHz.
    pub qubit_frequencies_ghz: Vec<f64>,
    /// Energy relaxation times T1 in microseconds.
    pub t1_us: Vec<f64>,
    /// Dephasing times T2 in microseconds.
    pub t2_us: Vec<f64>,
    /// Readout assignment fidelities per qubit (0.0 to 1.0).
    pub readout_fidelities: Vec<f64>,
    /// Single-qubit gate fidelities per qubit (0.0 to 1.0).
    pub single_gate_fidelities: Vec<f64>,
    /// Two-qubit gate fidelities keyed by (qubit_a, qubit_b).
    pub two_qubit_fidelities: HashMap<(usize, usize), f64>,
    /// Coupling map as a list of (qubit_a, qubit_b) pairs.
    pub coupling_map: Vec<(usize, usize)>,
    /// Native gate family name: "ecr", "sqrt_iswap", "cz".
    pub native_gate_family: String,
    /// Device name for identification.
    pub device_name: String,
}

impl CalibrationData {
    /// Number of qubits in this calibration snapshot.
    pub fn num_qubits(&self) -> usize {
        self.qubit_frequencies_ghz.len()
    }

    /// Parse IBM Quantum backend properties JSON format.
    ///
    /// Expects the structure from IBM's `/properties` endpoint:
    /// ```json
    /// {
    ///   "backend_name": "ibm_brisbane",
    ///   "qubits": [
    ///     [{"name": "T1", "value": 120.5, "unit": "us"}, {"name": "T2", ...}, {"name": "frequency", ...}, {"name": "readout_assignment_error", ...}],
    ///     ...
    ///   ],
    ///   "gates": [
    ///     {"gate": "sx", "qubits": [0], "parameters": [{"name": "gate_error", "value": 0.0003}]},
    ///     {"gate": "ecr", "qubits": [0, 1], "parameters": [{"name": "gate_error", "value": 0.008}]},
    ///     ...
    ///   ]
    /// }
    /// ```
    #[cfg(feature = "serde")]
    pub fn from_ibm_json(json_str: &str) -> DigitalTwinResult<Self> {
        let root: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| DigitalTwinError::ParseError(format!("JSON parse error: {}", e)))?;

        let backend_name = root
            .get("backend_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown_ibm_device")
            .to_string();

        // Parse qubit properties.
        let qubits_arr = root
            .get("qubits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                DigitalTwinError::ParseError("Missing 'qubits' array".to_string())
            })?;

        let n = qubits_arr.len();
        let mut frequencies = vec![5.0; n];
        let mut t1 = vec![100.0; n];
        let mut t2 = vec![80.0; n];
        let mut readout_fid = vec![0.95; n];

        for (qi, qubit_props) in qubits_arr.iter().enumerate() {
            if let Some(props) = qubit_props.as_array() {
                for prop in props {
                    let name = prop.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let value = prop.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    match name {
                        "T1" => t1[qi] = value,
                        "T2" => t2[qi] = value,
                        "frequency" => frequencies[qi] = value / 1e9, // Hz -> GHz
                        "readout_assignment_error" => readout_fid[qi] = 1.0 - value,
                        _ => {}
                    }
                }
            }
        }

        // Parse gate properties.
        let mut single_fid = vec![0.9995; n];
        let mut two_q_fid = HashMap::new();
        let mut coupling_map = Vec::new();
        let mut native_gate = "ecr".to_string();

        if let Some(gates_arr) = root.get("gates").and_then(|v| v.as_array()) {
            for gate_entry in gates_arr {
                let gate_name = gate_entry
                    .get("gate")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let gate_qubits: Vec<usize> = gate_entry
                    .get("qubits")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|u| u as usize))
                            .collect()
                    })
                    .unwrap_or_default();

                let gate_error = gate_entry
                    .get("parameters")
                    .and_then(|v| v.as_array())
                    .and_then(|params| {
                        params.iter().find_map(|p| {
                            if p.get("name").and_then(|n| n.as_str()) == Some("gate_error") {
                                p.get("value").and_then(|v| v.as_f64())
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or(0.001);

                match gate_name {
                    "sx" | "rz" | "x" => {
                        if let Some(&q) = gate_qubits.first() {
                            if q < n {
                                single_fid[q] = 1.0 - gate_error;
                            }
                        }
                    }
                    "ecr" | "cx" | "cz" => {
                        if gate_qubits.len() == 2 {
                            let (a, b) = (gate_qubits[0], gate_qubits[1]);
                            two_q_fid.insert((a, b), 1.0 - gate_error);
                            coupling_map.push((a, b));
                            if gate_name == "ecr" || gate_name == "cx" {
                                native_gate = "ecr".to_string();
                            } else {
                                native_gate = "cz".to_string();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(CalibrationData {
            qubit_frequencies_ghz: frequencies,
            t1_us: t1,
            t2_us: t2,
            readout_fidelities: readout_fid,
            single_gate_fidelities: single_fid,
            two_qubit_fidelities: two_q_fid,
            coupling_map,
            native_gate_family: native_gate,
            device_name: backend_name,
        })
    }

    /// Parse Google Cirq device specification JSON format.
    ///
    /// Expects the structure from Cirq's device metadata:
    /// ```json
    /// {
    ///   "device_name": "google_sycamore",
    ///   "qubits": [
    ///     {"id": "0_0", "frequency_ghz": 5.5, "t1_us": 16.0, "t2_us": 12.0, "readout_fidelity": 0.965}
    ///   ],
    ///   "couplers": [
    ///     {"qubit_a": 0, "qubit_b": 1, "two_qubit_fidelity": 0.995}
    ///   ],
    ///   "native_gate": "sqrt_iswap"
    /// }
    /// ```
    #[cfg(feature = "serde")]
    pub fn from_google_json(json_str: &str) -> DigitalTwinResult<Self> {
        let root: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| DigitalTwinError::ParseError(format!("JSON parse error: {}", e)))?;

        let device_name = root
            .get("device_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown_google_device")
            .to_string();

        let native_gate = root
            .get("native_gate")
            .and_then(|v| v.as_str())
            .unwrap_or("sqrt_iswap")
            .to_string();

        // Parse qubits.
        let qubits_arr = root
            .get("qubits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                DigitalTwinError::ParseError("Missing 'qubits' array".to_string())
            })?;

        let n = qubits_arr.len();
        let mut frequencies = Vec::with_capacity(n);
        let mut t1 = Vec::with_capacity(n);
        let mut t2 = Vec::with_capacity(n);
        let mut readout_fid = Vec::with_capacity(n);
        let mut single_fid = Vec::with_capacity(n);

        for qubit in qubits_arr {
            frequencies.push(
                qubit
                    .get("frequency_ghz")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(5.5),
            );
            t1.push(
                qubit
                    .get("t1_us")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(20.0),
            );
            t2.push(
                qubit
                    .get("t2_us")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(10.0),
            );
            readout_fid.push(
                qubit
                    .get("readout_fidelity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.96),
            );
            single_fid.push(
                qubit
                    .get("single_gate_fidelity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.999),
            );
        }

        // Parse couplers.
        let mut two_q_fid = HashMap::new();
        let mut coupling_map = Vec::new();

        if let Some(couplers_arr) = root.get("couplers").and_then(|v| v.as_array()) {
            for coupler in couplers_arr {
                let a = coupler
                    .get("qubit_a")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let b = coupler
                    .get("qubit_b")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let fid = coupler
                    .get("two_qubit_fidelity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.99);
                coupling_map.push((a, b));
                two_q_fid.insert((a, b), fid);
            }
        }

        Ok(CalibrationData {
            qubit_frequencies_ghz: frequencies,
            t1_us: t1,
            t2_us: t2,
            readout_fidelities: readout_fid,
            single_gate_fidelities: single_fid,
            two_qubit_fidelities: two_q_fid,
            coupling_map,
            native_gate_family: native_gate,
            device_name,
        })
    }

    /// Build CalibrationData without serde, using a simple key-value format.
    ///
    /// This is always available regardless of feature flags. For production
    /// use with real vendor JSON, enable the `serde` feature.
    pub fn from_processor(proc: &TransmonProcessor, device_name: &str) -> Self {
        let n = proc.num_qubits();
        let mut two_q_fid = HashMap::new();
        let mut coupling_map = Vec::new();

        for coupler in &proc.topology.couplers {
            coupling_map.push((coupler.qubit_a, coupler.qubit_b));
            two_q_fid.insert(
                (coupler.qubit_a, coupler.qubit_b),
                proc.two_qubit_fidelity,
            );
        }

        let native_gate_family = match proc.native_2q_gate {
            NativeGateFamily::ECR => "ecr".to_string(),
            NativeGateFamily::SqrtISWAP => "sqrt_iswap".to_string(),
            NativeGateFamily::CZ => "cz".to_string(),
            NativeGateFamily::FSim => "fsim".to_string(),
        };

        CalibrationData {
            qubit_frequencies_ghz: proc.qubits.iter().map(|q| q.frequency_ghz).collect(),
            t1_us: proc.qubits.iter().map(|q| q.t1_us).collect(),
            t2_us: proc.qubits.iter().map(|q| q.t2_us).collect(),
            readout_fidelities: proc.qubits.iter().map(|q| q.readout_fidelity).collect(),
            single_gate_fidelities: proc
                .qubits
                .iter()
                .map(|q| q.single_gate_fidelity)
                .collect(),
            two_qubit_fidelities: two_q_fid,
            coupling_map,
            native_gate_family,
            device_name: device_name.to_string(),
        }
    }

    /// Validate that calibration data is physically consistent.
    pub fn validate(&self) -> DigitalTwinResult<()> {
        let n = self.num_qubits();
        if n == 0 {
            return Err(DigitalTwinError::InvalidCalibration(
                "Zero qubits".to_string(),
            ));
        }
        if self.t1_us.len() != n
            || self.t2_us.len() != n
            || self.readout_fidelities.len() != n
            || self.single_gate_fidelities.len() != n
        {
            return Err(DigitalTwinError::InvalidCalibration(
                "Mismatched array lengths".to_string(),
            ));
        }
        for i in 0..n {
            if self.t2_us[i] > 2.0 * self.t1_us[i] * 1.01 {
                // Allow 1% tolerance for rounding
                return Err(DigitalTwinError::InvalidCalibration(format!(
                    "Qubit {}: T2={:.1} > 2*T1={:.1} violates physics",
                    i,
                    self.t2_us[i],
                    2.0 * self.t1_us[i]
                )));
            }
            if self.readout_fidelities[i] < 0.0 || self.readout_fidelities[i] > 1.0 {
                return Err(DigitalTwinError::InvalidCalibration(format!(
                    "Qubit {}: readout fidelity {} out of [0, 1]",
                    i, self.readout_fidelities[i]
                )));
            }
        }
        Ok(())
    }

    /// Average single-qubit gate fidelity across all qubits.
    pub fn avg_single_gate_fidelity(&self) -> f64 {
        if self.single_gate_fidelities.is_empty() {
            return 0.0;
        }
        self.single_gate_fidelities.iter().sum::<f64>()
            / self.single_gate_fidelities.len() as f64
    }

    /// Average two-qubit gate fidelity across all edges.
    pub fn avg_two_qubit_fidelity(&self) -> f64 {
        if self.two_qubit_fidelities.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.two_qubit_fidelities.values().sum();
        sum / self.two_qubit_fidelities.len() as f64
    }
}

// ===================================================================
// VALIDATION REPORT
// ===================================================================

/// Results from running a QCVV validation suite on a digital twin.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Bell state preparation fidelity (ideal = 1.0).
    pub bell_fidelity: f64,
    /// GHZ state preparation fidelity (ideal = 1.0).
    pub ghz_fidelity: f64,
    /// Randomized benchmarking error per Clifford gate.
    pub rb_error_per_gate: f64,
    /// Estimated quantum volume (log2).
    pub quantum_volume: u32,
    /// Predicted fidelity for a shallow circuit (depth 10, 10 2Q gates).
    pub predicted_shallow_fidelity: f64,
    /// Predicted fidelity for a deep circuit (depth 100, 100 2Q gates).
    pub predicted_deep_fidelity: f64,
    /// Device name for identification.
    pub device_name: String,
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Validation Report: {} ===", self.device_name)?;
        writeln!(f, "  Bell fidelity:            {:.6}", self.bell_fidelity)?;
        writeln!(f, "  GHZ fidelity:             {:.6}", self.ghz_fidelity)?;
        writeln!(
            f,
            "  RB error/gate:            {:.6}",
            self.rb_error_per_gate
        )?;
        writeln!(f, "  Quantum volume:           2^{}", self.quantum_volume)?;
        writeln!(
            f,
            "  Shallow circuit fidelity: {:.6}",
            self.predicted_shallow_fidelity
        )?;
        writeln!(
            f,
            "  Deep circuit fidelity:    {:.6}",
            self.predicted_deep_fidelity
        )?;
        Ok(())
    }
}

// ===================================================================
// DIGITAL TWIN
// ===================================================================

/// A digital twin of a superconducting quantum processor.
///
/// Wraps a `TransmonProcessor` with calibration-derived parameters and
/// provides fidelity prediction, QCVV validation, and cross-device
/// comparison capabilities.
#[derive(Debug, Clone)]
pub struct DigitalTwin {
    /// The underlying processor model.
    pub processor: TransmonProcessor,
    /// Calibration data used to build this twin.
    pub calibration: CalibrationData,
}

impl DigitalTwin {
    /// Build a digital twin from calibration data.
    ///
    /// Constructs a `TransmonProcessor` whose per-qubit parameters and
    /// coupling map match the supplied calibration snapshot.
    pub fn from_calibration(data: &CalibrationData) -> DigitalTwinResult<Self> {
        data.validate()?;

        let n = data.num_qubits();

        let native_gate = match data.native_gate_family.as_str() {
            "ecr" | "ECR" => NativeGateFamily::ECR,
            "sqrt_iswap" | "SqrtISWAP" | "√iSWAP" => NativeGateFamily::SqrtISWAP,
            "cz" | "CZ" => NativeGateFamily::CZ,
            "fsim" | "FSim" => NativeGateFamily::FSim,
            other => {
                return Err(DigitalTwinError::InvalidCalibration(format!(
                    "Unknown native gate family: '{}'",
                    other
                )));
            }
        };

        // Default anharmonicity by gate family (MHz).
        let anharmonicity_mhz = match native_gate {
            NativeGateFamily::ECR => -330.0,
            NativeGateFamily::SqrtISWAP | NativeGateFamily::FSim => -225.0,
            NativeGateFamily::CZ => -280.0,
        };

        let qubits: Vec<TransmonQubit> = (0..n)
            .map(|i| TransmonQubit {
                index: i,
                frequency_ghz: data.qubit_frequencies_ghz[i],
                anharmonicity_mhz,
                t1_us: data.t1_us[i],
                t2_us: data.t2_us[i],
                single_gate_fidelity: data.single_gate_fidelities[i],
                readout_fidelity: data.readout_fidelities[i],
                thermal_population: 0.01,
                gate_time_ns: 25.0,
            })
            .collect();

        let coupling_mhz = 3.5; // Default coupling strength.
        let couplers = data
            .coupling_map
            .iter()
            .map(|&(a, b)| super::superconducting::CouplerLink {
                qubit_a: a,
                qubit_b: b,
                coupling_mhz,
                zz_khz: None,
            })
            .collect();

        let avg_2q_fidelity = if data.two_qubit_fidelities.is_empty() {
            0.99
        } else {
            data.avg_two_qubit_fidelity()
        };

        let topology = ChipTopology::custom(n, couplers);

        let processor = TransmonProcessor {
            qubits,
            topology,
            native_2q_gate: native_gate,
            two_qubit_fidelity: avg_2q_fidelity,
            two_qubit_gate_time_ns: match native_gate {
                NativeGateFamily::ECR => 250.0,
                NativeGateFamily::SqrtISWAP => 30.0,
                NativeGateFamily::CZ => 180.0,
                NativeGateFamily::FSim => 30.0,
            },
            readout_time_ns: 800.0,
            measurement_crosstalk: 0.015,
            temperature_mk: 15.0,
        };

        Ok(Self {
            processor,
            calibration: data.clone(),
        })
    }

    // ---------------------------------------------------------------
    // Preset constructors
    // ---------------------------------------------------------------

    /// IBM Eagle digital twin (127 qubits, heavy-hex, ECR).
    ///
    /// Uses published specifications: T1 ~ 120 us, T2 ~ 80 us,
    /// 2Q fidelity ~ 99.0%, ECR gate at 300 ns.
    pub fn ibm_eagle(n: usize) -> Self {
        let proc = build_preset_processor(
            n,
            4.8,
            0.05,
            8,
            -340.0,
            120.0,
            80.0,
            0.9996,
            0.98,
            0.015,
            35.0,
            NativeGateFamily::ECR,
            0.990,
            300.0,
            3.5,
            true, // heavy-hex
        );
        let cal = CalibrationData::from_processor(&proc, "ibm_eagle");
        Self {
            processor: proc,
            calibration: cal,
        }
    }

    /// IBM Heron digital twin (156 qubits, heavy-hex, ECR, improved coherence).
    ///
    /// Uses published specifications: T1 ~ 300 us, T2 ~ 200 us,
    /// 2Q fidelity ~ 99.5%, ECR gate at 200 ns.
    pub fn ibm_heron(n: usize) -> Self {
        let proc = build_preset_processor(
            n,
            4.9,
            0.04,
            10,
            -320.0,
            300.0,
            200.0,
            0.9998,
            0.995,
            0.008,
            25.0,
            NativeGateFamily::ECR,
            0.995,
            200.0,
            3.0,
            true, // heavy-hex
        );
        let cal = CalibrationData::from_processor(&proc, "ibm_heron");
        Self {
            processor: proc,
            calibration: cal,
        }
    }

    /// Google Sycamore digital twin (53 qubits, grid, SqrtISWAP).
    ///
    /// Uses published specifications: T1 ~ 16 us, T2 ~ 12 us,
    /// 2Q fidelity ~ 99.5%, SqrtISWAP gate at 32 ns.
    pub fn google_sycamore(n: usize) -> Self {
        let proc = build_preset_processor(
            n,
            5.5,
            0.15,
            7,
            -220.0,
            16.0,
            12.0,
            0.9985,
            0.965,
            0.02,
            25.0,
            NativeGateFamily::SqrtISWAP,
            0.995,
            32.0,
            5.0,
            false, // grid
        );
        let cal = CalibrationData::from_processor(&proc, "google_sycamore");
        Self {
            processor: proc,
            calibration: cal,
        }
    }

    /// Google Willow digital twin (105 qubits, grid, SqrtISWAP).
    ///
    /// Uses published specifications: T1 ~ 68 us, T2 ~ 30 us,
    /// 2Q fidelity ~ 99.7%, SqrtISWAP gate at 26 ns.
    pub fn google_willow(n: usize) -> Self {
        let proc = build_preset_processor(
            n,
            5.2,
            0.12,
            8,
            -230.0,
            68.0,
            30.0,
            0.9993,
            0.993,
            0.006,
            22.0,
            NativeGateFamily::SqrtISWAP,
            0.997,
            26.0,
            4.5,
            false, // grid
        );
        let cal = CalibrationData::from_processor(&proc, "google_willow");
        Self {
            processor: proc,
            calibration: cal,
        }
    }

    // ---------------------------------------------------------------
    // Validation and prediction
    // ---------------------------------------------------------------

    /// Run a QCVV validation suite and return aggregate results.
    ///
    /// Evaluates:
    /// - Bell state fidelity (1 CNOT on 2 qubits)
    /// - GHZ fidelity (n-1 CNOTs in a chain, for min(n, 5) qubits)
    /// - Randomized benchmarking error rate (average single-qubit error)
    /// - Quantum volume estimate
    /// - Shallow/deep circuit fidelity predictions
    pub fn validate(&self) -> ValidationReport {
        let n = self.processor.num_qubits();
        let avg_1q_error = self.avg_single_qubit_error();
        let avg_2q_error = 1.0 - self.processor.two_qubit_fidelity;

        // Bell fidelity: H + CNOT on qubits 0,1.
        // F_bell = (1 - e_1q) * (1 - e_2q) * (1 - e_readout)^2
        let avg_readout_error = 1.0 - self.avg_readout_fidelity();
        let bell_fidelity =
            (1.0 - avg_1q_error) * (1.0 - avg_2q_error) * (1.0 - avg_readout_error).powi(2);

        // GHZ fidelity: 1 H + (k-1) CNOTs for k = min(n, 5).
        let k = n.min(5);
        let ghz_fidelity = (1.0 - avg_1q_error)
            * (1.0 - avg_2q_error).powi((k - 1) as i32)
            * (1.0 - avg_readout_error).powi(k as i32);

        // RB error per gate: average single-qubit depolarizing error.
        let rb_error_per_gate = avg_1q_error;

        // Quantum volume estimate: largest m such that QV circuit succeeds.
        // QV_circuit(m) has m layers of random SU(4) on m qubits.
        // Success threshold: heavy output probability > 2/3.
        // Approximate: F_QV = (1 - e_2q)^(m * (m/2)) * (1 - e_1q)^(m * m * 2)
        let mut qv = 1u32;
        for m in 1..=n.min(32) {
            let num_2q_gates = m * (m / 2);
            let num_1q_gates = m * m * 2;
            let fid = (1.0 - avg_2q_error).powi(num_2q_gates as i32)
                * (1.0 - avg_1q_error).powi(num_1q_gates as i32);
            if fid > 2.0 / 3.0 {
                qv = m as u32;
            } else {
                break;
            }
        }

        let predicted_shallow = self.predict_circuit_fidelity(20, 10, 10);
        let predicted_deep = self.predict_circuit_fidelity(200, 100, 100);

        ValidationReport {
            bell_fidelity,
            ghz_fidelity,
            rb_error_per_gate,
            quantum_volume: qv,
            predicted_shallow_fidelity: predicted_shallow,
            predicted_deep_fidelity: predicted_deep,
            device_name: self.calibration.device_name.clone(),
        }
    }

    /// Predict the output fidelity of a circuit using a multiplicative error model.
    ///
    /// Models circuit fidelity as:
    ///   F = (1 - e_1q)^{num_1q} * (1 - e_2q)^{num_2q} * (1 - e_idle)^{depth}
    ///
    /// where e_idle accounts for T1/T2 decoherence during idle time between layers.
    pub fn predict_circuit_fidelity(&self, num_1q: usize, num_2q: usize, depth: usize) -> f64 {
        let avg_1q_error = self.avg_single_qubit_error();
        let avg_2q_error = 1.0 - self.processor.two_qubit_fidelity;

        // Idle decoherence per layer: depends on gate time relative to T1/T2.
        let avg_t1_ns = self.calibration.t1_us.iter().sum::<f64>()
            / self.calibration.t1_us.len() as f64
            * 1000.0;
        let layer_time_ns = self.processor.two_qubit_gate_time_ns;
        let idle_error_per_layer = 1.0 - (-layer_time_ns / avg_t1_ns).exp();

        let fidelity = (1.0 - avg_1q_error).powi(num_1q as i32)
            * (1.0 - avg_2q_error).powi(num_2q as i32)
            * (1.0 - idle_error_per_layer).powi(depth as i32);

        fidelity.max(0.0)
    }

    /// Compare all four preset digital twins and return their validation reports.
    pub fn compare_presets(n: usize) -> Vec<(String, ValidationReport)> {
        let presets: Vec<(&str, DigitalTwin)> = vec![
            ("ibm_eagle", Self::ibm_eagle(n)),
            ("ibm_heron", Self::ibm_heron(n)),
            ("google_sycamore", Self::google_sycamore(n)),
            ("google_willow", Self::google_willow(n)),
        ];

        presets
            .into_iter()
            .map(|(name, twin)| (name.to_string(), twin.validate()))
            .collect()
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    fn avg_single_qubit_error(&self) -> f64 {
        if self.processor.qubits.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .processor
            .qubits
            .iter()
            .map(|q| 1.0 - q.single_gate_fidelity)
            .sum();
        sum / self.processor.qubits.len() as f64
    }

    fn avg_readout_fidelity(&self) -> f64 {
        if self.processor.qubits.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .processor
            .qubits
            .iter()
            .map(|q| q.readout_fidelity)
            .sum();
        sum / self.processor.qubits.len() as f64
    }
}

// ===================================================================
// PRESET PROCESSOR BUILDER
// ===================================================================

/// Internal helper to build a TransmonProcessor from preset parameters.
#[allow(clippy::too_many_arguments)]
fn build_preset_processor(
    num_qubits: usize,
    base_freq_ghz: f64,
    freq_spread: f64,
    freq_mod: usize,
    anharmonicity_mhz: f64,
    t1_us: f64,
    t2_us: f64,
    single_gate_fidelity: f64,
    readout_fidelity: f64,
    thermal_population: f64,
    gate_time_ns: f64,
    native_gate: NativeGateFamily,
    two_qubit_fidelity: f64,
    two_qubit_gate_time_ns: f64,
    coupling_mhz: f64,
    heavy_hex: bool,
) -> TransmonProcessor {
    let qubits: Vec<TransmonQubit> = (0..num_qubits)
        .map(|i| TransmonQubit {
            index: i,
            frequency_ghz: base_freq_ghz + freq_spread * (i as f64 % freq_mod as f64),
            anharmonicity_mhz,
            t1_us,
            t2_us,
            single_gate_fidelity,
            readout_fidelity,
            thermal_population,
            gate_time_ns,
        })
        .collect();

    let topology = if heavy_hex {
        ChipTopology::heavy_hex(num_qubits, coupling_mhz)
    } else {
        let cols = ((num_qubits as f64).sqrt().ceil()) as usize;
        let rows = (num_qubits + cols - 1) / cols;
        ChipTopology::grid(rows, cols, coupling_mhz)
    };

    TransmonProcessor {
        qubits,
        topology,
        native_2q_gate: native_gate,
        two_qubit_fidelity,
        two_qubit_gate_time_ns,
        readout_time_ns: 800.0,
        measurement_crosstalk: 0.015,
        temperature_mk: 15.0,
    }
}

// ===================================================================
// CALIBRATION DRIFT (STOCHASTIC MODEL)
// ===================================================================

/// Stochastic calibration drift model for superconducting processors.
///
/// Simulates how device parameters evolve over time between calibration
/// cycles using physics-motivated stochastic processes:
///
/// - **T1**: Wiener process with 2% volatility per sqrt(hour), capturing
///   the 1/f^alpha noise spectrum of TLS-dominated relaxation.
/// - **T2**: Correlated with T1 (maintains T2 <= 2*T1 constraint) plus
///   independent pure dephasing noise from flux and charge noise.
/// - **Frequency**: Poisson process for TLS frequency jumps (typically
///   ~100 kHz amplitude, rate ~0.5 events/qubit/hour).
/// - **Gate fidelity**: Sinusoidal aging with period ~12 hours from
///   slow parameter drift (e.g., flux bias point drift).
#[derive(Debug, Clone)]
pub struct CalibrationDriftModel {
    /// Base calibration (time = 0 snapshot).
    base: CalibrationData,
    /// PRNG seed for reproducible drift trajectories.
    seed: u64,
    /// T1 volatility: fractional RMS change per sqrt(hour).
    pub t1_volatility: f64,
    /// Frequency jump amplitude in GHz.
    pub freq_jump_ghz: f64,
    /// Poisson rate of frequency jumps per qubit per hour.
    pub freq_jump_rate_per_hr: f64,
    /// Gate fidelity aging period in hours.
    pub fidelity_aging_period_hr: f64,
    /// Peak fidelity degradation amplitude.
    pub fidelity_aging_amplitude: f64,
}

impl CalibrationDriftModel {
    /// Create a new drift model from base calibration data.
    ///
    /// Uses physics-motivated default parameters:
    /// - T1 volatility: 2% per sqrt(hour)
    /// - TLS jumps: 100 kHz, rate 0.5/qubit/hr
    /// - Fidelity aging: 0.2% amplitude, 12-hour period
    pub fn new(base: CalibrationData, seed: u64) -> Self {
        Self {
            base,
            seed,
            t1_volatility: 0.02,
            freq_jump_ghz: 100e-6, // 100 kHz in GHz
            freq_jump_rate_per_hr: 0.5,
            fidelity_aging_period_hr: 12.0,
            fidelity_aging_amplitude: 0.002,
        }
    }

    /// Generate drifted calibration data at a given time (in hours).
    ///
    /// Each qubit gets an independent drift trajectory seeded by
    /// `(self.seed, qubit_index)` for reproducibility.
    pub fn at_time(&self, hours: f64) -> CalibrationData {
        let n = self.base.num_qubits();
        let mut t1_us = Vec::with_capacity(n);
        let mut t2_us = Vec::with_capacity(n);
        let mut frequencies = Vec::with_capacity(n);
        let mut single_fid = Vec::with_capacity(n);
        let mut readout_fid = self.base.readout_fidelities.clone();

        for i in 0..n {
            let mut rng = Xorshift64::new(self.seed.wrapping_add(i as u64));

            // --- T1 drift: Wiener process ---
            // W(t) is normally distributed with variance t.
            // We discretize: dW = N(0,1) * sqrt(dt) for each time step.
            // For a single point query, W(hours) ~ N(0, hours).
            let z1 = rng.normal();
            let t1_drift_frac = self.t1_volatility * z1 * hours.sqrt();
            let new_t1 = (self.base.t1_us[i] * (1.0 + t1_drift_frac)).max(1.0);
            t1_us.push(new_t1);

            // --- T2 drift: correlated with T1 + independent noise ---
            // T2 <= 2*T1 always. Model: T2/T1 ratio drifts independently.
            let z2 = rng.normal();
            let t2_drift_frac = self.t1_volatility * 0.7 * z1 // correlated with T1
                + self.t1_volatility * 0.3 * z2; // independent noise
            let new_t2_raw = self.base.t2_us[i] * (1.0 + t2_drift_frac);
            let new_t2 = new_t2_raw.max(0.5).min(2.0 * new_t1); // enforce T2 <= 2*T1
            t2_us.push(new_t2);

            // --- Frequency drift: Poisson TLS jumps ---
            // Expected number of jumps in `hours` at rate lambda.
            let expected_jumps = self.freq_jump_rate_per_hr * hours;
            let num_jumps = rng.poisson(expected_jumps);
            // Each jump is random direction with fixed amplitude.
            let mut freq_shift = 0.0;
            for _ in 0..num_jumps {
                let direction = if rng.uniform() > 0.5 { 1.0 } else { -1.0 };
                freq_shift += direction * self.freq_jump_ghz;
            }
            frequencies.push(self.base.qubit_frequencies_ghz[i] + freq_shift);

            // --- Gate fidelity aging: sinusoidal ---
            // Models slow drift of optimal gate parameters (e.g., from flux bias drift).
            let phase = 2.0 * PI * hours / self.fidelity_aging_period_hr + (i as f64) * 0.5;
            let aging_penalty = self.fidelity_aging_amplitude * (1.0 - phase.cos()) / 2.0;
            let new_fid = (self.base.single_gate_fidelities[i] - aging_penalty).max(0.5);
            single_fid.push(new_fid);

            // Readout fidelity: minor degradation proportional to T1 change.
            let t1_ratio = new_t1 / self.base.t1_us[i];
            // If T1 drops, readout fidelity drops (measurement takes longer relative to T1).
            let readout_correction = if t1_ratio < 1.0 {
                (1.0 - t1_ratio) * 0.01 // 1% readout degradation per 100% T1 drop
            } else {
                0.0
            };
            readout_fid[i] = (self.base.readout_fidelities[i] - readout_correction).max(0.5);
        }

        // Two-qubit fidelities also drift (correlated with single-qubit aging).
        let mut two_q_fid = self.base.two_qubit_fidelities.clone();
        for ((_a, _b), fid) in two_q_fid.iter_mut() {
            let phase = 2.0 * PI * hours / self.fidelity_aging_period_hr;
            let penalty = self.fidelity_aging_amplitude * 1.5 * (1.0 - phase.cos()) / 2.0;
            *fid = (*fid - penalty).max(0.5);
        }

        CalibrationData {
            qubit_frequencies_ghz: frequencies,
            t1_us,
            t2_us,
            readout_fidelities: readout_fid,
            single_gate_fidelities: single_fid,
            two_qubit_fidelities: two_q_fid,
            coupling_map: self.base.coupling_map.clone(),
            native_gate_family: self.base.native_gate_family.clone(),
            device_name: format!("{}_t+{:.1}h", self.base.device_name, hours),
        }
    }

    /// Generate a stability report over a given duration.
    ///
    /// Samples the drift model at regular intervals and computes
    /// aggregate statistics for T1/T2 variation, frequency jumps,
    /// and fidelity range.
    pub fn stability_report(&self, duration_hours: f64) -> StabilityReport {
        let num_samples = 20;
        let dt = duration_hours / num_samples as f64;
        let n = self.base.num_qubits();

        let mut all_t1: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut all_t2: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut all_fid: Vec<f64> = Vec::new();
        let mut total_jumps = 0u32;

        for step in 0..=num_samples {
            let t = step as f64 * dt;
            let snapshot = self.at_time(t);

            for i in 0..n {
                all_t1[i].push(snapshot.t1_us[i]);
                all_t2[i].push(snapshot.t2_us[i]);
            }

            all_fid.push(snapshot.avg_single_gate_fidelity());

            // Count frequency jumps by comparing to base.
            for i in 0..n {
                let delta = (snapshot.qubit_frequencies_ghz[i]
                    - self.base.qubit_frequencies_ghz[i])
                    .abs();
                if delta > self.freq_jump_ghz * 0.5 {
                    total_jumps += 1;
                }
            }
        }

        // Compute T1 variation: max coefficient of variation across qubits.
        let t1_variation_pct = compute_max_cv_pct(&all_t1);
        let t2_variation_pct = compute_max_cv_pct(&all_t2);

        let fid_min = all_fid.iter().cloned().fold(f64::INFINITY, f64::min);
        let fid_max = all_fid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Recommended recalibration interval: when fidelity drops > 0.1%.
        let mut recal_hours = duration_hours;
        for step in 1..=num_samples {
            let t = step as f64 * dt;
            let snapshot = self.at_time(t);
            let fid_drop = self.base.avg_single_gate_fidelity() - snapshot.avg_single_gate_fidelity();
            if fid_drop > 0.001 {
                recal_hours = t;
                break;
            }
        }

        StabilityReport {
            t1_variation_pct,
            t2_variation_pct,
            frequency_jumps: total_jumps,
            fidelity_range: (fid_min, fid_max),
            recommended_recal_interval_hours: recal_hours,
        }
    }
}

/// Compute the maximum coefficient of variation (in %) across multiple time series.
fn compute_max_cv_pct(series: &[Vec<f64>]) -> f64 {
    let mut max_cv = 0.0f64;
    for s in series {
        if s.is_empty() {
            continue;
        }
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        if mean.abs() < 1e-12 {
            continue;
        }
        let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / s.len() as f64;
        let cv = var.sqrt() / mean;
        if cv > max_cv {
            max_cv = cv;
        }
    }
    max_cv * 100.0
}

// ===================================================================
// STABILITY REPORT
// ===================================================================

/// Aggregate stability statistics from a calibration drift analysis.
#[derive(Debug, Clone)]
pub struct StabilityReport {
    /// Maximum T1 coefficient of variation across qubits (%).
    pub t1_variation_pct: f64,
    /// Maximum T2 coefficient of variation across qubits (%).
    pub t2_variation_pct: f64,
    /// Total number of frequency jumps observed across all qubits.
    pub frequency_jumps: u32,
    /// Range of average single-qubit gate fidelity: (min, max).
    pub fidelity_range: (f64, f64),
    /// Recommended recalibration interval in hours.
    pub recommended_recal_interval_hours: f64,
}

impl std::fmt::Display for StabilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Stability Report ===")?;
        writeln!(f, "  T1 variation:       {:.2}%", self.t1_variation_pct)?;
        writeln!(f, "  T2 variation:       {:.2}%", self.t2_variation_pct)?;
        writeln!(f, "  Frequency jumps:    {}", self.frequency_jumps)?;
        writeln!(
            f,
            "  Fidelity range:     [{:.6}, {:.6}]",
            self.fidelity_range.0, self.fidelity_range.1
        )?;
        writeln!(
            f,
            "  Recal interval:     {:.1} hours",
            self.recommended_recal_interval_hours
        )?;
        Ok(())
    }
}

// ===================================================================
// COMPILER BENCHMARK
// ===================================================================

/// Compilation overhead result for a single (circuit, gate family) pair.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Name of the benchmark circuit.
    pub circuit_name: String,
    /// Native gate family used for compilation.
    pub native_gate_family: String,
    /// Number of single-qubit native operations after compilation.
    pub num_1q: usize,
    /// Number of two-qubit native operations after compilation.
    pub num_2q: usize,
    /// Circuit depth in native operations.
    pub depth: usize,
    /// Overhead ratio: (compiled 2Q gates) / (abstract 2Q gates).
    pub overhead_ratio: f64,
}

impl std::fmt::Display for CompilationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<12} | {:<12} | 1Q: {:>3} | 2Q: {:>3} | depth: {:>3} | overhead: {:.2}x",
            self.circuit_name,
            self.native_gate_family,
            self.num_1q,
            self.num_2q,
            self.depth,
            self.overhead_ratio
        )
    }
}

/// Benchmark native-gate compilation overhead across gate families.
///
/// Compiles standard circuits (Bell, GHZ, QFT) to each native gate set
/// (ECR, SqrtISWAP, CZ) and measures the resulting gate count, depth,
/// and overhead ratio.
pub struct CompilerBenchmark;

impl CompilerBenchmark {
    /// Run compilation benchmarks for all standard circuits on all gate families.
    pub fn benchmark_all(num_qubits: usize) -> Vec<CompilationResult> {
        let families = [
            ("ECR", NativeGateFamily::ECR),
            ("SqrtISWAP", NativeGateFamily::SqrtISWAP),
            ("CZ", NativeGateFamily::CZ),
        ];

        let circuits = Self::standard_circuits(num_qubits);
        let mut results = Vec::new();

        for (family_name, family) in &families {
            // Build a minimal processor with this gate family for compilation.
            let proc = build_preset_processor(
                num_qubits,
                5.0,
                0.1,
                5,
                -300.0,
                100.0,
                80.0,
                0.999,
                0.98,
                0.01,
                25.0,
                *family,
                0.99,
                200.0,
                3.5,
                false,
            );

            for (circuit_name, abstract_circuit, abstract_2q_count) in &circuits {
                let mut total_1q = 0usize;
                let mut total_2q = 0usize;
                let mut max_qubit_depth: HashMap<usize, usize> = HashMap::new();

                for gate in abstract_circuit {
                    let native_ops = compile_to_native(gate, &proc);
                    for op in &native_ops {
                        match op {
                            NativeGate::Rz { qubit, .. }
                            | NativeGate::SX { qubit }
                            | NativeGate::X { qubit } => {
                                total_1q += 1;
                                *max_qubit_depth.entry(*qubit).or_insert(0) += 1;
                            }
                            NativeGate::ECR { qubit_a, qubit_b }
                            | NativeGate::SqrtISWAP { qubit_a, qubit_b }
                            | NativeGate::CZGate { qubit_a, qubit_b } => {
                                total_2q += 1;
                                *max_qubit_depth.entry(*qubit_a).or_insert(0) += 1;
                                *max_qubit_depth.entry(*qubit_b).or_insert(0) += 1;
                            }
                            NativeGate::FSim {
                                qubit_a, qubit_b, ..
                            } => {
                                total_2q += 1;
                                *max_qubit_depth.entry(*qubit_a).or_insert(0) += 1;
                                *max_qubit_depth.entry(*qubit_b).or_insert(0) += 1;
                            }
                            NativeGate::Measure { qubit } | NativeGate::Reset { qubit } => {
                                *max_qubit_depth.entry(*qubit).or_insert(0) += 1;
                            }
                        }
                    }
                }

                let depth = max_qubit_depth.values().max().copied().unwrap_or(0);
                let overhead = if *abstract_2q_count > 0 {
                    total_2q as f64 / *abstract_2q_count as f64
                } else {
                    1.0
                };

                results.push(CompilationResult {
                    circuit_name: circuit_name.to_string(),
                    native_gate_family: family_name.to_string(),
                    num_1q: total_1q,
                    num_2q: total_2q,
                    depth,
                    overhead_ratio: overhead,
                });
            }
        }

        results
    }

    /// Generate standard benchmark circuits.
    ///
    /// Returns (name, gates, abstract_2q_count) tuples.
    fn standard_circuits(n: usize) -> Vec<(String, Vec<Gate>, usize)> {
        let mut circuits = Vec::new();

        // 1. Bell state: H(0), CNOT(0,1)
        if n >= 2 {
            let bell = vec![
                Gate::single(GateType::H, 0),
                Gate::two(GateType::CNOT, 0, 1),
            ];
            circuits.push(("Bell".to_string(), bell, 1));
        }

        // 2. GHZ state: H(0), CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1)
        if n >= 2 {
            let mut ghz = vec![Gate::single(GateType::H, 0)];
            for i in 0..(n - 1) {
                ghz.push(Gate::two(GateType::CNOT, i, i + 1));
            }
            circuits.push(("GHZ".to_string(), ghz, n - 1));
        }

        // 3. QFT: H, controlled-phase rotations, SWAPs
        if n >= 2 {
            let mut qft = Vec::new();
            let qft_size = n.min(6); // cap at 6 to keep benchmarks fast
            for i in 0..qft_size {
                qft.push(Gate::single(GateType::H, i));
                for j in (i + 1)..qft_size {
                    let angle = PI / (1 << (j - i)) as f64;
                    // Controlled-Phase as: CNOT + Rz dressing
                    // Abstract: 1 CNOT per controlled rotation
                    qft.push(Gate::two(GateType::CNOT, j, i));
                    qft.push(Gate::single(GateType::Rz(angle), i));
                    qft.push(Gate::two(GateType::CNOT, j, i));
                }
            }
            let num_2q = qft
                .iter()
                .filter(|g| {
                    matches!(g.gate_type, GateType::CNOT | GateType::CZ | GateType::SWAP)
                })
                .count();
            circuits.push(("QFT".to_string(), qft, num_2q));
        }

        circuits
    }
}

// ===================================================================
// SIMPLE XORSHIFT PRNG (deterministic, no external dependency)
// ===================================================================

/// Minimal xorshift64* PRNG for reproducible drift trajectories.
///
/// Avoids pulling in `rand` for this module -- the drift model needs
/// determinism more than statistical quality.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state.
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed },
        }
    }

    /// Generate a uniform f64 in [0, 1).
    fn uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Generate an approximate standard normal using Box-Muller transform.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15); // avoid log(0)
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Generate a Poisson-distributed count with given mean (lambda).
    ///
    /// Uses Knuth's algorithm for small lambda, adequate for our drift
    /// model where lambda is typically < 50.
    fn poisson(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }
        let l = (-lambda).exp();
        let mut k = 0u32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= self.uniform();
            if p <= l {
                break;
            }
        }
        k.saturating_sub(1)
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // 1. CalibrationData construction and validation
    // ---------------------------------------------------------------

    fn sample_calibration(n: usize) -> CalibrationData {
        let mut two_q_fid = HashMap::new();
        let mut coupling_map = Vec::new();
        for i in 0..(n - 1) {
            coupling_map.push((i, i + 1));
            two_q_fid.insert((i, i + 1), 0.99);
        }

        CalibrationData {
            qubit_frequencies_ghz: vec![5.0; n],
            t1_us: vec![100.0; n],
            t2_us: vec![80.0; n],
            readout_fidelities: vec![0.98; n],
            single_gate_fidelities: vec![0.9995; n],
            two_qubit_fidelities: two_q_fid,
            coupling_map,
            native_gate_family: "ecr".to_string(),
            device_name: "test_device".to_string(),
        }
    }

    #[test]
    fn test_calibration_num_qubits() {
        let cal = sample_calibration(10);
        assert_eq!(cal.num_qubits(), 10);
    }

    #[test]
    fn test_calibration_validation_passes() {
        let cal = sample_calibration(5);
        assert!(cal.validate().is_ok());
    }

    #[test]
    fn test_calibration_validation_t2_exceeds_2t1() {
        let mut cal = sample_calibration(3);
        cal.t2_us[1] = 300.0; // T2 > 2*T1 = 200
        assert!(cal.validate().is_err());
    }

    #[test]
    fn test_calibration_validation_bad_readout() {
        let mut cal = sample_calibration(3);
        cal.readout_fidelities[0] = 1.5; // out of [0, 1]
        assert!(cal.validate().is_err());
    }

    #[test]
    fn test_calibration_avg_fidelities() {
        let cal = sample_calibration(4);
        assert!((cal.avg_single_gate_fidelity() - 0.9995).abs() < 1e-8);
        assert!((cal.avg_two_qubit_fidelity() - 0.99).abs() < 1e-8);
    }

    // ---------------------------------------------------------------
    // 2. CalibrationData round-trip through from_processor
    // ---------------------------------------------------------------

    #[test]
    fn test_calibration_round_trip() {
        let cal_orig = sample_calibration(5);
        let twin = DigitalTwin::from_calibration(&cal_orig).unwrap();
        let cal_back = CalibrationData::from_processor(&twin.processor, "round_trip");

        assert_eq!(cal_back.num_qubits(), cal_orig.num_qubits());
        for i in 0..cal_orig.num_qubits() {
            assert!(
                (cal_back.qubit_frequencies_ghz[i] - cal_orig.qubit_frequencies_ghz[i]).abs()
                    < 1e-6,
                "Frequency mismatch at qubit {}",
                i
            );
            assert!(
                (cal_back.t1_us[i] - cal_orig.t1_us[i]).abs() < 1e-6,
                "T1 mismatch at qubit {}",
                i
            );
            assert!(
                (cal_back.t2_us[i] - cal_orig.t2_us[i]).abs() < 1e-6,
                "T2 mismatch at qubit {}",
                i
            );
        }
    }

    // ---------------------------------------------------------------
    // 3. Digital twin preset construction
    // ---------------------------------------------------------------

    #[test]
    fn test_ibm_eagle_preset() {
        let twin = DigitalTwin::ibm_eagle(20);
        assert_eq!(twin.processor.num_qubits(), 20);
        assert_eq!(twin.processor.native_2q_gate, NativeGateFamily::ECR);
        assert!(twin.processor.two_qubit_fidelity > 0.98);
        assert_eq!(twin.calibration.device_name, "ibm_eagle");
        // Eagle: T1 ~ 120 us
        assert!((twin.processor.qubits[0].t1_us - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_ibm_heron_preset() {
        let twin = DigitalTwin::ibm_heron(20);
        assert_eq!(twin.processor.num_qubits(), 20);
        assert_eq!(twin.processor.native_2q_gate, NativeGateFamily::ECR);
        assert!(twin.processor.two_qubit_fidelity > 0.99);
        assert_eq!(twin.calibration.device_name, "ibm_heron");
        // Heron has better coherence than Eagle.
        assert!(twin.processor.qubits[0].t1_us > 200.0);
    }

    #[test]
    fn test_google_sycamore_preset() {
        let twin = DigitalTwin::google_sycamore(20);
        assert_eq!(twin.processor.num_qubits(), 20);
        assert_eq!(twin.processor.native_2q_gate, NativeGateFamily::SqrtISWAP);
        assert_eq!(twin.calibration.device_name, "google_sycamore");
        // Sycamore: T1 ~ 16 us
        assert!((twin.processor.qubits[0].t1_us - 16.0).abs() < 1.0);
    }

    #[test]
    fn test_google_willow_preset() {
        let twin = DigitalTwin::google_willow(20);
        assert_eq!(twin.processor.num_qubits(), 20);
        assert_eq!(twin.processor.native_2q_gate, NativeGateFamily::SqrtISWAP);
        assert_eq!(twin.calibration.device_name, "google_willow");
        // Willow: T1 ~ 68 us (much better than Sycamore)
        assert!(twin.processor.qubits[0].t1_us > 50.0);
    }

    // ---------------------------------------------------------------
    // 4. Validation report
    // ---------------------------------------------------------------

    #[test]
    fn test_validation_report_reasonable() {
        let twin = DigitalTwin::ibm_heron(10);
        let report = twin.validate();

        // Bell fidelity should be high for Heron.
        assert!(
            report.bell_fidelity > 0.95,
            "Bell fidelity too low: {}",
            report.bell_fidelity
        );
        assert!(
            report.bell_fidelity <= 1.0,
            "Bell fidelity > 1: {}",
            report.bell_fidelity
        );

        // GHZ fidelity should be lower than Bell (more gates).
        assert!(report.ghz_fidelity <= report.bell_fidelity);
        assert!(report.ghz_fidelity > 0.5);

        // QV should be at least 1.
        assert!(report.quantum_volume >= 1);

        // Deep fidelity < shallow fidelity.
        assert!(report.predicted_deep_fidelity < report.predicted_shallow_fidelity);
    }

    #[test]
    fn test_compare_presets() {
        let results = DigitalTwin::compare_presets(10);
        assert_eq!(results.len(), 4);

        let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"ibm_eagle"));
        assert!(names.contains(&"ibm_heron"));
        assert!(names.contains(&"google_sycamore"));
        assert!(names.contains(&"google_willow"));

        // Heron should have better Bell fidelity than Eagle (better coherence).
        let eagle_bell = results
            .iter()
            .find(|(n, _)| n == "ibm_eagle")
            .unwrap()
            .1
            .bell_fidelity;
        let heron_bell = results
            .iter()
            .find(|(n, _)| n == "ibm_heron")
            .unwrap()
            .1
            .bell_fidelity;
        assert!(
            heron_bell > eagle_bell,
            "Heron ({}) should beat Eagle ({})",
            heron_bell,
            eagle_bell
        );
    }

    // ---------------------------------------------------------------
    // 5. Fidelity prediction
    // ---------------------------------------------------------------

    #[test]
    fn test_predict_fidelity_monotonic() {
        let twin = DigitalTwin::ibm_heron(10);
        let f_shallow = twin.predict_circuit_fidelity(10, 5, 5);
        let f_deep = twin.predict_circuit_fidelity(100, 50, 50);

        assert!(f_shallow > f_deep, "Shallow {} should exceed deep {}", f_shallow, f_deep);
        assert!(f_shallow > 0.0 && f_shallow <= 1.0);
        assert!(f_deep >= 0.0 && f_deep <= 1.0);
    }

    #[test]
    fn test_predict_fidelity_zero_gates() {
        let twin = DigitalTwin::ibm_heron(10);
        let f = twin.predict_circuit_fidelity(0, 0, 0);
        assert!((f - 1.0).abs() < 1e-10, "Zero gates should give fidelity 1.0, got {}", f);
    }

    // ---------------------------------------------------------------
    // 6. Calibration drift model
    // ---------------------------------------------------------------

    #[test]
    fn test_drift_at_time_zero_matches_base() {
        let cal = sample_calibration(5);
        let drift = CalibrationDriftModel::new(cal.clone(), 42);
        let drifted = drift.at_time(0.0);

        for i in 0..cal.num_qubits() {
            // At t=0, Wiener process has zero variance, so T1 should match.
            // Poisson with lambda=0 gives 0 jumps, so frequency should match.
            assert!(
                (drifted.qubit_frequencies_ghz[i] - cal.qubit_frequencies_ghz[i]).abs() < 1e-10,
                "Frequency drift at t=0: {} vs {}",
                drifted.qubit_frequencies_ghz[i],
                cal.qubit_frequencies_ghz[i]
            );
            // T1 drift factor: volatility * z * sqrt(0) = 0
            assert!(
                (drifted.t1_us[i] - cal.t1_us[i]).abs() < 1e-10,
                "T1 drift at t=0: {} vs {}",
                drifted.t1_us[i],
                cal.t1_us[i]
            );
        }
    }

    #[test]
    fn test_drift_changes_t1_over_time() {
        let cal = sample_calibration(10);
        let drift = CalibrationDriftModel::new(cal.clone(), 12345);
        let drifted = drift.at_time(24.0); // 24 hours of drift

        // At least one qubit should have a different T1.
        let any_changed = (0..cal.num_qubits()).any(|i| {
            (drifted.t1_us[i] - cal.t1_us[i]).abs() > 0.01
        });
        assert!(
            any_changed,
            "After 24 hours, at least one T1 should have drifted"
        );
    }

    #[test]
    fn test_drift_t2_bounded_by_2t1() {
        let cal = sample_calibration(10);
        let drift = CalibrationDriftModel::new(cal, 99);

        for hours in [1.0, 6.0, 12.0, 24.0, 48.0] {
            let drifted = drift.at_time(hours);
            for i in 0..drifted.num_qubits() {
                assert!(
                    drifted.t2_us[i] <= 2.0 * drifted.t1_us[i] + 1e-10,
                    "T2={} > 2*T1={} at qubit {} after {}h",
                    drifted.t2_us[i],
                    drifted.t1_us[i],
                    i,
                    hours
                );
            }
        }
    }

    #[test]
    fn test_drift_frequency_jumps() {
        let cal = sample_calibration(20);
        let drift = CalibrationDriftModel::new(cal.clone(), 7777);
        let drifted = drift.at_time(48.0); // 48 hours, expect some jumps

        // With 20 qubits, rate 0.5/qubit/hr, in 48 hrs: expected ~480 jumps.
        // At least some should be visible.
        let total_shift: f64 = (0..cal.num_qubits())
            .map(|i| {
                (drifted.qubit_frequencies_ghz[i] - cal.qubit_frequencies_ghz[i]).abs()
            })
            .sum();
        assert!(
            total_shift > 0.0,
            "Expected frequency shifts after 48 hours"
        );
    }

    #[test]
    fn test_drift_deterministic() {
        let cal = sample_calibration(5);
        let drift1 = CalibrationDriftModel::new(cal.clone(), 42);
        let drift2 = CalibrationDriftModel::new(cal, 42);

        let d1 = drift1.at_time(10.0);
        let d2 = drift2.at_time(10.0);

        for i in 0..d1.num_qubits() {
            assert!(
                (d1.t1_us[i] - d2.t1_us[i]).abs() < 1e-12,
                "Same seed should produce identical drift"
            );
            assert!(
                (d1.qubit_frequencies_ghz[i] - d2.qubit_frequencies_ghz[i]).abs() < 1e-12,
                "Same seed should produce identical frequency"
            );
        }
    }

    #[test]
    fn test_stability_report() {
        let cal = sample_calibration(5);
        let drift = CalibrationDriftModel::new(cal, 42);
        let report = drift.stability_report(24.0);

        assert!(report.t1_variation_pct >= 0.0);
        assert!(report.t2_variation_pct >= 0.0);
        assert!(report.fidelity_range.0 <= report.fidelity_range.1);
        assert!(report.recommended_recal_interval_hours > 0.0);
        assert!(report.recommended_recal_interval_hours <= 24.0);
    }

    // ---------------------------------------------------------------
    // 7. JSON parsing (requires serde feature)
    // ---------------------------------------------------------------

    #[cfg(feature = "serde")]
    #[test]
    fn test_ibm_json_parsing() {
        let json = r#"{
            "backend_name": "ibm_test",
            "qubits": [
                [{"name": "T1", "value": 120.5, "unit": "us"}, {"name": "T2", "value": 80.0, "unit": "us"}, {"name": "frequency", "value": 5000000000.0, "unit": "Hz"}, {"name": "readout_assignment_error", "value": 0.02}],
                [{"name": "T1", "value": 100.0, "unit": "us"}, {"name": "T2", "value": 70.0, "unit": "us"}, {"name": "frequency", "value": 5100000000.0, "unit": "Hz"}, {"name": "readout_assignment_error", "value": 0.03}]
            ],
            "gates": [
                {"gate": "sx", "qubits": [0], "parameters": [{"name": "gate_error", "value": 0.0003}]},
                {"gate": "sx", "qubits": [1], "parameters": [{"name": "gate_error", "value": 0.0004}]},
                {"gate": "ecr", "qubits": [0, 1], "parameters": [{"name": "gate_error", "value": 0.008}]}
            ]
        }"#;

        let cal = CalibrationData::from_ibm_json(json).unwrap();
        assert_eq!(cal.num_qubits(), 2);
        assert_eq!(cal.device_name, "ibm_test");
        assert!((cal.t1_us[0] - 120.5).abs() < 1e-6);
        assert!((cal.t1_us[1] - 100.0).abs() < 1e-6);
        assert!((cal.qubit_frequencies_ghz[0] - 5.0).abs() < 0.001);
        assert!((cal.qubit_frequencies_ghz[1] - 5.1).abs() < 0.001);
        assert!((cal.readout_fidelities[0] - 0.98).abs() < 1e-6);
        assert!((cal.single_gate_fidelities[0] - 0.9997).abs() < 1e-6);
        assert_eq!(cal.coupling_map.len(), 1);
        assert!(cal.two_qubit_fidelities.contains_key(&(0, 1)));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_google_json_parsing() {
        let json = r#"{
            "device_name": "google_test",
            "qubits": [
                {"id": "0_0", "frequency_ghz": 5.5, "t1_us": 16.0, "t2_us": 12.0, "readout_fidelity": 0.965, "single_gate_fidelity": 0.9985},
                {"id": "0_1", "frequency_ghz": 5.6, "t1_us": 18.0, "t2_us": 14.0, "readout_fidelity": 0.970, "single_gate_fidelity": 0.999}
            ],
            "couplers": [
                {"qubit_a": 0, "qubit_b": 1, "two_qubit_fidelity": 0.995}
            ],
            "native_gate": "sqrt_iswap"
        }"#;

        let cal = CalibrationData::from_google_json(json).unwrap();
        assert_eq!(cal.num_qubits(), 2);
        assert_eq!(cal.device_name, "google_test");
        assert!((cal.qubit_frequencies_ghz[0] - 5.5).abs() < 1e-6);
        assert!((cal.t1_us[0] - 16.0).abs() < 1e-6);
        assert_eq!(cal.native_gate_family, "sqrt_iswap");
        assert!(cal.two_qubit_fidelities.contains_key(&(0, 1)));
    }

    // ---------------------------------------------------------------
    // 8. Compiler benchmark
    // ---------------------------------------------------------------

    #[test]
    fn test_compiler_benchmark_runs() {
        let results = CompilerBenchmark::benchmark_all(4);
        // 3 gate families * 3 circuits = 9 results
        assert_eq!(results.len(), 9);

        for result in &results {
            assert!(result.num_1q > 0 || result.num_2q > 0);
            assert!(result.depth > 0);
            assert!(result.overhead_ratio > 0.0);
        }
    }

    #[test]
    fn test_ecr_fewer_2q_gates_than_sqrt_iswap_for_cnot() {
        let results = CompilerBenchmark::benchmark_all(4);

        // For Bell circuit (1 abstract CNOT), ECR should need 1 native 2Q gate
        // while SqrtISWAP needs 2 native 2Q gates.
        let ecr_bell = results
            .iter()
            .find(|r| r.circuit_name == "Bell" && r.native_gate_family == "ECR")
            .unwrap();
        let sqrt_iswap_bell = results
            .iter()
            .find(|r| r.circuit_name == "Bell" && r.native_gate_family == "SqrtISWAP")
            .unwrap();

        assert!(
            ecr_bell.num_2q <= sqrt_iswap_bell.num_2q,
            "ECR ({} 2Q gates) should use <= SqrtISWAP ({} 2Q gates) for Bell",
            ecr_bell.num_2q,
            sqrt_iswap_bell.num_2q
        );
    }

    #[test]
    fn test_compiler_overhead_ratios() {
        let results = CompilerBenchmark::benchmark_all(4);

        for result in &results {
            // Overhead should be >= 1.0 (can't do better than abstract).
            assert!(
                result.overhead_ratio >= 1.0 - 1e-10,
                "{} on {}: overhead {} < 1.0",
                result.circuit_name,
                result.native_gate_family,
                result.overhead_ratio
            );
        }
    }

    // ---------------------------------------------------------------
    // 9. PRNG determinism
    // ---------------------------------------------------------------

    #[test]
    fn test_xorshift_determinism() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.uniform().to_bits(), rng2.uniform().to_bits());
        }
    }

    #[test]
    fn test_xorshift_normal_distribution() {
        let mut rng = Xorshift64::new(12345);
        let samples: Vec<f64> = (0..10000).map(|_| rng.normal()).collect();

        // Mean should be near 0.
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            mean.abs() < 0.1,
            "Normal mean {} too far from 0",
            mean
        );

        // Std dev should be near 1.
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 0.15,
            "Normal std {} too far from 1",
            std
        );
    }

    #[test]
    fn test_xorshift_poisson() {
        let mut rng = Xorshift64::new(54321);
        let lambda = 5.0;
        let samples: Vec<u32> = (0..10000).map(|_| rng.poisson(lambda)).collect();

        let mean = samples.iter().sum::<u32>() as f64 / samples.len() as f64;
        assert!(
            (mean - lambda).abs() < 0.5,
            "Poisson mean {} too far from lambda {}",
            mean,
            lambda
        );
    }
}
