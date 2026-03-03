//! Live hardware calibration data fetching and noise model construction.
//!
//! Provides a unified interface for fetching real-time calibration data from
//! quantum hardware providers (IBM, IonQ, Rigetti, Google) and converting it
//! into simulation-ready noise models. Includes an LRU cache with TTL to
//! avoid excessive API calls.
//!
//! # Architecture
//!
//! ```text
//! LiveCalibrationProvider (trait)
//!     |
//!     +-- IbmCalibrationProvider     (IBM Quantum backend properties)
//!     +-- IonqCalibrationProvider     (IonQ characterization data)
//!     +-- RigettiCalibrationProvider  (Rigetti ISA calibration)
//!     +-- GoogleCalibrationProvider   (Google device metrics)
//!     |
//!     v
//! CalibrationData  --->  NoiseModelBuilder  --->  SimulationNoiseModel
//!     |
//! CalibrationCache (LRU + TTL)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::live_calibration::*;
//!
//! // Parse IBM calibration JSON
//! let provider = IbmCalibrationProvider::new("YOUR_API_TOKEN");
//! let cal = provider.parse_calibration_json(ibm_json)?;
//!
//! // Build a realistic noise model
//! let noise = NoiseModelBuilder::build_realistic(&cal)?;
//!
//! // Or use the cache
//! let mut cache = CalibrationCache::new(Duration::from_secs(300), 16);
//! let cal = cache.get_or_fetch(&provider, "ibm_brisbane")?;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors from live calibration fetching and parsing.
#[derive(Debug, Clone)]
pub enum LiveCalibrationError {
    /// HTTP or network error when fetching calibration data.
    FetchError(String),
    /// JSON parsing failed.
    ParseError(String),
    /// Calibration data violates physical constraints.
    ValidationError(String),
    /// Requested device not found.
    DeviceNotFound(String),
    /// Provider API returned an error.
    ProviderError(String),
    /// Cache-related error.
    CacheError(String),
}

impl fmt::Display for LiveCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiveCalibrationError::FetchError(msg) => write!(f, "Fetch error: {}", msg),
            LiveCalibrationError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LiveCalibrationError::ValidationError(msg) => {
                write!(f, "Validation error: {}", msg)
            }
            LiveCalibrationError::DeviceNotFound(msg) => {
                write!(f, "Device not found: {}", msg)
            }
            LiveCalibrationError::ProviderError(msg) => {
                write!(f, "Provider error: {}", msg)
            }
            LiveCalibrationError::CacheError(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for LiveCalibrationError {}

// ============================================================
// CALIBRATION DATA STRUCTURES
// ============================================================

/// Per-qubit calibration properties from hardware.
#[derive(Debug, Clone)]
pub struct QubitCalibration {
    /// Qubit index on the device.
    pub qubit_index: usize,
    /// T1 relaxation time in microseconds.
    pub t1_us: f64,
    /// T2 dephasing time in microseconds.
    pub t2_us: f64,
    /// Qubit drive frequency in GHz.
    pub frequency_ghz: f64,
    /// Readout assignment error (probability of misclassification).
    pub readout_error: f64,
    /// Anharmonicity in GHz (negative for transmons, 0 for trapped ions).
    pub anharmonicity_ghz: f64,
}

/// Per-gate calibration properties.
#[derive(Debug, Clone)]
pub struct GateCalibration {
    /// Gate type name (e.g., "cx", "sx", "syc", "ms", "cz").
    pub gate_type: String,
    /// Qubit indices this gate acts on.
    pub qubits: Vec<usize>,
    /// Gate error rate from randomized benchmarking.
    pub error_rate: f64,
    /// Gate execution time in nanoseconds.
    pub gate_time_ns: f64,
}

/// Crosstalk coefficient between a pair of qubits.
#[derive(Debug, Clone)]
pub struct CrosstalkEntry {
    /// Source qubit (the one being driven).
    pub source: usize,
    /// Target qubit (the one experiencing spurious coupling).
    pub target: usize,
    /// Crosstalk coefficient (dimensionless, typically 0.0001 to 0.01).
    pub coefficient: f64,
}

/// Complete calibration snapshot from a quantum device.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Device name (e.g., "ibm_brisbane", "ionq_aria").
    pub device_name: String,
    /// Number of qubits on the device.
    pub num_qubits: usize,
    /// Per-qubit calibration data, indexed by qubit index.
    pub qubit_calibrations: Vec<QubitCalibration>,
    /// Per-gate calibration data for all available gates.
    pub gate_calibrations: Vec<GateCalibration>,
    /// Crosstalk coefficients between qubit pairs.
    pub crosstalk_entries: Vec<CrosstalkEntry>,
    /// Unix timestamp (seconds) when this calibration was measured.
    pub timestamp_secs: u64,
    /// Provider-specific metadata (e.g., processor version, firmware).
    pub metadata: HashMap<String, String>,
}

impl CalibrationData {
    /// Look up the gate error rate for a specific gate type on specific qubits.
    ///
    /// Returns `None` if no matching gate calibration exists.
    pub fn gate_error(&self, gate_type: &str, qubits: &[usize]) -> Option<f64> {
        self.gate_calibrations
            .iter()
            .find(|g| g.gate_type == gate_type && g.qubits == qubits)
            .map(|g| g.error_rate)
    }

    /// Look up gate time in nanoseconds for a specific gate on specific qubits.
    pub fn gate_time_ns(&self, gate_type: &str, qubits: &[usize]) -> Option<f64> {
        self.gate_calibrations
            .iter()
            .find(|g| g.gate_type == gate_type && g.qubits == qubits)
            .map(|g| g.gate_time_ns)
    }

    /// Average single-qubit gate error across the device.
    pub fn avg_single_qubit_error(&self) -> f64 {
        let sq_gates: Vec<f64> = self
            .gate_calibrations
            .iter()
            .filter(|g| g.qubits.len() == 1)
            .map(|g| g.error_rate)
            .collect();
        if sq_gates.is_empty() {
            return 0.0;
        }
        sq_gates.iter().sum::<f64>() / sq_gates.len() as f64
    }

    /// Average two-qubit gate error across the device.
    pub fn avg_two_qubit_error(&self) -> f64 {
        let tq_gates: Vec<f64> = self
            .gate_calibrations
            .iter()
            .filter(|g| g.qubits.len() == 2)
            .map(|g| g.error_rate)
            .collect();
        if tq_gates.is_empty() {
            return 0.0;
        }
        tq_gates.iter().sum::<f64>() / tq_gates.len() as f64
    }

    /// Average readout error across all qubits.
    pub fn avg_readout_error(&self) -> f64 {
        if self.qubit_calibrations.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.qubit_calibrations.iter().map(|q| q.readout_error).sum();
        sum / self.qubit_calibrations.len() as f64
    }

    /// Build the crosstalk matrix as a dense 2D array.
    ///
    /// Returns an `n x n` matrix where `matrix[i][j]` is the crosstalk
    /// coefficient from qubit `i` to qubit `j`. Diagonal is 0.
    pub fn crosstalk_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.num_qubits;
        let mut matrix = vec![vec![0.0; n]; n];
        for entry in &self.crosstalk_entries {
            if entry.source < n && entry.target < n {
                matrix[entry.source][entry.target] = entry.coefficient;
            }
        }
        matrix
    }

    /// Validate that calibration data obeys physical constraints.
    pub fn validate(&self) -> Result<(), LiveCalibrationError> {
        if self.num_qubits == 0 {
            return Err(LiveCalibrationError::ValidationError(
                "Device must have at least 1 qubit".to_string(),
            ));
        }

        for qc in &self.qubit_calibrations {
            if qc.t1_us < 0.0 {
                return Err(LiveCalibrationError::ValidationError(format!(
                    "Qubit {} has negative T1: {} us",
                    qc.qubit_index, qc.t1_us
                )));
            }
            if qc.t2_us < 0.0 {
                return Err(LiveCalibrationError::ValidationError(format!(
                    "Qubit {} has negative T2: {} us",
                    qc.qubit_index, qc.t2_us
                )));
            }
            // T2 <= 2*T1 is a fundamental physical constraint
            if qc.t1_us > 0.0 && qc.t2_us > 2.0 * qc.t1_us + 1e-6 {
                return Err(LiveCalibrationError::ValidationError(format!(
                    "Qubit {} violates T2 <= 2*T1: T2={} us, T1={} us",
                    qc.qubit_index, qc.t2_us, qc.t1_us
                )));
            }
            if qc.readout_error < 0.0 || qc.readout_error > 1.0 {
                return Err(LiveCalibrationError::ValidationError(format!(
                    "Qubit {} has invalid readout error: {}",
                    qc.qubit_index, qc.readout_error
                )));
            }
        }

        for gc in &self.gate_calibrations {
            if gc.error_rate < 0.0 || gc.error_rate > 1.0 {
                return Err(LiveCalibrationError::ValidationError(format!(
                    "Gate {} on qubits {:?} has invalid error rate: {}",
                    gc.gate_type, gc.qubits, gc.error_rate
                )));
            }
            for &q in &gc.qubits {
                if q >= self.num_qubits {
                    return Err(LiveCalibrationError::ValidationError(format!(
                        "Gate {} references qubit {} but device has only {} qubits",
                        gc.gate_type, q, self.num_qubits
                    )));
                }
            }
        }

        for ct in &self.crosstalk_entries {
            if ct.source >= self.num_qubits || ct.target >= self.num_qubits {
                return Err(LiveCalibrationError::ValidationError(format!(
                    "Crosstalk entry ({}, {}) out of range for {}-qubit device",
                    ct.source, ct.target, self.num_qubits
                )));
            }
        }

        Ok(())
    }
}

// ============================================================
// SIMULATION NOISE MODEL (output of the builder)
// ============================================================

/// Noise model suitable for direct use in quantum simulation.
///
/// This is the output of `NoiseModelBuilder` -- it contains all per-qubit
/// and per-gate noise parameters pre-computed from calibration data.
#[derive(Debug, Clone)]
pub struct SimulationNoiseModel {
    /// Per-qubit depolarizing error probability (from average gate error).
    pub qubit_depolarizing: Vec<f64>,
    /// Per-qubit T1 amplitude damping probability per gate cycle.
    pub qubit_t1_damping: Vec<f64>,
    /// Per-qubit T2 phase damping probability per gate cycle.
    pub qubit_t2_damping: Vec<f64>,
    /// Per-qubit readout error probability.
    pub qubit_readout_error: Vec<f64>,
    /// Per-gate depolarizing error: key = (gate_type, qubits), value = error_rate.
    pub gate_errors: HashMap<(String, Vec<usize>), f64>,
    /// Crosstalk matrix (dense, n x n). Element [i][j] = coupling strength i->j.
    pub crosstalk_matrix: Vec<Vec<f64>>,
    /// Overall average depolarizing probability (for uniform models).
    pub avg_depolarizing_prob: f64,
    /// Device name for provenance tracking.
    pub device_name: String,
    /// Number of qubits.
    pub num_qubits: usize,
}

// ============================================================
// PROVIDER TRAIT
// ============================================================

/// Trait for quantum hardware calibration providers.
///
/// Each provider knows how to parse its vendor-specific JSON format
/// into the unified `CalibrationData` structure. The `fetch_calibration`
/// method is designed as a synchronous call; actual HTTP access can be
/// implemented behind feature gates.
pub trait LiveCalibrationProvider {
    /// Fetch calibration data for the named device.
    ///
    /// This is synchronous. Implementations may use blocking HTTP (e.g., ureq)
    /// or return cached/mock data.
    fn fetch_calibration(
        &self,
        device_name: &str,
    ) -> Result<CalibrationData, LiveCalibrationError>;

    /// Parse a raw JSON string into `CalibrationData`.
    ///
    /// This allows offline usage with saved calibration files.
    fn parse_calibration_json(
        &self,
        json: &str,
    ) -> Result<CalibrationData, LiveCalibrationError>;

    /// Build a noise model from calibration data using the provider's
    /// recommended settings (e.g., gate set, typical gate times).
    fn build_noise_model(
        &self,
        calibration: &CalibrationData,
    ) -> Result<SimulationNoiseModel, LiveCalibrationError> {
        NoiseModelBuilder::build_realistic(calibration)
    }

    /// Return the provider name (e.g., "ibm", "ionq", "rigetti", "google").
    fn provider_name(&self) -> &str;
}

// ============================================================
// IBM CALIBRATION PROVIDER
// ============================================================

/// Fetches and parses IBM Quantum backend properties.
///
/// IBM calibration JSON follows the Qiskit BackendProperties schema:
/// ```json
/// {
///   "backend_name": "ibm_brisbane",
///   "qubits": [ [ {"name": "T1", "value": 300.0}, ... ], ... ],
///   "gates": [ {"gate": "cx", "qubits": [0,1], "parameters": [...]} ],
///   "timestamp": 1707900000
/// }
/// ```
pub struct IbmCalibrationProvider {
    /// API token for IBM Quantum (optional; required for live fetching).
    pub api_token: Option<String>,
    /// Base URL override (default: https://api.quantum-computing.ibm.com).
    pub base_url: String,
}

impl IbmCalibrationProvider {
    pub fn new(api_token: &str) -> Self {
        Self {
            api_token: Some(api_token.to_string()),
            base_url: "https://api.quantum-computing.ibm.com".to_string(),
        }
    }

    pub fn without_token() -> Self {
        Self {
            api_token: None,
            base_url: "https://api.quantum-computing.ibm.com".to_string(),
        }
    }

    /// Extract a named parameter value from an IBM-style parameter array.
    ///
    /// IBM stores qubit params as `[{"name": "T1", "value": 300.0, "unit": "us"}, ...]`.
    fn named_param(params: &[serde_json::Value], names: &[&str]) -> Option<f64> {
        for p in params {
            if let Some(name) = p.get("name").and_then(|v| v.as_str()) {
                if names.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                    return p.get("value").and_then(|v| v.as_f64());
                }
            }
        }
        None
    }
}

impl LiveCalibrationProvider for IbmCalibrationProvider {
    fn fetch_calibration(
        &self,
        device_name: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        // Live HTTP fetch requires the `qpu` feature. Without it, return a clear error.
        #[cfg(not(feature = "qpu"))]
        {
            let _ = device_name;
            Err(LiveCalibrationError::FetchError(
                "Live IBM fetch requires the 'qpu' feature. \
                 Use parse_calibration_json() with saved data instead."
                    .to_string(),
            ))
        }
        #[cfg(feature = "qpu")]
        {
            let _url = format!(
                "{}/api/backends/{}/properties",
                self.base_url, device_name
            );
            // Actual HTTP implementation would go here with reqwest blocking client.
            // For now, return a clear error indicating the stub.
            Err(LiveCalibrationError::FetchError(format!(
                "IBM live fetch for '{}' not yet wired to HTTP transport",
                device_name
            )))
        }
    }

    fn parse_calibration_json(
        &self,
        json: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        let root: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            LiveCalibrationError::ParseError(format!("IBM JSON parse failed: {}", e))
        })?;

        let device_name = root
            .get("backend_name")
            .or_else(|| root.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("ibm_device")
            .to_string();

        // Parse qubit properties
        let qubits_arr = root
            .get("qubits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                LiveCalibrationError::ParseError(
                    "IBM JSON missing 'qubits' array".to_string(),
                )
            })?;

        let mut qubit_calibrations = Vec::with_capacity(qubits_arr.len());
        for (qi, q_entry) in qubits_arr.iter().enumerate() {
            let params = q_entry.as_array().ok_or_else(|| {
                LiveCalibrationError::ParseError(format!(
                    "IBM qubit {} entry is not an array of parameters",
                    qi
                ))
            })?;

            let t1_us = Self::named_param(params, &["T1"]).unwrap_or(100.0);
            let t2_us = Self::named_param(params, &["T2"])
                .unwrap_or((2.0 * t1_us).min(100.0));
            let frequency_ghz =
                Self::named_param(params, &["frequency"]).unwrap_or(5.0);
            let anharmonicity_ghz =
                Self::named_param(params, &["anharmonicity"]).unwrap_or(-0.34);
            let readout_error =
                Self::named_param(params, &["readout_error"]).unwrap_or(0.02);

            qubit_calibrations.push(QubitCalibration {
                qubit_index: qi,
                t1_us,
                t2_us,
                frequency_ghz,
                readout_error: readout_error.clamp(0.0, 1.0),
                anharmonicity_ghz,
            });
        }

        // Parse gate properties
        let gates_arr = root
            .get("gates")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                LiveCalibrationError::ParseError(
                    "IBM JSON missing 'gates' array".to_string(),
                )
            })?;

        let mut gate_calibrations = Vec::new();
        for g in gates_arr {
            let gate_type = g
                .get("gate")
                .or_else(|| g.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let qubits: Vec<usize> = g
                .get("qubits")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect()
                })
                .unwrap_or_default();

            if qubits.is_empty() {
                continue;
            }

            let params = g.get("parameters").and_then(|v| v.as_array());
            let error_rate = params
                .and_then(|p| {
                    Self::named_param(p, &["gate_error", "error"])
                })
                .or_else(|| g.get("gate_error").and_then(|v| v.as_f64()))
                .unwrap_or(0.0);
            let gate_time_ns = params
                .and_then(|p| {
                    Self::named_param(p, &["gate_length", "gate_time"])
                })
                .or_else(|| g.get("gate_length").and_then(|v| v.as_f64()))
                .unwrap_or(if qubits.len() >= 2 { 300.0 } else { 35.0 });

            gate_calibrations.push(GateCalibration {
                gate_type,
                qubits,
                error_rate: error_rate.clamp(0.0, 1.0),
                gate_time_ns,
            });
        }

        let timestamp_secs = root
            .get("timestamp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let num_qubits = qubit_calibrations.len();
        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), "ibm".to_string());
        if let Some(ver) = root.get("backend_version").and_then(|v| v.as_str()) {
            metadata.insert("backend_version".to_string(), ver.to_string());
        }

        let cal = CalibrationData {
            device_name,
            num_qubits,
            qubit_calibrations,
            gate_calibrations,
            crosstalk_entries: Vec::new(), // IBM doesn't expose crosstalk directly
            timestamp_secs,
            metadata,
        };
        cal.validate()?;
        Ok(cal)
    }

    fn provider_name(&self) -> &str {
        "ibm"
    }
}

// ============================================================
// IONQ CALIBRATION PROVIDER
// ============================================================

/// Fetches and parses IonQ device characterization data.
///
/// IonQ characterization JSON provides aggregate fidelities:
/// ```json
/// {
///   "name": "ionq_aria",
///   "num_qubits": 25,
///   "fidelity": { "1q": 0.9998, "2q": 0.995, "readout": 0.997 },
///   "timing": { "single_ns": 135000, "two_ns": 600000 }
/// }
/// ```
pub struct IonqCalibrationProvider {
    /// API key for IonQ (optional; required for live fetching).
    pub api_key: Option<String>,
    /// Base URL override (default: https://api.ionq.co/v0.2).
    pub base_url: String,
}

impl IonqCalibrationProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: Some(api_key.to_string()),
            base_url: "https://api.ionq.co/v0.2".to_string(),
        }
    }

    pub fn without_key() -> Self {
        Self {
            api_key: None,
            base_url: "https://api.ionq.co/v0.2".to_string(),
        }
    }

    /// Resolve a numeric value from multiple possible JSON paths.
    fn number_from_paths(root: &serde_json::Value, paths: &[&[&str]]) -> Option<f64> {
        for path in paths {
            let mut current = root;
            let mut found = true;
            for key in *path {
                match current.get(key) {
                    Some(v) => current = v,
                    None => {
                        found = false;
                        break;
                    }
                }
            }
            if found {
                if let Some(n) = current.as_f64() {
                    return Some(n);
                }
            }
        }
        None
    }
}

impl LiveCalibrationProvider for IonqCalibrationProvider {
    fn fetch_calibration(
        &self,
        device_name: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        #[cfg(not(feature = "qpu"))]
        {
            let _ = device_name;
            Err(LiveCalibrationError::FetchError(
                "Live IonQ fetch requires the 'qpu' feature. \
                 Use parse_calibration_json() with saved data instead."
                    .to_string(),
            ))
        }
        #[cfg(feature = "qpu")]
        {
            Err(LiveCalibrationError::FetchError(format!(
                "IonQ live fetch for '{}' not yet wired to HTTP transport",
                device_name
            )))
        }
    }

    fn parse_calibration_json(
        &self,
        json: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        let root: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            LiveCalibrationError::ParseError(format!("IonQ JSON parse failed: {}", e))
        })?;

        let num_qubits = Self::number_from_paths(
            &root,
            &[&["num_qubits"], &["qubits"], &["device", "num_qubits"]],
        )
        .map(|x| x as usize)
        .filter(|&x| x > 0)
        .ok_or_else(|| {
            LiveCalibrationError::ParseError(
                "IonQ JSON missing qubit count".to_string(),
            )
        })?;

        let device_name = root
            .get("name")
            .or_else(|| root.get("device").and_then(|d| d.get("name")))
            .and_then(|v| v.as_str())
            .unwrap_or("ionq_device")
            .to_string();

        let sq_fidelity = Self::number_from_paths(
            &root,
            &[
                &["fidelity", "1q"],
                &["single_qubit_fidelity"],
                &["single_qubit", "fidelity"],
            ],
        )
        .unwrap_or(0.999);

        let tq_fidelity = Self::number_from_paths(
            &root,
            &[
                &["fidelity", "2q"],
                &["two_qubit_fidelity"],
                &["two_qubit", "fidelity"],
            ],
        )
        .unwrap_or(0.995);

        let readout_fidelity = Self::number_from_paths(
            &root,
            &[
                &["fidelity", "readout"],
                &["readout_fidelity"],
                &["readout", "fidelity"],
            ],
        )
        .unwrap_or(0.997);

        let sq_gate_ns = Self::number_from_paths(
            &root,
            &[
                &["timing", "single_ns"],
                &["single_qubit", "duration_ns"],
                &["single_qubit", "gate_time_ns"],
            ],
        )
        .unwrap_or(135_000.0);

        let tq_gate_ns = Self::number_from_paths(
            &root,
            &[
                &["timing", "two_ns"],
                &["two_qubit", "duration_ns"],
                &["two_qubit", "gate_time_ns"],
            ],
        )
        .unwrap_or(600_000.0);

        // IonQ trapped ions: extremely long coherence times
        let qubit_calibrations: Vec<QubitCalibration> = (0..num_qubits)
            .map(|qi| QubitCalibration {
                qubit_index: qi,
                t1_us: 10_000_000.0, // ~10 seconds
                t2_us: 1_000_000.0,  // ~1 second
                frequency_ghz: 12.6, // Hyperfine transition
                readout_error: (1.0 - readout_fidelity).clamp(0.0, 1.0),
                anharmonicity_ghz: 0.0, // Not applicable for trapped ions
            })
            .collect();

        // Single-qubit gates for every qubit
        let mut gate_calibrations = Vec::new();
        for q in 0..num_qubits {
            gate_calibrations.push(GateCalibration {
                gate_type: "gpi".to_string(),
                qubits: vec![q],
                error_rate: (1.0 - sq_fidelity).clamp(0.0, 1.0),
                gate_time_ns: sq_gate_ns,
            });
        }
        // All-to-all two-qubit connectivity (trapped ions)
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                gate_calibrations.push(GateCalibration {
                    gate_type: "ms".to_string(),
                    qubits: vec![i, j],
                    error_rate: (1.0 - tq_fidelity).clamp(0.0, 1.0),
                    gate_time_ns: tq_gate_ns,
                });
            }
        }

        let timestamp_secs = root
            .get("timestamp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), "ionq".to_string());

        let cal = CalibrationData {
            device_name,
            num_qubits,
            qubit_calibrations,
            gate_calibrations,
            crosstalk_entries: Vec::new(),
            timestamp_secs,
            metadata,
        };
        cal.validate()?;
        Ok(cal)
    }

    fn provider_name(&self) -> &str {
        "ionq"
    }
}

// ============================================================
// RIGETTI CALIBRATION PROVIDER
// ============================================================

/// Fetches and parses Rigetti ISA calibration data.
///
/// Rigetti calibration provides per-edge gate fidelities:
/// ```json
/// {
///   "name": "rigetti_ankaa2",
///   "num_qubits": 84,
///   "isa": {
///     "1q": { "0": {"fidelity": 0.998, "t1_us": 20.0, "t2_us": 12.0} },
///     "2q": { "0-1": {"fidelity": 0.95, "gate_time_ns": 200.0} }
///   }
/// }
/// ```
pub struct RigettiCalibrationProvider {
    /// API key for Rigetti (optional; required for live fetching).
    pub api_key: Option<String>,
    /// Base URL override.
    pub base_url: String,
}

impl RigettiCalibrationProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: Some(api_key.to_string()),
            base_url: "https://api.qcs.rigetti.com".to_string(),
        }
    }

    pub fn without_key() -> Self {
        Self {
            api_key: None,
            base_url: "https://api.qcs.rigetti.com".to_string(),
        }
    }
}

impl LiveCalibrationProvider for RigettiCalibrationProvider {
    fn fetch_calibration(
        &self,
        device_name: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        #[cfg(not(feature = "qpu"))]
        {
            let _ = device_name;
            Err(LiveCalibrationError::FetchError(
                "Live Rigetti fetch requires the 'qpu' feature. \
                 Use parse_calibration_json() with saved data instead."
                    .to_string(),
            ))
        }
        #[cfg(feature = "qpu")]
        {
            Err(LiveCalibrationError::FetchError(format!(
                "Rigetti live fetch for '{}' not yet wired to HTTP transport",
                device_name
            )))
        }
    }

    fn parse_calibration_json(
        &self,
        json: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        let root: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            LiveCalibrationError::ParseError(format!("Rigetti JSON parse failed: {}", e))
        })?;

        let device_name = root
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("rigetti_device")
            .to_string();

        let isa = root.get("isa").ok_or_else(|| {
            LiveCalibrationError::ParseError(
                "Rigetti JSON missing 'isa' object".to_string(),
            )
        })?;

        // Parse 1Q (single-qubit) entries
        let mut qubit_calibrations = Vec::new();
        let mut gate_calibrations = Vec::new();

        if let Some(sq_obj) = isa.get("1q").and_then(|v| v.as_object()) {
            for (key, val) in sq_obj {
                let qi: usize = key.parse().map_err(|_| {
                    LiveCalibrationError::ParseError(format!(
                        "Rigetti: cannot parse qubit index '{}'",
                        key
                    ))
                })?;

                let fidelity = val.get("fidelity").and_then(|v| v.as_f64()).unwrap_or(0.998);
                let t1_us = val.get("t1_us").and_then(|v| v.as_f64()).unwrap_or(20.0);
                let t2_us = val.get("t2_us").and_then(|v| v.as_f64()).unwrap_or(12.0);
                let frequency_ghz = val
                    .get("frequency_ghz")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(5.5);
                let readout_error = val
                    .get("readout_error")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.04);

                qubit_calibrations.push(QubitCalibration {
                    qubit_index: qi,
                    t1_us,
                    t2_us,
                    frequency_ghz,
                    readout_error: readout_error.clamp(0.0, 1.0),
                    anharmonicity_ghz: -0.30,
                });

                gate_calibrations.push(GateCalibration {
                    gate_type: "rx".to_string(),
                    qubits: vec![qi],
                    error_rate: (1.0 - fidelity).clamp(0.0, 1.0),
                    gate_time_ns: val
                        .get("gate_time_ns")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(40.0),
                });
            }
        }

        // Sort by qubit index for consistency
        qubit_calibrations.sort_by_key(|q| q.qubit_index);

        // Parse 2Q (two-qubit) entries
        if let Some(tq_obj) = isa.get("2q").and_then(|v| v.as_object()) {
            for (key, val) in tq_obj {
                // Keys like "0-1", "1-2", etc.
                let parts: Vec<&str> = key.split('-').collect();
                if parts.len() != 2 {
                    continue;
                }
                let q0: usize = match parts[0].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let q1: usize = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let fidelity = val.get("fidelity").and_then(|v| v.as_f64()).unwrap_or(0.95);
                let gate_time_ns = val
                    .get("gate_time_ns")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(200.0);

                gate_calibrations.push(GateCalibration {
                    gate_type: "cz".to_string(),
                    qubits: vec![q0, q1],
                    error_rate: (1.0 - fidelity).clamp(0.0, 1.0),
                    gate_time_ns,
                });
            }
        }

        let num_qubits = root
            .get("num_qubits")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or_else(|| {
                qubit_calibrations
                    .iter()
                    .map(|q| q.qubit_index + 1)
                    .max()
                    .unwrap_or(0)
            });

        let timestamp_secs = root
            .get("timestamp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), "rigetti".to_string());

        let cal = CalibrationData {
            device_name,
            num_qubits,
            qubit_calibrations,
            gate_calibrations,
            crosstalk_entries: Vec::new(),
            timestamp_secs,
            metadata,
        };
        cal.validate()?;
        Ok(cal)
    }

    fn provider_name(&self) -> &str {
        "rigetti"
    }
}

// ============================================================
// GOOGLE CALIBRATION PROVIDER
// ============================================================

/// Fetches and parses Google Quantum AI device metrics.
///
/// Google calibration provides per-qubit and per-gate metrics:
/// ```json
/// {
///   "device_name": "sycamore",
///   "num_qubits": 53,
///   "qubit_metrics": { "0": {"t1_us": 15.0, "t2_us": 10.0, ...} },
///   "gate_metrics": { "syc": { "0,1": {"error": 0.006, "duration_ns": 12.0} } }
/// }
/// ```
pub struct GoogleCalibrationProvider {
    /// API key (optional; required for live fetching).
    pub api_key: Option<String>,
    /// Base URL override.
    pub base_url: String,
}

impl GoogleCalibrationProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: Some(api_key.to_string()),
            base_url: "https://quantum.googleapis.com/v1alpha1".to_string(),
        }
    }

    pub fn without_key() -> Self {
        Self {
            api_key: None,
            base_url: "https://quantum.googleapis.com/v1alpha1".to_string(),
        }
    }
}

impl LiveCalibrationProvider for GoogleCalibrationProvider {
    fn fetch_calibration(
        &self,
        device_name: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        #[cfg(not(feature = "qpu"))]
        {
            let _ = device_name;
            Err(LiveCalibrationError::FetchError(
                "Live Google fetch requires the 'qpu' feature. \
                 Use parse_calibration_json() with saved data instead."
                    .to_string(),
            ))
        }
        #[cfg(feature = "qpu")]
        {
            Err(LiveCalibrationError::FetchError(format!(
                "Google live fetch for '{}' not yet wired to HTTP transport",
                device_name
            )))
        }
    }

    fn parse_calibration_json(
        &self,
        json: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        let root: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            LiveCalibrationError::ParseError(format!("Google JSON parse failed: {}", e))
        })?;

        let device_name = root
            .get("device_name")
            .or_else(|| root.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("google_device")
            .to_string();

        let mut qubit_calibrations = Vec::new();
        let mut gate_calibrations = Vec::new();

        // Parse qubit metrics
        if let Some(qm) = root.get("qubit_metrics").and_then(|v| v.as_object()) {
            for (key, val) in qm {
                let qi: usize = match key.parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let t1_us = val.get("t1_us").and_then(|v| v.as_f64()).unwrap_or(15.0);
                let t2_us = val.get("t2_us").and_then(|v| v.as_f64()).unwrap_or(10.0);
                let frequency_ghz = val
                    .get("frequency_ghz")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(6.0);
                let readout_error = val
                    .get("readout_error")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.04);
                let anharmonicity_ghz = val
                    .get("anharmonicity_ghz")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(-0.22);

                qubit_calibrations.push(QubitCalibration {
                    qubit_index: qi,
                    t1_us,
                    t2_us,
                    frequency_ghz,
                    readout_error: readout_error.clamp(0.0, 1.0),
                    anharmonicity_ghz,
                });

                // Single-qubit gate from qubit metrics
                let sq_error = val
                    .get("single_qubit_error")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.001);
                gate_calibrations.push(GateCalibration {
                    gate_type: "phased_xz".to_string(),
                    qubits: vec![qi],
                    error_rate: sq_error.clamp(0.0, 1.0),
                    gate_time_ns: val
                        .get("single_qubit_gate_ns")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(25.0),
                });
            }
        }

        qubit_calibrations.sort_by_key(|q| q.qubit_index);

        // Parse gate metrics (keyed by gate type, then by "q0,q1")
        if let Some(gm) = root.get("gate_metrics").and_then(|v| v.as_object()) {
            for (gate_type, edges_obj) in gm {
                if let Some(edges) = edges_obj.as_object() {
                    for (edge_key, val) in edges {
                        let qubits: Vec<usize> = edge_key
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if qubits.len() != 2 {
                            continue;
                        }

                        let error = val
                            .get("error")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.006);
                        let duration_ns = val
                            .get("duration_ns")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(12.0);

                        gate_calibrations.push(GateCalibration {
                            gate_type: gate_type.clone(),
                            qubits,
                            error_rate: error.clamp(0.0, 1.0),
                            gate_time_ns: duration_ns,
                        });
                    }
                }
            }
        }

        let num_qubits = root
            .get("num_qubits")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or_else(|| {
                qubit_calibrations
                    .iter()
                    .map(|q| q.qubit_index + 1)
                    .max()
                    .unwrap_or(0)
            });

        let timestamp_secs = root
            .get("timestamp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        // Parse crosstalk if present
        let mut crosstalk_entries = Vec::new();
        if let Some(ct) = root.get("crosstalk").and_then(|v| v.as_object()) {
            for (key, val) in ct {
                let parts: Vec<usize> = key
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if parts.len() == 2 {
                    if let Some(coeff) = val.as_f64() {
                        crosstalk_entries.push(CrosstalkEntry {
                            source: parts[0],
                            target: parts[1],
                            coefficient: coeff,
                        });
                    }
                }
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), "google".to_string());

        let cal = CalibrationData {
            device_name,
            num_qubits,
            qubit_calibrations,
            gate_calibrations,
            crosstalk_entries,
            timestamp_secs,
            metadata,
        };
        cal.validate()?;
        Ok(cal)
    }

    fn provider_name(&self) -> &str {
        "google"
    }
}

// ============================================================
// CALIBRATION CACHE
// ============================================================

/// Cached calibration entry with timestamp for TTL expiry.
struct CacheEntry {
    data: CalibrationData,
    fetched_at: Instant,
}

/// LRU-style calibration cache with TTL expiry.
///
/// Avoids hammering provider APIs by caching calibration data for a
/// configurable duration (default: 5 minutes). When the cache is full,
/// the oldest entry is evicted.
pub struct CalibrationCache {
    /// TTL for cache entries.
    ttl: Duration,
    /// Maximum number of entries.
    max_entries: usize,
    /// Cached entries keyed by "provider:device_name".
    entries: HashMap<String, CacheEntry>,
    /// Insertion order for LRU eviction.
    insertion_order: Vec<String>,
    /// Cache statistics.
    pub hits: u64,
    pub misses: u64,
}

impl CalibrationCache {
    /// Create a new cache with the given TTL and max capacity.
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            ttl,
            max_entries: max_entries.max(1),
            entries: HashMap::new(),
            insertion_order: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Create a cache with default settings (5 min TTL, 16 entries).
    pub fn default_cache() -> Self {
        Self::new(Duration::from_secs(300), 16)
    }

    /// Cache key for a provider + device combination.
    fn cache_key(provider_name: &str, device_name: &str) -> String {
        format!("{}:{}", provider_name, device_name)
    }

    /// Get calibration from cache, or fetch from provider if missing/expired.
    pub fn get_or_fetch(
        &mut self,
        provider: &dyn LiveCalibrationProvider,
        device_name: &str,
    ) -> Result<CalibrationData, LiveCalibrationError> {
        let key = Self::cache_key(provider.provider_name(), device_name);

        // Check if cached and not expired
        if let Some(entry) = self.entries.get(&key) {
            if entry.fetched_at.elapsed() < self.ttl {
                self.hits += 1;
                return Ok(entry.data.clone());
            }
            // Expired -- remove it
            self.entries.remove(&key);
            self.insertion_order.retain(|k| k != &key);
        }

        // Cache miss -- fetch from provider
        self.misses += 1;
        let data = provider.fetch_calibration(device_name)?;

        // Evict oldest if at capacity
        while self.entries.len() >= self.max_entries {
            if let Some(oldest_key) = self.insertion_order.first().cloned() {
                self.entries.remove(&oldest_key);
                self.insertion_order.remove(0);
            } else {
                break;
            }
        }

        // Insert into cache
        self.entries.insert(
            key.clone(),
            CacheEntry {
                data: data.clone(),
                fetched_at: Instant::now(),
            },
        );
        self.insertion_order.push(key);

        Ok(data)
    }

    /// Manually insert calibration data into the cache (e.g., from parsed JSON).
    pub fn insert(
        &mut self,
        provider_name: &str,
        device_name: &str,
        data: CalibrationData,
    ) {
        let key = Self::cache_key(provider_name, device_name);

        // Remove old entry if present
        if self.entries.contains_key(&key) {
            self.entries.remove(&key);
            self.insertion_order.retain(|k| k != &key);
        }

        // Evict oldest if at capacity
        while self.entries.len() >= self.max_entries {
            if let Some(oldest_key) = self.insertion_order.first().cloned() {
                self.entries.remove(&oldest_key);
                self.insertion_order.remove(0);
            } else {
                break;
            }
        }

        self.entries.insert(
            key.clone(),
            CacheEntry {
                data,
                fetched_at: Instant::now(),
            },
        );
        self.insertion_order.push(key);
    }

    /// Check if a given provider + device combination is cached and not expired.
    pub fn is_cached(&self, provider_name: &str, device_name: &str) -> bool {
        let key = Self::cache_key(provider_name, device_name);
        self.entries
            .get(&key)
            .map(|e| e.fetched_at.elapsed() < self.ttl)
            .unwrap_or(false)
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================
// NOISE MODEL BUILDER
// ============================================================

/// Converts `CalibrationData` into simulation-ready `SimulationNoiseModel`.
///
/// Provides three fidelity levels:
/// - `build_depolarizing`: uniform depolarizing from average gate error (fastest)
/// - `build_simplified`: per-qubit T1/T2 only (no gate-level detail)
/// - `build_realistic`: full per-qubit + per-gate + readout + crosstalk (most accurate)
pub struct NoiseModelBuilder;

impl NoiseModelBuilder {
    /// Build a uniform depolarizing noise model from average gate error.
    ///
    /// All qubits get the same depolarizing probability, derived from the
    /// average single-qubit and two-qubit gate error rates. This is the
    /// fastest model for coarse-grained simulations.
    pub fn build_depolarizing(
        cal: &CalibrationData,
    ) -> Result<SimulationNoiseModel, LiveCalibrationError> {
        cal.validate()?;

        let n = cal.num_qubits;
        let avg_sq_error = cal.avg_single_qubit_error();
        let avg_tq_error = cal.avg_two_qubit_error();
        let avg_readout = cal.avg_readout_error();

        // Uniform depolarizing probability: combine 1Q and 2Q average errors
        let avg_depol = if avg_tq_error > 0.0 {
            (avg_sq_error + avg_tq_error) / 2.0
        } else {
            avg_sq_error
        };

        Ok(SimulationNoiseModel {
            qubit_depolarizing: vec![avg_depol; n],
            qubit_t1_damping: vec![0.0; n],
            qubit_t2_damping: vec![0.0; n],
            qubit_readout_error: vec![avg_readout; n],
            gate_errors: HashMap::new(),
            crosstalk_matrix: vec![vec![0.0; n]; n],
            avg_depolarizing_prob: avg_depol,
            device_name: cal.device_name.clone(),
            num_qubits: n,
        })
    }

    /// Build a simplified noise model using T1/T2 only.
    ///
    /// Per-qubit amplitude damping (T1) and phase damping (T2) are computed
    /// from a reference gate time. No per-gate error detail or crosstalk.
    /// Good for quick simulations that need qubit-level variation.
    pub fn build_simplified(
        cal: &CalibrationData,
    ) -> Result<SimulationNoiseModel, LiveCalibrationError> {
        cal.validate()?;

        let n = cal.num_qubits;

        // Determine a reference gate time from calibration data
        let ref_gate_time_ns = cal
            .gate_calibrations
            .iter()
            .filter(|g| g.qubits.len() == 1 && g.gate_time_ns > 0.0)
            .map(|g| g.gate_time_ns)
            .fold(f64::MAX, f64::min);
        let ref_gate_time_ns = if ref_gate_time_ns == f64::MAX {
            35.0 // Default: IBM sx gate time
        } else {
            ref_gate_time_ns
        };

        let mut qubit_t1_damping = vec![0.0; n];
        let mut qubit_t2_damping = vec![0.0; n];
        let mut qubit_readout_error = vec![0.0; n];

        for qc in &cal.qubit_calibrations {
            let qi = qc.qubit_index;
            if qi >= n {
                continue;
            }

            // Amplitude damping: gamma_1 = 1 - exp(-t_gate / T1)
            let t1_ns = qc.t1_us * 1000.0;
            if t1_ns > 0.0 {
                qubit_t1_damping[qi] = 1.0 - (-ref_gate_time_ns / t1_ns).exp();
            }

            // Phase damping: gamma_phi = 1/T2 - 1/(2*T1), then multiply by gate time
            let t2_ns = qc.t2_us * 1000.0;
            if t2_ns > 0.0 && t1_ns > 0.0 {
                let t2_clamped = t2_ns.min(2.0 * t1_ns);
                let rate_phi = (1.0 / t2_clamped - 1.0 / (2.0 * t1_ns)).max(0.0);
                qubit_t2_damping[qi] = (rate_phi * ref_gate_time_ns).min(1.0);
            }

            qubit_readout_error[qi] = qc.readout_error;
        }

        let avg_depol = cal.avg_single_qubit_error();

        Ok(SimulationNoiseModel {
            qubit_depolarizing: vec![0.0; n], // T1/T2 only, no depolarizing
            qubit_t1_damping,
            qubit_t2_damping,
            qubit_readout_error,
            gate_errors: HashMap::new(),
            crosstalk_matrix: vec![vec![0.0; n]; n],
            avg_depolarizing_prob: avg_depol,
            device_name: cal.device_name.clone(),
            num_qubits: n,
        })
    }

    /// Build a full realistic noise model from calibration data.
    ///
    /// Includes per-qubit T1/T2 damping, per-gate depolarizing errors,
    /// readout errors, and crosstalk coupling. This is the most accurate
    /// model and should be used when simulation fidelity is critical.
    pub fn build_realistic(
        cal: &CalibrationData,
    ) -> Result<SimulationNoiseModel, LiveCalibrationError> {
        cal.validate()?;

        let n = cal.num_qubits;

        // Start with simplified T1/T2 as the base
        let mut model = Self::build_simplified(cal)?;

        // Per-qubit depolarizing from single-qubit gate errors
        for qc in &cal.qubit_calibrations {
            let qi = qc.qubit_index;
            if qi >= n {
                continue;
            }

            // Find best single-qubit gate error for this qubit
            let sq_error = cal
                .gate_calibrations
                .iter()
                .filter(|g| g.qubits.len() == 1 && g.qubits[0] == qi)
                .map(|g| g.error_rate)
                .fold(f64::MAX, f64::min);

            if sq_error < f64::MAX {
                // Convert gate error to depolarizing rate: p_depol = (4/3) * error
                model.qubit_depolarizing[qi] = (4.0 / 3.0) * sq_error;
            }
        }

        // Per-gate error map
        for gc in &cal.gate_calibrations {
            model.gate_errors.insert(
                (gc.gate_type.clone(), gc.qubits.clone()),
                gc.error_rate,
            );
        }

        // Crosstalk matrix
        model.crosstalk_matrix = cal.crosstalk_matrix();

        // Recompute average depolarizing from per-qubit values
        let sum: f64 = model.qubit_depolarizing.iter().sum();
        model.avg_depolarizing_prob = if n > 0 { sum / n as f64 } else { 0.0 };

        Ok(model)
    }
}

// ============================================================
// HELPER: CREATE MOCK CALIBRATION DATA FOR TESTING
// ============================================================

/// Create a minimal CalibrationData for testing purposes.
///
/// This produces a small device with realistic-looking parameters
/// without requiring any JSON parsing.
pub fn mock_calibration(
    device_name: &str,
    num_qubits: usize,
) -> CalibrationData {
    let qubit_calibrations: Vec<QubitCalibration> = (0..num_qubits)
        .map(|qi| {
            let variation = 1.0 + 0.1 * ((qi as f64 * 0.7).sin());
            QubitCalibration {
                qubit_index: qi,
                t1_us: 100.0 * variation,
                t2_us: 80.0 * variation,
                frequency_ghz: 5.0 + 0.2 * ((qi as f64 * 0.3).sin()),
                readout_error: 0.02 * variation,
                anharmonicity_ghz: -0.34,
            }
        })
        .collect();

    let mut gate_calibrations = Vec::new();
    for qi in 0..num_qubits {
        gate_calibrations.push(GateCalibration {
            gate_type: "sx".to_string(),
            qubits: vec![qi],
            error_rate: 0.0003,
            gate_time_ns: 35.0,
        });
    }
    for i in 0..num_qubits.saturating_sub(1) {
        gate_calibrations.push(GateCalibration {
            gate_type: "cx".to_string(),
            qubits: vec![i, i + 1],
            error_rate: 0.01,
            gate_time_ns: 300.0,
        });
    }

    let mut crosstalk_entries = Vec::new();
    for i in 0..num_qubits.saturating_sub(1) {
        crosstalk_entries.push(CrosstalkEntry {
            source: i,
            target: i + 1,
            coefficient: 0.002,
        });
        crosstalk_entries.push(CrosstalkEntry {
            source: i + 1,
            target: i,
            coefficient: 0.001,
        });
    }

    let mut metadata = HashMap::new();
    metadata.insert("provider".to_string(), "mock".to_string());

    CalibrationData {
        device_name: device_name.to_string(),
        num_qubits,
        qubit_calibrations,
        gate_calibrations,
        crosstalk_entries,
        timestamp_secs: 1707900000,
        metadata,
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Sample JSON strings for each provider ----

    fn sample_ibm_json() -> &'static str {
        r#"{
            "backend_name": "ibm_test_5q",
            "backend_version": "1.2.3",
            "timestamp": 1707900000,
            "qubits": [
                [
                    {"name": "T1", "value": 150.0, "unit": "us"},
                    {"name": "T2", "value": 120.0, "unit": "us"},
                    {"name": "frequency", "value": 5.1, "unit": "GHz"},
                    {"name": "anharmonicity", "value": -0.34, "unit": "GHz"},
                    {"name": "readout_error", "value": 0.02}
                ],
                [
                    {"name": "T1", "value": 200.0, "unit": "us"},
                    {"name": "T2", "value": 180.0, "unit": "us"},
                    {"name": "frequency", "value": 5.2, "unit": "GHz"},
                    {"name": "anharmonicity", "value": -0.33, "unit": "GHz"},
                    {"name": "readout_error", "value": 0.015}
                ],
                [
                    {"name": "T1", "value": 180.0, "unit": "us"},
                    {"name": "T2", "value": 140.0, "unit": "us"},
                    {"name": "frequency", "value": 5.0, "unit": "GHz"},
                    {"name": "anharmonicity", "value": -0.35, "unit": "GHz"},
                    {"name": "readout_error", "value": 0.025}
                ],
                [
                    {"name": "T1", "value": 160.0, "unit": "us"},
                    {"name": "T2", "value": 130.0, "unit": "us"},
                    {"name": "frequency", "value": 4.9, "unit": "GHz"},
                    {"name": "anharmonicity", "value": -0.34, "unit": "GHz"},
                    {"name": "readout_error", "value": 0.018}
                ],
                [
                    {"name": "T1", "value": 190.0, "unit": "us"},
                    {"name": "T2", "value": 170.0, "unit": "us"},
                    {"name": "frequency", "value": 5.15, "unit": "GHz"},
                    {"name": "anharmonicity", "value": -0.33, "unit": "GHz"},
                    {"name": "readout_error", "value": 0.022}
                ]
            ],
            "gates": [
                {"gate": "sx", "qubits": [0], "parameters": [{"name": "gate_error", "value": 0.0003}, {"name": "gate_length", "value": 35.0}]},
                {"gate": "sx", "qubits": [1], "parameters": [{"name": "gate_error", "value": 0.0002}, {"name": "gate_length", "value": 35.0}]},
                {"gate": "sx", "qubits": [2], "parameters": [{"name": "gate_error", "value": 0.0004}, {"name": "gate_length", "value": 35.0}]},
                {"gate": "sx", "qubits": [3], "parameters": [{"name": "gate_error", "value": 0.0003}, {"name": "gate_length", "value": 35.0}]},
                {"gate": "sx", "qubits": [4], "parameters": [{"name": "gate_error", "value": 0.0002}, {"name": "gate_length", "value": 35.0}]},
                {"gate": "cx", "qubits": [0, 1], "parameters": [{"name": "gate_error", "value": 0.008}, {"name": "gate_length", "value": 300.0}]},
                {"gate": "cx", "qubits": [1, 2], "parameters": [{"name": "gate_error", "value": 0.009}, {"name": "gate_length", "value": 310.0}]},
                {"gate": "cx", "qubits": [2, 3], "parameters": [{"name": "gate_error", "value": 0.012}, {"name": "gate_length", "value": 290.0}]},
                {"gate": "cx", "qubits": [3, 4], "parameters": [{"name": "gate_error", "value": 0.010}, {"name": "gate_length", "value": 320.0}]}
            ]
        }"#
    }

    fn sample_ionq_json() -> &'static str {
        r#"{
            "name": "ionq_harmony",
            "num_qubits": 11,
            "fidelity": {
                "1q": 0.9998,
                "2q": 0.975,
                "readout": 0.997
            },
            "timing": {
                "single_ns": 135000,
                "two_ns": 600000
            },
            "timestamp": 1707900000
        }"#
    }

    fn sample_rigetti_json() -> &'static str {
        r#"{
            "name": "rigetti_ankaa2",
            "num_qubits": 4,
            "isa": {
                "1q": {
                    "0": {"fidelity": 0.998, "t1_us": 22.0, "t2_us": 13.0, "frequency_ghz": 5.4, "readout_error": 0.035, "gate_time_ns": 40.0},
                    "1": {"fidelity": 0.997, "t1_us": 18.0, "t2_us": 11.0, "frequency_ghz": 5.6, "readout_error": 0.040, "gate_time_ns": 42.0},
                    "2": {"fidelity": 0.999, "t1_us": 25.0, "t2_us": 15.0, "frequency_ghz": 5.3, "readout_error": 0.030, "gate_time_ns": 38.0},
                    "3": {"fidelity": 0.996, "t1_us": 20.0, "t2_us": 12.0, "frequency_ghz": 5.5, "readout_error": 0.045, "gate_time_ns": 41.0}
                },
                "2q": {
                    "0-1": {"fidelity": 0.95, "gate_time_ns": 200.0},
                    "1-2": {"fidelity": 0.94, "gate_time_ns": 210.0},
                    "2-3": {"fidelity": 0.96, "gate_time_ns": 195.0}
                }
            },
            "timestamp": 1707900000
        }"#
    }

    fn sample_google_json() -> &'static str {
        r#"{
            "device_name": "weber",
            "num_qubits": 4,
            "qubit_metrics": {
                "0": {"t1_us": 16.0, "t2_us": 11.0, "frequency_ghz": 6.1, "readout_error": 0.035, "anharmonicity_ghz": -0.22, "single_qubit_error": 0.001, "single_qubit_gate_ns": 25.0},
                "1": {"t1_us": 14.0, "t2_us": 9.5, "frequency_ghz": 5.9, "readout_error": 0.040, "anharmonicity_ghz": -0.21, "single_qubit_error": 0.0012, "single_qubit_gate_ns": 26.0},
                "2": {"t1_us": 18.0, "t2_us": 12.0, "frequency_ghz": 6.0, "readout_error": 0.032, "anharmonicity_ghz": -0.23, "single_qubit_error": 0.0008, "single_qubit_gate_ns": 24.0},
                "3": {"t1_us": 15.0, "t2_us": 10.0, "frequency_ghz": 6.2, "readout_error": 0.038, "anharmonicity_ghz": -0.22, "single_qubit_error": 0.0011, "single_qubit_gate_ns": 25.0}
            },
            "gate_metrics": {
                "syc": {
                    "0,1": {"error": 0.006, "duration_ns": 12.0},
                    "1,2": {"error": 0.007, "duration_ns": 12.0},
                    "2,3": {"error": 0.005, "duration_ns": 12.0}
                }
            },
            "crosstalk": {
                "0,1": 0.003,
                "1,0": 0.002,
                "1,2": 0.004,
                "2,1": 0.003,
                "2,3": 0.002,
                "3,2": 0.001
            },
            "timestamp": 1707900000
        }"#
    }

    // ========== IBM PARSING TESTS ==========

    #[test]
    fn test_parse_ibm_calibration() {
        let provider = IbmCalibrationProvider::without_token();
        let cal = provider.parse_calibration_json(sample_ibm_json()).unwrap();

        assert_eq!(cal.device_name, "ibm_test_5q");
        assert_eq!(cal.num_qubits, 5);
        assert_eq!(cal.qubit_calibrations.len(), 5);
        assert_eq!(cal.timestamp_secs, 1707900000);

        // Verify qubit 0 T1/T2
        let q0 = &cal.qubit_calibrations[0];
        assert!((q0.t1_us - 150.0).abs() < 1e-6);
        assert!((q0.t2_us - 120.0).abs() < 1e-6);
        assert!((q0.frequency_ghz - 5.1).abs() < 1e-6);
        assert!((q0.readout_error - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_ibm_gate_parsing() {
        let provider = IbmCalibrationProvider::without_token();
        let cal = provider.parse_calibration_json(sample_ibm_json()).unwrap();

        // 5 single-qubit sx + 4 two-qubit cx = 9 gates
        assert_eq!(cal.gate_calibrations.len(), 9);

        // Check cx(0,1) error
        let cx_01 = cal.gate_error("cx", &[0, 1]);
        assert!(cx_01.is_some());
        assert!((cx_01.unwrap() - 0.008).abs() < 1e-6);

        // Check sx(2) error
        let sx_2 = cal.gate_error("sx", &[2]);
        assert!(sx_2.is_some());
        assert!((sx_2.unwrap() - 0.0004).abs() < 1e-6);
    }

    #[test]
    fn test_ibm_metadata() {
        let provider = IbmCalibrationProvider::without_token();
        let cal = provider.parse_calibration_json(sample_ibm_json()).unwrap();

        assert_eq!(cal.metadata.get("provider").unwrap(), "ibm");
        assert_eq!(cal.metadata.get("backend_version").unwrap(), "1.2.3");
    }

    // ========== IONQ PARSING TESTS ==========

    #[test]
    fn test_parse_ionq_calibration() {
        let provider = IonqCalibrationProvider::without_key();
        let cal = provider.parse_calibration_json(sample_ionq_json()).unwrap();

        assert_eq!(cal.device_name, "ionq_harmony");
        assert_eq!(cal.num_qubits, 11);
        assert_eq!(cal.qubit_calibrations.len(), 11);

        // IonQ: all qubits have same long T1 (~10s)
        assert!(cal.qubit_calibrations[0].t1_us > 1_000_000.0);
        // IonQ: readout error = 1 - 0.997 = 0.003
        assert!((cal.qubit_calibrations[0].readout_error - 0.003).abs() < 1e-6);
    }

    #[test]
    fn test_ionq_all_to_all_connectivity() {
        let provider = IonqCalibrationProvider::without_key();
        let cal = provider.parse_calibration_json(sample_ionq_json()).unwrap();

        // 11 qubits: C(11,2) = 55 two-qubit gates + 11 single-qubit = 66
        let sq_count = cal
            .gate_calibrations
            .iter()
            .filter(|g| g.qubits.len() == 1)
            .count();
        let tq_count = cal
            .gate_calibrations
            .iter()
            .filter(|g| g.qubits.len() == 2)
            .count();

        assert_eq!(sq_count, 11);
        assert_eq!(tq_count, 55); // C(11,2) = 55

        // Every pair should be connected
        assert!(cal.gate_error("ms", &[3, 7]).is_some());
        assert!(cal.gate_error("ms", &[0, 10]).is_some());
    }

    #[test]
    fn test_ionq_fidelities() {
        let provider = IonqCalibrationProvider::without_key();
        let cal = provider.parse_calibration_json(sample_ionq_json()).unwrap();

        // Single-qubit error = 1 - 0.9998 = 0.0002
        let sq_err = cal.gate_error("gpi", &[0]).unwrap();
        assert!((sq_err - 0.0002).abs() < 1e-6);

        // Two-qubit error = 1 - 0.975 = 0.025
        let tq_err = cal.gate_error("ms", &[0, 1]).unwrap();
        assert!((tq_err - 0.025).abs() < 1e-6);
    }

    // ========== RIGETTI PARSING TESTS ==========

    #[test]
    fn test_parse_rigetti_calibration() {
        let provider = RigettiCalibrationProvider::without_key();
        let cal = provider
            .parse_calibration_json(sample_rigetti_json())
            .unwrap();

        assert_eq!(cal.device_name, "rigetti_ankaa2");
        assert_eq!(cal.num_qubits, 4);
        assert_eq!(cal.qubit_calibrations.len(), 4);
    }

    #[test]
    fn test_rigetti_per_qubit_variation() {
        let provider = RigettiCalibrationProvider::without_key();
        let cal = provider
            .parse_calibration_json(sample_rigetti_json())
            .unwrap();

        // Different T1 values per qubit
        let t1s: Vec<f64> = cal.qubit_calibrations.iter().map(|q| q.t1_us).collect();
        assert!((t1s[0] - 22.0).abs() < 1e-6);
        assert!((t1s[1] - 18.0).abs() < 1e-6);
        assert!((t1s[2] - 25.0).abs() < 1e-6);
        assert!((t1s[3] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_rigetti_two_qubit_gates() {
        let provider = RigettiCalibrationProvider::without_key();
        let cal = provider
            .parse_calibration_json(sample_rigetti_json())
            .unwrap();

        // CZ gate errors: 1 - fidelity
        let cz_01 = cal.gate_error("cz", &[0, 1]).unwrap();
        assert!((cz_01 - 0.05).abs() < 1e-6); // 1 - 0.95

        let cz_23 = cal.gate_error("cz", &[2, 3]).unwrap();
        assert!((cz_23 - 0.04).abs() < 1e-6); // 1 - 0.96
    }

    // ========== GOOGLE PARSING TESTS ==========

    #[test]
    fn test_parse_google_calibration() {
        let provider = GoogleCalibrationProvider::without_key();
        let cal = provider
            .parse_calibration_json(sample_google_json())
            .unwrap();

        assert_eq!(cal.device_name, "weber");
        assert_eq!(cal.num_qubits, 4);
        assert_eq!(cal.qubit_calibrations.len(), 4);
    }

    #[test]
    fn test_google_crosstalk() {
        let provider = GoogleCalibrationProvider::without_key();
        let cal = provider
            .parse_calibration_json(sample_google_json())
            .unwrap();

        assert!(!cal.crosstalk_entries.is_empty());

        let matrix = cal.crosstalk_matrix();
        assert_eq!(matrix.len(), 4);
        assert_eq!(matrix[0].len(), 4);

        // Check specific crosstalk values
        assert!((matrix[0][1] - 0.003).abs() < 1e-6);
        assert!((matrix[1][0] - 0.002).abs() < 1e-6);
        assert!((matrix[2][3] - 0.002).abs() < 1e-6);

        // Diagonal should be zero
        for i in 0..4 {
            assert!((matrix[i][i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_google_gate_metrics() {
        let provider = GoogleCalibrationProvider::without_key();
        let cal = provider
            .parse_calibration_json(sample_google_json())
            .unwrap();

        let syc_01 = cal.gate_error("syc", &[0, 1]).unwrap();
        assert!((syc_01 - 0.006).abs() < 1e-6);

        let syc_time = cal.gate_time_ns("syc", &[0, 1]).unwrap();
        assert!((syc_time - 12.0).abs() < 1e-6);
    }

    // ========== CALIBRATION DATA VALIDATION ==========

    #[test]
    fn test_validation_valid() {
        let cal = mock_calibration("test_device", 5);
        assert!(cal.validate().is_ok());
    }

    #[test]
    fn test_validation_negative_t1() {
        let mut cal = mock_calibration("test_device", 3);
        cal.qubit_calibrations[1].t1_us = -10.0;
        assert!(cal.validate().is_err());
    }

    #[test]
    fn test_validation_t2_exceeds_2t1() {
        let mut cal = mock_calibration("test_device", 3);
        cal.qubit_calibrations[0].t1_us = 50.0;
        cal.qubit_calibrations[0].t2_us = 200.0; // > 2*50 = 100
        let result = cal.validate();
        assert!(result.is_err());
        if let Err(LiveCalibrationError::ValidationError(msg)) = result {
            assert!(msg.contains("T2 <= 2*T1"));
        }
    }

    #[test]
    fn test_validation_invalid_gate_error() {
        let mut cal = mock_calibration("test_device", 3);
        cal.gate_calibrations[0].error_rate = 1.5; // > 1.0
        assert!(cal.validate().is_err());
    }

    #[test]
    fn test_validation_gate_qubit_out_of_range() {
        let mut cal = mock_calibration("test_device", 3);
        cal.gate_calibrations.push(GateCalibration {
            gate_type: "cx".to_string(),
            qubits: vec![0, 99], // qubit 99 does not exist
            error_rate: 0.01,
            gate_time_ns: 300.0,
        });
        assert!(cal.validate().is_err());
    }

    #[test]
    fn test_validation_zero_qubits() {
        let cal = CalibrationData {
            device_name: "empty".to_string(),
            num_qubits: 0,
            qubit_calibrations: Vec::new(),
            gate_calibrations: Vec::new(),
            crosstalk_entries: Vec::new(),
            timestamp_secs: 0,
            metadata: HashMap::new(),
        };
        assert!(cal.validate().is_err());
    }

    // ========== CACHE TESTS ==========

    #[test]
    fn test_cache_insert_and_lookup() {
        let mut cache = CalibrationCache::new(Duration::from_secs(300), 16);

        let cal = mock_calibration("test_device", 5);
        cache.insert("mock", "test_device", cal.clone());

        assert!(cache.is_cached("mock", "test_device"));
        assert!(!cache.is_cached("mock", "other_device"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_expiry() {
        let mut cache = CalibrationCache::new(Duration::from_millis(1), 16);

        let cal = mock_calibration("test_device", 5);
        cache.insert("mock", "test_device", cal);

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(10));

        assert!(!cache.is_cached("mock", "test_device"));
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = CalibrationCache::new(Duration::from_secs(300), 3);

        // Fill cache to capacity
        for i in 0..3 {
            let name = format!("device_{}", i);
            cache.insert("mock", &name, mock_calibration(&name, 2));
        }
        assert_eq!(cache.len(), 3);

        // Adding one more should evict the oldest (device_0)
        cache.insert("mock", "device_new", mock_calibration("device_new", 2));
        assert_eq!(cache.len(), 3);
        assert!(!cache.is_cached("mock", "device_0")); // Evicted
        assert!(cache.is_cached("mock", "device_1"));
        assert!(cache.is_cached("mock", "device_2"));
        assert!(cache.is_cached("mock", "device_new"));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = CalibrationCache::new(Duration::from_secs(300), 16);
        cache.insert("mock", "d1", mock_calibration("d1", 2));
        cache.insert("mock", "d2", mock_calibration("d2", 3));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit_miss_stats() {
        let mut cache = CalibrationCache::new(Duration::from_secs(300), 16);

        let cal = mock_calibration("test", 3);
        cache.insert("mock", "test", cal.clone());

        // Manual stats check: insert doesn't count as hit/miss
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 0);
    }

    // ========== NOISE MODEL BUILDER TESTS ==========

    #[test]
    fn test_build_depolarizing() {
        let cal = mock_calibration("test", 5);
        let model = NoiseModelBuilder::build_depolarizing(&cal).unwrap();

        assert_eq!(model.num_qubits, 5);
        assert_eq!(model.device_name, "test");

        // All qubits should have the same depolarizing prob
        let first = model.qubit_depolarizing[0];
        for &p in &model.qubit_depolarizing {
            assert!((p - first).abs() < 1e-10);
        }

        // T1/T2 damping should be zero in depolarizing model
        for &t in &model.qubit_t1_damping {
            assert!(t.abs() < 1e-10);
        }
    }

    #[test]
    fn test_build_simplified() {
        let cal = mock_calibration("test", 5);
        let model = NoiseModelBuilder::build_simplified(&cal).unwrap();

        assert_eq!(model.num_qubits, 5);

        // T1 damping should be > 0 (since T1 is finite)
        for &t in &model.qubit_t1_damping {
            assert!(t >= 0.0);
        }

        // Depolarizing should be zero in simplified model
        for &p in &model.qubit_depolarizing {
            assert!(p.abs() < 1e-10);
        }

        // Gate errors map should be empty
        assert!(model.gate_errors.is_empty());
    }

    #[test]
    fn test_build_realistic() {
        let cal = mock_calibration("test", 5);
        let model = NoiseModelBuilder::build_realistic(&cal).unwrap();

        assert_eq!(model.num_qubits, 5);

        // Realistic model should have non-zero depolarizing, T1, T2, and gate errors
        assert!(model.qubit_depolarizing.iter().any(|&p| p > 0.0));
        assert!(model.qubit_t1_damping.iter().any(|&p| p > 0.0));
        assert!(!model.gate_errors.is_empty());

        // Crosstalk matrix should have non-zero off-diagonal entries
        let ct = &model.crosstalk_matrix;
        let has_crosstalk = ct
            .iter()
            .enumerate()
            .any(|(i, row)| row.iter().enumerate().any(|(j, &v)| i != j && v > 0.0));
        assert!(has_crosstalk);
    }

    #[test]
    fn test_realistic_from_ibm_json() {
        let provider = IbmCalibrationProvider::without_token();
        let cal = provider.parse_calibration_json(sample_ibm_json()).unwrap();
        let model = NoiseModelBuilder::build_realistic(&cal).unwrap();

        assert_eq!(model.num_qubits, 5);
        assert_eq!(model.device_name, "ibm_test_5q");

        // cx(0,1) should appear in gate_errors
        let cx_err = model
            .gate_errors
            .get(&("cx".to_string(), vec![0, 1]));
        assert!(cx_err.is_some());
        assert!((cx_err.unwrap() - 0.008).abs() < 1e-6);
    }

    #[test]
    fn test_realistic_from_google_json() {
        let provider = GoogleCalibrationProvider::without_key();
        let cal = provider
            .parse_calibration_json(sample_google_json())
            .unwrap();
        let model = NoiseModelBuilder::build_realistic(&cal).unwrap();

        assert_eq!(model.num_qubits, 4);

        // Should have crosstalk from Google data
        assert!((model.crosstalk_matrix[0][1] - 0.003).abs() < 1e-6);
    }

    // ========== CALIBRATION DATA UTILITY TESTS ==========

    #[test]
    fn test_avg_errors() {
        let cal = mock_calibration("test", 5);

        let avg_sq = cal.avg_single_qubit_error();
        assert!(avg_sq > 0.0);
        assert!(avg_sq < 0.01); // Should be small

        let avg_tq = cal.avg_two_qubit_error();
        assert!((avg_tq - 0.01).abs() < 1e-6); // All cx gates have error 0.01

        let avg_ro = cal.avg_readout_error();
        assert!(avg_ro > 0.0);
    }

    #[test]
    fn test_gate_lookup_nonexistent() {
        let cal = mock_calibration("test", 5);

        // Gate that doesn't exist
        assert!(cal.gate_error("ryy", &[0, 1]).is_none());
        // Qubit pair that doesn't have a gate
        assert!(cal.gate_error("cx", &[0, 3]).is_none());
    }

    #[test]
    fn test_crosstalk_matrix_construction() {
        let cal = mock_calibration("test", 5);
        let matrix = cal.crosstalk_matrix();

        assert_eq!(matrix.len(), 5);
        assert_eq!(matrix[0].len(), 5);

        // Adjacent qubits should have non-zero crosstalk
        assert!(matrix[0][1] > 0.0);
        assert!(matrix[1][0] > 0.0);

        // Non-adjacent qubits should have zero crosstalk (in this mock)
        assert!((matrix[0][3]).abs() < 1e-10);
        assert!((matrix[0][4]).abs() < 1e-10);
    }

    // ========== ROUND-TRIP TESTS ==========

    #[test]
    fn test_roundtrip_cal_to_noise_to_params() {
        // Verify that calibration -> noise model -> usable parameters round-trips cleanly
        let cal = mock_calibration("roundtrip_device", 4);

        // Build each model type and verify they produce valid output
        let depol = NoiseModelBuilder::build_depolarizing(&cal).unwrap();
        let simpl = NoiseModelBuilder::build_simplified(&cal).unwrap();
        let real = NoiseModelBuilder::build_realistic(&cal).unwrap();

        // All models should have correct qubit count
        assert_eq!(depol.num_qubits, 4);
        assert_eq!(simpl.num_qubits, 4);
        assert_eq!(real.num_qubits, 4);

        // All noise parameters should be physically valid (non-negative, <= 1)
        for model in &[&depol, &simpl, &real] {
            for &p in &model.qubit_depolarizing {
                assert!(p >= 0.0 && p <= 1.0, "depolarizing out of range: {}", p);
            }
            for &p in &model.qubit_t1_damping {
                assert!(p >= 0.0 && p <= 1.0, "T1 damping out of range: {}", p);
            }
            for &p in &model.qubit_t2_damping {
                assert!(p >= 0.0 && p <= 1.0, "T2 damping out of range: {}", p);
            }
            for &p in &model.qubit_readout_error {
                assert!(p >= 0.0 && p <= 1.0, "readout error out of range: {}", p);
            }
        }

        // Realistic model should be strictly more detailed than depolarizing
        assert!(real.gate_errors.len() > depol.gate_errors.len());
    }

    #[test]
    fn test_provider_name() {
        assert_eq!(IbmCalibrationProvider::without_token().provider_name(), "ibm");
        assert_eq!(IonqCalibrationProvider::without_key().provider_name(), "ionq");
        assert_eq!(
            RigettiCalibrationProvider::without_key().provider_name(),
            "rigetti"
        );
        assert_eq!(
            GoogleCalibrationProvider::without_key().provider_name(),
            "google"
        );
    }

    #[test]
    fn test_invalid_json_error() {
        let provider = IbmCalibrationProvider::without_token();
        let result = provider.parse_calibration_json("not valid json");
        assert!(result.is_err());
        if let Err(LiveCalibrationError::ParseError(msg)) = result {
            assert!(msg.contains("IBM JSON parse failed"));
        } else {
            panic!("Expected ParseError");
        }
    }

    #[test]
    fn test_ibm_missing_qubits_error() {
        let provider = IbmCalibrationProvider::without_token();
        let result = provider.parse_calibration_json(r#"{"gates": []}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_calibration_sizes() {
        let cal_3 = mock_calibration("small", 3);
        assert_eq!(cal_3.num_qubits, 3);
        assert_eq!(cal_3.qubit_calibrations.len(), 3);
        // 3 sx gates + 2 cx gates = 5
        assert_eq!(cal_3.gate_calibrations.len(), 5);
        // 2 pairs * 2 directions = 4 crosstalk entries
        assert_eq!(cal_3.crosstalk_entries.len(), 4);

        let cal_1 = mock_calibration("tiny", 1);
        assert_eq!(cal_1.num_qubits, 1);
        assert_eq!(cal_1.qubit_calibrations.len(), 1);
        assert_eq!(cal_1.gate_calibrations.len(), 1); // Just 1 sx
        assert_eq!(cal_1.crosstalk_entries.len(), 0); // No pairs
    }

    #[test]
    fn test_error_display() {
        let errs = vec![
            LiveCalibrationError::FetchError("network timeout".to_string()),
            LiveCalibrationError::ParseError("invalid JSON".to_string()),
            LiveCalibrationError::ValidationError("T2 > 2*T1".to_string()),
            LiveCalibrationError::DeviceNotFound("ibm_unicorn".to_string()),
            LiveCalibrationError::ProviderError("rate limited".to_string()),
            LiveCalibrationError::CacheError("full".to_string()),
        ];
        for err in &errs {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
        }
    }
}
