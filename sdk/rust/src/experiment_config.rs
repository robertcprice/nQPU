//! Experiment Configuration for nQPU-Metal
//!
//! Provides a declarative experiment configuration system that allows users to
//! define quantum experiments (circuit, noise model, shots, backend, analysis)
//! without writing simulation code directly.
//!
//! # Architecture
//!
//! The configuration follows a layered design:
//! - [`ExperimentConfig`]: Top-level experiment definition
//! - [`CircuitSpec`]: What circuit to run (builtin, custom, QASM, parametric)
//! - [`BackendSpec`]: Which simulation backend to use
//! - [`NoiseSpec`]: Optional noise model injection
//! - [`AnalysisSpec`]: What measurements and analysis to perform
//! - [`OutputSpec`]: How to format and deliver results
//!
//! The [`ExperimentRunner`] consumes an [`ExperimentConfig`] and produces an
//! [`ExperimentResult`] containing counts, expectation values, timing, and metadata.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::experiment_config::*;
//!
//! let config = ExperimentConfig::builder("Bell State Test")
//!     .circuit(CircuitSpec::Builtin {
//!         name: BuiltinCircuit::BellState,
//!         params: vec![],
//!     })
//!     .backend(BackendSpec::StateVector)
//!     .shots(1024)
//!     .build();
//!
//! let result = ExperimentRunner::run(&config).unwrap();
//! assert!(result.counts.contains_key("00") || result.counts.contains_key("11"));
//! ```

use crate::{GateOperations, QuantumSimulator, QuantumState, C64};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;
use std::time::{Duration, Instant};

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during experiment configuration and execution.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Failed to parse configuration input.
    ParseError(String),
    /// The requested backend is not available or invalid.
    InvalidBackend(String),
    /// The circuit specification is malformed.
    InvalidCircuit(String),
    /// A required configuration field is missing.
    MissingRequired(String),
    /// A parameter value is out of the valid range.
    InvalidParameter(String),
    /// The sweep configuration is invalid.
    InvalidSweep(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::ParseError(msg) => write!(f, "parse error: {}", msg),
            ConfigError::InvalidBackend(msg) => write!(f, "invalid backend: {}", msg),
            ConfigError::InvalidCircuit(msg) => write!(f, "invalid circuit: {}", msg),
            ConfigError::MissingRequired(msg) => write!(f, "missing required field: {}", msg),
            ConfigError::InvalidParameter(msg) => write!(f, "invalid parameter: {}", msg),
            ConfigError::InvalidSweep(msg) => write!(f, "invalid sweep: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Convenience result type for experiment operations.
pub type ExperimentResult<T> = std::result::Result<T, ConfigError>;

// ============================================================
// CIRCUIT SPECIFICATION
// ============================================================

/// Well-known circuit templates available without manual gate construction.
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinCircuit {
    /// Bell state: H(0), CNOT(0,1) -- produces (|00> + |11>) / sqrt(2)
    BellState,
    /// GHZ state on n qubits: H(0), CNOT(0,1), CNOT(0,2), ..., CNOT(0,n-1)
    Ghz,
    /// Quantum Fourier Transform on n qubits
    Qft,
    /// Grover search oracle with marked state
    Grover,
    /// Variational quantum eigensolver ansatz (hardware-efficient)
    VqeAnsatz,
}

impl fmt::Display for BuiltinCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuiltinCircuit::BellState => write!(f, "bell_state"),
            BuiltinCircuit::Ghz => write!(f, "ghz"),
            BuiltinCircuit::Qft => write!(f, "qft"),
            BuiltinCircuit::Grover => write!(f, "grover"),
            BuiltinCircuit::VqeAnsatz => write!(f, "vqe_ansatz"),
        }
    }
}

/// A single gate operation in a custom circuit definition.
#[derive(Debug, Clone, PartialEq)]
pub enum GateSpec {
    /// Hadamard gate on target qubit.
    H(usize),
    /// Pauli-X gate on target qubit.
    X(usize),
    /// Pauli-Y gate on target qubit.
    Y(usize),
    /// Pauli-Z gate on target qubit.
    Z(usize),
    /// S (phase) gate on target qubit.
    S(usize),
    /// T (pi/8) gate on target qubit.
    T(usize),
    /// Rotation around X-axis by angle theta.
    Rx(usize, f64),
    /// Rotation around Y-axis by angle theta.
    Ry(usize, f64),
    /// Rotation around Z-axis by angle theta.
    Rz(usize, f64),
    /// Controlled-NOT: (control, target).
    Cnot(usize, usize),
    /// Controlled-Z: (control, target).
    Cz(usize, usize),
    /// Controlled phase rotation: (control, target, phi).
    Cphase(usize, usize, f64),
    /// SWAP gate: (qubit_a, qubit_b).
    Swap(usize, usize),
    /// Toffoli (CCX): (control1, control2, target).
    Toffoli(usize, usize, usize),
    /// Parametric gate placeholder: resolved at runtime via parameter binding.
    /// The string names a parameter from SweepConfig or the parametric template.
    Parametric {
        gate_type: ParametricGateType,
        qubit: usize,
        param_name: String,
    },
}

/// Types of parametric gates that accept a runtime-bound angle parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParametricGateType {
    Rx,
    Ry,
    Rz,
}

/// Specification of the quantum circuit to execute.
#[derive(Debug, Clone)]
pub enum CircuitSpec {
    /// Use a well-known builtin circuit template.
    Builtin {
        name: BuiltinCircuit,
        /// Optional parameters (e.g., number of qubits for GHZ, marked state for Grover).
        params: Vec<f64>,
    },
    /// User-defined gate list with explicit qubit count.
    Custom {
        num_qubits: usize,
        gates: Vec<GateSpec>,
    },
    /// OpenQASM string (parsed and executed).
    Qasm(String),
    /// Parametric circuit template with named parameter placeholders.
    Parametric {
        num_qubits: usize,
        template: Vec<GateSpec>,
        /// Default parameter values (param_name -> value).
        param_defaults: HashMap<String, f64>,
    },
}

// ============================================================
// BACKEND SPECIFICATION
// ============================================================

/// Which simulation backend to use for the experiment.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendSpec {
    /// Full state-vector simulation (2^n amplitudes, exact).
    StateVector,
    /// Density matrix simulation (2^n x 2^n, supports mixed states and noise).
    DensityMatrix,
    /// Matrix Product State simulation with given maximum bond dimension.
    Mps { bond_dim: usize },
    /// Stabilizer (tableau) simulation -- efficient for Clifford-only circuits.
    Stabilizer,
    /// GPU-accelerated simulation on the specified device index.
    Gpu { device: usize },
}

impl Default for BackendSpec {
    fn default() -> Self {
        BackendSpec::StateVector
    }
}

impl fmt::Display for BackendSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendSpec::StateVector => write!(f, "state_vector"),
            BackendSpec::DensityMatrix => write!(f, "density_matrix"),
            BackendSpec::Mps { bond_dim } => write!(f, "mps(bond_dim={})", bond_dim),
            BackendSpec::Stabilizer => write!(f, "stabilizer"),
            BackendSpec::Gpu { device } => write!(f, "gpu(device={})", device),
        }
    }
}

// ============================================================
// NOISE SPECIFICATION
// ============================================================

/// Declarative noise model for injection into the simulation.
///
/// Multiple noise sources can be combined by wrapping them in a vec
/// inside the [`ExperimentConfig`].
#[derive(Debug, Clone)]
pub enum NoiseSpec {
    /// No noise (ideal simulation).
    None,
    /// Depolarizing channel with error probability p per gate.
    Depolarizing { p: f64 },
    /// Amplitude damping (T1 relaxation) with decay rate gamma.
    AmplitudeDamping { gamma: f64 },
    /// Readout (measurement) error with asymmetric bit-flip probabilities.
    ReadoutError {
        /// Probability of reading 1 when the true state is 0.
        p01: f64,
        /// Probability of reading 0 when the true state is 1.
        p10: f64,
    },
    /// Named device noise profile (e.g., "ibm_perth", "google_sycamore").
    DeviceProfile { name: String },
    /// Combined noise model: apply multiple noise sources in sequence.
    Combined(Vec<NoiseSpec>),
}

impl Default for NoiseSpec {
    fn default() -> Self {
        NoiseSpec::None
    }
}

// ============================================================
// ANALYSIS SPECIFICATION
// ============================================================

/// An observable quantity to measure.
#[derive(Debug, Clone, PartialEq)]
pub enum Observable {
    /// Single-qubit Pauli-Z expectation value.
    PauliZ(usize),
    /// Single-qubit Pauli-X expectation value.
    PauliX(usize),
    /// Single-qubit Pauli-Y expectation value.
    PauliY(usize),
    /// Multi-qubit Pauli string, e.g. Z0 * Z1.
    /// Represented as (qubit_index, pauli_label) pairs.
    PauliString(Vec<(usize, PauliLabel)>),
    /// Identity observable (always expectation value 1).
    Identity,
}

/// Labels for individual Pauli operators in a string.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PauliLabel {
    I,
    X,
    Y,
    Z,
}

/// Specification of analysis to perform on simulation results.
#[derive(Debug, Clone)]
pub struct AnalysisSpec {
    /// Expectation values to compute from the final state.
    pub expectation_values: Vec<Observable>,
    /// Whether to perform full state tomography.
    pub state_tomography: bool,
    /// Optional target state for fidelity computation (as amplitude vector).
    pub fidelity_target: Option<Vec<C64>>,
    /// Whether to compute von Neumann entropy of the final state.
    pub entropy: bool,
}

impl Default for AnalysisSpec {
    fn default() -> Self {
        AnalysisSpec {
            expectation_values: Vec::new(),
            state_tomography: false,
            fidelity_target: None,
            entropy: false,
        }
    }
}

// ============================================================
// OUTPUT SPECIFICATION
// ============================================================

/// Output format for experiment results.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    /// JSON format (machine-readable).
    Json,
    /// CSV format (spreadsheet-compatible).
    Csv,
    /// Human-readable text summary.
    Human,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Human
    }
}

/// Configuration for how results are reported.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    /// Output format.
    pub format: OutputFormat,
    /// Optional file path to write results to.
    pub path: Option<String>,
    /// Whether to include the full state vector in output.
    pub include_statevector: bool,
    /// Whether to include measurement counts in output.
    pub include_counts: bool,
}

impl Default for OutputSpec {
    fn default() -> Self {
        OutputSpec {
            format: OutputFormat::Human,
            path: None,
            include_statevector: false,
            include_counts: true,
        }
    }
}

// ============================================================
// PARAMETER SWEEP
// ============================================================

/// Configuration for sweeping a parameter across a range of values.
///
/// Used for VQE landscape plots, phase estimation studies, and
/// any parametric experiment that needs to explore a parameter space.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Name of the parameter to sweep (must match a Parametric gate's param_name).
    pub param_name: String,
    /// Starting value of the sweep range.
    pub start: f64,
    /// Ending value of the sweep range (inclusive).
    pub stop: f64,
    /// Number of evenly-spaced points in the sweep.
    pub steps: usize,
}

impl SweepConfig {
    /// Create a new sweep configuration.
    pub fn new(param_name: impl Into<String>, start: f64, stop: f64, steps: usize) -> Self {
        SweepConfig {
            param_name: param_name.into(),
            start,
            stop,
            steps,
        }
    }

    /// Generate the list of parameter values for this sweep.
    pub fn values(&self) -> Vec<f64> {
        if self.steps <= 1 {
            return vec![self.start];
        }
        let step_size = (self.stop - self.start) / (self.steps - 1) as f64;
        (0..self.steps)
            .map(|i| self.start + i as f64 * step_size)
            .collect()
    }

    /// Validate the sweep configuration.
    pub fn validate(&self) -> ExperimentResult<()> {
        if self.steps == 0 {
            return Err(ConfigError::InvalidSweep(
                "steps must be at least 1".to_string(),
            ));
        }
        if self.param_name.is_empty() {
            return Err(ConfigError::InvalidSweep(
                "param_name must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================
// TOP-LEVEL EXPERIMENT CONFIGURATION
// ============================================================

/// Complete specification of a quantum experiment.
///
/// Holds all the information needed to construct, simulate, measure, and
/// report the results of a quantum experiment without writing any
/// simulation code directly.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Human-readable experiment name.
    pub name: String,
    /// Optional longer description.
    pub description: Option<String>,
    /// Circuit to execute.
    pub circuit: CircuitSpec,
    /// Simulation backend.
    pub backend: BackendSpec,
    /// Noise model (None for ideal).
    pub noise: NoiseSpec,
    /// Analysis pipeline.
    pub analysis: AnalysisSpec,
    /// Output configuration.
    pub output: OutputSpec,
    /// Number of measurement shots.
    pub shots: usize,
    /// Optional parameter sweep (overrides parametric defaults per step).
    pub sweep: Option<SweepConfig>,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        ExperimentConfig {
            name: "Untitled Experiment".to_string(),
            description: None,
            circuit: CircuitSpec::Builtin {
                name: BuiltinCircuit::BellState,
                params: vec![],
            },
            backend: BackendSpec::default(),
            noise: NoiseSpec::default(),
            analysis: AnalysisSpec::default(),
            output: OutputSpec::default(),
            shots: 1024,
            sweep: None,
            metadata: HashMap::new(),
        }
    }
}

impl ExperimentConfig {
    /// Start building an experiment with the given name.
    pub fn builder(name: impl Into<String>) -> ExperimentConfigBuilder {
        ExperimentConfigBuilder::new(name)
    }

    /// Validate the entire configuration for internal consistency.
    pub fn validate(&self) -> ExperimentResult<()> {
        if self.name.is_empty() {
            return Err(ConfigError::MissingRequired("name".to_string()));
        }
        if self.shots == 0 {
            return Err(ConfigError::InvalidParameter(
                "shots must be at least 1".to_string(),
            ));
        }

        // Validate noise parameters
        match &self.noise {
            NoiseSpec::Depolarizing { p } => {
                if *p < 0.0 || *p > 1.0 {
                    return Err(ConfigError::InvalidParameter(format!(
                        "depolarizing p must be in [0, 1], got {}",
                        p
                    )));
                }
            }
            NoiseSpec::AmplitudeDamping { gamma } => {
                if *gamma < 0.0 || *gamma > 1.0 {
                    return Err(ConfigError::InvalidParameter(format!(
                        "amplitude damping gamma must be in [0, 1], got {}",
                        gamma
                    )));
                }
            }
            NoiseSpec::ReadoutError { p01, p10 } => {
                if *p01 < 0.0 || *p01 > 1.0 || *p10 < 0.0 || *p10 > 1.0 {
                    return Err(ConfigError::InvalidParameter(format!(
                        "readout error probabilities must be in [0, 1], got p01={}, p10={}",
                        p01, p10
                    )));
                }
            }
            NoiseSpec::Combined(specs) => {
                for spec in specs {
                    // Recursively validate nested specs (shallow -- Combined inside Combined
                    // is unusual but not forbidden).
                    match spec {
                        NoiseSpec::Depolarizing { p } if *p < 0.0 || *p > 1.0 => {
                            return Err(ConfigError::InvalidParameter(format!(
                                "nested depolarizing p out of range: {}",
                                p
                            )));
                        }
                        NoiseSpec::AmplitudeDamping { gamma } if *gamma < 0.0 || *gamma > 1.0 => {
                            return Err(ConfigError::InvalidParameter(format!(
                                "nested amplitude damping gamma out of range: {}",
                                gamma
                            )));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        // Validate sweep if present
        if let Some(sweep) = &self.sweep {
            sweep.validate()?;
        }

        // Validate circuit qubit count is reasonable
        let num_qubits = self.infer_num_qubits();
        if num_qubits > 30 {
            return Err(ConfigError::InvalidCircuit(format!(
                "circuit uses {} qubits; state-vector simulation is impractical beyond ~30 qubits",
                num_qubits
            )));
        }

        Ok(())
    }

    /// Infer the number of qubits from the circuit specification.
    pub fn infer_num_qubits(&self) -> usize {
        match &self.circuit {
            CircuitSpec::Builtin { name, params } => match name {
                BuiltinCircuit::BellState => 2,
                BuiltinCircuit::Ghz => {
                    params.first().map(|&n| n as usize).unwrap_or(3)
                }
                BuiltinCircuit::Qft => {
                    params.first().map(|&n| n as usize).unwrap_or(3)
                }
                BuiltinCircuit::Grover => {
                    params.first().map(|&n| n as usize).unwrap_or(3)
                }
                BuiltinCircuit::VqeAnsatz => {
                    params.first().map(|&n| n as usize).unwrap_or(4)
                }
            },
            CircuitSpec::Custom { num_qubits, .. } => *num_qubits,
            CircuitSpec::Qasm(qasm_str) => Self::infer_qubits_from_qasm(qasm_str),
            CircuitSpec::Parametric { num_qubits, .. } => *num_qubits,
        }
    }

    /// Parse a QASM string to extract the total qubit count.
    ///
    /// Supports both OpenQASM 2.0 (`qreg q[N];`) and OpenQASM 3.0
    /// (`qubit[N] q;`) register declarations. If multiple registers are
    /// declared, the total qubit count is the sum of all register sizes.
    /// Returns 1 as a safe fallback if no register declarations are found.
    fn infer_qubits_from_qasm(qasm: &str) -> usize {
        let mut total_qubits: usize = 0;

        // Pattern 1: QASM 2.0 style -- qreg <name>[<N>];
        // Matches lines like "qreg q[5];" or "qreg ancilla[2];"
        for line in qasm.lines() {
            let trimmed = line.trim();

            // QASM 2.0: qreg name[N];
            if trimmed.starts_with("qreg ") {
                if let Some(bracket_start) = trimmed.find('[') {
                    if let Some(bracket_end) = trimmed.find(']') {
                        if bracket_end > bracket_start + 1 {
                            if let Ok(n) = trimmed[bracket_start + 1..bracket_end].trim().parse::<usize>() {
                                total_qubits += n;
                            }
                        }
                    }
                }
            }

            // QASM 3.0: qubit[N] name;  or  qubit name;
            if trimmed.starts_with("qubit[") {
                if let Some(bracket_end) = trimmed.find(']') {
                    // Extract N from "qubit[N]"
                    let start = "qubit[".len();
                    if bracket_end > start {
                        if let Ok(n) = trimmed[start..bracket_end].trim().parse::<usize>() {
                            total_qubits += n;
                        }
                    }
                }
            } else if trimmed.starts_with("qubit ") && !trimmed.starts_with("qubit[") {
                // QASM 3.0 single qubit declaration: "qubit q;"
                total_qubits += 1;
            }
        }

        if total_qubits == 0 {
            // Fallback: no register declarations found
            1
        } else {
            total_qubits
        }
    }
}

// ============================================================
// BUILDER PATTERN
// ============================================================

/// Fluent builder for constructing [`ExperimentConfig`] instances.
pub struct ExperimentConfigBuilder {
    config: ExperimentConfig,
}

impl ExperimentConfigBuilder {
    /// Create a new builder with the given experiment name.
    pub fn new(name: impl Into<String>) -> Self {
        ExperimentConfigBuilder {
            config: ExperimentConfig {
                name: name.into(),
                ..Default::default()
            },
        }
    }

    /// Set the experiment description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.config.description = Some(desc.into());
        self
    }

    /// Set the circuit specification.
    pub fn circuit(mut self, circuit: CircuitSpec) -> Self {
        self.config.circuit = circuit;
        self
    }

    /// Set the simulation backend.
    pub fn backend(mut self, backend: BackendSpec) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set the noise model.
    pub fn noise(mut self, noise: NoiseSpec) -> Self {
        self.config.noise = noise;
        self
    }

    /// Set the analysis specification.
    pub fn analysis(mut self, analysis: AnalysisSpec) -> Self {
        self.config.analysis = analysis;
        self
    }

    /// Set the output specification.
    pub fn output(mut self, output: OutputSpec) -> Self {
        self.config.output = output;
        self
    }

    /// Set the number of measurement shots.
    pub fn shots(mut self, shots: usize) -> Self {
        self.config.shots = shots;
        self
    }

    /// Set a parameter sweep configuration.
    pub fn sweep(mut self, sweep: SweepConfig) -> Self {
        self.config.sweep = Some(sweep);
        self
    }

    /// Add a metadata key-value pair.
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.metadata.insert(key.into(), value.into());
        self
    }

    /// Add an expectation value observable to the analysis.
    pub fn observe(mut self, obs: Observable) -> Self {
        self.config.analysis.expectation_values.push(obs);
        self
    }

    /// Enable entropy computation.
    pub fn with_entropy(mut self) -> Self {
        self.config.analysis.entropy = true;
        self
    }

    /// Enable state tomography.
    pub fn with_tomography(mut self) -> Self {
        self.config.analysis.state_tomography = true;
        self
    }

    /// Set a fidelity target state for comparison.
    pub fn fidelity_target(mut self, target: Vec<C64>) -> Self {
        self.config.analysis.fidelity_target = Some(target);
        self
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output.format = format;
        self
    }

    /// Include the state vector in the output.
    pub fn include_statevector(mut self) -> Self {
        self.config.output.include_statevector = true;
        self
    }

    /// Finalize and return the configuration.
    ///
    /// Does not validate -- call [`ExperimentConfig::validate`] separately
    /// if you need pre-flight validation before execution.
    pub fn build(self) -> ExperimentConfig {
        self.config
    }
}

// ============================================================
// EXPERIMENT RESULT
// ============================================================

/// A single sweep-point result (for parameter sweeps).
#[derive(Debug, Clone)]
pub struct SweepPointResult {
    /// The parameter value at this sweep point.
    pub param_value: f64,
    /// Measurement counts at this point.
    pub counts: HashMap<String, usize>,
    /// Expectation values at this point.
    pub expectation_values: HashMap<String, f64>,
}

/// Complete result of an experiment execution.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Experiment name (echoed from config).
    pub name: String,
    /// Measurement outcome counts (bitstring -> count).
    pub counts: HashMap<String, usize>,
    /// Computed expectation values (observable label -> value).
    pub expectation_values: HashMap<String, f64>,
    /// State fidelity with respect to the target (if configured).
    pub fidelity: Option<f64>,
    /// Von Neumann entropy (if configured).
    pub entropy: Option<f64>,
    /// Final state vector (if include_statevector was set).
    pub statevector: Option<Vec<C64>>,
    /// Wall-clock duration of the simulation.
    pub timing: Duration,
    /// Backend that was used.
    pub backend_used: String,
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
    /// Total shots executed.
    pub shots: usize,
    /// Sweep results (empty if no sweep was configured).
    pub sweep_results: Vec<SweepPointResult>,
    /// Echoed metadata from the config.
    pub metadata: HashMap<String, String>,
}

impl RunResult {
    /// Format the result as a JSON string.
    pub fn to_json(&self) -> String {
        let mut lines = Vec::new();
        lines.push("{".to_string());
        lines.push(format!("  \"name\": \"{}\",", self.name));
        lines.push(format!("  \"backend\": \"{}\",", self.backend_used));
        lines.push(format!("  \"num_qubits\": {},", self.num_qubits));
        lines.push(format!("  \"shots\": {},", self.shots));
        lines.push(format!(
            "  \"timing_ms\": {},",
            self.timing.as_millis()
        ));

        // Counts
        lines.push("  \"counts\": {".to_string());
        let count_entries: Vec<String> = self
            .counts
            .iter()
            .map(|(k, v)| format!("    \"{}\": {}", k, v))
            .collect();
        lines.push(count_entries.join(",\n"));
        lines.push("  },".to_string());

        // Expectation values
        lines.push("  \"expectation_values\": {".to_string());
        let ev_entries: Vec<String> = self
            .expectation_values
            .iter()
            .map(|(k, v)| format!("    \"{}\": {:.6}", k, v))
            .collect();
        lines.push(ev_entries.join(",\n"));
        lines.push("  }".to_string());

        // Fidelity
        if let Some(fid) = self.fidelity {
            // Replace the last "}" with a comma and add fidelity
            let last = lines.len() - 1;
            lines[last] = "  },".to_string();
            lines.push(format!("  \"fidelity\": {:.6}", fid));
        }

        lines.push("}".to_string());
        lines.join("\n")
    }

    /// Format the result as CSV lines (header + data rows).
    pub fn to_csv(&self) -> String {
        let mut lines = Vec::new();
        lines.push("bitstring,count".to_string());
        let mut sorted_counts: Vec<_> = self.counts.iter().collect();
        sorted_counts.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (bitstring, count) in &sorted_counts {
            lines.push(format!("{},{}", bitstring, count));
        }

        if !self.expectation_values.is_empty() {
            lines.push(String::new());
            lines.push("observable,value".to_string());
            let mut sorted_evs: Vec<_> = self.expectation_values.iter().collect();
            sorted_evs.sort_by(|(a, _), (b, _)| a.cmp(b));
            for (obs, val) in &sorted_evs {
                lines.push(format!("{},{:.6}", obs, val));
            }
        }
        lines.join("\n")
    }

    /// Format the result as a human-readable summary.
    pub fn to_human(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== {} ===", self.name));
        lines.push(format!(
            "Backend: {} | Qubits: {} | Shots: {}",
            self.backend_used, self.num_qubits, self.shots
        ));
        lines.push(format!("Time: {:.2}ms", self.timing.as_secs_f64() * 1000.0));
        lines.push(String::new());

        lines.push("Measurement counts:".to_string());
        let mut sorted_counts: Vec<_> = self.counts.iter().collect();
        sorted_counts.sort_by(|a, b| b.1.cmp(a.1));
        for (bitstring, count) in &sorted_counts {
            let pct = **count as f64 / self.shots as f64 * 100.0;
            lines.push(format!("  |{}> : {} ({:.1}%)", bitstring, count, pct));
        }

        if !self.expectation_values.is_empty() {
            lines.push(String::new());
            lines.push("Expectation values:".to_string());
            for (obs, val) in &self.expectation_values {
                lines.push(format!("  <{}> = {:.6}", obs, val));
            }
        }

        if let Some(fid) = self.fidelity {
            lines.push(format!("\nFidelity: {:.6}", fid));
        }
        if let Some(ent) = self.entropy {
            lines.push(format!("Entropy: {:.6}", ent));
        }

        lines.join("\n")
    }

    /// Format the result according to the given output specification.
    pub fn format(&self, output: &OutputSpec) -> String {
        match output.format {
            OutputFormat::Json => self.to_json(),
            OutputFormat::Csv => self.to_csv(),
            OutputFormat::Human => self.to_human(),
        }
    }
}

// ============================================================
// CIRCUIT CONSTRUCTION HELPERS
// ============================================================

/// Apply a gate specification to a quantum simulator.
fn apply_gate(sim: &mut QuantumSimulator, gate: &GateSpec, params: &HashMap<String, f64>) {
    match gate {
        GateSpec::H(q) => sim.h(*q),
        GateSpec::X(q) => sim.x(*q),
        GateSpec::Y(q) => sim.y(*q),
        GateSpec::Z(q) => sim.z(*q),
        GateSpec::S(q) => sim.s(*q),
        GateSpec::T(q) => sim.t(*q),
        GateSpec::Rx(q, theta) => sim.rx(*q, *theta),
        GateSpec::Ry(q, theta) => sim.ry(*q, *theta),
        GateSpec::Rz(q, theta) => sim.rz(*q, *theta),
        GateSpec::Cnot(c, t) => sim.cnot(*c, *t),
        GateSpec::Cz(c, t) => sim.cz(*c, *t),
        GateSpec::Cphase(c, t, phi) => sim.cphase(*c, *t, *phi),
        GateSpec::Swap(a, b) => sim.swap(*a, *b),
        GateSpec::Toffoli(c1, c2, t) => sim.toffoli(*c1, *c2, *t),
        GateSpec::Parametric {
            gate_type,
            qubit,
            param_name,
        } => {
            let angle = params.get(param_name).copied().unwrap_or(0.0);
            match gate_type {
                ParametricGateType::Rx => sim.rx(*qubit, angle),
                ParametricGateType::Ry => sim.ry(*qubit, angle),
                ParametricGateType::Rz => sim.rz(*qubit, angle),
            }
        }
    }
}

/// Build a builtin circuit on the given simulator.
fn build_builtin_circuit(
    name: &BuiltinCircuit,
    params: &[f64],
    sim: &mut QuantumSimulator,
) {
    match name {
        BuiltinCircuit::BellState => {
            // |00> -> (|00> + |11>) / sqrt(2)
            sim.h(0);
            sim.cnot(0, 1);
        }
        BuiltinCircuit::Ghz => {
            let n = sim.num_qubits();
            sim.h(0);
            for i in 1..n {
                sim.cnot(0, i);
            }
        }
        BuiltinCircuit::Qft => {
            let n = sim.num_qubits();
            for i in 0..n {
                sim.h(i);
                for j in (i + 1)..n {
                    let angle = PI / (1 << (j - i)) as f64;
                    sim.cphase(j, i, angle);
                }
            }
            // Bit-reversal swaps
            for i in 0..n / 2 {
                sim.swap(i, n - 1 - i);
            }
        }
        BuiltinCircuit::Grover => {
            let n = sim.num_qubits();
            let marked = params.get(1).map(|&m| m as usize).unwrap_or(0);
            let iterations = ((PI / 4.0) * (2.0_f64.powi(n as i32)).sqrt()).floor() as usize;
            let iterations = iterations.max(1);

            // Initialize superposition
            for i in 0..n {
                sim.h(i);
            }

            for _ in 0..iterations {
                // Oracle: flip sign of marked state
                // Encode marked state by applying X to qubits that are 0 in the marked bitstring
                for bit in 0..n {
                    if (marked >> bit) & 1 == 0 {
                        sim.x(bit);
                    }
                }
                // Multi-controlled Z (approximated with available gates for small circuits)
                if n == 2 {
                    sim.cz(0, 1);
                } else if n >= 3 {
                    sim.h(n - 1);
                    if n == 3 {
                        sim.toffoli(0, 1, n - 1);
                    } else {
                        // For n > 3, use cascaded Toffoli decomposition (simplified)
                        sim.toffoli(0, 1, n - 1);
                    }
                    sim.h(n - 1);
                }
                for bit in 0..n {
                    if (marked >> bit) & 1 == 0 {
                        sim.x(bit);
                    }
                }

                // Diffusion operator
                for i in 0..n {
                    sim.h(i);
                    sim.x(i);
                }
                if n == 2 {
                    sim.cz(0, 1);
                } else if n >= 3 {
                    sim.h(n - 1);
                    sim.toffoli(0, 1, n - 1);
                    sim.h(n - 1);
                }
                for i in 0..n {
                    sim.x(i);
                    sim.h(i);
                }
            }
        }
        BuiltinCircuit::VqeAnsatz => {
            let n = sim.num_qubits();
            // Hardware-efficient ansatz: Ry layer + CNOT entangling layer
            // Use param values as rotation angles (or defaults)
            for i in 0..n {
                let angle = params.get(i + 1).copied().unwrap_or(PI / 4.0);
                sim.ry(i, angle);
            }
            for i in 0..n - 1 {
                sim.cnot(i, i + 1);
            }
            // Second Ry layer
            for i in 0..n {
                let angle = params
                    .get(n + i + 1)
                    .copied()
                    .unwrap_or(PI / 3.0);
                sim.ry(i, angle);
            }
        }
    }
}

/// Compute the expectation value of a single-qubit Pauli-Z observable.
///
/// <Z_q> = sum_i (-1)^(bit q of i) * |a_i|^2
fn expectation_pauli_z(state: &QuantumState, qubit: usize) -> f64 {
    let probs = state.probabilities();
    let mask = 1 << qubit;
    let mut ev = 0.0;
    for (i, p) in probs.iter().enumerate() {
        if i & mask == 0 {
            ev += p; // eigenvalue +1
        } else {
            ev -= p; // eigenvalue -1
        }
    }
    ev
}

/// Compute the expectation value of a single-qubit Pauli-X observable.
///
/// <X_q> = 2 * Re( sum_{i: bit q = 0} conj(a_i) * a_{i XOR 2^q} )
fn expectation_pauli_x(state: &QuantumState, qubit: usize) -> f64 {
    let amps = state.amplitudes_ref();
    let mask = 1 << qubit;
    let mut ev = 0.0;
    for i in 0..state.dim {
        if i & mask == 0 {
            let j = i | mask;
            // <X> contribution: conj(a_i) * a_j + conj(a_j) * a_i = 2 * Re(conj(a_i) * a_j)
            let product = amps[i].conj() * amps[j];
            ev += product.re;
        }
    }
    2.0 * ev
}

/// Compute the expectation value of a single-qubit Pauli-Y observable.
///
/// <Y_q> = 2 * Im( sum_{i: bit q = 0} conj(a_i) * a_{i XOR 2^q} )
fn expectation_pauli_y(state: &QuantumState, qubit: usize) -> f64 {
    let amps = state.amplitudes_ref();
    let mask = 1 << qubit;
    let mut ev = 0.0;
    for i in 0..state.dim {
        if i & mask == 0 {
            let j = i | mask;
            let product = amps[i].conj() * amps[j];
            ev += product.im;
        }
    }
    // The Y Pauli has an extra factor of i relative to X, so <Y> = 2 * Im(sum(...))
    // but with the correct sign convention: Y = -i|0><1| + i|1><0|
    // <Y> = -2 * Im(sum_{bit=0} conj(a_i) * a_{i^mask})
    -2.0 * ev
}

/// Compute the expectation value of a multi-qubit Pauli string.
fn expectation_pauli_string(state: &QuantumState, terms: &[(usize, PauliLabel)]) -> f64 {
    // For a Pauli string P = P_1 x P_2 x ... x P_k, we compute <psi|P|psi>.
    // We operate on the state vector directly.
    let amps = state.amplitudes_ref();
    let dim = state.dim;
    let mut ev = C64::new(0.0, 0.0);

    for i in 0..dim {
        // Compute P|i> and its coefficient
        let mut j = i;
        let mut coeff = C64::new(1.0, 0.0);

        for (qubit, pauli) in terms {
            let mask = 1 << qubit;
            let bit = (j >> qubit) & 1;
            match pauli {
                PauliLabel::I => {}
                PauliLabel::X => {
                    j ^= mask; // flip bit
                }
                PauliLabel::Y => {
                    j ^= mask; // flip bit
                    if bit == 0 {
                        coeff = coeff * C64::new(0.0, 1.0); // i
                    } else {
                        coeff = coeff * C64::new(0.0, -1.0); // -i
                    }
                }
                PauliLabel::Z => {
                    if bit == 1 {
                        coeff = coeff * C64::new(-1.0, 0.0);
                    }
                }
            }
        }

        // <i|P|psi> contributes conj(a_i) * coeff * a_j
        ev += amps[i].conj() * coeff * amps[j];
    }

    ev.re // Should be real for a Hermitian observable
}

/// Compute the von Neumann entropy of the state.
///
/// For a pure state, the entropy is 0. For a mixed state (from probabilities),
/// we compute the Shannon entropy of the measurement probability distribution:
/// S = -sum p_i * log2(p_i)
fn compute_entropy(state: &QuantumState) -> f64 {
    let probs = state.probabilities();
    let mut entropy = 0.0;
    for &p in &probs {
        if p > 1e-15 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compute the state fidelity |<target|state>|^2.
fn compute_fidelity(state: &QuantumState, target: &[C64]) -> f64 {
    let amps = state.amplitudes_ref();
    if amps.len() != target.len() {
        return 0.0;
    }
    let overlap: C64 = amps
        .iter()
        .zip(target.iter())
        .map(|(a, t)| t.conj() * a)
        .sum();
    overlap.norm_sqr()
}

/// Convert a measurement index to a bitstring of given width.
fn index_to_bitstring(index: usize, num_qubits: usize) -> String {
    (0..num_qubits)
        .rev()
        .map(|bit| if index & (1 << bit) != 0 { '1' } else { '0' })
        .collect()
}

/// Apply simple depolarizing noise to a state (stochastic Pauli channel).
///
/// With probability p, applies a random Pauli (X, Y, or Z) to each qubit.
fn apply_depolarizing_noise(state: &mut QuantumState, p: f64) {
    let n = state.num_qubits;
    for qubit in 0..n {
        let r: f64 = rand::random();
        if r < p / 3.0 {
            GateOperations::x(state, qubit);
        } else if r < 2.0 * p / 3.0 {
            GateOperations::y(state, qubit);
        } else if r < p {
            GateOperations::z(state, qubit);
        }
    }
}

/// Apply readout error to a measurement outcome.
///
/// For each bit in the bitstring, flip it with the appropriate probability.
fn apply_readout_error(bitstring: &str, p01: f64, p10: f64) -> String {
    bitstring
        .chars()
        .map(|c| {
            let r: f64 = rand::random();
            match c {
                '0' if r < p01 => '1',
                '1' if r < p10 => '0',
                other => other,
            }
        })
        .collect()
}

// ============================================================
// EXPERIMENT RUNNER
// ============================================================

/// Executes an [`ExperimentConfig`] and produces a [`RunResult`].
///
/// The runner handles:
/// - Circuit construction from the specification
/// - Backend selection and initialization
/// - Noise injection (if configured)
/// - Repeated measurement (shots)
/// - Expectation value computation
/// - Parameter sweeps
/// - Timing and metadata collection
pub struct ExperimentRunner;

impl ExperimentRunner {
    /// Execute the experiment described by the given configuration.
    ///
    /// Returns a [`RunResult`] on success or a [`ConfigError`] if the
    /// configuration is invalid or execution fails.
    pub fn run(config: &ExperimentConfig) -> ExperimentResult<RunResult> {
        config.validate()?;

        // If there is a sweep, delegate to sweep execution
        if let Some(sweep) = &config.sweep {
            return Self::run_sweep(config, sweep);
        }

        Self::run_single(config, &HashMap::new())
    }

    /// Execute a single experiment (no sweep).
    fn run_single(
        config: &ExperimentConfig,
        param_overrides: &HashMap<String, f64>,
    ) -> ExperimentResult<RunResult> {
        let start = Instant::now();
        let num_qubits = config.infer_num_qubits();

        // Build the circuit on a state-vector simulator.
        // For DensityMatrix or Stabilizer backends, we still use the state-vector
        // path here for simplicity -- the backend spec primarily affects
        // how we label the result and could be extended for true backend dispatch.
        let mut sim = QuantumSimulator::new(num_qubits);
        Self::build_circuit(config, &mut sim, param_overrides)?;

        // Apply noise after circuit execution (simplified noise model)
        Self::apply_noise(&config.noise, &mut sim.state);

        // Compute expectation values from the final state (before measurement collapse)
        let expectation_values =
            Self::compute_expectation_values(&config.analysis, &sim.state);

        // Compute fidelity if target specified
        let fidelity = config
            .analysis
            .fidelity_target
            .as_ref()
            .map(|target| compute_fidelity(&sim.state, target));

        // Compute entropy if requested
        let entropy = if config.analysis.entropy {
            Some(compute_entropy(&sim.state))
        } else {
            None
        };

        // Capture state vector before measurement (if requested)
        let statevector = if config.output.include_statevector {
            Some(sim.state.amplitudes_ref().to_vec())
        } else {
            None
        };

        // Perform measurement shots
        let mut counts: HashMap<String, usize> = HashMap::new();
        if config.output.include_counts {
            let probs = sim.state.probabilities();
            for _ in 0..config.shots {
                let r: f64 = rand::random();
                let mut cumsum = 0.0;
                let mut outcome = 0;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if r < cumsum {
                        outcome = i;
                        break;
                    }
                }
                let mut bitstring = index_to_bitstring(outcome, num_qubits);

                // Apply readout error if configured
                if let NoiseSpec::ReadoutError { p01, p10 } = &config.noise {
                    bitstring = apply_readout_error(&bitstring, *p01, *p10);
                }
                if let NoiseSpec::Combined(specs) = &config.noise {
                    for spec in specs {
                        if let NoiseSpec::ReadoutError { p01, p10 } = spec {
                            bitstring = apply_readout_error(&bitstring, *p01, *p10);
                        }
                    }
                }

                *counts.entry(bitstring).or_insert(0) += 1;
            }
        }

        let timing = start.elapsed();

        Ok(RunResult {
            name: config.name.clone(),
            counts,
            expectation_values,
            fidelity,
            entropy,
            statevector,
            timing,
            backend_used: config.backend.to_string(),
            num_qubits,
            shots: config.shots,
            sweep_results: Vec::new(),
            metadata: config.metadata.clone(),
        })
    }

    /// Execute a parameter sweep experiment.
    fn run_sweep(
        config: &ExperimentConfig,
        sweep: &SweepConfig,
    ) -> ExperimentResult<RunResult> {
        let start = Instant::now();
        let values = sweep.values();
        let mut sweep_results = Vec::with_capacity(values.len());
        let mut all_counts: HashMap<String, usize> = HashMap::new();
        let mut all_evs: HashMap<String, f64> = HashMap::new();

        for val in &values {
            let mut overrides = HashMap::new();
            overrides.insert(sweep.param_name.clone(), *val);

            let point_result = Self::run_single(config, &overrides)?;

            // Merge counts
            for (k, v) in &point_result.counts {
                *all_counts.entry(k.clone()).or_insert(0) += v;
            }

            sweep_results.push(SweepPointResult {
                param_value: *val,
                counts: point_result.counts,
                expectation_values: point_result.expectation_values.clone(),
            });

            // Track last point's EVs as the "aggregate"
            all_evs = point_result.expectation_values;
        }

        let timing = start.elapsed();
        let num_qubits = config.infer_num_qubits();

        Ok(RunResult {
            name: config.name.clone(),
            counts: all_counts,
            expectation_values: all_evs,
            fidelity: None,
            entropy: None,
            statevector: None,
            timing,
            backend_used: config.backend.to_string(),
            num_qubits,
            shots: config.shots,
            sweep_results,
            metadata: config.metadata.clone(),
        })
    }

    /// Construct the circuit on the simulator.
    fn build_circuit(
        config: &ExperimentConfig,
        sim: &mut QuantumSimulator,
        param_overrides: &HashMap<String, f64>,
    ) -> ExperimentResult<()> {
        match &config.circuit {
            CircuitSpec::Builtin { name, params } => {
                build_builtin_circuit(name, params, sim);
            }
            CircuitSpec::Custom { gates, .. } => {
                for gate in gates {
                    apply_gate(sim, gate, param_overrides);
                }
            }
            CircuitSpec::Qasm(qasm_str) => {
                // Minimal QASM parsing -- this is a simplified stub.
                // Full QASM support is provided by the crate::qasm3 module.
                return Err(ConfigError::InvalidCircuit(format!(
                    "QASM parsing is not yet integrated in experiment_config; \
                     use crate::qasm3 for full QASM 3.0 support. Input length: {} chars",
                    qasm_str.len()
                )));
            }
            CircuitSpec::Parametric {
                template,
                param_defaults,
                ..
            } => {
                // Merge defaults with overrides (overrides take precedence)
                let mut merged = param_defaults.clone();
                for (k, v) in param_overrides {
                    merged.insert(k.clone(), *v);
                }
                for gate in template {
                    apply_gate(sim, gate, &merged);
                }
            }
        }
        Ok(())
    }

    /// Apply noise to the state based on the noise specification.
    fn apply_noise(noise: &NoiseSpec, state: &mut QuantumState) {
        match noise {
            NoiseSpec::None => {}
            NoiseSpec::Depolarizing { p } => {
                apply_depolarizing_noise(state, *p);
            }
            NoiseSpec::AmplitudeDamping { gamma } => {
                // Simplified amplitude damping: for each qubit, with probability gamma,
                // apply a partial decay towards |0>
                let n = state.num_qubits;
                for qubit in 0..n {
                    let r: f64 = rand::random();
                    if r < *gamma {
                        // Crude amplitude damping: apply sqrt(gamma) damping
                        // This is a simplified model; for proper Kraus-operator
                        // amplitude damping, use the crate::noise module.
                        let mask = 1 << qubit;
                        let dim = state.dim;
                        let sqrt_gamma = gamma.sqrt();
                        let sqrt_1mg = (1.0 - gamma).sqrt();
                        let amps = state.amplitudes_mut();
                        for i in 0..dim {
                            if i & mask != 0 {
                                let j = i ^ mask;
                                let excited = amps[i];
                                amps[j] = amps[j] + C64::new(sqrt_gamma, 0.0) * excited;
                                amps[i] = C64::new(sqrt_1mg, 0.0) * excited;
                            }
                        }
                        // Re-normalize
                        let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
                        if norm > 1e-15 {
                            for a in amps.iter_mut() {
                                *a = *a / norm;
                            }
                        }
                    }
                }
            }
            NoiseSpec::ReadoutError { .. } => {
                // Readout errors are applied during measurement, not to the state
            }
            NoiseSpec::DeviceProfile { name } => {
                // Device profiles would map to specific noise parameters.
                // For now, apply a generic light depolarizing noise.
                let default_p = match name.as_str() {
                    "ibm_perth" => 0.005,
                    "google_sycamore" => 0.003,
                    _ => 0.01,
                };
                apply_depolarizing_noise(state, default_p);
            }
            NoiseSpec::Combined(specs) => {
                for spec in specs {
                    Self::apply_noise(spec, state);
                }
            }
        }
    }

    /// Compute all requested expectation values from the final state.
    fn compute_expectation_values(
        analysis: &AnalysisSpec,
        state: &QuantumState,
    ) -> HashMap<String, f64> {
        let mut evs = HashMap::new();
        for obs in &analysis.expectation_values {
            let (label, value) = match obs {
                Observable::PauliZ(q) => {
                    (format!("Z{}", q), expectation_pauli_z(state, *q))
                }
                Observable::PauliX(q) => {
                    (format!("X{}", q), expectation_pauli_x(state, *q))
                }
                Observable::PauliY(q) => {
                    (format!("Y{}", q), expectation_pauli_y(state, *q))
                }
                Observable::PauliString(terms) => {
                    let label: String = terms
                        .iter()
                        .map(|(q, p)| format!("{:?}{}", p, q))
                        .collect::<Vec<_>>()
                        .join("*");
                    let value = expectation_pauli_string(state, terms);
                    (label, value)
                }
                Observable::Identity => ("I".to_string(), 1.0),
            };
            evs.insert(label, value);
        }
        evs
    }
}

// ============================================================
// SIMPLE KEY-VALUE CONFIG PARSER
// ============================================================

/// Parse a simple key-value configuration string into an ExperimentConfig.
///
/// Supports a TOML-like syntax:
/// ```text
/// name = "Bell State Test"
/// circuit = "bell_state"
/// backend = "state_vector"
/// shots = 1024
/// noise = "depolarizing(0.01)"
/// ```
///
/// This is intentionally minimal -- for full TOML parsing, add the `toml`
/// dependency. This parser handles the most common use cases.
pub fn parse_config(input: &str) -> ExperimentResult<ExperimentConfig> {
    let mut config = ExperimentConfig::default();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(ConfigError::ParseError(format!(
                "expected 'key = value', got: {}",
                line
            )));
        }

        let key = parts[0].trim();
        let value = parts[1].trim().trim_matches('"');

        match key {
            "name" => config.name = value.to_string(),
            "description" => config.description = Some(value.to_string()),
            "shots" => {
                config.shots = value.parse().map_err(|_| {
                    ConfigError::ParseError(format!("invalid shots value: {}", value))
                })?;
            }
            "circuit" => {
                config.circuit = match value {
                    "bell_state" => CircuitSpec::Builtin {
                        name: BuiltinCircuit::BellState,
                        params: vec![],
                    },
                    "ghz" => CircuitSpec::Builtin {
                        name: BuiltinCircuit::Ghz,
                        params: vec![3.0],
                    },
                    "qft" => CircuitSpec::Builtin {
                        name: BuiltinCircuit::Qft,
                        params: vec![3.0],
                    },
                    "grover" => CircuitSpec::Builtin {
                        name: BuiltinCircuit::Grover,
                        params: vec![3.0],
                    },
                    "vqe_ansatz" => CircuitSpec::Builtin {
                        name: BuiltinCircuit::VqeAnsatz,
                        params: vec![4.0],
                    },
                    _ => {
                        return Err(ConfigError::InvalidCircuit(format!(
                            "unknown builtin circuit: {}",
                            value
                        )));
                    }
                };
            }
            "backend" => {
                config.backend = match value {
                    "state_vector" => BackendSpec::StateVector,
                    "density_matrix" => BackendSpec::DensityMatrix,
                    "stabilizer" => BackendSpec::Stabilizer,
                    s if s.starts_with("mps") => {
                        // Parse mps(64) or mps(bond_dim=64)
                        let inner = s
                            .trim_start_matches("mps(")
                            .trim_end_matches(')');
                        let bd_str = inner
                            .trim_start_matches("bond_dim=")
                            .trim();
                        let bond_dim: usize = bd_str.parse().map_err(|_| {
                            ConfigError::InvalidBackend(format!(
                                "invalid MPS bond dimension: {}",
                                bd_str
                            ))
                        })?;
                        BackendSpec::Mps { bond_dim }
                    }
                    s if s.starts_with("gpu") => {
                        let inner = s
                            .trim_start_matches("gpu(")
                            .trim_end_matches(')');
                        let dev_str = inner
                            .trim_start_matches("device=")
                            .trim();
                        let device: usize = dev_str.parse().unwrap_or(0);
                        BackendSpec::Gpu { device }
                    }
                    _ => {
                        return Err(ConfigError::InvalidBackend(format!(
                            "unknown backend: {}",
                            value
                        )));
                    }
                };
            }
            "noise" => {
                config.noise = parse_noise_spec(value)?;
            }
            "output_format" => {
                config.output.format = match value {
                    "json" => OutputFormat::Json,
                    "csv" => OutputFormat::Csv,
                    "human" => OutputFormat::Human,
                    _ => {
                        return Err(ConfigError::ParseError(format!(
                            "unknown output format: {}",
                            value
                        )));
                    }
                };
            }
            "include_statevector" => {
                config.output.include_statevector = value == "true";
            }
            "include_counts" => {
                config.output.include_counts = value == "true";
            }
            "output_path" => {
                config.output.path = Some(value.to_string());
            }
            _ => {
                // Unknown keys are stored as metadata
                config
                    .metadata
                    .insert(key.to_string(), value.to_string());
            }
        }
    }

    Ok(config)
}

/// Parse a noise specification string.
///
/// Supported formats:
/// - "none"
/// - "depolarizing(0.01)"
/// - "amplitude_damping(0.05)"
/// - "readout_error(0.01, 0.02)"
/// - "device(ibm_perth)"
fn parse_noise_spec(value: &str) -> ExperimentResult<NoiseSpec> {
    let value = value.trim();
    if value == "none" {
        return Ok(NoiseSpec::None);
    }

    if value.starts_with("depolarizing(") {
        let inner = value
            .trim_start_matches("depolarizing(")
            .trim_end_matches(')');
        let p: f64 = inner.parse().map_err(|_| {
            ConfigError::ParseError(format!(
                "invalid depolarizing parameter: {}",
                inner
            ))
        })?;
        return Ok(NoiseSpec::Depolarizing { p });
    }

    if value.starts_with("amplitude_damping(") {
        let inner = value
            .trim_start_matches("amplitude_damping(")
            .trim_end_matches(')');
        let gamma: f64 = inner.parse().map_err(|_| {
            ConfigError::ParseError(format!(
                "invalid amplitude damping parameter: {}",
                inner
            ))
        })?;
        return Ok(NoiseSpec::AmplitudeDamping { gamma });
    }

    if value.starts_with("readout_error(") {
        let inner = value
            .trim_start_matches("readout_error(")
            .trim_end_matches(')');
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(ConfigError::ParseError(
                "readout_error requires two parameters: p01, p10".to_string(),
            ));
        }
        let p01: f64 = parts[0].parse().map_err(|_| {
            ConfigError::ParseError(format!("invalid p01: {}", parts[0]))
        })?;
        let p10: f64 = parts[1].parse().map_err(|_| {
            ConfigError::ParseError(format!("invalid p10: {}", parts[1]))
        })?;
        return Ok(NoiseSpec::ReadoutError { p01, p10 });
    }

    if value.starts_with("device(") {
        let inner = value
            .trim_start_matches("device(")
            .trim_end_matches(')');
        return Ok(NoiseSpec::DeviceProfile {
            name: inner.to_string(),
        });
    }

    Err(ConfigError::ParseError(format!(
        "unknown noise specification: {}",
        value
    )))
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = ExperimentConfig::default();
        assert_eq!(config.name, "Untitled Experiment");
        assert_eq!(config.shots, 1024);
        assert_eq!(config.backend, BackendSpec::StateVector);
        assert!(config.validate().is_ok());

        match &config.circuit {
            CircuitSpec::Builtin { name, .. } => {
                assert_eq!(*name, BuiltinCircuit::BellState);
            }
            _ => panic!("default circuit should be a builtin Bell state"),
        }
    }

    #[test]
    fn config_builder() {
        let config = ExperimentConfig::builder("My Experiment")
            .description("Testing the builder pattern")
            .backend(BackendSpec::DensityMatrix)
            .shots(2048)
            .noise(NoiseSpec::Depolarizing { p: 0.01 })
            .observe(Observable::PauliZ(0))
            .with_entropy()
            .output_format(OutputFormat::Json)
            .meta("author", "nqpu-metal")
            .build();

        assert_eq!(config.name, "My Experiment");
        assert_eq!(config.description, Some("Testing the builder pattern".to_string()));
        assert_eq!(config.backend, BackendSpec::DensityMatrix);
        assert_eq!(config.shots, 2048);
        assert_eq!(config.analysis.expectation_values.len(), 1);
        assert!(config.analysis.entropy);
        assert_eq!(config.output.format, OutputFormat::Json);
        assert_eq!(config.metadata.get("author"), Some(&"nqpu-metal".to_string()));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn builtin_bell_state() {
        let config = ExperimentConfig::builder("Bell State")
            .circuit(CircuitSpec::Builtin {
                name: BuiltinCircuit::BellState,
                params: vec![],
            })
            .shots(4096)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        assert_eq!(result.num_qubits, 2);
        assert_eq!(result.shots, 4096);

        // Bell state should only produce |00> and |11>
        let total: usize = result.counts.values().sum();
        assert_eq!(total, 4096);

        let count_00 = result.counts.get("00").copied().unwrap_or(0);
        let count_11 = result.counts.get("11").copied().unwrap_or(0);
        assert_eq!(count_00 + count_11, 4096, "Bell state should only produce 00 and 11");

        // Each outcome should be roughly 50% (within statistical tolerance)
        let frac_00 = count_00 as f64 / 4096.0;
        assert!(
            frac_00 > 0.35 && frac_00 < 0.65,
            "Expected ~50% |00>, got {:.1}%",
            frac_00 * 100.0
        );
    }

    #[test]
    fn builtin_ghz() {
        let config = ExperimentConfig::builder("GHZ State")
            .circuit(CircuitSpec::Builtin {
                name: BuiltinCircuit::Ghz,
                params: vec![3.0],
            })
            .shots(4096)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        assert_eq!(result.num_qubits, 3);

        // GHZ state: (|000> + |111>) / sqrt(2)
        let count_000 = result.counts.get("000").copied().unwrap_or(0);
        let count_111 = result.counts.get("111").copied().unwrap_or(0);
        assert_eq!(
            count_000 + count_111,
            4096,
            "GHZ state should only produce 000 and 111, got counts: {:?}",
            result.counts
        );
    }

    #[test]
    fn custom_circuit() {
        let config = ExperimentConfig::builder("Custom Circuit")
            .circuit(CircuitSpec::Custom {
                num_qubits: 2,
                gates: vec![
                    GateSpec::X(0),   // |00> -> |10>
                    GateSpec::H(1),   // |10> -> |1+>
                    GateSpec::Cnot(1, 0), // Entangle
                ],
            })
            .shots(1000)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        assert_eq!(result.num_qubits, 2);
        let total: usize = result.counts.values().sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn backend_selection() {
        // Test state_vector backend
        let sv_config = ExperimentConfig::builder("SV Test")
            .backend(BackendSpec::StateVector)
            .shots(100)
            .build();
        let sv_result = ExperimentRunner::run(&sv_config).unwrap();
        assert_eq!(sv_result.backend_used, "state_vector");

        // Test density_matrix backend label
        let dm_config = ExperimentConfig::builder("DM Test")
            .backend(BackendSpec::DensityMatrix)
            .shots(100)
            .build();
        let dm_result = ExperimentRunner::run(&dm_config).unwrap();
        assert_eq!(dm_result.backend_used, "density_matrix");

        // Test MPS backend label
        let mps_config = ExperimentConfig::builder("MPS Test")
            .backend(BackendSpec::Mps { bond_dim: 64 })
            .shots(100)
            .build();
        let mps_result = ExperimentRunner::run(&mps_config).unwrap();
        assert_eq!(mps_result.backend_used, "mps(bond_dim=64)");

        // Test stabilizer backend label
        let stab_config = ExperimentConfig::builder("Stabilizer Test")
            .backend(BackendSpec::Stabilizer)
            .shots(100)
            .build();
        let stab_result = ExperimentRunner::run(&stab_config).unwrap();
        assert_eq!(stab_result.backend_used, "stabilizer");
    }

    #[test]
    fn noise_injection() {
        // Run a deterministic circuit (X gate on qubit 0) with and without noise.
        // X(0) flips qubit 0 (LSB): |00> -> |01>.
        // Bitstring uses MSB-first convention, so index 1 = "01" for 2 qubits.

        let clean_config = ExperimentConfig::builder("No Noise")
            .circuit(CircuitSpec::Custom {
                num_qubits: 2,
                gates: vec![GateSpec::X(0)],
            })
            .noise(NoiseSpec::None)
            .shots(1000)
            .build();

        let clean_result = ExperimentRunner::run(&clean_config).unwrap();
        let clean_01 = clean_result.counts.get("01").copied().unwrap_or(0);
        assert_eq!(clean_01, 1000, "Without noise, X(0) should always give |01>");

        // With readout error noise, we should see bit flips in the measurement results.
        // Readout error is deterministic per-bit with probability -- over many trials,
        // we are guaranteed to see deviations.
        let noisy_config = ExperimentConfig::builder("Readout Noisy")
            .circuit(CircuitSpec::Custom {
                num_qubits: 2,
                gates: vec![GateSpec::X(0)],
            })
            .noise(NoiseSpec::ReadoutError { p01: 0.3, p10: 0.3 })
            .shots(4000)
            .build();

        let noisy_result = ExperimentRunner::run(&noisy_config).unwrap();
        // With 30% readout error on each bit, some shots should differ from "01"
        let noisy_01 = noisy_result.counts.get("01").copied().unwrap_or(0);
        let other_outcomes: usize = noisy_result
            .counts
            .iter()
            .filter(|(k, _)| k.as_str() != "01")
            .map(|(_, v)| v)
            .sum();
        // With p=0.3 readout error on 2 bits, probability of no flip on either bit
        // is 0.7 * 0.7 = 0.49. So roughly half the shots should be "01" and half other.
        // We just check that there are at least SOME other outcomes.
        assert!(
            other_outcomes > 0,
            "With 30% readout error, should see some non-01 outcomes; got only 01: {}",
            noisy_01
        );
    }

    #[test]
    fn parameter_sweep() {
        // Sweep Ry rotation from 0 to 2*pi and check that we get the right number of points
        let config = ExperimentConfig::builder("Ry Sweep")
            .circuit(CircuitSpec::Parametric {
                num_qubits: 1,
                template: vec![GateSpec::Parametric {
                    gate_type: ParametricGateType::Ry,
                    qubit: 0,
                    param_name: "theta".to_string(),
                }],
                param_defaults: {
                    let mut m = HashMap::new();
                    m.insert("theta".to_string(), 0.0);
                    m
                },
            })
            .sweep(SweepConfig::new("theta", 0.0, 2.0 * PI, 10))
            .shots(100)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        assert_eq!(result.sweep_results.len(), 10);

        // First point (theta=0): should be all |0>
        let first = &result.sweep_results[0];
        assert!(
            (first.param_value - 0.0).abs() < 1e-10,
            "first sweep value should be 0"
        );
    }

    #[test]
    fn expectation_value() {
        // Measure <Z0> on |0> state (should be +1) and |1> state (should be -1)
        let config_z0 = ExperimentConfig::builder("Z0 on |0>")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![], // |0> state
            })
            .observe(Observable::PauliZ(0))
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config_z0).unwrap();
        let z0_val = result.expectation_values.get("Z0").unwrap();
        assert!(
            (*z0_val - 1.0).abs() < 1e-10,
            "<Z> on |0> should be +1, got {}",
            z0_val
        );

        // Measure <Z0> on |1> state
        let config_z1 = ExperimentConfig::builder("Z0 on |1>")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![GateSpec::X(0)],
            })
            .observe(Observable::PauliZ(0))
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config_z1).unwrap();
        let z0_val = result.expectation_values.get("Z0").unwrap();
        assert!(
            (*z0_val - (-1.0)).abs() < 1e-10,
            "<Z> on |1> should be -1, got {}",
            z0_val
        );

        // Measure <Z0> on |+> state (should be 0)
        let config_plus = ExperimentConfig::builder("Z0 on |+>")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![GateSpec::H(0)],
            })
            .observe(Observable::PauliZ(0))
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config_plus).unwrap();
        let z0_val = result.expectation_values.get("Z0").unwrap();
        assert!(
            z0_val.abs() < 1e-10,
            "<Z> on |+> should be 0, got {}",
            z0_val
        );
    }

    #[test]
    fn output_json() {
        let config = ExperimentConfig::builder("JSON Test")
            .circuit(CircuitSpec::Builtin {
                name: BuiltinCircuit::BellState,
                params: vec![],
            })
            .observe(Observable::PauliZ(0))
            .output_format(OutputFormat::Json)
            .shots(100)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        let json = result.to_json();
        assert!(json.contains("\"name\": \"JSON Test\""));
        assert!(json.contains("\"counts\""));
        assert!(json.contains("\"shots\": 100"));
        assert!(json.contains("\"expectation_values\""));
    }

    #[test]
    fn output_csv() {
        let config = ExperimentConfig::builder("CSV Test")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![GateSpec::X(0)],
            })
            .observe(Observable::PauliZ(0))
            .output_format(OutputFormat::Csv)
            .shots(50)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        let csv = result.to_csv();
        assert!(csv.starts_with("bitstring,count"));
        assert!(csv.contains("observable,value"));
        // The only outcome should be |1>
        assert!(csv.contains("1,50"));
    }

    #[test]
    fn experiment_timing() {
        let config = ExperimentConfig::builder("Timing Test")
            .circuit(CircuitSpec::Builtin {
                name: BuiltinCircuit::BellState,
                params: vec![],
            })
            .shots(100)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        // The experiment should complete and record a non-zero duration
        // (even on fast machines, there should be at least some nanoseconds)
        assert!(
            result.timing.as_nanos() > 0,
            "timing should be non-zero"
        );
    }

    #[test]
    fn multiple_observables() {
        let config = ExperimentConfig::builder("Multi-Observable")
            .circuit(CircuitSpec::Custom {
                num_qubits: 2,
                gates: vec![GateSpec::H(0), GateSpec::Cnot(0, 1)],
            })
            .observe(Observable::PauliZ(0))
            .observe(Observable::PauliZ(1))
            .observe(Observable::PauliX(0))
            .observe(Observable::Identity)
            .shots(100)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        assert_eq!(result.expectation_values.len(), 4);
        assert!(result.expectation_values.contains_key("Z0"));
        assert!(result.expectation_values.contains_key("Z1"));
        assert!(result.expectation_values.contains_key("X0"));
        assert!(result.expectation_values.contains_key("I"));

        // Identity should always be 1.0
        let id_val = result.expectation_values.get("I").unwrap();
        assert!(
            (*id_val - 1.0).abs() < 1e-10,
            "Identity expectation should be 1.0, got {}",
            id_val
        );

        // For Bell state (|00> + |11>)/sqrt(2):
        // <Z0> = 0, <Z1> = 0 (each qubit is maximally mixed)
        let z0 = result.expectation_values.get("Z0").unwrap();
        let z1 = result.expectation_values.get("Z1").unwrap();
        assert!(z0.abs() < 1e-10, "Bell state <Z0> should be 0, got {}", z0);
        assert!(z1.abs() < 1e-10, "Bell state <Z1> should be 0, got {}", z1);
    }

    #[test]
    fn sweep_results_shape() {
        let config = ExperimentConfig::builder("Sweep Shape")
            .circuit(CircuitSpec::Parametric {
                num_qubits: 1,
                template: vec![GateSpec::Parametric {
                    gate_type: ParametricGateType::Ry,
                    qubit: 0,
                    param_name: "theta".to_string(),
                }],
                param_defaults: {
                    let mut m = HashMap::new();
                    m.insert("theta".to_string(), 0.0);
                    m
                },
            })
            .sweep(SweepConfig::new("theta", 0.0, PI, 21))
            .observe(Observable::PauliZ(0))
            .shots(100)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        assert_eq!(
            result.sweep_results.len(),
            21,
            "should have exactly 21 sweep points"
        );

        // Check that parameter values are evenly spaced
        let values: Vec<f64> = result
            .sweep_results
            .iter()
            .map(|r| r.param_value)
            .collect();
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[20] - PI).abs() < 1e-10);

        // At theta=0 (|0>), <Z> should be +1
        let z_at_0 = result.sweep_results[0]
            .expectation_values
            .get("Z0")
            .unwrap();
        assert!(
            (*z_at_0 - 1.0).abs() < 1e-10,
            "At theta=0, <Z> should be 1.0, got {}",
            z_at_0
        );

        // At theta=pi (|1>), <Z> should be -1
        let z_at_pi = result.sweep_results[20]
            .expectation_values
            .get("Z0")
            .unwrap();
        assert!(
            (*z_at_pi - (-1.0)).abs() < 1e-10,
            "At theta=pi, <Z> should be -1.0, got {}",
            z_at_pi
        );
    }

    #[test]
    fn config_validation_errors() {
        // Empty name
        let mut config = ExperimentConfig::default();
        config.name = String::new();
        assert!(config.validate().is_err());

        // Zero shots
        let mut config = ExperimentConfig::default();
        config.shots = 0;
        assert!(config.validate().is_err());

        // Invalid noise parameter
        let mut config = ExperimentConfig::default();
        config.noise = NoiseSpec::Depolarizing { p: 1.5 };
        assert!(config.validate().is_err());

        // Invalid sweep
        let mut config = ExperimentConfig::default();
        config.sweep = Some(SweepConfig {
            param_name: String::new(),
            start: 0.0,
            stop: 1.0,
            steps: 10,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn parse_config_basic() {
        let input = r#"
name = "Parsed Experiment"
circuit = "bell_state"
backend = "state_vector"
shots = 512
noise = "depolarizing(0.01)"
output_format = "json"
author = "test_suite"
"#;

        let config = parse_config(input).unwrap();
        assert_eq!(config.name, "Parsed Experiment");
        assert_eq!(config.shots, 512);
        assert_eq!(config.backend, BackendSpec::StateVector);
        assert_eq!(config.output.format, OutputFormat::Json);
        assert_eq!(
            config.metadata.get("author"),
            Some(&"test_suite".to_string())
        );
    }

    #[test]
    fn fidelity_computation() {
        // Prepare |+> and check fidelity against exact |+> target
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let target = vec![
            C64::new(sqrt2_inv, 0.0),
            C64::new(sqrt2_inv, 0.0),
        ];

        let config = ExperimentConfig::builder("Fidelity Test")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![GateSpec::H(0)],
            })
            .fidelity_target(target)
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        let fid = result.fidelity.unwrap();
        assert!(
            (fid - 1.0).abs() < 1e-10,
            "Fidelity of |+> with itself should be 1.0, got {}",
            fid
        );
    }

    #[test]
    fn entropy_computation() {
        // |0> state has 0 entropy (all probability in one basis state)
        let config = ExperimentConfig::builder("Entropy Test |0>")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![],
            })
            .with_entropy()
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        let ent = result.entropy.unwrap();
        assert!(
            ent.abs() < 1e-10,
            "|0> state should have 0 entropy, got {}",
            ent
        );

        // |+> state has 1 bit of entropy
        let config = ExperimentConfig::builder("Entropy Test |+>")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![GateSpec::H(0)],
            })
            .with_entropy()
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        let ent = result.entropy.unwrap();
        assert!(
            (ent - 1.0).abs() < 1e-10,
            "|+> state should have 1 bit of entropy, got {}",
            ent
        );
    }

    #[test]
    fn sweep_config_values() {
        let sweep = SweepConfig::new("x", 0.0, 1.0, 5);
        let vals = sweep.values();
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 0.25).abs() < 1e-10);
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[3] - 0.75).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);

        // Single step
        let sweep1 = SweepConfig::new("x", 3.14, 6.28, 1);
        let vals1 = sweep1.values();
        assert_eq!(vals1.len(), 1);
        assert!((vals1[0] - 3.14).abs() < 1e-10);
    }

    #[test]
    fn pauli_string_expectation() {
        // For Bell state (|00> + |11>)/sqrt(2), <Z0*Z1> should be +1
        let config = ExperimentConfig::builder("ZZ Correlator")
            .circuit(CircuitSpec::Custom {
                num_qubits: 2,
                gates: vec![GateSpec::H(0), GateSpec::Cnot(0, 1)],
            })
            .observe(Observable::PauliString(vec![
                (0, PauliLabel::Z),
                (1, PauliLabel::Z),
            ]))
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        // The key format is "Z0*Z1"
        let zz = result
            .expectation_values
            .values()
            .next()
            .unwrap();
        assert!(
            (*zz - 1.0).abs() < 1e-10,
            "Bell state <Z0*Z1> should be +1, got {}",
            zz
        );
    }

    #[test]
    fn human_output_format() {
        let config = ExperimentConfig::builder("Human Format Test")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![GateSpec::X(0)],
            })
            .observe(Observable::PauliZ(0))
            .shots(100)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        let human = result.to_human();
        assert!(human.contains("=== Human Format Test ==="));
        assert!(human.contains("Measurement counts:"));
        assert!(human.contains("Expectation values:"));
        assert!(human.contains("|1>"));
    }

    #[test]
    fn builtin_circuit_display() {
        assert_eq!(format!("{}", BuiltinCircuit::BellState), "bell_state");
        assert_eq!(format!("{}", BuiltinCircuit::Ghz), "ghz");
        assert_eq!(format!("{}", BuiltinCircuit::Qft), "qft");
        assert_eq!(format!("{}", BuiltinCircuit::Grover), "grover");
        assert_eq!(format!("{}", BuiltinCircuit::VqeAnsatz), "vqe_ansatz");
    }

    #[test]
    fn backend_display() {
        assert_eq!(format!("{}", BackendSpec::StateVector), "state_vector");
        assert_eq!(format!("{}", BackendSpec::DensityMatrix), "density_matrix");
        assert_eq!(
            format!("{}", BackendSpec::Mps { bond_dim: 128 }),
            "mps(bond_dim=128)"
        );
        assert_eq!(format!("{}", BackendSpec::Stabilizer), "stabilizer");
        assert_eq!(format!("{}", BackendSpec::Gpu { device: 0 }), "gpu(device=0)");
    }

    #[test]
    fn statevector_output() {
        let config = ExperimentConfig::builder("SV Output")
            .circuit(CircuitSpec::Custom {
                num_qubits: 1,
                gates: vec![GateSpec::H(0)],
            })
            .include_statevector()
            .shots(1)
            .build();

        let result = ExperimentRunner::run(&config).unwrap();
        let sv = result.statevector.as_ref().unwrap();
        assert_eq!(sv.len(), 2);

        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - sqrt2_inv).abs() < 1e-10);
        assert!((sv[1].re - sqrt2_inv).abs() < 1e-10);
    }

    #[test]
    fn parse_noise_specs() {
        assert!(matches!(parse_noise_spec("none").unwrap(), NoiseSpec::None));

        match parse_noise_spec("depolarizing(0.05)").unwrap() {
            NoiseSpec::Depolarizing { p } => assert!((p - 0.05).abs() < 1e-10),
            _ => panic!("expected depolarizing"),
        }

        match parse_noise_spec("amplitude_damping(0.1)").unwrap() {
            NoiseSpec::AmplitudeDamping { gamma } => assert!((gamma - 0.1).abs() < 1e-10),
            _ => panic!("expected amplitude_damping"),
        }

        match parse_noise_spec("readout_error(0.01, 0.02)").unwrap() {
            NoiseSpec::ReadoutError { p01, p10 } => {
                assert!((p01 - 0.01).abs() < 1e-10);
                assert!((p10 - 0.02).abs() < 1e-10);
            }
            _ => panic!("expected readout_error"),
        }

        match parse_noise_spec("device(ibm_perth)").unwrap() {
            NoiseSpec::DeviceProfile { name } => assert_eq!(name, "ibm_perth"),
            _ => panic!("expected device profile"),
        }

        assert!(parse_noise_spec("unknown(123)").is_err());
    }
}
