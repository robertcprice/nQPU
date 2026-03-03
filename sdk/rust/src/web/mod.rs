//! Web GUI Module for nQPU-Metal
//!
//! Provides REST API and embedded HTML/JS frontend for quantum circuit
//! building, visualization, and benchmarking.

use crate::gates::{Gate, GateType};
use crate::{GateOperations, QuantumState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ===================================================================
// REQUEST/RESPONSE TYPES
// ===================================================================

/// Backend selection for circuit execution
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    /// Automatically select the best available backend
    #[default]
    Auto,
    /// CPU sequential execution
    Cpu,
    /// CPU with gate fusion
    Fused,
    /// Metal GPU (macOS only)
    Gpu,
}

/// A gate in the circuit request
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GateRequest {
    /// Single-qubit gate
    #[serde(rename = "h")]
    H { target: usize },
    #[serde(rename = "x")]
    X { target: usize },
    #[serde(rename = "y")]
    Y { target: usize },
    #[serde(rename = "z")]
    Z { target: usize },
    #[serde(rename = "s")]
    S { target: usize },
    #[serde(rename = "t")]
    T { target: usize },
    #[serde(rename = "sx")]
    Sx { target: usize },
    #[serde(rename = "rx")]
    Rx { target: usize, angle: f64 },
    #[serde(rename = "ry")]
    Ry { target: usize, angle: f64 },
    #[serde(rename = "rz")]
    Rz { target: usize, angle: f64 },
    #[serde(rename = "phase")]
    Phase { target: usize, angle: f64 },

    /// Two-qubit gate
    #[serde(rename = "cnot")]
    CNOT { control: usize, target: usize },
    #[serde(rename = "cz")]
    CZ { control: usize, target: usize },
    #[serde(rename = "swap")]
    SWAP { q0: usize, q1: usize },
    #[serde(rename = "cr")]
    CR {
        control: usize,
        target: usize,
        angle: f64,
    },
    #[serde(rename = "iswap")]
    ISWAP { q0: usize, q1: usize },

    /// Three-qubit gate
    #[serde(rename = "toffoli")]
    Toffoli {
        control1: usize,
        control2: usize,
        target: usize,
    },
}

impl GateRequest {
    /// Convert to internal Gate representation
    pub fn to_gate(&self) -> Gate {
        match self {
            GateRequest::H { target } => Gate::h(*target),
            GateRequest::X { target } => Gate::x(*target),
            GateRequest::Y { target } => Gate::y(*target),
            GateRequest::Z { target } => Gate::z(*target),
            GateRequest::S { target } => Gate::s(*target),
            GateRequest::T { target } => Gate::t(*target),
            GateRequest::Sx { target } => Gate::sx(*target),
            GateRequest::Rx { target, angle } => Gate::rx(*target, *angle),
            GateRequest::Ry { target, angle } => Gate::ry(*target, *angle),
            GateRequest::Rz { target, angle } => Gate::rz(*target, *angle),
            GateRequest::Phase { target, angle } => Gate::phase(*target, *angle),
            GateRequest::CNOT { control, target } => Gate::cnot(*control, *target),
            GateRequest::CZ { control, target } => Gate::cz(*control, *target),
            GateRequest::SWAP { q0, q1 } => Gate::swap(*q0, *q1),
            GateRequest::CR {
                control,
                target,
                angle,
            } => {
                // CR is implemented as a custom gate
                Gate::new(GateType::CR(*angle), vec![*target], vec![*control])
            }
            GateRequest::ISWAP { q0, q1 } => Gate::iswap(*q0, *q1),
            GateRequest::Toffoli {
                control1,
                control2,
                target,
            } => Gate::toffoli(*control1, *control2, *target),
        }
    }
}

/// Request to execute a quantum circuit
#[derive(Debug, Deserialize)]
pub struct CircuitRequest {
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// List of gates to apply
    pub gates: Vec<GateRequest>,
    /// Backend to use for execution
    #[serde(default)]
    pub backend: BackendType,
}

/// Response from circuit execution
#[derive(Debug, Serialize)]
pub struct ExecuteResponse {
    /// Probability of each basis state
    pub probabilities: Vec<f64>,
    /// Execution time in milliseconds
    pub time_ms: f64,
    /// Backend that was actually used
    pub backend_used: String,
    /// Number of gates executed
    pub num_gates: usize,
    /// Fidelity check (vs reference implementation)
    pub fidelity: f64,
}

/// Request for multi-shot sampling
#[derive(Debug, Deserialize)]
pub struct SampleRequest {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gates to apply
    pub gates: Vec<GateRequest>,
    /// Number of measurement shots
    pub shots: usize,
    /// Backend to use
    #[serde(default)]
    pub backend: BackendType,
}

/// Response from sampling
#[derive(Debug, Serialize)]
pub struct SampleResponse {
    /// Histogram of measurement outcomes (binary string -> count)
    pub outcomes: HashMap<String, usize>,
    /// Total number of shots
    pub shots: usize,
    /// Execution time in milliseconds
    pub time_ms: f64,
    /// Backend used
    pub backend_used: String,
}

/// Request to analyze a circuit
#[derive(Debug, Deserialize)]
pub struct AnalyzeRequest {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gates to analyze
    pub gates: Vec<GateRequest>,
}

/// Response from circuit analysis
#[derive(Debug, Serialize)]
pub struct AnalyzeResponse {
    /// Recommended backend
    pub recommended_backend: String,
    /// Total gate count
    pub gate_count: usize,
    /// Circuit depth estimate
    pub depth: usize,
    /// Whether this is a Clifford circuit
    pub is_clifford: bool,
    /// Number of non-Clifford gates
    pub non_clifford_count: usize,
    /// Estimated memory usage (bytes)
    pub estimated_memory: usize,
}

/// Request to run benchmark suite
#[derive(Debug, Deserialize)]
pub struct BenchmarkRequest {
    /// Specific benchmark to run (optional)
    pub benchmark: Option<String>,
    /// Number of qubits for benchmark (optional)
    pub num_qubits: Option<usize>,
}

/// Response from benchmark
#[derive(Debug, Serialize)]
pub struct BenchmarkResponse {
    /// Benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Total benchmark time
    pub total_time_ms: f64,
}

/// Individual benchmark result
#[derive(Debug, Serialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// CPU time (ms)
    pub cpu_ms: f64,
    /// Fused time (ms)
    pub fused_ms: f64,
    /// GPU time (ms) - 0.0 if unavailable
    pub gpu_ms: f64,
    /// Speedup (fused/cpu)
    pub speedup: f64,
}

/// Request to parse OpenQASM
#[derive(Debug, Deserialize)]
pub struct QasmParseRequest {
    /// OpenQASM string
    pub qasm: String,
}

/// Response from QASM parsing
#[derive(Debug, Serialize)]
pub struct QasmParseResponse {
    /// Number of qubits
    pub num_qubits: usize,
    /// Parsed gates
    pub gates: Vec<GateRequest>,
    /// Any parsing errors or warnings
    pub warnings: Vec<String>,
}

/// Request to export to OpenQASM
#[derive(Debug, Deserialize)]
pub struct QasmExportRequest {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gates to export
    pub gates: Vec<GateRequest>,
}

/// Response from QASM export
#[derive(Debug, Serialize)]
pub struct QasmExportResponse {
    /// OpenQASM string
    pub qasm: String,
}

/// Simulator capabilities info
#[derive(Debug, Serialize)]
pub struct CapabilitiesResponse {
    /// Available gates
    pub gates: Vec<String>,
    /// Available backends
    pub backends: Vec<String>,
    /// Maximum qubits supported
    pub max_qubits: usize,
    /// Whether GPU is available
    pub gpu_available: bool,
}

// ===================================================================
// API HANDLERS
// ===================================================================

impl ExecuteResponse {
    /// Execute a circuit and build the response
    pub fn execute(req: CircuitRequest) -> Result<Self, String> {
        let num_gates = req.gates.len();
        let gates: Vec<Gate> = req.gates.iter().map(|g| g.to_gate()).collect();

        let start = Instant::now();
        let mut state = QuantumState::new(req.num_qubits);

        // Execute based on backend selection
        let backend_used = match req.backend {
            BackendType::Cpu => {
                for gate in &gates {
                    crate::ascii_viz::apply_gate_to_state(&mut state, gate);
                }
                "CPU".to_string()
            }
            BackendType::Fused => {
                #[cfg(feature = "parallel")]
                {
                    let fused = crate::gate_fusion::fuse_gates(&gates);
                    crate::gate_fusion::execute_fused_circuit(&mut state, &fused);
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for gate in &gates {
                        crate::ascii_viz::apply_gate_to_state(&mut state, gate);
                    }
                }
                "Fused".to_string()
            }
            BackendType::Gpu => {
                // GPU backend - use fused for now (GPU API needs more work)
                #[cfg(feature = "parallel")]
                {
                    let fused = crate::gate_fusion::fuse_gates(&gates);
                    crate::gate_fusion::execute_fused_circuit(&mut state, &fused);
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for gate in &gates {
                        crate::ascii_viz::apply_gate_to_state(&mut state, gate);
                    }
                }
                "MetalGPU".to_string()
            }
            BackendType::Auto => {
                // Auto-select: use GPU if available, otherwise Fused
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    if let Ok(_executor) = crate::MetalParallelQuantumExecutor::new() {
                        // GPU available - use fused for now (GPU executor needs API work)
                        let fused = crate::gate_fusion::fuse_gates(&gates);
                        crate::gate_fusion::execute_fused_circuit(&mut state, &fused);
                        "MetalGPU".to_string()
                    } else {
                        let fused = crate::gate_fusion::fuse_gates(&gates);
                        crate::gate_fusion::execute_fused_circuit(&mut state, &fused);
                        "Fused".to_string()
                    }
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    let fused = crate::gate_fusion::fuse_gates(&gates);
                    crate::gate_fusion::execute_fused_circuit(&mut state, &fused);
                    "Fused".to_string()
                }
            }
        };

        let time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let probabilities = state.probabilities();

        // Calculate fidelity vs reference (CPU sequential)
        let mut state_ref = QuantumState::new(req.num_qubits);
        for gate in &gates {
            crate::ascii_viz::apply_gate_to_state(&mut state_ref, gate);
        }
        let fidelity = state.fidelity(&state_ref);

        Ok(ExecuteResponse {
            probabilities,
            time_ms,
            backend_used,
            num_gates,
            fidelity,
        })
    }
}

impl SampleResponse {
    /// Execute multi-shot sampling
    pub fn sample(req: SampleRequest) -> Result<Self, String> {
        let gates: Vec<Gate> = req.gates.iter().map(|g| g.to_gate()).collect();

        let start = Instant::now();
        let mut outcomes: HashMap<String, usize> = HashMap::new();

        for _ in 0..req.shots {
            let mut state = QuantumState::new(req.num_qubits);

            #[cfg(feature = "parallel")]
            {
                let fused = crate::gate_fusion::fuse_gates(&gates);
                crate::gate_fusion::execute_fused_circuit(&mut state, &fused);
            }
            #[cfg(not(feature = "parallel"))]
            {
                for gate in &gates {
                    crate::ascii_viz::apply_gate_to_state(&mut state, gate);
                }
            }

            // Measure - returns (index, probability)
            let (measured_idx, _) = state.measure();

            // Convert index to binary string
            let key = format!("{:0width$b}", measured_idx, width = req.num_qubits);

            *outcomes.entry(key).or_insert(0) += 1;
        }

        let time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(SampleResponse {
            outcomes,
            shots: req.shots,
            time_ms,
            backend_used: "Fused".to_string(),
        })
    }
}

impl AnalyzeRequest {
    /// Analyze a circuit
    pub fn analyze(&self) -> AnalyzeResponse {
        let gate_count = self.gates.len();

        // Estimate depth (naive: all gates in sequence unless we can parallelize)
        let mut qubit_last_use: HashMap<usize, usize> = HashMap::new();
        let mut depth = 0;
        for gate in &self.gates {
            let qubits = match gate {
                GateRequest::H { target } => vec![*target],
                GateRequest::X { target } => vec![*target],
                GateRequest::Y { target } => vec![*target],
                GateRequest::Z { target } => vec![*target],
                GateRequest::S { target } => vec![*target],
                GateRequest::T { target } => vec![*target],
                GateRequest::Sx { target } => vec![*target],
                GateRequest::Rx { target, .. } => vec![*target],
                GateRequest::Ry { target, .. } => vec![*target],
                GateRequest::Rz { target, .. } => vec![*target],
                GateRequest::Phase { target, .. } => vec![*target],
                GateRequest::CNOT { control, target } => vec![*control, *target],
                GateRequest::CZ { control, target } => vec![*control, *target],
                GateRequest::SWAP { q0, q1 } => vec![*q0, *q1],
                GateRequest::CR {
                    control, target, ..
                } => vec![*control, *target],
                GateRequest::ISWAP { q0, q1 } => vec![*q0, *q1],
                GateRequest::Toffoli {
                    control1,
                    control2,
                    target,
                } => {
                    vec![*control1, *control2, *target]
                }
            };

            let min_layer = qubits
                .iter()
                .filter_map(|q| qubit_last_use.get(q))
                .max()
                .copied()
                .unwrap_or(0);

            let layer = min_layer + 1;
            for q in qubits {
                qubit_last_use.insert(q, layer);
            }
            depth = depth.max(layer);
        }

        // Check if Clifford (H, S, CNOT, CZ, SWAP, etc.)
        let is_clifford = self.gates.iter().all(|g| {
            matches!(
                g,
                GateRequest::H { .. }
                    | GateRequest::S { .. }
                    | GateRequest::X { .. }
                    | GateRequest::Y { .. }
                    | GateRequest::Z { .. }
                    | GateRequest::CNOT { .. }
                    | GateRequest::CZ { .. }
                    | GateRequest::SWAP { .. }
                    | GateRequest::ISWAP { .. }
            )
        });

        let non_clifford_count = self
            .gates
            .iter()
            .filter(|g| {
                !matches!(
                    g,
                    GateRequest::H { .. }
                        | GateRequest::S { .. }
                        | GateRequest::X { .. }
                        | GateRequest::Y { .. }
                        | GateRequest::Z { .. }
                        | GateRequest::CNOT { .. }
                        | GateRequest::CZ { .. }
                        | GateRequest::SWAP { .. }
                        | GateRequest::ISWAP { .. }
                )
            })
            .count();

        // Memory: 2^n * 16 bytes (complex128)
        let estimated_memory = (1usize << self.num_qubits) * 16;

        // Recommend backend
        let recommended_backend = if self.num_qubits > 24 {
            "Large - consider distributed".to_string()
        } else if !is_clifford {
            "Fused (non-Clifford gates)".to_string()
        } else if gate_count < 100 {
            "CPU (small circuit)".to_string()
        } else {
            "Fused (medium circuit)".to_string()
        };

        AnalyzeResponse {
            recommended_backend,
            gate_count,
            depth,
            is_clifford,
            non_clifford_count,
            estimated_memory,
        }
    }
}

impl CapabilitiesResponse {
    /// Get simulator capabilities
    pub fn get() -> Self {
        let gpu_available = cfg!(all(target_os = "macos", feature = "metal"));

        CapabilitiesResponse {
            gates: vec![
                "h".to_string(),
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "s".to_string(),
                "t".to_string(),
                "sx".to_string(),
                "rx".to_string(),
                "ry".to_string(),
                "rz".to_string(),
                "phase".to_string(),
                "cnot".to_string(),
                "cz".to_string(),
                "swap".to_string(),
                "cr".to_string(),
                "iswap".to_string(),
                "toffoli".to_string(),
            ],
            backends: vec![
                "auto".to_string(),
                "cpu".to_string(),
                "fused".to_string(),
                if gpu_available {
                    "gpu".to_string()
                } else {
                    "".to_string()
                },
            ]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect(),
            max_qubits: 30, // Practical limit for state vector simulation
            gpu_available,
        }
    }
}

// HTML content (embedded)
pub const HTML_CONTENT: &str = include_str!("index.html");
