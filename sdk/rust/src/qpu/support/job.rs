//! Job configuration, status, and result types for QPU operations.

use std::collections::HashMap;
use std::time::Duration;

/// Configuration for submitting a quantum job.
#[derive(Debug, Clone)]
pub struct JobConfig {
    /// Number of measurement shots
    pub shots: usize,
    /// Transpiler optimization level (0=none, 1=light, 2=medium, 3=heavy)
    pub optimization_level: u8,
    /// Whether to auto-transpile for the target backend
    pub auto_transpile: bool,
    /// Maximum time to wait in queue before cancelling
    pub max_queue_time: Option<Duration>,
    /// Job name/tag for identification
    pub name: Option<String>,
    /// Provider-specific options
    pub extra: HashMap<String, String>,
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            shots: 1024,
            optimization_level: 1,
            auto_transpile: true,
            max_queue_time: None,
            name: None,
            extra: HashMap::new(),
        }
    }
}

/// Current status of a quantum job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Job has been submitted but not yet queued
    Initializing,
    /// Job is waiting in queue
    Queued,
    /// Job is being validated/transpiled
    Validating,
    /// Job is executing on quantum hardware
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed during execution
    Failed(String),
    /// Job was cancelled by user
    Cancelled,
}

impl JobStatus {
    /// Returns true if the job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobStatus::Completed | JobStatus::Failed(_) | JobStatus::Cancelled
        )
    }

    /// Returns true if the job completed successfully.
    pub fn is_success(&self) -> bool {
        matches!(self, JobStatus::Completed)
    }
}

/// Result from a completed quantum job.
#[derive(Debug, Clone)]
pub struct JobResult {
    /// Job identifier
    pub job_id: String,
    /// Backend that executed the job
    pub backend: String,
    /// Measurement counts: bitstring → count
    pub counts: HashMap<String, usize>,
    /// Total number of shots executed
    pub shots: usize,
    /// Whether the results are from a simulator
    pub is_simulator: bool,
    /// Execution time on the quantum device
    pub execution_time: Option<Duration>,
    /// Time spent in queue
    pub queue_time: Option<Duration>,
    /// Raw provider-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl JobResult {
    /// Get the most frequently measured bitstring.
    pub fn most_frequent(&self) -> Option<(&str, usize)> {
        self.counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(bitstring, count)| (bitstring.as_str(), *count))
    }

    /// Get measurement probabilities (counts / total shots).
    pub fn probabilities(&self) -> HashMap<String, f64> {
        let total = self.shots as f64;
        self.counts
            .iter()
            .map(|(k, v)| (k.clone(), *v as f64 / total))
            .collect()
    }
}

/// Cost estimate for a quantum job.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Provider name
    pub provider: String,
    /// Backend name
    pub backend: String,
    /// Number of shots
    pub shots: usize,
    /// Estimated cost in USD
    pub estimated_cost_usd: f64,
    /// Estimated queue wait time
    pub estimated_queue_time: Option<Duration>,
    /// Whether the backend is free tier
    pub is_free_tier: bool,
    /// Additional cost details
    pub details: Option<String>,
}

/// Information about a quantum backend/device.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend identifier (e.g., "ibm_brisbane", "ionq_aria")
    pub name: String,
    /// Provider name (e.g., "IBM Quantum", "Amazon Braket")
    pub provider: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Whether this is a simulator
    pub is_simulator: bool,
    /// Backend status (online/offline/maintenance)
    pub status: BackendStatus,
    /// Number of jobs in queue
    pub queue_length: usize,
    /// Supported basis gates
    pub basis_gates: Vec<String>,
    /// Qubit connectivity (pairs of connected qubits)
    pub coupling_map: Option<Vec<(usize, usize)>>,
    /// Maximum number of shots per job
    pub max_shots: usize,
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Average single-qubit gate error rate
    pub avg_gate_error_1q: Option<f64>,
    /// Average two-qubit gate error rate
    pub avg_gate_error_2q: Option<f64>,
    /// Average readout error rate
    pub avg_readout_error: Option<f64>,
    /// Average T1 relaxation time in microseconds
    pub avg_t1_us: Option<f64>,
    /// Average T2 dephasing time in microseconds
    pub avg_t2_us: Option<f64>,
    /// Provider-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Backend availability status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendStatus {
    Online,
    Offline,
    Maintenance,
    Retired,
}

/// Validation report for a circuit against a backend.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the circuit passed validation (no errors)
    pub is_valid: bool,
    /// Critical errors that prevent submission
    pub errors: Vec<String>,
    /// Non-critical warnings
    pub warnings: Vec<String>,
    /// Estimated circuit fidelity (0.0 - 1.0) based on noise model
    pub estimated_fidelity: Option<f64>,
}
