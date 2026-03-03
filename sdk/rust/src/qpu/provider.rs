//! QPU provider and job traits — the unified interface for all quantum hardware.

use async_trait::async_trait;
use std::time::Duration;

use crate::qpu::error::QPUError;
use crate::qpu::job::{BackendInfo, CostEstimate, JobConfig, JobResult, JobStatus, ValidationReport};
use crate::qpu::QPUCircuit;

/// Unified trait for all quantum hardware providers.
///
/// Implementations exist for IBM Quantum, Amazon Braket, Azure Quantum,
/// IonQ Direct, and Google Quantum AI. All methods are async since they
/// involve network I/O.
///
/// # Example
/// ```no_run
/// use nqpu_metal::qpu::{IBMProvider, QPUProvider, QPUCircuit, JobConfig};
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = IBMProvider::from_env()?;
/// let backends = provider.list_backends().await?;
/// println!("Available backends: {:?}", backends.len());
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait QPUProvider: Send + Sync {
    /// Human-readable provider name (e.g., "IBM Quantum", "Amazon Braket")
    fn name(&self) -> &str;

    /// List all available backends/devices from this provider.
    async fn list_backends(&self) -> Result<Vec<BackendInfo>, QPUError>;

    /// Get detailed info about a specific backend by name.
    async fn get_backend(&self, name: &str) -> Result<BackendInfo, QPUError>;

    /// Submit a circuit for execution on quantum hardware.
    ///
    /// Returns a `QPUJob` handle that can be used to poll status and retrieve results.
    async fn submit_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        config: &JobConfig,
    ) -> Result<Box<dyn QPUJob>, QPUError>;

    /// Estimate the cost of running a circuit.
    ///
    /// Returns `None` if the provider doesn't expose cost information.
    async fn estimate_cost(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        shots: usize,
    ) -> Result<Option<CostEstimate>, QPUError>;

    /// Validate a circuit against a specific backend's constraints.
    ///
    /// Checks qubit count, gate support, connectivity, and depth limits.
    fn validate_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &BackendInfo,
    ) -> Result<ValidationReport, QPUError>;
}

/// Handle for a submitted quantum job.
///
/// Provides methods to poll status, wait for completion, retrieve results,
/// and cancel the job.
#[async_trait]
pub trait QPUJob: Send + Sync {
    /// Provider-assigned job identifier.
    fn id(&self) -> &str;

    /// Backend name where the job is running.
    fn backend(&self) -> &str;

    /// Provider name.
    fn provider(&self) -> &str;

    /// Get the current job status.
    async fn status(&self) -> Result<JobStatus, QPUError>;

    /// Poll until the job reaches a terminal state (completed, failed, or cancelled).
    ///
    /// Uses exponential backoff starting from `poll_interval`, with a maximum
    /// wait of `timeout`. Returns the final result on success.
    async fn wait_for_completion(
        &self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<JobResult, QPUError>;

    /// Get the result if the job is already complete.
    ///
    /// Returns `None` if the job hasn't finished yet.
    async fn result(&self) -> Result<Option<JobResult>, QPUError>;

    /// Cancel the job.
    async fn cancel(&self) -> Result<(), QPUError>;
}
