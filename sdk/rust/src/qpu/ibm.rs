//! IBM Quantum REST API provider.
//!
//! Connects to IBM Quantum Platform via the v2 REST API at
//! `https://api.quantum-computing.ibm.com/`. Authentication uses a Bearer
//! token read from the `IBM_QUANTUM_TOKEN` environment variable.
//!
//! # Endpoints
//!
//! | Method | Path                         | Purpose            |
//! |--------|------------------------------|--------------------|
//! | GET    | `/api/backends`              | List backends      |
//! | POST   | `/api/jobs`                  | Submit job         |
//! | GET    | `/api/jobs/{id}`             | Poll job status    |
//! | GET    | `/api/jobs/{id}/results`     | Retrieve results   |
//! | POST   | `/api/jobs/{id}/cancel`      | Cancel a job       |

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::qpu::error::QPUError;
use crate::qpu::job::{
    BackendInfo, BackendStatus, CostEstimate, JobConfig, JobResult, JobStatus, ValidationReport,
};
use crate::qpu::provider::{QPUJob, QPUProvider};
use crate::qpu::validation::CircuitValidator;
use crate::qpu::QPUCircuit;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const IBM_BASE_URL: &str = "https://api.quantum-computing.ibm.com";
const IBM_PROVIDER_NAME: &str = "IBM Quantum";
const MAX_BACKOFF_SECS: u64 = 60;

// ---------------------------------------------------------------------------
// IBMProvider
// ---------------------------------------------------------------------------

/// IBM Quantum REST API provider.
///
/// Submits circuits as OpenQASM 2.0 strings via the IBM Quantum Platform v2
/// API.  Requires a valid API token which can be obtained from
/// <https://quantum.ibm.com/>.
///
/// # Construction
///
/// ```no_run
/// # use nqpu_metal::qpu::ibm::IBMProvider;
/// // From an explicit token
/// let provider = IBMProvider::new("my_token".into());
///
/// // From the IBM_QUANTUM_TOKEN environment variable
/// let provider = IBMProvider::from_env().expect("IBM_QUANTUM_TOKEN not set");
/// ```
pub struct IBMProvider {
    client: reqwest::Client,
    token: String,
    base_url: String,
}

impl IBMProvider {
    /// Create a new IBM provider with an explicit API token.
    pub fn new(token: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            token,
            base_url: IBM_BASE_URL.to_string(),
        }
    }

    /// Create a new IBM provider reading the token from `IBM_QUANTUM_TOKEN`.
    pub fn from_env() -> Result<Self, QPUError> {
        let token = std::env::var("IBM_QUANTUM_TOKEN").map_err(|_| {
            QPUError::ConfigError(
                "IBM_QUANTUM_TOKEN environment variable not set. \
                 Get your token at https://quantum.ibm.com/"
                    .into(),
            )
        })?;
        Ok(Self::new(token))
    }

    /// Override the base URL (useful for testing against a mock server).
    #[cfg(test)]
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    // -- internal helpers ---------------------------------------------------

    /// Build an authenticated GET request.
    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.client
            .get(format!("{}{}", self.base_url, path))
            .bearer_auth(&self.token)
    }

    /// Build an authenticated POST request.
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}{}", self.base_url, path))
            .bearer_auth(&self.token)
    }

    /// Map an HTTP error response to a [`QPUError`].
    fn map_http_error(status: reqwest::StatusCode, body: &str) -> QPUError {
        match status.as_u16() {
            401 | 403 => QPUError::AuthenticationError(format!(
                "IBM Quantum authentication failed (HTTP {}): {}",
                status, body
            )),
            429 => {
                // Try to parse Retry-After header value from body (fallback 30s)
                let retry_secs = parse_retry_after_from_body(body).unwrap_or(30);
                QPUError::RateLimited {
                    retry_after: Duration::from_secs(retry_secs),
                }
            }
            404 => QPUError::BackendUnavailable(format!("Not found: {}", body)),
            _ => QPUError::ProviderError {
                provider: IBM_PROVIDER_NAME.into(),
                message: body.to_string(),
                status_code: Some(status.as_u16()),
            },
        }
    }

    /// Execute a request, handling common error cases.
    async fn execute(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<serde_json::Value, QPUError> {
        let response = request
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("IBM request failed: {}", e)))?;

        let status = response.status();

        // Check for rate-limiting via the Retry-After header before consuming body.
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(30);
            return Err(QPUError::RateLimited {
                retry_after: Duration::from_secs(retry_after),
            });
        }

        let body = response
            .text()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Failed to read IBM response: {}", e)))?;

        if !status.is_success() {
            return Err(Self::map_http_error(status, &body));
        }

        serde_json::from_str(&body).map_err(|e| {
            QPUError::ProviderError {
                provider: IBM_PROVIDER_NAME.into(),
                message: format!("Invalid JSON from IBM: {}", e),
                status_code: None,
            }
        })
    }
}

// ---------------------------------------------------------------------------
// QPUProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl QPUProvider for IBMProvider {
    fn name(&self) -> &str {
        IBM_PROVIDER_NAME
    }

    async fn list_backends(&self) -> Result<Vec<BackendInfo>, QPUError> {
        let json = self.execute(self.get("/api/backends")).await?;

        let backends_arr = json
            .as_array()
            .ok_or_else(|| QPUError::ProviderError {
                provider: IBM_PROVIDER_NAME.into(),
                message: "Expected JSON array from /api/backends".into(),
                status_code: None,
            })?;

        let mut backends = Vec::with_capacity(backends_arr.len());
        for entry in backends_arr {
            if let Some(info) = parse_ibm_backend(entry) {
                backends.push(info);
            }
        }
        Ok(backends)
    }

    async fn get_backend(&self, name: &str) -> Result<BackendInfo, QPUError> {
        // IBM v2 API: individual backend info via /api/backends/{name}
        let json = self
            .execute(self.get(&format!("/api/backends/{}", name)))
            .await?;
        parse_ibm_backend(&json).ok_or_else(|| {
            QPUError::BackendUnavailable(format!("Could not parse backend info for '{}'", name))
        })
    }

    async fn submit_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        config: &JobConfig,
    ) -> Result<Box<dyn QPUJob>, QPUError> {
        let qasm = circuit.to_qasm2();

        let mut body = serde_json::json!({
            "backend": backend,
            "qasm": qasm,
            "shots": config.shots,
            "optimization_level": config.optimization_level,
        });

        if let Some(ref name) = config.name {
            body["name"] = serde_json::Value::String(name.clone());
        }

        // Merge provider-specific extra options.
        for (k, v) in &config.extra {
            body[k] = serde_json::Value::String(v.clone());
        }

        let json = self
            .execute(self.post("/api/jobs").json(&body))
            .await?;

        let job_id = json["id"]
            .as_str()
            .ok_or_else(|| QPUError::SubmissionError("Missing 'id' in job response".into()))?
            .to_string();

        Ok(Box::new(IBMJob {
            client: self.client.clone(),
            token: self.token.clone(),
            base_url: self.base_url.clone(),
            job_id,
            backend_name: backend.to_string(),
        }))
    }

    async fn estimate_cost(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        shots: usize,
    ) -> Result<Option<CostEstimate>, QPUError> {
        // IBM Quantum has a free tier for open-plan users and usage-based
        // pricing for premium instances.  The public API does not expose a
        // cost-estimation endpoint, so we return a best-effort estimate based
        // on known pricing tiers.
        let gate_count = circuit.gate_count();
        let is_simulator = backend.contains("simulator") || backend.contains("sim");

        // IBM simulators are free; hardware ~ $1.60 per runtime-second.
        // Rough heuristic: each shot takes ~10us of QPU time per gate.
        let estimated_seconds = (shots as f64) * (gate_count as f64) * 1e-5;
        let cost = if is_simulator {
            0.0
        } else {
            estimated_seconds * 1.60
        };

        Ok(Some(CostEstimate {
            provider: IBM_PROVIDER_NAME.into(),
            backend: backend.into(),
            shots,
            estimated_cost_usd: cost,
            estimated_queue_time: None,
            is_free_tier: is_simulator,
            details: Some(format!(
                "Estimated {} runtime-seconds at $1.60/s (hardware) or free (simulator)",
                estimated_seconds
            )),
        }))
    }

    fn validate_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &BackendInfo,
    ) -> Result<ValidationReport, QPUError> {
        let validator = CircuitValidator::new(backend);
        Ok(validator.validate(circuit))
    }
}

// ---------------------------------------------------------------------------
// IBMJob
// ---------------------------------------------------------------------------

/// Handle for a job submitted to IBM Quantum.
pub struct IBMJob {
    client: reqwest::Client,
    token: String,
    base_url: String,
    job_id: String,
    backend_name: String,
}

impl IBMJob {
    /// Execute an authenticated GET and return parsed JSON.
    async fn get_json(&self, path: &str) -> Result<serde_json::Value, QPUError> {
        let resp = self
            .client
            .get(format!("{}{}", self.base_url, path))
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("IBM request failed: {}", e)))?;

        let status = resp.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(30);
            return Err(QPUError::RateLimited {
                retry_after: Duration::from_secs(retry_after),
            });
        }

        let body = resp
            .text()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Failed to read IBM response: {}", e)))?;

        if !status.is_success() {
            return Err(IBMProvider::map_http_error(status, &body));
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ProviderError {
            provider: IBM_PROVIDER_NAME.into(),
            message: format!("Invalid JSON: {}", e),
            status_code: None,
        })
    }
}

#[async_trait]
impl QPUJob for IBMJob {
    fn id(&self) -> &str {
        &self.job_id
    }

    fn backend(&self) -> &str {
        &self.backend_name
    }

    fn provider(&self) -> &str {
        IBM_PROVIDER_NAME
    }

    async fn status(&self) -> Result<JobStatus, QPUError> {
        let json = self
            .get_json(&format!("/api/jobs/{}", self.job_id))
            .await?;
        Ok(parse_ibm_job_status(&json))
    }

    async fn wait_for_completion(
        &self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<JobResult, QPUError> {
        let start = Instant::now();
        let mut current_interval = poll_interval;

        loop {
            if start.elapsed() > timeout {
                return Err(QPUError::Timeout);
            }

            let json = self
                .get_json(&format!("/api/jobs/{}", self.job_id))
                .await?;

            let status = parse_ibm_job_status(&json);

            match status {
                JobStatus::Completed => {
                    return self.fetch_result().await;
                }
                JobStatus::Failed(msg) => {
                    return Err(QPUError::ExecutionError(format!(
                        "IBM job {} failed: {}",
                        self.job_id, msg
                    )));
                }
                JobStatus::Cancelled => {
                    return Err(QPUError::ExecutionError(format!(
                        "IBM job {} was cancelled",
                        self.job_id
                    )));
                }
                _ => {
                    // Exponential back-off capped at MAX_BACKOFF_SECS
                    tokio::time::sleep(current_interval).await;
                    current_interval = current_interval
                        .mul_f64(1.5)
                        .min(Duration::from_secs(MAX_BACKOFF_SECS));
                }
            }
        }
    }

    async fn result(&self) -> Result<Option<JobResult>, QPUError> {
        let json = self
            .get_json(&format!("/api/jobs/{}", self.job_id))
            .await?;
        let status = parse_ibm_job_status(&json);

        if status == JobStatus::Completed {
            self.fetch_result().await.map(Some)
        } else {
            Ok(None)
        }
    }

    async fn cancel(&self) -> Result<(), QPUError> {
        let resp = self
            .client
            .post(format!(
                "{}/api/jobs/{}/cancel",
                self.base_url, self.job_id
            ))
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Cancel request failed: {}", e)))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(QPUError::ProviderError {
                provider: IBM_PROVIDER_NAME.into(),
                message: format!("Failed to cancel job: {}", body),
                status_code: None,
            });
        }
        Ok(())
    }
}

impl IBMJob {
    /// Fetch full results from the results endpoint.
    async fn fetch_result(&self) -> Result<JobResult, QPUError> {
        let json = self
            .get_json(&format!("/api/jobs/{}/results", self.job_id))
            .await?;

        let counts = parse_ibm_counts(&json);

        let shots = json["shots"]
            .as_u64()
            .unwrap_or_else(|| counts.values().sum::<usize>() as u64)
            as usize;

        let execution_time = json["execution_time"]
            .as_f64()
            .map(|s| Duration::from_secs_f64(s));

        let queue_time = json["queue_time"]
            .as_f64()
            .map(|s| Duration::from_secs_f64(s));

        let is_simulator = self.backend_name.contains("simulator")
            || self.backend_name.contains("sim");

        let mut metadata = HashMap::new();
        if let Some(obj) = json.as_object() {
            for (k, v) in obj {
                if !["counts", "shots", "execution_time", "queue_time"].contains(&k.as_str()) {
                    metadata.insert(k.clone(), v.clone());
                }
            }
        }

        Ok(JobResult {
            job_id: self.job_id.clone(),
            backend: self.backend_name.clone(),
            counts,
            shots,
            is_simulator,
            execution_time,
            queue_time,
            metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// JSON parsing helpers
// ---------------------------------------------------------------------------

/// Parse an IBM backend JSON object into a [`BackendInfo`].
fn parse_ibm_backend(json: &serde_json::Value) -> Option<BackendInfo> {
    let name = json["name"].as_str()?.to_string();

    let num_qubits = json["num_qubits"]
        .as_u64()
        .or_else(|| json["n_qubits"].as_u64())
        .unwrap_or(0) as usize;

    let is_simulator = json["simulator"].as_bool().unwrap_or(false)
        || name.contains("simulator")
        || name.contains("sim");

    let status_str = json["status"]
        .as_str()
        .or_else(|| json["operational"].as_bool().map(|b| if b { "online" } else { "offline" }))
        .unwrap_or("offline");
    let status = match status_str.to_lowercase().as_str() {
        "online" | "active" | "true" => BackendStatus::Online,
        "maintenance" => BackendStatus::Maintenance,
        "retired" => BackendStatus::Retired,
        _ => BackendStatus::Offline,
    };

    let queue_length = json["pending_jobs"]
        .as_u64()
        .or_else(|| json["queue_length"].as_u64())
        .unwrap_or(0) as usize;

    let basis_gates: Vec<String> = json["basis_gates"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let coupling_map: Option<Vec<(usize, usize)>> = json["coupling_map"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|pair| {
                    let p = pair.as_array()?;
                    Some((p.first()?.as_u64()? as usize, p.get(1)?.as_u64()? as usize))
                })
                .collect()
        });

    let max_shots = json["max_shots"].as_u64().unwrap_or(100_000) as usize;
    let max_depth = json["max_depth"].as_u64().map(|d| d as usize);

    // Error rates from properties
    let props = &json["properties"];
    let avg_gate_error_1q = props["avg_gate_error_1q"]
        .as_f64()
        .or_else(|| props["gate_error_1q"].as_f64());
    let avg_gate_error_2q = props["avg_gate_error_2q"]
        .as_f64()
        .or_else(|| props["gate_error_2q"].as_f64());
    let avg_readout_error = props["avg_readout_error"]
        .as_f64()
        .or_else(|| props["readout_error"].as_f64());
    let avg_t1_us = props["avg_t1_us"].as_f64().or_else(|| props["t1"].as_f64());
    let avg_t2_us = props["avg_t2_us"].as_f64().or_else(|| props["t2"].as_f64());

    let mut metadata = HashMap::new();
    if let Some(desc) = json["description"].as_str() {
        metadata.insert(
            "description".into(),
            serde_json::Value::String(desc.into()),
        );
    }
    if let Some(version) = json["version"].as_str() {
        metadata.insert("version".into(), serde_json::Value::String(version.into()));
    }
    if let Some(processor) = json["processor_type"].as_object() {
        metadata.insert("processor_type".into(), serde_json::Value::Object(processor.clone()));
    }

    Some(BackendInfo {
        name,
        provider: IBM_PROVIDER_NAME.into(),
        num_qubits,
        is_simulator,
        status,
        queue_length,
        basis_gates,
        coupling_map,
        max_shots,
        max_depth,
        avg_gate_error_1q,
        avg_gate_error_2q,
        avg_readout_error,
        avg_t1_us,
        avg_t2_us,
        metadata,
    })
}

/// Parse IBM job status string.
fn parse_ibm_job_status(json: &serde_json::Value) -> JobStatus {
    match json["status"]
        .as_str()
        .unwrap_or("UNKNOWN")
        .to_uppercase()
        .as_str()
    {
        "INITIALIZING" | "CREATING" => JobStatus::Initializing,
        "QUEUED" | "PENDING" => JobStatus::Queued,
        "VALIDATING" | "TRANSPILING" => JobStatus::Validating,
        "RUNNING" | "EXECUTING" => JobStatus::Running,
        "COMPLETED" | "DONE" => JobStatus::Completed,
        "FAILED" | "ERROR" => {
            let msg = json["error"]
                .as_str()
                .or_else(|| json["error_message"].as_str())
                .unwrap_or("Unknown error")
                .to_string();
            JobStatus::Failed(msg)
        }
        "CANCELLED" | "CANCELED" => JobStatus::Cancelled,
        other => JobStatus::Failed(format!("Unknown IBM status: {}", other)),
    }
}

/// Parse IBM measurement counts from a results response.
fn parse_ibm_counts(json: &serde_json::Value) -> HashMap<String, usize> {
    let mut counts = HashMap::new();

    // IBM returns counts under various paths depending on API version.
    let counts_obj = json["counts"]
        .as_object()
        .or_else(|| json["results"][0]["data"]["counts"].as_object())
        .or_else(|| json["quasi_dists"][0].as_object());

    if let Some(obj) = counts_obj {
        for (bitstring, count_val) in obj {
            // IBM sometimes uses hex keys like "0x0", "0x3"
            let key = if bitstring.starts_with("0x") {
                if let Ok(num) = u64::from_str_radix(&bitstring[2..], 16) {
                    format!("{:b}", num)
                } else {
                    bitstring.clone()
                }
            } else {
                // Strip spaces IBM sometimes puts in bitstrings
                bitstring.replace(' ', "")
            };

            let count = count_val
                .as_u64()
                .or_else(|| count_val.as_f64().map(|f| f as u64))
                .unwrap_or(0) as usize;

            counts.insert(key, count);
        }
    }

    counts
}

/// Attempt to parse a retry-after value from a JSON error body.
fn parse_retry_after_from_body(body: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()?["retry_after"]
        .as_u64()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Backend parsing ----------------------------------------------------

    fn sample_backend_json() -> serde_json::Value {
        serde_json::json!({
            "name": "ibm_brisbane",
            "num_qubits": 127,
            "simulator": false,
            "status": "online",
            "pending_jobs": 42,
            "basis_gates": ["cx", "id", "rz", "sx", "x"],
            "coupling_map": [[0,1],[1,0],[1,2],[2,1]],
            "max_shots": 100000,
            "max_depth": 300,
            "description": "Eagle r3 processor",
            "version": "1.2.3",
            "processor_type": {"family": "Eagle", "revision": 3},
            "properties": {
                "avg_gate_error_1q": 0.00034,
                "avg_gate_error_2q": 0.0089,
                "avg_readout_error": 0.012,
                "avg_t1_us": 245.0,
                "avg_t2_us": 120.0
            }
        })
    }

    #[test]
    fn test_parse_ibm_backend_basic() {
        let info = parse_ibm_backend(&sample_backend_json()).expect("should parse");
        assert_eq!(info.name, "ibm_brisbane");
        assert_eq!(info.provider, "IBM Quantum");
        assert_eq!(info.num_qubits, 127);
        assert!(!info.is_simulator);
        assert_eq!(info.status, BackendStatus::Online);
        assert_eq!(info.queue_length, 42);
        assert_eq!(info.basis_gates, vec!["cx", "id", "rz", "sx", "x"]);
        assert_eq!(
            info.coupling_map,
            Some(vec![(0, 1), (1, 0), (1, 2), (2, 1)])
        );
        assert_eq!(info.max_shots, 100_000);
        assert_eq!(info.max_depth, Some(300));
        assert!((info.avg_gate_error_1q.unwrap() - 0.00034).abs() < 1e-10);
        assert!((info.avg_gate_error_2q.unwrap() - 0.0089).abs() < 1e-10);
        assert!((info.avg_readout_error.unwrap() - 0.012).abs() < 1e-10);
        assert!((info.avg_t1_us.unwrap() - 245.0).abs() < 1e-10);
        assert!((info.avg_t2_us.unwrap() - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_ibm_backend_simulator() {
        let json = serde_json::json!({
            "name": "ibmq_qasm_simulator",
            "num_qubits": 32,
            "simulator": true,
            "status": "online",
            "basis_gates": ["cx", "id", "rz", "sx", "x", "u3"],
            "max_shots": 100000
        });
        let info = parse_ibm_backend(&json).expect("should parse simulator");
        assert!(info.is_simulator);
        assert_eq!(info.num_qubits, 32);
        assert!(info.coupling_map.is_none());
    }

    #[test]
    fn test_parse_ibm_backend_missing_name() {
        let json = serde_json::json!({"num_qubits": 5});
        assert!(parse_ibm_backend(&json).is_none());
    }

    #[test]
    fn test_parse_ibm_backend_minimal() {
        let json = serde_json::json!({"name": "minimal"});
        let info = parse_ibm_backend(&json).expect("should parse minimal");
        assert_eq!(info.name, "minimal");
        assert_eq!(info.num_qubits, 0);
        assert_eq!(info.status, BackendStatus::Offline);
        assert!(info.avg_gate_error_1q.is_none());
    }

    #[test]
    fn test_parse_ibm_backend_alternative_fields() {
        // Some IBM API versions use different field names
        let json = serde_json::json!({
            "name": "ibm_alt",
            "n_qubits": 65,
            "operational": true,
            "queue_length": 10,
            "basis_gates": ["ecr", "rz", "sx"],
            "properties": {
                "gate_error_1q": 0.0005,
                "gate_error_2q": 0.01,
                "readout_error": 0.02,
                "t1": 180.0,
                "t2": 90.0
            }
        });
        let info = parse_ibm_backend(&json).expect("should parse alternative fields");
        assert_eq!(info.num_qubits, 65);
        assert_eq!(info.status, BackendStatus::Online);
        assert_eq!(info.queue_length, 10);
        assert!((info.avg_gate_error_1q.unwrap() - 0.0005).abs() < 1e-10);
    }

    // -- Status parsing -----------------------------------------------------

    #[test]
    fn test_parse_ibm_job_status_all_states() {
        let cases = vec![
            ("INITIALIZING", false, false),
            ("CREATING", false, false),
            ("QUEUED", false, false),
            ("PENDING", false, false),
            ("VALIDATING", false, false),
            ("TRANSPILING", false, false),
            ("RUNNING", false, false),
            ("EXECUTING", false, false),
            ("COMPLETED", true, true),
            ("DONE", true, true),
            ("CANCELLED", true, false),
            ("CANCELED", true, false),
        ];

        for (status_str, is_terminal, is_success) in cases {
            let json = serde_json::json!({"status": status_str});
            let status = parse_ibm_job_status(&json);
            assert_eq!(
                status.is_terminal(),
                is_terminal,
                "Status '{}' terminal check failed",
                status_str
            );
            assert_eq!(
                status.is_success(),
                is_success,
                "Status '{}' success check failed",
                status_str
            );
        }
    }

    #[test]
    fn test_parse_ibm_job_status_failed_with_message() {
        let json = serde_json::json!({
            "status": "FAILED",
            "error": "Calibration data stale"
        });
        let status = parse_ibm_job_status(&json);
        match status {
            JobStatus::Failed(msg) => assert_eq!(msg, "Calibration data stale"),
            other => panic!("Expected Failed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_ibm_job_status_unknown() {
        let json = serde_json::json!({"status": "WEIRD_STATE"});
        let status = parse_ibm_job_status(&json);
        assert!(status.is_terminal());
        match status {
            JobStatus::Failed(msg) => assert!(msg.contains("WEIRD_STATE")),
            other => panic!("Expected Failed for unknown, got {:?}", other),
        }
    }

    // -- Counts parsing -----------------------------------------------------

    #[test]
    fn test_parse_ibm_counts_direct() {
        let json = serde_json::json!({
            "counts": {
                "00": 512,
                "11": 512
            }
        });
        let counts = parse_ibm_counts(&json);
        assert_eq!(counts.get("00"), Some(&512));
        assert_eq!(counts.get("11"), Some(&512));
    }

    #[test]
    fn test_parse_ibm_counts_hex_keys() {
        let json = serde_json::json!({
            "counts": {
                "0x0": 500,
                "0x3": 524
            }
        });
        let counts = parse_ibm_counts(&json);
        assert_eq!(counts.get("0"), Some(&500));
        assert_eq!(counts.get("11"), Some(&524));
    }

    #[test]
    fn test_parse_ibm_counts_nested_results() {
        let json = serde_json::json!({
            "results": [{
                "data": {
                    "counts": {
                        "0x0": 450,
                        "0x1": 274,
                        "0x2": 200,
                        "0x3": 100
                    }
                }
            }]
        });
        let counts = parse_ibm_counts(&json);
        assert_eq!(counts.len(), 4);
        assert_eq!(counts.get("0"), Some(&450));
        assert_eq!(counts.get("1"), Some(&274));
        assert_eq!(counts.get("10"), Some(&200));
        assert_eq!(counts.get("11"), Some(&100));
    }

    #[test]
    fn test_parse_ibm_counts_float_values() {
        // quasi_dists sometimes returns float probabilities that we treat as counts
        let json = serde_json::json!({
            "counts": {
                "00": 512.0,
                "11": 512.0
            }
        });
        let counts = parse_ibm_counts(&json);
        assert_eq!(counts.get("00"), Some(&512));
    }

    #[test]
    fn test_parse_ibm_counts_empty() {
        let json = serde_json::json!({});
        let counts = parse_ibm_counts(&json);
        assert!(counts.is_empty());
    }

    // -- Cost estimation ----------------------------------------------------

    #[tokio::test]
    async fn test_ibm_cost_estimate_simulator() {
        let provider = IBMProvider::new("fake_token".into());
        let circuit = QPUCircuit::bell_state();
        let cost = provider
            .estimate_cost(&circuit, "ibmq_qasm_simulator", 1024)
            .await
            .expect("cost estimation should not fail");
        let cost = cost.expect("should return some estimate");
        assert!(cost.is_free_tier);
        assert_eq!(cost.estimated_cost_usd, 0.0);
    }

    #[tokio::test]
    async fn test_ibm_cost_estimate_hardware() {
        let provider = IBMProvider::new("fake_token".into());
        let circuit = QPUCircuit::ghz_state(5);
        let cost = provider
            .estimate_cost(&circuit, "ibm_brisbane", 4096)
            .await
            .expect("cost estimation should not fail");
        let cost = cost.expect("should return some estimate");
        assert!(!cost.is_free_tier);
        assert!(cost.estimated_cost_usd > 0.0);
        assert_eq!(cost.provider, "IBM Quantum");
    }

    // -- Validation ---------------------------------------------------------

    #[test]
    fn test_ibm_validate_circuit_valid() {
        let provider = IBMProvider::new("fake_token".into());
        let circuit = QPUCircuit::bell_state();
        let backend = BackendInfo {
            name: "ibm_brisbane".into(),
            provider: "IBM Quantum".into(),
            num_qubits: 127,
            is_simulator: false,
            status: BackendStatus::Online,
            queue_length: 0,
            basis_gates: vec!["h".into(), "cx".into(), "rz".into(), "sx".into(), "x".into()],
            coupling_map: Some(vec![(0, 1), (1, 0)]),
            max_shots: 100_000,
            max_depth: Some(300),
            avg_gate_error_1q: Some(0.0003),
            avg_gate_error_2q: Some(0.009),
            avg_readout_error: Some(0.012),
            avg_t1_us: Some(245.0),
            avg_t2_us: Some(120.0),
            metadata: HashMap::new(),
        };
        let report = provider
            .validate_circuit(&circuit, &backend)
            .expect("validation should succeed");
        assert!(report.is_valid);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_ibm_validate_circuit_too_many_qubits() {
        let provider = IBMProvider::new("fake_token".into());
        let circuit = QPUCircuit::ghz_state(10);
        let backend = BackendInfo {
            name: "small_backend".into(),
            provider: "IBM Quantum".into(),
            num_qubits: 5,
            is_simulator: false,
            status: BackendStatus::Online,
            queue_length: 0,
            basis_gates: vec!["h".into(), "cx".into()],
            coupling_map: None,
            max_shots: 100_000,
            max_depth: None,
            avg_gate_error_1q: None,
            avg_gate_error_2q: None,
            avg_readout_error: None,
            avg_t1_us: None,
            avg_t2_us: None,
            metadata: HashMap::new(),
        };
        let report = provider
            .validate_circuit(&circuit, &backend)
            .expect("validation should succeed");
        assert!(!report.is_valid);
        assert!(report.errors.iter().any(|e| e.contains("qubits")));
    }

    // -- Retry-After parsing ------------------------------------------------

    #[test]
    fn test_parse_retry_after_from_body() {
        let body = r#"{"error":"rate limited","retry_after":45}"#;
        assert_eq!(parse_retry_after_from_body(body), Some(45));
    }

    #[test]
    fn test_parse_retry_after_missing() {
        let body = r#"{"error":"rate limited"}"#;
        assert_eq!(parse_retry_after_from_body(body), None);
    }

    #[test]
    fn test_parse_retry_after_invalid_json() {
        assert_eq!(parse_retry_after_from_body("not json"), None);
    }

    // -- HTTP error mapping -------------------------------------------------

    #[test]
    fn test_map_http_error_auth() {
        let err = IBMProvider::map_http_error(
            reqwest::StatusCode::UNAUTHORIZED,
            "bad token",
        );
        match err {
            QPUError::AuthenticationError(msg) => assert!(msg.contains("401")),
            other => panic!("Expected AuthenticationError, got {:?}", other),
        }
    }

    #[test]
    fn test_map_http_error_rate_limited() {
        let err = IBMProvider::map_http_error(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"retry_after":60}"#,
        );
        match err {
            QPUError::RateLimited { retry_after } => {
                assert_eq!(retry_after, Duration::from_secs(60));
            }
            other => panic!("Expected RateLimited, got {:?}", other),
        }
    }

    #[test]
    fn test_map_http_error_not_found() {
        let err = IBMProvider::map_http_error(
            reqwest::StatusCode::NOT_FOUND,
            "no such backend",
        );
        match err {
            QPUError::BackendUnavailable(msg) => assert!(msg.contains("no such backend")),
            other => panic!("Expected BackendUnavailable, got {:?}", other),
        }
    }

    #[test]
    fn test_map_http_error_server() {
        let err = IBMProvider::map_http_error(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "something broke",
        );
        match err {
            QPUError::ProviderError {
                provider,
                message,
                status_code,
            } => {
                assert_eq!(provider, "IBM Quantum");
                assert_eq!(message, "something broke");
                assert_eq!(status_code, Some(500));
            }
            other => panic!("Expected ProviderError, got {:?}", other),
        }
    }
}
