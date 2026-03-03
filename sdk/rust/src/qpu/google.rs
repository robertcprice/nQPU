//! Google Quantum AI provider — submit circuits to Sycamore processors.
//!
//! API: https://quantum.googleapis.com/v1alpha1/
//! Auth: Google Cloud service account / OAuth2 bearer token
//! Note: Google Quantum API access requires partnership/approval from Google.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::qpu::error::QPUError;
use crate::qpu::job::*;
use crate::qpu::provider::{QPUJob, QPUProvider};
use crate::qpu::validation::CircuitValidator;
use crate::qpu::QPUCircuit;

const GOOGLE_API_BASE: &str = "https://quantum.googleapis.com/v1alpha1";

/// Google Quantum AI provider.
///
/// Connects to Google's Quantum Computing Service for access to
/// Sycamore and future processors. Requires Google Cloud project
/// with Quantum Engine API enabled and appropriate permissions.
pub struct GoogleProvider {
    client: reqwest::Client,
    project_id: String,
    access_token: Arc<RwLock<Option<CachedToken>>>,
    credentials_json: Option<String>,
}

struct CachedToken {
    token: String,
    expires_at: Instant,
}

impl GoogleProvider {
    /// Create with explicit project ID and access token.
    pub fn new(project_id: String, access_token: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        Self {
            client,
            project_id,
            access_token: Arc::new(RwLock::new(Some(CachedToken {
                token: access_token,
                expires_at: Instant::now() + Duration::from_secs(3600),
            }))),
            credentials_json: None,
        }
    }

    /// Create from environment variables.
    /// Reads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_QUANTUM_ACCESS_TOKEN` or
    /// `GOOGLE_APPLICATION_CREDENTIALS` for service account auth.
    pub fn from_env() -> Result<Self, QPUError> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
            QPUError::ConfigError("GOOGLE_CLOUD_PROJECT not set".into())
        })?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let access_token = std::env::var("GOOGLE_QUANTUM_ACCESS_TOKEN").ok();
        let credentials_json = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok();

        let cached = access_token.map(|t| CachedToken {
            token: t,
            expires_at: Instant::now() + Duration::from_secs(3600),
        });

        Ok(Self {
            client,
            project_id,
            access_token: Arc::new(RwLock::new(cached)),
            credentials_json,
        })
    }

    async fn get_token(&self) -> Result<String, QPUError> {
        // Check cache
        {
            let cache = self.access_token.read().unwrap();
            if let Some(ref cached) = *cache {
                if cached.expires_at > Instant::now() {
                    return Ok(cached.token.clone());
                }
            }
        }

        // If we have service account credentials, refresh via OAuth2
        if let Some(ref _creds_path) = self.credentials_json {
            // Service account token refresh via metadata server or JWT
            // For now, try the metadata server (works in GCP environments)
            let resp = self
                .client
                .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
                .header("Metadata-Flavor", "Google")
                .send()
                .await
                .map_err(|e| QPUError::AuthenticationError(
                    format!("Failed to get Google token: {}. Set GOOGLE_QUANTUM_ACCESS_TOKEN manually.", e)
                ))?;

            let body: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| QPUError::AuthenticationError(e.to_string()))?;

            let token = body["access_token"]
                .as_str()
                .ok_or_else(|| QPUError::AuthenticationError("No access_token in response".into()))?
                .to_string();

            let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

            let mut cache = self.access_token.write().unwrap();
            *cache = Some(CachedToken {
                token: token.clone(),
                expires_at: Instant::now() + Duration::from_secs(expires_in - 60),
            });

            Ok(token)
        } else {
            Err(QPUError::AuthenticationError(
                "No Google credentials available. Set GOOGLE_QUANTUM_ACCESS_TOKEN or \
                 GOOGLE_APPLICATION_CREDENTIALS."
                    .into(),
            ))
        }
    }

    async fn get_json(&self, path: &str) -> Result<serde_json::Value, QPUError> {
        let token = self.get_token().await?;
        let url = format!("{}{}", GOOGLE_API_BASE, path);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.text().await.map_err(|e| QPUError::NetworkError(e.to_string()))?;

        if status == 401 || status == 403 {
            return Err(QPUError::AuthenticationError(
                "Google Cloud authentication failed. Check your credentials.".into(),
            ));
        }
        if status >= 400 {
            return Err(QPUError::ProviderError {
                provider: "Google Quantum".into(),
                message: body,
                status_code: Some(status),
            });
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ConversionError(e.to_string()))
    }

    async fn post_json(
        &self,
        path: &str,
        payload: &serde_json::Value,
    ) -> Result<serde_json::Value, QPUError> {
        let token = self.get_token().await?;
        let url = format!("{}{}", GOOGLE_API_BASE, path);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(payload)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.text().await.map_err(|e| QPUError::NetworkError(e.to_string()))?;

        if status >= 400 {
            return Err(QPUError::ProviderError {
                provider: "Google Quantum".into(),
                message: body,
                status_code: Some(status),
            });
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ConversionError(e.to_string()))
    }

    /// Convert QPUCircuit to Google Cirq-style JSON format.
    fn circuit_to_cirq_json(circuit: &QPUCircuit) -> serde_json::Value {
        let mut moments = Vec::new();
        // Group gates into moments (layers)
        let mut current_moment: Vec<serde_json::Value> = Vec::new();
        let mut used_qubits: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for gate in &circuit.gates {
            let qubits = gate.qubits();
            let has_conflict = qubits.iter().any(|q| used_qubits.contains(q));

            if has_conflict && !current_moment.is_empty() {
                moments.push(serde_json::json!({"operations": current_moment}));
                current_moment = Vec::new();
                used_qubits.clear();
            }

            let op = gate_to_cirq_op(gate);
            if let Some(op) = op {
                current_moment.push(op);
                for q in qubits {
                    used_qubits.insert(q);
                }
            }
        }

        if !current_moment.is_empty() {
            moments.push(serde_json::json!({"operations": current_moment}));
        }

        // Add measurement moment
        if !circuit.measurements.is_empty() {
            let meas_qubits: Vec<serde_json::Value> = circuit
                .measurements
                .iter()
                .map(|&(q, _)| serde_json::json!({"id": format!("{}", q)}))
                .collect();

            moments.push(serde_json::json!({
                "operations": [{
                    "gate": {"id": "meas"},
                    "qubits": meas_qubits,
                    "key": "result"
                }]
            }));
        }

        serde_json::json!({
            "language": {
                "gate_set": "sqrt_iswap"
            },
            "circuit": {
                "scheduling_strategy": "MOMENT_BY_MOMENT",
                "moments": moments
            },
            "qubits": (0..circuit.num_qubits).map(|i| serde_json::json!({"id": format!("{}", i)})).collect::<Vec<_>>()
        })
    }
}

fn gate_to_cirq_op(gate: &crate::qpu::QPUGate) -> Option<serde_json::Value> {
    use crate::qpu::QPUGate;
    match gate {
        QPUGate::H(q) => Some(serde_json::json!({
            "gate": {"id": "h"},
            "qubits": [{"id": format!("{}", q)}]
        })),
        QPUGate::X(q) => Some(serde_json::json!({
            "gate": {"id": "x"},
            "qubits": [{"id": format!("{}", q)}]
        })),
        QPUGate::Y(q) => Some(serde_json::json!({
            "gate": {"id": "y"},
            "qubits": [{"id": format!("{}", q)}]
        })),
        QPUGate::Z(q) => Some(serde_json::json!({
            "gate": {"id": "z"},
            "qubits": [{"id": format!("{}", q)}]
        })),
        QPUGate::CX(c, t) => Some(serde_json::json!({
            "gate": {"id": "cnot"},
            "qubits": [{"id": format!("{}", c)}, {"id": format!("{}", t)}]
        })),
        QPUGate::CZ(a, b) => Some(serde_json::json!({
            "gate": {"id": "cz"},
            "qubits": [{"id": format!("{}", a)}, {"id": format!("{}", b)}]
        })),
        QPUGate::Rz(q, theta) => Some(serde_json::json!({
            "gate": {"id": "rz", "args": [{"arg_value": {"float_value": theta}}]},
            "qubits": [{"id": format!("{}", q)}]
        })),
        QPUGate::SycamoreGate(a, b) => Some(serde_json::json!({
            "gate": {"id": "syc"},
            "qubits": [{"id": format!("{}", a)}, {"id": format!("{}", b)}]
        })),
        QPUGate::PhasedXZ(q, x_exp, z_exp, axis_phase) => Some(serde_json::json!({
            "gate": {
                "id": "phased_xz",
                "args": [
                    {"arg_value": {"float_value": x_exp}},
                    {"arg_value": {"float_value": z_exp}},
                    {"arg_value": {"float_value": axis_phase}},
                ]
            },
            "qubits": [{"id": format!("{}", q)}]
        })),
        QPUGate::Barrier(_) => None,
        _ => {
            // Generic single-qubit gate
            let qs = gate.qubits();
            if qs.len() == 1 {
                Some(serde_json::json!({
                    "gate": {"id": gate.name()},
                    "qubits": [{"id": format!("{}", qs[0])}]
                }))
            } else if qs.len() == 2 {
                Some(serde_json::json!({
                    "gate": {"id": gate.name()},
                    "qubits": [{"id": format!("{}", qs[0])}, {"id": format!("{}", qs[1])}]
                }))
            } else {
                None
            }
        }
    }
}

#[async_trait]
impl QPUProvider for GoogleProvider {
    fn name(&self) -> &str {
        "Google Quantum AI"
    }

    async fn list_backends(&self) -> Result<Vec<BackendInfo>, QPUError> {
        let path = format!(
            "/projects/{}/processors",
            self.project_id
        );
        match self.get_json(&path).await {
            Ok(resp) => {
                let mut backends = Vec::new();
                if let Some(processors) = resp.get("processors").and_then(|p| p.as_array()) {
                    for proc in processors {
                        if let Some(bi) = parse_google_processor(proc) {
                            backends.push(bi);
                        }
                    }
                }
                if backends.is_empty() {
                    backends = known_google_backends();
                }
                Ok(backends)
            }
            Err(_) => {
                // If API is unreachable, return known devices
                Ok(known_google_backends())
            }
        }
    }

    async fn get_backend(&self, name: &str) -> Result<BackendInfo, QPUError> {
        let backends = self.list_backends().await?;
        backends
            .into_iter()
            .find(|b| b.name == name)
            .ok_or_else(|| QPUError::BackendUnavailable(name.into()))
    }

    async fn submit_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        config: &JobConfig,
    ) -> Result<Box<dyn QPUJob>, QPUError> {
        let processor_id = match backend {
            "google_sycamore" | "sycamore" => "rainbow",
            "google_weber" | "weber" => "weber",
            other => other,
        };

        let program_json = Self::circuit_to_cirq_json(circuit);

        let path = format!(
            "/projects/{}/programs",
            self.project_id,
        );

        let payload = serde_json::json!({
            "name": config.name.as_deref().unwrap_or("nqpu-metal"),
            "code": program_json,
            "scheduling_config": {
                "processor_selector": {
                    "processor": format!("projects/{}/processors/{}", self.project_id, processor_id)
                }
            },
        });

        let program = self.post_json(&path, &payload).await?;
        let program_name = program["name"].as_str().unwrap_or("").to_string();

        // Create a job under the program
        let job_path = format!("{}/jobs", program_name);
        let job_payload = serde_json::json!({
            "run_context": {
                "parameter_sweeps": [{
                    "repetitions": config.shots
                }]
            }
        });

        let job_resp = self.post_json(&job_path, &job_payload).await?;
        let job_name = job_resp["name"].as_str().unwrap_or("").to_string();

        Ok(Box::new(GoogleJob {
            client: self.client.clone(),
            access_token: self.access_token.clone(),
            job_name,
            backend: backend.into(),
            shots: config.shots,
            num_qubits: circuit.num_qubits,
        }))
    }

    async fn estimate_cost(
        &self,
        _circuit: &QPUCircuit,
        backend: &str,
        shots: usize,
    ) -> Result<Option<CostEstimate>, QPUError> {
        // Google Quantum doesn't have public pricing (research access)
        Ok(Some(CostEstimate {
            provider: "Google Quantum AI".into(),
            backend: backend.into(),
            shots,
            estimated_cost_usd: 0.0,
            estimated_queue_time: None,
            is_free_tier: true,
            details: Some("Google Quantum access is via research partnership — no per-job cost".into()),
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

/// Google Quantum job handle.
pub struct GoogleJob {
    client: reqwest::Client,
    access_token: Arc<RwLock<Option<CachedToken>>>,
    job_name: String,
    backend: String,
    shots: usize,
    num_qubits: usize,
}

impl GoogleJob {
    fn get_token(&self) -> Result<String, QPUError> {
        let cache = self.access_token.read().unwrap();
        match &*cache {
            Some(cached) if cached.expires_at > Instant::now() => Ok(cached.token.clone()),
            _ => Err(QPUError::AuthenticationError("Token expired".into())),
        }
    }
}

#[async_trait]
impl QPUJob for GoogleJob {
    fn id(&self) -> &str {
        &self.job_name
    }

    fn backend(&self) -> &str {
        &self.backend
    }

    fn provider(&self) -> &str {
        "Google Quantum AI"
    }

    async fn status(&self) -> Result<JobStatus, QPUError> {
        let token = self.get_token()?;
        let url = format!("{}/{}", GOOGLE_API_BASE, self.job_name);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        parse_google_status(&body)
    }

    async fn wait_for_completion(
        &self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<JobResult, QPUError> {
        let start = Instant::now();
        let mut interval = poll_interval;

        loop {
            if start.elapsed() > timeout {
                return Err(QPUError::Timeout);
            }

            let token = self.get_token()?;
            let url = format!("{}/{}", GOOGLE_API_BASE, self.job_name);
            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", token))
                .send()
                .await
                .map_err(|e| QPUError::NetworkError(e.to_string()))?;

            let body: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| QPUError::NetworkError(e.to_string()))?;

            let status = parse_google_status(&body)?;

            match status {
                JobStatus::Completed => {
                    return parse_google_result(&body, &self.backend, self.shots, self.num_qubits);
                }
                JobStatus::Failed(msg) => {
                    return Err(QPUError::ExecutionError(msg));
                }
                JobStatus::Cancelled => {
                    return Err(QPUError::ExecutionError("Job cancelled".into()));
                }
                _ => {
                    tokio::time::sleep(interval).await;
                    interval = (interval * 2).min(Duration::from_secs(60));
                }
            }
        }
    }

    async fn result(&self) -> Result<Option<JobResult>, QPUError> {
        let token = self.get_token()?;
        let url = format!("{}/{}", GOOGLE_API_BASE, self.job_name);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let status = parse_google_status(&body)?;
        if status.is_success() {
            Ok(Some(parse_google_result(&body, &self.backend, self.shots, self.num_qubits)?))
        } else {
            Ok(None)
        }
    }

    async fn cancel(&self) -> Result<(), QPUError> {
        let token = self.get_token()?;
        let url = format!("{}/{}:cancel", GOOGLE_API_BASE, self.job_name);
        self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;
        Ok(())
    }
}

fn parse_google_status(body: &serde_json::Value) -> Result<JobStatus, QPUError> {
    let state = body["executionStatus"]["state"]
        .as_str()
        .unwrap_or("UNKNOWN");
    match state {
        "READY" | "CREATED" => Ok(JobStatus::Queued),
        "RUNNING" => Ok(JobStatus::Running),
        "SUCCESS" => Ok(JobStatus::Completed),
        "FAILURE" => {
            let msg = body["executionStatus"]["failureMessage"]
                .as_str()
                .unwrap_or("Unknown failure")
                .to_string();
            Ok(JobStatus::Failed(msg))
        }
        "CANCELLED" => Ok(JobStatus::Cancelled),
        _ => Ok(JobStatus::Queued),
    }
}

fn parse_google_result(
    body: &serde_json::Value,
    backend: &str,
    shots: usize,
    num_qubits: usize,
) -> Result<JobResult, QPUError> {
    let job_id = body["name"].as_str().unwrap_or("").to_string();
    let mut counts = HashMap::new();

    // Google returns results as measurement arrays
    if let Some(results) = body.get("result").and_then(|r| r.get("measurementResults")) {
        if let Some(arr) = results.as_array() {
            for result in arr {
                if let Some(measurements) = result.get("measurements").and_then(|m| m.as_array()) {
                    for meas in measurements {
                        if let Some(bits) = meas.as_array() {
                            let bitstring: String =
                                bits.iter().map(|b| if b.as_i64() == Some(1) { '1' } else { '0' }).collect();
                            *counts.entry(bitstring).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(JobResult {
        job_id,
        backend: backend.into(),
        counts,
        shots,
        is_simulator: false,
        execution_time: None,
        queue_time: None,
        metadata: HashMap::new(),
    })
}

fn parse_google_processor(proc: &serde_json::Value) -> Option<BackendInfo> {
    let name = proc["name"].as_str()?;
    let short_name = name.rsplit('/').next().unwrap_or(name);

    Some(BackendInfo {
        name: format!("google_{}", short_name),
        provider: "Google Quantum AI".into(),
        num_qubits: proc["deviceSpec"]["numQubits"].as_u64().unwrap_or(53) as usize,
        is_simulator: false,
        status: if proc["health"] == "OK" {
            BackendStatus::Online
        } else {
            BackendStatus::Offline
        },
        queue_length: 0,
        basis_gates: vec![
            "syc".into(), "phased_xz".into(), "cz".into(),
            "h".into(), "x".into(), "y".into(), "z".into(),
        ],
        coupling_map: None,
        max_shots: 1_000_000,
        max_depth: None,
        avg_gate_error_1q: Some(0.001),
        avg_gate_error_2q: Some(0.005), // Sycamore typical
        avg_readout_error: Some(0.01),
        avg_t1_us: Some(20.0),
        avg_t2_us: Some(10.0),
        metadata: HashMap::new(),
    })
}

fn known_google_backends() -> Vec<BackendInfo> {
    vec![
        BackendInfo {
            name: "google_sycamore".into(),
            provider: "Google Quantum AI".into(),
            num_qubits: 53,
            is_simulator: false,
            status: BackendStatus::Online,
            queue_length: 0,
            basis_gates: vec![
                "syc".into(), "phased_xz".into(), "cz".into(),
            ],
            coupling_map: None,
            max_shots: 1_000_000,
            max_depth: None,
            avg_gate_error_1q: Some(0.001),
            avg_gate_error_2q: Some(0.005),
            avg_readout_error: Some(0.01),
            avg_t1_us: Some(20.0),
            avg_t2_us: Some(10.0),
            metadata: HashMap::new(),
        },
        BackendInfo {
            name: "google_weber".into(),
            provider: "Google Quantum AI".into(),
            num_qubits: 53,
            is_simulator: false,
            status: BackendStatus::Offline,
            queue_length: 0,
            basis_gates: vec![
                "syc".into(), "phased_xz".into(), "cz".into(),
            ],
            coupling_map: None,
            max_shots: 1_000_000,
            max_depth: None,
            avg_gate_error_1q: Some(0.001),
            avg_gate_error_2q: Some(0.005),
            avg_readout_error: Some(0.01),
            avg_t1_us: Some(20.0),
            avg_t2_us: Some(10.0),
            metadata: HashMap::new(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_google_status_success() {
        let body = serde_json::json!({
            "executionStatus": {"state": "SUCCESS"}
        });
        let status = parse_google_status(&body).unwrap();
        assert!(matches!(status, JobStatus::Completed));
    }

    #[test]
    fn test_parse_google_status_failure() {
        let body = serde_json::json!({
            "executionStatus": {
                "state": "FAILURE",
                "failureMessage": "Calibration error"
            }
        });
        let status = parse_google_status(&body).unwrap();
        assert!(matches!(status, JobStatus::Failed(msg) if msg == "Calibration error"));
    }

    #[test]
    fn test_known_backends() {
        let backends = known_google_backends();
        assert_eq!(backends.len(), 2);
        assert!(backends.iter().any(|b| b.name == "google_sycamore"));
        assert!(backends.iter().any(|b| b.name == "google_weber"));
    }

    #[test]
    fn test_circuit_to_cirq_json() {
        let circuit = QPUCircuit::bell_state();
        let json = GoogleProvider::circuit_to_cirq_json(&circuit);
        assert!(json.get("circuit").is_some());
        assert!(json.get("qubits").is_some());
        let moments = json["circuit"]["moments"].as_array().unwrap();
        assert!(moments.len() >= 2); // gates + measurement
    }

    #[test]
    fn test_gate_to_cirq_op_h() {
        let gate = crate::qpu::QPUGate::H(0);
        let op = gate_to_cirq_op(&gate).unwrap();
        assert_eq!(op["gate"]["id"], "h");
        assert_eq!(op["qubits"][0]["id"], "0");
    }

    #[test]
    fn test_gate_to_cirq_op_cx() {
        let gate = crate::qpu::QPUGate::CX(0, 1);
        let op = gate_to_cirq_op(&gate).unwrap();
        assert_eq!(op["gate"]["id"], "cnot");
        assert_eq!(op["qubits"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_parse_google_result() {
        let body = serde_json::json!({
            "name": "projects/test/programs/p1/jobs/j1",
            "executionStatus": {"state": "SUCCESS"},
            "result": {
                "measurementResults": [{
                    "measurements": [
                        [0, 0],
                        [0, 0],
                        [1, 1],
                        [1, 1],
                    ]
                }]
            }
        });
        let result = parse_google_result(&body, "google_sycamore", 4, 2).unwrap();
        assert_eq!(result.counts.get("00"), Some(&2));
        assert_eq!(result.counts.get("11"), Some(&2));
    }
}
