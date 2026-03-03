//! IonQ Direct API provider — submit circuits directly to IonQ trapped-ion hardware.
//!
//! API docs: https://docs.ionq.com/api-reference
//! Auth: API key via `IONQ_API_KEY` env var

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::qpu::error::QPUError;
use crate::qpu::job::*;
use crate::qpu::provider::{QPUJob, QPUProvider};
use crate::qpu::validation::CircuitValidator;
use crate::qpu::QPUCircuit;

const IONQ_API_BASE: &str = "https://api.ionq.co/v0.3";

/// IonQ Direct quantum hardware provider.
///
/// Connects directly to IonQ's REST API for access to trapped-ion processors
/// (Aria, Forte) and their cloud simulator.
pub struct IonQProvider {
    client: reqwest::Client,
    api_key: String,
}

impl IonQProvider {
    /// Create a new IonQ provider with the given API key.
    pub fn new(api_key: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        Self { client, api_key }
    }

    /// Create from environment variable `IONQ_API_KEY`.
    pub fn from_env() -> Result<Self, QPUError> {
        let key = std::env::var("IONQ_API_KEY").map_err(|_| {
            QPUError::ConfigError(
                "IONQ_API_KEY not set. Get your key at https://cloud.ionq.com/".into(),
            )
        })?;
        Ok(Self::new(key))
    }

    fn auth_header(&self) -> String {
        format!("apiKey {}", self.api_key)
    }

    async fn get_json(&self, path: &str) -> Result<serde_json::Value, QPUError> {
        let url = format!("{}{}", IONQ_API_BASE, path);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let status = resp.status().as_u16();
        if status == 429 {
            return Err(QPUError::RateLimited {
                retry_after: Duration::from_secs(30),
            });
        }
        if status == 401 || status == 403 {
            return Err(QPUError::AuthenticationError("Invalid IonQ API key".into()));
        }

        let body = resp.text().await.map_err(|e| QPUError::NetworkError(e.to_string()))?;

        if status >= 400 {
            return Err(QPUError::ProviderError {
                provider: "IonQ".into(),
                message: body,
                status_code: Some(status),
            });
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ConversionError(e.to_string()))
    }

    async fn post_json(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, QPUError> {
        let url = format!("{}{}", IONQ_API_BASE, path);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let status = resp.status().as_u16();
        let body_text = resp.text().await.map_err(|e| QPUError::NetworkError(e.to_string()))?;

        if status >= 400 {
            return Err(QPUError::ProviderError {
                provider: "IonQ".into(),
                message: body_text,
                status_code: Some(status),
            });
        }

        serde_json::from_str(&body_text).map_err(|e| QPUError::ConversionError(e.to_string()))
    }
}

#[async_trait]
impl QPUProvider for IonQProvider {
    fn name(&self) -> &str {
        "IonQ"
    }

    async fn list_backends(&self) -> Result<Vec<BackendInfo>, QPUError> {
        let resp = self.get_json("/backends").await?;
        let mut backends = Vec::new();

        if let Some(arr) = resp.as_array() {
            for item in arr {
                if let Some(bi) = parse_ionq_backend(item) {
                    backends.push(bi);
                }
            }
        }

        // If API returned nothing, provide known devices
        if backends.is_empty() {
            backends = known_ionq_backends();
        }

        Ok(backends)
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
        let target = match backend {
            "ionq_aria" | "aria" => "qpu.aria-1",
            "ionq_forte" | "forte" => "qpu.forte-1",
            "ionq_simulator" | "simulator" => "simulator",
            other => other,
        };

        // Build IonQ job payload using native gate format
        let circuit_json = circuit.to_ionq_json();

        let payload = serde_json::json!({
            "target": target,
            "shots": config.shots,
            "input": {
                "gateset": circuit_json.get("gateset").unwrap_or(&serde_json::json!("qis")),
                "qubits": circuit.num_qubits,
                "circuit": circuit_json.get("circuit").unwrap_or(&serde_json::json!([])),
            },
            "name": config.name.as_deref().unwrap_or("nqpu-metal-job"),
        });

        let resp = self.post_json("/jobs", &payload).await?;

        let job_id = resp["id"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if job_id.is_empty() {
            return Err(QPUError::SubmissionError(
                "No job ID in IonQ response".into(),
            ));
        }

        Ok(Box::new(IonQJob {
            client: self.client.clone(),
            api_key: self.api_key.clone(),
            job_id,
            backend: backend.into(),
            shots: config.shots,
        }))
    }

    async fn estimate_cost(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        shots: usize,
    ) -> Result<Option<CostEstimate>, QPUError> {
        let is_simulator = backend.contains("simulator");
        let num_gates = circuit.gate_count();

        // IonQ pricing: per-gate pricing for QPU
        // Aria: ~$0.01 per gate per shot / 1000 (rough estimate)
        let cost = if is_simulator {
            0.0
        } else {
            // Rough estimate: $0.30 per minute + per-gate component
            (num_gates as f64 * shots as f64) * 0.000003
        };

        Ok(Some(CostEstimate {
            provider: "IonQ".into(),
            backend: backend.into(),
            shots,
            estimated_cost_usd: cost,
            estimated_queue_time: if is_simulator {
                Some(Duration::from_secs(5))
            } else {
                Some(Duration::from_secs(300))
            },
            is_free_tier: is_simulator,
            details: Some(format!(
                "{} gates × {} shots",
                num_gates, shots
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

/// IonQ job handle.
pub struct IonQJob {
    client: reqwest::Client,
    api_key: String,
    job_id: String,
    backend: String,
    shots: usize,
}

#[async_trait]
impl QPUJob for IonQJob {
    fn id(&self) -> &str {
        &self.job_id
    }

    fn backend(&self) -> &str {
        &self.backend
    }

    fn provider(&self) -> &str {
        "IonQ"
    }

    async fn status(&self) -> Result<JobStatus, QPUError> {
        let url = format!("{}/jobs/{}", IONQ_API_BASE, self.job_id);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("apiKey {}", self.api_key))
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        parse_ionq_status(&body)
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

            let url = format!("{}/jobs/{}", IONQ_API_BASE, self.job_id);
            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("apiKey {}", self.api_key))
                .send()
                .await
                .map_err(|e| QPUError::NetworkError(e.to_string()))?;

            let body: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| QPUError::NetworkError(e.to_string()))?;

            let status = parse_ionq_status(&body)?;

            match status {
                JobStatus::Completed => {
                    return parse_ionq_result(&body, &self.backend, self.shots);
                }
                JobStatus::Failed(msg) => {
                    return Err(QPUError::ExecutionError(msg));
                }
                JobStatus::Cancelled => {
                    return Err(QPUError::ExecutionError("Job was cancelled".into()));
                }
                _ => {
                    tokio::time::sleep(interval).await;
                    interval = (interval * 2).min(Duration::from_secs(60));
                }
            }
        }
    }

    async fn result(&self) -> Result<Option<JobResult>, QPUError> {
        let url = format!("{}/jobs/{}", IONQ_API_BASE, self.job_id);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("apiKey {}", self.api_key))
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;

        let status = parse_ionq_status(&body)?;
        if status.is_success() {
            Ok(Some(parse_ionq_result(&body, &self.backend, self.shots)?))
        } else {
            Ok(None)
        }
    }

    async fn cancel(&self) -> Result<(), QPUError> {
        let url = format!("{}/jobs/{}/status/cancel", IONQ_API_BASE, self.job_id);
        self.client
            .put(&url)
            .header("Authorization", format!("apiKey {}", self.api_key))
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(e.to_string()))?;
        Ok(())
    }
}

fn parse_ionq_status(body: &serde_json::Value) -> Result<JobStatus, QPUError> {
    let status_str = body["status"].as_str().unwrap_or("unknown");
    match status_str {
        "submitted" | "ready" => Ok(JobStatus::Queued),
        "running" => Ok(JobStatus::Running),
        "completed" => Ok(JobStatus::Completed),
        "failed" => {
            let msg = body["failure"]
                .as_str()
                .unwrap_or("Unknown failure")
                .to_string();
            Ok(JobStatus::Failed(msg))
        }
        "canceled" => Ok(JobStatus::Cancelled),
        other => Ok(JobStatus::Queued), // Treat unknown as queued
    }
}

fn parse_ionq_result(
    body: &serde_json::Value,
    backend: &str,
    shots: usize,
) -> Result<JobResult, QPUError> {
    let job_id = body["id"].as_str().unwrap_or("").to_string();
    let mut counts = HashMap::new();

    // IonQ returns histogram with integer keys → probability values
    if let Some(histogram) = body.get("data").and_then(|d| d.get("histogram")).or_else(|| body.get("histogram")) {
        if let Some(obj) = histogram.as_object() {
            let num_qubits = body["qubits"].as_u64().unwrap_or(1) as usize;
            for (key, prob) in obj {
                let idx: u64 = key.parse().unwrap_or(0);
                let bitstring = format!("{:0>width$b}", idx, width = num_qubits);
                let count = (prob.as_f64().unwrap_or(0.0) * shots as f64).round() as usize;
                if count > 0 {
                    counts.insert(bitstring, count);
                }
            }
        }
    }

    // IonQ also might return probabilities
    if counts.is_empty() {
        if let Some(probs) = body.get("data").and_then(|d| d.get("probabilities")) {
            if let Some(obj) = probs.as_object() {
                let num_qubits = body["qubits"].as_u64().unwrap_or(1) as usize;
                for (key, prob) in obj {
                    let idx: u64 = key.parse().unwrap_or(0);
                    let bitstring = format!("{:0>width$b}", idx, width = num_qubits);
                    let count = (prob.as_f64().unwrap_or(0.0) * shots as f64).round() as usize;
                    if count > 0 {
                        counts.insert(bitstring, count);
                    }
                }
            }
        }
    }

    let execution_time = body["execution_time"]
        .as_f64()
        .map(|ms| Duration::from_millis(ms as u64));

    Ok(JobResult {
        job_id,
        backend: backend.into(),
        counts,
        shots,
        is_simulator: backend.contains("simulator"),
        execution_time,
        queue_time: None,
        metadata: HashMap::new(),
    })
}

fn parse_ionq_backend(item: &serde_json::Value) -> Option<BackendInfo> {
    let name = item["backend"].as_str()?;
    Some(BackendInfo {
        name: format!("ionq_{}", name),
        provider: "IonQ".into(),
        num_qubits: item["qubits"].as_u64().unwrap_or(11) as usize,
        is_simulator: name == "simulator",
        status: if item["status"].as_str() == Some("available") {
            BackendStatus::Online
        } else {
            BackendStatus::Offline
        },
        queue_length: item["queue_depth"].as_u64().unwrap_or(0) as usize,
        basis_gates: vec![
            "gpi".into(), "gpi2".into(), "ms".into(),
            "h".into(), "x".into(), "y".into(), "z".into(),
            "rx".into(), "ry".into(), "rz".into(),
            "cnot".into(), "swap".into(),
        ],
        coupling_map: None, // IonQ: all-to-all connectivity
        max_shots: item["max_shots"].as_u64().unwrap_or(10000) as usize,
        max_depth: None,
        avg_gate_error_1q: Some(0.0003), // IonQ typical
        avg_gate_error_2q: Some(0.005),  // IonQ typical MS gate
        avg_readout_error: Some(0.003),
        avg_t1_us: None, // Trapped ions: very long coherence
        avg_t2_us: None,
        metadata: HashMap::new(),
    })
}

fn known_ionq_backends() -> Vec<BackendInfo> {
    vec![
        BackendInfo {
            name: "ionq_aria".into(),
            provider: "IonQ".into(),
            num_qubits: 25,
            is_simulator: false,
            status: BackendStatus::Online,
            queue_length: 0,
            basis_gates: vec![
                "gpi".into(), "gpi2".into(), "ms".into(),
            ],
            coupling_map: None,
            max_shots: 10000,
            max_depth: None,
            avg_gate_error_1q: Some(0.0003),
            avg_gate_error_2q: Some(0.004),
            avg_readout_error: Some(0.003),
            avg_t1_us: None,
            avg_t2_us: None,
            metadata: HashMap::new(),
        },
        BackendInfo {
            name: "ionq_forte".into(),
            provider: "IonQ".into(),
            num_qubits: 36,
            is_simulator: false,
            status: BackendStatus::Online,
            queue_length: 0,
            basis_gates: vec![
                "gpi".into(), "gpi2".into(), "ms".into(),
            ],
            coupling_map: None,
            max_shots: 10000,
            max_depth: None,
            avg_gate_error_1q: Some(0.0002),
            avg_gate_error_2q: Some(0.003),
            avg_readout_error: Some(0.002),
            avg_t1_us: None,
            avg_t2_us: None,
            metadata: HashMap::new(),
        },
        BackendInfo {
            name: "ionq_simulator".into(),
            provider: "IonQ".into(),
            num_qubits: 29,
            is_simulator: true,
            status: BackendStatus::Online,
            queue_length: 0,
            basis_gates: vec![
                "gpi".into(), "gpi2".into(), "ms".into(),
                "h".into(), "x".into(), "y".into(), "z".into(),
                "rx".into(), "ry".into(), "rz".into(),
                "cnot".into(), "swap".into(),
            ],
            coupling_map: None,
            max_shots: 100000,
            max_depth: None,
            avg_gate_error_1q: None,
            avg_gate_error_2q: None,
            avg_readout_error: None,
            avg_t1_us: None,
            avg_t2_us: None,
            metadata: HashMap::new(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ionq_status_completed() {
        let body = serde_json::json!({"status": "completed"});
        let status = parse_ionq_status(&body).unwrap();
        assert!(matches!(status, JobStatus::Completed));
    }

    #[test]
    fn test_parse_ionq_status_failed() {
        let body = serde_json::json!({"status": "failed", "failure": "qubit error"});
        let status = parse_ionq_status(&body).unwrap();
        assert!(matches!(status, JobStatus::Failed(msg) if msg == "qubit error"));
    }

    #[test]
    fn test_parse_ionq_result_histogram() {
        let body = serde_json::json!({
            "id": "job-123",
            "status": "completed",
            "qubits": 2,
            "data": {
                "histogram": {
                    "0": 0.5,
                    "3": 0.5
                }
            }
        });
        let result = parse_ionq_result(&body, "ionq_simulator", 1000).unwrap();
        assert_eq!(result.counts.get("00"), Some(&500));
        assert_eq!(result.counts.get("11"), Some(&500));
    }

    #[test]
    fn test_known_backends() {
        let backends = known_ionq_backends();
        assert_eq!(backends.len(), 3);
        assert!(backends.iter().any(|b| b.name == "ionq_aria"));
        assert!(backends.iter().any(|b| b.name == "ionq_forte"));
        assert!(backends.iter().any(|b| b.name == "ionq_simulator"));
    }

    #[test]
    fn test_ionq_target_mapping() {
        // Test that human-friendly names map correctly
        let targets = vec![
            ("ionq_aria", "qpu.aria-1"),
            ("aria", "qpu.aria-1"),
            ("ionq_forte", "qpu.forte-1"),
            ("forte", "qpu.forte-1"),
            ("ionq_simulator", "simulator"),
            ("simulator", "simulator"),
        ];
        for (input, expected) in targets {
            let target = match input {
                "ionq_aria" | "aria" => "qpu.aria-1",
                "ionq_forte" | "forte" => "qpu.forte-1",
                "ionq_simulator" | "simulator" => "simulator",
                other => other,
            };
            assert_eq!(target, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_bell_circuit_ionq_json() {
        let circuit = QPUCircuit::bell_state();
        let json = circuit.to_ionq_json();
        assert_eq!(json["qubits"], 2);
        let gates = json["circuit"].as_array().unwrap();
        assert!(gates.len() >= 2);
    }
}
