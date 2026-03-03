//! Azure Quantum REST API provider.
//!
//! Connects to Azure Quantum workspaces at
//! `https://{workspace}.{region}.quantum.azure.com/`.  Authentication uses
//! OAuth2 client credentials (Azure AD), with automatic token refresh.
//!
//! # Endpoints
//!
//! | Method | Path             | Purpose                        |
//! |--------|------------------|--------------------------------|
//! | GET    | `/providers`     | List available providers/targets |
//! | POST   | `/jobs`          | Submit a quantum job           |
//! | GET    | `/jobs/{jobId}`  | Get job status and results     |
//!
//! # Supported Targets
//!
//! - `ionq.simulator` — IonQ cloud simulator
//! - `ionq.qpu.aria-1` — IonQ Aria trapped-ion QPU
//! - `quantinuum.sim.h1-1sc` — Quantinuum H1-1 syntax checker / simulator
//! - `quantinuum.qpu.h1-1` — Quantinuum H1-1 trapped-ion QPU

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

const AZURE_PROVIDER_NAME: &str = "Azure Quantum";
const MAX_BACKOFF_SECS: u64 = 60;
const TOKEN_REFRESH_MARGIN_SECS: u64 = 60;
const AZURE_QUANTUM_SCOPE: &str = "https://quantum.microsoft.com/.default";

/// Well-known Azure Quantum target IDs and their properties.
const KNOWN_TARGETS: &[(&str, &str, usize, bool)] = &[
    ("ionq.simulator", "IonQ", 29, true),
    ("ionq.qpu.aria-1", "IonQ", 25, false),
    ("ionq.qpu.aria-2", "IonQ", 25, false),
    ("ionq.qpu.forte-1", "IonQ", 36, false),
    ("quantinuum.sim.h1-1sc", "Quantinuum", 20, true),
    ("quantinuum.sim.h1-1e", "Quantinuum", 20, true),
    ("quantinuum.qpu.h1-1", "Quantinuum", 20, false),
    ("quantinuum.qpu.h1-2", "Quantinuum", 32, false),
    ("rigetti.sim.qvm", "Rigetti", 32, true),
    ("rigetti.qpu.ankaa-2", "Rigetti", 84, false),
];

// ---------------------------------------------------------------------------
// Cached OAuth2 token
// ---------------------------------------------------------------------------

/// Thread-safe cached bearer token with expiration tracking.
#[derive(Debug, Clone)]
struct CachedToken {
    access_token: String,
    /// Epoch seconds when the token expires.
    expires_at: u64,
}

impl CachedToken {
    /// Returns true if the token is still valid (with a safety margin).
    fn is_valid(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now + TOKEN_REFRESH_MARGIN_SECS < self.expires_at
    }
}

// ---------------------------------------------------------------------------
// AzureProvider
// ---------------------------------------------------------------------------

/// Azure Quantum REST API provider.
///
/// Submits circuits as OpenQASM strings to quantum targets available through
/// an Azure Quantum workspace.  Handles OAuth2 token acquisition and automatic
/// refresh via Azure AD client credentials flow.
///
/// # Construction
///
/// ```no_run
/// # use nqpu_metal::qpu::azure::AzureProvider;
/// // From environment variables
/// let provider = AzureProvider::from_env().expect("Azure credentials not set");
/// ```
pub struct AzureProvider {
    client: reqwest::Client,
    subscription_id: String,
    resource_group: String,
    workspace: String,
    location: String,
    tenant_id: String,
    client_id: String,
    client_secret: String,
    base_url: String,
    /// Cached OAuth2 bearer token (thread-safe).
    token_cache: Arc<RwLock<Option<CachedToken>>>,
}

impl AzureProvider {
    /// Create a provider from explicit credentials.
    pub fn new(
        subscription_id: String,
        resource_group: String,
        workspace: String,
        location: String,
        tenant_id: String,
        client_id: String,
        client_secret: String,
    ) -> Self {
        let base_url = format!(
            "https://{}.{}.quantum.azure.com",
            workspace, location
        );
        Self {
            client: reqwest::Client::new(),
            subscription_id,
            resource_group,
            workspace,
            location,
            tenant_id,
            client_id,
            client_secret,
            base_url,
            token_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a provider from Azure environment variables.
    ///
    /// Reads:
    /// - `AZURE_QUANTUM_SUBSCRIPTION_ID`
    /// - `AZURE_QUANTUM_RESOURCE_GROUP`
    /// - `AZURE_QUANTUM_WORKSPACE`
    /// - `AZURE_QUANTUM_LOCATION` (defaults to `eastus`)
    /// - `AZURE_TENANT_ID`
    /// - `AZURE_CLIENT_ID`
    /// - `AZURE_CLIENT_SECRET`
    pub fn from_env() -> Result<Self, QPUError> {
        let subscription_id = required_env("AZURE_QUANTUM_SUBSCRIPTION_ID")?;
        let resource_group = required_env("AZURE_QUANTUM_RESOURCE_GROUP")?;
        let workspace = required_env("AZURE_QUANTUM_WORKSPACE")?;
        let location = std::env::var("AZURE_QUANTUM_LOCATION")
            .unwrap_or_else(|_| "eastus".to_string());
        let tenant_id = required_env("AZURE_TENANT_ID")?;
        let client_id = required_env("AZURE_CLIENT_ID")?;
        let client_secret = required_env("AZURE_CLIENT_SECRET")?;

        Ok(Self::new(
            subscription_id,
            resource_group,
            workspace,
            location,
            tenant_id,
            client_id,
            client_secret,
        ))
    }

    /// Override the base URL (useful for testing against a mock server).
    #[cfg(test)]
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    // -- Token management ---------------------------------------------------

    /// Get a valid bearer token, refreshing from Azure AD if necessary.
    async fn get_token(&self) -> Result<String, QPUError> {
        // Check cache first.
        {
            let cache = self.token_cache.read().map_err(|_| {
                QPUError::ProviderError {
                    provider: AZURE_PROVIDER_NAME.into(),
                    message: "Token cache lock poisoned".into(),
                    status_code: None,
                }
            })?;
            if let Some(ref cached) = *cache {
                if cached.is_valid() {
                    return Ok(cached.access_token.clone());
                }
            }
        }

        // Token expired or not yet acquired; refresh.
        let new_token = self.fetch_oauth2_token().await?;

        // Store in cache.
        {
            let mut cache = self.token_cache.write().map_err(|_| {
                QPUError::ProviderError {
                    provider: AZURE_PROVIDER_NAME.into(),
                    message: "Token cache lock poisoned".into(),
                    status_code: None,
                }
            })?;
            *cache = Some(new_token.clone());
        }

        Ok(new_token.access_token)
    }

    /// Acquire a fresh OAuth2 token via the Azure AD client credentials flow.
    ///
    /// POST `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`
    async fn fetch_oauth2_token(&self) -> Result<CachedToken, QPUError> {
        let token_url = format!(
            "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
            self.tenant_id
        );

        let params = [
            ("grant_type", "client_credentials"),
            ("client_id", &self.client_id),
            ("client_secret", &self.client_secret),
            ("scope", AZURE_QUANTUM_SCOPE),
        ];

        let response = self
            .client
            .post(&token_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| {
                QPUError::AuthenticationError(format!(
                    "Failed to reach Azure AD token endpoint: {}",
                    e
                ))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            QPUError::AuthenticationError(format!("Failed to read Azure AD response: {}", e))
        })?;

        if !status.is_success() {
            return Err(QPUError::AuthenticationError(format!(
                "Azure AD token request failed (HTTP {}): {}",
                status,
                &body[..body.len().min(500)]
            )));
        }

        let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            QPUError::AuthenticationError(format!("Invalid token response JSON: {}", e))
        })?;

        let access_token = json["access_token"]
            .as_str()
            .ok_or_else(|| {
                QPUError::AuthenticationError("Missing access_token in Azure AD response".into())
            })?
            .to_string();

        let expires_in = json["expires_in"].as_u64().unwrap_or(3600);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(CachedToken {
            access_token,
            expires_at: now + expires_in,
        })
    }

    // -- HTTP helpers -------------------------------------------------------

    /// Build the API version query parameter.
    fn api_version_param() -> &'static str {
        "api-version=2024-03-01-preview"
    }

    /// Build a full URL with the API version query parameter.
    fn url(&self, path: &str) -> String {
        let sep = if path.contains('?') { '&' } else { '?' };
        format!(
            "{}{}{}{}",
            self.base_url,
            path,
            sep,
            Self::api_version_param()
        )
    }

    /// Execute an authenticated GET and return JSON.
    async fn auth_get(&self, path: &str) -> Result<serde_json::Value, QPUError> {
        let token = self.get_token().await?;
        let url = self.url(path);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Azure request failed: {}", e)))?;

        self.handle_response(response).await
    }

    /// Execute an authenticated POST with a JSON body and return JSON.
    async fn auth_post(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, QPUError> {
        let token = self.get_token().await?;
        let url = self.url(path);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&token)
            .json(body)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Azure request failed: {}", e)))?;

        self.handle_response(response).await
    }

    /// Process an HTTP response, handling common error codes.
    async fn handle_response(
        &self,
        response: reqwest::Response,
    ) -> Result<serde_json::Value, QPUError> {
        let status = response.status();

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
            .map_err(|e| QPUError::NetworkError(format!("Read Azure response: {}", e)))?;

        if !status.is_success() {
            return Err(Self::map_http_error(status, &body));
        }

        // Some Azure endpoints return 204 No Content.
        if body.is_empty() {
            return Ok(serde_json::json!({}));
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ProviderError {
            provider: AZURE_PROVIDER_NAME.into(),
            message: format!(
                "Invalid JSON from Azure: {} (body: {})",
                e,
                &body[..body.len().min(200)]
            ),
            status_code: None,
        })
    }

    /// Map an HTTP error to a [`QPUError`].
    fn map_http_error(status: reqwest::StatusCode, body: &str) -> QPUError {
        match status.as_u16() {
            401 | 403 => QPUError::AuthenticationError(format!(
                "Azure authentication failed (HTTP {}): {}",
                status,
                &body[..body.len().min(300)]
            )),
            404 => QPUError::BackendUnavailable(format!(
                "Not found: {}",
                &body[..body.len().min(300)]
            )),
            _ => QPUError::ProviderError {
                provider: AZURE_PROVIDER_NAME.into(),
                message: body[..body.len().min(500)].to_string(),
                status_code: Some(status.as_u16()),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// QPUProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl QPUProvider for AzureProvider {
    fn name(&self) -> &str {
        AZURE_PROVIDER_NAME
    }

    async fn list_backends(&self) -> Result<Vec<BackendInfo>, QPUError> {
        let json = self.auth_get("/providers").await?;

        // Azure returns a structure with provider objects, each containing targets.
        let mut backends = Vec::new();

        let providers = json["value"]
            .as_array()
            .or_else(|| json.as_array());

        if let Some(providers_arr) = providers {
            for provider in providers_arr {
                let provider_id = provider["providerId"]
                    .as_str()
                    .or_else(|| provider["id"].as_str())
                    .unwrap_or("unknown");

                if let Some(targets) = provider["targets"].as_array() {
                    for target in targets {
                        if let Some(info) = parse_azure_target(target, provider_id) {
                            backends.push(info);
                        }
                    }
                }
            }
        }

        // If the API returned an empty list, supplement with known targets.
        if backends.is_empty() {
            for &(target_id, provider_name, qubits, is_sim) in KNOWN_TARGETS {
                backends.push(make_known_target_info(target_id, provider_name, qubits, is_sim));
            }
        }

        Ok(backends)
    }

    async fn get_backend(&self, name: &str) -> Result<BackendInfo, QPUError> {
        // Try the API first.
        let backends = self.list_backends().await?;
        if let Some(info) = backends.into_iter().find(|b| b.name == name) {
            return Ok(info);
        }

        // Fall back to known targets.
        for &(target_id, provider_name, qubits, is_sim) in KNOWN_TARGETS {
            if target_id == name {
                return Ok(make_known_target_info(
                    target_id,
                    provider_name,
                    qubits,
                    is_sim,
                ));
            }
        }

        Err(QPUError::BackendUnavailable(format!(
            "Target '{}' not found in Azure Quantum workspace",
            name
        )))
    }

    async fn submit_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        config: &JobConfig,
    ) -> Result<Box<dyn QPUJob>, QPUError> {
        // Determine the input format based on the target.
        let (input_format, input_data) = if backend.contains("ionq") {
            // IonQ targets accept their native JSON format.
            let ionq_json = circuit.to_ionq_json();
            ("ionq.circuit.v1", serde_json::to_string(&ionq_json).map_err(|e| {
                QPUError::ConversionError(format!("Failed to serialize IonQ circuit: {}", e))
            })?)
        } else {
            // Most targets accept OpenQASM.
            ("openqasm3", circuit.to_qasm3())
        };

        let mut body = serde_json::json!({
            "id": format!(
                "nqpu-metal-{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            ),
            "name": config.name.clone().unwrap_or_else(|| "nqpu-metal-job".to_string()),
            "providerId": target_provider_id(backend),
            "target": backend,
            "inputDataFormat": input_format,
            "outputDataFormat": "microsoft.quantum-results.v2",
            "inputData": input_data,
            "inputParams": {
                "shots": config.shots
            }
        });

        // Merge extra parameters.
        if let Some(params) = body["inputParams"].as_object_mut() {
            for (k, v) in &config.extra {
                params.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
        }

        let json = self.auth_post("/jobs", &body).await?;

        let job_id = json["id"]
            .as_str()
            .ok_or_else(|| QPUError::SubmissionError("Missing 'id' in Azure job response".into()))?
            .to_string();

        Ok(Box::new(AzureJob {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            token_cache: self.token_cache.clone(),
            tenant_id: self.tenant_id.clone(),
            client_id: self.client_id.clone(),
            client_secret: self.client_secret.clone(),
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
        let is_simulator = backend.contains("sim");

        // Azure Quantum pricing varies by provider.
        let (cost, details) = if backend.contains("ionq") {
            if is_simulator {
                (0.0, "IonQ simulator: free on Azure Quantum")
            } else {
                // IonQ on Azure: ~$0.01/gate per shot (single-qubit)
                //                ~$0.03/gate per shot (multi-qubit)
                let num_1q: usize = circuit
                    .gates
                    .iter()
                    .filter(|g| g.two_qubit_operands().is_none())
                    .count();
                let num_2q: usize = circuit
                    .gates
                    .iter()
                    .filter(|g| g.two_qubit_operands().is_some())
                    .count();
                let gate_cost = (num_1q as f64 * 0.00003 + num_2q as f64 * 0.0003) * shots as f64;
                (
                    gate_cost,
                    "IonQ QPU: ~$0.00003/1Q-gate-shot + $0.0003/2Q-gate-shot",
                )
            }
        } else if backend.contains("quantinuum") {
            if is_simulator {
                // H1 emulator uses H-System Credits (HQC)
                (0.0, "Quantinuum simulator: free (syntax checker)")
            } else {
                // Quantinuum: pricing in HQC, roughly $1/HQC
                // HQC formula: 5 + N_1q * 0.0625 + N_2q * 10 + N_m * 6.4166
                let n_1q = circuit
                    .gates
                    .iter()
                    .filter(|g| g.two_qubit_operands().is_none())
                    .count() as f64;
                let n_2q = circuit
                    .gates
                    .iter()
                    .filter(|g| g.two_qubit_operands().is_some())
                    .count() as f64;
                let n_m = circuit.measurements.len() as f64;
                let hqc = 5.0 + n_1q * 0.0625 + n_2q * 10.0 + n_m * 6.4166;
                let cost = hqc * shots as f64;
                (
                    cost,
                    "Quantinuum QPU: HQC-based pricing (~$1/HQC)",
                )
            }
        } else {
            (0.0, "Unknown target: cost estimation unavailable")
        };

        Ok(Some(CostEstimate {
            provider: AZURE_PROVIDER_NAME.into(),
            backend: backend.into(),
            shots,
            estimated_cost_usd: cost,
            estimated_queue_time: None,
            is_free_tier: is_simulator,
            details: Some(details.into()),
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
// AzureJob
// ---------------------------------------------------------------------------

/// Handle for a job submitted to Azure Quantum.
pub struct AzureJob {
    client: reqwest::Client,
    base_url: String,
    token_cache: Arc<RwLock<Option<CachedToken>>>,
    tenant_id: String,
    client_id: String,
    client_secret: String,
    job_id: String,
    backend_name: String,
}

impl AzureJob {
    /// Get a valid bearer token, refreshing if necessary.
    async fn get_token(&self) -> Result<String, QPUError> {
        // Check cache.
        {
            let cache = self.token_cache.read().map_err(|_| {
                QPUError::ProviderError {
                    provider: AZURE_PROVIDER_NAME.into(),
                    message: "Token cache lock poisoned".into(),
                    status_code: None,
                }
            })?;
            if let Some(ref cached) = *cache {
                if cached.is_valid() {
                    return Ok(cached.access_token.clone());
                }
            }
        }

        // Refresh token.
        let token_url = format!(
            "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
            self.tenant_id
        );

        let params = [
            ("grant_type", "client_credentials"),
            ("client_id", self.client_id.as_str()),
            ("client_secret", self.client_secret.as_str()),
            ("scope", AZURE_QUANTUM_SCOPE),
        ];

        let response = self
            .client
            .post(&token_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| {
                QPUError::AuthenticationError(format!("Azure AD token refresh failed: {}", e))
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            QPUError::AuthenticationError(format!("Read Azure AD response: {}", e))
        })?;

        if !status.is_success() {
            return Err(QPUError::AuthenticationError(format!(
                "Azure AD token refresh failed (HTTP {}): {}",
                status,
                &body[..body.len().min(500)]
            )));
        }

        let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            QPUError::AuthenticationError(format!("Invalid token JSON: {}", e))
        })?;

        let access_token = json["access_token"]
            .as_str()
            .ok_or_else(|| {
                QPUError::AuthenticationError("Missing access_token in refresh response".into())
            })?
            .to_string();

        let expires_in = json["expires_in"].as_u64().unwrap_or(3600);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let new_token = CachedToken {
            access_token: access_token.clone(),
            expires_at: now + expires_in,
        };

        // Update cache.
        if let Ok(mut cache) = self.token_cache.write() {
            *cache = Some(new_token);
        }

        Ok(access_token)
    }

    /// Build the full URL for a job endpoint.
    fn job_url(&self) -> String {
        format!(
            "{}/jobs/{}?{}",
            self.base_url,
            self.job_id,
            AzureProvider::api_version_param()
        )
    }

    /// Fetch the job JSON from Azure.
    async fn fetch_job_json(&self) -> Result<serde_json::Value, QPUError> {
        let token = self.get_token().await?;
        let url = self.job_url();

        let response = self
            .client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Azure request failed: {}", e)))?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(30);
            return Err(QPUError::RateLimited {
                retry_after: Duration::from_secs(retry),
            });
        }

        let body = response
            .text()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Read Azure response: {}", e)))?;

        if !status.is_success() {
            return Err(AzureProvider::map_http_error(status, &body));
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ProviderError {
            provider: AZURE_PROVIDER_NAME.into(),
            message: format!("Invalid JSON: {}", e),
            status_code: None,
        })
    }
}

#[async_trait]
impl QPUJob for AzureJob {
    fn id(&self) -> &str {
        &self.job_id
    }

    fn backend(&self) -> &str {
        &self.backend_name
    }

    fn provider(&self) -> &str {
        AZURE_PROVIDER_NAME
    }

    async fn status(&self) -> Result<JobStatus, QPUError> {
        let json = self.fetch_job_json().await?;
        Ok(parse_azure_job_status(&json))
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

            let json = self.fetch_job_json().await?;
            let status = parse_azure_job_status(&json);

            match status {
                JobStatus::Completed => {
                    return parse_azure_result(&json, &self.job_id, &self.backend_name);
                }
                JobStatus::Failed(msg) => {
                    return Err(QPUError::ExecutionError(format!(
                        "Azure job {} failed: {}",
                        self.job_id, msg
                    )));
                }
                JobStatus::Cancelled => {
                    return Err(QPUError::ExecutionError(format!(
                        "Azure job {} was cancelled",
                        self.job_id
                    )));
                }
                _ => {
                    tokio::time::sleep(interval).await;
                    interval = interval
                        .mul_f64(1.5)
                        .min(Duration::from_secs(MAX_BACKOFF_SECS));
                }
            }
        }
    }

    async fn result(&self) -> Result<Option<JobResult>, QPUError> {
        let json = self.fetch_job_json().await?;
        let status = parse_azure_job_status(&json);

        if status == JobStatus::Completed {
            parse_azure_result(&json, &self.job_id, &self.backend_name).map(Some)
        } else {
            Ok(None)
        }
    }

    async fn cancel(&self) -> Result<(), QPUError> {
        let token = self.get_token().await?;
        let url = format!(
            "{}/jobs/{}?{}",
            self.base_url,
            self.job_id,
            AzureProvider::api_version_param()
        );

        let response = self
            .client
            .delete(&url)
            .bearer_auth(&token)
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Cancel request failed: {}", e)))?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(QPUError::ProviderError {
                provider: AZURE_PROVIDER_NAME.into(),
                message: format!("Failed to cancel job: {}", body),
                status_code: None,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JSON parsing helpers
// ---------------------------------------------------------------------------

/// Read a required environment variable.
fn required_env(name: &str) -> Result<String, QPUError> {
    std::env::var(name).map_err(|_| {
        QPUError::ConfigError(format!("{} environment variable not set", name))
    })
}

/// Map a target ID to its provider ID for the Azure API.
fn target_provider_id(target: &str) -> &str {
    if target.starts_with("ionq") {
        "ionq"
    } else if target.starts_with("quantinuum") {
        "quantinuum"
    } else if target.starts_with("rigetti") {
        "rigetti"
    } else if target.starts_with("microsoft") {
        "microsoft-qc"
    } else {
        "unknown"
    }
}

/// Create a [`BackendInfo`] for a known Azure Quantum target.
fn make_known_target_info(
    target_id: &str,
    provider_name: &str,
    qubits: usize,
    is_simulator: bool,
) -> BackendInfo {
    let basis_gates = if target_id.contains("ionq") {
        vec!["gpi".into(), "gpi2".into(), "ms".into()]
    } else if target_id.contains("quantinuum") {
        vec!["rz".into(), "ry".into(), "rx".into(), "zz".into()]
    } else if target_id.contains("rigetti") {
        vec!["rx".into(), "rz".into(), "cz".into()]
    } else {
        vec![]
    };

    BackendInfo {
        name: target_id.to_string(),
        provider: format!("Azure Quantum ({})", provider_name),
        num_qubits: qubits,
        is_simulator,
        status: BackendStatus::Online,
        queue_length: 0,
        basis_gates,
        coupling_map: None, // Azure does not typically expose coupling maps
        max_shots: if target_id.contains("ionq") {
            10_000
        } else if target_id.contains("quantinuum") {
            10_000
        } else {
            100_000
        },
        max_depth: None,
        avg_gate_error_1q: None,
        avg_gate_error_2q: None,
        avg_readout_error: None,
        avg_t1_us: None,
        avg_t2_us: None,
        metadata: HashMap::new(),
    }
}

/// Parse an Azure Quantum target JSON into a [`BackendInfo`].
fn parse_azure_target(json: &serde_json::Value, provider_id: &str) -> Option<BackendInfo> {
    let target_id = json["id"]
        .as_str()
        .or_else(|| json["targetId"].as_str())?
        .to_string();

    let current_availability = json["currentAvailability"]
        .as_str()
        .unwrap_or("Unavailable");
    let status = match current_availability.to_lowercase().as_str() {
        "available" => BackendStatus::Online,
        "degraded" => BackendStatus::Maintenance,
        _ => BackendStatus::Offline,
    };

    let is_simulator = target_id.contains("sim")
        || json["targetType"].as_str().map(|t| t.contains("Simulator")).unwrap_or(false);

    // Try to find qubit count from known targets.
    let num_qubits = KNOWN_TARGETS
        .iter()
        .find(|(id, _, _, _)| *id == target_id)
        .map(|(_, _, q, _)| *q)
        .unwrap_or(0);

    let basis_gates = if target_id.contains("ionq") {
        vec!["gpi".into(), "gpi2".into(), "ms".into()]
    } else if target_id.contains("quantinuum") {
        vec!["rz".into(), "ry".into(), "rx".into(), "zz".into()]
    } else if target_id.contains("rigetti") {
        vec!["rx".into(), "rz".into(), "cz".into()]
    } else {
        vec![]
    };

    let mut metadata = HashMap::new();
    if let Some(desc) = json["description"].as_str() {
        metadata.insert(
            "description".into(),
            serde_json::Value::String(desc.into()),
        );
    }
    metadata.insert(
        "provider_id".into(),
        serde_json::Value::String(provider_id.into()),
    );

    Some(BackendInfo {
        name: target_id,
        provider: format!("Azure Quantum ({})", provider_id),
        num_qubits,
        is_simulator,
        status,
        queue_length: 0,
        basis_gates,
        coupling_map: None,
        max_shots: 100_000,
        max_depth: None,
        avg_gate_error_1q: None,
        avg_gate_error_2q: None,
        avg_readout_error: None,
        avg_t1_us: None,
        avg_t2_us: None,
        metadata,
    })
}

/// Parse Azure Quantum job status.
fn parse_azure_job_status(json: &serde_json::Value) -> JobStatus {
    match json["status"]
        .as_str()
        .unwrap_or("Unknown")
        .to_lowercase()
        .as_str()
    {
        "waiting" | "submitting" => JobStatus::Initializing,
        "executing" | "running" => JobStatus::Running,
        "succeeded" | "completed" => JobStatus::Completed,
        "failed" => {
            let msg = json["errorData"]["message"]
                .as_str()
                .or_else(|| json["error"]["message"].as_str())
                .or_else(|| json["errorMessage"].as_str())
                .unwrap_or("Unknown error")
                .to_string();
            JobStatus::Failed(msg)
        }
        "cancelled" | "canceled" => JobStatus::Cancelled,
        other => JobStatus::Failed(format!("Unknown Azure status: {}", other)),
    }
}

/// Parse a completed Azure Quantum job into a [`JobResult`].
fn parse_azure_result(
    json: &serde_json::Value,
    job_id: &str,
    backend_name: &str,
) -> Result<JobResult, QPUError> {
    let mut counts = HashMap::new();

    // Azure returns results in different formats depending on the provider.
    let output = &json["outputData"];
    let output_parsed: serde_json::Value = if let Some(s) = output.as_str() {
        serde_json::from_str(s).unwrap_or(serde_json::json!({}))
    } else {
        output.clone()
    };

    // IonQ format: {"histogram": {"0": 0.5, "3": 0.5}}
    if let Some(histogram) = output_parsed["histogram"].as_object() {
        let shots = json["inputParams"]["shots"]
            .as_u64()
            .unwrap_or(1000) as f64;
        for (key, prob) in histogram {
            let p = prob.as_f64().unwrap_or(0.0);
            let c = (p * shots).round() as usize;
            if c > 0 {
                // Convert integer key to binary string
                if let Ok(num) = key.parse::<u64>() {
                    counts.insert(format!("{:b}", num), c);
                } else {
                    counts.insert(key.clone(), c);
                }
            }
        }
    }
    // Quantinuum / generic format: {"c": ["00", "11", "00", ...]}
    else if let Some(results) = output_parsed["c"].as_array() {
        for result in results {
            if let Some(bitstring) = result.as_str() {
                *counts.entry(bitstring.to_string()).or_insert(0) += 1;
            } else if let Some(bits) = result.as_array() {
                let bitstring: String = bits
                    .iter()
                    .map(|b| if b.as_u64().unwrap_or(0) == 1 { '1' } else { '0' })
                    .collect();
                *counts.entry(bitstring).or_insert(0) += 1;
            }
        }
    }
    // Microsoft format: {"Counts": {"00": 512, "11": 512}}
    else if let Some(count_obj) = output_parsed["Counts"].as_object()
        .or_else(|| output_parsed["counts"].as_object())
    {
        for (bitstring, count) in count_obj {
            let c = count.as_u64().unwrap_or(0) as usize;
            counts.insert(bitstring.clone(), c);
        }
    }

    let shots = json["inputParams"]["shots"]
        .as_u64()
        .unwrap_or_else(|| counts.values().sum::<usize>() as u64)
        as usize;

    let is_simulator = backend_name.contains("sim");

    let execution_time = json["beginExecutionTime"]
        .as_str()
        .and_then(|_begin| {
            json["endExecutionTime"].as_str().map(|_end| {
                // In a real implementation we would parse ISO 8601 timestamps
                // and compute the difference.  For now return a placeholder.
                Duration::from_secs(0)
            })
        });

    let mut metadata = HashMap::new();
    if let Some(cost) = json["costEstimate"].as_object() {
        metadata.insert("cost_estimate".into(), serde_json::Value::Object(cost.clone()));
    }
    if let Some(target) = json["target"].as_str() {
        metadata.insert("target".into(), serde_json::Value::String(target.into()));
    }
    if let Some(provider) = json["providerId"].as_str() {
        metadata.insert(
            "provider_id".into(),
            serde_json::Value::String(provider.into()),
        );
    }

    Ok(JobResult {
        job_id: job_id.to_string(),
        backend: backend_name.to_string(),
        counts,
        shots,
        is_simulator,
        execution_time,
        queue_time: None,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Known target info --------------------------------------------------

    #[test]
    fn test_known_targets_completeness() {
        assert!(KNOWN_TARGETS.len() >= 8);
        // Verify each entry has a non-empty target ID.
        for (id, provider, qubits, _) in KNOWN_TARGETS {
            assert!(!id.is_empty());
            assert!(!provider.is_empty());
            assert!(*qubits > 0);
        }
    }

    #[test]
    fn test_make_known_target_info_ionq_simulator() {
        let info = make_known_target_info("ionq.simulator", "IonQ", 29, true);
        assert_eq!(info.name, "ionq.simulator");
        assert!(info.provider.contains("IonQ"));
        assert_eq!(info.num_qubits, 29);
        assert!(info.is_simulator);
        assert_eq!(info.status, BackendStatus::Online);
        assert!(info.basis_gates.contains(&"gpi".to_string()));
        assert!(info.basis_gates.contains(&"ms".to_string()));
    }

    #[test]
    fn test_make_known_target_info_quantinuum_qpu() {
        let info = make_known_target_info("quantinuum.qpu.h1-1", "Quantinuum", 20, false);
        assert!(!info.is_simulator);
        assert_eq!(info.num_qubits, 20);
        assert!(info.basis_gates.contains(&"zz".to_string()));
    }

    // -- Target provider ID mapping -----------------------------------------

    #[test]
    fn test_target_provider_id() {
        assert_eq!(target_provider_id("ionq.simulator"), "ionq");
        assert_eq!(target_provider_id("ionq.qpu.aria-1"), "ionq");
        assert_eq!(target_provider_id("quantinuum.sim.h1-1sc"), "quantinuum");
        assert_eq!(target_provider_id("quantinuum.qpu.h1-1"), "quantinuum");
        assert_eq!(target_provider_id("rigetti.sim.qvm"), "rigetti");
        assert_eq!(target_provider_id("microsoft.estimator"), "microsoft-qc");
        assert_eq!(target_provider_id("unknown.device"), "unknown");
    }

    // -- Target parsing -----------------------------------------------------

    fn sample_azure_target_json() -> serde_json::Value {
        serde_json::json!({
            "id": "ionq.qpu.aria-1",
            "currentAvailability": "Available",
            "targetType": "QuantumProcessor",
            "description": "IonQ Aria 25-qubit QPU"
        })
    }

    #[test]
    fn test_parse_azure_target_basic() {
        let info =
            parse_azure_target(&sample_azure_target_json(), "ionq").expect("should parse target");
        assert_eq!(info.name, "ionq.qpu.aria-1");
        assert!(info.provider.contains("ionq"));
        assert_eq!(info.num_qubits, 25); // from KNOWN_TARGETS
        assert!(!info.is_simulator);
        assert_eq!(info.status, BackendStatus::Online);
    }

    #[test]
    fn test_parse_azure_target_simulator() {
        let json = serde_json::json!({
            "id": "ionq.simulator",
            "currentAvailability": "Available",
            "targetType": "Simulator"
        });
        let info = parse_azure_target(&json, "ionq").expect("should parse simulator");
        assert!(info.is_simulator);
    }

    #[test]
    fn test_parse_azure_target_degraded() {
        let json = serde_json::json!({
            "id": "quantinuum.qpu.h1-1",
            "currentAvailability": "Degraded"
        });
        let info = parse_azure_target(&json, "quantinuum").expect("should parse");
        assert_eq!(info.status, BackendStatus::Maintenance);
    }

    #[test]
    fn test_parse_azure_target_unavailable() {
        let json = serde_json::json!({
            "id": "rigetti.qpu.ankaa-2",
            "currentAvailability": "Unavailable"
        });
        let info = parse_azure_target(&json, "rigetti").expect("should parse");
        assert_eq!(info.status, BackendStatus::Offline);
    }

    #[test]
    fn test_parse_azure_target_missing_id() {
        let json = serde_json::json!({"currentAvailability": "Available"});
        assert!(parse_azure_target(&json, "ionq").is_none());
    }

    #[test]
    fn test_parse_azure_target_with_target_id_field() {
        let json = serde_json::json!({
            "targetId": "quantinuum.sim.h1-1sc",
            "currentAvailability": "Available"
        });
        let info = parse_azure_target(&json, "quantinuum").expect("should parse targetId");
        assert_eq!(info.name, "quantinuum.sim.h1-1sc");
        assert!(info.is_simulator);
    }

    // -- Job status parsing -------------------------------------------------

    #[test]
    fn test_parse_azure_job_status_all_states() {
        let cases = vec![
            ("Waiting", false, false),
            ("Submitting", false, false),
            ("Executing", false, false),
            ("Running", false, false),
            ("Succeeded", true, true),
            ("Completed", true, true),
            ("Cancelled", true, false),
            ("Canceled", true, false),
        ];

        for (status_str, is_terminal, is_success) in cases {
            let json = serde_json::json!({"status": status_str});
            let status = parse_azure_job_status(&json);
            assert_eq!(
                status.is_terminal(),
                is_terminal,
                "Status '{}' terminal check",
                status_str
            );
            assert_eq!(
                status.is_success(),
                is_success,
                "Status '{}' success check",
                status_str
            );
        }
    }

    #[test]
    fn test_parse_azure_job_status_failed() {
        let json = serde_json::json!({
            "status": "Failed",
            "errorData": {
                "message": "Circuit too deep for target"
            }
        });
        match parse_azure_job_status(&json) {
            JobStatus::Failed(msg) => assert_eq!(msg, "Circuit too deep for target"),
            other => panic!("Expected Failed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_azure_job_status_failed_alt_field() {
        let json = serde_json::json!({
            "status": "Failed",
            "error": {"message": "Backend offline"}
        });
        match parse_azure_job_status(&json) {
            JobStatus::Failed(msg) => assert_eq!(msg, "Backend offline"),
            other => panic!("Expected Failed, got {:?}", other),
        }
    }

    // -- Result parsing: IonQ histogram format ------------------------------

    #[test]
    fn test_parse_azure_result_ionq_histogram() {
        let json = serde_json::json!({
            "status": "Succeeded",
            "target": "ionq.qpu.aria-1",
            "inputParams": {"shots": 1000},
            "outputData": serde_json::json!({
                "histogram": {
                    "0": 0.5,
                    "3": 0.5
                }
            }).to_string()
        });
        let result =
            parse_azure_result(&json, "job-1", "ionq.qpu.aria-1").expect("should parse IonQ");
        assert_eq!(result.shots, 1000);
        assert_eq!(result.counts.get("0"), Some(&500));
        assert_eq!(result.counts.get("11"), Some(&500)); // 3 -> "11"
        assert!(!result.is_simulator);
    }

    // -- Result parsing: Quantinuum array format ----------------------------

    #[test]
    fn test_parse_azure_result_quantinuum_array() {
        let json = serde_json::json!({
            "status": "Succeeded",
            "target": "quantinuum.sim.h1-1sc",
            "inputParams": {"shots": 4},
            "outputData": serde_json::json!({
                "c": ["00", "00", "11", "11"]
            }).to_string()
        });
        let result = parse_azure_result(&json, "job-2", "quantinuum.sim.h1-1sc")
            .expect("should parse Quantinuum");
        assert_eq!(result.shots, 4);
        assert_eq!(result.counts.get("00"), Some(&2));
        assert_eq!(result.counts.get("11"), Some(&2));
        assert!(result.is_simulator);
    }

    // -- Result parsing: Direct counts format -------------------------------

    #[test]
    fn test_parse_azure_result_direct_counts() {
        let json = serde_json::json!({
            "status": "Succeeded",
            "target": "ionq.simulator",
            "inputParams": {"shots": 1024},
            "outputData": serde_json::json!({
                "Counts": {
                    "00": 512,
                    "11": 512
                }
            }).to_string()
        });
        let result =
            parse_azure_result(&json, "job-3", "ionq.simulator").expect("should parse Counts");
        assert_eq!(result.counts.get("00"), Some(&512));
        assert_eq!(result.counts.get("11"), Some(&512));
    }

    // -- Result parsing: Empty output ---------------------------------------

    #[test]
    fn test_parse_azure_result_empty_output() {
        let json = serde_json::json!({
            "status": "Succeeded",
            "target": "ionq.simulator",
            "inputParams": {"shots": 100}
        });
        let result =
            parse_azure_result(&json, "job-4", "ionq.simulator").expect("should handle empty");
        assert!(result.counts.is_empty());
    }

    // -- Result parsing: output as JSON object (not string) -----------------

    #[test]
    fn test_parse_azure_result_output_as_object() {
        let json = serde_json::json!({
            "status": "Succeeded",
            "target": "ionq.simulator",
            "inputParams": {"shots": 100},
            "outputData": {
                "counts": {
                    "01": 40,
                    "10": 60
                }
            }
        });
        let result = parse_azure_result(&json, "job-5", "ionq.simulator")
            .expect("should parse object outputData");
        assert_eq!(result.counts.get("01"), Some(&40));
        assert_eq!(result.counts.get("10"), Some(&60));
    }

    // -- Token cache --------------------------------------------------------

    #[test]
    fn test_cached_token_valid() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let token = CachedToken {
            access_token: "test".into(),
            expires_at: now + 3600,
        };
        assert!(token.is_valid());
    }

    #[test]
    fn test_cached_token_expired() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let token = CachedToken {
            access_token: "test".into(),
            expires_at: now - 1,
        };
        assert!(!token.is_valid());
    }

    #[test]
    fn test_cached_token_within_margin() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Token expires in 30 seconds, but margin is 60 seconds
        let token = CachedToken {
            access_token: "test".into(),
            expires_at: now + 30,
        };
        assert!(!token.is_valid()); // Should be considered expired due to margin
    }

    // -- Cost estimation ----------------------------------------------------

    #[tokio::test]
    async fn test_azure_cost_estimate_ionq_simulator() {
        let provider = AzureProvider::new(
            "sub".into(),
            "rg".into(),
            "ws".into(),
            "eastus".into(),
            "tenant".into(),
            "client".into(),
            "secret".into(),
        );
        let circuit = QPUCircuit::bell_state();
        let cost = provider
            .estimate_cost(&circuit, "ionq.simulator", 1000)
            .await
            .expect("should estimate")
            .expect("should return some");
        assert!(cost.is_free_tier);
        assert_eq!(cost.estimated_cost_usd, 0.0);
    }

    #[tokio::test]
    async fn test_azure_cost_estimate_ionq_qpu() {
        let provider = AzureProvider::new(
            "sub".into(),
            "rg".into(),
            "ws".into(),
            "eastus".into(),
            "tenant".into(),
            "client".into(),
            "secret".into(),
        );
        let circuit = QPUCircuit::bell_state(); // 1 H + 1 CX = 1 1Q + 1 2Q
        let cost = provider
            .estimate_cost(&circuit, "ionq.qpu.aria-1", 1000)
            .await
            .expect("should estimate")
            .expect("should return some");
        assert!(!cost.is_free_tier);
        assert!(cost.estimated_cost_usd > 0.0);
    }

    #[tokio::test]
    async fn test_azure_cost_estimate_quantinuum_qpu() {
        let provider = AzureProvider::new(
            "sub".into(),
            "rg".into(),
            "ws".into(),
            "eastus".into(),
            "tenant".into(),
            "client".into(),
            "secret".into(),
        );
        let circuit = QPUCircuit::bell_state();
        let cost = provider
            .estimate_cost(&circuit, "quantinuum.qpu.h1-1", 100)
            .await
            .expect("should estimate")
            .expect("should return some");
        assert!(!cost.is_free_tier);
        // Quantinuum is expensive: at least base 5 HQC per shot
        assert!(cost.estimated_cost_usd > 100.0);
    }

    // -- Validation ---------------------------------------------------------

    #[test]
    fn test_azure_validate_circuit() {
        let provider = AzureProvider::new(
            "sub".into(),
            "rg".into(),
            "ws".into(),
            "eastus".into(),
            "tenant".into(),
            "client".into(),
            "secret".into(),
        );
        let circuit = QPUCircuit::bell_state();
        let backend = make_known_target_info("ionq.qpu.aria-1", "IonQ", 25, false);
        let report = provider
            .validate_circuit(&circuit, &backend)
            .expect("validation should succeed");
        assert!(report.is_valid);
    }

    #[test]
    fn test_azure_validate_too_many_qubits() {
        let provider = AzureProvider::new(
            "sub".into(),
            "rg".into(),
            "ws".into(),
            "eastus".into(),
            "tenant".into(),
            "client".into(),
            "secret".into(),
        );
        let circuit = QPUCircuit::ghz_state(30);
        let backend = make_known_target_info("ionq.qpu.aria-1", "IonQ", 25, false);
        let report = provider
            .validate_circuit(&circuit, &backend)
            .expect("validation should succeed");
        assert!(!report.is_valid);
    }

    // -- HTTP error mapping -------------------------------------------------

    #[test]
    fn test_azure_map_http_error_auth() {
        let err = AzureProvider::map_http_error(
            reqwest::StatusCode::FORBIDDEN,
            "access denied",
        );
        match err {
            QPUError::AuthenticationError(msg) => assert!(msg.contains("403")),
            other => panic!("Expected AuthenticationError, got {:?}", other),
        }
    }

    #[test]
    fn test_azure_map_http_error_not_found() {
        let err = AzureProvider::map_http_error(
            reqwest::StatusCode::NOT_FOUND,
            "workspace not found",
        );
        match err {
            QPUError::BackendUnavailable(msg) => assert!(msg.contains("workspace")),
            other => panic!("Expected BackendUnavailable, got {:?}", other),
        }
    }

    #[test]
    fn test_azure_map_http_error_server() {
        let err = AzureProvider::map_http_error(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "internal error",
        );
        match err {
            QPUError::ProviderError {
                status_code,
                ..
            } => assert_eq!(status_code, Some(500)),
            other => panic!("Expected ProviderError, got {:?}", other),
        }
    }

    // -- API version --------------------------------------------------------

    #[test]
    fn test_api_version_param() {
        let param = AzureProvider::api_version_param();
        assert!(param.starts_with("api-version="));
        assert!(param.contains("2024"));
    }
}
