//! Amazon Braket REST API provider.
//!
//! Connects to the Amazon Braket service using AWS Signature Version 4
//! authentication.  The base URL varies by region:
//! `https://braket.{region}.amazonaws.com/`.
//!
//! # Endpoints
//!
//! | Method | Path                      | Purpose                   |
//! |--------|---------------------------|---------------------------|
//! | GET    | `/devices`                | List quantum devices      |
//! | POST   | `/quantum-tasks`          | Submit a quantum task     |
//! | GET    | `/quantum-tasks/{taskId}` | Get task status & results |
//!
//! # Authentication
//!
//! Uses HMAC-SHA256 based AWS Signature V4 signing, reading credentials from
//! `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and
//! optionally `AWS_SESSION_TOKEN`.

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
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

const BRAKET_PROVIDER_NAME: &str = "Amazon Braket";
const BRAKET_SERVICE: &str = "braket";
const MAX_BACKOFF_SECS: u64 = 60;

/// Well-known device ARN mappings.
const DEVICE_ARN_MAP: &[(&str, &str)] = &[
    (
        "ionq_aria",
        "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    ),
    (
        "ionq_forte",
        "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    ),
    (
        "rigetti_aspen",
        "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
    ),
    (
        "oqc_lucy",
        "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
    ),
    (
        "quera_aquila",
        "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    ),
    (
        "sv1",
        "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    ),
];

// ---------------------------------------------------------------------------
// AWS Signature V4 helpers
// ---------------------------------------------------------------------------

/// HMAC-SHA256 using the `sha2` crate (no dependency on `hmac` crate).
///
/// HMAC(K, m) = H((K' xor opad) || H((K' xor ipad) || m))
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 64;
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;

    // If key is longer than block size, hash it first.
    let key_prime = if key.len() > BLOCK_SIZE {
        let mut h = Sha256::new();
        h.update(key);
        h.finalize().to_vec()
    } else {
        key.to_vec()
    };

    // Pad to block size.
    let mut padded = vec![0u8; BLOCK_SIZE];
    padded[..key_prime.len()].copy_from_slice(&key_prime);

    // Inner hash: H((K' xor ipad) || data)
    let mut inner = Sha256::new();
    let inner_key: Vec<u8> = padded.iter().map(|b| b ^ IPAD).collect();
    inner.update(&inner_key);
    inner.update(data);
    let inner_hash = inner.finalize();

    // Outer hash: H((K' xor opad) || inner_hash)
    let mut outer = Sha256::new();
    let outer_key: Vec<u8> = padded.iter().map(|b| b ^ OPAD).collect();
    outer.update(&outer_key);
    outer.update(&inner_hash);
    outer.finalize().to_vec()
}

/// Hex-encode a byte slice.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// SHA-256 hash of a byte string, returned as hex.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex_encode(&hasher.finalize())
}

/// Get the current UTC time formatted for AWS Sig V4.
fn aws_datetime_now() -> (String, String) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();

    // Convert epoch seconds to YYYYMMDD'T'HHMMSS'Z'
    // We do basic arithmetic rather than pulling in chrono.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since epoch to Y/M/D using a simplified algorithm.
    let (year, month, day) = epoch_days_to_ymd(days as i64);

    let date_stamp = format!("{:04}{:02}{:02}", year, month, day);
    let amz_date = format!("{}T{:02}{:02}{:02}Z", date_stamp, hours, minutes, seconds);

    (amz_date, date_stamp)
}

/// Convert days since Unix epoch to (year, month, day).
fn epoch_days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from Howard Hinnant's civil_from_days
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// AWS Credential scope: `{date}/{region}/{service}/aws4_request`
fn credential_scope(date_stamp: &str, region: &str) -> String {
    format!("{}/{}/{}/aws4_request", date_stamp, region, BRAKET_SERVICE)
}

/// Derive the AWS Sig V4 signing key.
///
/// kDate = HMAC("AWS4" + secret, date)
/// kRegion = HMAC(kDate, region)
/// kService = HMAC(kRegion, service)
/// kSigning = HMAC(kService, "aws4_request")
fn derive_signing_key(secret: &str, date_stamp: &str, region: &str) -> Vec<u8> {
    let k_date = hmac_sha256(
        format!("AWS4{}", secret).as_bytes(),
        date_stamp.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, BRAKET_SERVICE.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// Build the Authorization header value for an AWS Sig V4 signed request.
fn sign_request(
    method: &str,
    uri: &str,
    query_string: &str,
    headers_map: &[(&str, &str)],
    payload: &[u8],
    access_key: &str,
    secret_key: &str,
    region: &str,
    amz_date: &str,
    date_stamp: &str,
) -> String {
    // 1. Canonical headers (must be sorted by lowercase key)
    let mut sorted_headers: Vec<(&str, &str)> = headers_map.to_vec();
    sorted_headers.sort_by_key(|(k, _)| k.to_lowercase());

    let canonical_headers: String = sorted_headers
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k.to_lowercase(), v.trim()))
        .collect();

    let signed_headers: String = sorted_headers
        .iter()
        .map(|(k, _)| k.to_lowercase())
        .collect::<Vec<_>>()
        .join(";");

    // 2. Canonical request
    let payload_hash = sha256_hex(payload);
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method, uri, query_string, canonical_headers, signed_headers, payload_hash
    );

    // 3. String to sign
    let scope = credential_scope(date_stamp, region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        scope,
        sha256_hex(canonical_request.as_bytes())
    );

    // 4. Signature
    let signing_key = derive_signing_key(secret_key, date_stamp, region);
    let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    // 5. Authorization header
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        access_key, scope, signed_headers, signature
    )
}

// ---------------------------------------------------------------------------
// BraketProvider
// ---------------------------------------------------------------------------

/// Amazon Braket REST API provider.
///
/// Submits circuits as OpenQASM 3.0 strings to quantum devices accessible
/// through Amazon Braket.  Supports IonQ, Rigetti, OQC, QuEra hardware and
/// the SV1 state-vector simulator.
///
/// # Construction
///
/// ```no_run
/// # use nqpu_metal::qpu::braket::BraketProvider;
/// // From environment variables
/// let provider = BraketProvider::from_env().expect("AWS credentials not set");
/// ```
pub struct BraketProvider {
    client: reqwest::Client,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    region: String,
    base_url: String,
}

impl BraketProvider {
    /// Create a provider from explicit credentials.
    pub fn new(
        access_key_id: String,
        secret_access_key: String,
        region: String,
        session_token: Option<String>,
    ) -> Self {
        let base_url = format!("https://braket.{}.amazonaws.com", region);
        Self {
            client: reqwest::Client::new(),
            access_key_id,
            secret_access_key,
            session_token,
            region,
            base_url,
        }
    }

    /// Create a provider from AWS environment variables.
    ///
    /// Reads: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    /// `AWS_DEFAULT_REGION` (defaults to `us-east-1`), and optionally
    /// `AWS_SESSION_TOKEN`.
    pub fn from_env() -> Result<Self, QPUError> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| {
            QPUError::ConfigError("AWS_ACCESS_KEY_ID environment variable not set".into())
        })?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| {
            QPUError::ConfigError("AWS_SECRET_ACCESS_KEY environment variable not set".into())
        })?;
        let region =
            std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        Ok(Self::new(access_key, secret_key, region, session_token))
    }

    /// Override the base URL (useful for testing against a mock server).
    #[cfg(test)]
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    // -- ARN resolution -----------------------------------------------------

    /// Resolve a human-friendly device name to an AWS device ARN.
    ///
    /// If the input already looks like an ARN (starts with `arn:`), it is
    /// returned as-is.
    fn resolve_device_arn(name: &str) -> String {
        if name.starts_with("arn:") {
            return name.to_string();
        }
        for (alias, arn) in DEVICE_ARN_MAP {
            if *alias == name {
                return arn.to_string();
            }
        }
        // If not found, assume it is a full ARN the caller knows about.
        name.to_string()
    }

    /// Derive a short display name from an ARN.
    fn arn_to_display_name(arn: &str) -> String {
        // Try reverse lookup first.
        for (alias, a) in DEVICE_ARN_MAP {
            if *a == arn {
                return alias.to_string();
            }
        }
        // Fallback: last path segment of the ARN.
        arn.rsplit('/')
            .next()
            .unwrap_or(arn)
            .to_string()
    }

    // -- Signed HTTP helpers ------------------------------------------------

    /// Execute a signed GET request and return JSON.
    async fn signed_get(&self, path: &str) -> Result<serde_json::Value, QPUError> {
        let (amz_date, date_stamp) = aws_datetime_now();
        let host = self
            .base_url
            .trim_start_matches("https://")
            .trim_start_matches("http://");

        let mut headers_list: Vec<(&str, String)> = vec![
            ("host", host.to_string()),
            ("x-amz-date", amz_date.clone()),
        ];
        if let Some(ref tok) = self.session_token {
            headers_list.push(("x-amz-security-token", tok.clone()));
        }
        let headers_ref: Vec<(&str, &str)> = headers_list
            .iter()
            .map(|(k, v)| (*k, v.as_str()))
            .collect();

        let auth = sign_request(
            "GET",
            path,
            "",
            &headers_ref,
            b"",
            &self.access_key_id,
            &self.secret_access_key,
            &self.region,
            &amz_date,
            &date_stamp,
        );

        let mut req = self
            .client
            .get(format!("{}{}", self.base_url, path))
            .header("x-amz-date", &amz_date)
            .header("Authorization", &auth);

        if let Some(ref tok) = self.session_token {
            req = req.header("x-amz-security-token", tok);
        }

        self.send_and_parse(req).await
    }

    /// Execute a signed POST request with a JSON body and return JSON.
    async fn signed_post(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, QPUError> {
        let payload = serde_json::to_vec(body).map_err(|e| {
            QPUError::ConversionError(format!("Failed to serialize request body: {}", e))
        })?;

        let (amz_date, date_stamp) = aws_datetime_now();
        let host = self
            .base_url
            .trim_start_matches("https://")
            .trim_start_matches("http://");

        let content_type = "application/json";
        let payload_hash = sha256_hex(&payload);

        let mut headers_list: Vec<(&str, String)> = vec![
            ("content-type", content_type.to_string()),
            ("host", host.to_string()),
            ("x-amz-content-sha256", payload_hash.clone()),
            ("x-amz-date", amz_date.clone()),
        ];
        if let Some(ref tok) = self.session_token {
            headers_list.push(("x-amz-security-token", tok.clone()));
        }
        let headers_ref: Vec<(&str, &str)> = headers_list
            .iter()
            .map(|(k, v)| (*k, v.as_str()))
            .collect();

        let auth = sign_request(
            "POST",
            path,
            "",
            &headers_ref,
            &payload,
            &self.access_key_id,
            &self.secret_access_key,
            &self.region,
            &amz_date,
            &date_stamp,
        );

        let mut req = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .header("Content-Type", content_type)
            .header("x-amz-date", &amz_date)
            .header("x-amz-content-sha256", &payload_hash)
            .header("Authorization", &auth)
            .body(payload);

        if let Some(ref tok) = self.session_token {
            req = req.header("x-amz-security-token", tok);
        }

        self.send_and_parse(req).await
    }

    /// Send a request and parse the response body as JSON.
    async fn send_and_parse(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<serde_json::Value, QPUError> {
        let response = request
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Braket request failed: {}", e)))?;

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
            .map_err(|e| QPUError::NetworkError(format!("Failed to read Braket response: {}", e)))?;

        if !status.is_success() {
            return Err(Self::map_http_error(status, &body));
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ProviderError {
            provider: BRAKET_PROVIDER_NAME.into(),
            message: format!("Invalid JSON from Braket: {} (body: {})", e, &body[..body.len().min(200)]),
            status_code: None,
        })
    }

    /// Map an HTTP error to a [`QPUError`].
    fn map_http_error(status: reqwest::StatusCode, body: &str) -> QPUError {
        match status.as_u16() {
            401 | 403 => QPUError::AuthenticationError(format!(
                "AWS authentication failed (HTTP {}): {}",
                status,
                &body[..body.len().min(300)]
            )),
            404 => QPUError::BackendUnavailable(format!("Not found: {}", &body[..body.len().min(300)])),
            _ => QPUError::ProviderError {
                provider: BRAKET_PROVIDER_NAME.into(),
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
impl QPUProvider for BraketProvider {
    fn name(&self) -> &str {
        BRAKET_PROVIDER_NAME
    }

    async fn list_backends(&self) -> Result<Vec<BackendInfo>, QPUError> {
        let json = self.signed_get("/devices").await?;

        let devices = json["devices"]
            .as_array()
            .or_else(|| json.as_array())
            .ok_or_else(|| QPUError::ProviderError {
                provider: BRAKET_PROVIDER_NAME.into(),
                message: "Expected JSON array from /devices".into(),
                status_code: None,
            })?;

        let mut backends = Vec::with_capacity(devices.len());
        for dev in devices {
            if let Some(info) = parse_braket_device(dev) {
                backends.push(info);
            }
        }
        Ok(backends)
    }

    async fn get_backend(&self, name: &str) -> Result<BackendInfo, QPUError> {
        let arn = Self::resolve_device_arn(name);
        let json = self
            .signed_get(&format!("/devices/{}", urlencoding_arn(&arn)))
            .await?;
        parse_braket_device(&json).ok_or_else(|| {
            QPUError::BackendUnavailable(format!("Could not parse device info for '{}'", name))
        })
    }

    async fn submit_circuit(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        config: &JobConfig,
    ) -> Result<Box<dyn QPUJob>, QPUError> {
        let device_arn = Self::resolve_device_arn(backend);
        let qasm3 = circuit.to_qasm3();

        let mut body = serde_json::json!({
            "deviceArn": device_arn,
            "action": serde_json::json!({
                "braketSchemaHeader": {
                    "name": "braket.ir.openqasm.program",
                    "version": "1"
                },
                "source": qasm3
            }).to_string(),
            "shots": config.shots,
            "outputS3Bucket": config.extra.get("output_s3_bucket")
                .cloned()
                .unwrap_or_else(|| "braket-output".to_string()),
            "outputS3KeyPrefix": config.extra.get("output_s3_prefix")
                .cloned()
                .unwrap_or_else(|| "nqpu-metal".to_string()),
        });

        if let Some(ref name) = config.name {
            body["tags"] = serde_json::json!({"name": name});
        }

        let json = self.signed_post("/quantum-tasks", &body).await?;

        let task_id = json["quantumTaskArn"]
            .as_str()
            .or_else(|| json["taskArn"].as_str())
            .ok_or_else(|| {
                QPUError::SubmissionError("Missing task ARN in Braket response".into())
            })?
            .to_string();

        let display_name = Self::arn_to_display_name(&device_arn);

        Ok(Box::new(BraketJob {
            client: self.client.clone(),
            access_key_id: self.access_key_id.clone(),
            secret_access_key: self.secret_access_key.clone(),
            session_token: self.session_token.clone(),
            region: self.region.clone(),
            base_url: self.base_url.clone(),
            task_id,
            backend_name: display_name,
        }))
    }

    async fn estimate_cost(
        &self,
        circuit: &QPUCircuit,
        backend: &str,
        shots: usize,
    ) -> Result<Option<CostEstimate>, QPUError> {
        let arn = Self::resolve_device_arn(backend);
        let is_simulator = arn.contains("simulator");

        // Braket pricing (approximate, varies by device):
        //   IonQ: $0.30/task + $0.01/shot
        //   Rigetti: $0.30/task + $0.00035/shot
        //   OQC: $0.30/task + $0.00035/shot
        //   SV1: $0.075/minute of simulation
        let (task_cost, shot_cost, details) = if arn.contains("ionq") {
            (0.30, 0.01, "IonQ: $0.30/task + $0.01/shot")
        } else if arn.contains("rigetti") {
            (0.30, 0.00035, "Rigetti: $0.30/task + $0.00035/shot")
        } else if arn.contains("oqc") {
            (0.30, 0.00035, "OQC: $0.30/task + $0.00035/shot")
        } else if is_simulator {
            // SV1: rough estimate based on circuit complexity
            let minutes = (circuit.gate_count() as f64 * shots as f64 * 1e-7).max(0.01);
            let cost = minutes * 0.075;
            return Ok(Some(CostEstimate {
                provider: BRAKET_PROVIDER_NAME.into(),
                backend: backend.into(),
                shots,
                estimated_cost_usd: cost,
                estimated_queue_time: None,
                is_free_tier: false,
                details: Some(format!(
                    "SV1 simulator: ~{:.3} min at $0.075/min",
                    minutes
                )),
            }));
        } else {
            (0.30, 0.01, "Unknown device: using IonQ-equivalent pricing")
        };

        let total = task_cost + (shot_cost * shots as f64);

        Ok(Some(CostEstimate {
            provider: BRAKET_PROVIDER_NAME.into(),
            backend: backend.into(),
            shots,
            estimated_cost_usd: total,
            estimated_queue_time: None,
            is_free_tier: false,
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
// BraketJob
// ---------------------------------------------------------------------------

/// Handle for a quantum task submitted to Amazon Braket.
pub struct BraketJob {
    client: reqwest::Client,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    region: String,
    base_url: String,
    task_id: String,
    backend_name: String,
}

impl BraketJob {
    /// Execute a signed GET and return JSON.
    async fn signed_get(&self, path: &str) -> Result<serde_json::Value, QPUError> {
        let (amz_date, date_stamp) = aws_datetime_now();
        let host = self
            .base_url
            .trim_start_matches("https://")
            .trim_start_matches("http://");

        let mut headers_list: Vec<(&str, String)> = vec![
            ("host", host.to_string()),
            ("x-amz-date", amz_date.clone()),
        ];
        if let Some(ref tok) = self.session_token {
            headers_list.push(("x-amz-security-token", tok.clone()));
        }
        let headers_ref: Vec<(&str, &str)> = headers_list
            .iter()
            .map(|(k, v)| (*k, v.as_str()))
            .collect();

        let auth = sign_request(
            "GET",
            path,
            "",
            &headers_ref,
            b"",
            &self.access_key_id,
            &self.secret_access_key,
            &self.region,
            &amz_date,
            &date_stamp,
        );

        let mut req = self
            .client
            .get(format!("{}{}", self.base_url, path))
            .header("x-amz-date", &amz_date)
            .header("Authorization", &auth);

        if let Some(ref tok) = self.session_token {
            req = req.header("x-amz-security-token", tok);
        }

        let response = req
            .send()
            .await
            .map_err(|e| QPUError::NetworkError(format!("Braket request failed: {}", e)))?;

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
            .map_err(|e| QPUError::NetworkError(format!("Read Braket response: {}", e)))?;

        if !status.is_success() {
            return Err(BraketProvider::map_http_error(status, &body));
        }

        serde_json::from_str(&body).map_err(|e| QPUError::ProviderError {
            provider: BRAKET_PROVIDER_NAME.into(),
            message: format!("Invalid JSON: {}", e),
            status_code: None,
        })
    }
}

#[async_trait]
impl QPUJob for BraketJob {
    fn id(&self) -> &str {
        &self.task_id
    }

    fn backend(&self) -> &str {
        &self.backend_name
    }

    fn provider(&self) -> &str {
        BRAKET_PROVIDER_NAME
    }

    async fn status(&self) -> Result<JobStatus, QPUError> {
        let task_path = braket_task_path(&self.task_id);
        let json = self.signed_get(&task_path).await?;
        Ok(parse_braket_task_status(&json))
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

            let task_path = braket_task_path(&self.task_id);
            let json = self.signed_get(&task_path).await?;
            let status = parse_braket_task_status(&json);

            match status {
                JobStatus::Completed => {
                    return parse_braket_result(&json, &self.task_id, &self.backend_name);
                }
                JobStatus::Failed(msg) => {
                    return Err(QPUError::ExecutionError(format!(
                        "Braket task {} failed: {}",
                        self.task_id, msg
                    )));
                }
                JobStatus::Cancelled => {
                    return Err(QPUError::ExecutionError(format!(
                        "Braket task {} was cancelled",
                        self.task_id
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
        let task_path = braket_task_path(&self.task_id);
        let json = self.signed_get(&task_path).await?;
        let status = parse_braket_task_status(&json);

        if status == JobStatus::Completed {
            parse_braket_result(&json, &self.task_id, &self.backend_name).map(Some)
        } else {
            Ok(None)
        }
    }

    async fn cancel(&self) -> Result<(), QPUError> {
        // Braket cancellation is done via a PUT to the task with status CANCELLING.
        // For simplicity we use the existing signed_get infrastructure and note
        // that a proper implementation would use a DELETE or PUT.  The API also
        // supports cancellation via the SDK.  Here we send a best-effort request.
        let task_path = braket_task_path(&self.task_id);
        // Note: real cancellation requires a different HTTP method; this is a
        // stub that will be replaced once the full Braket SDK integration is done.
        let _ = self.signed_get(&format!("{}/cancel", task_path)).await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JSON parsing helpers
// ---------------------------------------------------------------------------

/// Build the API path for a quantum task.
fn braket_task_path(task_id: &str) -> String {
    // Task IDs may be full ARNs or just the UUID portion.
    if task_id.starts_with("arn:") {
        format!("/quantum-tasks/{}", urlencoding_arn(task_id))
    } else {
        format!("/quantum-tasks/{}", task_id)
    }
}

/// Percent-encode an ARN for use in a URL path segment.
fn urlencoding_arn(arn: &str) -> String {
    arn.replace(':', "%3A").replace('/', "%2F")
}

/// Parse a Braket device JSON into a [`BackendInfo`].
fn parse_braket_device(json: &serde_json::Value) -> Option<BackendInfo> {
    let device_arn = json["deviceArn"]
        .as_str()
        .or_else(|| json["arn"].as_str())?;

    let name = json["deviceName"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| BraketProvider::arn_to_display_name(device_arn));

    let provider_name = json["providerName"]
        .as_str()
        .unwrap_or("Unknown")
        .to_string();

    let device_type = json["deviceType"].as_str().unwrap_or("");
    let is_simulator = device_type.contains("SIMULATOR")
        || device_arn.contains("simulator")
        || name.to_lowercase().contains("sim");

    let status_str = json["deviceStatus"].as_str().unwrap_or("OFFLINE");
    let status = match status_str.to_uppercase().as_str() {
        "ONLINE" => BackendStatus::Online,
        "OFFLINE" => BackendStatus::Offline,
        "RETIRED" => BackendStatus::Retired,
        _ => BackendStatus::Offline,
    };

    // Device capabilities may be a JSON string that needs parsing.
    let caps_str = json["deviceCapabilities"].as_str().unwrap_or("{}");
    let caps: serde_json::Value =
        serde_json::from_str(caps_str).unwrap_or(serde_json::json!({}));

    let num_qubits = caps["paradigm"]["qubitCount"]
        .as_u64()
        .or_else(|| json["qubitCount"].as_u64())
        .unwrap_or(0) as usize;

    let basis_gates: Vec<String> = caps["paradigm"]["nativeGateSet"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let coupling_map: Option<Vec<(usize, usize)>> = caps["paradigm"]["connectivity"]["fullyConnected"]
        .as_bool()
        .and_then(|fully| {
            if fully {
                None // fully connected = no explicit map needed
            } else {
                caps["paradigm"]["connectivity"]["connectivityGraph"]
                    .as_object()
                    .map(|graph| {
                        let mut edges = Vec::new();
                        for (src, targets) in graph {
                            if let (Ok(s), Some(arr)) = (src.parse::<usize>(), targets.as_array()) {
                                for t in arr {
                                    if let Some(tgt) = t.as_u64() {
                                        edges.push((s, tgt as usize));
                                    }
                                }
                            }
                        }
                        edges
                    })
            }
        });

    let max_shots = caps["service"]["shotsRange"]
        .as_array()
        .and_then(|arr| arr.last()?.as_u64())
        .unwrap_or(100_000) as usize;

    let queue_length = json["queueDepth"]
        .as_u64()
        .or_else(|| json["queue_depth"].as_u64())
        .unwrap_or(0) as usize;

    let mut metadata = HashMap::new();
    metadata.insert(
        "device_arn".into(),
        serde_json::Value::String(device_arn.to_string()),
    );
    metadata.insert(
        "provider_name".into(),
        serde_json::Value::String(provider_name.clone()),
    );

    Some(BackendInfo {
        name,
        provider: format!("Amazon Braket ({})", provider_name),
        num_qubits,
        is_simulator,
        status,
        queue_length,
        basis_gates,
        coupling_map,
        max_shots,
        max_depth: None,
        avg_gate_error_1q: None,
        avg_gate_error_2q: None,
        avg_readout_error: None,
        avg_t1_us: None,
        avg_t2_us: None,
        metadata,
    })
}

/// Parse Braket task status.
fn parse_braket_task_status(json: &serde_json::Value) -> JobStatus {
    match json["status"]
        .as_str()
        .unwrap_or("UNKNOWN")
        .to_uppercase()
        .as_str()
    {
        "CREATED" => JobStatus::Initializing,
        "QUEUED" => JobStatus::Queued,
        "RUNNING" => JobStatus::Running,
        "COMPLETED" => JobStatus::Completed,
        "FAILED" => {
            let msg = json["failureReason"]
                .as_str()
                .unwrap_or("Unknown failure")
                .to_string();
            JobStatus::Failed(msg)
        }
        "CANCELLING" | "CANCELLED" => JobStatus::Cancelled,
        other => JobStatus::Failed(format!("Unknown Braket status: {}", other)),
    }
}

/// Parse a completed Braket task into a [`JobResult`].
fn parse_braket_result(
    json: &serde_json::Value,
    task_id: &str,
    backend_name: &str,
) -> Result<JobResult, QPUError> {
    let mut counts = HashMap::new();

    // Braket returns measurement results in various formats.
    let measurements = &json["result"]["measurements"];
    let measurement_counts = &json["result"]["measurementCounts"];
    let measurement_probabilities = &json["result"]["measurementProbabilities"];

    if let Some(obj) = measurement_counts.as_object() {
        // Direct bitstring -> count mapping
        for (bitstring, count) in obj {
            let c = count.as_u64().unwrap_or(0) as usize;
            counts.insert(bitstring.clone(), c);
        }
    } else if let Some(arr) = measurements.as_array() {
        // Array of measurement arrays: [[0,1],[1,0],...]
        for shot in arr {
            if let Some(bits) = shot.as_array() {
                let bitstring: String = bits
                    .iter()
                    .map(|b| if b.as_u64().unwrap_or(0) == 1 { '1' } else { '0' })
                    .collect();
                *counts.entry(bitstring).or_insert(0) += 1;
            }
        }
    } else if let Some(obj) = measurement_probabilities.as_object() {
        // Probabilities -> approximate counts (using 1000 as denominator)
        let total_shots = json["shots"].as_u64().unwrap_or(1000) as f64;
        for (bitstring, prob) in obj {
            let p = prob.as_f64().unwrap_or(0.0);
            let c = (p * total_shots).round() as usize;
            if c > 0 {
                counts.insert(bitstring.clone(), c);
            }
        }
    }

    let shots = json["shots"]
        .as_u64()
        .unwrap_or_else(|| counts.values().sum::<usize>() as u64)
        as usize;

    let is_simulator = json["deviceArn"]
        .as_str()
        .map(|a| a.contains("simulator"))
        .unwrap_or(false);

    let execution_time = json["result"]["taskMetadata"]["executionDuration"]
        .as_f64()
        .map(Duration::from_secs_f64);

    let mut metadata = HashMap::new();
    if let Some(arn) = json["quantumTaskArn"].as_str() {
        metadata.insert("task_arn".into(), serde_json::Value::String(arn.into()));
    }
    if let Some(device) = json["deviceArn"].as_str() {
        metadata.insert(
            "device_arn".into(),
            serde_json::Value::String(device.into()),
        );
    }

    Ok(JobResult {
        job_id: task_id.to_string(),
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

    // -- HMAC-SHA256 --------------------------------------------------------

    #[test]
    fn test_hmac_sha256_known_vector() {
        // RFC 4231 Test Case 2
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
        let result = hex_encode(&hmac_sha256(key, data));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_hmac_sha256_empty_data() {
        let key = b"key";
        let result = hmac_sha256(key, b"");
        assert_eq!(result.len(), 32); // SHA-256 output is always 32 bytes
    }

    #[test]
    fn test_hmac_sha256_long_key() {
        // Key longer than block size (64 bytes) should be hashed first
        let key = vec![0xAA; 131];
        let data = b"test data";
        let result = hmac_sha256(&key, data);
        assert_eq!(result.len(), 32);
    }

    // -- SHA-256 helpers ----------------------------------------------------

    #[test]
    fn test_sha256_hex_empty() {
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(sha256_hex(b""), expected);
    }

    #[test]
    fn test_sha256_hex_hello() {
        let expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        assert_eq!(sha256_hex(b"hello"), expected);
    }

    // -- Date helpers -------------------------------------------------------

    #[test]
    fn test_epoch_days_to_ymd_epoch() {
        assert_eq!(epoch_days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn test_epoch_days_to_ymd_known_date() {
        // 2024-01-01 = day 19723
        let (y, m, d) = epoch_days_to_ymd(19723);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 1);
    }

    #[test]
    fn test_epoch_days_to_ymd_leap_year() {
        // 2024-02-29 = day 19782
        let (y, m, d) = epoch_days_to_ymd(19782);
        assert_eq!(y, 2024);
        assert_eq!(m, 2);
        assert_eq!(d, 29);
    }

    // -- ARN resolution -----------------------------------------------------

    #[test]
    fn test_resolve_known_arns() {
        assert_eq!(
            BraketProvider::resolve_device_arn("ionq_aria"),
            "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
        );
        assert_eq!(
            BraketProvider::resolve_device_arn("sv1"),
            "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        );
    }

    #[test]
    fn test_resolve_raw_arn_passthrough() {
        let arn = "arn:aws:braket:us-east-1::device/qpu/custom/Device-1";
        assert_eq!(BraketProvider::resolve_device_arn(arn), arn);
    }

    #[test]
    fn test_resolve_unknown_name() {
        assert_eq!(
            BraketProvider::resolve_device_arn("some_new_device"),
            "some_new_device"
        );
    }

    #[test]
    fn test_arn_to_display_name_known() {
        assert_eq!(
            BraketProvider::arn_to_display_name(
                "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
            ),
            "ionq_aria"
        );
    }

    #[test]
    fn test_arn_to_display_name_unknown() {
        assert_eq!(
            BraketProvider::arn_to_display_name(
                "arn:aws:braket:us-east-1::device/qpu/new/Widget-9"
            ),
            "Widget-9"
        );
    }

    // -- URL encoding -------------------------------------------------------

    #[test]
    fn test_urlencoding_arn() {
        let arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1";
        let encoded = urlencoding_arn(arn);
        assert!(!encoded.contains(':'));
        assert!(!encoded.contains('/'));
        assert!(encoded.contains("%3A"));
        assert!(encoded.contains("%2F"));
    }

    // -- Device parsing -----------------------------------------------------

    fn sample_braket_device_json() -> serde_json::Value {
        serde_json::json!({
            "deviceArn": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
            "deviceName": "Aria 1",
            "providerName": "IonQ",
            "deviceType": "QPU",
            "deviceStatus": "ONLINE",
            "queueDepth": 7,
            "deviceCapabilities": serde_json::json!({
                "paradigm": {
                    "qubitCount": 25,
                    "nativeGateSet": ["gpi", "gpi2", "ms"],
                    "connectivity": {
                        "fullyConnected": true
                    }
                },
                "service": {
                    "shotsRange": [1, 10000]
                }
            }).to_string()
        })
    }

    #[test]
    fn test_parse_braket_device_qpu() {
        let info = parse_braket_device(&sample_braket_device_json()).expect("should parse");
        assert_eq!(info.name, "Aria 1");
        assert!(info.provider.contains("IonQ"));
        assert_eq!(info.num_qubits, 25);
        assert!(!info.is_simulator);
        assert_eq!(info.status, BackendStatus::Online);
        assert_eq!(info.queue_length, 7);
        assert_eq!(info.basis_gates, vec!["gpi", "gpi2", "ms"]);
        assert!(info.coupling_map.is_none()); // fully connected
        assert_eq!(info.max_shots, 10_000);
    }

    #[test]
    fn test_parse_braket_device_simulator() {
        let json = serde_json::json!({
            "deviceArn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            "deviceName": "SV1",
            "providerName": "Amazon",
            "deviceType": "SIMULATOR",
            "deviceStatus": "ONLINE",
            "deviceCapabilities": serde_json::json!({
                "paradigm": {
                    "qubitCount": 34,
                    "nativeGateSet": ["h", "cx", "rz", "ry", "rx", "swap"]
                },
                "service": {
                    "shotsRange": [0, 100000]
                }
            }).to_string()
        });
        let info = parse_braket_device(&json).expect("should parse simulator");
        assert!(info.is_simulator);
        assert_eq!(info.num_qubits, 34);
        assert_eq!(info.max_shots, 100_000);
    }

    #[test]
    fn test_parse_braket_device_minimal() {
        let json = serde_json::json!({
            "deviceArn": "arn:aws:braket:us-east-1::device/qpu/new/Dev-1"
        });
        let info = parse_braket_device(&json).expect("should parse minimal");
        assert_eq!(info.name, "Dev-1");
        assert_eq!(info.num_qubits, 0);
    }

    #[test]
    fn test_parse_braket_device_missing_arn() {
        let json = serde_json::json!({"deviceName": "no-arn"});
        assert!(parse_braket_device(&json).is_none());
    }

    #[test]
    fn test_parse_braket_device_with_connectivity_graph() {
        let json = serde_json::json!({
            "deviceArn": "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
            "deviceName": "Aspen-M-3",
            "providerName": "Rigetti",
            "deviceType": "QPU",
            "deviceStatus": "ONLINE",
            "deviceCapabilities": serde_json::json!({
                "paradigm": {
                    "qubitCount": 80,
                    "nativeGateSet": ["cz", "rx", "rz"],
                    "connectivity": {
                        "fullyConnected": false,
                        "connectivityGraph": {
                            "0": [1, 7],
                            "1": [0, 2],
                            "2": [1, 3]
                        }
                    }
                },
                "service": {
                    "shotsRange": [1, 100000]
                }
            }).to_string()
        });
        let info = parse_braket_device(&json).expect("should parse with graph");
        assert_eq!(info.num_qubits, 80);
        let cm = info.coupling_map.expect("should have coupling map");
        assert!(cm.contains(&(0, 1)));
        assert!(cm.contains(&(0, 7)));
        assert!(cm.contains(&(1, 0)));
        assert!(cm.contains(&(1, 2)));
    }

    // -- Task status parsing ------------------------------------------------

    #[test]
    fn test_parse_braket_task_status_all_states() {
        let cases = vec![
            ("CREATED", false, false),
            ("QUEUED", false, false),
            ("RUNNING", false, false),
            ("COMPLETED", true, true),
            ("CANCELLING", true, false),
            ("CANCELLED", true, false),
        ];

        for (status_str, is_terminal, is_success) in cases {
            let json = serde_json::json!({"status": status_str});
            let status = parse_braket_task_status(&json);
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
    fn test_parse_braket_task_status_failed() {
        let json = serde_json::json!({
            "status": "FAILED",
            "failureReason": "Device calibration error"
        });
        match parse_braket_task_status(&json) {
            JobStatus::Failed(msg) => assert_eq!(msg, "Device calibration error"),
            other => panic!("Expected Failed, got {:?}", other),
        }
    }

    // -- Result parsing -----------------------------------------------------

    #[test]
    fn test_parse_braket_result_measurement_counts() {
        let json = serde_json::json!({
            "status": "COMPLETED",
            "shots": 1024,
            "deviceArn": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
            "result": {
                "measurements": null,
                "measurementCounts": {
                    "00": 512,
                    "11": 512
                },
                "taskMetadata": {
                    "executionDuration": 0.045
                }
            }
        });
        let result =
            parse_braket_result(&json, "task-123", "ionq_aria").expect("should parse result");
        assert_eq!(result.job_id, "task-123");
        assert_eq!(result.backend, "ionq_aria");
        assert_eq!(result.shots, 1024);
        assert_eq!(result.counts.get("00"), Some(&512));
        assert_eq!(result.counts.get("11"), Some(&512));
        assert!(!result.is_simulator);
        assert!(result.execution_time.is_some());
    }

    #[test]
    fn test_parse_braket_result_measurement_arrays() {
        let json = serde_json::json!({
            "status": "COMPLETED",
            "shots": 4,
            "deviceArn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            "result": {
                "measurements": [
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [1, 1]
                ]
            }
        });
        let result = parse_braket_result(&json, "task-456", "sv1").expect("should parse arrays");
        assert_eq!(result.shots, 4);
        assert_eq!(result.counts.get("00"), Some(&2));
        assert_eq!(result.counts.get("11"), Some(&2));
        assert!(result.is_simulator);
    }

    #[test]
    fn test_parse_braket_result_probabilities() {
        let json = serde_json::json!({
            "status": "COMPLETED",
            "shots": 1000,
            "deviceArn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            "result": {
                "measurementProbabilities": {
                    "00": 0.5,
                    "11": 0.5
                }
            }
        });
        let result =
            parse_braket_result(&json, "task-789", "sv1").expect("should parse probabilities");
        assert_eq!(result.counts.get("00"), Some(&500));
        assert_eq!(result.counts.get("11"), Some(&500));
    }

    // -- AWS Sig V4 integration test ----------------------------------------

    #[test]
    fn test_sign_request_deterministic() {
        let auth = sign_request(
            "GET",
            "/devices",
            "",
            &[
                ("host", "braket.us-east-1.amazonaws.com"),
                ("x-amz-date", "20240101T000000Z"),
            ],
            b"",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "us-east-1",
            "20240101T000000Z",
            "20240101",
        );
        assert!(auth.starts_with("AWS4-HMAC-SHA256 "));
        assert!(auth.contains("Credential=AKIAIOSFODNN7EXAMPLE/20240101/us-east-1/braket/aws4_request"));
        assert!(auth.contains("SignedHeaders=host;x-amz-date"));
        assert!(auth.contains("Signature="));
        // Signature should be a 64-char hex string.
        let sig_part = auth.split("Signature=").nth(1).expect("should have Signature=");
        assert_eq!(sig_part.len(), 64);
        assert!(sig_part.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_sign_request_with_payload() {
        let payload = br#"{"deviceArn":"arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"}"#;
        let auth = sign_request(
            "POST",
            "/quantum-tasks",
            "",
            &[
                ("content-type", "application/json"),
                ("host", "braket.us-east-1.amazonaws.com"),
                ("x-amz-date", "20240101T120000Z"),
            ],
            payload,
            "AKID",
            "SECRET",
            "us-east-1",
            "20240101T120000Z",
            "20240101",
        );
        assert!(auth.starts_with("AWS4-HMAC-SHA256 "));
        assert!(auth.contains("content-type;host;x-amz-date"));
    }

    // -- Cost estimation ----------------------------------------------------

    #[tokio::test]
    async fn test_braket_cost_estimate_ionq() {
        let provider = BraketProvider::new(
            "AKID".into(),
            "SECRET".into(),
            "us-east-1".into(),
            None,
        );
        let circuit = QPUCircuit::bell_state();
        let cost = provider
            .estimate_cost(&circuit, "ionq_aria", 1000)
            .await
            .expect("should estimate")
            .expect("should return some");
        // $0.30 + 1000 * $0.01 = $10.30
        assert!((cost.estimated_cost_usd - 10.30).abs() < 0.01);
        assert!(!cost.is_free_tier);
    }

    #[tokio::test]
    async fn test_braket_cost_estimate_simulator() {
        let provider = BraketProvider::new(
            "AKID".into(),
            "SECRET".into(),
            "us-east-1".into(),
            None,
        );
        let circuit = QPUCircuit::bell_state();
        let cost = provider
            .estimate_cost(&circuit, "sv1", 1024)
            .await
            .expect("should estimate")
            .expect("should return some");
        // Simulator cost should be small
        assert!(cost.estimated_cost_usd < 1.0);
        assert!(!cost.is_free_tier); // Braket SV1 is not free
    }

    // -- Credential scope ---------------------------------------------------

    #[test]
    fn test_credential_scope() {
        assert_eq!(
            credential_scope("20240101", "us-east-1"),
            "20240101/us-east-1/braket/aws4_request"
        );
    }

    // -- Signing key derivation ---------------------------------------------

    #[test]
    fn test_derive_signing_key_length() {
        let key = derive_signing_key("secret", "20240101", "us-east-1");
        assert_eq!(key.len(), 32); // HMAC-SHA256 always produces 32 bytes
    }

    #[test]
    fn test_derive_signing_key_deterministic() {
        let key1 = derive_signing_key("secret", "20240101", "us-east-1");
        let key2 = derive_signing_key("secret", "20240101", "us-east-1");
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_derive_signing_key_varies_by_date() {
        let key1 = derive_signing_key("secret", "20240101", "us-east-1");
        let key2 = derive_signing_key("secret", "20240102", "us-east-1");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_derive_signing_key_varies_by_region() {
        let key1 = derive_signing_key("secret", "20240101", "us-east-1");
        let key2 = derive_signing_key("secret", "20240101", "eu-west-2");
        assert_ne!(key1, key2);
    }
}
