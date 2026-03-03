//! QPU error types for quantum hardware provider operations.

use std::fmt;
use std::time::Duration;

/// Unified error type for all QPU provider operations.
#[derive(Debug, Clone)]
pub enum QPUError {
    /// Authentication failed (invalid token, expired credentials, etc.)
    AuthenticationError(String),

    /// HTTP/network error during API communication
    NetworkError(String),

    /// Circuit validation failed before submission
    ValidationError(ValidationError),

    /// Job submission was rejected by the provider
    SubmissionError(String),

    /// Job failed during execution on quantum hardware
    ExecutionError(String),

    /// Job did not complete within the specified timeout
    Timeout,

    /// Provider rate-limited the request
    RateLimited {
        /// Suggested wait time before retrying
        retry_after: Duration,
    },

    /// Requested backend/device is not available
    BackendUnavailable(String),

    /// Circuit format conversion error
    ConversionError(String),

    /// Provider-specific error not covered by other variants
    ProviderError {
        provider: String,
        message: String,
        status_code: Option<u16>,
    },

    /// Feature not supported by this provider
    UnsupportedFeature(String),

    /// Configuration error (missing env vars, invalid config)
    ConfigError(String),
}

/// Circuit validation errors detected before submission.
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Circuit uses gates not supported by the target backend
    UnsupportedGates(Vec<String>),

    /// Circuit requires more qubits than the backend has
    TooManyQubits {
        circuit: usize,
        backend: usize,
    },

    /// Circuit violates hardware connectivity constraints
    ConnectivityViolation(Vec<(usize, usize)>),

    /// Circuit exceeds maximum depth for the backend
    CircuitTooDeep {
        depth: usize,
        max_depth: usize,
    },

    /// Circuit has no measurements (nothing to return)
    NoMeasurements,

    /// Circuit is empty
    EmptyCircuit,
}

impl fmt::Display for QPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QPUError::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
            QPUError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            QPUError::ValidationError(ve) => write!(f, "Validation error: {}", ve),
            QPUError::SubmissionError(msg) => write!(f, "Submission error: {}", msg),
            QPUError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            QPUError::Timeout => write!(f, "Job timed out"),
            QPUError::RateLimited { retry_after } => {
                write!(f, "Rate limited, retry after {:?}", retry_after)
            }
            QPUError::BackendUnavailable(name) => write!(f, "Backend unavailable: {}", name),
            QPUError::ConversionError(msg) => write!(f, "Circuit conversion error: {}", msg),
            QPUError::ProviderError {
                provider,
                message,
                status_code,
            } => {
                if let Some(code) = status_code {
                    write!(f, "[{}] HTTP {}: {}", provider, code, message)
                } else {
                    write!(f, "[{}] {}", provider, message)
                }
            }
            QPUError::UnsupportedFeature(msg) => write!(f, "Unsupported feature: {}", msg),
            QPUError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::UnsupportedGates(gates) => {
                write!(f, "Unsupported gates: {}", gates.join(", "))
            }
            ValidationError::TooManyQubits { circuit, backend } => {
                write!(
                    f,
                    "Circuit requires {} qubits but backend has {}",
                    circuit, backend
                )
            }
            ValidationError::ConnectivityViolation(pairs) => {
                write!(f, "Connectivity violations on {} qubit pairs", pairs.len())
            }
            ValidationError::CircuitTooDeep { depth, max_depth } => {
                write!(
                    f,
                    "Circuit depth {} exceeds backend max {}",
                    depth, max_depth
                )
            }
            ValidationError::NoMeasurements => write!(f, "Circuit has no measurements"),
            ValidationError::EmptyCircuit => write!(f, "Circuit is empty"),
        }
    }
}

impl std::error::Error for QPUError {}
impl std::error::Error for ValidationError {}
