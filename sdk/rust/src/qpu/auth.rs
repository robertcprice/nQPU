//! Authentication configuration for QPU providers.

use crate::qpu::error::QPUError;
use std::collections::HashMap;

/// Authentication configuration for a QPU provider.
#[derive(Debug, Clone)]
pub enum AuthConfig {
    /// API token/key authentication (IBM Quantum, IonQ)
    Token(String),

    /// AWS credentials for Amazon Braket
    AwsCredentials {
        access_key_id: String,
        secret_access_key: String,
        region: String,
        session_token: Option<String>,
    },

    /// Azure credentials for Azure Quantum
    AzureCredentials {
        subscription_id: String,
        resource_group: String,
        workspace: String,
        tenant_id: String,
        client_id: String,
        client_secret: String,
    },

    /// Google Cloud credentials
    GoogleCredentials {
        project_id: String,
        /// Service account JSON key or OAuth2 token
        credentials_json: Option<String>,
    },

    /// Load from environment variables (provider-specific)
    FromEnvironment,
}

impl AuthConfig {
    /// Load IBM Quantum credentials from environment.
    /// Reads `IBM_QUANTUM_TOKEN` env var.
    pub fn ibm_from_env() -> Result<Self, QPUError> {
        let token = std::env::var("IBM_QUANTUM_TOKEN").map_err(|_| {
            QPUError::ConfigError(
                "IBM_QUANTUM_TOKEN environment variable not set. \
                 Get your token at https://quantum.ibm.com/"
                    .into(),
            )
        })?;
        Ok(AuthConfig::Token(token))
    }

    /// Load IonQ credentials from environment.
    /// Reads `IONQ_API_KEY` env var.
    pub fn ionq_from_env() -> Result<Self, QPUError> {
        let key = std::env::var("IONQ_API_KEY").map_err(|_| {
            QPUError::ConfigError(
                "IONQ_API_KEY environment variable not set. \
                 Get your API key at https://cloud.ionq.com/"
                    .into(),
            )
        })?;
        Ok(AuthConfig::Token(key))
    }

    /// Load AWS Braket credentials from environment.
    /// Reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.
    pub fn braket_from_env() -> Result<Self, QPUError> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| {
            QPUError::ConfigError("AWS_ACCESS_KEY_ID environment variable not set".into())
        })?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| {
            QPUError::ConfigError("AWS_SECRET_ACCESS_KEY environment variable not set".into())
        })?;
        let region = std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into());
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        Ok(AuthConfig::AwsCredentials {
            access_key_id: access_key,
            secret_access_key: secret_key,
            region,
            session_token,
        })
    }

    /// Load Azure Quantum credentials from environment.
    pub fn azure_from_env() -> Result<Self, QPUError> {
        let subscription_id = std::env::var("AZURE_QUANTUM_SUBSCRIPTION_ID").map_err(|_| {
            QPUError::ConfigError(
                "AZURE_QUANTUM_SUBSCRIPTION_ID environment variable not set".into(),
            )
        })?;
        let resource_group = std::env::var("AZURE_QUANTUM_RESOURCE_GROUP").map_err(|_| {
            QPUError::ConfigError(
                "AZURE_QUANTUM_RESOURCE_GROUP environment variable not set".into(),
            )
        })?;
        let workspace = std::env::var("AZURE_QUANTUM_WORKSPACE").map_err(|_| {
            QPUError::ConfigError("AZURE_QUANTUM_WORKSPACE environment variable not set".into())
        })?;
        let tenant_id = std::env::var("AZURE_TENANT_ID").map_err(|_| {
            QPUError::ConfigError("AZURE_TENANT_ID environment variable not set".into())
        })?;
        let client_id = std::env::var("AZURE_CLIENT_ID").map_err(|_| {
            QPUError::ConfigError("AZURE_CLIENT_ID environment variable not set".into())
        })?;
        let client_secret = std::env::var("AZURE_CLIENT_SECRET").map_err(|_| {
            QPUError::ConfigError("AZURE_CLIENT_SECRET environment variable not set".into())
        })?;

        Ok(AuthConfig::AzureCredentials {
            subscription_id,
            resource_group,
            workspace,
            tenant_id,
            client_id,
            client_secret,
        })
    }

    /// Load Google Cloud credentials from environment.
    pub fn google_from_env() -> Result<Self, QPUError> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
            QPUError::ConfigError("GOOGLE_CLOUD_PROJECT environment variable not set".into())
        })?;
        let credentials_json = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok();

        Ok(AuthConfig::GoogleCredentials {
            project_id,
            credentials_json,
        })
    }

    /// Extract bearer token for HTTP Authorization header.
    pub fn bearer_token(&self) -> Option<String> {
        match self {
            AuthConfig::Token(t) => Some(t.clone()),
            _ => None,
        }
    }

    /// Build HTTP headers for this auth config.
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        match self {
            AuthConfig::Token(token) => {
                headers.insert(
                    "Authorization".into(),
                    format!("Bearer {}", token),
                );
            }
            _ => {}
        }
        headers
    }
}
