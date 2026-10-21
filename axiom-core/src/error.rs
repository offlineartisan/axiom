//! Error types and result handling for Axiom

use std::sync::Arc;

use serde::de::Error as _;
use thiserror::Error;

/// Main error type for Axiom operations
#[derive(Error, Debug)]
pub enum AxiomError {
    #[error("Configuration error: {0}")]
    Config(String),

    /// A permanent provider error (bad request, unsupported model, context too long, ...).
    /// Never retried.
    #[error("LLM provider error: {0}")]
    LlmProvider(String),

    /// A transient provider failure (5xx, 529 overloaded). Retried with backoff.
    #[error("Provider unavailable: {0}")]
    ProviderUnavailable(String),

    /// Transport failure. Shared so the error can be cloned without losing its variant.
    #[error("Network error: {0}")]
    Network(Arc<reqwest::Error>),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Tool execution error: {0}")]
    ToolExecution(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Chain execution error: {0}")]
    ChainExecution(String),

    #[error("WASM execution error: {0}")]
    WasmExecution(String),

    #[error("Monitoring error: {0}")]
    Monitoring(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convenient result type alias
pub type Result<T> = std::result::Result<T, AxiomError>;

impl From<reqwest::Error> for AxiomError {
    fn from(err: reqwest::Error) -> Self {
        AxiomError::Network(Arc::new(err))
    }
}

impl AxiomError {
    /// Check if this error is worth retrying.
    ///
    /// Transport failures, timeouts, rate-limit rejections and transient provider
    /// failures are retryable. Everything else, including `LlmProvider` errors, is
    /// permanent.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            AxiomError::Network(_)
                | AxiomError::Timeout(_)
                | AxiomError::RateLimit(_)
                | AxiomError::ProviderUnavailable(_)
        )
    }

    /// Get a user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            AxiomError::Config(msg) => msg.clone(),
            AxiomError::LlmProvider(msg) => msg.clone(),
            AxiomError::ProviderUnavailable(msg) => msg.clone(),
            AxiomError::Network(err) => err.to_string(),
            AxiomError::Serialization(err) => err.to_string(),
            AxiomError::Io(err) => err.to_string(),
            AxiomError::Timeout(msg) => msg.clone(),
            AxiomError::RateLimit(msg) => msg.clone(),
            AxiomError::Authentication(msg) => msg.clone(),
            AxiomError::Validation(msg) => msg.clone(),
            AxiomError::ToolExecution(msg) => msg.clone(),
            AxiomError::Memory(msg) => msg.clone(),
            AxiomError::ChainExecution(msg) => msg.clone(),
            AxiomError::WasmExecution(msg) => msg.clone(),
            AxiomError::Monitoring(msg) => msg.clone(),
            AxiomError::Internal(msg) => msg.clone(),
        }
    }
}

impl Clone for AxiomError {
    /// Every variant keeps its variant and retryability when cloned.
    ///
    /// `Network` shares its `reqwest::Error`. `Serialization` and `Io` wrap types that
    /// are not `Clone`, so they are rebuilt with the same message (and, for IO, the same kind).
    fn clone(&self) -> Self {
        match self {
            AxiomError::Config(msg) => AxiomError::Config(msg.clone()),
            AxiomError::LlmProvider(msg) => AxiomError::LlmProvider(msg.clone()),
            AxiomError::ProviderUnavailable(msg) => AxiomError::ProviderUnavailable(msg.clone()),
            AxiomError::Network(err) => AxiomError::Network(Arc::clone(err)),
            AxiomError::Serialization(err) => {
                AxiomError::Serialization(serde_json::Error::custom(err.to_string()))
            }
            AxiomError::Io(err) => AxiomError::Io(std::io::Error::new(err.kind(), err.to_string())),
            AxiomError::Timeout(msg) => AxiomError::Timeout(msg.clone()),
            AxiomError::RateLimit(msg) => AxiomError::RateLimit(msg.clone()),
            AxiomError::Authentication(msg) => AxiomError::Authentication(msg.clone()),
            AxiomError::Validation(msg) => AxiomError::Validation(msg.clone()),
            AxiomError::ToolExecution(msg) => AxiomError::ToolExecution(msg.clone()),
            AxiomError::Memory(msg) => AxiomError::Memory(msg.clone()),
            AxiomError::ChainExecution(msg) => AxiomError::ChainExecution(msg.clone()),
            AxiomError::WasmExecution(msg) => AxiomError::WasmExecution(msg.clone()),
            AxiomError::Monitoring(msg) => AxiomError::Monitoring(msg.clone()),
            AxiomError::Internal(msg) => AxiomError::Internal(msg.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn network_error() -> AxiomError {
        // An invalid URL fails in the builder, so no network access happens.
        let err = reqwest::Client::new()
            .get("not a url")
            .build()
            .expect_err("invalid URL should fail to build");
        AxiomError::from(err)
    }

    #[test]
    fn network_error_clone_keeps_variant_and_retryability() {
        let original = network_error();
        let cloned = original.clone();

        assert!(matches!(cloned, AxiomError::Network(_)));
        assert!(cloned.is_retryable());
        assert_eq!(cloned.to_string(), original.to_string());
    }

    #[test]
    fn clone_keeps_variant_for_non_clone_payloads() {
        let io = AxiomError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
        let cloned_io = io.clone();
        assert!(matches!(&cloned_io, AxiomError::Io(e) if e.kind() == std::io::ErrorKind::NotFound));
        assert_eq!(cloned_io.to_string(), io.to_string());

        let serde = AxiomError::from(serde_json::from_str::<u32>("nope").unwrap_err());
        let cloned_serde = serde.clone();
        assert!(matches!(cloned_serde, AxiomError::Serialization(_)));
        assert_eq!(cloned_serde.to_string(), serde.to_string());
    }

    #[test]
    fn only_transient_errors_are_retryable() {
        assert!(AxiomError::ProviderUnavailable("503".into()).is_retryable());
        assert!(AxiomError::RateLimit("slow down".into()).is_retryable());
        assert!(AxiomError::Timeout("slow".into()).is_retryable());

        assert!(!AxiomError::LlmProvider("model not supported".into()).is_retryable());
        assert!(!AxiomError::Authentication("bad key".into()).is_retryable());
        assert!(!AxiomError::Config("bad config".into()).is_retryable());
    }
}
