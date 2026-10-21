//! HTTP transport seam between the providers and the network

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::header::HeaderMap;
use reqwest::StatusCode;

use axiom_ai_core::{AxiomError, Result};

use super::ProviderConfig;

/// A stream of raw response body bytes
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

/// Sends JSON POST requests on behalf of a provider.
///
/// Implementations own the network: the production adapter is [`ReqwestTransport`],
/// and tests use [`crate::providers::testing::FakeTransport`].
#[async_trait]
pub trait HttpTransport: Send + Sync {
    /// POST `body` as JSON and return the status with the full response body
    async fn post_json(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &serde_json::Value,
    ) -> Result<(StatusCode, String)>;

    /// POST `body` as JSON and return the status with the response body as a byte stream
    async fn post_json_stream(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &serde_json::Value,
    ) -> Result<(StatusCode, ByteStream)>;
}

/// Production transport backed by `reqwest`
#[derive(Debug, Clone)]
pub struct ReqwestTransport {
    client: reqwest::Client,
    timeout: Option<Duration>,
}

impl ReqwestTransport {
    /// Create a transport from a provider configuration.
    ///
    /// `timeout_seconds` bounds connection setup and, for non-streaming requests,
    /// the whole request. Streaming responses are not cut off by the total timeout.
    pub fn new(config: &ProviderConfig) -> Result<Self> {
        let mut builder = reqwest::Client::builder();
        if let Some(seconds) = config.timeout_seconds {
            builder = builder.connect_timeout(Duration::from_secs(seconds));
        }
        let client = builder
            .build()
            .map_err(|e| AxiomError::LlmProvider(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            timeout: config.timeout_seconds.map(Duration::from_secs),
        })
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn post_json(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &serde_json::Value,
    ) -> Result<(StatusCode, String)> {
        let mut request = self.client.post(url).headers(headers).json(body);
        if let Some(timeout) = self.timeout {
            request = request.timeout(timeout);
        }
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        Ok((status, text))
    }

    async fn post_json_stream(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &serde_json::Value,
    ) -> Result<(StatusCode, ByteStream)> {
        let response = self.client.post(url).headers(headers).json(body).send().await?;
        let status = response.status();
        let stream = response.bytes_stream().map(|chunk| chunk.map_err(AxiomError::from));
        Ok((status, Box::pin(stream)))
    }
}

/// Read a byte stream to completion and decode it as UTF-8 (lossily)
pub async fn collect_text(mut bytes: ByteStream) -> Result<String> {
    let mut buffer = Vec::new();
    while let Some(chunk) = bytes.next().await {
        buffer.extend_from_slice(&chunk?);
    }
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}
