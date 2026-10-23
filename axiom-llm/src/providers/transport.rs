//! HTTP transport seam between the providers and the network

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, Stream, StreamExt};
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
    /// `timeout_seconds` bounds connection setup. For non-streaming requests it also bounds the
    /// whole request. For streaming responses it bounds the silence between chunks, so a stream
    /// that stops sending fails instead of hanging.
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
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(AxiomError::from))
            .boxed();
        let stream = match self.timeout {
            Some(idle) => with_idle_timeout(stream, idle),
            None => stream,
        };
        Ok((status, stream))
    }
}

/// Fail a byte stream that goes quiet for longer than `idle`.
///
/// Each wait for the next chunk is bounded. On timeout the stream yields a
/// [`AxiomError::Timeout`] and then ends.
fn with_idle_timeout(stream: ByteStream, idle: Duration) -> ByteStream {
    let bounded = stream::unfold(Some(stream), move |state| async move {
        let mut stream = state?;
        match tokio::time::timeout(idle, stream.next()).await {
            Ok(Some(item)) => Some((item, Some(stream))),
            Ok(None) => None,
            Err(_elapsed) => Some((
                Err(AxiomError::Timeout(format!("No data received for {idle:?}"))),
                None,
            )),
        }
    });
    Box::pin(bounded)
}

/// Read a byte stream to completion and decode it as UTF-8 (lossily)
pub async fn collect_text(mut bytes: ByteStream) -> Result<String> {
    let mut buffer = Vec::new();
    while let Some(chunk) = bytes.next().await {
        buffer.extend_from_slice(&chunk?);
    }
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn quiet_stream_fails_after_the_idle_period() {
        let chunks = stream::iter(vec![Ok::<Bytes, AxiomError>(Bytes::from_static(b"hi"))])
            .chain(stream::pending());
        let mut bounded = with_idle_timeout(Box::pin(chunks), Duration::from_millis(20));

        assert_eq!(bounded.next().await.unwrap().unwrap(), Bytes::from_static(b"hi"));
        assert!(matches!(bounded.next().await, Some(Err(AxiomError::Timeout(_)))));
        assert!(bounded.next().await.is_none());
    }
}
