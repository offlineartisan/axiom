//! In-memory [`HttpTransport`] for exercising providers without the network

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream;
use futures::StreamExt;
use reqwest::header::HeaderMap;
use reqwest::StatusCode;

use axiom_ai_core::{AxiomError, Result};

use super::transport::{ByteStream, HttpTransport};

/// A request as the provider sent it
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    /// Full request URL
    pub url: String,
    /// Headers the provider set
    pub headers: HeaderMap,
    /// JSON body
    pub body: serde_json::Value,
}

#[derive(Debug, Clone)]
struct ScriptedResponse {
    status: StatusCode,
    chunks: Vec<Bytes>,
}

#[derive(Debug, Default)]
struct FakeState {
    script: VecDeque<ScriptedResponse>,
    requests: Vec<RecordedRequest>,
}

/// Records requests and replays scripted responses, in the order they were queued.
///
/// Clones share state, so keep one clone for assertions and give another to the provider.
#[derive(Debug, Clone, Default)]
pub struct FakeTransport {
    state: Arc<Mutex<FakeState>>,
}

impl FakeTransport {
    /// Create an empty fake transport
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a response whose entire body is `body`
    pub fn respond(&self, status: StatusCode, body: impl Into<String>) {
        self.push(status, vec![Bytes::from(body.into())]);
    }

    /// Queue a response whose body arrives as the given chunks, e.g. an SSE
    /// stream split at arbitrary byte boundaries
    pub fn respond_chunks(&self, status: StatusCode, chunks: &[&str]) {
        let chunks = chunks.iter().map(|chunk| Bytes::from(chunk.to_string())).collect();
        self.push(status, chunks);
    }

    /// Every request received so far, in order
    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.lock().requests.clone()
    }

    fn push(&self, status: StatusCode, chunks: Vec<Bytes>) {
        self.lock().script.push_back(ScriptedResponse { status, chunks });
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FakeState> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record the request and pop the next scripted response
    fn serve(&self, url: &str, headers: HeaderMap, body: &serde_json::Value) -> Result<ScriptedResponse> {
        let mut state = self.lock();
        state.requests.push(RecordedRequest {
            url: url.to_string(),
            headers,
            body: body.clone(),
        });
        state.script.pop_front().ok_or_else(|| {
            AxiomError::LlmProvider(format!("FakeTransport has no scripted response for {}", url))
        })
    }
}

#[async_trait]
impl HttpTransport for FakeTransport {
    async fn post_json(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &serde_json::Value,
    ) -> Result<(StatusCode, String)> {
        let scripted = self.serve(url, headers, body)?;
        let bytes: Vec<u8> = scripted.chunks.iter().flat_map(|chunk| chunk.iter().copied()).collect();
        Ok((scripted.status, String::from_utf8_lossy(&bytes).into_owned()))
    }

    async fn post_json_stream(
        &self,
        url: &str,
        headers: HeaderMap,
        body: &serde_json::Value,
    ) -> Result<(StatusCode, ByteStream)> {
        let scripted = self.serve(url, headers, body)?;
        let chunks = stream::iter(scripted.chunks.into_iter().map(Ok::<Bytes, AxiomError>));
        Ok((scripted.status, chunks.boxed()))
    }
}
