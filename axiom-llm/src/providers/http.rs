//! Generic HTTP provider, parameterised by a wire-format dialect

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::stream;
use futures::StreamExt;
use reqwest::header::HeaderMap;

use axiom_ai_core::{AxiomError, ChunkType, Message, MessageContent, MessagePart, Result, StreamChunk, StreamResponse};

use crate::gateway::{LlmRequest, LlmResponse};

use super::base::{handle_provider_error, LlmProvider};
use super::transport::{collect_text, ByteStream, HttpTransport, ReqwestTransport};
use super::ProviderConfig;

/// Token counts reported by a provider. Either side may be unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    /// Tokens in the prompt
    pub prompt_tokens: Option<u32>,
    /// Tokens in the completion
    pub completion_tokens: Option<u32>,
}

impl TokenUsage {
    /// Total tokens, if at least one side is known
    pub fn total(&self) -> Option<u32> {
        match (self.prompt_tokens, self.completion_tokens) {
            (None, None) => None,
            (prompt, completion) => {
                Some(prompt.unwrap_or(0).saturating_add(completion.unwrap_or(0)))
            }
        }
    }

    /// Merge newer counts into these, keeping existing values where the new ones are unknown
    pub fn merge(&mut self, other: TokenUsage) {
        if other.prompt_tokens.is_some() {
            self.prompt_tokens = other.prompt_tokens;
        }
        if other.completion_tokens.is_some() {
            self.completion_tokens = other.completion_tokens;
        }
    }
}

/// A complete (non-streaming) completion parsed from a provider response
#[derive(Debug, Clone, Default)]
pub struct ParsedCompletion {
    /// Generated text
    pub content: String,
    /// Token usage reported with the completion
    pub usage: TokenUsage,
}

/// One meaningful event decoded from a provider's stream
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// A piece of generated text
    TextDelta(String),
    /// Token counts; merged into the running totals
    Usage(TokenUsage),
    /// The provider has finished the response
    Done,
}

/// Wire-format details for one LLM API.
///
/// A dialect supplies only what differs between APIs. [`HttpProvider`] owns the
/// transport, timing, status handling, SSE parsing and `LlmProvider` plumbing.
pub trait Dialect: Send + Sync + 'static {
    /// Provider name, e.g. `"openai"`
    fn name(&self) -> &'static str;

    /// Human-readable description
    fn description(&self) -> &'static str;

    /// Base URL used when the configuration does not override it
    fn default_base_url(&self) -> &'static str;

    /// Path appended to the base URL for chat requests
    fn endpoint_path(&self) -> &'static str;

    /// Authentication headers for an API key (empty when there is no key)
    fn auth_headers(&self, api_key: Option<&str>) -> Result<HeaderMap>;

    /// Translate a request into the provider's JSON body
    fn build_request_body(&self, request: &LlmRequest, stream: bool) -> Result<serde_json::Value>;

    /// Parse a complete (non-streaming) response body
    fn parse_response(&self, body: &str) -> Result<ParsedCompletion>;

    /// Parse one SSE `data:` payload into zero or more stream events
    fn parse_stream_event(&self, data: &str) -> Result<Vec<StreamEvent>>;

    /// Models this dialect accepts
    fn models(&self) -> &'static [&'static str];

    /// Cost per token (USD) for a model, if known
    fn get_cost_per_token(&self, model: &str) -> Option<f64>;

    /// Maximum context length for a model, if known
    fn get_max_context_length(&self, model: &str) -> Option<usize>;
}

/// An LLM provider that speaks a [`Dialect`] over an [`HttpTransport`]
pub struct HttpProvider<D: Dialect> {
    dialect: Arc<D>,
    base_url: String,
    api_key: Option<String>,
    transport: Arc<dyn HttpTransport>,
}

impl<D: Dialect + Default> HttpProvider<D> {
    /// Create a provider that talks to the network through reqwest
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let transport = ReqwestTransport::new(&config)?;
        Ok(Self::with_transport(config, Arc::new(transport)))
    }

    /// Create a provider that sends requests through the given transport
    pub fn with_transport(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> Self {
        let dialect = D::default();
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| dialect.default_base_url().to_string());
        Self {
            dialect: Arc::new(dialect),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: config.api_key,
            transport,
        }
    }
}

impl<D: Dialect> HttpProvider<D> {
    /// Full URL for chat requests
    fn endpoint(&self) -> String {
        format!("{}{}", self.base_url, self.dialect.endpoint_path())
    }

    /// Headers for a request, failing if the key cannot be encoded as a header value
    fn headers(&self) -> Result<HeaderMap> {
        self.dialect.auth_headers(self.api_key.as_deref())
    }
}

#[async_trait]
impl<D: Dialect> LlmProvider for HttpProvider<D> {
    fn name(&self) -> &str {
        self.dialect.name()
    }

    fn description(&self) -> &str {
        self.dialect.description()
    }

    fn supported_models(&self) -> Vec<String> {
        self.dialect.models().iter().map(|model| model.to_string()).collect()
    }

    async fn generate(&self, request: LlmRequest) -> Result<LlmResponse> {
        self.validate_request(&request)?;

        let start_time = Instant::now();
        let body = self.dialect.build_request_body(&request, false)?;
        let (status, text) = self.transport.post_json(&self.endpoint(), self.headers()?, &body).await?;

        if !status.is_success() {
            return Err(handle_provider_error(status, &text));
        }

        let parsed = self.dialect.parse_response(&text)?;
        let mut response = LlmResponse::new(parsed.content, &request.model, self.dialect.name());

        if let Some(tokens) = parsed.usage.total() {
            response = response.with_tokens_used(tokens);
            if let Some(rate) = self.dialect.get_cost_per_token(&request.model) {
                response = response.with_cost(rate * f64::from(tokens));
            }
        }

        Ok(response.with_generation_time(start_time.elapsed().as_millis() as u64))
    }

    async fn generate_stream(&self, request: LlmRequest) -> Result<StreamResponse> {
        self.validate_request(&request)?;

        let body = self.dialect.build_request_body(&request, true)?;
        let (status, bytes) = self.transport.post_json_stream(&self.endpoint(), self.headers()?, &body).await?;

        if !status.is_success() {
            let text = collect_text(bytes).await?;
            return Err(handle_provider_error(status, &text));
        }

        let state = StreamState::new(bytes, Arc::clone(&self.dialect));
        let chunks = stream::unfold(state, |mut state| async move {
            let item = state.next_item().await?;
            Some((item, state))
        });
        Ok(StreamResponse::new(chunks))
    }

    fn get_cost_per_token(&self, model: &str) -> Option<f64> {
        self.dialect.get_cost_per_token(model)
    }

    fn get_max_context_length(&self, model: &str) -> Option<usize> {
        self.dialect.get_max_context_length(model)
    }
}

/// Mutable state behind a streaming response
struct StreamState<D: Dialect> {
    bytes: ByteStream,
    decoder: SseDecoder,
    events: VecDeque<StreamEvent>,
    dialect: Arc<D>,
    usage: TokenUsage,
    eof: bool,
    finished: bool,
}

impl<D: Dialect> StreamState<D> {
    fn new(bytes: ByteStream, dialect: Arc<D>) -> Self {
        Self {
            bytes,
            decoder: SseDecoder::default(),
            events: VecDeque::new(),
            dialect,
            usage: TokenUsage::default(),
            eof: false,
            finished: false,
        }
    }

    /// Produce the next chunk, or `None` once the stream has ended
    async fn next_item(&mut self) -> Option<Result<StreamChunk>> {
        loop {
            if let Some(event) = self.events.pop_front() {
                match event {
                    StreamEvent::TextDelta(text) => return Some(Ok(text_chunk(text, false))),
                    StreamEvent::Usage(usage) => self.usage.merge(usage),
                    StreamEvent::Done => {
                        self.finished = true;
                        self.events.clear();
                        return Some(Ok(final_chunk(self.usage)));
                    }
                }
                continue;
            }

            if self.finished {
                return None;
            }

            if self.eof {
                self.finished = true;
                return Some(Err(AxiomError::LlmProvider(
                    "Stream ended before the provider signalled completion".to_string(),
                )));
            }

            match self.bytes.next().await {
                Some(Ok(chunk)) => {
                    let payloads = self.decoder.push(&chunk);
                    if let Err(e) = self.absorb(payloads) {
                        self.finished = true;
                        return Some(Err(e));
                    }
                }
                Some(Err(e)) => {
                    self.finished = true;
                    return Some(Err(e));
                }
                None => {
                    self.eof = true;
                    let payloads = self.decoder.finish();
                    if let Err(e) = self.absorb(payloads) {
                        self.finished = true;
                        return Some(Err(e));
                    }
                }
            }
        }
    }

    /// Decode SSE payloads into queued stream events
    fn absorb(&mut self, payloads: Vec<String>) -> Result<()> {
        for payload in payloads {
            if payload == "[DONE]" {
                self.events.push_back(StreamEvent::Done);
            } else {
                let events = self.dialect.parse_stream_event(&payload)?;
                self.events.extend(events);
            }
        }
        Ok(())
    }
}

fn text_chunk(content: String, is_final: bool) -> StreamChunk {
    StreamChunk {
        content,
        chunk_type: ChunkType::Text,
        metadata: None,
        is_final,
    }
}

fn final_chunk(usage: TokenUsage) -> StreamChunk {
    StreamChunk {
        content: String::new(),
        chunk_type: ChunkType::Text,
        metadata: usage.total().map(|tokens| serde_json::json!({ "tokens_used": tokens })),
        is_final: true,
    }
}

/// Incremental decoder for Server-Sent Events.
///
/// Feed it arbitrary byte chunks; it returns the joined `data:` payload of each
/// event as the blank line that terminates it arrives.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    data: Vec<String>,
}

impl SseDecoder {
    /// Feed bytes; returns the payloads of events completed by them
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|&b| b == b'\n') {
            let mut raw: Vec<u8> = self.buffer.drain(..=newline).collect();
            raw.pop();
            self.handle_line(&raw, &mut events);
        }
        events
    }

    /// Flush at end of input; returns any event still pending
    pub fn finish(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let raw = std::mem::take(&mut self.buffer);
            self.handle_line(&raw, &mut events);
        }
        self.dispatch(&mut events);
        events
    }

    fn handle_line(&mut self, raw: &[u8], events: &mut Vec<String>) {
        let text = String::from_utf8_lossy(raw);
        let line = text.strip_suffix('\r').unwrap_or(&text);
        if line.is_empty() {
            self.dispatch(events);
        } else if let Some(value) = line.strip_prefix("data:") {
            self.data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
        // Other fields (event, id, retry) and comments carry nothing we need.
    }

    fn dispatch(&mut self, events: &mut Vec<String>) {
        if !self.data.is_empty() {
            events.push(self.data.drain(..).collect::<Vec<_>>().join("\n"));
        }
    }
}

/// Text of a message, with multi-part content reduced to its text parts joined by spaces
pub(crate) fn message_text(message: &Message) -> String {
    match &message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_decoder_handles_events_split_across_chunks() {
        let mut decoder = SseDecoder::default();
        let mut events = Vec::new();
        events.extend(decoder.push(b"event: a\r\ndata: {\"x\"")); // split mid-line
        events.extend(decoder.push(b":1}\r\n\r\n: comment\n"));
        events.extend(decoder.push(b"data: one\ndata: two\n\ndata: [DONE]\n\n"));
        assert_eq!(events, vec!["{\"x\":1}", "one\ntwo", "[DONE]"]);
    }

    #[test]
    fn sse_decoder_flushes_unterminated_final_event() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: last").is_empty());
        assert_eq!(decoder.finish(), vec!["last"]);
    }
}
