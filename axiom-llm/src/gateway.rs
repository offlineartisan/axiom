//! LLM Gateway - Unified interface for all LLM providers

use std::collections::HashMap;
use std::sync::Arc;

use axiom_ai_core::{Message, StreamResponse, Result, AxiomError, ChainInput, ChainOutput, Tool};
use crate::clock::Clock;
use crate::monitoring::{LlmMetrics, ProviderMetrics};
use crate::providers::{estimate_tokens, LlmProvider};
use crate::rate_limit::RateLimitConfig;
use crate::resilience::ResilientCall;
use crate::retry::RetryConfig;

/// Main LLM Gateway that provides a unified interface to all providers.
///
/// Every call goes through a [`ResilientCall`], so rate limits, retries and metrics apply to each attempt.
pub struct LlmGateway {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    default_provider: String,
    resilience: ResilientCall,
}

impl LlmGateway {
    /// Create a new LLM Gateway
    pub fn new(default_provider: impl Into<String>) -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: default_provider.into(),
            resilience: ResilientCall::default(),
        }
    }

    /// Add a provider to the gateway
    pub fn add_provider(mut self, name: impl Into<String>, provider: Arc<dyn LlmProvider>) -> Self {
        self.providers.insert(name.into(), provider);
        self
    }

    /// Set the default provider
    pub fn set_default_provider(mut self, name: impl Into<String>) -> Self {
        self.default_provider = name.into();
        self
    }

    /// Set retry configuration
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.resilience = self.resilience.with_retry_config(config);
        self
    }

    /// Set rate limiting configuration. Providers without a limit are not rate limited.
    pub fn with_rate_limit_config(mut self, config: RateLimitConfig) -> Self {
        self.resilience = self.resilience.with_rate_limit_config(config);
        self
    }

    /// Set the clock used for rate limiting and retry backoff
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.resilience = self.resilience.with_clock(clock);
        self
    }

    /// Get a provider by name
    pub fn get_provider(&self, name: &str) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(name)
    }

    /// Get the default provider
    pub fn default_provider(&self) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(&self.default_provider)
    }

    /// List all available providers
    pub fn list_providers(&self) -> Vec<&String> {
        self.providers.keys().collect()
    }

    /// Generate a completion using the default provider
    pub async fn generate(&self, request: LlmRequest) -> Result<LlmResponse> {
        self.generate_with_provider(&self.default_provider, request).await
    }

    /// Generate a completion using a specific provider
    pub async fn generate_with_provider(
        &self,
        provider_name: &str,
        request: LlmRequest,
    ) -> Result<LlmResponse> {
        let provider = self.require_provider(provider_name)?;
        let tokens = estimated_tokens(&request);

        let response = self
            .resilience
            .call(provider_name, tokens, || provider.generate(request.clone()))
            .await?;

        let tokens_used = u64::from(response.tokens_used.unwrap_or(0));
        self.resilience
            .metrics()
            .write()
            .await
            .record_usage(provider_name, tokens_used, response.cost.unwrap_or(0.0));

        Ok(response)
    }

    /// Generate a streaming completion using the default provider
    pub async fn generate_stream(&self, request: LlmRequest) -> Result<StreamResponse> {
        self.generate_stream_with_provider(&self.default_provider, request).await
    }

    /// Generate a streaming completion using a specific provider.
    ///
    /// The retry policy covers establishing the stream. Errors after the stream has started are
    /// the caller's to handle.
    pub async fn generate_stream_with_provider(
        &self,
        provider_name: &str,
        request: LlmRequest,
    ) -> Result<StreamResponse> {
        let provider = self.require_provider(provider_name)?;
        let tokens = estimated_tokens(&request);

        self.resilience
            .call(provider_name, tokens, || provider.generate_stream(request.clone()))
            .await
    }

    /// Get metrics for all providers (aggregate totals plus a per-provider breakdown)
    pub async fn get_metrics(&self) -> LlmMetrics {
        self.resilience.metrics().read().await.clone()
    }

    /// Get metrics for a single provider. `None` if the provider has not been called.
    pub async fn get_provider_metrics(&self, provider_name: &str) -> Option<ProviderMetrics> {
        self.resilience
            .metrics()
            .read()
            .await
            .get_provider_metrics(provider_name)
            .cloned()
    }

    fn require_provider(&self, provider_name: &str) -> Result<&Arc<dyn LlmProvider>> {
        self.get_provider(provider_name)
            .ok_or_else(|| AxiomError::LlmProvider(format!("Provider not found: {}", provider_name)))
    }
}

/// Estimated token cost of one attempt: the prompt plus the requested output budget
fn estimated_tokens(request: &LlmRequest) -> u32 {
    let prompt = u32::try_from(estimate_tokens(&request.messages)).unwrap_or(u32::MAX);
    prompt.saturating_add(request.max_tokens.unwrap_or(0))
}

/// Request to an LLM provider
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// Messages in the conversation
    pub messages: Vec<Message>,
    /// Model to use
    pub model: String,
    /// Temperature for generation
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Whether to stream the response
    pub stream: bool,
    /// Tools referenced by tool calls in `messages`.
    ///
    /// Providers send these as context only and tell the model not to call tools itself:
    /// tool calls in this crate are issued by the caller, not by the model.
    pub tools: Vec<ToolDefinition>,
    /// Additional parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Description of a tool, in the form providers send to the model
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// Tool name, matching the `name` of tool calls in the conversation
    pub name: String,
    /// What the tool does
    pub description: String,
    /// JSON Schema of the tool's arguments
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Describe a tool from its name, description and parameter schema
    pub fn from_tool(tool: &dyn Tool) -> Self {
        Self {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.parameters().to_json_schema(),
        }
    }
}

impl LlmRequest {
    /// Create a new LLM request
    pub fn new(messages: Vec<Message>, model: impl Into<String>) -> Self {
        Self {
            messages,
            model: model.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop_sequences: Vec::new(),
            stream: false,
            tools: Vec::new(),
            parameters: HashMap::new(),
        }
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-p parameter
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the top-k parameter
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set presence penalty
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Set frequency penalty
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Add stop sequences
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set the tools referenced by tool calls in the conversation
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a custom parameter
    pub fn with_parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Convert from chain input
    pub fn from_chain_input(input: ChainInput) -> Self {
        Self {
            messages: input.messages,
            model: "gpt-3.5-turbo".to_string(), // Default model
            temperature: input.config.temperature,
            max_tokens: input.config.max_tokens,
            top_p: input.config.top_p,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop_sequences: Vec::new(),
            stream: input.config.stream,
            tools: Vec::new(),
            parameters: input.config.parameters,
        }
    }
}

/// Response from an LLM provider
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Generated content
    pub content: String,
    /// Model used for generation
    pub model: String,
    /// Provider used
    pub provider: String,
    /// Number of tokens used
    pub tokens_used: Option<u32>,
    /// Cost of the request
    pub cost: Option<f64>,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LlmResponse {
    /// Create a new LLM response
    pub fn new(content: impl Into<String>, model: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            model: model.into(),
            provider: provider.into(),
            tokens_used: None,
            cost: None,
            generation_time_ms: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set the number of tokens used
    pub fn with_tokens_used(mut self, tokens: u32) -> Self {
        self.tokens_used = Some(tokens);
        self
    }

    /// Set the cost
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Set the generation time
    pub fn with_generation_time(mut self, time_ms: u64) -> Self {
        self.generation_time_ms = time_ms;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Convert to chain output
    pub fn to_chain_output(self) -> ChainOutput {
        ChainOutput {
            messages: vec![axiom_ai_core::Message::assistant(self.content)],
            data: self.metadata,
            metadata: axiom_ai_core::ChainMetadata {
                execution_time_ms: self.generation_time_ms,
                tokens_used: self.tokens_used,
                cost: self.cost,
                data: HashMap::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FakeClock;
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    /// Provider that returns scripted generate results in order and counts every call
    struct ScriptedProvider {
        name: String,
        script: Mutex<VecDeque<Result<LlmResponse>>>,
        calls: AtomicU32,
    }

    impl ScriptedProvider {
        fn new(name: &str, script: Vec<Result<LlmResponse>>) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                script: Mutex::new(script.into()),
                calls: AtomicU32::new(0),
            })
        }

        fn calls(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Scripted provider for tests"
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["test-model".to_string()]
        }

        async fn generate(&self, _request: LlmRequest) -> Result<LlmResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.script
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Err(AxiomError::Internal("script exhausted".to_string())))
        }

        async fn generate_stream(&self, _request: LlmRequest) -> Result<StreamResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(AxiomError::LlmProvider("streaming not implemented".to_string()))
        }

        fn get_cost_per_token(&self, _model: &str) -> Option<f64> {
            None
        }

        fn get_max_context_length(&self, _model: &str) -> Option<usize> {
            None
        }
    }

    fn request() -> LlmRequest {
        LlmRequest::new(Vec::new(), "test-model")
    }

    fn ok_response() -> LlmResponse {
        LlmResponse::new("hello", "test-model", "a").with_tokens_used(42)
    }

    fn unavailable() -> AxiomError {
        AxiomError::ProviderUnavailable("503 Service Unavailable".to_string())
    }

    fn make_gateway(clock: &Arc<FakeClock>) -> LlmGateway {
        LlmGateway::new("a")
            .with_clock(clock.clone())
            .with_retry_config(RetryConfig::new(3).with_initial_delay(Duration::from_millis(100)))
    }

    #[tokio::test]
    async fn transient_failure_is_retried_and_each_attempt_is_recorded() {
        let clock = Arc::new(FakeClock::new());
        let provider = ScriptedProvider::new("a", vec![Err(unavailable()), Ok(ok_response())]);
        let gateway = make_gateway(&clock).add_provider("a", provider.clone());

        let response = gateway.generate_with_provider("a", request()).await.unwrap();

        assert_eq!(response.content, "hello");
        assert_eq!(provider.calls(), 2);
        assert_eq!(clock.sleeps(), vec![Duration::from_millis(100)]);

        let metrics = gateway.get_provider_metrics("a").await.expect("provider a recorded");
        assert_eq!(
            (metrics.total_requests, metrics.successful_requests, metrics.failed_requests),
            (2, 1, 1)
        );
        assert_eq!(metrics.total_tokens, 42);
    }

    #[tokio::test]
    async fn fatal_provider_error_is_not_retried() {
        let clock = Arc::new(FakeClock::new());
        let provider = ScriptedProvider::new(
            "a",
            vec![Err(AxiomError::LlmProvider("model not supported".to_string())), Ok(ok_response())],
        );
        let gateway = make_gateway(&clock).add_provider("a", provider.clone());

        let err = gateway.generate_with_provider("a", request()).await.unwrap_err();

        assert!(matches!(err, AxiomError::LlmProvider(_)));
        assert_eq!(provider.calls(), 1);
        assert!(clock.sleeps().is_empty());

        let metrics = gateway.get_provider_metrics("a").await.expect("provider a recorded");
        assert_eq!(metrics.failed_requests, 1);
    }

    #[tokio::test]
    async fn unknown_provider_is_an_error_and_records_nothing() {
        let clock = Arc::new(FakeClock::new());
        let gateway = make_gateway(&clock);

        let err = gateway.generate_with_provider("missing", request()).await.unwrap_err();

        assert!(matches!(err, AxiomError::LlmProvider(_)));
        assert!(gateway.get_provider_metrics("missing").await.is_none());
    }

    #[tokio::test]
    async fn failures_are_counted_for_the_provider_that_failed_only() {
        let clock = Arc::new(FakeClock::new());
        let failing = ScriptedProvider::new(
            "a",
            vec![Err(AxiomError::LlmProvider("bad request".to_string()))],
        );
        let healthy = ScriptedProvider::new("b", vec![Ok(ok_response())]);
        let gateway = make_gateway(&clock)
            .add_provider("a", failing)
            .add_provider("b", healthy);

        assert!(gateway.generate_with_provider("a", request()).await.is_err());
        assert!(gateway.generate_with_provider("b", request()).await.is_ok());

        let a = gateway.get_provider_metrics("a").await.expect("a recorded");
        assert_eq!((a.failed_requests, a.successful_requests), (1, 0));

        let b = gateway.get_provider_metrics("b").await.expect("b recorded");
        assert_eq!((b.failed_requests, b.successful_requests), (0, 1));

        let all = gateway.get_metrics().await;
        assert_eq!((all.total_requests, all.failed_requests, all.successful_requests), (2, 1, 1));
    }

    #[tokio::test]
    async fn rate_limit_is_checked_before_every_retry_attempt() {
        let clock = Arc::new(FakeClock::new());
        let rate_limits = RateLimitConfig::new();
        rate_limits.add_limiter("a".to_string(), Some(1), None).await;
        let provider = ScriptedProvider::new("a", vec![Err(unavailable()), Ok(ok_response())]);
        let gateway = make_gateway(&clock)
            .with_rate_limit_config(rate_limits)
            .add_provider("a", provider.clone());

        let err = gateway.generate_with_provider("a", request()).await.unwrap_err();

        // The first attempt used the only permit. Every retry is rejected before reaching the provider.
        assert_eq!(provider.calls(), 1);
        assert!(matches!(err, AxiomError::RateLimit(_)));
    }

    #[tokio::test]
    async fn stream_establishment_is_not_retried_for_fatal_errors() {
        let clock = Arc::new(FakeClock::new());
        let provider = ScriptedProvider::new("a", Vec::new());
        let gateway = make_gateway(&clock).add_provider("a", provider.clone());

        let err = gateway
            .generate_stream_with_provider("a", request())
            .await
            .err()
            .expect("streaming is not implemented by the test provider");

        assert!(matches!(err, AxiomError::LlmProvider(_)));
        assert_eq!(provider.calls(), 1);
        assert!(clock.sleeps().is_empty());

        let metrics = gateway.get_provider_metrics("a").await.expect("provider a recorded");
        assert_eq!(metrics.failed_requests, 1);
    }
}
