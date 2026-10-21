//! OpenAI provider implementation (chat completions API)

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};

use axiom_ai_core::{AxiomError, Message, MessageContent, MessagePart, MessageRole, Result};

use crate::gateway::{LlmRequest, ToolDefinition};

use super::http::{message_text, Dialect, HttpProvider, ParsedCompletion, StreamEvent, TokenUsage};

/// OpenAI provider: [`HttpProvider`] speaking the OpenAI chat completions dialect
pub type OpenAIProvider = HttpProvider<OpenAiDialect>;

/// Wire format for the OpenAI chat completions API
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiDialect;

const MODELS: &[&str] = &[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-32k",
    "gpt-4-turbo",
    "text-davinci-003",
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
];

impl Dialect for OpenAiDialect {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn description(&self) -> &'static str {
        "OpenAI GPT models provider"
    }

    fn default_base_url(&self) -> &'static str {
        "https://api.openai.com/v1"
    }

    fn endpoint_path(&self) -> &'static str {
        "/chat/completions"
    }

    fn auth_headers(&self, api_key: Option<&str>) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        if let Some(key) = api_key {
            let value = HeaderValue::from_str(&format!("Bearer {}", key)).map_err(|e| {
                AxiomError::Config(format!("OpenAI API key is not a valid header value: {}", e))
            })?;
            headers.insert(AUTHORIZATION, value);
        }
        Ok(headers)
    }

    fn build_request_body(&self, request: &LlmRequest, stream: bool) -> Result<serde_json::Value> {
        let body = OpenAiRequest {
            model: request.model.clone(),
            messages: request.messages.iter().flat_map(to_openai_messages).collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stop: (!request.stop_sequences.is_empty()).then(|| request.stop_sequences.clone()),
            stream,
            stream_options: stream.then_some(StreamOptions { include_usage: true }),
            tools: (!request.tools.is_empty())
                .then(|| request.tools.iter().map(OpenAiTool::from).collect()),
            // Tool calls are issued by the caller; tools are sent so the history is valid
            tool_choice: (!request.tools.is_empty()).then_some("none"),
        };
        Ok(serde_json::to_value(body)?)
    }

    fn parse_response(&self, body: &str) -> Result<ParsedCompletion> {
        let response: OpenAiResponse = serde_json::from_str(body)?;
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_default();
        Ok(ParsedCompletion {
            content,
            usage: response.usage.map(TokenUsage::from).unwrap_or_default(),
        })
    }

    fn parse_stream_event(&self, data: &str) -> Result<Vec<StreamEvent>> {
        let chunk: OpenAiStreamChunk = serde_json::from_str(data)?;
        let mut events = Vec::new();

        let text = chunk
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.delta.content)
            .filter(|text| !text.is_empty());
        if let Some(text) = text {
            events.push(StreamEvent::TextDelta(text));
        }

        if let Some(usage) = chunk.usage {
            events.push(StreamEvent::Usage(usage.into()));
        }

        Ok(events)
    }

    fn models(&self) -> &'static [&'static str] {
        MODELS
    }

    fn get_cost_per_token(&self, model: &str) -> Option<f64> {
        match model {
            "gpt-3.5-turbo" => Some(0.0005 / 1000.0), // $0.0005 per 1K tokens
            "gpt-3.5-turbo-16k" => Some(0.003 / 1000.0),
            "gpt-4" => Some(0.03 / 1000.0),
            "gpt-4-32k" => Some(0.06 / 1000.0),
            "gpt-4-turbo" => Some(0.01 / 1000.0),
            _ => None,
        }
    }

    fn get_max_context_length(&self, model: &str) -> Option<usize> {
        match model {
            "gpt-3.5-turbo" => Some(4096),
            "gpt-3.5-turbo-16k" => Some(16384),
            "gpt-4" => Some(8192),
            "gpt-4-32k" => Some(32768),
            "gpt-4-turbo" => Some(128000),
            _ => None,
        }
    }
}

/// OpenAI role name for a message role
fn role_name(role: &MessageRole) -> String {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
    .to_string()
}

/// Translate one message into OpenAI messages.
///
/// Tool calls become `tool_calls` on the assistant message, and each tool result becomes a
/// `tool` message with its `tool_call_id`. A tool-role message without a tool result part
/// has no call to answer, so it is sent as user text.
fn to_openai_messages(message: &Message) -> Vec<OpenAiMessage> {
    let parts = match &message.content {
        MessageContent::Parts(parts) => parts.as_slice(),
        MessageContent::Text(_) => &[],
    };

    match message.role {
        MessageRole::Tool => {
            let results: Vec<OpenAiMessage> = parts
                .iter()
                .filter_map(|part| match part {
                    MessagePart::ToolResult { tool_call_id, content } => Some(OpenAiMessage {
                        tool_call_id: Some(tool_call_id.clone()),
                        ..OpenAiMessage::text("tool", content.clone())
                    }),
                    _ => None,
                })
                .collect();
            if results.is_empty() {
                vec![OpenAiMessage::text("user", message_text(message))]
            } else {
                results
            }
        }
        MessageRole::Assistant => {
            let calls: Vec<OpenAiToolCall> = parts
                .iter()
                .filter_map(|part| match part {
                    MessagePart::ToolCall { id, name, arguments } => Some(OpenAiToolCall {
                        id: id.clone(),
                        kind: "function",
                        function: OpenAiFunctionCall {
                            name: name.clone(),
                            arguments: arguments.to_string(),
                        },
                    }),
                    _ => None,
                })
                .collect();
            let text = message_text(message);
            if calls.is_empty() {
                vec![OpenAiMessage::text("assistant", text)]
            } else {
                vec![OpenAiMessage {
                    content: (!text.is_empty()).then_some(text),
                    tool_calls: Some(calls),
                    ..OpenAiMessage::text("assistant", String::new())
                }]
            }
        }
        MessageRole::System | MessageRole::User => {
            vec![OpenAiMessage::text(&role_name(&message.role), message_text(message))]
        }
    }
}

/// OpenAI API request format
#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

/// Tool definition in a request
#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunction,
}

/// Function described by a tool definition
#[derive(Debug, Serialize)]
struct OpenAiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl From<&ToolDefinition> for OpenAiTool {
    fn from(tool: &ToolDefinition) -> Self {
        OpenAiTool {
            kind: "function",
            function: OpenAiFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        }
    }
}

/// Streaming options; asks for token usage in the final chunk
#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// OpenAI message format
#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OpenAiMessage {
    /// A plain text message
    fn text(role: &str, content: String) -> Self {
        OpenAiMessage {
            role: role.to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Tool call on an assistant message
#[derive(Debug, Serialize)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionCall,
}

/// Function name and JSON-encoded arguments of a tool call
#[derive(Debug, Serialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

/// OpenAI API response format
#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

/// OpenAI choice format
#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

/// Assistant message in a response; content is null for pure tool calls
#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
}

/// OpenAI usage format
#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl From<OpenAiUsage> for TokenUsage {
    fn from(usage: OpenAiUsage) -> Self {
        TokenUsage {
            prompt_tokens: Some(usage.prompt_tokens),
            completion_tokens: Some(usage.completion_tokens),
        }
    }
}

/// OpenAI streaming chunk format
#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
    usage: Option<OpenAiUsage>,
}

/// OpenAI streaming choice format
#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    #[serde(default)]
    delta: OpenAiDelta,
}

/// Incremental message content in a streaming chunk
#[derive(Debug, Default, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use axiom_ai_core::{AxiomError, Message};
    use futures::StreamExt;
    use reqwest::header::AUTHORIZATION;
    use reqwest::StatusCode;

    use crate::gateway::{LlmRequest, ToolDefinition};
    use crate::providers::testing::FakeTransport;
    use crate::providers::{LlmProvider, ProviderConfig};

    use super::OpenAIProvider;

    const BASE: &str = "https://api.openai.com/v1";

    fn provider(transport: &FakeTransport) -> OpenAIProvider {
        let config = ProviderConfig::new().with_api_key("sk-test");
        OpenAIProvider::with_transport(config, std::sync::Arc::new(transport.clone()))
    }

    fn request() -> LlmRequest {
        LlmRequest::new(
            vec![Message::system("Be brief."), Message::user("Hello")],
            "gpt-4",
        )
        .with_temperature(0.2)
        .with_max_tokens(64)
    }

    #[tokio::test]
    async fn request_has_bearer_auth_path_and_body() {
        let transport = FakeTransport::new();
        transport.respond(
            StatusCode::OK,
            r#"{"choices":[{"message":{"role":"assistant","content":"Hi"}}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#,
        );

        provider(&transport).generate(request()).await.unwrap();

        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        let sent = &requests[0];
        assert_eq!(sent.url, format!("{}/chat/completions", BASE));
        assert_eq!(sent.headers.get(AUTHORIZATION).unwrap(), "Bearer sk-test");
        assert_eq!(sent.body["model"], "gpt-4");
        assert_eq!(sent.body["stream"], false);
        assert_eq!(sent.body["max_tokens"], 64);
        assert_eq!(sent.body["messages"][0]["role"], "system");
        assert_eq!(sent.body["messages"][0]["content"], "Be brief.");
        assert_eq!(sent.body["messages"][1]["role"], "user");
        assert_eq!(sent.body["messages"][1]["content"], "Hello");
    }

    #[tokio::test]
    async fn generate_parses_content_usage_and_cost() {
        let transport = FakeTransport::new();
        transport.respond(
            StatusCode::OK,
            r#"{"choices":[{"message":{"role":"assistant","content":"Hello there"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
        );

        let response = provider(&transport).generate(request()).await.unwrap();

        assert_eq!(response.content, "Hello there");
        assert_eq!(response.provider, "openai");
        assert_eq!(response.tokens_used, Some(15));
        let expected_cost = 0.03 / 1000.0 * 15.0;
        assert!((response.cost.unwrap() - expected_cost).abs() < 1e-12);
    }

    #[tokio::test]
    async fn stream_yields_ordered_text_chunks_and_final_chunk() {
        let transport = FakeTransport::new();
        // Split mid-event to exercise the SSE buffering
        transport.respond_chunks(
            StatusCode::OK,
            &[
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"con",
                "tent\":\"lo\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
                "data: [DONE]\n\n",
            ],
        );

        let stream = provider(&transport).generate_stream(request()).await.unwrap();
        let chunks = stream.collect_all().await.unwrap();

        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).filter(|c| !c.is_empty()).collect();
        assert_eq!(texts, vec!["Hel", "lo"]);
        let last = chunks.last().unwrap();
        assert!(last.is_final);
        assert!(chunks[..chunks.len() - 1].iter().all(|c| !c.is_final));
        assert_eq!(last.metadata.as_ref().unwrap()["tokens_used"], 5);

        let sent = &transport.requests()[0];
        assert_eq!(sent.body["stream"], true);
        assert_eq!(sent.body["stream_options"]["include_usage"], true);
    }

    #[tokio::test]
    async fn non_success_status_maps_through_handle_provider_error() {
        let transport = FakeTransport::new();
        transport.respond(StatusCode::UNAUTHORIZED, r#"{"error":"bad key"}"#);

        let err = provider(&transport).generate(request()).await.unwrap_err();
        assert!(matches!(err, AxiomError::Authentication(_)), "got {:?}", err);

        transport.respond(StatusCode::UNAUTHORIZED, r#"{"error":"bad key"}"#);
        let err = provider(&transport).generate_stream(request()).await.err().unwrap();
        assert!(matches!(err, AxiomError::Authentication(_)), "got {:?}", err);
    }

    #[tokio::test]
    async fn truncated_stream_is_an_error() {
        let transport = FakeTransport::new();
        transport.respond_chunks(
            StatusCode::OK,
            &["data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"],
        );

        let stream = provider(&transport).generate_stream(request()).await.unwrap();
        let results: Vec<_> = stream.collect::<Vec<_>>().await;
        assert!(results[0].is_ok());
        assert!(results.last().unwrap().is_err());
    }

    #[tokio::test]
    async fn unencodable_api_key_is_a_config_error_not_a_panic() {
        let transport = FakeTransport::new();
        let config = ProviderConfig::new().with_api_key("bad\nkey");
        let provider = OpenAIProvider::with_transport(config, std::sync::Arc::new(transport.clone()));

        let err = provider.generate(request()).await.unwrap_err();
        assert!(matches!(err, AxiomError::Config(_)), "got {:?}", err);
        assert!(transport.requests().is_empty());
    }

    fn tool_request(model: &str) -> LlmRequest {
        let tool = ToolDefinition {
            name: "calculator".to_string(),
            description: "Evaluates arithmetic".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            }),
        };
        LlmRequest::new(
            vec![
                Message::user("What is 2+2?"),
                Message::tool_call("call_1", "calculator", serde_json::json!({"expression": "2+2"})),
                Message::tool("call_1", "4"),
            ],
            model,
        )
        .with_tools(vec![tool])
    }

    #[tokio::test]
    async fn tool_call_and_result_are_sent_as_tool_messages() {
        let transport = FakeTransport::new();
        transport.respond(
            StatusCode::OK,
            r#"{"choices":[{"message":{"role":"assistant","content":"4"}}]}"#,
        );

        provider(&transport).generate(tool_request("gpt-4")).await.unwrap();

        let body = &transport.requests()[0].body;
        let call = &body["messages"][1];
        assert_eq!(call["role"], "assistant");
        assert!(call.get("content").is_none());
        assert_eq!(call["tool_calls"][0]["id"], "call_1");
        assert_eq!(call["tool_calls"][0]["type"], "function");
        assert_eq!(call["tool_calls"][0]["function"]["name"], "calculator");
        assert_eq!(call["tool_calls"][0]["function"]["arguments"], r#"{"expression":"2+2"}"#);
        let result = &body["messages"][2];
        assert_eq!(result["role"], "tool");
        assert_eq!(result["tool_call_id"], "call_1");
        assert_eq!(result["content"], "4");
        assert_eq!(body["tools"][0]["function"]["name"], "calculator");
        assert_eq!(body["tools"][0]["function"]["parameters"]["required"][0], "expression");
        assert_eq!(body["tool_choice"], "none");
    }

    #[tokio::test]
    async fn request_without_tools_omits_tool_fields() {
        let transport = FakeTransport::new();
        transport.respond(StatusCode::OK, r#"{"choices":[{"message":{"content":"Hi"}}]}"#);

        provider(&transport).generate(request()).await.unwrap();

        let body = &transport.requests()[0].body;
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }
}
