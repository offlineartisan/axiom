//! Anthropic Claude provider implementation (messages API)

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

use axiom_ai_core::{AxiomError, Message, MessageContent, MessagePart, MessageRole, Result};

use crate::gateway::{LlmRequest, ToolDefinition};

use super::http::{message_text, Dialect, HttpProvider, ParsedCompletion, StreamEvent, TokenUsage};

/// Anthropic provider: [`HttpProvider`] speaking the Anthropic messages dialect
pub type AnthropicProvider = HttpProvider<AnthropicDialect>;

/// Wire format for the Anthropic messages API
#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicDialect;

/// API version sent with every request
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// `max_tokens` is required by the API; used when the request does not set one
const DEFAULT_MAX_TOKENS: u32 = 1000;

const MODELS: &[&str] = &[
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
];

impl Dialect for AnthropicDialect {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn description(&self) -> &'static str {
        "Anthropic Claude models provider"
    }

    fn default_base_url(&self) -> &'static str {
        "https://api.anthropic.com/v1"
    }

    fn endpoint_path(&self) -> &'static str {
        "/messages"
    }

    fn auth_headers(&self, api_key: Option<&str>) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        if let Some(key) = api_key {
            let value = HeaderValue::from_str(key).map_err(|e| {
                AxiomError::Config(format!("Anthropic API key is not a valid header value: {}", e))
            })?;
            headers.insert(HeaderName::from_static("x-api-key"), value);
        }
        Ok(headers)
    }

    fn build_request_body(&self, request: &LlmRequest, stream: bool) -> Result<serde_json::Value> {
        let (system, messages) = split_system(&request.messages);

        let body = AnthropicRequest {
            model: request.model.clone(),
            max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages,
            system: (!system.is_empty()).then_some(system),
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: (!request.stop_sequences.is_empty()).then(|| request.stop_sequences.clone()),
            stream,
            tools: (!request.tools.is_empty())
                .then(|| request.tools.iter().map(AnthropicTool::from).collect()),
            // Tool calls are issued by the caller; tools are sent because the API rejects
            // tool_use/tool_result blocks in a request that defines no tools
            tool_choice: (!request.tools.is_empty()).then(|| serde_json::json!({"type": "none"})),
        };
        Ok(serde_json::to_value(body)?)
    }

    fn parse_response(&self, body: &str) -> Result<ParsedCompletion> {
        let response: AnthropicResponse = serde_json::from_str(body)?;
        let content = response
            .content
            .iter()
            .filter_map(|block| block.text.as_deref())
            .collect::<String>();
        let usage = response
            .usage
            .map(|usage| TokenUsage {
                prompt_tokens: Some(usage.input_tokens),
                completion_tokens: Some(usage.output_tokens),
            })
            .unwrap_or_default();
        Ok(ParsedCompletion { content, usage })
    }

    fn parse_stream_event(&self, data: &str) -> Result<Vec<StreamEvent>> {
        let event: AnthropicStreamEvent = serde_json::from_str(data)?;
        let mut events = Vec::new();

        match event.kind.as_str() {
            "content_block_delta" => {
                let text = event.delta.and_then(|delta| delta.text).filter(|text| !text.is_empty());
                if let Some(text) = text {
                    events.push(StreamEvent::TextDelta(text));
                }
            }
            "message_start" => {
                if let Some(usage) = event.message.and_then(|message| message.usage) {
                    events.push(StreamEvent::Usage(usage.into()));
                }
            }
            "message_delta" => {
                if let Some(usage) = event.usage {
                    events.push(StreamEvent::Usage(usage.into()));
                }
            }
            "message_stop" => events.push(StreamEvent::Done),
            "error" => {
                let message = event
                    .error
                    .map(|error| error.message)
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(AxiomError::LlmProvider(format!("Anthropic stream error: {}", message)));
            }
            // ping, content_block_start/stop and similar carry nothing we need
            _ => {}
        }

        Ok(events)
    }

    fn models(&self) -> &'static [&'static str] {
        MODELS
    }

    fn get_cost_per_token(&self, model: &str) -> Option<f64> {
        match model {
            "claude-3-opus-20240229" => Some(0.015 / 1000.0), // $0.015 per 1K tokens
            "claude-3-sonnet-20240229" => Some(0.003 / 1000.0),
            "claude-3-haiku-20240307" => Some(0.00025 / 1000.0),
            "claude-2.1" => Some(0.008 / 1000.0),
            "claude-2.0" => Some(0.008 / 1000.0),
            "claude-instant-1.2" => Some(0.0008 / 1000.0),
            _ => None,
        }
    }

    fn get_max_context_length(&self, model: &str) -> Option<usize> {
        match model {
            "claude-3-opus-20240229" => Some(200000),
            "claude-3-sonnet-20240229" => Some(200000),
            "claude-3-haiku-20240307" => Some(200000),
            "claude-2.1" => Some(100000),
            "claude-2.0" => Some(100000),
            "claude-instant-1.2" => Some(100000),
            _ => None,
        }
    }
}

/// Split messages into the top-level system prompt and the user/assistant turns
fn split_system(messages: &[Message]) -> (String, Vec<AnthropicMessage>) {
    let mut system = String::new();
    let mut conversation = Vec::new();

    for message in messages {
        match message.role {
            MessageRole::System => {
                system.push_str(&message_text(message));
                system.push('\n');
            }
            MessageRole::User => conversation.push(AnthropicMessage::text("user", message)),
            MessageRole::Assistant => conversation.push(assistant_message(message)),
            MessageRole::Tool => conversation.push(tool_message(message)),
        }
    }

    (system.trim().to_string(), conversation)
}

/// Content parts of a message; empty for plain text
fn parts(message: &Message) -> &[MessagePart] {
    match &message.content {
        MessageContent::Parts(parts) => parts.as_slice(),
        MessageContent::Text(_) => &[],
    }
}

/// Assistant turn; tool calls become `tool_use` blocks after any text
fn assistant_message(message: &Message) -> AnthropicMessage {
    let tool_uses: Vec<AnthropicBlock> = parts(message)
        .iter()
        .filter_map(|part| match part {
            MessagePart::ToolCall { id, name, arguments } => Some(AnthropicBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: arguments.clone(),
            }),
            _ => None,
        })
        .collect();
    if tool_uses.is_empty() {
        return AnthropicMessage::text("assistant", message);
    }

    let text = message_text(message);
    let blocks = (!text.is_empty())
        .then(|| AnthropicBlock::Text { text })
        .into_iter()
        .chain(tool_uses)
        .collect();
    AnthropicMessage {
        role: "assistant".to_string(),
        content: AnthropicContent::Blocks(blocks),
    }
}

/// The messages API has no tool role: tool results are `tool_result` blocks in a user turn.
/// A tool-role message without a tool result part has no call to answer, so it is sent as text.
fn tool_message(message: &Message) -> AnthropicMessage {
    let results: Vec<AnthropicBlock> = parts(message)
        .iter()
        .filter_map(|part| match part {
            MessagePart::ToolResult { tool_call_id, content } => Some(AnthropicBlock::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: content.clone(),
            }),
            _ => None,
        })
        .collect();
    if results.is_empty() {
        return AnthropicMessage::text("user", message);
    }
    AnthropicMessage {
        role: "user".to_string(),
        content: AnthropicContent::Blocks(results),
    }
}

/// Anthropic API request format
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

/// Tool definition in a request
#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

impl From<&ToolDefinition> for AnthropicTool {
    fn from(tool: &ToolDefinition) -> Self {
        AnthropicTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        }
    }
}

/// Anthropic message format
#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

impl AnthropicMessage {
    /// A turn carrying the message's text
    fn text(role: &str, message: &Message) -> Self {
        AnthropicMessage {
            role: role.to_string(),
            content: AnthropicContent::Text(message_text(message)),
        }
    }
}

/// Message content: a plain string, or content blocks when tools are involved
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

/// Content block in a request message
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String },
}

/// Anthropic API response format
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
}

/// Anthropic content block; only text blocks carry a `text` field
#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    text: Option<String>,
}

/// Anthropic usage format
#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Anthropic streaming event. Only the fields used by [`AnthropicDialect::parse_stream_event`] are read.
#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    kind: String,
    delta: Option<AnthropicStreamDelta>,
    message: Option<AnthropicStreamMessage>,
    usage: Option<AnthropicUsageDelta>,
    error: Option<AnthropicStreamError>,
}

/// `delta` of a `content_block_delta` event
#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    text: Option<String>,
}

/// `message` of a `message_start` event
#[derive(Debug, Deserialize)]
struct AnthropicStreamMessage {
    usage: Option<AnthropicUsageDelta>,
}

/// Usage in a stream event; counts are cumulative and may be partial
#[derive(Debug, Deserialize)]
struct AnthropicUsageDelta {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

impl From<AnthropicUsageDelta> for TokenUsage {
    fn from(usage: AnthropicUsageDelta) -> Self {
        TokenUsage {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
        }
    }
}

/// `error` of an `error` event
#[derive(Debug, Deserialize)]
struct AnthropicStreamError {
    message: String,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axiom_ai_core::{AxiomError, Message};
    use futures::StreamExt;
    use reqwest::header::{AUTHORIZATION};
    use reqwest::StatusCode;

    use crate::gateway::{LlmRequest, ToolDefinition};
    use crate::providers::testing::FakeTransport;
    use crate::providers::{LlmProvider, ProviderConfig};

    use super::AnthropicProvider;

    const BASE: &str = "https://api.anthropic.com/v1";
    const MODEL: &str = "claude-3-haiku-20240307";

    fn provider(transport: &FakeTransport) -> AnthropicProvider {
        let config = ProviderConfig::new().with_api_key("sk-ant-test");
        AnthropicProvider::with_transport(config, Arc::new(transport.clone()))
    }

    fn request() -> LlmRequest {
        LlmRequest::new(
            vec![
                Message::system("You are terse."),
                Message::system("Answer in English."),
                Message::user("Hello"),
            ],
            MODEL,
        )
        .with_max_tokens(128)
    }

    #[tokio::test]
    async fn request_uses_x_api_key_and_version_header_with_top_level_system() {
        let transport = FakeTransport::new();
        transport.respond(
            StatusCode::OK,
            r#"{"content":[{"type":"text","text":"Hi"}],"usage":{"input_tokens":4,"output_tokens":1}}"#,
        );

        provider(&transport).generate(request()).await.unwrap();

        let requests = transport.requests();
        let sent = &requests[0];
        assert_eq!(sent.url, format!("{}/messages", BASE));
        assert_eq!(sent.headers.get("x-api-key").unwrap(), "sk-ant-test");
        assert_eq!(sent.headers.get("anthropic-version").unwrap(), "2023-06-01");
        assert!(sent.headers.get(AUTHORIZATION).is_none());

        assert_eq!(sent.body["model"], MODEL);
        assert_eq!(sent.body["max_tokens"], 128);
        assert_eq!(sent.body["system"], "You are terse.\nAnswer in English.");
        assert_eq!(sent.body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(sent.body["messages"][0]["role"], "user");
        assert_eq!(sent.body["messages"][0]["content"], "Hello");
    }

    #[tokio::test]
    async fn generate_parses_content_usage_and_cost() {
        let transport = FakeTransport::new();
        transport.respond(
            StatusCode::OK,
            r#"{"content":[{"type":"text","text":"Hello"},{"type":"text","text":" world"}],"usage":{"input_tokens":10,"output_tokens":6}}"#,
        );

        let response = provider(&transport).generate(request()).await.unwrap();

        assert_eq!(response.content, "Hello world");
        assert_eq!(response.provider, "anthropic");
        assert_eq!(response.tokens_used, Some(16));
        let expected_cost = 0.00025 / 1000.0 * 16.0;
        assert!((response.cost.unwrap() - expected_cost).abs() < 1e-12);
    }

    #[tokio::test]
    async fn stream_yields_ordered_text_chunks_and_final_chunk() {
        let transport = FakeTransport::new();
        transport.respond_chunks(
            StatusCode::OK,
            &[
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel",
                "lo\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            ],
        );

        let stream = provider(&transport).generate_stream(request()).await.unwrap();
        let chunks = stream.collect_all().await.unwrap();

        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).filter(|c| !c.is_empty()).collect();
        assert_eq!(texts, vec!["Hello", " there"]);
        let last = chunks.last().unwrap();
        assert!(last.is_final);
        assert!(chunks[..chunks.len() - 1].iter().all(|c| !c.is_final));
        assert_eq!(last.metadata.as_ref().unwrap()["tokens_used"], 11);

        let sent = &transport.requests()[0];
        assert_eq!(sent.body["stream"], true);
    }

    #[tokio::test]
    async fn stream_error_event_surfaces_as_error() {
        let transport = FakeTransport::new();
        transport.respond_chunks(
            StatusCode::OK,
            &["data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"],
        );

        let stream = provider(&transport).generate_stream(request()).await.unwrap();
        let results: Vec<_> = stream.collect::<Vec<_>>().await;
        match results.last().unwrap() {
            Err(AxiomError::LlmProvider(msg)) => assert!(msg.contains("Overloaded")),
            other => panic!("expected provider error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn non_success_status_maps_through_handle_provider_error() {
        let transport = FakeTransport::new();
        transport.respond(StatusCode::UNAUTHORIZED, r#"{"type":"error"}"#);

        let err = provider(&transport).generate(request()).await.unwrap_err();
        assert!(matches!(err, AxiomError::Authentication(_)), "got {:?}", err);
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
    async fn tool_call_and_result_are_sent_as_content_blocks() {
        let transport = FakeTransport::new();
        transport.respond(StatusCode::OK, r#"{"content":[{"type":"text","text":"4"}]}"#);

        provider(&transport)
            .generate(tool_request("claude-3-haiku-20240307"))
            .await
            .unwrap();

        let body = &transport.requests()[0].body;
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["content"], "What is 2+2?");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(
            messages[1]["content"],
            serde_json::json!([{"type": "tool_use", "id": "call_1", "name": "calculator", "input": {"expression": "2+2"}}])
        );
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(
            messages[2]["content"],
            serde_json::json!([{"type": "tool_result", "tool_use_id": "call_1", "content": "4"}])
        );
        assert_eq!(body["tools"][0]["name"], "calculator");
        assert_eq!(body["tools"][0]["input_schema"]["required"][0], "expression");
        assert_eq!(body["tool_choice"], serde_json::json!({"type": "none"}));
    }
}
