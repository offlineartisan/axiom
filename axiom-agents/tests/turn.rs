//! Turn behaviour, exercised through the `Agent` interface with a scripted LLM,
//! a scripted planner and a small fake tool.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axiom_ai_agents::{
    Agent, AgentConfig, ExecutionPlan, PlanAction, PlanStep, PlanningRequest, Planner,
    SafetyGuard, SimpleSafetyGuard, StepStatus, StopReason, TurnOutcome,
};
use axiom_ai_core::{
    AxiomError, InMemoryMemory, Message, MessageContent, MessagePart, MessageRole, ParameterType,
    Result, StreamResponse, Tool, ToolParameter, ToolParameters, ToolResult,
};
use axiom_ai_llm::{LlmGateway, LlmProvider, LlmRequest, LlmResponse, RateLimitConfig, RetryConfig};
use serde_json::Value;

/// LLM provider that records every request and replies with scripted text
struct FakeLlm {
    requests: Arc<Mutex<Vec<LlmRequest>>>,
    replies: Mutex<VecDeque<String>>,
}

impl FakeLlm {
    fn new(replies: &[&str]) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            replies: Mutex::new(replies.iter().map(|r| r.to_string()).collect()),
        }
    }
}

#[async_trait]
impl LlmProvider for FakeLlm {
    fn name(&self) -> &str {
        "fake"
    }

    fn description(&self) -> &str {
        "Scripted provider for turn tests"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["fake-model".to_string()]
    }

    async fn generate(&self, request: LlmRequest) -> Result<LlmResponse> {
        self.requests.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| "default reply".to_string());
        Ok(LlmResponse::new(reply, "fake-model", "fake"))
    }

    async fn generate_stream(&self, _request: LlmRequest) -> Result<StreamResponse> {
        Err(AxiomError::LlmProvider("streaming not used in turn tests".to_string()))
    }

    fn get_cost_per_token(&self, _model: &str) -> Option<f64> {
        None
    }

    fn get_max_context_length(&self, _model: &str) -> Option<usize> {
        None
    }
}

/// Planner that always returns the same steps
struct ScriptedPlanner {
    steps: Vec<PlanStep>,
}

#[async_trait]
impl Planner for ScriptedPlanner {
    async fn create_plan(&self, _request: PlanningRequest) -> Result<ExecutionPlan> {
        Ok(ExecutionPlan {
            steps: self.steps.clone(),
            metadata: HashMap::new(),
            estimated_time_ms: 0,
        })
    }
}

/// Tool named `lookup` that requires a `query` argument, counts its runs,
/// and sleeps for `delay` before answering
struct LookupTool {
    runs: Arc<AtomicUsize>,
    delay: Duration,
}

#[async_trait]
impl Tool for LookupTool {
    fn name(&self) -> &str {
        "lookup"
    }

    fn description(&self) -> &str {
        "Look up a fact"
    }

    fn parameters(&self) -> ToolParameters {
        ToolParameters::new()
            .add_parameter(
                "query".to_string(),
                ToolParameter::new(ParameterType::String, "What to look up").required(),
            )
            .required("query")
    }

    async fn execute(&self, _arguments: HashMap<String, Value>) -> Result<ToolResult> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        Ok(ToolResult::success("MARKER-42 is the answer"))
    }
}

/// Build a plan step with an id derived from its position
fn step(index: usize, action: PlanAction) -> PlanStep {
    PlanStep {
        id: format!("step_{index}"),
        action,
        dependencies: Vec::new(),
        estimated_time_ms: 0,
        priority: 1,
    }
}

/// A `lookup` call with the given query
fn lookup(query: &str) -> PlanAction {
    let mut arguments = HashMap::new();
    arguments.insert("query".to_string(), Value::String(query.to_string()));
    PlanAction::CallTool {
        tool_name: "lookup".to_string(),
        arguments,
    }
}

/// A `lookup` call with no arguments, which fails validation
fn lookup_without_query() -> PlanAction {
    PlanAction::CallTool {
        tool_name: "lookup".to_string(),
        arguments: HashMap::new(),
    }
}

fn generate() -> PlanAction {
    PlanAction::GenerateResponse {
        prompt: String::new(),
    }
}

/// Gateway over one fake provider, with no retry delay and a rate limit that never trips
fn gateway(llm: FakeLlm) -> Arc<LlmGateway> {
    Arc::new(
        LlmGateway::new("fake")
            .add_provider("fake", Arc::new(llm))
            .with_retry_config(RetryConfig::new(0))
            .with_rate_limit_config(
                RateLimitConfig::new().with_default_limits(Some(10_000), None),
            ),
    )
}

fn config() -> AgentConfig {
    AgentConfig {
        execution_model: "fake-model".to_string(),
        ..AgentConfig::default()
    }
}

/// Agent wired to the fakes; `tool` is registered under the name `lookup`
fn agent(
    llm: FakeLlm,
    steps: Vec<PlanStep>,
    tool: LookupTool,
    safety: Box<dyn SafetyGuard>,
    config: AgentConfig,
) -> Agent {
    Agent::new(
        gateway(llm),
        Box::new(InMemoryMemory::new()),
        Box::new(ScriptedPlanner { steps }),
        safety,
        config,
    )
    .add_tool(Box::new(tool))
}

/// Text of every message in a recorded request
fn request_texts(request: &LlmRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter_map(|m| m.text_content().map(str::to_string))
        .collect()
}

#[tokio::test]
async fn tool_result_reaches_next_llm_request() {
    let llm = FakeLlm::new(&["The answer is 42"]);
    let requests = Arc::clone(&llm.requests);
    let runs = Arc::new(AtomicUsize::new(0));
    let steps = vec![step(0, lookup("answer")), step(1, generate())];
    let tool = LookupTool { runs: Arc::clone(&runs), delay: Duration::ZERO };
    let mut agent = agent(llm, steps, tool, Box::new(SimpleSafetyGuard::new()), config());

    let outcome = agent
        .process_message(Message::user("what is the answer?"))
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1, "one LLM call for the generate step");
    assert!(
        request_texts(&requests[0]).iter().any(|t| t.contains("MARKER-42")),
        "tool output must be in the request sent to the LLM"
    );

    // The result answers a tool call with the same id, so providers accept it
    let sent = &requests[0].messages;
    let call = sent.iter().position(|m| m.has_tool_calls()).expect("assistant tool call");
    let MessageContent::Parts(call_parts) = &sent[call].content else { panic!("call has parts") };
    let MessagePart::ToolCall { id: call_id, name, .. } = &call_parts[0] else { panic!("tool call part") };
    assert_eq!(sent[call].role, MessageRole::Assistant);
    assert_eq!(name, "lookup");
    let result = &sent[call + 1];
    assert_eq!(result.role, MessageRole::Tool);
    let MessageContent::Parts(result_parts) = &result.content else { panic!("result has parts") };
    let MessagePart::ToolResult { tool_call_id, .. } = &result_parts[0] else { panic!("tool result part") };
    assert_eq!(tool_call_id, call_id);
    assert_eq!(requests[0].tools.len(), 1, "tool definitions travel with the request");
    assert_eq!(requests[0].tools[0].name, "lookup");

    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(outcome.tool_calls.len(), 1);
    assert_eq!(outcome.stop_reason, StopReason::ResponseGenerated);
}

#[tokio::test]
async fn max_tool_calls_stops_extra_tool_calls() {
    let llm = FakeLlm::new(&["done"]);
    let runs = Arc::new(AtomicUsize::new(0));
    let steps = vec![
        step(0, lookup("first")),
        step(1, lookup("second")),
        step(2, lookup("third")),
        step(3, generate()),
    ];
    let tool = LookupTool { runs: Arc::clone(&runs), delay: Duration::ZERO };
    let config = AgentConfig {
        max_tool_calls: 1,
        ..config()
    };
    let mut agent = agent(llm, steps, tool, Box::new(SimpleSafetyGuard::new()), config);

    let outcome = agent.process_message(Message::user("look things up")).await.unwrap();

    assert_eq!(runs.load(Ordering::SeqCst), 1, "only one tool execution allowed");
    assert_eq!(outcome.tool_calls.len(), 1);
    assert_eq!(
        statuses(&outcome),
        vec![
            StepStatus::Completed,
            StepStatus::LimitReached,
            StepStatus::LimitReached,
            StepStatus::Completed,
        ]
    );
    assert_eq!(outcome.stop_reason, StopReason::ResponseGenerated);
}

#[tokio::test]
async fn tool_blocked_by_safety_is_not_executed_and_verdict_is_traced() {
    let llm = FakeLlm::new(&["ok"]);
    let runs = Arc::new(AtomicUsize::new(0));
    let steps = vec![step(0, lookup("anything")), step(1, generate())];
    let tool = LookupTool { runs: Arc::clone(&runs), delay: Duration::ZERO };
    let safety = SimpleSafetyGuard::new().block_tool("lookup");
    let mut agent = agent(llm, steps, tool, Box::new(safety), config());

    let outcome = agent.process_message(Message::user("find it")).await.unwrap();

    assert_eq!(runs.load(Ordering::SeqCst), 0, "blocked tool must not run");
    assert!(outcome.tool_calls.is_empty());
    let tool_verdict = outcome
        .safety_verdicts
        .iter()
        .find(|v| v.subject == "tool:lookup")
        .expect("trace should record the tool safety verdict");
    assert!(!tool_verdict.result.is_safe);
    assert!(matches!(outcome.steps[0].status, StepStatus::Blocked { .. }));
}

#[tokio::test]
async fn invalid_arguments_are_rejected_without_executing() {
    let llm = FakeLlm::new(&["ok"]);
    let runs = Arc::new(AtomicUsize::new(0));
    let steps = vec![step(0, lookup_without_query()), step(1, generate())];
    let tool = LookupTool { runs: Arc::clone(&runs), delay: Duration::ZERO };
    let mut agent = agent(llm, steps, tool, Box::new(SimpleSafetyGuard::new()), config());

    let outcome = agent.process_message(Message::user("find it")).await.unwrap();

    assert_eq!(runs.load(Ordering::SeqCst), 0, "invalid call must not run");
    assert!(outcome.tool_calls.is_empty());
    match &outcome.steps[0].status {
        StepStatus::Rejected { reason } => assert!(reason.contains("query"), "reason: {reason}"),
        other => panic!("expected Rejected, got {other:?}"),
    }
}

#[tokio::test]
async fn slow_tool_hits_tool_call_timeout() {
    let llm = FakeLlm::new(&["sorry, the lookup timed out"]);
    let requests = Arc::clone(&llm.requests);
    let runs = Arc::new(AtomicUsize::new(0));
    let steps = vec![step(0, lookup("slow")), step(1, generate())];
    let tool = LookupTool { runs: Arc::clone(&runs), delay: Duration::from_secs(60) };
    let config = AgentConfig {
        tool_call_timeout: 1,
        ..config()
    };
    let mut agent = agent(llm, steps, tool, Box::new(SimpleSafetyGuard::new()), config);

    let started = Instant::now();
    let outcome = agent.process_message(Message::user("look it up")).await.unwrap();

    assert!(started.elapsed() < Duration::from_secs(10), "turn must not wait for the slow tool");
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(outcome.tool_calls.len(), 1);
    let record = &outcome.tool_calls[0];
    assert!(!record.result.success);
    assert!(
        record.result.error.as_deref().unwrap_or("").contains("timed out"),
        "error: {:?}",
        record.result.error
    );
    let requests = requests.lock().unwrap();
    assert!(
        request_texts(&requests[0]).iter().any(|t| t.contains("timed out")),
        "the timeout must be visible to the LLM"
    );
}

#[tokio::test]
async fn unsafe_generated_response_is_replaced() {
    let llm = FakeLlm::new(&["here is how to hack the server"]);
    let runs = Arc::new(AtomicUsize::new(0));
    let steps = vec![step(0, generate())];
    let tool = LookupTool { runs: Arc::clone(&runs), delay: Duration::ZERO };
    let mut agent = agent(llm, steps, tool, Box::new(SimpleSafetyGuard::new()), config());

    let outcome = agent.process_message(Message::user("tell me")).await.unwrap();

    assert_eq!(outcome.stop_reason, StopReason::ResponseBlocked);
    assert!(outcome
        .safety_verdicts
        .iter()
        .any(|v| v.subject == "response" && !v.result.is_safe));
    let last = outcome.messages.last().and_then(|m| m.text_content()).unwrap_or("");
    assert!(!last.contains("hack"), "unsafe text must not reach the user: {last}");
}

fn statuses(outcome: &TurnOutcome) -> Vec<StepStatus> {
    outcome.steps.iter().map(|s| s.status.clone()).collect()
}
