//! One agent turn: plan, run each step, and record what happened.
//!
//! [`run_turn`] is the only code that dispatches plan steps. For each step it
//! checks safety, dispatches (tool call, memory read or write, or LLM response),
//! and appends the step's result messages to the conversation before the next
//! step runs. The turn stops when a response is generated; if the plan runs out
//! first, the turn generates one.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axiom_ai_core::{Memory, MemoryItem, MemoryType, Message, Result, Tool, ToolResult};
use axiom_ai_llm::{LlmGateway, LlmRequest, ToolDefinition};
use serde_json::Value;

use crate::agent::{AgentConfig, PlanningRequest, ToolInfo};
use crate::planner::{ExecutionPlan, PlanAction, PlanStep, Planner};
use crate::safety::{SafetyGuard, SafetyResult};

/// Text returned to the user in place of a generated response that failed the safety check
const BLOCKED_RESPONSE: &str = "I can't provide that response.";

/// Collaborators a turn needs from its owning agent
pub struct TurnDeps<'a> {
    /// LLM gateway used to generate responses
    pub llm: &'a LlmGateway,
    /// Memory used for context retrieval and memory steps
    pub memory: &'a mut dyn Memory,
    /// Planner that produces the plan for the turn
    pub planner: &'a dyn Planner,
    /// Safety guard checked before tool calls and on generated responses
    pub safety: &'a dyn SafetyGuard,
    /// Tools available to the agent, keyed by name
    pub tools: &'a HashMap<String, Arc<dyn Tool>>,
    /// Agent configuration (limits, models, timeouts)
    pub config: &'a AgentConfig,
}

/// Structured trace of one turn
#[derive(Debug, Clone)]
pub struct TurnOutcome {
    /// Plan produced by the planner for this turn
    pub plan: ExecutionPlan,
    /// One record per plan step that was reached, in plan order
    pub steps: Vec<StepRecord>,
    /// Tool calls that were executed in this turn, including timed-out ones
    pub tool_calls: Vec<ToolCallRecord>,
    /// Every safety verdict rendered during the turn, in order
    pub safety_verdicts: Vec<SafetyVerdict>,
    /// Messages appended to the conversation during the turn, in order
    pub messages: Vec<Message>,
    /// Why the turn stopped
    pub stop_reason: StopReason,
    /// Wall-clock duration of the turn in milliseconds
    pub execution_time_ms: u64,
}

/// Record of one executed tool call
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Tool name
    pub tool_name: String,
    /// Arguments passed to the tool
    pub arguments: HashMap<String, Value>,
    /// Result of the tool call
    pub result: ToolResult,
    /// Timestamp of the call
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// What happened to one plan step
#[derive(Debug, Clone)]
pub struct StepRecord {
    /// Identifier of the plan step
    pub step_id: String,
    /// Disposition of the step
    pub status: StepStatus,
}

/// Disposition of a plan step
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepStatus {
    /// The step ran and produced its messages
    Completed,
    /// A tool call was not run: unknown tool or invalid arguments
    Rejected {
        /// Why the call was rejected
        reason: String,
    },
    /// A tool call was not run, or a generated response was replaced, by the safety guard
    Blocked {
        /// Why the step was blocked
        reason: String,
    },
    /// A tool call was not run because `max_tool_calls` was already reached
    LimitReached,
}

/// A safety decision and what it applied to
#[derive(Debug, Clone)]
pub struct SafetyVerdict {
    /// What was checked, such as `tool:calculator` or `response`
    pub subject: String,
    /// The guard's decision
    pub result: SafetyResult,
}

/// Why a turn stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The plan ran out without a response step, and the turn answered with a fallback response
    PlanExhausted,
    /// A response was generated and the turn ended
    ResponseGenerated,
    /// The generated response failed the safety check and was replaced
    ResponseBlocked,
}

/// Step id recorded when a plan has no response step and the turn answers anyway
const FALLBACK_STEP_ID: &str = "fallback_response";

/// Run one turn for `incoming`.
///
/// `incoming` is appended to `conversation`, the planner builds a plan from it, and
/// the plan's steps run in order. Each step's result messages are appended to
/// `conversation` before the next step runs, so later steps and the final LLM call
/// see tool outputs and retrieved memories.
///
/// Every turn that succeeds ends with an assistant message. If the plan runs out
/// without a response step, the turn generates one from the conversation.
///
/// Errors are returned only when a collaborator fails (planner, memory, LLM gateway,
/// safety guard). Tool failures, timeouts and rejections are recorded in the outcome.
/// On error, `conversation` is restored to its length before the call and the
/// incoming query is removed from memory, so the same message can be retried cleanly.
/// Memory writes made by plan steps before the failure are not undone.
pub async fn run_turn(
    mut deps: TurnDeps<'_>,
    conversation: &mut Vec<Message>,
    incoming: Message,
) -> Result<TurnOutcome> {
    let started = Instant::now();
    let query = incoming.text_content().unwrap_or_default().to_string();
    let conversation_len = conversation.len();

    let stored = MemoryItem::new(query.clone(), MemoryType::ShortTerm, 0.5);
    let stored_id = stored.id;
    deps.memory.store(stored).await?;
    conversation.push(incoming);

    match run_plan(&mut deps, conversation, &query, started).await {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            conversation.truncate(conversation_len);
            // Best effort: the error being returned is the one that matters to the caller.
            let _ = deps.memory.delete(stored_id).await;
            Err(err)
        }
    }
}

/// Plan the turn and run its steps; the body of [`run_turn`] after the incoming message is recorded
async fn run_plan(
    deps: &mut TurnDeps<'_>,
    conversation: &mut Vec<Message>,
    query: &str,
    started: Instant,
) -> Result<TurnOutcome> {
    let context = deps.memory.retrieve(query, Some(10)).await?;
    let plan = deps
        .planner
        .create_plan(PlanningRequest {
            conversation: conversation.clone(),
            context,
            available_tools: tool_infos(deps.tools),
            config: deps.config.clone(),
        })
        .await?;

    let mut steps = Vec::with_capacity(plan.steps.len());
    let mut tool_calls = Vec::new();
    let mut safety_verdicts = Vec::new();
    let mut messages = Vec::new();
    let mut stop_reason = StopReason::PlanExhausted;

    for step in &plan.steps {
        let output = dispatch(deps, conversation.as_slice(), tool_calls.len(), step).await?;

        conversation.extend(output.messages.iter().cloned());
        messages.extend(output.messages);
        steps.push(StepRecord {
            step_id: step.id.clone(),
            status: output.status,
        });
        safety_verdicts.extend(output.verdicts);
        tool_calls.extend(output.tool_call);

        if let Some(reason) = output.stop {
            stop_reason = reason;
            break;
        }
    }

    if stop_reason == StopReason::PlanExhausted {
        // No response step ran, so answer from the conversation as it now stands
        let output = generate_response(deps, conversation.as_slice()).await?;
        conversation.extend(output.messages.iter().cloned());
        messages.extend(output.messages);
        steps.push(StepRecord {
            step_id: FALLBACK_STEP_ID.to_string(),
            status: output.status.clone(),
        });
        safety_verdicts.extend(output.verdicts);
        if output.status != StepStatus::Completed {
            stop_reason = StopReason::ResponseBlocked;
        }
    }

    Ok(TurnOutcome {
        plan,
        steps,
        tool_calls,
        safety_verdicts,
        messages,
        stop_reason,
        execution_time_ms: started.elapsed().as_millis() as u64,
    })
}

/// Result of dispatching one step, before it is folded into the turn
struct StepOutput {
    /// Disposition of the step
    status: StepStatus,
    /// Messages to append to the conversation
    messages: Vec<Message>,
    /// Safety verdicts rendered while dispatching the step
    verdicts: Vec<SafetyVerdict>,
    /// Record of the tool call, when a tool was executed
    tool_call: Option<ToolCallRecord>,
    /// Set when the step ends the turn
    stop: Option<StopReason>,
}

impl StepOutput {
    /// Create an output with no verdicts, tool call or stop condition
    fn new(status: StepStatus, messages: Vec<Message>) -> Self {
        Self {
            status,
            messages,
            verdicts: Vec::new(),
            tool_call: None,
            stop: None,
        }
    }
}

/// Dispatch one plan step to the component that handles its action
async fn dispatch(
    deps: &mut TurnDeps<'_>,
    conversation: &[Message],
    executed_tool_calls: usize,
    step: &PlanStep,
) -> Result<StepOutput> {
    match &step.action {
        PlanAction::GenerateResponse { .. } => generate_response(deps, conversation).await,
        PlanAction::CallTool { tool_name, arguments } => {
            call_tool(deps, executed_tool_calls, tool_name, arguments).await
        }
        PlanAction::UpdateMemory { content, memory_type } => {
            update_memory(&mut *deps.memory, content, memory_type.clone()).await
        }
        PlanAction::RetrieveMemory { query, limit } => {
            retrieve_memory(&mut *deps.memory, query, *limit).await
        }
    }
}

/// Validate, safety-check, and execute a tool call under the turn's limits
async fn call_tool(
    deps: &TurnDeps<'_>,
    executed_tool_calls: usize,
    tool_name: &str,
    arguments: &HashMap<String, Value>,
) -> Result<StepOutput> {
    let Some(tool) = deps.tools.get(tool_name) else {
        let reason = format!("unknown tool '{tool_name}'");
        return Ok(StepOutput::new(
            StepStatus::Rejected { reason },
            vec![Message::system(format!("Tool '{tool_name}' does not exist and was not run."))],
        ));
    };

    if let Err(err) = tool.validate_arguments(arguments) {
        let reason = err.user_message();
        return Ok(StepOutput::new(
            StepStatus::Rejected { reason: reason.clone() },
            vec![Message::system(format!("Tool '{tool_name}' was not run: {reason}"))],
        ));
    }

    let limit = deps.config.max_tool_calls as usize;
    if executed_tool_calls >= limit {
        return Ok(StepOutput::new(
            StepStatus::LimitReached,
            vec![Message::system(format!(
                "Tool call limit of {limit} reached; '{tool_name}' was not run."
            ))],
        ));
    }

    let mut verdicts = Vec::new();
    if deps.config.enable_safety_checks {
        let result = deps.safety.check_tool_call(tool_name, arguments).await?;
        let blocked = (!result.is_safe).then(|| result.reason.clone());
        verdicts.push(SafetyVerdict {
            subject: format!("tool:{tool_name}"),
            result,
        });
        if let Some(reason) = blocked {
            let mut output = StepOutput::new(
                StepStatus::Blocked { reason: reason.clone() },
                vec![Message::system(format!(
                    "Tool '{tool_name}' was blocked by the safety check and not run: {reason}"
                ))],
            );
            output.verdicts = verdicts;
            return Ok(output);
        }
    }

    let started = Instant::now();
    let timeout_secs = deps.config.tool_call_timeout;
    let result = match tokio::time::timeout(
        Duration::from_secs(timeout_secs),
        tool.execute(arguments.clone()),
    )
    .await
    {
        Ok(Ok(result)) => result,
        Ok(Err(err)) => ToolResult::error(err.user_message()),
        Err(_elapsed) => ToolResult::error(format!("timed out after {timeout_secs}s")),
    };

    let messages = tool_exchange_messages(tool_name, arguments, &result);
    let record = ToolCallRecord {
        tool_name: tool_name.to_string(),
        arguments: arguments.clone(),
        result,
        timestamp: chrono::Utc::now(),
        execution_time_ms: started.elapsed().as_millis() as u64,
    };

    let mut output = StepOutput::new(StepStatus::Completed, messages);
    output.verdicts = verdicts;
    output.tool_call = Some(record);
    Ok(output)
}

/// Store a memory item
async fn update_memory(
    memory: &mut dyn Memory,
    content: &str,
    memory_type: MemoryType,
) -> Result<StepOutput> {
    memory.store(MemoryItem::new(content, memory_type, 0.5)).await?;
    Ok(StepOutput::new(StepStatus::Completed, Vec::new()))
}

/// Retrieve memories matching a query and expose them to the model
async fn retrieve_memory(memory: &mut dyn Memory, query: &str, limit: usize) -> Result<StepOutput> {
    let items = memory.retrieve(query, Some(limit)).await?;
    let messages = if items.is_empty() {
        Vec::new()
    } else {
        let text = items
            .iter()
            .map(|item| item.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        vec![Message::system(format!("Relevant memory:\n{text}"))]
    };
    Ok(StepOutput::new(StepStatus::Completed, messages))
}

/// Generate a response from the conversation so far, then safety-check it
async fn generate_response(deps: &TurnDeps<'_>, conversation: &[Message]) -> Result<StepOutput> {
    let request = LlmRequest::new(conversation.to_vec(), deps.config.execution_model.clone())
        .with_temperature(deps.config.execution_temperature)
        .with_tools(tool_definitions(deps.tools));
    let response = deps.llm.generate(request).await?;

    let mut verdicts = Vec::new();
    if deps.config.enable_safety_checks {
        let result = deps.safety.check_response(&response.content).await?;
        let blocked = (!result.is_safe).then(|| result.reason.clone());
        verdicts.push(SafetyVerdict {
            subject: "response".to_string(),
            result,
        });
        if let Some(reason) = blocked {
            let mut output = StepOutput::new(
                StepStatus::Blocked { reason },
                vec![Message::assistant(BLOCKED_RESPONSE)],
            );
            output.verdicts = verdicts;
            output.stop = Some(StopReason::ResponseBlocked);
            return Ok(output);
        }
    }

    let mut output = StepOutput::new(
        StepStatus::Completed,
        vec![Message::assistant(response.content)],
    );
    output.verdicts = verdicts;
    output.stop = Some(StopReason::ResponseGenerated);
    Ok(output)
}

/// Build the messages that report an executed tool call to the model.
///
/// Providers only accept a tool result that answers a tool call earlier in the
/// conversation. The planner, not the model, issued this call, so the turn records
/// both halves: an assistant tool call and the tool result with the same id.
fn tool_exchange_messages(
    tool_name: &str,
    arguments: &HashMap<String, Value>,
    result: &ToolResult,
) -> Vec<Message> {
    let id = format!("call_{}", uuid::Uuid::new_v4().simple());
    let content = if result.success {
        result.content.clone()
    } else {
        let error = result.error.as_deref().unwrap_or("unknown error");
        format!("Error: {error}")
    };
    let arguments = Value::Object(arguments.clone().into_iter().collect());
    vec![
        Message::tool_call(id.clone(), tool_name, arguments),
        Message::tool(id, content),
    ]
}

/// Describe the agent's tools for the LLM, sorted by name
fn tool_definitions(tools: &HashMap<String, Arc<dyn Tool>>) -> Vec<ToolDefinition> {
    let mut definitions: Vec<ToolDefinition> = tools
        .values()
        .map(|tool| ToolDefinition::from_tool(tool.as_ref()))
        .collect();
    definitions.sort_by(|a, b| a.name.cmp(&b.name));
    definitions
}

/// Describe the agent's tools for the planner, sorted by name
fn tool_infos(tools: &HashMap<String, Arc<dyn Tool>>) -> Vec<ToolInfo> {
    let mut infos: Vec<ToolInfo> = tools
        .values()
        .map(|tool| ToolInfo {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.parameters(),
        })
        .collect();
    infos.sort_by(|a, b| a.name.cmp(&b.name));
    infos
}
