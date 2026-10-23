//! Agent facade: owns the collaborators and conversation, and delegates each turn to [`crate::turn`]

use std::collections::HashMap;
use std::sync::Arc;

use axiom_ai_core::{Memory, Message, MessageRole, Result, StreamResponse, Tool};
use axiom_ai_llm::LlmGateway;

use crate::planner::{ExecutionPlan, Planner};
use crate::safety::SafetyGuard;
use crate::turn::{run_turn, ToolCallRecord, TurnDeps, TurnOutcome};

/// Main Agent that orchestrates planning, execution, and tool usage
pub struct Agent {
    /// LLM gateway for language model interactions
    llm_gateway: Arc<LlmGateway>,
    /// Memory system for context and conversation
    memory: Box<dyn Memory>,
    /// Tools available to plans, keyed by name
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Planner for creating execution plans
    planner: Box<dyn Planner>,
    /// Safety guard for tool calls and generated responses
    safety_guard: Box<dyn SafetyGuard>,
    /// Agent configuration
    config: AgentConfig,
    /// Current agent state
    state: AgentState,
}

/// Configuration for the agent
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of tool calls per turn
    pub max_tool_calls: u32,
    /// Tool call timeout in seconds
    pub tool_call_timeout: u64,
    /// Enable tool retries
    pub enable_tool_retries: bool,
    /// Maximum tool retries
    pub max_tool_retries: u32,
    /// Enable safety checks
    pub enable_safety_checks: bool,
    /// Safety check timeout in seconds
    pub safety_check_timeout: u64,
    /// Model to use for planning
    pub planning_model: String,
    /// Model to use for execution
    pub execution_model: String,
    /// Temperature for planning
    pub planning_temperature: f32,
    /// Temperature for execution
    pub execution_temperature: f32,
    /// Maximum context length
    pub max_context_length: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_calls: 10,
            tool_call_timeout: 30,
            enable_tool_retries: true,
            max_tool_retries: 3,
            enable_safety_checks: true,
            safety_check_timeout: 5,
            planning_model: "gpt-3.5-turbo".to_string(),
            execution_model: "gpt-3.5-turbo".to_string(),
            planning_temperature: 0.7,
            execution_temperature: 0.7,
            max_context_length: 4000,
        }
    }
}

/// Current state of the agent
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Current conversation
    pub conversation: Vec<Message>,
    /// Plan produced by the most recent turn
    pub current_plan: Option<ExecutionPlan>,
    /// Tool call history across turns
    pub tool_call_history: Vec<ToolCallRecord>,
    /// Agent metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Agent {
    /// Create a new agent
    pub fn new(
        llm_gateway: Arc<LlmGateway>,
        memory: Box<dyn Memory>,
        planner: Box<dyn Planner>,
        safety_guard: Box<dyn SafetyGuard>,
        config: AgentConfig,
    ) -> Self {
        Self {
            llm_gateway,
            memory,
            tools: HashMap::new(),
            planner,
            safety_guard,
            config,
            state: AgentState {
                conversation: Vec::new(),
                current_plan: None,
                tool_call_history: Vec::new(),
                metadata: HashMap::new(),
            },
        }
    }

    /// Add a tool to the agent, replacing any tool with the same name
    pub fn add_tool(mut self, tool: Box<dyn Tool>) -> Self {
        let tool: Arc<dyn Tool> = Arc::from(tool);
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    /// Process a user message and run one turn
    pub async fn process_message(&mut self, message: Message) -> Result<TurnOutcome> {
        let outcome = self.run(message).await?;
        self.state.current_plan = Some(outcome.plan.clone());
        self.state.tool_call_history.extend(outcome.tool_calls.iter().cloned());
        Ok(outcome)
    }

    /// Process a message and stream the assistant's reply.
    ///
    /// Tool results, memory notes and safety notices from the turn are not streamed; they
    /// stay in the outcome of [`process_message`](Self::process_message). The reply is the
    /// assistant text the turn produced, and only its last chunk is marked final.
    pub async fn process_message_stream(&mut self, message: Message) -> Result<StreamResponse> {
        let outcome = self.process_message(message).await?;

        let mut replies: Vec<String> = outcome
            .messages
            .iter()
            .filter(|msg| msg.role == MessageRole::Assistant)
            .filter_map(|msg| msg.text_content().map(str::to_string))
            .collect();
        if replies.is_empty() {
            replies.push(String::new());
        }

        let last = replies.len() - 1;
        let chunks: Vec<Result<axiom_ai_core::StreamChunk>> = replies
            .into_iter()
            .enumerate()
            .map(|(index, content)| {
                Ok(axiom_ai_core::StreamChunk {
                    content,
                    chunk_type: axiom_ai_core::ChunkType::Text,
                    metadata: None,
                    is_final: index == last,
                })
            })
            .collect();

        let stream = tokio_stream::iter(chunks);
        Ok(StreamResponse::new(stream))
    }

    /// Hand the agent's collaborators and conversation to the turn module
    async fn run(&mut self, message: Message) -> Result<TurnOutcome> {
        let deps = TurnDeps {
            llm: self.llm_gateway.as_ref(),
            memory: self.memory.as_mut(),
            planner: self.planner.as_ref(),
            safety: self.safety_guard.as_ref(),
            tools: &self.tools,
            config: &self.config,
        };
        run_turn(deps, &mut self.state.conversation, message).await
    }

    /// Get the current agent state
    pub fn get_state(&self) -> &AgentState {
        &self.state
    }

    /// Update agent configuration
    pub fn update_config(&mut self, config: AgentConfig) {
        self.config = config;
    }

    /// Clear conversation history
    pub async fn clear_conversation(&mut self) -> Result<()> {
        self.state.conversation.clear();
        self.state.tool_call_history.clear();
        self.memory.clear().await?;
        Ok(())
    }
}

/// Information about an available tool
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool parameters schema
    pub parameters: axiom_ai_core::ToolParameters,
}

/// Request for planning
#[derive(Debug, Clone)]
pub struct PlanningRequest {
    /// Current conversation
    pub conversation: Vec<Message>,
    /// Relevant context from memory
    pub context: Vec<axiom_ai_core::MemoryItem>,
    /// Available tools
    pub available_tools: Vec<ToolInfo>,
    /// Agent configuration
    pub config: AgentConfig,
}
