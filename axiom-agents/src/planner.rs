//! Planning system for agent execution

use async_trait::async_trait;
use std::collections::HashMap;

use axiom_ai_core::{Result, AxiomError, MemoryType, ParameterType};
// Types are defined in this module
use crate::agent::{PlanningRequest, ToolInfo};

/// Trait for planning agent execution
#[async_trait]
pub trait Planner: Send + Sync {
    /// Create an execution plan based on the request
    async fn create_plan(&self, request: PlanningRequest) -> Result<ExecutionPlan>;
}

/// An execution plan consisting of multiple steps
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Steps in the plan
    pub steps: Vec<PlanStep>,
    /// Plan metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Estimated execution time in milliseconds
    pub estimated_time_ms: u64,
}

/// A single step in an execution plan
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Step identifier
    pub id: String,
    /// Action to perform
    pub action: PlanAction,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Estimated execution time in milliseconds
    pub estimated_time_ms: u64,
    /// Priority (higher = more important)
    pub priority: u32,
}

/// Actions that can be performed in a plan step
#[derive(Debug, Clone)]
pub enum PlanAction {
    /// Generate a response using the LLM
    GenerateResponse { prompt: String },
    /// Call a tool
    CallTool { tool_name: String, arguments: HashMap<String, serde_json::Value> },
    /// Update memory
    UpdateMemory { content: String, memory_type: MemoryType },
    /// Retrieve memory
    RetrieveMemory { query: String, limit: usize },
}

/// Simple planner that creates basic execution plans
pub struct SimplePlanner {
    /// Maximum number of steps in a plan
    max_steps: usize,
    /// Whether to enable parallel execution
    enable_parallel: bool,
}

impl SimplePlanner {
    /// Create a new simple planner
    pub fn new(max_steps: usize) -> Self {
        Self {
            max_steps,
            enable_parallel: false,
        }
    }

    /// Enable parallel execution
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }
}

#[async_trait]
impl Planner for SimplePlanner {
    async fn create_plan(&self, request: PlanningRequest) -> Result<ExecutionPlan> {
        let mut steps = Vec::new();
        let mut step_counter = 0;

        // Analyze the conversation to determine what needs to be done
        let last_message = request.conversation.last()
            .and_then(|msg| msg.text_content())
            .unwrap_or("");

        // Check if we need to retrieve memory
        if self.should_retrieve_memory(last_message, &request.context) {
            steps.push(PlanStep {
                id: format!("retrieve_memory_{}", step_counter),
                action: PlanAction::RetrieveMemory {
                    query: last_message.to_string(),
                    limit: 5,
                },
                dependencies: Vec::new(),
                estimated_time_ms: 100,
                priority: 1,
            });
            step_counter += 1;
        }

        // Check if we need to call tools
        let tool_calls = self.identify_tool_calls(last_message, &request.available_tools);
        for tool_call in tool_calls {
            steps.push(PlanStep {
                id: format!("call_tool_{}", step_counter),
                action: PlanAction::CallTool {
                    tool_name: tool_call.tool_name,
                    arguments: tool_call.arguments,
                },
                dependencies: Vec::new(),
                estimated_time_ms: 1000,
                priority: 2,
            });
            step_counter += 1;
        }

        // Limit the number of steps, keeping room for the response
        steps.truncate(self.max_steps.saturating_sub(1));

        // Always generate a response, after everything else
        steps.push(PlanStep {
            id: format!("generate_response_{}", step_counter),
            action: PlanAction::GenerateResponse {
                prompt: last_message.to_string(),
            },
            dependencies: steps.iter().map(|step| step.id.clone()).collect(),
            estimated_time_ms: 2000,
            priority: 3,
        });

        // Calculate total estimated time
        let estimated_time = steps.iter().map(|s| s.estimated_time_ms).sum();

        Ok(ExecutionPlan {
            steps,
            metadata: HashMap::new(),
            estimated_time_ms: estimated_time,
        })
    }
}

impl SimplePlanner {
    /// Check if we should retrieve memory
    fn should_retrieve_memory(&self, message: &str, context: &[axiom_ai_core::MemoryItem]) -> bool {
        // Simple heuristic: retrieve memory if context is empty or message asks for information
        context.is_empty() || message.to_lowercase().contains("remember") || message.to_lowercase().contains("recall")
    }

    /// Identify tool calls needed based on the message
    fn identify_tool_calls(&self, message: &str, available_tools: &[ToolInfo]) -> Vec<ToolCall> {
        let mut tool_calls = Vec::new();
        let message_lower = message.to_lowercase();

        for tool in available_tools {
            if !self.should_call_tool(&message_lower, tool) {
                continue;
            }
            // A call whose required arguments can't be filled would be rejected; skip it
            if let Some(arguments) = self.extract_tool_arguments(message, tool) {
                tool_calls.push(ToolCall {
                    tool_name: tool.name.clone(),
                    arguments,
                });
            }
        }

        tool_calls
    }

    /// Check if a tool should be called based on the message
    fn should_call_tool(&self, message: &str, tool: &ToolInfo) -> bool {
        let tool_name_lower = tool.name.to_lowercase();
        let description_lower = tool.description.to_lowercase();

        // Simple keyword matching; stems so "calculate" matches "calculations"
        message.contains(&tool_name_lower)
            || ["search", "calculat", "weather", "time"]
                .iter()
                .any(|stem| message.contains(stem) && description_lower.contains(stem))
    }

    /// Fill the tool's required arguments from the message.
    ///
    /// Arguments are chosen by the tool's parameter schema: `expression` gets the math in the
    /// message, `location` the place after "in", and any other string parameter the whole
    /// message. Returns `None` when a required argument can't be filled this way.
    fn extract_tool_arguments(&self, message: &str, tool: &ToolInfo) -> Option<HashMap<String, serde_json::Value>> {
        let mut arguments = HashMap::new();

        for name in tool.parameters.required_names() {
            let param = tool.parameters.properties.get(&name)?;
            if param.parameter_type != ParameterType::String {
                return None;
            }
            let value = match name.as_str() {
                "expression" => self.extract_math_expression(message),
                "location" => self.extract_location(message),
                _ => message.to_string(),
            };
            if value.is_empty() {
                return None;
            }
            arguments.insert(name, serde_json::Value::String(value));
        }

        Some(arguments)
    }

    /// Extract mathematical expression from message
    fn extract_math_expression(&self, message: &str) -> String {
        // Simple regex-like extraction for math expressions
        // In a real implementation, you'd use a proper regex library
        let words: Vec<&str> = message.split_whitespace().collect();
        let mut expr = String::new();
        
        for word in words {
            if word.chars().any(|c| c.is_numeric() || "+-*/()".contains(c)) {
                expr.push_str(word);
                expr.push(' ');
            }
        }
        
        expr.trim().to_string()
    }

    /// Extract location from message
    fn extract_location(&self, message: &str) -> String {
        // Simple location extraction
        // In a real implementation, you'd use NER or more sophisticated methods
        let words: Vec<&str> = message.split_whitespace().collect();
        let mut location = String::new();
        
        for (i, word) in words.iter().enumerate() {
            if word.to_lowercase() == "in" && i + 1 < words.len() {
                location = words[i + 1..].join(" ");
                break;
            }
        }
        
        if location.is_empty() {
            "current location".to_string()
        } else {
            location
        }
    }
}

/// A tool call identified by the planner
#[derive(Debug, Clone)]
struct ToolCall {
    tool_name: String,
    arguments: HashMap<String, serde_json::Value>,
}

/// Advanced planner that uses LLM for planning
pub struct LLMPlanner {
    /// LLM gateway for planning
    llm_gateway: std::sync::Arc<axiom_ai_llm::LlmGateway>,
    /// Maximum number of steps
    max_steps: usize,
    /// Planning model
    model: String,
}

impl LLMPlanner {
    /// Create a new LLM planner
    pub fn new(llm_gateway: std::sync::Arc<axiom_ai_llm::LlmGateway>, model: String) -> Self {
        Self {
            llm_gateway,
            max_steps: 10,
            model,
        }
    }

    /// Set maximum number of steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

#[async_trait]
impl Planner for LLMPlanner {
    async fn create_plan(&self, request: PlanningRequest) -> Result<ExecutionPlan> {
        // Create a planning prompt
        let prompt = self.create_planning_prompt(&request);
        
        // Call LLM to generate plan
        let llm_request = axiom_ai_llm::LlmRequest::new(
            vec![axiom_ai_core::Message::user(prompt)],
            self.model.clone(),
        );

        let response = self.llm_gateway.generate(llm_request).await?;
        
        // Parse the response into a plan
        self.parse_plan_response(&response.content, &request)
    }
}

impl LLMPlanner {
    /// Create a planning prompt
    fn create_planning_prompt(&self, request: &PlanningRequest) -> String {
        let mut prompt = String::new();
        
        prompt.push_str("You are an AI planning agent. Create an execution plan based on the following conversation and available tools.\n\n");
        
        prompt.push_str("Conversation:\n");
        for message in &request.conversation {
            prompt.push_str(&format!("{}: {}\n", 
                match message.role {
                    axiom_ai_core::MessageRole::User => "User",
                    axiom_ai_core::MessageRole::Assistant => "Assistant",
                    axiom_ai_core::MessageRole::System => "System",
                    axiom_ai_core::MessageRole::Tool => "Tool",
                },
                message.text_content().unwrap_or("")
            ));
        }
        
        prompt.push_str("\nAvailable tools (arguments as JSON Schema):\n");
        for tool in &request.available_tools {
            prompt.push_str(&format!(
                "- {}: {}\n  arguments: {}\n",
                tool.name,
                tool.description,
                tool.parameters.to_json_schema()
            ));
        }
        
        prompt.push_str("\nCreate a step-by-step execution plan. Each step should be in the format:\n");
        prompt.push_str("STEP: <action_type> | <parameters> | <dependencies>\n");
        prompt.push_str("Available actions: generate_response, call_tool, update_memory, retrieve_memory\n");
        prompt.push_str("For call_tool, <parameters> is the tool name followed by its arguments as one JSON object, e.g.\n");
        prompt.push_str("STEP: call_tool | calculator {\"expression\": \"2 + 2\"} |\n");
        
        prompt
    }

    /// Parse LLM response into a plan
    fn parse_plan_response(&self, response: &str, request: &PlanningRequest) -> Result<ExecutionPlan> {
        let mut steps = Vec::new();
        let mut step_counter = 0;

        for line in response.lines() {
            if line.starts_with("STEP:") {
                let step = self.parse_step_line(line, &mut step_counter, request)?;
                steps.push(step);
            }
        }

        // Limit steps
        if steps.len() > self.max_steps {
            steps.truncate(self.max_steps);
        }

        let estimated_time = steps.iter().map(|s| s.estimated_time_ms).sum();

        Ok(ExecutionPlan {
            steps,
            metadata: HashMap::new(),
            estimated_time_ms: estimated_time,
        })
    }

    /// Parse a single step line
    fn parse_step_line(&self, line: &str, step_counter: &mut usize, request: &PlanningRequest) -> Result<PlanStep> {
        let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
        
        if parts.len() < 2 {
            return Err(AxiomError::ChainExecution("Invalid step format".to_string()));
        }

        let action_part = parts[0].replace("STEP:", "").trim().to_string();
        let params_part = parts[1];
        let deps_part = if parts.len() > 2 { parts[2] } else { "" };

        let action = self.parse_action(&action_part, params_part, request)?;
        let dependencies = if deps_part.is_empty() {
            Vec::new()
        } else {
            deps_part.split(',').map(|s| s.trim().to_string()).collect()
        };

        *step_counter += 1;

        Ok(PlanStep {
            id: format!("step_{}", step_counter),
            action,
            dependencies,
            estimated_time_ms: 1000, // Default estimate
            priority: 1,
        })
    }

    /// Parse action from text
    fn parse_action(&self, action_type: &str, params: &str, request: &PlanningRequest) -> Result<PlanAction> {
        match action_type.trim() {
            "generate_response" => {
                Ok(PlanAction::GenerateResponse {
                    prompt: params.to_string(),
                })
            }
            "call_tool" => {
                let params = params.trim();
                let (tool_name, raw_arguments) = params
                    .split_once(char::is_whitespace)
                    .unwrap_or((params, ""));
                if tool_name.is_empty() {
                    return Err(AxiomError::ChainExecution("Missing tool name".to_string()));
                }
                
                // Find the tool in available tools
                let tool = request.available_tools.iter()
                    .find(|t| t.name == tool_name)
                    .ok_or_else(|| AxiomError::ChainExecution(format!("Tool not found: {}", tool_name)))?;
                
                let arguments = parse_tool_arguments(tool, raw_arguments.trim())?;
                
                Ok(PlanAction::CallTool {
                    tool_name: tool_name.to_string(),
                    arguments,
                })
            }
            "update_memory" => {
                Ok(PlanAction::UpdateMemory {
                    content: params.to_string(),
                    memory_type: MemoryType::LongTerm,
                })
            }
            "retrieve_memory" => {
                Ok(PlanAction::RetrieveMemory {
                    query: params.to_string(),
                    limit: 5,
                })
            }
            _ => {
                Err(AxiomError::ChainExecution(format!("Unknown action type: {}", action_type)))
            }
        }
    }
}

/// Arguments for `tool` from the text after its name in a `call_tool` step.
///
/// The text is a JSON object of arguments. As a fallback, plain text is accepted for a tool
/// with exactly one required string parameter, and empty text for a tool with no required
/// parameters.
fn parse_tool_arguments(tool: &ToolInfo, text: &str) -> Result<HashMap<String, serde_json::Value>> {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(text) {
        return Ok(map.into_iter().collect());
    }

    let required = tool.parameters.required_names();
    match required.as_slice() {
        [] if text.is_empty() => Ok(HashMap::new()),
        [name]
            if tool
                .parameters
                .properties
                .get(name)
                .map_or(false, |param| param.parameter_type == ParameterType::String) =>
        {
            Ok(HashMap::from([(name.clone(), serde_json::Value::String(text.to_string()))]))
        }
        _ => Err(AxiomError::ChainExecution(format!(
            "Arguments for tool '{}' must be a JSON object, got: {}",
            tool.name, text
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axiom_ai_core::{Message, Tool};
    use axiom_ai_llm::LlmGateway;

    use super::*;
    use crate::agent::AgentConfig;
    use crate::tools::{CalculatorTool, FileTool, SearchTool, TimeTool, WeatherTool};

    fn info(tool: &dyn Tool) -> ToolInfo {
        ToolInfo {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.parameters(),
        }
    }

    fn request(message: &str, tools: &[&dyn Tool]) -> PlanningRequest {
        PlanningRequest {
            conversation: vec![Message::user(message)],
            // Non-empty context so the planner doesn't add a memory step
            context: vec![axiom_ai_core::MemoryItem::new("seen", MemoryType::ShortTerm, 0.5)],
            available_tools: tools.iter().map(|tool| info(*tool)).collect(),
            config: AgentConfig::default(),
        }
    }

    fn tool_calls(plan: &ExecutionPlan) -> Vec<(&str, &HashMap<String, serde_json::Value>)> {
        plan.steps
            .iter()
            .filter_map(|step| match &step.action {
                PlanAction::CallTool { tool_name, arguments } => Some((tool_name.as_str(), arguments)),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn simple_planner_fills_calculator_expression_that_validates() {
        let calculator = CalculatorTool::new();
        let plan = SimplePlanner::new(5)
            .create_plan(request("please calculate 2 + 2", &[&calculator]))
            .await
            .unwrap();

        let calls = tool_calls(&plan);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "calculator");
        assert_eq!(calls[0].1["expression"], "2 + 2");
        calculator.validate_arguments(calls[0].1).unwrap();
    }

    #[tokio::test]
    async fn simple_planner_arguments_validate_for_each_matched_builtin() {
        let search = SearchTool::new(None);
        let weather = WeatherTool::new(None);
        let time = TimeTool::new();
        let tools: [&dyn Tool; 3] = [&search, &weather, &time];
        let plan = SimplePlanner::new(10)
            .create_plan(request("search the weather in Oslo at this time", &tools))
            .await
            .unwrap();

        let calls = tool_calls(&plan);
        assert_eq!(calls.len(), 3);
        for (name, arguments) in calls {
            let tool = tools.iter().find(|tool| tool.name() == name).unwrap();
            tool.validate_arguments(arguments)
                .unwrap_or_else(|e| panic!("{name} arguments rejected: {e}"));
        }
    }

    #[tokio::test]
    async fn simple_planner_skips_tool_whose_required_arguments_it_cannot_fill() {
        let file = FileTool::new();
        let plan = SimplePlanner::new(5)
            .create_plan(request("read the file", &[&file]))
            .await
            .unwrap();

        assert!(tool_calls(&plan).is_empty());
    }

    #[tokio::test]
    async fn simple_planner_keeps_response_step_last_when_truncating() {
        let search = SearchTool::new(None);
        let time = TimeTool::new();
        let plan = SimplePlanner::new(2)
            .create_plan(request("search what time it is", &[&search, &time]))
            .await
            .unwrap();

        assert_eq!(plan.steps.len(), 2);
        let last = plan.steps.last().unwrap();
        assert!(matches!(last.action, PlanAction::GenerateResponse { .. }));
        assert_eq!(last.dependencies, vec![plan.steps[0].id.clone()]);
    }

    fn llm_planner() -> LLMPlanner {
        LLMPlanner::new(Arc::new(LlmGateway::new("none")), "model".to_string())
    }

    #[test]
    fn llm_planner_parses_json_tool_arguments() {
        let calculator = CalculatorTool::new();
        let plan = llm_planner()
            .parse_plan_response(
                "STEP: call_tool | calculator {\"expression\": \"2 + 2\"} |\nSTEP: generate_response | answer |",
                &request("", &[&calculator]),
            )
            .unwrap();

        let calls = tool_calls(&plan);
        assert_eq!(calls[0].1["expression"], "2 + 2");
        calculator.validate_arguments(calls[0].1).unwrap();
    }

    #[test]
    fn llm_planner_maps_plain_text_to_single_required_string_parameter() {
        let search = SearchTool::new(None);
        let plan = llm_planner()
            .parse_plan_response("STEP: call_tool | search rust async traits |", &request("", &[&search]))
            .unwrap();

        let calls = tool_calls(&plan);
        assert_eq!(calls[0].1["query"], "rust async traits");
        search.validate_arguments(calls[0].1).unwrap();
    }

    #[test]
    fn llm_planner_rejects_plain_text_for_multi_argument_tool() {
        let file = FileTool::new();
        let result = llm_planner()
            .parse_plan_response("STEP: call_tool | file read notes.txt |", &request("", &[&file]));

        assert!(result.is_err());
    }
}
