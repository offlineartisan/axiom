//! Axiom Agents - Agent framework with tools, memory, and execution planning
//!
//! This crate provides a comprehensive agent framework that includes:
//! - Tool execution and management
//! - Memory systems for conversation and context
//! - Execution planning and reasoning
//! - Safety guards and validation

pub mod agent;
pub mod planner;
pub mod safety;
pub mod tools;
pub mod turn;

pub use agent::*;
pub use planner::*;
pub use safety::*;
pub use tools::*;
pub use turn::*;

/// Re-exports commonly used types for convenience
pub mod prelude {
    pub use crate::{
        Agent, AgentConfig, AgentState,
        Planner, ExecutionPlan, PlanStep,
        TurnOutcome, StopReason,
        SafetyGuard, SafetyResult,
    };
    pub use axiom_ai_core::prelude::*;
    pub use axiom_ai_llm::prelude::*;
}
