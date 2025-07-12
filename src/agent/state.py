"""State objects shared across LangGraph nodes.

This module defines the **static** type contracts (using Pydantic) that govern
information flow between nodes in the Computer-Use Agent graph. Keeping these
schemas centralised makes it easy to reason about what each node may read or
write and enables automatic validation.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Literal, Optional
from enum import Enum
from dataclasses import dataclass, field

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, validator, root_validator

__all__ = [
    "GraphInput",
    "ExecutionState",
    "Status",
]

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

Status = Literal[
    "in_progress",
    "ambiguity",
    "human-intervention",
    "sensitive-action",
    "problem",
    "done",
    "completed",
    "failed",
]


class TaskStatus(str, Enum):
    """Status of the overall task execution."""
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class ActionType(str, Enum):
    """Types of actions the executor can perform."""
    screenshot = "screenshot"
    click_element = "click_element"  # Click on a described element
    type_text = "type_text"
    press_key = "press_key"
    scroll = "scroll"
    wait = "wait"
    navigate = "navigate"


@dataclass
class SemanticInstruction:
    """Represents a semantic instruction from the brain to the executor."""
    action_type: ActionType
    description: str  # Human-readable description of what to do
    target_element: str = ""  # Description of element to interact with
    text_content: str = ""  # Text to type
    key_sequence: List[str] = field(default_factory=list)  # Keys to press
    scroll_direction: str = "down"  # Scroll direction
    wait_seconds: int = 2  # Wait time
    url: str = ""  # URL to navigate to
    reasoning: str = ""  # Why this action is needed
    is_task_complete: bool = False
    completion_message: str = ""


@dataclass
class ExecutorReport:
    """Represents a report from the executor about the current screen state."""
    screenshot_b64: str
    action_performed: Optional[str] = None
    action_result: Optional[str] = None
    error: Optional[str] = None
    elements_found: List[str] = field(default_factory=list)  # Elements the executor found
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


# Pydantic models for structured output from planner brain
class SemanticInstructionSchema(BaseModel):
    """Pydantic model for structured semantic instruction output."""
    action_type: str = Field(description="Type of action: screenshot, click_element, type_text, press_key, scroll, wait, navigate")
    description: str = Field(description="Human-readable description of what to do")
    target_element: str = Field(default="", description="Description of element to interact with (e.g., 'first search result', 'login button', 'search box')")
    text_content: str = Field(default="", description="Text to type")
    key_sequence: List[str] = Field(default_factory=list, description="Keys to press (e.g., ['enter', 'tab'])")
    scroll_direction: str = Field(default="down", description="Scroll direction: up or down")
    wait_seconds: int = Field(default=2, description="Number of seconds to wait")
    url: str = Field(default="", description="URL to navigate to")
    reasoning: str = Field(description="Explanation of why this action is needed")
    is_task_complete: bool = Field(default=False, description="Whether the task is fully completed")
    completion_message: str = Field(default="", description="Message to show user when task is complete")

    class Config:
        extra = "allow"


class ExecutionState(BaseModel):
    """State for the Computer Use Agent execution with semantic brain-executor communication."""
    
    # Basic execution state
    status: TaskStatus = TaskStatus.pending
    message_for_user: str = ""
    user_request: str = ""
    
    # Sandbox management
    sandbox_id: str = ""
    sandbox_url: str = ""
    sandbox_expires_at: Optional[datetime] = None
    app_launched: bool = False
    application_platform: str = "web"
    
    # Semantic Brain-Executor communication
    current_instruction: Optional[SemanticInstruction] = None
    last_executor_report: Optional[ExecutorReport] = None
    interaction_count: int = 0
    max_interactions: int = 30  # Prevent infinite loops
    
    # Action history for loop prevention
    action_history: List[str] = Field(default_factory=list)
    max_history_length: int = 10
    
    # Conversation tracking
    conversation_turn: int = 0
    last_response_id: Optional[str] = None
    
    # Screenshots and messages (for context)
    screenshots: List[str] = Field(default_factory=list)
    messages: List[dict] = Field(default_factory=list)
    
    # Legacy fields for backward compatibility (can be removed later)
    execution_plan: List[dict] = Field(default_factory=list)
    current_step_index: int = 0
    planning_completed: bool = False
    
    def add_screenshot(self, screenshot_data) -> None:
        """Add a screenshot to the state."""
        if isinstance(screenshot_data, (bytes, bytearray)):
            # Convert bytes to base64 string
            import base64
            screenshot_b64 = base64.b64encode(screenshot_data).decode()
            self.screenshots.append(screenshot_b64)
        elif isinstance(screenshot_data, str):
            # Already a string, assume it's base64 encoded
            self.screenshots.append(screenshot_data)
        else:
            raise ValueError(f"Unsupported screenshot data type: {type(screenshot_data)}")
    
    def create_executor_report(self, screenshot_data, action_performed: Optional[str] = None, 
                             action_result: Optional[str] = None, error: Optional[str] = None,
                             elements_found: List[str] = None) -> ExecutorReport:
        """Create an executor report with the current screen state."""
        if isinstance(screenshot_data, (bytes, bytearray)):
            import base64
            screenshot_b64 = base64.b64encode(screenshot_data).decode()
        elif isinstance(screenshot_data, str):
            screenshot_b64 = screenshot_data
        else:
            raise ValueError(f"Unsupported screenshot data type: {type(screenshot_data)}")
        
        return ExecutorReport(
            screenshot_b64=screenshot_b64,
            action_performed=action_performed,
            action_result=action_result,
            error=error,
            elements_found=elements_found or []
        )
    
    def set_brain_instruction(self, instruction: SemanticInstruction) -> None:
        """Set the current brain instruction."""
        self.current_instruction = instruction
        # Add to action history for loop prevention
        action_desc = f"{instruction.action_type}:{instruction.target_element or instruction.description}"
        self.action_history.append(action_desc)
        # Keep history limited
        if len(self.action_history) > self.max_history_length:
            self.action_history = self.action_history[-self.max_history_length:]
    
    def set_executor_report(self, report: ExecutorReport) -> None:
        """Set the last executor report."""
        self.last_executor_report = report
        self.add_screenshot(report.screenshot_b64)
    
    def increment_interaction(self) -> None:
        """Increment the interaction counter."""
        self.interaction_count += 1
    
    def is_max_interactions_reached(self) -> bool:
        """Check if maximum interactions have been reached."""
        return self.interaction_count >= self.max_interactions
    
    def is_repeating_action(self, action_desc: str, max_repeats: int = 3) -> bool:
        """Check if an action is being repeated too many times."""
        recent_actions = self.action_history[-max_repeats:]
        return recent_actions.count(action_desc) >= max_repeats
    
    def get_interaction_summary(self) -> str:
        """Get a summary of the current interaction state."""
        return f"Interaction {self.interaction_count}/{self.max_interactions}"
    
    def get_recent_actions_summary(self) -> str:
        """Get a summary of recent actions for context."""
        if not self.action_history:
            return "No previous actions"
        recent = self.action_history[-5:]  # Last 5 actions
        return f"Recent actions: {' -> '.join(recent)}"

    # Legacy methods for backward compatibility
    def get_current_step(self) -> Optional[dict]:
        """Legacy method - returns None in new architecture."""
        return None
    
    def mark_current_step_completed(self, result: Optional[str] = None) -> None:
        """Legacy method - no-op in new architecture."""
        pass
    
    def mark_current_step_failed(self, error: str) -> None:
        """Legacy method - no-op in new architecture."""
        pass
    
    def is_plan_completed(self) -> bool:
        """Legacy method - returns completion status."""
        return self.status == TaskStatus.completed
    
    def get_plan_summary(self) -> str:
        """Get a summary of the current state."""
        if self.current_instruction:
            return f"Current Instruction: {self.current_instruction.description}"
        return "No current instruction"

    # Make the model mutable for LangGraph
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "populate_by_name": True,
    }


class GraphInput(BaseModel):
    """Input schema for the graph."""
    user_request: str
