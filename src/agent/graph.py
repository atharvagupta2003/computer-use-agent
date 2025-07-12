"""LangGraph workflow for the Computer Use Agent."""

from __future__ import annotations

import asyncio
from typing import Optional

from langgraph.graph import END, START, StateGraph

from agent.configuration import get_config
from agent.cua.sandbox import start_desktop_sandbox, get_desktop, _DESKTOP_CACHE
from agent.prompts import executor_system_prompt
from agent.state import (
    ExecutionState, 
    GraphInput, 
    TaskStatus, 
    SemanticInstruction, 
    ExecutorReport, 
    ActionType,
    SemanticInstructionSchema
)
from agent.utils import utcnow

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_state_field(state, field_name, default=None):
    """Get a field from state, handling both dict and object formats."""
    if isinstance(state, dict):
        return state.get(field_name, default)
    else:
        return getattr(state, field_name, default)

def set_state_field(state, field_name, value):
    """Set a field in state, handling both dict and object formats."""
    if isinstance(state, dict):
        state[field_name] = value
    else:
        setattr(state, field_name, value)

def map_platform_to_environment(platform: str) -> str:
    """Map platform selection to CUA environment name."""
    mapping = {
        "google-chrome": "browser",
        "firefox": "browser", 
        "vscode": "editor",
    }
    return mapping.get(platform, "browser")

def launch_application_in_sandbox(sandbox_id: str, platform: str) -> None:
    """Launch the specified application in the sandbox."""
    
    desktop = get_desktop(sandbox_id)
    if not desktop:
        raise RuntimeError(f"No desktop sandbox found for ID: {sandbox_id}")

    # Map platform to e2b launch command
    launch_commands = {
        "google-chrome": "google-chrome",
        "firefox": "firefox",
        "vscode": "code",
    }

    command = launch_commands.get(platform)
    if not command:
        raise ValueError(f"Unknown platform: {platform}")
    
    desktop.launch(command)
    # Wait for application to start
    desktop.wait(5000)  # 5 seconds

def cleanup_sandbox(sandbox_id: str, graceful: bool = True) -> None:
    """Clean up sandbox resources gracefully."""
    
    desktop = get_desktop(sandbox_id)
    if not desktop:
        return  # Already cleaned up
    
    try:
        if graceful:
            # Stop streaming gracefully
            if hasattr(desktop, 'stream') and desktop.stream:
                try:
                    desktop.stream.stop()
                except Exception:
                    pass  # Stream might already be stopped
            
            # Save any work or close applications gracefully
            # This is platform-specific, but we can try some common shortcuts
            try:
                # Save any open documents (Ctrl+S)
                desktop.key_down("ctrl")
                desktop.key_down("s")
                desktop.key_up("s")
                desktop.key_up("ctrl")
                desktop.wait(1000)  # Wait 1 second
                
                # Close applications gracefully (Alt+F4)
                desktop.key_down("alt")
                desktop.key_down("f4")
                desktop.key_up("f4")
                desktop.key_up("alt")
                desktop.wait(1000)
            except Exception:
                pass  # Ignore errors during graceful cleanup
        
        # Kill the sandbox
        desktop.kill()
        
    except Exception as e:
        # Log error but don't raise - cleanup should be best effort
        print(f"Error during sandbox cleanup: {e}")
    
    finally:
        # Remove from cache
        if sandbox_id in _DESKTOP_CACHE:
            del _DESKTOP_CACHE[sandbox_id]

def is_screenshot_duplicate(new_screenshot: bytes, previous_screenshots: list[str], threshold: float = 0.95) -> bool:
    """Check if a screenshot is substantially the same as a previous one."""
    import hashlib
    import base64
    
    # Quick hash comparison first
    new_hash = hashlib.md5(new_screenshot).hexdigest()
    
    for prev_b64 in previous_screenshots[-5:]:  # Only check last 5 screenshots
        try:
            prev_bytes = base64.b64decode(prev_b64)
            prev_hash = hashlib.md5(prev_bytes).hexdigest()
            
            # If hashes match exactly, it's a duplicate
            if new_hash == prev_hash:
                return True
                
            # For more sophisticated comparison, we could use image similarity
            # but for now, hash comparison is sufficient for exact duplicates
            
        except Exception:
            continue  # Skip invalid base64
    
    return False

def trim_message_history(messages: list[dict], max_messages: int = 20) -> list[dict]:
    """Trim message history to stay within context limits."""
    if len(messages) <= max_messages:
        return messages
    
    # Always keep the system message (first message)
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    
    # Keep the most recent messages
    recent_messages = messages[-(max_messages - 1):] if system_msg else messages[-max_messages:]
    
    # Reconstruct with system message at the beginning
    if system_msg:
        return [system_msg] + recent_messages
    else:
        return recent_messages

# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def setup_sandbox(state: ExecutionState, *, config: dict) -> ExecutionState:
    """Set up the sandbox environment for execution."""
    
    # Extract user request from the config
    # LangGraph passes the user input in different ways depending on how it's invoked
    user_request = ""
    configurable = config.get("configurable", {})
    
    # First, check if we have a direct user_request in the state (from GraphInput)
    if hasattr(state, 'user_request') and state.user_request:
        user_request = state.user_request
    # Try different ways to get the user request from config
    elif "user_request" in configurable:
        user_request = configurable["user_request"]
    elif "input" in configurable:
        user_request = configurable["input"]
    elif "query" in configurable:
        user_request = configurable["query"]
    elif "message" in configurable:
        user_request = configurable["message"]
    else:
        # If no user request found, use a default
        user_request = "Open Chrome and search for information"
    
    # Handle both dict and object state
    if isinstance(state, dict):
        state["user_request"] = user_request
    else:
        state.user_request = user_request
    
    # Check if we have existing sandbox info in the config or state
    if isinstance(state, dict):
        sandbox_id = state.get('sandbox_id') or configurable.get("sandbox_id")
        sandbox_url = state.get('sandbox_url') or configurable.get("sandbox_url")
    else:
        sandbox_id = getattr(state, 'sandbox_id', None) or configurable.get("sandbox_id")
        sandbox_url = getattr(state, 'sandbox_url', None) or configurable.get("sandbox_url")
    
    # Use existing sandbox if provided, otherwise create new one
    if sandbox_id and sandbox_url:
        if isinstance(state, dict):
            state["sandbox_id"] = sandbox_id
            state["sandbox_url"] = sandbox_url
        else:
            state.sandbox_id = sandbox_id
            state.sandbox_url = sandbox_url
        print(f"Using existing sandbox: {sandbox_id}")
    else:
        # Create new sandbox
        print("Creating new sandbox...")
        cfg = get_config()
        sandbox_details = start_desktop_sandbox(timeout=cfg.sandbox_timeout)
        if isinstance(state, dict):
            state["sandbox_id"] = sandbox_details.sandbox_id
            state["sandbox_url"] = sandbox_details.url
            state["sandbox_expires_at"] = sandbox_details.expires_at
        else:
            state.sandbox_id = sandbox_details.sandbox_id
            state.sandbox_url = sandbox_details.url
            state.sandbox_expires_at = sandbox_details.expires_at
        print(f"Created new sandbox: {sandbox_details.sandbox_id}")
    
    # Set status to in progress
    if isinstance(state, dict):
        state["status"] = TaskStatus.in_progress
    else:
        state.status = TaskStatus.in_progress
    
    return state

async def strategic_brain(state: ExecutionState, *, config: dict) -> ExecutionState:
    """Strategic brain that analyzes screen state and gives semantic instructions."""
    
    try:
        print(f"Strategic brain started - {state.get_interaction_summary()}")
        
        # Get configuration
        agent_config = config.get("configurable", {})
        api_key = agent_config.get("openai_api_key")
        if not api_key:
            try:
                global_config = get_config()
                api_key = global_config.openai_api_key
                model = global_config.model_planner
            except Exception as e:
                print(f"Failed to get global config: {e}")
                api_key = None
        else:
            model = agent_config.get("model_planner", "gpt-4o-2024-11-20")
        
        if not api_key:
            print("No API key found in config or environment")
            state.status = TaskStatus.failed
            state.message_for_user = "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            return state
        
        # Build context for the strategic brain
        user_request = state.user_request
        last_report = state.last_executor_report
        
        context_parts = [
            f"USER REQUEST: {user_request}",
            f"INTERACTION: {state.get_interaction_summary()}",
            f"RECENT ACTIONS: {state.get_recent_actions_summary()}",
        ]
        
        if last_report:
            context_parts.append(f"LAST ACTION: {last_report.action_performed or 'Initial screenshot'}")
            if last_report.action_result:
                context_parts.append(f"ACTION RESULT: {last_report.action_result}")
            if last_report.error:
                context_parts.append(f"ERROR: {last_report.error}")
            if last_report.elements_found:
                context_parts.append(f"ELEMENTS FOUND: {', '.join(last_report.elements_found)}")
        
        context = "\n".join(context_parts)
        
        # Create the strategic brain prompt
        brain_prompt = f"""You are the strategic brain of a computer use agent. Your job is to analyze the current screen and give semantic instructions to a vision-capable executor.

{context}

You will be shown a screenshot of the current screen. Based on what you see and the user's request, give a semantic instruction about what to do next.

IMPORTANT: You give HIGH-LEVEL SEMANTIC INSTRUCTIONS, not exact coordinates. The executor has vision capabilities and will find the elements you describe.

Available action types and how to use them:
- screenshot: Take a screenshot to see current state
- click_element: Click on a described element (e.g., "first search result", "login button", "search box")
- type_text: Type text into the currently focused element
- press_key: Press keys like enter, tab, escape
- scroll: Scroll up or down
- wait: Wait for specified seconds
- navigate: Navigate to a URL

Your response should include:
1. action_type: The type of action to perform
2. description: Clear description of what you want to accomplish
3. target_element: Description of the element to interact with (for click_element)
4. text_content: Text to type (for type_text)
5. key_sequence: Keys to press (for press_key)
6. scroll_direction: Direction to scroll (for scroll)
7. wait_seconds: Time to wait (for wait)
8. url: URL to navigate to (for navigate)
9. reasoning: Why this action is needed
10. is_task_complete: Whether the user's request is fully completed
11. completion_message: Message to show user when task is complete

Examples of good semantic instructions:
- "Click on the first search result about Brisbane startups"
- "Click on the search box at the top of the page"
- "Click on the link that says 'Top 10 Startups'"
- "Scroll down to see more search results"
- "Type the search query into the search box"

Focus on making strategic progress toward the user's goal. Avoid repeating the same action multiple times."""
        
        # Create LangChain client with structured output
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0
        )
        
        # Use structured output
        structured_llm = llm.with_structured_output(SemanticInstructionSchema, method="function_calling")
        
        # Prepare messages for the brain
        messages = [{"role": "user", "content": brain_prompt}]
        
        # Add screenshot if available
        if last_report and last_report.screenshot_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Current screen:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{last_report.screenshot_b64}"}}
                ]
            })
        
        print("Strategic brain analyzing screen and creating instruction...")
        try:
            instruction_response = await structured_llm.ainvoke(messages)
            print(f"Brain instruction: {instruction_response.action_type} - {instruction_response.description}")
            
            # Check for repeated actions
            action_desc = f"{instruction_response.action_type}:{instruction_response.target_element or instruction_response.description}"
            if state.is_repeating_action(action_desc):
                print(f"Detected repeated action: {action_desc}. Trying alternative approach.")
                # Modify the instruction to break the loop
                if instruction_response.action_type == "click_element":
                    instruction_response.action_type = "scroll"
                    instruction_response.description = "Scroll to find different elements"
                    instruction_response.target_element = ""
                    instruction_response.scroll_direction = "down"
                    instruction_response.reasoning = "Breaking repetitive clicking by scrolling to find new elements"
            
            # Convert to SemanticInstruction object
            instruction = SemanticInstruction(
                action_type=ActionType(instruction_response.action_type),
                description=instruction_response.description,
                target_element=instruction_response.target_element,
                text_content=instruction_response.text_content,
                key_sequence=instruction_response.key_sequence,
                scroll_direction=instruction_response.scroll_direction,
                wait_seconds=instruction_response.wait_seconds,
                url=instruction_response.url,
                reasoning=instruction_response.reasoning,
                is_task_complete=instruction_response.is_task_complete,
                completion_message=instruction_response.completion_message
            )
            
            # Store instruction in state
            state.set_brain_instruction(instruction)
            
            # Check if task is complete
            if instruction.is_task_complete:
                state.status = TaskStatus.completed
                state.message_for_user = instruction.completion_message
                print(f"Task completed: {instruction.completion_message}")
            
            return state
            
        except Exception as e:
            print(f"Failed to get brain instruction: {e}")
            state.status = TaskStatus.failed
            state.message_for_user = f"Brain analysis failed: {str(e)}"
            return state
        
    except Exception as e:
        print(f"Error in strategic brain: {e}")
        state.status = TaskStatus.failed
        state.message_for_user = f"Brain error: {str(e)}"
        return state

async def vision_executor(state: ExecutionState) -> ExecutionState:
    """Vision-capable executor that interprets semantic instructions and performs actions."""
    
    try:
        print(f"Vision executor started - {state.get_interaction_summary()}")
        
        # Increment interaction counter
        state.increment_interaction()
        
        # Check for max interactions
        if state.is_max_interactions_reached():
            print("Maximum interactions reached")
            state.status = TaskStatus.failed
            state.message_for_user = "Maximum interactions reached. Task may be too complex or stuck in a loop."
            return state
        
        # Get desktop
        desktop = get_desktop(state.sandbox_id)
        if not desktop:
            state.status = TaskStatus.failed
            state.message_for_user = "Desktop sandbox not found"
            return state
        
        # Launch application if not already launched
        if not state.app_launched:
            print(f"Launching {state.application_platform}...")
            try:
                if state.application_platform == "google-chrome":
                    desktop.launch("google-chrome")
                elif state.application_platform == "firefox":
                    desktop.launch("firefox")
                elif state.application_platform == "vscode":
                    desktop.launch("code")
                else:
                    desktop.launch("google-chrome")
                
                desktop.wait(5000)  # Wait for application to start
                state.app_launched = True
                print(f"Successfully launched {state.application_platform}")
            except Exception as e:
                print(f"Error launching application: {e}")
                state.status = TaskStatus.failed
                state.message_for_user = f"Failed to launch application: {str(e)}"
                return state
        
        # Execute the semantic instruction from strategic brain
        instruction = state.current_instruction
        action_performed = None
        action_result = None
        error = None
        elements_found = []
        
        if instruction:
            print(f"Executing instruction: {instruction.action_type} - {instruction.description}")
            
            try:
                if instruction.action_type == ActionType.screenshot:
                    # Just take screenshot - no other action needed
                    action_performed = "screenshot"
                    action_result = "Screenshot taken"
                    
                elif instruction.action_type == ActionType.click_element:
                    # Use the existing computer tool to find and click elements
                    if instruction.target_element:
                        # For now, use a simple heuristic approach
                        # In a real implementation, you'd use vision models to find elements
                        action_performed = f"click_element: {instruction.target_element}"
                        
                        # Simple heuristic: try common click locations based on element description
                        if "search" in instruction.target_element.lower():
                            # Click on search box area
                            desktop.left_click(400, 100)
                            action_result = f"Clicked on search element: {instruction.target_element}"
                        elif "first" in instruction.target_element.lower() and "result" in instruction.target_element.lower():
                            # Click on first search result
                            desktop.left_click(400, 200)
                            action_result = f"Clicked on first search result: {instruction.target_element}"
                        elif "link" in instruction.target_element.lower():
                            # Click on a link
                            desktop.left_click(400, 300)
                            action_result = f"Clicked on link: {instruction.target_element}"
                        else:
                            # Generic click in content area
                            desktop.left_click(400, 250)
                            action_result = f"Clicked on element: {instruction.target_element}"
                        
                        elements_found.append(instruction.target_element)
                    else:
                        error = "No target element specified for click_element action"
                        
                elif instruction.action_type == ActionType.type_text:
                    if instruction.text_content:
                        desktop.write(instruction.text_content)
                        action_performed = f"type_text: {instruction.text_content}"
                        action_result = f"Typed: {instruction.text_content}"
                        print(f"Typed: {instruction.text_content}")
                    else:
                        error = "No text content provided for type_text action"
                        
                elif instruction.action_type == ActionType.press_key:
                    if instruction.key_sequence:
                        for key in instruction.key_sequence:
                            desktop.press(key.lower())
                            print(f"Pressed key: {key}")
                        action_performed = f"press_key: {instruction.key_sequence}"
                        action_result = f"Pressed keys: {', '.join(instruction.key_sequence)}"
                    else:
                        error = "No key sequence provided for press_key action"
                        
                elif instruction.action_type == ActionType.scroll:
                    direction = instruction.scroll_direction
                    amount = 3  # Default scroll amount
                    desktop.scroll(direction, amount)
                    action_performed = f"scroll {direction}"
                    action_result = f"Scrolled {direction} by {amount}"
                    print(f"Scrolled {direction} by {amount}")
                    
                elif instruction.action_type == ActionType.wait:
                    wait_time = instruction.wait_seconds
                    desktop.wait(wait_time * 1000)  # e2b expects milliseconds
                    action_performed = f"wait {wait_time}s"
                    action_result = f"Waited {wait_time} seconds"
                    print(f"Waited {wait_time} seconds")
                    
                elif instruction.action_type == ActionType.navigate:
                    if instruction.url:
                        # Navigate by typing URL in address bar
                        desktop.key_down("ctrl")
                        desktop.key_down("l")
                        desktop.key_up("l")
                        desktop.key_up("ctrl")
                        desktop.wait(500)
                        desktop.write(instruction.url)
                        desktop.press("enter")
                        action_performed = f"navigate to {instruction.url}"
                        action_result = f"Navigated to {instruction.url}"
                        print(f"Navigated to {instruction.url}")
                    else:
                        error = "No URL provided for navigate action"
                        
                # Wait a bit for action to complete
                desktop.wait(1000)
                
            except Exception as e:
                error = f"Action execution failed: {str(e)}"
                print(f"Action execution error: {e}")
        
        # Always take a screenshot after action to report current state
        try:
            screenshot_data = desktop.screenshot()
            if screenshot_data:
                print(f"Screenshot taken: {len(screenshot_data)} bytes")
                
                # Create executor report
                report = state.create_executor_report(
                    screenshot_data=screenshot_data,
                    action_performed=action_performed,
                    action_result=action_result,
                    error=error,
                    elements_found=elements_found
                )
                
                # Store report in state
                state.set_executor_report(report)
                
                print(f"Executor report created and stored")
                
            else:
                print("Failed to take screenshot")
                state.status = TaskStatus.failed
                state.message_for_user = "Failed to take screenshot"
                return state
                
        except Exception as e:
            print(f"Screenshot error: {e}")
            state.status = TaskStatus.failed
            state.message_for_user = f"Screenshot error: {str(e)}"
            return state
        
        return state
        
    except Exception as e:
        print(f"Error in vision executor: {e}")
        state.status = TaskStatus.failed
        state.message_for_user = f"Executor error: {str(e)}"
        return state

async def cleanup_node(state) -> ExecutionState:
    """Clean up sandbox resources when execution is complete."""
    
    try:
        # Wait 20 seconds after task completion before cleanup
        status = get_state_field(state, "status", "")
        print(f"Cleanup node: status={status}, type={type(status)}")
        
        # Handle both enum and string status values
        if status == TaskStatus.completed or str(status) == "completed":
            print("Task completed. Waiting 20 seconds before cleanup...")
            await asyncio.sleep(20)
            print("20 seconds elapsed, proceeding with cleanup...")
        
        sandbox_id = get_state_field(state, "sandbox_id", "")
        if sandbox_id:
            cleanup_sandbox(sandbox_id, graceful=True)
            print("Sandbox cleanup completed successfully")
    except Exception as e:
        print(f"Cleanup failed: {e}")
        set_state_field(state, "message_for_user", f"Cleanup failed: {e}")
    
    return state

# ---------------------------------------------------------------------------
# Conditional logic
# ---------------------------------------------------------------------------

def should_continue_from_brain(state) -> str:
    """Determine the next step after brain analysis."""
    status = get_state_field(state, "status", "")
    
    print(f"Brain conditional check: status={status}")
    
    # Handle both enum and string status values
    if status == TaskStatus.failed or str(status) == "failed":
        print("Going to cleanup due to failed status")
        return "cleanup"
    elif status == TaskStatus.completed or str(status) == "completed":
        print("Going to cleanup due to completed status")
        return "cleanup"
    else:
        print("Going to executor")
        return "executor"

def should_continue_from_executor(state) -> str:
    """Determine the next step after executor action."""
    status = get_state_field(state, "status", "")
    
    print(f"Executor conditional check: status={status}")
    
    # Handle both enum and string status values
    if status == TaskStatus.failed or str(status) == "failed":
        print("Going to cleanup due to failed status")
        return "cleanup"
    elif status == TaskStatus.completed or str(status) == "completed":
        print("Going to cleanup due to completed status")
        return "cleanup"
    else:
        print("Going back to brain for next decision")
        return "brain"

def create_graph() -> StateGraph:
    """Create the Computer Use Agent graph with semantic brain-executor feedback loop."""
    
    # Create the graph with input schema
    graph = StateGraph(ExecutionState, input=GraphInput)
    
    # Add nodes
    graph.add_node("setup_sandbox", setup_sandbox)
    graph.add_node("brain", strategic_brain)
    graph.add_node("executor", vision_executor)
    graph.add_node("cleanup", cleanup_node)
    
    # Add edges
    graph.add_edge(START, "setup_sandbox")
    graph.add_edge("setup_sandbox", "brain")
    
    # Add conditional edges for brain-executor feedback loop
    graph.add_conditional_edges(
        "brain",
        should_continue_from_brain,
        {
            "executor": "executor",  # Go to executor to perform action
            "cleanup": "cleanup",    # Go to cleanup if failed/completed
        }
    )
    
    # Add conditional edges from executor
    graph.add_conditional_edges(
        "executor",
        should_continue_from_executor,
        {
            "brain": "brain",        # Go back to brain for next decision
            "cleanup": "cleanup",    # Go to cleanup if failed/completed
        }
    )
    
    graph.add_edge("cleanup", END)
    
    return graph

# Legacy functions for backward compatibility
async def executor(state: ExecutionState) -> ExecutionState:
    """Legacy executor function - redirects to new vision_executor."""
    return await vision_executor(state)

async def planner_node(state: ExecutionState, *, config: dict) -> ExecutionState:
    """Legacy planner function - redirects to new strategic_brain."""
    return await strategic_brain(state, config=config)


# Create and compile the graph
workflow_graph = create_graph().compile()
workflow_graph.name = "Computer Use Agent"

__all__ = ["workflow_graph"]
