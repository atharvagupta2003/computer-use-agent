"""Vision utilities for the Computer Use Agent executor."""

from __future__ import annotations

from typing import List, Tuple, Optional
from agent.state import ActionType, SemanticInstruction


def execute_semantic_instruction(desktop, instruction: SemanticInstruction) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """
    Execute a semantic instruction using the desktop interface.
    
    Returns:
        Tuple of (action_performed, action_result, error, elements_found)
    """
    action_performed = None
    action_result = None
    error = None
    elements_found = []
    
    try:
        if instruction.action_type == ActionType.screenshot:
            # Just take screenshot - no other action needed
            action_performed = "screenshot"
            action_result = "Screenshot taken"
            
        elif instruction.action_type == ActionType.click_element:
            if instruction.target_element:
                # This case is now handled by the OpenAI Computer Use model in the executor
                # for visual analysis and precise coordinate detection
                action_performed = f"click_element: {instruction.target_element}"
                action_result = f"Attempted to click on element: {instruction.target_element}"
                elements_found = [instruction.target_element]
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
                # Navigate by typing URL in address bar using correct e2b API
                # Use Ctrl+L to focus address bar
                desktop.press("ctrl+l")
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
    
    return action_performed, action_result, error, elements_found


def build_brain_context(user_request: str, interaction_summary: str, recent_actions: str, recent_brain_instructions: str, previous_output: str, previous_user_request: str, last_report) -> str:
    """Build context string for the strategic brain."""
    context_parts = [
        f"USER REQUEST: {user_request}",
        f"INTERACTION: {interaction_summary}",
        f"RECENT ACTIONS: {recent_actions}",
        f"RECENT BRAIN INSTRUCTIONS: {recent_brain_instructions}",
    ]
    
    # Add previous conversation context if available
    if previous_user_request and previous_output:
        context_parts.append(f"PREVIOUS USER REQUEST: {previous_user_request}")
        context_parts.append(f"PREVIOUS OUTPUT: {previous_output}")
    elif previous_output:
        context_parts.append(f"PREVIOUS OUTPUT: {previous_output}")
    
    if last_report:
        context_parts.append(f"LAST ACTION: {last_report.action_performed or 'Initial screenshot'}")
        if last_report.action_result:
            context_parts.append(f"ACTION RESULT: {last_report.action_result}")
        if last_report.error:
            context_parts.append(f"ERROR: {last_report.error}")
        if last_report.elements_found:
            context_parts.append(f"ELEMENTS FOUND: {', '.join(last_report.elements_found)}")
    
    return "\n".join(context_parts)


def should_force_completion(user_request: str, interaction_count: int, recent_actions: List[str]) -> tuple[bool, str]:
    """
    Intelligently determine if task should be forced to completion based on request patterns.
    
    Returns:
        tuple: (should_complete, completion_message)
    """
    user_request_lower = user_request.lower()
    
    # For search requests, if we've taken multiple actions, likely complete
    if any(keyword in user_request_lower for keyword in ["search", "find", "look for", "top", "list", "dishes", "information"]):
        if interaction_count >= 6:
            return True, "Search task completed. The requested information should be visible on the current screen."
    
    # For navigation requests, if we've taken several actions, likely complete
    if any(keyword in user_request_lower for keyword in ["go to", "navigate", "visit", "open"]):
        if interaction_count >= 4:
            return True, "Navigation task completed. You should now be at the requested destination."
    
    # If we see repeated screenshot actions, task is likely complete
    if recent_actions.count("screenshot") >= 3:
        return True, "Task completed based on current screen content. The requested information should be visible."
    
    # If we see repeated type_text actions, task is likely complete
    if len([a for a in recent_actions if "type_text" in a]) >= 3:
        return True, "Task completed. The search query has been entered and results should be visible."
    
    return False, ""


def handle_repeated_action(instruction_response, state, action_desc: str):
    """Handle repeated actions by modifying the instruction to break loops."""
    if state.is_repeating_action(action_desc):
        print(f"Detected repeated action: {action_desc}. Trying alternative approach.")
        
        # Check if we've been repeating the same action too many times
        recent_actions = state.action_history[-5:]  # Last 5 actions
        same_action_count = recent_actions.count(action_desc)
        
        # If we've repeated the same action 3+ times, force task completion
        if same_action_count >= 3:
            print(f"Action repeated {same_action_count} times. Forcing task completion.")
            instruction_response.is_task_complete = True
            instruction_response.completion_message = "Task completed based on current screen state. The requested information should be visible."
            return instruction_response
        
        # Check if we should force completion based on request patterns
        should_complete, completion_msg = should_force_completion(
            state.user_request, state.interaction_count, state.action_history[-10:]
        )
        if should_complete:
            print(f"Forcing completion based on request pattern analysis.")
            instruction_response.is_task_complete = True
            instruction_response.completion_message = completion_msg
            return instruction_response
        
        # Otherwise, try to break the loop with alternative actions
        if instruction_response.action_type == "click_element":
            instruction_response.action_type = "scroll"
            instruction_response.description = "Scroll to find different elements"
            instruction_response.target_element = ""
            instruction_response.scroll_direction = "down"
            instruction_response.reasoning = "Breaking repetitive clicking by scrolling to find new elements"
        elif instruction_response.action_type == "type_text":
            instruction_response.action_type = "press_key"
            instruction_response.description = "Press Enter to execute the search"
            instruction_response.text_content = ""
            instruction_response.key_sequence = ["enter"]
            instruction_response.reasoning = "Breaking repetitive typing by pressing Enter to execute the search"
        elif instruction_response.action_type == "screenshot":
            # If we keep taking screenshots, the task is likely complete
            instruction_response.is_task_complete = True
            instruction_response.completion_message = "Task completed based on current screen content. The requested information should be visible in the screenshot."
            instruction_response.reasoning = "Breaking repetitive screenshots by marking task complete"
    
    return instruction_response 


def detect_human_intervention_needed(last_report, user_request: str, interaction_count: int) -> tuple[bool, str]:
    """
    Detect if human intervention is needed based on screen content and context.
    
    Returns:
        tuple: (needs_intervention, reason)
    """
    if not last_report or not last_report.screenshot_b64:
        return False, ""
    
    # Check for common intervention scenarios based on screen content
    # Note: This is a simplified check - in practice, you'd use computer vision
    # to analyze the actual screenshot content
    
    intervention_keywords = [
        "login", "sign in", "password", "username", "email",
        "captcha", "recaptcha", "verification", "verify",
        "two-factor", "2fa", "authentication", "auth",
        "blocked", "access denied", "forbidden",
        "rate limit", "too many requests",
        "human verification", "robot check",
        "confirm you are human", "prove you're human"
    ]
    
    # Check if any action results or errors indicate need for human intervention
    if last_report.action_result:
        action_result_lower = last_report.action_result.lower()
        if any(keyword in action_result_lower for keyword in intervention_keywords):
            return True, f"Authentication/verification required: {last_report.action_result}"
    
    if last_report.error:
        error_lower = last_report.error.lower()
        if any(keyword in error_lower for keyword in intervention_keywords):
            return True, f"Authentication/verification error: {last_report.error}"
    
    # Check for high interaction count suggesting the agent is stuck
    if interaction_count >= 18:
        return True, f"Agent appears to be stuck after {interaction_count} interactions. Human guidance needed to proceed."
    
    return False, ""


def should_request_human_intervention(state, instruction_response) -> tuple[bool, str]:
    """
    Determine if human intervention should be requested based on current state.
    
    Returns:
        tuple: (should_request, reason)
    """
    # Check if we're stuck in a loop with high interaction count
    if state.interaction_count >= 18:
        return True, f"Agent is stuck after {state.interaction_count} interactions. Human guidance needed to proceed or complete the task."
    
    # Check if we've detected authentication/verification needs
    needs_intervention, reason = detect_human_intervention_needed(
        state.last_executor_report, 
        state.user_request, 
        state.interaction_count
    )
    
    if needs_intervention:
        return True, reason
    
    # Check if we're repeating the same instruction too many times
    if hasattr(state, 'brain_instruction_history') and len(state.brain_instruction_history) >= 3:
        brain_instruction_desc = f"{instruction_response.action_type}:{instruction_response.description}"
        if state.is_repeating_brain_instruction(brain_instruction_desc):
            recent_same_count = state.brain_instruction_history[-5:].count(brain_instruction_desc)
            if recent_same_count >= 3 and state.interaction_count >= 15:
                return True, f"Agent is stuck repeating the same instruction '{brain_instruction_desc}' {recent_same_count} times. Human guidance needed."
    
    return False, "" 