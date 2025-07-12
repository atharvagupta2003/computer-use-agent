"""Map CUA tool-call **actions** onto e2b_desktop sandbox operations."""

from __future__ import annotations

import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .sandbox import get_desktop


# Mapping of CUA key names → e2b_desktop friendly strings or behaviours.
_CUA_KEY_TO_DESKTOP = {
    "/": "slash",
    "\\": "backslash",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "backspace": "BackSpace",
    "capslock": "Caps_Lock",
    "delete": "Delete",
    "end": "End",
    "enter": "Return",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "pagedown": "Page_Down",
    "pageup": "Page_Up",
    "tab": "Tab",
    "ctrl": "Control_L",
    "alt": "Alt_L",
    "shift": "Shift_L",
    "cmd": "Meta_L",
    "win": "Meta_L",
    "meta": "Meta_L",
    "space": "space",
}

# Set up logging
logger = logging.getLogger("cua.actions")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def execute_action(action: str | Dict[str, Any], sandbox_id: str) -> Dict[str, Any]:
    """Execute an action that can be either a string or a structured dict.
    
    This function handles both formats:
    - String actions (from older format): "Click the search box"
    - Dict actions (structured format): {"action": "click", "coordinate": [x, y]}
    """
    if isinstance(action, str):
        # For string actions, we need to parse the intent and convert to structured format
        # This is a simplified parser - in practice, you might want more sophisticated parsing
        action_lower = action.lower()
        
        if "click" in action_lower:
            # For now, we'll just return a generic click action
            # In practice, you'd want to extract coordinates from the string
            return {"success": False, "error": "String actions not fully supported - use structured format"}
        elif "type" in action_lower:
            # Extract text to type
            # This is a simplified extraction
            return {"success": False, "error": "String actions not fully supported - use structured format"}
        else:
            return {"success": False, "error": f"Unknown string action: {action}"}
    
    # Handle structured dict actions
    return perform_action(sandbox_id, action)


def perform_action(sandbox_id: str, action: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Execute a single *action* against the desktop sandbox identified by *sandbox_id*.
    
    Returns a dict with the result of the action, including success status and any error.
    """
    result = {
        "success": False,
        "action_type": action.get("type"),
        "error": None,
        "details": {}
    }

    try:
        desktop = get_desktop(sandbox_id)
        if desktop is None:
            raise RuntimeError(f"Desktop sandbox {sandbox_id} not found in cache.")

        action_type = action.get("type")
        logger.info(f"Executing action: {action_type}")

        if action_type == "click":
            _click(desktop, action)
            result["details"]["coordinates"] = (action.get("x"), action.get("y"))
            
        elif action_type == "double_click":
            _double_click(desktop, action)
            result["details"]["coordinates"] = (action.get("x"), action.get("y"))
            
        elif action_type == "move":
            desktop.move_mouse(action.get("x"), action.get("y"))  # type: ignore[attr-defined]
            result["details"]["coordinates"] = (action.get("x"), action.get("y"))
            
        elif action_type == "drag":
            _drag(desktop, action)
            path = action.get("path", [])
            if path:
                result["details"]["start"] = (path[0].get("x"), path[0].get("y"))
                result["details"]["end"] = (path[-1].get("x"), path[-1].get("y"))
                
        elif action_type == "keypress":
            keys = _keypress(desktop, action)
            result["details"]["keys"] = keys
            
        elif action_type == "type":
            text = action.get("text", "")
            _type_text(desktop, text)
            result["details"]["text"] = text
            
        elif action_type == "scroll":
            _scroll(desktop, action)
            result["details"]["delta_x"] = action.get("scroll_x", 0)
            result["details"]["delta_y"] = action.get("scroll_y", 0)
            
        elif action_type == "wait":
            duration = action.get("duration", 2)
            time.sleep(duration)
            result["details"]["duration"] = duration
            
        elif action_type == "screenshot":
            # Use screenshot method if available
            if hasattr(desktop, "screenshot"):
                desktop.screenshot()  # type: ignore[attr-defined]
                
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
        
        # Add a small delay after each action to allow the UI to update
        time.sleep(0.1)
        result["success"] = True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Action failed: {error_msg}")
        result["success"] = False
        result["error"] = error_msg
    
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _click(desktop, action):  # noqa: D401 – helper
    x, y = action.get("x"), action.get("y")
    button = action.get("button", "left")
    
    # Move mouse first
    desktop.move_mouse(x, y)  # type: ignore[attr-defined]
    
    # Small delay to ensure mouse has moved
    time.sleep(0.05)
    
    # Then click based on button type
    if button == "right":
        if hasattr(desktop, "right_click"):
            desktop.right_click()  # type: ignore[attr-defined]
        else:
            # Fallback using mouse_press/mouse_release if available
            if hasattr(desktop, "mouse_press") and hasattr(desktop, "mouse_release"):
                desktop.mouse_press("right")  # type: ignore[attr-defined]
                time.sleep(0.05)
                desktop.mouse_release("right")  # type: ignore[attr-defined]
    elif button == "middle":
        if hasattr(desktop, "middle_click"):
            desktop.middle_click()  # type: ignore[attr-defined]
        else:
            # Fallback using mouse_press/mouse_release if available
            if hasattr(desktop, "mouse_press") and hasattr(desktop, "mouse_release"):
                desktop.mouse_press("middle")  # type: ignore[attr-defined]
                time.sleep(0.05)
                desktop.mouse_release("middle")  # type: ignore[attr-defined]
    else:
        if hasattr(desktop, "left_click"):
            desktop.left_click()  # type: ignore[attr-defined]
        else:
            # Fallback using mouse_press/mouse_release if available
            if hasattr(desktop, "mouse_press") and hasattr(desktop, "mouse_release"):
                desktop.mouse_press("left")  # type: ignore[attr-defined]
                time.sleep(0.05)
                desktop.mouse_release("left")  # type: ignore[attr-defined]


def _double_click(desktop, action):  # noqa: D401 – helper
    x, y = action.get("x"), action.get("y")
    
    # Move mouse first
    desktop.move_mouse(x, y)  # type: ignore[attr-defined]
    
    # Small delay to ensure mouse has moved
    time.sleep(0.05)
    
    # Then double click
    if hasattr(desktop, "double_click"):
        desktop.double_click()  # type: ignore[attr-defined]
    else:
        # Fallback to two left clicks with a short delay
        if hasattr(desktop, "left_click"):
            desktop.left_click()  # type: ignore[attr-defined]
            time.sleep(0.1)  # Short delay between clicks for double-click
            desktop.left_click()  # type: ignore[attr-defined]
        else:
            # Fallback using mouse_press/mouse_release if available
            if hasattr(desktop, "mouse_press") and hasattr(desktop, "mouse_release"):
                desktop.mouse_press("left")  # type: ignore[attr-defined]
                desktop.mouse_release("left")  # type: ignore[attr-defined]
                time.sleep(0.1)
                desktop.mouse_press("left")  # type: ignore[attr-defined]
                desktop.mouse_release("left")  # type: ignore[attr-defined]


def _drag(desktop, action):  # noqa: D401 – helper
    path: List[Dict[str, int]] = action.get("path", [])
    if not path:
        return
    
    if len(path) < 2:
        return
    
    start = path[0]
    end = path[-1]
    
    # Use drag method if available
    if hasattr(desktop, "drag"):
        desktop.drag((start.get("x"), start.get("y")), (end.get("x"), end.get("y")))  # type: ignore[attr-defined]
        return
    
    # Otherwise fallback to move + press + move + release
    desktop.move_mouse(start.get("x"), start.get("y"))  # type: ignore[attr-defined]
    time.sleep(0.1)  # Ensure mouse has moved
    
    if hasattr(desktop, "mouse_press"):
        desktop.mouse_press("left")  # type: ignore[attr-defined]
        time.sleep(0.1)  # Short delay after press
        
        # Move through path points
        for point in path[1:]:
            desktop.move_mouse(point.get("x"), point.get("y"))  # type: ignore[attr-defined]
            time.sleep(0.05)  # Small delay between moves for smoother dragging
            
        time.sleep(0.1)  # Short delay before release
        if hasattr(desktop, "mouse_release"):
            desktop.mouse_release("left")  # type: ignore[attr-defined]
    else:
        # Very basic fallback
        desktop.left_click()  # type: ignore[attr-defined]
        time.sleep(0.1)
        desktop.move_mouse(end.get("x"), end.get("y"))  # type: ignore[attr-defined]
        time.sleep(0.1)
        desktop.left_click()  # type: ignore[attr-defined]


def _keypress(desktop, action) -> List[str]:  # noqa: D401 – helper
    keys = action.get("keys", [])
    pressed_keys = []
    
    for key in keys:
        # Convert key to lowercase for e2b_desktop API
        key_lower = key.lower()
        
        # Map some common key names to e2b_desktop format
        key_mapping = {
            "return": "enter",
            "ret": "enter", 
            "enter": "enter",
            "space": "space",
            "tab": "tab",
            "backspace": "backspace",
            "delete": "delete",
            "escape": "escape",
            "esc": "escape",
            "shift": "shift",
            "ctrl": "ctrl",
            "alt": "alt",
            "cmd": "cmd",
            "meta": "cmd",
            "arrowup": "up",
            "arrowdown": "down",
            "arrowleft": "left",
            "arrowright": "right",
            "up": "up",
            "down": "down", 
            "left": "left",
            "right": "right",
            "home": "home",
            "end": "end",
            "pageup": "page_up",
            "pagedown": "page_down",
        }
        
        # Get the correct key name for e2b_desktop
        desktop_key = key_mapping.get(key_lower, key_lower)
        pressed_keys.append(desktop_key)
        
        try:
            # Use the e2b_desktop press method
            if hasattr(desktop, "press"):
                desktop.press(desktop_key)  # type: ignore[attr-defined]
                print(f"Pressed key: {desktop_key}")
            else:
                print(f"Warning: desktop.press method not available for key: {desktop_key}")
                
        except Exception as e:
            print(f"Error pressing key {desktop_key}: {e}")
            # Try alternative methods if available
            if hasattr(desktop, "press_key"):
                try:
                    desktop.press_key(desktop_key)  # type: ignore[attr-defined]
                    print(f"Pressed key using press_key: {desktop_key}")
                except Exception as e2:
                    print(f"Error with press_key for {desktop_key}: {e2}")
    
    return pressed_keys


def _type_text(desktop, text: str) -> None:
    """Type text efficiently based on available methods."""
    if not text:
        return
        
    # Use the most efficient method available
    if hasattr(desktop, "write"):
        desktop.write(text)  # type: ignore[attr-defined]
    elif hasattr(desktop, "type_text"):
        desktop.type_text(text)  # type: ignore[attr-defined]
    else:
        # Fallback to pressing keys one by one
        for char in text:
            if hasattr(desktop, "press"):
                desktop.press(char)  # type: ignore[attr-defined]
                time.sleep(0.01)  # Small delay to avoid overwhelming the system


def _scroll(desktop, action):  # noqa: D401 – helper
    dx = action.get("scroll_x", 0)
    dy = action.get("scroll_y", 0)
    x = action.get("x", 0)
    y = action.get("y", 0)
    
    # Move mouse to position first if coordinates are provided
    if x != 0 or y != 0:
        desktop.move_mouse(x, y)  # type: ignore[attr-defined]
        time.sleep(0.05)
    
    if hasattr(desktop, "scroll"):
        # Determine direction and amount
        if dy != 0:
            direction = "down" if dy > 0 else "up"
            amount = abs(dy) // 20 or 1  # Normalize the amount, minimum 1
            desktop.scroll(direction=direction, amount=amount)  # type: ignore[attr-defined]
        elif dx != 0:
            direction = "right" if dx > 0 else "left"
            amount = abs(dx) // 20 or 1  # Normalize the amount, minimum 1
            desktop.scroll(direction=direction, amount=amount)  # type: ignore[attr-defined]
    else:
        # No scroll method available, try to use a more generic method if exists
        pass


def capture_screenshot(sandbox_id: str) -> bytes | None:  # noqa: D401
    """Capture a screenshot from the sandbox.
    
    Returns bytes of the screenshot or None if not available.
    """
    desktop = get_desktop(sandbox_id)
    if desktop is None:
        logger.error(f"Cannot capture screenshot: sandbox {sandbox_id} not found")
        return None
        
    # Try different screenshot methods
    try:
        if hasattr(desktop, "capture_screen"):
            return desktop.capture_screen()  # type: ignore[attr-defined]
        elif hasattr(desktop, "screenshot"):
            # Check if it returns bytes
            try:
                screenshot = desktop.screenshot(format="bytes")  # type: ignore[attr-defined]
                if isinstance(screenshot, (bytes, bytearray)):
                    return screenshot
            except Exception as e:
                logger.warning(f"Error capturing screenshot with format=bytes: {e}")
                # Try without format parameter
                screenshot = desktop.screenshot()  # type: ignore[attr-defined]
                if isinstance(screenshot, (bytes, bytearray)):
                    return screenshot
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
    
    return None 