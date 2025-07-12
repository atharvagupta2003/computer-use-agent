"""Thin async wrapper around OpenAI's *Computer Use* model family.

The OpenAI Python SDK exposes the computer use models via the regular chat
completion endpoint with special tool binding. This wrapper standardises the
request / response schema for the executor node and encapsulates retries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

import openai

from ..utils import async_retry

__all__ = ["take_action"]


@async_retry(attempts=3)
async def take_action(
    messages: list[dict], *, model: str, api_key: str, environment: str = "browser", previous_response_id: Optional[str] = None
) -> Dict[str, Any]:
    """Call the OpenAI chat/completion endpoint with retries.

    Parameters
    ----------
    messages
        List of dicts in OpenAI chat format.
    model
        Name of the model to use (e.g. gpt-4o).
    api_key
        OpenAI API key.
    environment
        Environment type (browser, ubuntu, windows).
    previous_response_id
        Previous response ID for conversation continuity (not used in chat completions).

    Returns
    -------
    Dict[str, Any]
        Raw response from the OpenAI API.
    """
    # Create OpenAI client
    client = openai.AsyncOpenAI(api_key=api_key)
    
    # Define the computer use tool
    computer_tool = {
        "type": "function",
        "function": {
            "name": "computer",
            "description": "Use a computer to perform actions like clicking, typing, and taking screenshots",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["screenshot", "click", "type", "keypress", "scroll", "wait"],
                        "description": "The action to perform"
                    },
                    "coordinate": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The [x, y] coordinate for click actions"
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to type"
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The keys to press"
                    },
                    "scroll_direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "The direction to scroll"
                    },
                    "scroll_amount": {
                        "type": "number",
                        "description": "The amount to scroll"
                    }
                },
                "required": ["action"]
            }
        }
    }
    
    # Create request parameters
    params = {
        "model": model,
        "messages": messages,
        "tools": [computer_tool],
        "tool_choice": "auto"
    }

    response = await client.chat.completions.create(**params)
    
    # Convert response to dict format
    return response.model_dump() 