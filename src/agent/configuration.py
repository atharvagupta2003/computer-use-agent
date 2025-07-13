"""Configuration utilities for the Computer-Use Agent.

All static settings are centralised here. Values are primarily sourced from
environment variables to keep the public API surface minimal and avoid passing
secrets across function boundaries.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationInfo
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

__all__ = ["AgentConfig", "get_config"]


class AgentConfig(BaseModel):
    """Static configuration for the Computer-Use Agent.

    Most fields are optional and will fall back to sane defaults when omitted.
    You can supply values either via constructor kwargs *or* via environment
    variables whose names are documented alongside each field.
    """

    # ---------------------------------------------------------------------
    # Credential & API keys
    # ---------------------------------------------------------------------
    openai_api_key: str = Field(  # noqa: D401 – Provide docstring in comment
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="Secret key for the OpenAI API. Set via $OPENAI_API_KEY.",
    )
    anthropic_api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""),
        description="Secret key for the Anthropic API. Set via $ANTHROPIC_API_KEY.",
    )
    e2b_api_key: str = Field(
        default_factory=lambda: os.getenv("E2B_API_KEY", ""),
        description="Secret key for the e2b API. Set via $E2B_API_KEY.",
    )

    # ---------------------------------------------------------------------
    # Model names
    # ---------------------------------------------------------------------
    model_planner: str = Field(
        default=os.getenv("CUA_PLANNER_MODEL", "gpt-4o-mini"),
        description="Chat model used for the planner node.",
    )
    model_executor: str = Field(
        default=os.getenv("CUA_EXECUTOR_MODEL", "gpt-4o"),
        description="Computer-Use model used for the executor node.",
    )
    brain_provider: str = Field(
        default=os.getenv("CUA_BRAIN_PROVIDER", "anthropic"),
        description="LLM provider for the brain node. Options: 'openai' or 'anthropic'.",
    )
    brain_model: str = Field(
        default=os.getenv("CUA_BRAIN_MODEL", "claude-3-5-sonnet-20241022"),
        description="Model name for the brain node.",
    )

    # ---------------------------------------------------------------------
    # Sandbox settings
    # ---------------------------------------------------------------------
    sandbox_template: str = Field(
        default=os.getenv("CUA_SANDBOX_TEMPLATE", "browser-python"),
        description="Name of the e2b sandbox template to launch when a sandbox is required.",
    )
    sandbox_timeout: int = Field(
        default_factory=lambda: int(os.getenv("CUA_SANDBOX_TIMEOUT", "900")),
        description="Maximum sandbox lifetime in seconds (default 900 = 15 min).",
    )

    # If the caller already has a sandbox they want to re-use, they can
    # populate these two fields in the initial GraphInput instead of relying
    # on configuration.
    sandbox_id: Optional[str] = Field(default=None, description="Pre-existing sandbox id (optional).")
    sandbox_url: Optional[HttpUrl] = Field(default=None, description="Pre-existing sandbox URL (optional).")

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------
    iteration_limit: int = Field(
        default_factory=lambda: int(os.getenv("CUA_ITERATION_LIMIT", "30")),
        description="Safety cap on max executor iterations in a single invocation.",
    )
    
    recursion_limit: int = Field(
        default_factory=lambda: int(os.getenv("CUA_RECURSION_LIMIT", "50")),
        description="LangGraph recursion limit for brain-executor feedback loops.",
    )

    # ---------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------
    @field_validator("e2b_api_key", mode="before")
    def _not_empty(cls, v: str, info: ValidationInfo):  # noqa: D401 – internal helper
        if not v:
            raise ValueError(
                f"{info.field_name} is required. Set it via environment variable or pass it explicitly."
            )
        return v


@lru_cache(maxsize=1)
def get_config() -> AgentConfig:
    """Return a **cached** instance of `AgentConfig`.

    This ensures expensive env parsing occurs only once per process.
    """

    return AgentConfig()
