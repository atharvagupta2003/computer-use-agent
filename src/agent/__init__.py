"""Top-level import convenience for the Computer-Use Agent package."""

from .graph import workflow_graph as _graph
from .state import GraphInput, ExecutionState

__all__ = ["create_agent", "GraphInput", "ExecutionState"]

def create_agent():  # noqa: D401 – simple factory
    """Return a configured LangGraph callable ready for invocation."""

    return _graph
