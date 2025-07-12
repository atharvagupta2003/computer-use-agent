# Implementation Roadmap

> **Legend**  
> ⬜︎ pending &nbsp;&nbsp;🔄 in progress &nbsp;&nbsp;✅ done

| # | Task | Owner | Status | Dependencies |
|---|-------|-------|--------|--------------|
| 1 | **Dependencies** – add OpenAI, e2b-python, langgraph, pydantic, python-dateutil to `pyproject.toml` | — | ⬜︎ | — |
| 2 | **configuration.py** – create `AgentConfig` model + `load_config` helper (env var & file) | — | ⬜︎ | 1 |
| 3 | **state.py** – implement `GraphInput`, `ExecutionState`, enums, helper getters | — | ⬜︎ | 2 |
| 4 | **prompts.py** – write planner + executor Jinja templates & constants | — | ⬜︎ | 2 |
| 5 | **utils.py** – time utilities, `retry_async`, logging wrappers | — | ⬜︎ | 2 |
| 6 | **cua/** – wrapper:
|   6.1 | `client.py` – thin OpenAI CUA client (async + retries) | — | ⬜︎ | 1 |
|   6.2 | `sandbox.py` – e2b helpers: create, get, wait_ready | — | ⬜︎ | 1 |
| 7 | **tools.py** – optional LangChain toolset (e.g. `CurrentTimeTool`) | — | ⬜︎ | 5 |
| 8 | **graph.py** – compose LangGraph:
|   8.1 | Node: `sandbox_manager` | — | ⬜︎ | 3,6 |
|   8.2 | Node: `planner` | — | ⬜︎ | 3,4,6 |
|   8.3 | Node: `executor` | — | ⬜︎ | 3,4,6 |
|   8.4 | Node: `end` | — | ⬜︎ | 3 |
| 9 | **tests/** – unit tests for planner parsing, sandbox utils | — | ⬜︎ | 3-8 |
|10 | **integration/** – happy-path scenario (Google search) | — | ⬜︎ | 8 |
|11 | **docs/** – update README + API usage examples | — | ⬜︎ | 8 |

---

## Detailed Steps

### 1. Dependencies
```toml
[tool.poetry.dependencies]
python = "^3.10"
openai = ">=1.0.0"
e2b = "^1.5.6"
langgraph = "^0.0.6"
pydantic = "^2.6"
python-dateutil = "^2.8"
```
*Run*: `poetry add …` or update `pyproject.toml` + `pdm sync`.

### 2. configuration.py
```python
from pydantic import BaseModel, Field, HttpUrl

class AgentConfig(BaseModel):
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    e2b_api_key: str = Field(env="E2B_API_KEY")
    timeout: int = 900  # seconds
    model_planner: str = "gpt-4o-mini"
    model_executor: str = "cua-large-2024-05-17"
    sandbox_template: str = "browser-python"
    # … more

def load_config() -> AgentConfig:
    return AgentConfig()  # env vars picked automatically
```

### 3. state.py
* Implement dataclasses shown in architecture doc.
* Add helper: `is_expiring(self) -> bool`.
* NB: Use `datetime.now(tz=timezone.utc)`.

### 4. prompts.py
* Jinja/format strings with placeholders.
* Provide `format_plan_prompt(user_request)` etc.
* Keep tokens under 2k.

### 5. utils.py
* `async_retry(fn, attempts=3, backoff=…)`.
* `utcnow()` wrapper.
* `to_b64(img_bytes)`.

### 6. cua/
* `sandbox.py`:
  * `create_or_get(config, sandbox_id, url)`.
  * `expires_in(sandbox) -> timedelta`.
* `client.py` wrapper around `openai.beta.computer_vision…` (Pseudo-API for CUA).

### 8. graph.py
```python
from langgraph.graph import StateGraph

graph = StateGraph(ExecutionState)

graph.add_node("sandbox_manager", sandbox_manager)
# … add edges

app = graph.compile()
```
Invoke with `app.invoke(GraphInput(...))`.

### 9-10. Tests
* Use `pytest` w/ `pytest-asyncio`.
* Mock network via `respx`.

---

## Milestone Timeline (optimistic)
1. Core plumbing (tasks 1-5) – **1 day**
2. Sandbox & CUA wrappers – **1 day**
3. Nodes + graph – **2 days**
4. Tests & docs – **1 day**

Total **≈5 days** of focused work. 