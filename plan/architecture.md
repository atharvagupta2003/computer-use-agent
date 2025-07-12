# Computer Use Agent – Architecture & Workflow

## 1. Goal
Build a smart, fast agent that automates tasks in a disposable **e2b** virtual machine ("sandbox") using OpenAI **Computer-Use Assistant** (CUA) models and orchestrated by a **LangGraph** state machine.

*   Re-use an existing sandbox when `sandbox_id` & `sandbox_url` are supplied.
*   Spin up a fresh sandbox when none is provided (respecting `timeout` and other config limits).
*   Split user requests into minimal actionable steps ("planner" node) to keep CUA prompts small & deterministic.
*   Continually validate sandbox health / expiry and gracefully end when < 60 s remain.
*   Detect and surface special conditions (`ambiguity`, `human-intervention`, `sensitive-action`, `problem`).
*   Persist rich state (screenshots, intermediate outputs) for observability & debugging.

---

## 2. High-Level Components
| Module | Responsibility |
|--------|----------------|
| `configuration.py` | **Static** settings & Pydantic schema loaded at start-up (API keys, sandbox template, timeout, model names, etc.). |
| `state.py` | **Dynamic** LangGraph state definitions (input, working, and output objects). |
| `prompts.py` | All LLM prompt templates (planner, executor, system messages). |
| `cua/` | Typed wrappers around OpenAI CUA API (screenshot, action execution, polling helpers). |
| `utils.py` | Generic helpers (time, retries, serialization, logging). |
| `tools.py` | Optional LangChain tools (e.g. HTTP, file read) exposed to planner LLM. |
| `graph.py` | Builds & runs the LangGraph with concrete node functions. |

---

## 3. LangGraph Workflow
```
┌──────────────┐   healthy      ┌─────────────┐    plan        ┌────────────┐
│  sandbox_mgr │ ───────────▶ │   planner   │ ───────────▶ │  executor  │
└──────────────┘  expiry<60s    └─────┬──────┘  all steps     └─────┬─────┘
        ▲   │                        │  │  step-done             │  │
        │   └──────── end ───────────┘  └───────── next / end ───┘  │
        └────────────────────────────────────────────────────────────┘
```

1. **sandbox_mgr** – Ensure a sandbox is running & not about to expire.<br/>
   • If it would expire in < 60 s ⇒ output=`problem` → **end**.<br/>
   • Otherwise pass state forward.
2. **planner** – Small LLM that:
   • Parses `user_request` → high-level plan (ordered `steps`).<br/>
   • Selects `application_platform` (`google-chrome`, `firefox`, `vscode`).<br/>
   • If request is ambiguous ⇒ output=`ambiguity` → **end**.
3. **executor** – Loop over `steps[index]` until done.
   • Uses CUA model to screenshot → reason → act.
   • After each action: screenshot → check success.
   • If human-verification or sensitive action detected ⇒ output respective flag & break.
4. **end** – Aggregates final result & returns to caller.

---

## 4. State Design (`state.py`)
```
class GraphInput(BaseModel):
    user_request: str
    sandbox_id: Optional[str] = None
    sandbox_url: Optional[HttpUrl] = None

class ExecutionState(BaseModel):
    plan: List[str] = []
    current_step: int = 0
    application_platform: Optional[Literal["google-chrome", "firefox", "vscode"]]
    sandbox_id: str
    sandbox_url: HttpUrl
    sandbox_expires_at: datetime
    screenshots: List[bytes] = []  # base64 PNGs
    last_executor_output: Optional[str]
    status: Literal[
        "in_progress", "ambiguity", "human-intervention", "sensitive-action", "problem", "done"
    ] = "in_progress"
```
(Add validation utilities & convenience getters.)

---

## 5. Prompt Strategy (`prompts.py`)
1. **Planner System** – "You are a task planner… produce numbered JSON list of concise steps … choose platform…"
2. **Executor System** – "You have the following step: <step>. First screenshot then …"
3. **Safety / Guard** messages for sensitive actions.

---

## 6. Error & Control Codes
| Code | Trigger | Consumer |
|------|---------|----------|
| `ambiguity` | Planner cannot confidently derive plan. | Caller / UI |
| `human-intervention` | CAPTCHA, login, puzzle, etc. | Human operator |
| `sensitive-action` | Delete files, send emails, purchase, etc. | Human operator |
| `problem` | Sandbox expired, API failure, etc. | Planner or top-level caller |
| `done` | All steps executed successfully. | – |

---

## 7. Interaction with **e2b** Sandbox
*   **Create** – `e2b.Client().sandboxes.create(template=…, timeout=…)` → returns `sandbox_id`, `url`, `expires_at`.
*   **Check status** – `client.sandboxes.get(id)` & metrics call provides `expires_at`.
*   **Execute commands** – Browser/VSCode launched via internal helper that opens web sockets via `sandbox_url`.

Expiration check occurs: `expires_at - now < 60s` ⇒ `problem`.

---

## 8. Performance Considerations
*   Keep planner prompts ≤2-3 sentences per step (< 4 k-tokens total).
*   Deduplicate screenshots (store hash).
*   Use async I/O when calling CUA & e2b APIs.

---

## 9. Security
*   Never expose user credentials in prompts.
*   Sensitive actions gated by `sensitive-action` status.
*   Sandbox is isolated & auto-deleted at timeout.

---

## 10. Next Steps
See `implementation_steps.md` for sequenced coding tasks. 