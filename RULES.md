# Project Rules and Workflow (Python FastAPI + Transformers)

These rules are binding for every change. Keep code, docs, and behavior synchronized at all times.

Files referenced below:
- [README.md](README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TODO.md](TODO.md)
- [CLAUDE.md](CLAUDE.md)
- [.env.example](.env.example)
- [.gitignore](.gitignore)
- [requirements.txt](requirements.txt)
- [Python.main()](main.py:1)

## 1) Documentation rules (must-do on every change)

Always update documentation when code or behavior changes.

Minimum documentation checklist:
- What changed and where (filenames, sections, or callable links like [Python.function chat_completions()](main.py:591)).
- Why the change was made (problem or requirement).
- How to operate or verify (commands, endpoints, examples).
- Follow-ups or known limitations.

Where to update:
- Operator-facing: [README.md](README.md)
- Developer-facing: [CLAUDE.md](CLAUDE.md) (rationale, alternatives, caveats)
- Architecture or flows: [ARCHITECTURE.md](ARCHITECTURE.md)
- Tasks and statuses: [TODO.md](TODO.md)

Never skip documentation. If a change is reverted, document the revert.

## 2) Git discipline (mandatory)

- Always use Git. Every change or progress step MUST be committed and pushed.
  - Windows CMD example:
    - git add .
    - git commit -m "type(scope): short description"
    - git push
- No exceptions. If no remote exists, commit locally and configure a remote as soon as possible. Record any temporary push limitations in [README.md](README.md) and [CLAUDE.md](CLAUDE.md), but commits are still required locally.
- Commit style:
  - Conventional types: chore, docs, feat, fix, refactor, perf, test, build, ci
  - Keep commits small and atomic (one concern per commit).
  - Reference important files in the commit body, for example: updated [Python.function chat_completions()](main.py:591), [README.md](README.md).
- After updating code or docs, commit immediately. Do not batch unrelated changes.

## 3) Large artifacts policy (.gitignore)

Never commit large/generated artifacts. Keep the repository lean and reproducible.

Must be ignored:
- models/ (downloaded by HF/Transformers cache or tools at runtime)
- .venv/, venv/
- __pycache__/
- .cache/
- uploads/, data/, tmp/

See [.gitignore](.gitignore) and extend as needed for new generated outputs. If you add ignores, document the rationale in [CLAUDE.md](CLAUDE.md).

## 4) Model policy (Hugging Face / Transformers)

Target default model:
- Qwen/Qwen3-VL-2B-Thinking (Transformers; multimodal).

Rules:
- Use Hugging Face Transformers (AutoModelForCausalLM + AutoProcessor) with trust_remote_code=True.
- Do not commit model weights or caches. Let from_pretrained() download to local caches.
- Handle authentication for gated models via HF_TOKEN in [.env.example](.env.example).
- The server must remain OpenAI-compatible at /v1/chat/completions and support multimodal inputs (text, images, videos).
- Keep configuration via environment variables (see [Python.os.getenv()](main.py:67)).

## 5) API contract

Provide an OpenAI-compatible endpoint:
- POST /v1/chat/completions

Minimum behavior:
- Accept model and messages per OpenAI schema (we honor messages; model is informational since server is pinned via env).
- Non-streaming JSON response.
- Streaming SSE response when body.stream=true:
  - Emit OpenAI-style chat.completion.chunk deltas.
  - Include SSE id lines "session_id:index" to support resume via Last-Event-ID.

Resume semantics:
- Client provides a session_id (or server generates one).
- Client may reconnect and send Last-Event-ID: session_id:index to replay missed chunks.
- Session data can be persisted (SQLite) if enabled.

Manual cancel (custom extension):
- POST /v1/cancel/{session_id} cancels a streaming generation.
- Note: Not part of legacy OpenAI Chat Completions spec. It mirrors the spirit of the newer OpenAI Responses API cancel endpoint.

All endpoints must validate inputs, handle timeouts/failures, and return structured JSON errors.

## 6) Streaming, persistence, and cancellation

- Streaming is implemented via SSE in [Python.function chat_completions()](main.py:591) with token iteration in [Python.function infer_stream](main.py:375).
- In-memory ring buffer per session and optional SQLite persistence for replay across restarts:
  - In-memory: [Python.class _SSESession](main.py:435), [Python.class _SessionStore](main.py:449)
  - SQLite: [Python.class _SQLiteStore](main.py:482) (enabled with PERSIST_SESSIONS=1)
- Resume:
  - Uses SSE id "session_id:index" and Last-Event-ID header (or ?last_event_id=...).
- Auto-cancel on disconnect:
  - If all clients disconnect, generation is cancelled after CANCEL_AFTER_DISCONNECT_SECONDS (default 3600 sec). Configurable via env.
  - Cooperative stop via StoppingCriteria in [Python.function infer_stream](main.py:375).
- Manual cancel:
  - [Python.function cancel_session](main.py:792) to stop a session on demand.

## 7) Logging and error handling

- Log key lifecycle stages (startup, model load, stream start/stop, resume).
- Redact sensitive fields (e.g., tokens, credentials).
- User errors → 400; model-not-ready → 503; unexpected failures → 500.
- Optionally add structured logging and request IDs in a follow-up.

## 8) Architecture documentation

Keep [ARCHITECTURE.md](ARCHITECTURE.md) authoritative for:
- Startup flow and lazy model load
- Multimodal preprocessing (images/videos)
- Streaming, resume, persistence, and cancellation flows
- Error/timeout handling
- Extensibility (persistence strategies, cancellation hooks, scaling patterns)

Update when code paths or data flows change.

## 9) TODO hygiene

Track all planned work in [TODO.md](TODO.md):
- Update statuses immediately when tasks start/complete.
- Add newly discovered tasks as soon as they are identified.
- Keep TODO focused, scoped, and prioritized.

## 10) Operational requirements and environment

Required:
- Python: >= 3.10
- pip
- PyTorch: install a wheel matching platform/CUDA (see [requirements.txt](requirements.txt) notes)

Recommended:
- GPU with sufficient VRAM for the chosen model
- Windows 11 supported; Linux/macOS should also work

Environment variables (see [.env.example](.env.example)):
- PORT=3000
- MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Thinking
- HF_TOKEN=
- MAX_TOKENS=256
- TEMPERATURE=0.7
- MAX_VIDEO_FRAMES=16
- DEVICE_MAP=auto
- TORCH_DTYPE=auto
- PERSIST_SESSIONS=1|0, SESSIONS_DB_PATH, SESSIONS_TTL_SECONDS
- CANCEL_AFTER_DISCONNECT_SECONDS=3600 (0 to disable)

## 11) File responsibilities overview

- Server: [Python.main()](main.py:1)
  - API routing, model singleton, inference, streaming, resume, cancel
- Docs: [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md)
- Dev log: [CLAUDE.md](CLAUDE.md)
- Tasks: [TODO.md](TODO.md)
- Config template: [.env.example](.env.example)
- Dependencies: [requirements.txt](requirements.txt)
- Ignores: [.gitignore](.gitignore)

## 12) Workflow example (single iteration)

1) Make a small, isolated change (e.g., enable SQLite persistence).
2) Update docs:
   - [CLAUDE.md](CLAUDE.md): what/why/how
   - [README.md](README.md): operator usage changes
   - [ARCHITECTURE.md](ARCHITECTURE.md): persistence/resume flow
   - [TODO.md](TODO.md): status changes
3) Commit and push:
   - git add .
   - git commit -m "feat(stream): add SQLite persistence for SSE resume"
   - git push
4) Verify locally; record any issues or follow-ups in [CLAUDE.md](CLAUDE.md).

## 13) Compliance checklist (pre-merge / pre-push)

- Code runs locally (uvicorn main:app …).
- Docs updated ([README.md](README.md), [CLAUDE.md](CLAUDE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [TODO.md](TODO.md)).
- No large artifacts added to git.
- Commit message follows conventional style.
- Endpoint contract honored (including streaming/resume semantics and cancel extension).
