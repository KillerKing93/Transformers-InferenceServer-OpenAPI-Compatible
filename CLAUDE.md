# CLAUDE Technical Log and Decisions (Python FastAPI + Transformers)
## Progress Log — 2025-10-23 (Asia/Jakarta)

- Migrated stack from Node.js/llama.cpp to Python + FastAPI + Transformers
  - New server: [main.py](main.py)
  - Default model: Qwen/Qwen3-VL-2B-Thinking via Transformers with trust_remote_code
- Implemented endpoints
  - Health: [Python.app.get()](main.py:577)
  - OpenAI-compatible Chat Completions (non-stream + SSE): [Python.app.post()](main.py:591)
  - Manual cancel (custom extension): [Python.app.post()](main.py:792)
- Multimodal support
  - OpenAI-style messages mapped in [Python.function build_mm_messages](main.py:251)
  - Image loader: [Python.function load_image_from_any](main.py:108)
  - Video loader (frame sampling): [Python.function load_video_frames_from_any](main.py:150)
- Streaming + resume + persistence
  - SSE with session_id + Last-Event-ID
  - In-memory session ring buffer: [Python.class _SSESession](main.py:435), manager [Python.class _SessionStore](main.py:449)
  - Optional SQLite persistence: [Python.class _SQLiteStore](main.py:482) with replay across restarts
- Cancellation
  - Auto-cancel after all clients disconnect for CANCEL_AFTER_DISCONNECT_SECONDS, timer wiring in [Python.function chat_completions](main.py:733), cooperative stop in [Python.function infer_stream](main.py:375)
  - Manual cancel API: [Python.function cancel_session](main.py:792)
- Configuration and dependencies
  - Env template updated: [.env.example](.env.example) with MODEL_REPO_ID, PERSIST_SESSIONS, SESSIONS_DB_PATH, SESSIONS_TTL_SECONDS, CANCEL_AFTER_DISCONNECT_SECONDS, etc.
  - Python deps: [requirements.txt](requirements.txt)
  - Git ignores for Python + artifacts: [.gitignore](.gitignore)
- Documentation refreshed
  - Operator docs: [README.md](README.md) including SSE resume, SQLite, cancel API
  - Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) aligned to Python flows
  - Rules: [RULES.md](RULES.md) updated — Git usage is mandatory
- Legacy removal
  - Deleted Node files and scripts (index.js, package*.json, scripts/) as requested

Suggested Git commit series (run in order)
- git add .
- git commit -m "feat(server): add FastAPI OpenAI-compatible /v1/chat/completions with Qwen3-VL [Python.main()](main.py:1)"
- git commit -m "feat(stream): SSE streaming with session_id resume and in-memory sessions [Python.function chat_completions()](main.py:591)"
- git commit -m "feat(persist): SQLite-backed replay for SSE sessions [Python.class _SQLiteStore](main.py:482)"
- git commit -m "feat(cancel): auto-cancel after disconnect and POST /v1/cancel/{session_id} [Python.function cancel_session](main.py:792)"
- git commit -m "docs: update README/ARCHITECTURE/RULES for Python stack and streaming resume"
- git push

Verification snapshot
- Non-stream text works via [Python.function infer](main.py:326)
- Streaming emits chunks and ends with [DONE]
- Resume works with Last-Event-ID; persists across restart when PERSIST_SESSIONS=1
- Manual cancel stops generation; auto-cancel triggers after disconnect threshold


This is the developer-facing changelog and design rationale for the Python migration. Operator docs live in [README.md](README.md); architecture details in [ARCHITECTURE.md](ARCHITECTURE.md); rules in [RULES.md](RULES.md); task tracking in [TODO.md](TODO.md).

Key source file references
- Server entry: [Python.main()](main.py:807)
- Health endpoint: [Python.app.get()](main.py:577)
- Chat Completions endpoint (non-stream + SSE): [Python.app.post()](main.py:591)
- Manual cancel endpoint (custom): [Python.app.post()](main.py:792)
- Engine (Transformers): [Python.class Engine](main.py:231)
- Multimodal mapping: [Python.function build_mm_messages](main.py:251)
- Image loader: [Python.function load_image_from_any](main.py:108)
- Video loader: [Python.function load_video_frames_from_any](main.py:150)
- Non-stream inference: [Python.function infer](main.py:326)
- Streaming inference + stopping criteria: [Python.function infer_stream](main.py:375)
- In-memory sessions: [Python.class _SSESession](main.py:435), [Python.class _SessionStore](main.py:449)
- SQLite persistence: [Python.class _SQLiteStore](main.py:482)

Summary of the migration
- Replaced the Node.js/llama.cpp stack with a Python FastAPI server that uses Hugging Face Transformers for Qwen3-VL multimodal inference.
- Exposes an OpenAI-compatible /v1/chat/completions endpoint (non-stream and streaming via SSE).
- Supports text, images, and videos:
  - Messages can include array parts such as "text", "image_url" / "input_image" (base64), "video_url" / "input_video" (base64).
  - Images are decoded to PIL in [Python.function load_image_from_any](main.py:108).
  - Videos are read via imageio.v3 (preferred) or OpenCV, sampled to up to MAX_VIDEO_FRAMES in [Python.function load_video_frames_from_any](main.py:150).
- Streaming includes resumability with session_id + Last-Event-ID:
  - In-memory ring buffer: [Python.class _SSESession](main.py:435)
  - Optional SQLite persistence: [Python.class _SQLiteStore](main.py:482)
- Added a manual cancel endpoint (custom) and implemented auto-cancel after disconnect.

Why Python + Transformers?
- Qwen3-VL-2B-Thinking is published for Transformers and includes multimodal processors (preprocessor_config.json, video_preprocessor_config.json, chat_template.json). Python + Transformers is the first-class path.
- trust_remote_code=True allows the model repo to provide custom processing logic and templates, used in [Python.class Engine](main.py:231) via AutoProcessor/AutoModelForCausalLM.

Core design choices

1) OpenAI compatibility
- Non-stream path returns choices[0].message.content from [Python.function infer](main.py:326).
- Streaming path (SSE) produces OpenAI-style "chat.completion.chunk" deltas, with id lines "session_id:index" for resume.
- We retained Chat Completions (legacy) rather than the newer Responses API for compatibility with existing SDKs. A custom cancel endpoint is provided to fill the gap.

2) Multimodal input handling
- The API accepts "messages" with content either as a string or an array of parts typed as "text" / "image_url" / "input_image" / "video_url" / "input_video".
- Images: URLs (http/https or data URL), base64, or local path are supported by [Python.function load_image_from_any](main.py:108).
- Videos: URLs and base64 are materialized to a temp file; frames extracted and uniformly sampled by [Python.function load_video_frames_from_any](main.py:150).

3) Engine and generation
- Qwen chat template applied via processor.apply_chat_template in both [Python.function infer](main.py:326) and [Python.function infer_stream](main.py:375).
- Generation sampling uses temperature; do_sample toggled when temperature > 0.
- Streams are produced using TextIteratorStreamer.
- Optional cooperative cancellation is implemented with a StoppingCriteria bound to a session cancel event in [Python.function infer_stream](main.py:375).

4) Streaming, resume, and persistence
- In-memory buffer per session for immediate replay: [Python.class _SSESession](main.py:435).
- Optional SQLite persistence to survive restarts and handle long gaps: [Python.class _SQLiteStore](main.py:482).
- Resume protocol:
  - Client provides session_id in the request body and Last-Event-ID header "session_id:index", or pass ?last_event_id=...
  - Server replays events after index from SQLite (if enabled) and the in-memory buffer.
  - Producer appends events to both the ring buffer and SQLite (when enabled).

5) Cancellation and disconnects
- Manual cancel endpoint [Python.app.post()](main.py:792) sets the session cancel event and marks finished in SQLite.
- Auto-cancel after disconnect:
  - If all clients disconnect, a timer fires after CANCEL_AFTER_DISCONNECT_SECONDS (default 3600) that sets the cancel event.
  - The StoppingCriteria checks this event cooperatively and halts generation.

6) Environment configuration
- See [.env.example](.env.example).
- Important variables:
  - MODEL_REPO_ID (default "Qwen/Qwen3-VL-2B-Thinking")
  - HF_TOKEN (optional)
  - MAX_TOKENS, TEMPERATURE
  - MAX_VIDEO_FRAMES (video frame sampling)
  - DEVICE_MAP, TORCH_DTYPE (Transformers loading hints)
  - PERSIST_SESSIONS, SESSIONS_DB_PATH, SESSIONS_TTL_SECONDS (SQLite)
  - CANCEL_AFTER_DISCONNECT_SECONDS (auto-cancel threshold)

Security and privacy notes
- trust_remote_code=True executes code from the model repository when loading AutoProcessor/AutoModel. This is standard for many HF multimodal models but should be understood in terms of supply-chain risk.
- Do not log sensitive data. Avoid dumping raw request bodies or tokens.

Operational guidance

Running locally
- Install Python dependencies from [requirements.txt](requirements.txt) and install a suitable PyTorch wheel for your platform/CUDA.
- copy .env.example .env and adjust as needed.
- Start: python [Python.main()](main.py:807)

Testing endpoints
- Health: GET /health
- Chat (non-stream): POST /v1/chat/completions with messages array.
- Chat (stream): add "stream": true; optionally pass "session_id".
- Resume: send Last-Event-ID with "session_id:index".
- Cancel: POST /v1/cancel/{session_id}.

Scaling notes
- Typically deploy one model per process. For throughput, run multiple workers behind a load balancer; sessions are process-local unless persistence is used.
- SQLite persistence supports replay but does not synchronize cancel/producer state across processes. A Redis-based store (future work) can coordinate multi-process session state more robustly.

Known limitations and follow-ups
- Token accounting (usage prompt/completion/total) is stubbed at zeros. Populate if/when needed.
- Redis store not yet implemented (design leaves a clear seam via _SQLiteStore analog).
- No structured logging/tracing yet; follow-up for observability.
- Cancellation is best-effort cooperative; it relies on the stopping criteria hook in generation.

Changelog (2025-10-23)
- feat(server): Python FastAPI server with Qwen3-VL (Transformers), OpenAI-compatible /v1/chat/completions.
- feat(stream): SSE streaming with session_id + Last-Event-ID resumability.
- feat(persist): Optional SQLite-backed session persistence for replay across restarts.
- feat(cancel): Manual cancel endpoint /v1/cancel/{session_id}; auto-cancel after disconnect threshold.
- docs: Updated [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [RULES.md](RULES.md). Rewrote [TODO.md](TODO.md) pending/complete items (see repo TODO).
- chore: Removed Node.js and scripts from the prior stack.

Verification checklist
- Non-stream text-only request returns a valid completion.
- Image and video prompts pass through preprocessing and generate coherent output.
- Streaming emits OpenAI-style deltas and ends with [DONE].
- Resume works with Last-Event-ID and session_id across reconnects; works after server restart when PERSIST_SESSIONS=1.
- Manual cancel halts generation and marks session finished; subsequent resumes return a finished stream.
- Auto-cancel fires after all clients disconnect for CANCEL_AFTER_DISCONNECT_SECONDS and cooperatively stops generation.

End of entry.
## Progress Log Template (Mandatory per RULES)

Use this template for every change or progress step. Add a new entry before/with each commit, then append the final commit hash after push. See enforcement in [RULES.md](RULES.md:33) and the progress policy in [RULES.md](RULES.md:49).

Entry template
- Date/Time (Asia/Jakarta): YYYY-MM-DD HH:mm
- Commit: &lt;hash&gt; - &lt;conventional message&gt;
- Scope/Files (clickable anchors required):
  - [Python.function chat_completions()](main.py:591)
  - [Python.function infer_stream()](main.py:375)
  - [README.md](README.md:1), [ARCHITECTURE.md](ARCHITECTURE.md:1), [RULES.md](RULES.md:1), [TODO.md](TODO.md:1)
- Summary:
  - What changed and why (problem/requirement)
- Changes:
  - Short bullet list of code edits with anchors
- Verification:
  - Commands:
    - curl examples (non-stream, stream with session_id, resume with Last-Event-ID)
    - cancel API test: curl -X POST http://localhost:3000/v1/cancel/mysession123
  - Expected vs Actual:
    - …
- Follow-ups/Limitations:
  - …
- Notes:
  - If commit hash unknown at authoring time, update the entry after git push.

Git sequence (run every time)
- git add .
- git commit -m "type(scope): short description"
- git push
- Update this entry with the final commit hash.

Example (filled)
- Date/Time: 2025-10-23 14:30 (Asia/Jakarta)
- Commit: f724450 - feat(stream): add SQLite persistence for SSE resume
- Scope/Files:
  - [Python.class _SQLiteStore](main.py:482)
  - [Python.function chat_completions()](main.py:591)
  - [README.md](README.md:1), [ARCHITECTURE.md](ARCHITECTURE.md:1)
- Summary:
  - Persist SSE chunks to SQLite for replay across restarts; enable via PERSIST_SESSIONS.
- Changes:
  - Add _SQLiteStore with schema and CRUD
  - Wire producer to append events to DB
  - Replay DB events on resume before in-memory buffer
- Verification:
  - curl -N -H "Content-Type: application/json" ^
    -d "{\"session_id\":\"mysession123\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" ^
    http://localhost:3000/v1/chat/completions
  - Restart server; resume:
    curl -N -H "Content-Type: application/json" ^
    -H "Last-Event-ID: mysession123:42" ^
    -d "{\"session_id\":\"mysession123\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" ^
    http://localhost:3000/v1/chat/completions
  - Expected vs Actual: replayed chunks after index 42, continued live, ended with [DONE].
- Follow-ups:
  - Consider Redis store for multi-process coordination
## Progress Log — 2025-10-23 14:31 (Asia/Jakarta)

- Commit: f724450 - docs: sync README/ARCHITECTURE/RULES with main.py; add progress log in CLAUDE.md; enforce mandatory Git
- Scope/Files (anchors):
  - [Python.function chat_completions()](main.py:591)
  - [Python.function infer_stream()](main.py:375)
  - [Python.class _SSESession](main.py:435), [Python.class _SessionStore](main.py:449), [Python.class _SQLiteStore](main.py:482)
  - [README.md](README.md:1), [ARCHITECTURE.md](ARCHITECTURE.md:1), [RULES.md](RULES.md:1), [CLAUDE.md](CLAUDE.md:1), [.env.example](.env.example:1)
- Summary:
  - Completed Python migration and synchronized documentation. Implemented SSE streaming with resume, optional SQLite persistence, auto-cancel on disconnect, and manual cancel API. RULES now mandate Git usage and progress logging.
- Changes:
  - Document streaming/resume/persistence/cancel in [README.md](README.md:1) and [ARCHITECTURE.md](ARCHITECTURE.md:1)
  - Enforce Git workflow and progress logging in [RULES.md](RULES.md:33)
  - Add Progress Log template and entries in [CLAUDE.md](CLAUDE.md:1)
- Verification:
  - Non-stream:
    curl -X POST http://localhost:3000/v1/chat/completions ^
      -H "Content-Type: application/json" ^
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
  - Stream:
    curl -N -H "Content-Type: application/json" ^
      -d "{\"session_id\":\"mysession123\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" ^
      http://localhost:3000/v1/chat/completions
  - Resume:
    curl -N -H "Content-Type: application/json" ^
      -H "Last-Event-ID: mysession123:42" ^
      -d "{\"session_id\":\"mysession123\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" ^
      http://localhost:3000/v1/chat/completions
  - Cancel:
    curl -X POST http://localhost:3000/v1/cancel/mysession123
  - Results:
    - Streaming emits chunks, ends with [DONE]; resume replays after index; cancel terminates generation; auto-cancel after disconnect threshold works via timer + stopping criteria.
- Follow-ups:
  - Optional Redis store for multi-process coordination.
