# Architecture (Python FastAPI + Transformers)

This document describes the Python-based, OpenAI-compatible inference server for Qwen3-VL, replacing the previous Node.js/llama.cpp stack.

Key source files
- Server entry: [main.py](main.py)
- Inference engine: [Python.class Engine](main.py:231)
- Multimodal parsing: [Python.function build_mm_messages](main.py:251), [Python.function load_image_from_any](main.py:108), [Python.function load_video_frames_from_any](main.py:150)
- Endpoints: Health [Python.app.get()](main.py:577), Chat Completions [Python.app.post()](main.py:591), Cancel [Python.app.post()](main.py:792)
- Streaming + resume: [Python.class _SSESession](main.py:435), [Python.class _SessionStore](main.py:449), [Python.class _SQLiteStore](main.py:482), [Python.function chat_completions](main.py:591)
- Local run (uvicorn): [Python.main()](main.py:807)
- Configuration template: [.env.example](.env.example)
- Dependencies: [requirements.txt](requirements.txt)

Model target (default)
- Hugging Face: Qwen/Qwen3-VL-2B-Thinking (Transformers, multimodal)
- Overridable via environment variable: MODEL_REPO_ID

## Overview

The server exposes an OpenAI-compatible endpoint for chat completions that supports:
- Text-only prompts
- Images (URL or base64)
- Videos (URL or base64; frames sampled)

Two response modes are implemented:
- Non-streaming JSON
- Streaming via Server-Sent Events (SSE) with resumable delivery using Last-Event-ID. Resumability is achieved with an in‑memory ring buffer and optional SQLite persistence.

## Components

1) FastAPI application
- Instantiated in [Python.main module](main.py:541) and endpoints mounted at:
  - Health: [Python.app.get()](main.py:577)
  - Chat Completions (non-stream + SSE): [Python.app.post()](main.py:591)
  - Manual cancel (custom): [Python.app.post()](main.py:792)
- CORS is enabled for simplicity.

2) Inference Engine (Transformers)
- Class: [Python.class Engine](main.py:231)
- Loads:
  - Processor: AutoProcessor(trust_remote_code=True)
  - Model: AutoModelForCausalLM (device_map, dtype configurable via env)
- Core methods:
  - Input building: [Python.function build_mm_messages](main.py:251)
  - Text-only generate: [Python.function infer](main.py:326)
  - Streaming generate (iterator): [Python.function infer_stream](main.py:375)

3) Multimodal preprocessing
- Images:
  - URL (http/https), data URL, base64, or local path
  - Loader: [Python.function load_image_from_any](main.py:108)
- Videos:
  - URL (downloaded to temp), base64 to temp file, or local path
  - Frame extraction via imageio.v3 (preferred) or OpenCV fallback
  - Uniform sampling up to MAX_VIDEO_FRAMES
  - Loader: [Python.function load_video_frames_from_any](main.py:150)

4) SSE streaming with resume
- Session objects:
  - [Python.class _SSESession](main.py:435): ring buffer, condition variable, producer thread reference, cancellation event, listener count, and disconnect timer
  - [Python.class _SessionStore](main.py:449): in-memory map with TTL + GC
  - Optional persistence: [Python.class _SQLiteStore](main.py:482) for replaying chunks across restarts
- SSE id format: "session_id:index"
- Resume:
  - Client sends Last-Event-ID header (or query ?last_event_id=...) and the same session_id in the body
  - Server replays cached/persisted chunks after the provided index, then continues live streaming
- Producer:
  - Created on demand per session; runs generation in a daemon thread and pushes chunks into the ring buffer and SQLite (if enabled)
  - See producer closure inside [Python.function chat_completions](main.py:591)
- Auto-cancel on disconnect:
  - If all clients disconnect for CANCEL_AFTER_DISCONNECT_SECONDS (default 3600s), a timer signals cancellation via a stopping criteria in [Python.function infer_stream](main.py:375)

## Request flow

Non-streaming (POST /v1/chat/completions)
1. Validate input, load engine singleton via [Python.function get_engine](main.py:558)
2. Convert OpenAI-style messages to Qwen chat template via [Python.function build_mm_messages](main.py:251) and apply_chat_template
3. Preprocess images/videos into processor inputs
4. Generate with [Python.function infer](main.py:326)
5. Return OpenAI-compatible response (choices[0].message.content)

Streaming (POST /v1/chat/completions with "stream": true)
1. Determine session_id:
   - Use body.session_id if provided; otherwise generated server-side
2. Parse Last-Event-ID (or query ?last_event_id) to get last delivered index
3. Create/start or reuse producer thread for this session
4. StreamingResponse generator:
   - Replays persisted events (SQLite, if enabled) and in-memory buffer after last index
   - Waits on condition variable for new tokens
   - Emits "[DONE]" at the end or upon buffer completion
5. Clients can reconnect and resume by sending Last-Event-ID: "session_id:index"
6. If all clients disconnect, an auto-cancel timer can stop generation (configurable via env)

Manual cancel (POST /v1/cancel/{session_id})
- Custom operational shortcut to cancel an in-flight generation for a session id.
- This is not part of the legacy OpenAI Chat Completions spec (OpenAI’s newer Responses API defines cancel); it is provided for practical control.

## Message and content mapping

Input format (OpenAI-like):
- "messages" list of role/content entries
- content can be:
  - string (text)
  - array of parts with "type":
    - "text": { text: "..."}
    - "image_url": { image_url: { url: "..." } } or { image_url: "..." }
    - "input_image": { b64_json: "..." } or { image: "..." }
    - "video_url": { video_url: { url: "..." } } or { video_url: "..." }
    - "input_video": { b64_json: "..." } or { video: "..." }

Conversion:
- [Python.function build_mm_messages](main.py:251) constructs a multimodal content list per message:
  - { type: "text", text: ... }
  - { type: "image", image: PIL.Image }
  - { type: "video", video: [PIL.Image frames] }

Template:
- Qwen apply_chat_template:
  - See usage in [Python.function infer](main.py:326) and [Python.function infer_stream](main.py:375)

## Configuration (.env)

See [.env.example](.env.example)
- PORT (default 3000)
- MODEL_REPO_ID (default "Qwen/Qwen3-VL-2B-Thinking")
- HF_TOKEN (optional)
- MAX_TOKENS (default 256)
- TEMPERATURE (default 0.7)
- MAX_VIDEO_FRAMES (default 16)
- DEVICE_MAP (default "auto")
- TORCH_DTYPE (default "auto")
- PERSIST_SESSIONS (default 0; set 1 to enable SQLite persistence)
- SESSIONS_DB_PATH (default sessions.db)
- SESSIONS_TTL_SECONDS (default 600)
- CANCEL_AFTER_DISCONNECT_SECONDS (default 3600; set 0 to disable)

## Error handling and readiness

- Health endpoint: [Python.app.get()](main.py:577)
  - Returns { ok, modelReady, modelId, error }
- Chat endpoint:
  - 400 for invalid messages or multimodal parsing errors
  - 503 when model failed to load
  - 500 for unexpected generation errors
- During first request, the model is lazily loaded; subsequent requests reuse the singleton

## Performance and scaling

- GPU recommended:
  - Set DEVICE_MAP=auto and TORCH_DTYPE=bfloat16/float16 if supported
- Reduce MAX_VIDEO_FRAMES to speed up video processing
- For concurrency:
  - FastAPI/Uvicorn workers and model sharing: typically 1 model per process
  - For high throughput, prefer multiple processes or a queueing layer

## Data and directories

- models/ contains downloaded model artifacts (implicitly created by Transformers cache); ignored by git
- tmp/ used transiently for video decoding (temporary files)

Ignored artifacts (see [.gitignore](.gitignore))
- Python: .venv/, __pycache__/, .cache/, etc.
- Large artifacts: models/, data/, uploads/, tmp/

## Streaming resume details

- Session store:
  - In-memory ring buffer for fast replay
  - Optional SQLite persistence for robust replay across process restarts
  - See GC in [Python.class _SessionStore](main.py:449) and [Python.method _SQLiteStore.gc](main.py:526)
- Limits:
  - Ring buffer stores ~2048 SSE events per session by default
  - If the buffer overflows before a client resumes and persistence is disabled, the earliest chunks may be unavailable
- End-of-stream:
  - Final chunk contains finish_reason: "stop"
  - "[DONE]" sentinel is emitted afterwards

## Future enhancements

- Redis persistence:
  - Add a Redis-backed store as a drop-in alongside SQLite
- Token accounting:
  - Populate usage prompt/completion/total tokens when model exposes tokenization costs
- Logging/observability:
  - Structured logs, request IDs, and metrics

## Migration notes (from Node.js)

- All Node.js server files and scripts were removed (index.js, package*.json, scripts/)
- The server now targets Transformers models directly and supports multimodal inputs out of the box
- The API remains OpenAI-compatible on /v1/chat/completions with resumable SSE and optional SQLite persistence
