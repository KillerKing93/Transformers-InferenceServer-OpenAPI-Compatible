# Python FastAPI Inference Server (OpenAI-Compatible) for Qwen3-VL-2B-Thinking

This repository has been migrated from a Node.js/llama.cpp stack to a Python/Transformers stack to fully support multimodal inference (text, images, videos) with the Hugging Face Qwen3 models.

Key files:

- Server entry: [main.py](main.py)
- Environment template: [.env.example](.env.example)
- Python dependencies: [requirements.txt](requirements.txt)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) (will be updated to reflect the Python stack)

Model:

- Default: Qwen/Qwen3-VL-2B-Thinking (Transformers; supports multimodal)
- You can change the model via environment variable MODEL_REPO_ID.

Node.js artifacts and scripts from the previous project have been removed.

Requirements

- Python 3.10+
- pip
- PyTorch (install a wheel matching your platform/CUDA)
- Optionally a GPU with enough VRAM for the chosen model

Install

1. Create and activate a virtual environment (Windows CMD):
   python -m venv .venv
   .venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Install PyTorch appropriate for your platform (examples):
   CPU-only:
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   CUDA 12.4 example:
   pip install torch --index-url https://download.pytorch.org/whl/cu124

4. Create a .env from the template and adjust if needed:
   copy .env.example .env
   - Set HF_TOKEN if the model is gated
   - Adjust MAX_TOKENS, TEMPERATURE, DEVICE_MAP, TORCH_DTYPE, MAX_VIDEO_FRAMES as desired

Configuration via .env
See [.env.example](.env.example). Important variables:

- PORT=3000
- MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Thinking
- HF_TOKEN= # optional if gated
- MAX_TOKENS=256
- TEMPERATURE=0.7
- MAX_VIDEO_FRAMES=16
- DEVICE_MAP=auto
- TORCH_DTYPE=auto

Additional streaming/persistence configuration
- PERSIST_SESSIONS=1                 # enable SQLite-backed resumable SSE
- SESSIONS_DB_PATH=sessions.db       # SQLite db path
- SESSIONS_TTL_SECONDS=600           # TTL for finished sessions before GC
- CANCEL_AFTER_DISCONNECT_SECONDS=3600  # auto-cancel generation if all clients disconnect for this many seconds (0=disable)

Cancel session API (custom extension)
- Endpoint: POST /v1/cancel/{session_id}
- Purpose: Manually cancel an in-flight streaming generation for the given session_id. Not part of OpenAI Chat Completions spec (the newer OpenAI Responses API has cancel), so this is provided as a practical extension.
- Example (Windows CMD):
  curl -X POST http://localhost:3000/v1/cancel/mysession123
Run

- Direct:
  python main.py

- Using uvicorn:
  uvicorn main:app --host 0.0.0.0 --port 3000

Endpoints (OpenAI-compatible)

- Health
  GET /health
  Example:
  curl http://localhost:3000/health
  Response:
  {
  "ok": true,
  "modelReady": true,
  "modelId": "Qwen/Qwen3-VL-2B-Thinking",
  "error": null
  }

- Chat Completions (non-streaming)
  POST /v1/chat/completions
  Example (Windows CMD):
  curl -X POST http://localhost:3000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen-local\",\"messages\":[{\"role\":\"user\",\"content\":\"Describe this image briefly\"}],\"max_tokens\":128}"

  Example (PowerShell):
  $body = @{
  model = "qwen-local"
  messages = @(@{ role = "user"; content = "Hello Qwen3!" })
  max_tokens = 128
  } | ConvertTo-Json -Depth 5
  curl -Method POST http://localhost:3000/v1/chat/completions -ContentType "application/json" -Body $body

- Chat Completions (streaming via Server-Sent Events)
  Set "stream": true to receive partial deltas as they are generated.
  Example (Windows CMD):
  curl -N -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen-local\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: what is 17 * 23?\"}],\"stream\":true}" ^
  http://localhost:3000/v1/chat/completions

  The stream format follows OpenAI-style SSE:
  data: { "id": "...", "object": "chat.completion.chunk", "choices":[{ "delta": {"role": "assistant"} }]}
  data: { "choices":[{ "delta": {"content": "To"} }]}
  data: { "choices":[{ "delta": {"content": " think..."} }]}
  ...
  data: { "choices":[{ "delta": {}, "finish_reason": "stop"}]}
  data: [DONE]

Multimodal Usage

- Text only:
  { "role": "user", "content": "Summarize: The quick brown fox ..." }

- Image by URL:
  {
  "role": "user",
  "content": [
  { "type": "text", "text": "What is in this image?" },
  { "type": "image_url", "image_url": { "url": "https://example.com/cat.jpg" } }
  ]
  }

- Image by base64:
  {
  "role": "user",
  "content": [
  { "type": "text", "text": "OCR this." },
  { "type": "input_image", "b64_json": "<base64 of image bytes>" }
  ]
  }

- Video by URL (frames are sampled up to MAX_VIDEO_FRAMES):
  {
  "role": "user",
  "content": [
  { "type": "text", "text": "Describe this clip." },
  { "type": "video_url", "video_url": { "url": "https://example.com/clip.mp4" } }
  ]
  }

- Video by base64:
  {
  "role": "user",
  "content": [
  { "type": "text", "text": "Count the number of cars." },
  { "type": "input_video", "b64_json": "<base64 of full video file>" }
  ]
  }

Implementation Notes

- Server code: [main.py](main.py)
  - FastAPI with CORS enabled
  - Non-streaming and streaming endpoints
  - Uses AutoProcessor and AutoModelForCausalLM with trust_remote_code=True
  - Converts OpenAI-style messages into the Qwen multimodal format
  - Images loaded via PIL; videos loaded via imageio.v3 (preferred) or OpenCV as fallback; frames sampled

Performance Tips

- On GPUs: set DEVICE_MAP=auto and TORCH_DTYPE=bfloat16 or float16 if supported
- Reduce MAX_VIDEO_FRAMES to speed up video processing
- Tune MAX_TOKENS and TEMPERATURE according to your needs

Troubleshooting

- ImportError or no CUDA found:
  - Ensure PyTorch is installed with the correct wheel for your environment.
- OOM / CUDA out of memory:
  - Use a smaller model, lower MAX_VIDEO_FRAMES, lower MAX_TOKENS, or run on CPU.
- 503 Model not ready:
  - The first request triggers model load; check /health for errors and HF_TOKEN if gated.

License

- See LICENSE for terms.

Changelog and Architecture

- We will update [ARCHITECTURE.md](ARCHITECTURE.md) to reflect the Python server flow.


## Streaming behavior, resume, and reconnections

The server streams responses using Server‑Sent Events (SSE) from [Python.function chat_completions()](main.py:457), driven by token iteration in [Python.function infer_stream](main.py:361). It now supports resumable streaming using an in‑memory ring buffer and SSE Last-Event-ID, with optional SQLite persistence (enable PERSIST_SESSIONS=1).

What’s implemented
- Per-session in-memory ring buffer keyed by session_id (no external storage).
- Each SSE event carries an SSE id line in the format "session_id:index" so clients can resume with Last-Event-ID.
- On reconnect:
  - Provide the same session_id in the request body, and
  - Provide "Last-Event-ID: session_id:index" header (or query ?last_event_id=session_id:index).
  - The server replays cached events after index and continues streaming new tokens.
- Session TTL: ~10 minutes, buffer capacity: ~2048 events. Old or finished sessions are garbage-collected in-memory.

How to start a streaming session
- Minimal (server generates a session_id internally for SSE id lines):
  Windows CMD:
    curl -N -H "Content-Type: application/json" ^
         -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" ^
         http://localhost:3000/v1/chat/completions

- With explicit session_id (recommended if you want to resume):
  Windows CMD:
    curl -N -H "Content-Type: application/json" ^
         -d "{\"session_id\":\"mysession123\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" ^
         http://localhost:3000/v1/chat/completions

How to resume after disconnect
- Use the same session_id and the SSE Last-Event-ID header (or ?last_event_id=...):
  Windows CMD (resume from index 42):
    curl -N -H "Content-Type: application/json" ^
         -H "Last-Event-ID: mysession123:42" ^
         -d "{\"session_id\":\"mysession123\",\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" ^
         http://localhost:3000/v1/chat/completions

  Alternatively with query string:
    http://localhost:3000/v1/chat/completions?last_event_id=mysession123:42

Event format
- Chunks follow the OpenAI-style "chat.completion.chunk" shape in data payloads, plus an SSE id:
  id: mysession123:5
  data: {"id":"mysession123","object":"chat.completion.chunk","created":..., "model":"...", "choices":[{"index":0,"delta":{"content":" token"},"finish_reason":null}]}

- The stream ends with:
  data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
  data: [DONE]

Notes and limits
- This implementation keeps session state only in memory; restarts will drop buffers.
- If the buffer overflows before you resume, the earliest chunks may be unavailable.
- Cancellation on client disconnect is not automatic; generation runs to completion in the background. A cancellable stopping-criteria can be added if required.


## Hugging Face repository files support

This server loads the Qwen3-VL model via Transformers with `trust_remote_code=True`, so the standard files from the repo are supported and consumed automatically. Summary for https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking/tree/main:

- Used by model weights and architecture
  - model.safetensors — main weights loaded by AutoModelForCausalLM
  - config.json — architecture/config
  - generation_config.json — default gen params (we may override via request or env)

- Used by tokenizer
  - tokenizer.json — primary tokenizer specification
  - tokenizer_config.json — tokenizer settings
  - merges.txt and vocab.json — fallback/compat files; if tokenizer.json exists, HF generally prefers it

- Used by processors (multimodal)
  - preprocessor_config.json — image/text processor config
  - video_preprocessor_config.json — video processor config (frame sampling, etc.)
  - chat_template.json — chat formatting used by [Python.function infer](main.py:312) and [Python.function infer_stream](main.py:361) via `processor.apply_chat_template(...)`

- Not required for runtime
  - README.md, .gitattributes — ignored by runtime

Notes:
- We rely on Transformers’ AutoModelForCausalLM and AutoProcessor to resolve and use the above files; no manual parsing is required in our code.
- With `trust_remote_code=True`, model-specific code from the repo may load additional assets transparently.
- If the repo updates configs (e.g., new chat template), the server will pick them up on next load.


## Cancellation and session persistence

- Auto-cancel on disconnect:
  - Generation is automatically cancelled if all clients disconnect for more than CANCEL_AFTER_DISCONNECT_SECONDS (default 3600 seconds = 1 hour). Configure in [.env.example](.env.example) via `CANCEL_AFTER_DISCONNECT_SECONDS`.
  - Implemented by a timer in [Python.function chat_completions](main.py:732) that triggers a cooperative stop through a stopping criteria in [Python.function infer_stream](main.py:375).

- Manual cancel API (custom extension):
  - Endpoint: `POST /v1/cancel/{session_id}`
  - Cancels an ongoing streaming session and marks it finished in the store. Example (Windows CMD):
    curl -X POST http://localhost:3000/v1/cancel/mysession123
  - This is not part of OpenAI’s legacy Chat Completions spec. OpenAI’s newer Responses API has a cancel endpoint, but Chat Completions does not. We provide this custom endpoint for operational control.

- Persistence:
  - Optional SQLite-backed persistence for resumable SSE (enable `PERSIST_SESSIONS=1` in [.env.example](.env.example)).
  - Database path: `SESSIONS_DB_PATH` (default: sessions.db)
  - Session TTL for GC: `SESSIONS_TTL_SECONDS` (default: 600)
  - See implementation in [Python.class _SQLiteStore](main.py:481) and integration in [Python.function chat_completions](main.py:591).
  - Redis is not implemented yet; the design isolates persistence so a Redis-backed store can be added as a drop-in.
