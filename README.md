---
title: "Transformers Inference Server (Qwen3‑VL)"
emoji: 🐍
colorFrom: purple
colorTo: green
sdk: docker
app_port: 3000
pinned: false
---

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

## Quick Start

### Option 1: Run with Docker (with-model images: CPU / NVIDIA / AMD)

Tags built by CI:
- ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-cpu
- ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-nvidia
- ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-amd

Pull:

```bash
# CPU
docker pull ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-cpu

# NVIDIA (CUDA 12.4 wheel)
docker pull ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-nvidia

# AMD (ROCm 6.2 wheel)
docker pull ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-amd
```

Run:

```bash
# CPU
docker run -p 3000:3000 \
  -e HF_TOKEN=your_hf_token_here \
  ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-cpu

# NVIDIA GPU (requires NVIDIA drivers + nvidia-container-toolkit on the host)
docker run --gpus all -p 3000:3000 \
  -e HF_TOKEN=your_hf_token_here \
  ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-nvidia

# AMD GPU ROCm (requires ROCm 6.2+ drivers on the host; Linux only)
# Map ROCm devices and video group (may vary by distro)
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
  -p 3000:3000 \
  -e HF_TOKEN=your_hf_token_here \
  ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-amd
```

Health check:
```bash
curl http://localhost:3000/health
```

Swagger UI:
http://localhost:3000/docs

OpenAPI (YAML):
http://localhost:3000/openapi.yaml

Notes:
- These are with-model images; the first pull is large. In CI, after "Model downloaded." BuildKit may appear idle while tarring/committing the multi‑GB layer.
- Host requirements:
  - NVIDIA: recent driver + nvidia-container-toolkit.
  - AMD: ROCm 6.2+ driver stack, supported GPU, and mapped /dev/kfd and /dev/dri devices.

### Option 2: Run Locally

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
- MAX_TOKENS=4096
- TEMPERATURE=0.7
- MAX_VIDEO_FRAMES=16
- DEVICE_MAP=auto
- TORCH_DTYPE=auto

Additional streaming/persistence configuration

- PERSIST_SESSIONS=1 # enable SQLite-backed resumable SSE
- SESSIONS_DB_PATH=sessions.db # SQLite db path
- SESSIONS_TTL_SECONDS=600 # TTL for finished sessions before GC
- CANCEL_AFTER_DISCONNECT_SECONDS=3600 # auto-cancel generation if all clients disconnect for this many seconds (0=disable)

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

- Swagger UI
  GET /docs
- OpenAPI (YAML)
  GET /openapi.yaml
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
  -d "{\"model\":\"qwen-local\",\"messages\":[{\"role\":\"user\",\"content\":\"Describe this image briefly\"}],\"max_tokens\":4096}"

  Example (PowerShell):
  $body = @{
  model = "qwen-local"
  messages = @(@{ role = "user"; content = "Hello Qwen3!" })
  max_tokens = 4096
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
  - See implementation in [Python.class \_SQLiteStore](main.py:481) and integration in [Python.function chat_completions](main.py:591).
  - Redis is not implemented yet; the design isolates persistence so a Redis-backed store can be added as a drop-in.

## Deploy on Render

Render has two easy options. Since our image already bakes the model, the fastest path is to deploy the public Docker image (CPU). Render currently doesn’t provide NVIDIA/AMD GPUs for standard Web Services, so use the CPU image.

Option A — Deploy public Docker image (recommended)
1) In Render Dashboard: New → Web Service
2) Environment: Docker → Public Docker image
3) Image
   - ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-cpu
4) Instance and region
   - Region: closest to your users
   - Instance type: pick a plan with at least 16 GB RAM (more if you see OOM)
5) Port/health
   - Render auto-injects PORT; the server binds to it via [Python.os.getenv()](main.py:71)
   - Health Check Path: /health (served by [Python.function health](main.py:871))
6) Start command
   - Leave blank; the image uses CMD ["python","main.py"] as defined in [Dockerfile](Dockerfile:54). The app entry is [Python.main()](main.py:1).
7) Environment variables
   - EAGER_LOAD_MODEL=1
   - MAX_TOKENS=4096
   - HF_TOKEN=your_hf_token_here (only if the model is gated)
   - Optional persistence:
     - PERSIST_SESSIONS=1
     - SESSIONS_DB_PATH=/data/sessions.db (requires a disk)
8) Persistent Disk (optional)
   - Add a Disk (e.g., 1–5 GB) and mount it at /data if you enable SQLite persistence
9) Create Web Service and wait for it to start
10) Verify
   - curl https://YOUR-SERVICE.onrender.com/health
   - OpenAPI YAML: https://YOUR-SERVICE.onrender.com/openapi.yaml (served by [Python.function openapi_yaml](main.py:863))
   - Chat endpoint: POST https://YOUR-SERVICE.onrender.com/v1/chat/completions (implemented in [Python.function chat_completions](main.py:891))

Option B — Build directly from this GitHub repo (Dockerfile)
1) In Render Dashboard: New → Web Service → Build from a Git repo (connect this repo)
2) Render will detect the Dockerfile automatically (no Build Command needed)
3) Advanced → Docker Build Args
   - BACKEND=cpu  (ensures CPU-only torch wheel)
4) Health and env vars
   - Health Check Path: /health
   - Set EAGER_LOAD_MODEL, MAX_TOKENS, HF_TOKEN as needed (same as Option A)
5) (Optional) Add a Disk and mount at /data, then set SESSIONS_DB_PATH=/data/sessions.db if you want resumable SSE across restarts
6) Deploy (first build can take a while due to the multi-GB model layer)

Notes and limits on Render
- GPU acceleration (NVIDIA/AMD) isn’t available for standard Web Services on Render; use the CPU image.
- The image already contains the Qwen3-VL model under /app/hf-cache, so there’s no model download at runtime.
- SSE is supported; streaming is produced by [Python.function chat_completions](main.py:891). Keep the connection open to avoid idle timeouts.
- If you enable SQLite persistence, remember to attach a Disk; otherwise, the DB is ephemeral.

Example render.yaml (optional IaC)
If you prefer infrastructure-as-code, you can use a render.yaml like:

services:
  - type: web
    name: qwen-vl-cpu
    env: docker
    image:
      url: ghcr.io/killerking93/transformers-inferenceserver-openapi-compatible:latest-with-model-cpu
    plan: standard
    region: oregon
    healthCheckPath: /health
    autoDeploy: true
    envVars:
      - key: EAGER_LOAD_MODEL
        value: "1"
      - key: MAX_TOKENS
        value: "4096"
      # - key: HF_TOKEN
      #   sync: false  # set in dashboard or use Render secrets
      # - key: PERSIST_SESSIONS
      #   value: "1"
      # - key: SESSIONS_DB_PATH
      #   value: "/data/sessions.db"
    disks:
      # Uncomment if using persistence
      # - name: data
      #   mountPath: /data
      #   sizeGB: 5

After deploy:
- Health: GET /health
- OpenAPI: GET /openapi.yaml
- Inference:
  curl -X POST https://YOUR-SERVICE.onrender.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":4096}"

## Deploy on Hugging Face Spaces

Recommended: Docker Space (works with our FastAPI app and preserves multimodal behavior). You can run CPU or GPU hardware. To persist the HF cache across restarts, enable Persistent Storage and point HF cache to /data.

A) Create the Space (Docker)
1) Install CLI and login:
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login

2) Create a Docker Space (public or private):
   huggingface-cli repo create my-qwen3-vl-server --type space --sdk docker

3) Add the Space as a remote and push this repo:
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/my-qwen3-vl-server
   git push hf main

This pushes Dockerfile, main.py, requirements.txt. The Space will auto-build your container.

B) Configure Space settings
- Hardware:
  - CPU: works out-of-the-box (fast to build, slower inference).
  - GPU: choose a GPU tier (e.g., T4/A10G/L4) for faster inference.

- Persistent Storage (recommended):
  - Enable Persistent storage (e.g., 10–30 GB).
  - This lets you cache models and sessions across restarts.

- Variables and Secrets:
  - Variables:
    - EAGER_LOAD_MODEL=1
    - MAX_TOKENS=4096
    - HF_HOME=/data/hf-cache
    - TRANSFORMERS_CACHE=/data/hf-cache
  - Secrets:
    - HF_TOKEN=your_hf_token_if_model_is_gated

C) CPU vs GPU on Spaces
- CPU: No change needed. Our Dockerfile defaults to CPU PyTorch and bakes the model during build. It will run on CPU Spaces.
- GPU: Edit the Space’s Dockerfile to switch the backend before the next build:
  - In the file editor of the Space UI, change:
      ARG BACKEND=cpu
    to:
      ARG BACKEND=nvidia
  - Save/commit; the Space rebuilds with a CUDA-enabled torch. Choose a GPU hardware tier in the Space settings. Note: Building the GPU image pulls CUDA torch wheels and increases build time.
- AMD ROCm is not available on Spaces; use NVIDIA GPUs on Spaces.

D) Speed up cold starts and caching
- With Persistent Storage enabled and HF_HOME/TRANSFORMERS_CACHE pointed to /data/hf-cache, the model cache persists across restarts (subsequent spins are much faster).
- Keep the Space “Always on” if available on your plan to avoid cold starts.

E) Space endpoints
- Base URL: https://huggingface.co/spaces/YOUR_USERNAME/my-qwen3-vl-server (Spaces proxy to your container)
- Swagger UI: GET /docs (interactive API with examples)
- Health: GET /health (implemented by [Python.function health](main.py:951))
- OpenAPI YAML: GET /openapi.yaml (implemented by [Python.openapi_yaml](main.py:943))
- Chat Completions: POST /v1/chat/completions (non-stream + SSE) [Python.function chat_completions](main.py:971)
- Cancel: POST /v1/cancel/{session_id} [Python.function cancel_session](main.py:1191)

F) Quick test after the Space is “Running”
- Health:
  curl -s https://YOUR-SPACE-Subdomain.hf.space/health
- Non-stream:
  curl -s -X POST https://YOUR-SPACE-Subdomain.hf.space/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello from HF Spaces!\"}],\"max_tokens\":4096}"
- Streaming:
  curl -N -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Think step by step: 17*23?\"}],\"stream\":true}" \
    https://YOUR-SPACE-Subdomain.hf.space/v1/chat/completions

Notes
- The Space build step can appear “idle” after “Model downloaded.” while Docker commits a multi‑GB layer; this is expected.
- If you hit OOM, increase the Space hardware memory or switch to a GPU tier. Reduce MAX_VIDEO_FRAMES and MAX_TOKENS if needed.
