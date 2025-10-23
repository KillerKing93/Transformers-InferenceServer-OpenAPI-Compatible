# NodeJS LLamaCPP Inference Server (OpenAI-Compatible)

An OpenAI API-compatible inference server powered by node-llama-cpp (llama.cpp bindings). This project automates:
- Downloading model weights from Hugging Face
- Attempting conversion to GGUF (for llama.cpp)
- Loading the converted model for inference
- Serving a basic OpenAI-compatible endpoint

Key files:
- Server entry: index.js
- Scripts: scripts/download-model.js, scripts/convert-model.js, scripts/setup-model.js, scripts/postinstall-checks.js
- Rules: RULES.md (authoritative workflow)
- Dev log: CLAUDE.md
- Architecture: ARCHITECTURE.md
- Tasks: TODO.md

Note on target model
By request, this project targets Qwen/Qwen3-VL-2B-Thinking-FP8 as the default. This is a TensorFlow FP8 vision-language (VLM) model; standard llama.cpp GGUF conversion is primarily designed for text LLM weights (often Transformers/PyTorch). If this conversion is unsupported, the scripts will fail fast with clear guidance and alternatives. You can easily point the server to a different, supported model using .env.

Requirements
- Node.js 20+
- Python 3.10+ (for conversion tooling)
- Git (recommended, used to auto-fetch llama.cpp converter if needed)
- Git LFS (recommended for some large model repos)
- Windows 11 supported (project is OS-agnostic with equivalent tooling on Linux/macOS)

Install
1) Clone the repository
   Windows CMD:
   git clone https://github.com/KillerKing93/NodeJS-LLamaCPP-InferenceServer-OpenAPI-Compatible.git
   cd NodeJS-LLamaCPP-InferenceServer-OpenAPI-Compatible

2) Install dependencies (postinstall checks will run)
   npm install

3) Optional: create a local .env to customize behavior (see “Configuration via .env”)

Configuration via .env
You can switch models and tune inference quickly via .env. Create a .env alongside package.json with variables like:

# Server
PORT=3000

# Model defaults (change these to switch models)
MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Thinking-FP8
MODELS_DIR=./models
OUT_GGUF_NAME=model.gguf
# Or point directly to a GGUF file (overrides the above three)
# MODEL_GGUF_PATH=./models/Qwen/Qwen3-VL-2B-Thinking-FP8/model.gguf

# Inference params
CTX_SIZE=4096
MAX_TOKENS=256
TEMPERATURE=0.7

# Optional
# HF_TOKEN=                # required for gated/private repos
# PYTHON_EXE=python        # or a full path to Python 3.10+
# LLAMACPP_DIR=./tools/llama.cpp
# LLAMACPP_CONVERTER=      # path to convert-hf-to-gguf.py if you have it locally
# CONVERT_ARGS=--outtype q8_0
# DOWNLOAD_CONCURRENCY=3
# HF_FILE_PATTERNS=*.json,*.md,*.safetensors,*.bin,*.pb,variables/*

Model lifecycle (download -> convert -> serve)
- Download from Hugging Face
  npm run download-model
  - Reads MODEL_REPO_ID, MODELS_DIR, HF_TOKEN, HF_FILE_PATTERNS, DOWNLOAD_CONCURRENCY from .env
  - Stores files under ./models/<org>/<repo>

- Convert to GGUF (attempt)
  npm run convert-model
  - Reads MODELS_DIR, OUT_GGUF_NAME, PYTHON_EXE, LLAMACPP_* from .env
  - Attempts to run llama.cpp’s convert-hf-to-gguf.py
  - If the model is a TF SavedModel (e.g. Qwen3-VL TF FP8), conversion will likely fail; the script fails fast with clear instructions

- Orchestrate both steps (idempotent)
  npm run setup-model
  - Skips steps if the target GGUF already exists

- Serve
  npm start
  - The server loads MODEL_GGUF_PATH if set; otherwise constructs the GGUF path from MODEL_REPO_ID, MODELS_DIR, OUT_GGUF_NAME
  - If the GGUF is missing, the server returns 503 for inference and advises running setup

OpenAI-compatible API
- Health
  GET /health
  Response example:
  {
    "ok": true,
    "modelReady": true,
    "modelPath": "./models/ORG/REPO/model.gguf",
    "error": null
  }

- Chat Completions (non-streaming)
  POST /v1/chat/completions
  Request example:
  curl -X POST http://localhost:3000/v1/chat/completions ^
       -H "Content-Type: application/json" ^
       -d "{\"model\":\"qwen-local\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
  Response example:
  {
    "id": "chatcmpl-<id>",
    "object": "chat.completion",
    "created": 173...,
    "model": "./models/ORG/REPO/model.gguf",
    "choices": [
      {
        "index": 0,
        "message": { "role": "assistant", "content": "..." },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0
    }
  }

Notes:
- messages must be a non-empty array
- Optional fields max_tokens and temperature are supported and default to MAX_TOKENS and TEMPERATURE from .env

Known limitations and guidance
- Qwen/Qwen3-VL-2B-Thinking-FP8 is a TF FP8 VLM; llama.cpp GGUF conversion is primarily for text LLMs with Transformers (PyTorch) weights. The provided convert script detects TF SavedModel artifacts and fails fast with actionable guidance.
- If you need an immediately working flow, consider:
  - Using a text-only Qwen/Qwen2/Qwen2.5 model with Transformers weights supported by convert-hf-to-gguf.py; or
  - Downloading a prebuilt GGUF known to work with llama.cpp, set MODEL_GGUF_PATH directly in .env, and skip conversion.
- node-llama-cpp may require Node 20+ and can trigger native builds. Prebuilt binaries are provided for many platforms but not all; see upstream docs if builds occur.

NPM scripts
- start: node index.js
- dev: node --watch index.js
- download-model: node scripts/download-model.js
- convert-model: node scripts/convert-model.js
- setup-model: node scripts/setup-model.js
- postinstall: node scripts/postinstall-checks.js
- test: placeholder, returns success

Troubleshooting
- 503 Model not ready
  - Run: npm run setup-model
  - Or set MODEL_GGUF_PATH directly to an existing .gguf
- Conversion fails (TF/VLM formats)
  - Use a supported Transformers (PyTorch) model or a prebuilt GGUF
  - Check CLAUDE.md for notes and alternatives
- Python not found
  - Install Python 3.10+ and set PYTHON_EXE in .env
- git not found
  - Install Git; auto-fetching llama.cpp converter or LFS may require it

Repository rules and docs
- Always document code changes in README.md and CLAUDE.md, and update ARCHITECTURE.md if design changes (see RULES.md).
- Never commit node_modules or models (see .gitignore).
- Commit and push after each progress step (see RULES.md for style).

License
- See LICENSE for the modified Apache 2.0 (Royalty-linked) terms.
