# Architecture

This document describes the end-to-end architecture of the NodeJS Llama.cpp Inference Server, its configuration model (.env-driven), model lifecycle (download → convert → serve), and the OpenAI-compatible request flow.

Key source files
- [index.js](index.js)
- [scripts/download-model.js](scripts/download-model.js)
- [scripts/convert-model.js](scripts/convert-model.js)
- [scripts/setup-model.js](scripts/setup-model.js)
- [scripts/postinstall-checks.js](scripts/postinstall-checks.js)
- [.env.example](.env.example)
- [README.md](README.md)
- [RULES.md](RULES.md)
- [CLAUDE.md](CLAUDE.md)

## Overview

The server exposes an OpenAI-compatible endpoint for chat completions and manages local model assets using a simple, reproducible pipeline:
1) Download model artifacts from Hugging Face into ./models
2) Attempt to convert those artifacts into a llama.cpp-compatible GGUF file
3) Load the GGUF at runtime using node-llama-cpp and serve inference

Configuration is fully environment-driven via .env. Swapping models does not require code changes: update .env, run setup (if needed), and restart.

## Components

- Server (Express)
  - Entry point: [index.js](index.js)
  - Endpoints:
    - Health: [JavaScript.app.get()](index.js:79)
    - OpenAI chat completions (non-stream): [JavaScript.app.post()](index.js:90)
- Inference runtime
  - node-llama-cpp bindings to llama.cpp
  - Model/context/session singletons lazily initialized in [JavaScript.initModel()](index.js:42)
- Model lifecycle scripts
  - Downloader: [JavaScript.require()](scripts/download-model.js:22) (HTTP download via Hugging Face API)
  - Converter (attempt): [JavaScript.require()](scripts/convert-model.js:24) (llama.cpp convert-hf-to-gguf.py)
  - Orchestrator: [JavaScript.require()](scripts/setup-model.js:23) (idempotent sequence)
  - Postinstall checks: [JavaScript.require()](scripts/postinstall-checks.js:12)
- Documentation and rules
  - Operator docs: [README.md](README.md)
  - Architecture (this file): [ARCHITECTURE.md](ARCHITECTURE.md)
  - Developer log/decisions: [CLAUDE.md](CLAUDE.md)
  - Binding rules/process: [RULES.md](RULES.md)

## Configuration (.env-driven)

All key behaviors are controlled via .env (see [`.env.example`](.env.example)):
- Server
  - PORT (default 3000)
- Model selection (swappable)
  - MODEL_REPO_ID (e.g., Qwen/Qwen3-VL-2B-Thinking-FP8)
  - MODELS_DIR (default ./models)
  - OUT_GGUF_NAME (default model.gguf)
  - MODEL_GGUF_PATH (optional absolute override)
- Inference parameters
  - CTX_SIZE, MAX_TOKENS, TEMPERATURE
- Download/convert options
  - HF_TOKEN, DOWNLOAD_CONCURRENCY, HF_FILE_PATTERNS
  - PYTHON_EXE, LLAMACPP_DIR, LLAMACPP_CONVERTER, CONVERT_ARGS

Swapping models
1) Edit .env (MODEL_REPO_ID and/or MODEL_GGUF_PATH)
2) Run setup if you need to download/convert: npm run setup-model
3) Restart the server

## Model lifecycle: Download → Convert → Serve

1) Download from Hugging Face
   - Command: npm run download-model
   - Script: [JavaScript.require()](scripts/download-model.js:22)
   - Uses Hugging Face model list API to enumerate files; downloads selected files to ./models/<org>/<repo> with basic resume/skip-by-size.

2) Convert to GGUF (attempt)
   - Command: npm run convert-model
   - Script: [JavaScript.require()](scripts/convert-model.js:24)
   - Acquires llama.cpp’s convert-hf-to-gguf.py (auto-clone if needed)
   - Detects unsupported formats (e.g., TensorFlow SavedModel for VLM) and fails fast with guidance

3) Orchestrate (idempotent)
   - Command: npm run setup-model
   - Script: [JavaScript.require()](scripts/setup-model.js:23)
   - Skips work if GGUF already exists at the target location

4) Serve
   - Command: npm start
   - Server: [JavaScript.app.listen()](index.js:165)
   - On boot, the server attempts to initialize the model via [JavaScript.initModel()](index.js:42)
   - If GGUF is missing, inference returns 503 with guidance; health endpoint remains available

Note about VLM conversion
- The default model Qwen/Qwen3-VL-2B-Thinking-FP8 is a TF FP8 vision-language model; llama.cpp GGUF conversion primarily targets text LLMs (Transformers/PyTorch). The converter detects TF SavedModel artifacts and fails fast with alternatives (use supported text models or prebuilt GGUF and set MODEL_GGUF_PATH).

## Request flow (OpenAI-compatible)

1) Client sends POST /v1/chat/completions to [JavaScript.app.post()](index.js:90)
2) Server validates input and synthesizes a simple prompt from messages
3) node-llama-cpp session runs prompt with configured sampling params
4) Server returns a non-streaming response conforming to OpenAI’s shape (choices[0].message.content)

Readiness and observability
- Health: [JavaScript.app.get()](index.js:79) returns modelReady along with modelPath and optional error
- Logs: server prints model initialization and path; structured logging and redaction are planned (see TODO)

## Error handling

Current
- Missing model GGUF: returns 503 with actionable guidance (run setup-model or set MODEL_GGUF_PATH)
- Conversion unsupported: scripts fail fast and print alternatives

Planned (see [TODO.md](TODO.md))
- Structured logs with redaction
- Timeouts and better error categories
- Basic request/response tracing

## Extensibility

- Swap models via .env without code changes
- Pluggable download filters (HF_FILE_PATTERNS)
- Conversion parameters via CONVERT_ARGS
- MODEL_GGUF_PATH override to use prebuilt GGUFs directly

## Data and directories

- ./models: local model cache (ignored by git)
- node_modules: dependencies (ignored by git)
- No large artifacts are committed; all binaries are downloaded/generated locally per [RULES.md](RULES.md)

## Compliance notes

- All changes must be documented (see [RULES.md](RULES.md))
- Always commit and push after meaningful progress
- Keep [ARCHITECTURE.md](ARCHITECTURE.md) authoritative for flows; update when code/data paths change
