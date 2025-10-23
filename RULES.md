# Project Rules and Workflow

Authoritative rules for this repository. These rules are binding for all changes and must be enforced on every iteration.

Files referenced below:
- [RULES.md](RULES.md)
- [CLAUDE.md](CLAUDE.md)
- [README.md](README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TODO.md](TODO.md)
- [.gitignore](.gitignore)
- [index.js](index.js)

## 1) Documentation rules (must-do on every change)
- Always document the change:
  - Add a short, operator-facing summary to [README.md](README.md).
  - Add detailed, developer-facing notes to [CLAUDE.md](CLAUDE.md) (what changed, why, alternatives considered, caveats).
  - If the change touches design or data flow, update [ARCHITECTURE.md](ARCHITECTURE.md).
  - If the change introduces new actionable work or completes existing tasks, update [TODO.md](TODO.md).
- Never skip documentation. If a change is reverted, document the revert as well.
- Keep documentation up-to-date with the code at all times (no stale docs).

Minimum documentation checklist per change:
- What was changed and where (filenames and sections).
- Why the change was made (problem or requirement).
- How to operate or verify (commands, endpoints, examples).
- Follow-ups or known limitations.

## 2) Git discipline (commit and push cadence)
- After every meaningful progress:
  - Stage, commit, and push: 
    - Windows CMD example:
      - git add .
      - git commit -m "type(scope): short description"
      - git push
- If no remote is configured, still commit locally and add a note in [CLAUDE.md](CLAUDE.md) that push was skipped.
- Use conventional commit style where possible:
  - chore, docs, feat, fix, refactor, perf, test, build, ci
  - Example: feat(api): add POST /v1/chat/completions
- Keep commits small and atomic; one concern per commit.
- Always include references to files touched in the commit body where appropriate.

## 3) Large artifacts policy (.gitignore)
- The following directories must never be committed and must be ignored:
  - node_modules
  - models
- These are already present in [.gitignore](.gitignore). If additional large or auto-generated artifacts appear, add them to [.gitignore](.gitignore) and document the rationale in [CLAUDE.md](CLAUDE.md).
- For any large file that can be downloaded or generated (models, caches, build outputs), prefer automatic retrieval/generation scripts rather than committing binaries.

## 4) Model lifecycle: download, convert, use
Goal: Automatically download the Qwen3 VL model, convert to GGUF, and use it via llama.cpp bindings.

Target model:
- Repository: https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8
- Format: TensorFlow FP8 (vision-language). This format/model may have limited support in llama.cpp GGUF conversion.

Rules:
- Provide commands/scripts to:
  - Download model artifacts into ./models/Qwen3-VL-2B-Thinking-FP8
  - Attempt conversion to GGUF using llama.cpp’s convert-hf-to-gguf.py
  - Select the converted .gguf for inference
- Prerequisites (document clearly in [README.md](README.md)):
  - Node.js 20+
  - Python 3.10+ with pip
  - git and git-lfs (for Hugging Face model pulls)
  - llama.cpp conversion script availability (either installed locally or auto-fetched)
- Unsupported-case handling:
  - Qwen3-VL is a multimodal VLM and the TensorFlow FP8 variant may not convert to GGUF with standard llama.cpp tooling.
  - If conversion is not supported, the script must:
    - Fail fast with a clear message
    - Print actionable alternatives (e.g., supported weights, manual steps)
    - Record details in [CLAUDE.md](CLAUDE.md) and [README.md](README.md)
- Never commit downloaded or converted models; they must remain in ./models (ignored by git).

## 5) API contract
- Provide an OpenAI-compatible endpoint:
  - POST /v1/chat/completions
- Minimum behavior:
  - Accept model and messages per OpenAI spec
  - Provide a non-streamed completion response initially
  - Add streaming later if applicable
- Validate inputs, handle timeouts, and return structured errors.
- Ensure the endpoint and its shape are documented in [README.md](README.md).

## 6) Logging and error handling
- Use structured logging (level, timestamp, correlation id).
- Redact sensitive request fields.
- Log:
  - Request received (without sensitive data)
  - Model selection and load status
  - Inference start/stop and token counts (if available)
  - Errors with actionable messages
- Document log conventions in [README.md](README.md).

## 7) Architecture documentation
- Keep [ARCHITECTURE.md](ARCHITECTURE.md) authoritative for:
  - Startup flow
  - Model download/convert pipeline
  - Inference path from HTTP to model
  - Error/timeout handling
  - Extensibility points (e.g., swapping models)
- Update whenever code paths or data flows change.

## 8) TODO hygiene
- All planned work must be tracked in [TODO.md](TODO.md).
- Update statuses as soon as work starts or completes.
- When new tasks are discovered mid-implementation, add them immediately, then continue.

## 9) Operational requirements and environment
- Node.js: >= 20 (required by express@5 and node-llama-cpp)
- Windows 11 supported; Linux/macOS should work with equivalent tooling.
- Python: >= 3.10 for conversion scripts.
- Build chain: node-llama-cpp may trigger native builds; ensure CMake and compiler toolchain are present or that prebuilt binaries exist.

## 10) File responsibilities overview
- [index.js](index.js): Express server, API routing, model lifecycle triggers, inference invocation.
- [README.md](README.md): How to install, run, configure, download/convert models, and call the API.
- [ARCHITECTURE.md](ARCHITECTURE.md): System design, component interactions, sequences.
- [TODO.md](TODO.md): Work plan and statuses.
- [CLAUDE.md](CLAUDE.md): Deep-dive notes, change log, important rules mirror, decisions.
- [RULES.md](RULES.md): You are here; do not drift from these rules.
- [.gitignore](.gitignore): Must exclude large/auto-generated directories (node_modules, models).

## 11) Workflow example (single iteration)
1) Make a small, isolated change (e.g., add a script for model download).
2) Update docs:
   - [CLAUDE.md](CLAUDE.md): What/why/how
   - [README.md](README.md): How to use it
   - [ARCHITECTURE.md](ARCHITECTURE.md): If it affects flows
   - [TODO.md](TODO.md): Update status
3) Commit and push:
   - git add .
   - git commit -m "feat(model): add auto-download script for Qwen3-VL-2B-Thinking-FP8"
   - git push
4) Verify locally, record any issues or follow-ups in [CLAUDE.md](CLAUDE.md).

## 12) Compliance checklist (pre-merge / pre-push)
- Code compiles and runs locally.
- Docs updated ([README.md](README.md), [CLAUDE.md](CLAUDE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [TODO.md](TODO.md)).
- No large artifacts added to git.
- Commit message follows conventional style.
- Endpoint contract honored and validated.
