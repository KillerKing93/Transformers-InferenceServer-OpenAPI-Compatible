# CLAUDE Technical Log and Decisions

This document is the developer-facing changelog, decisions log, and notes for this repository. It mirrors key rules and captures the rationale behind implementation choices. Operator-facing usage belongs in [README.md](README.md); architecture belongs in [ARCHITECTURE.md](ARCHITECTURE.md); binding rules live in [RULES.md](RULES.md); task tracking lives in [TODO.md](TODO.md).

Repository links:
- [RULES.md](RULES.md)
- [README.md](README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TODO.md](TODO.md)
- [index.js](index.js)

Summary of binding rules (mirror)
Refer to [RULES.md](RULES.md) for authoritative wording. Key obligations:
- Always document changes in [README.md](README.md), [CLAUDE.md](CLAUDE.md), and—when design is affected—[ARCHITECTURE.md](ARCHITECTURE.md). Update [TODO.md](TODO.md) statuses immediately.
- After every progress step, run git add ., git commit, and git push. Use conventional commit messages.
- Never commit large artifacts. Keep node_modules and models out of git via [.gitignore](.gitignore).
- Provide OpenAI-compatible POST /v1/chat/completions, with validation, logging, and error handling.
- Maintain Node.js >= 20, Python >= 3.10 for conversion tooling.

2025-10-23 – Change log
- Added [RULES.md](RULES.md) capturing workflow, documentation discipline, large-artifact policy, API contract, logging, and architecture/documentation responsibilities.
- Established plan to auto-download the model from Hugging Face, convert to GGUF, and load it at runtime via node-llama-cpp. Tracked in [TODO.md](TODO.md).
- Noted potential format support caveat: Target model is Qwen/Qwen3-VL-2B-Thinking-FP8 (TensorFlow, vision-language). Standard llama.cpp GGUF conversion primarily targets LLaMA/Qwen LLM weights in PyTorch/transformers formats; VLM TensorFlow FP8 may be unsupported. Implementation will detect and fail fast with guidance if conversion is not viable.

Model lifecycle design (download → convert → load)
- Download location: ./models/Qwen3-VL-2B-Thinking-FP8 (ignored by git).
- Download approach: node-based downloader with resume and checksum where available. Prefer direct file pulls from Hugging Face via HTTPS; fallback to git-lfs if installed.
- Conversion: attempt llama.cpp convert-hf-to-gguf.py with appropriate flags for Qwen models. If the TensorFlow FP8 layout is unsupported, emit a clear error and suggest alternatives (e.g., Qwen2/Qwen2.5 GGUF releases).
- Loading: configure node-llama-cpp to load the produced .gguf path; pass model options (context length, GPU layers) via environment variables.

Planned scripts and npm hooks
- download-model: downloads artifacts to ./models
- convert-model: runs conversion to GGUF (Python dependency)
- setup-model: orchestrates download + convert; idempotent and safe to re-run
- start/dev: runs server; server verifies presence of GGUF and prompts to run setup-model if missing
- postinstall: optional guard to warn if Node & Python versions are insufficient

Pre-requisites (developer machine)
- Node.js 20+
- Python 3.10+
- git and git-lfs recommended for large files
- CMake/Build tools may be required by node-llama-cpp if prebuilt binaries are not used

API contract notes
- Implement OpenAI-compatible endpoint: POST /v1/chat/completions
- Minimal viable response: non-streaming choices array with a single message
- Validation: ensure messages is an array; model is accepted but overridden by configured model if needed
- Logging: structured logs with redaction and correlation id

Error handling and observability
- Detect missing model or missing GGUF; return 503 with actionable message and include next-steps hint to run npm run setup-model
- Timeouts for long inferences; return 504 with details
- Wrap node-llama-cpp operations with try/catch and clear error mapping

License and metadata note
- LICENSE is a modified Apache 2.0 with royalty linkage. package.json lists "ISC". Aligning these will be scheduled as a follow-up task to avoid legal ambiguity.

Git discipline template
Commit after each progress step:

Conventional examples:
- feat(model): add auto-download script for Qwen3-VL-2B-Thinking-FP8
- chore(docs): add RULES and mirror in CLAUDE
- feat(api): implement POST /v1/chat/completions
- fix(model): handle unsupported conversion formats with clear guidance

Verification checklist per commit
- Code runs (or is behind a feature flag)
- [README.md](README.md) updated for operators
- [CLAUDE.md](CLAUDE.md) updated for developers
- [ARCHITECTURE.md](ARCHITECTURE.md) updated if flows changed
- [TODO.md](TODO.md) statuses reflect the new state
- No large artifacts were committed

Next actionable items (aligned with TODO)
- Finalize scripts: download-model, convert-model, setup-model (npm run scripts)
- Wire model load and implement /v1/chat/completions in [index.js](index.js)
- Update [ARCHITECTURE.md](ARCHITECTURE.md) with end-to-end flows
- Update [README.md](README.md) with quickstart and troubleshooting

Operational notes for Qwen3-VL TF FP8
- If conversion is unsupported, suggest alternatives in logs and docs:
  - Use an officially provided GGUF for a Qwen text-only model for immediate functionality
  - Or switch to a supported Qwen3 weight format (e.g., PyTorch/transformers) that convert-hf-to-gguf.py supports
- The server will still start and expose health, but return a helpful error for inference until a supported model is installed

Health and diagnostics
- Add GET /health for readiness reporting (model:ready true/false)
- Log model metadata after load (params, vocab size) if available

Contributing discipline
- Never bypass the documentation steps
- Keep commits small and atomic
- Prefer explicit, reproducible scripts over ad-hoc manual steps

End of entry.