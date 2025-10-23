/**
 * Postinstall environment checks
 * - Verifies Node.js version (>= 20)
 * - Warns if Python (>= 3.10) and git are missing (needed for model conversion / llama.cpp tooling)
 * - Prints .env guidance for easy model switching
 *
 * Skips checks when NO_POSTINSTALL_CHECKS=1
 *
 * This script should not fail installs unless Node is too old. Other tools are warned, not enforced.
 */

require('dotenv').config();
const { spawnSync } = require("child_process");

const SKIP = process.env.NO_POSTINSTALL_CHECKS === "1";

if (SKIP) {
  info("NO_POSTINSTALL_CHECKS=1 set, skipping postinstall checks.");
  process.exit(0);
}

// 1) Node version check (hard requirement)
try {
  const version = process.versions.node || "";
  const major = parseInt(version.split(".")[0] || "0", 10);
  if (Number.isFinite(major) && major < 20) {
    error(`Node.js ${version} detected. This project requires Node.js >= 20.`);
    error("Please upgrade Node and re-run npm install.");
    process.exit(1);
  } else {
    ok(`Node.js ${version} OK (>= 20)`);
  }
} catch (e) {
  warn(`Unable to detect Node.js version: ${e.message || e}`);
}

// 2) Python availability (soft requirement for GGUF conversion)
try {
  const py = process.env.PYTHON_EXE || "python";
  const out = spawnSync(py, ["--version"], { encoding: "utf8" });
  if (out.status === 0) {
    const v = (out.stdout || out.stderr || "").trim();
    ok(`Python available: ${v} (required for convert-model)`);
  } else {
    warn(
      "Python not found on PATH. GGUF conversion may fail. Set PYTHON_EXE or install Python >= 3.10."
    );
  }
} catch {
  warn(
    "Python not found on PATH. GGUF conversion may fail. Set PYTHON_EXE or install Python >= 3.10."
  );
}

// 3) git availability (soft requirement to auto-fetch llama.cpp converter or LFS model pulls)
try {
  const git = spawnSync("git", ["--version"], { encoding: "utf8" });
  if (git.status === 0) {
    ok(`git available: ${(git.stdout || "").trim()}`);
  } else {
    warn(
      "git not found on PATH. Auto-fetching llama.cpp or LFS-based downloads may fail."
    );
  }
} catch {
  warn(
    "git not found on PATH. Auto-fetching llama.cpp or LFS-based downloads may fail."
  );
}

// 4) Quick .env guidance for easy model switching
printEnvGuidance();

process.exit(0);

/* Helpers */

function printEnvGuidance() {
  const lines = [
    "",
    "Environment configuration (.env) tips:",
    "  # Server",
    "  PORT=3000",
    "",
    "  # Default model repo and output locations",
    "  MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Thinking-FP8",
    "  MODELS_DIR=./models",
    "  OUT_GGUF_NAME=model.gguf",
    "  # You can also point directly to a .gguf file, which overrides the above",
    "  # MODEL_GGUF_PATH=./models/Qwen/Qwen3-VL-2B-Thinking-FP8/model.gguf",
    "",
    "  # Inference params",
    "  CTX_SIZE=4096",
    "  MAX_TOKENS=256",
    "  TEMPERATURE=0.7",
    "",
    "  # Optional setup behavior",
    "  # AUTO_SETUP=1               # (future option) auto-run setup on server start if model missing",
    "  # HF_TOKEN=                   # set if repo is gated",
    "  # PYTHON_EXE=python",
    "  # LLAMACPP_DIR=./tools/llama.cpp",
    "  # LLAMACPP_CONVERTER=<path to convert-hf-to-gguf.py>",
    "  # CONVERT_ARGS=--outtype q8_0",
    "",
    "After setting .env, you can run:",
    "  npm run setup-model   # downloads then attempts conversion",
    "  npm start             # starts the server; /health and /v1/chat/completions are available",
    "",
  ];
  info(lines.join("\n"));
}

function ok(msg) {
  console.log(`[OK] ${msg}`);
}
function info(msg) {
  console.log(`[INFO] ${msg}`);
}
function warn(msg) {
  console.warn(`[WARN] ${msg}`);
}
function error(msg) {
  console.error(`[ERROR] ${msg}`);
}
