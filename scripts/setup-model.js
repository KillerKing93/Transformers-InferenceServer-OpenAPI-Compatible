/**
 * Orchestrate model setup:
 * 1) Download from Hugging Face into ./models/<repoId>
 * 2) Attempt conversion to GGUF via llama.cpp converter
 * 3) Idempotent: skips steps when artifacts already present
 *
 * Usage:
 *   node scripts/setup-model.js
 *   node scripts/setup-model.js Qwen/Qwen3-VL-2B-Thinking-FP8
 *
 * Environment:
 *   MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Thinking-FP8
 *   MODELS_DIR=./models
 *   OUT_GGUF_NAME=model.gguf
 *   HF_TOKEN=<optional HF token>
 *   HF_FILE_PATTERNS="saved_model.pb,variables/*,*.json,*.md" (defaults provided below)
 *   PYTHON_EXE=python
 *   LLAMACPP_DIR=./tools/llama.cpp
 *   LLAMACPP_CONVERTER=<path to convert-hf-to-gguf.py>
 *   CONVERT_ARGS="--outtype q8_0"
 */

const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const REPO_DEFAULT = 'Qwen/Qwen3-VL-2B-Thinking-FP8';
const repoId = (process.argv[2] || process.env.MODEL_REPO_ID || REPO_DEFAULT).trim();
const modelsRoot = path.resolve(process.env.MODELS_DIR || 'models');
const modelDir = path.join(modelsRoot, repoId.replace(/\//g, path.sep));
const outGgufName = process.env.OUT_GGUF_NAME || 'model.gguf';
const ggufPath = path.join(modelDir, outGgufName);

(async () => {
  try {
    banner('MODEL SETUP START');

    // Step 0: If GGUF already exists, skip all
    if (fs.existsSync(ggufPath)) {
      info(`GGUF already exists: ${ggufPath}`);
      info('Nothing to do. You may start the server.');
      done();
      return;
    }

    // Step 1: Download
    ensureScriptExists('scripts/download-model.js');
    const dlEnv = {
      ...process.env,
      MODEL_REPO_ID: repoId,
      MODELS_DIR: modelsRoot,
      // sensible defaults for TF SavedModel-based repos; user can override
      HF_FILE_PATTERNS:
        process.env.HF_FILE_PATTERNS ||
        'saved_model.pb,variables/*,*.json,*.md,*.txt,*.model,*.bin,*.safetensors',
    };

    banner('DOWNLOAD PHASE');
    runNodeScript('scripts/download-model.js', [repoId], { env: dlEnv });

    // Step 2: Convert to GGUF
    ensureScriptExists('scripts/convert-model.js');
    const cvEnv = {
      ...process.env,
      MODEL_REPO_ID: repoId,
      MODELS_DIR: modelsRoot,
      OUT_GGUF_NAME: outGgufName,
    };

    banner('CONVERSION PHASE');
    runNodeScript('scripts/convert-model.js', [repoId], { env: cvEnv });

    // Verify
    if (!fs.existsSync(ggufPath)) {
      throw new Error(
        `Conversion reported success but GGUF not found at: ${ggufPath}`
      );
    }

    banner('MODEL SETUP COMPLETE');
    success(`Ready: ${ggufPath}`);
    info('You can now run: npm start');
    done();
  } catch (err) {
    error(err.message || String(err));
    info('See CLAUDE.md and README.md for guidance and alternatives if conversion is unsupported.');
    process.exit(1);
  }
})();

function runNodeScript(scriptRel, args = [], options = {}) {
  const abs = path.resolve(scriptRel);
  const res = spawnSync(process.execPath, [abs, ...args], {
    stdio: 'inherit',
    ...options,
  });
  if (res.error) throw res.error;
  if (res.status !== 0) {
    throw new Error(`${scriptRel} exited with code ${res.status}`);
  }
}

function ensureScriptExists(rel) {
  const abs = path.resolve(rel);
  if (!fs.existsSync(abs)) {
    throw new Error(`Required script not found: ${rel}`);
  }
}

function banner(msg) {
  console.log(`\n========== ${msg} ==========\n`);
}
function info(msg) {
  console.log(`[INFO] ${msg}`);
}
function success(msg) {
  console.log(`[OK] ${msg}`);
}
function error(msg) {
  console.error(`[ERROR] ${msg}`);
}
function done() {
  console.log('');
}