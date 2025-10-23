/**
 * Convert Hugging Face model to GGUF using llama.cpp convert-hf-to-gguf.py
 *
 * Primary target (per requirements): Qwen/Qwen3-VL-2B-Thinking-FP8 (TensorFlow FP8, VLM)
 * IMPORTANT: This specific model format is likely NOT supported by llama.cpp GGUF conversion.
 * The script will detect TensorFlow SavedModel artifacts (saved_model.pb / variables/*) and fail fast
 * with actionable guidance. If you point this script to a supported HF Transformers (PyTorch) model,
 * it will attempt conversion.
 *
 * Usage:
 *   node scripts/convert-model.js
 *   node scripts/convert-model.js Qwen/Qwen3-VL-2B-Thinking-FP8
 *
 * Environment:
 *   MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Thinking-FP8
 *   MODELS_DIR=./models
 *   OUT_GGUF_NAME=model.gguf
 *   PYTHON_EXE=python (or full path)
 *   LLAMACPP_DIR=./tools/llama.cpp (optional; will auto-clone if missing and git available)
 *   LLAMACPP_CONVERTER=<path to convert-hf-to-gguf.py> (overrides auto-detection)
 *   CONVERT_ARGS="--outtype q8_0" (extra args appended to converter call)
 */

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const REPO_DEFAULT = 'Qwen/Qwen3-VL-2B-Thinking-FP8';
const repoId = (process.argv[2] || process.env.MODEL_REPO_ID || REPO_DEFAULT).trim();
const modelsRoot = path.resolve(process.env.MODELS_DIR || 'models');
const modelDir = path.join(modelsRoot, repoId.replace(/\//g, path.sep));
const outGgufName = process.env.OUT_GGUF_NAME || 'model.gguf';
const llcppDir = path.resolve(process.env.LLAMACPP_DIR || path.join('tools', 'llama.cpp'));
const pythonExe = process.env.PYTHON_EXE || 'python';
const extraArgs = (process.env.CONVERT_ARGS || '').trim();
const converterOverride = process.env.LLAMACPP_CONVERTER || '';

(async () => {
  try {
    assertExists(modelDir, `Model directory not found. Did you run "npm run download-model"? Expected: ${modelDir}`);

    // Detect TensorFlow SavedModel (commonly used by TF VLM exports like Qwen3-VL FP8)
    const tfSavedModel = fs.existsSync(path.join(modelDir, 'saved_model.pb'));
    const tfVariables = fs.existsSync(path.join(modelDir, 'variables'));
    if (tfSavedModel || tfVariables) {
      failUnsupportedTF(repoId, modelDir);
    }

    // Detect Transformers-style files (config.json, tokenizer.json, pytorch_model*.bin / safetensors)
    const transformersLikely = hasAny(modelDir, [
      'config.json',
      'tokenizer.json',
      'tokenizer.model',
      'pytorch_model.bin',
      'pytorch_model-00001-of-0000*.bin',
      'model.safetensors',
      'model-00001-of-0000*.safetensors',
    ]);

    if (!transformersLikely) {
      logWarn('This model does not look like a typical Transformers (PyTorch) export. Conversion may fail.');
    }

    const converter = await ensureConverterScript();
    const outDir = modelDir;
    const outFile = path.join(outDir, outGgufName);

    // Build converter args
    // Minimal safe defaults; callers can pass CONVERT_ARGS to customize
    const argv = [converter];
    if (extraArgs) {
      // naive split on spaces; for complex args, set via env variable appropriately quoted
      argv.push(...extraArgs.split(' ').filter(Boolean));
    }
    // Many models work with calling: python convert-hf-to-gguf.py /path/to/model -o /path/to/out.gguf
    argv.push(modelDir, '-o', outFile);

    // Run conversion
    logInfo(`Running converter: ${pythonExe} ${argv.map((a) => (a.includes(' ') ? `"${a}"` : a)).join(' ')}`);
    ensurePythonAvailable();
    const run = spawnSync(pythonExe, argv, { stdio: 'inherit' });

    if (run.error) {
      throw run.error;
    }
    if (run.status !== 0) {
      throw new Error(`Converter exited with code ${run.status}. See logs above for details.`);
    }

    assertExists(outFile, `Converter finished but GGUF not found at: ${outFile}`);

    logSuccess(`GGUF conversion complete: ${outFile}`);
    logInfo('Next: configure server to use this GGUF or run "npm start" to serve if already wired.');
    process.exit(0);
  } catch (err) {
    logError(err.message || String(err));
    logInfo('If you are trying to convert Qwen3-VL TF FP8, consider using a text-only Qwen model with Transformers weights that is known to convert to GGUF, or download a prebuilt GGUF.');
    process.exit(2);
  }
})();

/**
 * Ensures the convert-hf-to-gguf.py script is available.
 * Resolution order:
 *  1) LLAMACPP_CONVERTER env path if provided.
 *  2) ./tools/llama.cpp/scripts/convert-hf-to-gguf.py (auto-clone llama.cpp if missing and git is available).
 */
async function ensureConverterScript() {
  if (converterOverride) {
    const p = path.resolve(converterOverride);
    assertExists(p, `LLAMACPP_CONVERTER not found at: ${p}`);
    return p;
  }
  const candidate = path.join(llcppDir, 'convert-hf-to-gguf.py'); // recent repo moved script to repo root
  const altCandidate = path.join(llcppDir, 'scripts', 'convert-hf-to-gguf.py'); // older path
  if (fs.existsSync(candidate)) return candidate;
  if (fs.existsSync(altCandidate)) return altCandidate;

  logInfo('llama.cpp repo not found locally; attempting to clone a shallow copy to acquire the converter script...');
  ensureGitAvailable();
  fs.mkdirSync(path.dirname(llcppDir), { recursive: true });
  const gitUrl = 'https://github.com/ggerganov/llama.cpp.git';
  const clone = spawnSync('git', ['clone', '--filter=blob:none', '--depth', '1', gitUrl, llcppDir], { stdio: 'inherit' });
  if (clone.error) throw clone.error;
  if (clone.status !== 0) throw new Error(`git clone exited with code ${clone.status}.`);

  // Check both possible locations
  if (fs.existsSync(candidate)) return candidate;
  if (fs.existsSync(altCandidate)) return altCandidate;

  throw new Error('convert-hf-to-gguf.py not found after cloning llama.cpp. Repository layout may have changed.');
}

function ensurePythonAvailable() {
  const probe = spawnSync(pythonExe, ['--version'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) {
    throw new Error(`Python not available/executable: ${pythonExe}. Install Python 3.10+ and ensure it is on PATH, or set PYTHON_EXE.`);
  }
}

function ensureGitAvailable() {
  const probe = spawnSync('git', ['--version'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) {
    throw new Error('git is required to auto-fetch llama.cpp. Install Git or set LLAMACPP_CONVERTER to an existing convert-hf-to-gguf.py path.');
  }
}

function assertExists(p, msgIfMissing) {
  if (!fs.existsSync(p)) {
    throw new Error(msgIfMissing);
  }
}

function hasAny(baseDir, patterns) {
  // Very simple presence checks (wildcards supported only at the end)
  for (const p of patterns) {
    if (p.includes('*')) {
      const prefix = p.split('*')[0];
      const dir = path.dirname(prefix);
      const fnamePrefix = path.basename(prefix);
      const fullDir = path.join(baseDir, dir === '.' ? '' : dir);
      if (fs.existsSync(fullDir)) {
        const entries = fs.readdirSync(fullDir);
        if (entries.some((e) => e.startsWith(fnamePrefix))) return true;
      }
    } else {
      if (fs.existsSync(path.join(baseDir, p))) return true;
    }
  }
  return false;
}

function failUnsupportedTF(repo, dir) {
  const msg = [
    `Detected TensorFlow SavedModel artifacts in: ${dir}`,
    `The requested model "${repo}" (TF FP8 VLM) is likely NOT supported by llama.cpp GGUF converter.`,
    'Actionable alternatives:',
    '  - Use a text-only Qwen/Qwen2/Qwen2.5 model with Transformers (PyTorch) weights for GGUF conversion.',
    '  - Or download a prebuilt GGUF compatible with llama.cpp.',
    '  - If you require VLM, consult llama.cpp upstream for current multimodal support status.',
  ].join('\n');
  throw new Error(msg);
}

function logInfo(msg) {
  console.log(`[INFO] ${msg}`);
}
function logWarn(msg) {
  console.warn(`[WARN] ${msg}`);
}
function logError(msg) {
  console.error(`[ERROR] ${msg}`);
}
function logSuccess(msg) {
  console.log(`[OK] ${msg}`);
}