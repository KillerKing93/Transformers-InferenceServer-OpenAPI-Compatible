/**
 * Qwen3-VL-2B-Thinking-FP8 downloader
 * - Lists files from Hugging Face model repo
 * - Downloads into ./models/<repoId>/<files>
 * - Skips files that already exist with matching size (best-effort)
 * - Supports HF_TOKEN for private/gated models
 * - Concurrency controlled via DOWNLOAD_CONCURRENCY
 *
 * Usage:
 *   node scripts/download-model.js
 *   node scripts/download-model.js Qwen/Qwen3-VL-2B-Thinking-FP8
 *
 * Environment:
 *   MODEL_REPO_ID=Qwen/Qwen3-VL-2B-Thinking-FP8
 *   MODELS_DIR=./models
 *   HF_TOKEN=<hf_token_optional>
 *   DOWNLOAD_CONCURRENCY=3
 *   HF_FILE_PATTERNS="*.json,*.md,*.safetensors,*.bin,*.pb,variables/*"
 *   DRY_RUN=1  (only list planned downloads)
 */

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const https = require('https');
const { URL } = require('url');

const REPO_DEFAULT = 'Qwen/Qwen3-VL-2B-Thinking-FP8';
const repoId = (process.argv[2] || process.env.MODEL_REPO_ID || REPO_DEFAULT).trim();
const modelsRoot = path.resolve(process.env.MODELS_DIR || 'models');
const outDir = path.join(modelsRoot, repoId.replace(/\//g, path.sep));
const hfToken = process.env.HF_TOKEN || '';
const concurrency = Math.max(1, parseInt(process.env.DOWNLOAD_CONCURRENCY || '3', 10));
const dryRun = !!process.env.DRY_RUN;
const filePatterns = (process.env.HF_FILE_PATTERNS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

logInfo(`Repo: ${repoId}`);
logInfo(`Output dir: ${outDir}`);
if (filePatterns.length) logInfo(`Patterns: ${filePatterns.join(', ')}`);
if (dryRun) logWarn('DRY_RUN is enabled — no files will be downloaded.');

(async () => {
  try {
    await ensureDir(outDir);
    const files = await listRepoFiles(repoId);
    const filtered = filterFiles(files, filePatterns);

    if (filtered.length === 0) {
      logWarn('No files matched. Consider adjusting HF_FILE_PATTERNS or removing it to download all files.');
      process.exit(0);
    }

    logInfo(`Planned files: ${filtered.length}`);
    for (const f of filtered) {
      logInfo(` - ${f}`);
    }

    if (dryRun) {
      logInfo('DRY_RUN complete.');
      process.exit(0);
    }

    await downloadAll(repoId, filtered, outDir, concurrency);

    logSuccess('Download completed successfully.');
    // Hints for next steps
    logInfo('Next: npm run convert-model  (attempt GGUF conversion)');
    process.exit(0);
  } catch (err) {
    logError(err.message || String(err));
    process.exit(1);
  }
})();

/**
 * List repo files via Hugging Face API
 * https://huggingface.co/api/models/{repoId}
 */
async function listRepoFiles(repo) {
  const apiUrl = new URL(`https://huggingface.co/api/models/${encodeURIComponent(repo)}`);
  const json = await fetchJSON(apiUrl, {
    headers: baseHeaders(),
  });
  // The API returns "siblings" with rfilename
  if (!json || !Array.isArray(json.siblings)) {
    throw new Error('Unexpected HF API response. Cannot list files.');
  }
  const files = [];
  for (const s of json.siblings) {
    if (s && s.rfilename) {
      files.push(s.rfilename);
    }
  }
  // Note: this only lists top-level. For nested structures, rfilename includes the path.
  return files;
}

/**
 * Filter filenames by simple glob patterns (supports "*" wildcard only, no "**")
 */
function filterFiles(files, patterns) {
  if (!patterns || patterns.length === 0) return files;
  return files.filter((file) => patterns.some((p) => minimatchSimple(file, p)));
}

function minimatchSimple(str, pattern) {
  // Convert very simple glob "*" to a regex
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
  const re = new RegExp(`^${escaped}$`, 'i');
  return re.test(str);
}

/**
 * Download all files with a small concurrency pool
 */
async function downloadAll(repo, files, destDir, poolSize) {
  const tasks = files.slice();
  let active = 0;
  let resolveAll, rejectAll;
  const results = [];

  await new Promise((resolve, reject) => {
    resolveAll = resolve;
    rejectAll = reject;

    const pump = () => {
      if (tasks.length === 0 && active === 0) return resolveAll(results);
      while (active < poolSize && tasks.length) {
        const fileRel = tasks.shift();
        active++;
        downloadOne(repo, fileRel, destDir)
          .then((r) => results.push(r))
          .catch((err) => {
            logError(`Failed: ${fileRel} -> ${err.message || err}`);
            results.push({ file: fileRel, ok: false, error: err.message || String(err) });
          })
          .finally(() => {
            active--;
            setImmediate(pump);
          });
      }
    };
    pump();
  });

  // Count successes/failures
  const okCount = results.filter((r) => r.ok).length;
  const failCount = results.length - okCount;
  logInfo(`Downloaded OK: ${okCount}, Failed: ${failCount}`);
  if (failCount > 0) {
    logWarn('Some files failed to download. You may re-run to retry.');
  }
}

/**
 * Download a single file
 * - Skips if existing file matches remote size (best-effort)
 */
async function downloadOne(repo, fileRel, destDir) {
  const remoteUrl = new URL(
    `https://huggingface.co/${encodeURIComponent(repo)}/resolve/main/${encodeURIPath(fileRel)}?download=true`
  );
  const localPath = path.join(destDir, fileRel);
  await ensureDir(path.dirname(localPath));

  // Try HEAD to get length
  let remoteSize = 0;
  try {
    remoteSize = await headContentLength(remoteUrl, { headers: baseHeaders() });
  } catch {
    // Some endpoints may not support HEAD; proceed without it.
  }

  if (await fileHasSize(localPath, remoteSize)) {
    logInfo(`Skip (exists, size matches): ${fileRel}`);
    return { file: fileRel, ok: true, skipped: true };
  }

  await downloadToFile(remoteUrl, localPath, { headers: baseHeaders() });
  // Verify size if available
  if (remoteSize > 0) {
    const stat = await fsStatSafe(localPath);
    if (!stat || stat.size !== remoteSize) {
      throw new Error(`Size mismatch after download: expected=${remoteSize}, got=${stat ? stat.size : 'N/A'}`);
    }
  }

  logSuccess(`Downloaded: ${fileRel}`);
  return { file: fileRel, ok: true, skipped: false };
}

/**
 * Helpers
 */
function baseHeaders() {
  const h = {
    'User-Agent': 'nodejs-llamacpp-inferenceserver/1.0',
    Accept: 'application/json, text/plain, */*',
  };
  if (hfToken) {
    h.Authorization = `Bearer ${hfToken}`;
  }
  return h;
}

function ensureDir(dir) {
  return fs.promises.mkdir(dir, { recursive: true });
}

function fetchJSON(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, { method: 'GET', ...options }, (res) => {
      const { statusCode } = res;
      let data = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        if (statusCode && statusCode >= 200 && statusCode < 300) {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(new Error(`Invalid JSON from HF API: ${e.message}`));
          }
        } else {
          reject(new Error(`HF API error: ${statusCode} ${data}`));
        }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

function headContentLength(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, { method: 'HEAD', ...options }, (res) => {
      const len = parseInt(res.headers['content-length'] || '0', 10);
      if (res.statusCode && res.statusCode >= 200 && res.statusCode < 400) {
        resolve(Number.isFinite(len) ? len : 0);
      } else {
        reject(new Error(`HEAD failed: ${res.statusCode}`));
      }
    });
    req.on('error', reject);
    req.end();
  });
}

function downloadToFile(url, dest, options = {}) {
  return new Promise((resolve, reject) => {
    const tmp = `${dest}.part`;
    const file = fs.createWriteStream(tmp);
    const req = https.request(url, { method: 'GET', ...options }, (res) => {
      if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
        res.pipe(file);
        file.on('finish', () => {
          file.close(async () => {
            try {
              await fs.promises.rename(tmp, dest);
              resolve();
            } catch (e) {
              reject(e);
            }
          });
        });
      } else {
        file.close(() => {
          fs.promises.unlink(tmp).catch(() => {});
          reject(new Error(`GET failed: ${res.statusCode}`));
        });
      }
    });
    req.on('error', (err) => {
      file.close(() => {
        fs.promises.unlink(tmp).catch(() => {});
        reject(err);
      });
    });
    req.end();
  });
}

async function fileHasSize(filePath, size) {
  const stat = await fsStatSafe(filePath);
  if (!stat) return false;
  if (!size || size <= 0) return true; // if unknown remote size, assume ok to skip
  return stat.size === size;
}

async function fsStatSafe(p) {
  try {
    return await fs.promises.stat(p);
  } catch {
    return null;
  }
}

function encodeURIPath(p) {
  // Encode each segment separately to preserve slashes
  return p
    .split('/')
    .map((seg) => encodeURIComponent(seg))
    .join('/');
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