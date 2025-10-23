'use strict';

require('dotenv').config();

const fs = require('fs');
const path = require('path');
const express = require('express');

const PORT = Number(process.env.PORT || 3000);

// Model configuration via .env
const MODEL_REPO_ID = process.env.MODEL_REPO_ID || 'Qwen/Qwen3-VL-2B-Thinking-FP8';
const MODELS_DIR = process.env.MODELS_DIR || 'models';
const OUT_GGUF_NAME = process.env.OUT_GGUF_NAME || 'model.gguf';
const MODEL_GGUF_PATH =
  process.env.MODEL_GGUF_PATH ||
  path.join(MODELS_DIR, MODEL_REPO_ID.replace(/\//g, path.sep), OUT_GGUF_NAME);

// Inference parameters via .env
const CTX_SIZE = Number(process.env.CTX_SIZE || 4096);
const MAX_TOKENS = Number(process.env.MAX_TOKENS || 256);
const TEMPERATURE = Number(process.env.TEMPERATURE || 0.7);

const app = express();
app.use(express.json());

// Lazy model singletons
let modelReady = false;
let modelError = null;
let LlamaModel, LlamaContext, LlamaChatSession;
let model, context, session;

function fileExists(p) {
  try {
    fs.accessSync(p, fs.constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

async function initModel() {
  modelReady = false;
  modelError = null;

  if (!fileExists(MODEL_GGUF_PATH)) {
    modelError = new Error(
      `GGUF not found at ${MODEL_GGUF_PATH}. Run "npm run setup-model" or set MODEL_GGUF_PATH in .env`
    );
    return;
  }

  try {
    const nlc = require('node-llama-cpp');
    LlamaModel = nlc.LlamaModel;
    LlamaContext = nlc.LlamaContext;
    LlamaChatSession = nlc.LlamaChatSession;

    model = new LlamaModel({ modelPath: MODEL_GGUF_PATH });
    context = new LlamaContext({ model, contextSize: CTX_SIZE });
    session = new LlamaChatSession({ context });

    modelReady = true;
    console.log(`[model] Ready using ${MODEL_GGUF_PATH}`);
  } catch (e) {
    modelError = e;
    console.error('[model] init error:', e);
  }
}

// Initialize in background; will retry on demand in API if file appears later
initModel().catch((e) => console.error(e));

// Simple root
app.get('/', (req, res) => {
  res.send('OK');
});

// Health endpoint for readiness
app.get('/health', (req, res) => {
  res.json({
    ok: true,
    modelReady,
    modelPath: MODEL_GGUF_PATH,
    error: modelError ? String(modelError.message || modelError) : null,
  });
});

// OpenAI-compatible chat completions (non-streaming)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!modelReady) {
      // try once more if GGUF appeared after boot
      if (fileExists(MODEL_GGUF_PATH)) {
        await initModel();
      }
    }
    if (!modelReady) {
      return res.status(503).json({
        error: {
          message:
            modelError
              ? String(modelError.message || modelError)
              : `Model not ready. Expected GGUF at ${MODEL_GGUF_PATH}`,
          type: 'service_unavailable',
        },
      });
    }

    const { messages, max_tokens, temperature } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: { message: 'messages must be a non-empty array' },
      });
    }

    // Build a simple prompt from chat messages
    const prompt =
      messages
        .map((m) => {
          const role = m.role || 'user';
          const content =
            Array.isArray(m.content)
              ? m.content.map((c) => (typeof c === 'string' ? c : c.text || '')).join(' ')
              : m.content;
          return `${role}: ${content}`;
        })
        .join('\n') + '\nassistant:';

    const tokLimit = Number.isFinite(max_tokens) ? max_tokens : MAX_TOKENS;
    const temp = Number.isFinite(temperature) ? temperature : TEMPERATURE;

    const output = await session.prompt(String(prompt), {
      maxTokens: tokLimit,
      temperature: temp,
    });

    const now = Math.floor(Date.now() / 1000);
    res.json({
      id: `chatcmpl-${now}-${Math.random().toString(36).slice(2, 10)}`,
      object: 'chat.completion',
      created: now,
      model: MODEL_GGUF_PATH,
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: String(output) },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    });
  } catch (err) {
    console.error('[api] /v1/chat/completions error:', err);
    res.status(500).json({
      error: { message: 'Internal error running inference' },
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
  console.log(`Model: ${MODEL_GGUF_PATH}`);
});
