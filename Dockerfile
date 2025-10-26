# Use Python 3.12 slim for smaller image
FROM python:3.12-slim

# Install system deps for image/video processing and HF
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Backend selector: cpu | nvidia | amd
ARG BACKEND=cpu
# Pin torch versions per backend index
# - CPU index publishes newer (2.9.0 ok)
# - CUDA cu124 index publishes up to 2.6.0 (auto-resolves to +cu124)
# - ROCm 6.2 index publishes up to 2.5.1+rocm6.2 (must include local tag)
ARG TORCH_VER_CPU=2.9.0
ARG TORCH_VER_NVIDIA=2.6.0
ARG TORCH_VER_AMD=2.5.1+rocm6.2

# Control whether to bake the model into the image (1) or skip and download at runtime (0)
ARG BAKE_MODEL=1

ENV BACKEND=${BACKEND}
ENV BAKE_MODEL=${BAKE_MODEL}
ENV PIP_NO_CACHE_DIR=1

# Install appropriate PyTorch for the selected backend, then the rest
RUN if [ "$BACKEND" = "cpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VER_CPU}; \
    elif [ "$BACKEND" = "nvidia" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==${TORCH_VER_NVIDIA}; \
    elif [ "$BACKEND" = "amd" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/rocm6.2 "torch==${TORCH_VER_AMD}"; \
    else \
      echo "Unsupported BACKEND: $BACKEND" && exit 1; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .
COPY tests/ tests/

# Copy env template (users can override with volume or env)
COPY .env.example .env

# HF cache and optional model bake-in (skippable for huge GPU builds to avoid runner disk exhaustion)
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache
RUN mkdir -p /app/hf-cache && \
    if [ "$BAKE_MODEL" = "1" ]; then \
      python -c "import os; from huggingface_hub import snapshot_download; repo_id='Qwen/Qwen3-VL-2B-Thinking'; token=os.getenv('HF_TOKEN'); print(f'Downloading {repo_id}...'); snapshot_download(repo_id, token=token, local_dir='/app/hf-cache/Qwen_Qwen3-VL-2B-Thinking', local_dir_use_symlinks=False); print('Model downloaded.');"; \
    else \
      echo 'Skipping model bake-in (BAKE_MODEL=0). The server will prefetch to /app/hf-cache at startup.'; \
    fi

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run the server
CMD ["python", "main.py"]