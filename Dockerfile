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
ENV BACKEND=${BACKEND}
ENV PIP_NO_CACHE_DIR=1

# Install appropriate PyTorch for the selected backend, then the rest
RUN if [ "$BACKEND" = "cpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.0; \
    elif [ "$BACKEND" = "nvidia" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.9.0; \
    elif [ "$BACKEND" = "amd" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/rocm6.2 torch==2.9.0; \
    else \
      echo "Unsupported BACKEND: $BACKEND" && exit 1; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .
COPY tests/ tests/

# Copy env template (users can override with volume or env)
COPY .env.example .env

# Pre-download model during build (makes image self-contained)
# Use HF_TOKEN if provided, but for public models it's optional
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache
RUN mkdir -p /app/hf-cache && \
    python -c "\
import os; \
from huggingface_hub import snapshot_download; \
repo_id = 'Qwen/Qwen3-VL-2B-Thinking'; \
token = os.getenv('HF_TOKEN'); \
print(f'Downloading {repo_id}...'); \
snapshot_download(repo_id, token=token, local_dir='/app/hf-cache/Qwen_Qwen3-VL-2B-Thinking', local_dir_use_symlinks=False); \
print('Model downloaded.'); \
"

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run the server
CMD ["python", "main.py"]