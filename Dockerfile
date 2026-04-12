# ─────────────────────────────────────────────────────────────────────────────
# GST Intelligence RL — Docker Image
# Target: HuggingFace Spaces (2 vCPU, 8 GB RAM)
# Python: 3.11
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="gst-intelligence-rl"
LABEL description="GST Intelligence RL Environment — OpenEnv Hackathon Submission"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY . .

# ── Pre-generate data splits at build time (avoids cold-start latency) ───────
RUN python data/generate_splits.py

# ── Environment variables with defaults (HF_TOKEN injected at runtime) ───────
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED="1"

# ── Health check: verify env imports cleanly ─────────────────────────────────
RUN python -c "\
from env.gst_env import GSTEnvironment; \
from models.observation import GSTObservation; \
env = GSTEnvironment(); \
obs, _ = env.reset(task_id=1, seed=0); \
print('Health check OK: pending=' + str(obs.pending_count))"

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 7860

# ── Entry point: Gradio UI + FastAPI (openenv endpoints) ─────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
