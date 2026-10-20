# Core image: the webhook receiver + worker (src/api/webhooks.py, the actual
# product per README.md), and - same image, different command - the CLI
# (src/cli.py) and the incident eval (evals/). The demo layer (demo/) and the
# dashboard (demo-ui/) have their own images: demo/Dockerfile and
# demo-ui/Dockerfile. docker-compose.yml wires all of them together.
FROM python:3.13-slim

# git: src/github_integration/repo_client.py shells out to it to clone/
# fetch a PR's repo for drift checks. libgomp1: xgboost's OpenMP runtime,
# not always present on the slim base. bandit (the security agent's
# optional AST-scan corroboration, src/agents/security_agent_tools.py) is
# left out - the code already degrades gracefully without it; add
# `pip install bandit` below if you want that pass in the demo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# Dependencies first, so a source-only change reuses this layer. The cache
# mount keeps downloaded wheels (xgboost alone is ~60 MB) across builds and between this image and demo/Dockerfile, outside the
# image itself - a retried build after a dropped connection resumes instead
# of re-downloading.
COPY src/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY src/ ./src/
COPY evals/ ./evals/

# data/prs (settings.pr_cache_dir), data/monitoring (settings.monitoring_dir)
# and data/repos (repo_client.py's clone cache) are written at runtime - see
# the skynet-data volume in docker-compose.yml. A fresh named volume copies
# this directory's ownership, so the non-root user can write it. memory.db
# (settings.memory_db_path, CLI debug/compare history) lands in
# /app/src/memory and is not on the volume.
RUN useradd --create-home --uid 10001 skynet \
    && mkdir -p data/prs data/monitoring data/repos \
    && chown -R skynet:skynet /app
USER skynet

EXPOSE 8000

# No curl on the slim base; the stdlib does the probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

# Secrets (GITHUB_TOKEN, GITHUB_WEBHOOK_SECRET, RESULTS_API_TOKEN, GROQ_API_KEY,
# ...) are read from the environment at startup (src/core/config.py) - pass
# them with `docker run --env-file .env` or compose's env_file, never baked in.
CMD ["uvicorn", "src.api.webhooks:app", "--host", "0.0.0.0", "--port", "8000"]
