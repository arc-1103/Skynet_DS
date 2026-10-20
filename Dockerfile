# Runs the webhook receiver (src/api/webhooks.py) - the actual product per
# README.md. The demo/ scripts and demo-ui/ (NestJS) are presentation-only
# and intentionally not part of this image; see demo/README.md.
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

WORKDIR /app

COPY src/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# data/prs (settings.pr_cache_dir) and data/repos (repo_client.py's clone
# cache) are written at runtime - see the volume mount in
# docker-compose.yml. memory.db (settings.memory_db_path) lands at /app
# and is NOT covered by that mount; it's only used by the CLI's
# debug/compare path, not the webhook service this image runs.
RUN mkdir -p data/prs data/repos

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Secrets (GITHUB_TOKEN, GITHUB_WEBHOOK_SECRET, GROQ_API_KEY, ...) are read
# from the environment at startup (src/core/config.py) - pass them with
# `docker run --env-file .env` or compose's env_file, never baked in here.
CMD ["uvicorn", "src.api.webhooks:app", "--host", "0.0.0.0", "--port", "8000"]
