# Demo layer image (demo/run_demo.py): the multi-agent review of a real PR,
# Langfuse-traced, plus Evidently drift dashboards written as HTML. Build
# context is the repo root (it imports src/); its ignore rules are in
# demo/Dockerfile.dockerignore. Presentation only - see demo/README.md.
#
#   docker compose run --rm demo owner/repo 42          # review + dashboards
#   docker compose run --rm demo owner/repo 42 --post   # also comments on the PR
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# Same pip cache mount as the root Dockerfile, so wheels the core image
# already downloaded are reused here.
COPY src/requirements.txt ./requirements.txt
COPY demo/requirements-demo.txt ./requirements-demo.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt -r requirements-demo.txt

COPY src/ ./src/
COPY demo/ ./demo/

# demo/out/evidently: the HTML dashboards (bind-mounted out in compose so
# they land on the host). data/repos: the PR clone cache.
RUN useradd --create-home --uid 10001 skynet \
    && mkdir -p demo/out/evidently data/repos \
    && chown -R skynet:skynet /app
USER skynet

ENTRYPOINT ["python", "-m", "demo.run_demo"]
