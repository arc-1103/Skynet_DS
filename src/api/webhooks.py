"""GitHub webhook receiver (Architecture.md api/: 'HTTP surface... Thin
layer only - no business logic here'). Verifies the payload signature,
routes by X-GitHub-Event, and hands anything that takes real work off to
worker/ - this endpoint should ack fast."""

import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from src.core.config import Settings, get_settings
from src.schemas.webhook import GitHubDeploymentStatusEvent, GitHubPullRequestEvent
from src.worker.dispatcher import queue
from src.worker.jobs import run_deployment_regression_check, run_monitoring_sweep, run_pr_code_review

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Workflow 5: the always-on monitor lives in this process's worker
    queue, started with the server. Off unless both a sweep period and a
    Prometheus to read from are configured."""
    settings = get_settings()
    if settings.monitor_interval_seconds > 0 and settings.prometheus_url:
        queue.schedule_every(run_monitoring_sweep, interval_seconds=settings.monitor_interval_seconds)
        logger.info("monitoring sweep every %ss", settings.monitor_interval_seconds)
    yield
    queue.cancel_all()


app = FastAPI(title="ML Reliability Copilot - Webhook Receiver", lifespan=_lifespan)

# Wait this long post-deploy before checking metrics, per Workflow.md
# Workflow 4 ("1hr post-deploy, enough data to compare").
POST_DEPLOY_CHECK_DELAY_SECONDS = 60 * 60


def _verify_signature(secret: str, payload: bytes, signature_header: str | None) -> None:
    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing or malformed X-Hub-Signature-256")
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Signature mismatch")


def _validated(model, payload: dict):
    """Parse a payload into its event model, turning a schema mismatch into
    a 422 rather than a 500."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _ignored(event: str, reason: str) -> dict:
    return {"status": "ignored", "event": event, "reason": reason}


def _handle_deployment_status(event_name: str, payload: dict) -> dict:
    event = _validated(GitHubDeploymentStatusEvent, payload)
    if event.deployment_status.state != "success":
        return _ignored(event_name, f"state={event.deployment_status.state}")

    queue.schedule(
        run_deployment_regression_check,
        kwargs={
            "repo_full_name": event.repository.full_name,
            "deploy_sha": event.deployment.sha,
            "environment": event.deployment.environment,
        },
        delay_seconds=POST_DEPLOY_CHECK_DELAY_SECONDS,
    )
    return {"status": "accepted", "event": event_name, "scheduled": True}


def _handle_pull_request(event_name: str, payload: dict) -> dict:
    event = _validated(GitHubPullRequestEvent, payload)
    # Run on newly opened/reopened PRs or when new commits are pushed
    if event.action not in {"opened", "reopened", "synchronize"}:
        return _ignored(event_name, f"action={event.action}")

    queue.schedule(
        run_pr_code_review,
        kwargs={
            "repo_full_name": event.repository.full_name,
            "pr_number": event.number,
            "head_sha": event.pull_request.head.sha,
        },
        delay_seconds=0,  # PR reviews should run immediately
    )
    return {"status": "accepted", "event": event_name, "scheduled": True}


# X-GitHub-Event -> handler. Adding an event is a line here plus its handler,
# instead of another branch in an if/elif chain that repeats the whole
# validate / check-condition / schedule / respond shape each time.
_EVENT_HANDLERS = {
    "deployment_status": _handle_deployment_status,
    "pull_request": _handle_pull_request,
}


@app.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> dict:
    body = await request.body()

    if settings.github_webhook_secret:
        _verify_signature(settings.github_webhook_secret, body, x_hub_signature_256)

    if x_github_event is None:
        raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")

    if x_github_event == "ping":
        return {"status": "ok", "event": "ping"}

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Malformed JSON body: {exc}") from exc

    handler = _EVENT_HANDLERS.get(x_github_event)
    if handler is None:
        logger.info("Unhandled GitHub event type: %s", x_github_event)
        return {"status": "ignored", "event": x_github_event}

    return handler(x_github_event, payload)
