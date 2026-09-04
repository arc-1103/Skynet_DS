# src/api/webhooks.py
"""GitHub webhook receiver (Architecture.md api/: 'HTTP surface... Thin
layer only - no business logic here'). Verifies the payload signature,
routes by X-GitHub-Event, and hands anything that takes real work off to
worker/ - this endpoint should ack fast."""

import hashlib
import hmac
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from src.core.config import settings
from src.schemas.webhook import GitHubDeploymentStatusEvent
from src.worker.dispatcher import queue
from src.worker.jobs import run_deployment_regression_check

logger = logging.getLogger(__name__)

app = FastAPI(title="ML Reliability Copilot - Webhook Receiver")

# Wait this long post-deploy before checking metrics, per Workflow.md
# Workflow 4 ("1hr post-deploy, enough data to compare").
POST_DEPLOY_CHECK_DELAY_SECONDS = 60 * 60


def _verify_signature(secret: str, payload: bytes, signature_header: str | None) -> None:
    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing or malformed X-Hub-Signature-256")
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Signature mismatch")


@app.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    body = await request.body()

    if settings.github_webhook_secret:
        _verify_signature(settings.github_webhook_secret, body, x_hub_signature_256)

    if x_github_event is None:
        raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")

    if x_github_event == "ping":
        return {"status": "ok", "event": "ping"}

    payload = await request.json()

    if x_github_event == "deployment_status":
        try:
            event = GitHubDeploymentStatusEvent.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if event.deployment_status.state != "success":
            return {
                "status": "ignored",
                "event": x_github_event,
                "reason": f"state={event.deployment_status.state}",
            }

        queue.schedule(
            lambda: run_deployment_regression_check(
                repo_full_name=event.repository.full_name,
                deploy_sha=event.deployment.sha,
                environment=event.deployment.environment,
            ),
            delay_seconds=POST_DEPLOY_CHECK_DELAY_SECONDS,
        )
        return {"status": "accepted", "event": x_github_event, "scheduled": True}

    logger.info("Unhandled GitHub event type: %s", x_github_event)
    return {"status": "ignored", "event": x_github_event}
