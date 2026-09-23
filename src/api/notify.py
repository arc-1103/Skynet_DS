# src/api/notify.py
"""Outbound notifications - the "published to api/ (webhook out to Slack)"
output step of Workflow 2 (step 6) and Workflow 4 (step 5).

A Slack incoming webhook takes a JSON {"text": ...} POST; anything
Slack-compatible (Mattermost, Rocket.Chat) works the same way."""

import logging

import httpx

logger = logging.getLogger(__name__)


def _escape(text: str) -> str:
    """Slack treats &, < and > as control characters in message text -
    explanations are full of '->' and must not be parsed as links."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def notify_slack(webhook_url: str | None, title: str, body: str) -> bool:
    """Post a verdict to Slack. Never raises: a verdict is already logged
    and persisted to memory/, so a Slack outage must not fail the job.
    Returns whether a message was delivered."""
    if not webhook_url:
        return False
    try:
        response = httpx.post(
            webhook_url, json={"text": f"*{_escape(title)}*\n{_escape(body)}"}, timeout=10.0
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # the URL is a secret (it's the whole credential), so it isn't logged
        logger.warning("Slack notification failed: %s", type(exc).__name__)
        return False
    return True
