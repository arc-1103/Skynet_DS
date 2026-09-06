import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from src.api import webhooks
from src.core.config import Settings

DEPLOYMENT_STATUS_SUCCESS = {
    "action": "created",
    "deployment": {"sha": "abc123", "ref": "main", "environment": "production"},
    "deployment_status": {"state": "success"},
    "repository": {
        "full_name": "acme/fraud-clf",
        "clone_url": "https://github.com/acme/fraud-clf.git",
        "default_branch": "main",
    },
    "sender": {"login": "octocat"},
}

DEPLOYMENT_STATUS_PENDING = {
    **DEPLOYMENT_STATUS_SUCCESS,
    "deployment_status": {"state": "pending"},
}


@pytest.fixture
def client(monkeypatch):
    scheduled = []
    monkeypatch.setattr(webhooks.queue, "schedule", lambda fn, delay_seconds: scheduled.append((fn, delay_seconds)))
    monkeypatch.setattr(webhooks, "settings", Settings(github_webhook_secret=None))
    test_client = TestClient(webhooks.app)
    test_client.scheduled = scheduled
    return test_client


def test_ping_event_returns_ok(client):
    resp = client.post("/webhooks/github", json={"zen": "hi"}, headers={"X-GitHub-Event": "ping"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "event": "ping"}


def test_missing_event_header_returns_400(client):
    resp = client.post("/webhooks/github", json={})
    assert resp.status_code == 400


def test_deployment_status_success_schedules_job(client):
    resp = client.post(
        "/webhooks/github",
        json=DEPLOYMENT_STATUS_SUCCESS,
        headers={"X-GitHub-Event": "deployment_status"},
    )
    assert resp.status_code == 200
    assert resp.json()["scheduled"] is True
    assert len(client.scheduled) == 1
    _, delay = client.scheduled[0]
    assert delay == webhooks.POST_DEPLOY_CHECK_DELAY_SECONDS


def test_deployment_status_non_success_is_ignored(client):
    resp = client.post(
        "/webhooks/github",
        json=DEPLOYMENT_STATUS_PENDING,
        headers={"X-GitHub-Event": "deployment_status"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert client.scheduled == []


def test_unhandled_event_type_is_ignored(client):
    resp = client.post(
        "/webhooks/github",
        json={"anything": True},
        headers={"X-GitHub-Event": "issues"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ignored", "event": "issues"}


def test_malformed_deployment_payload_returns_422(client):
    resp = client.post(
        "/webhooks/github",
        json={"deployment": {}},
        headers={"X-GitHub-Event": "deployment_status"},
    )
    assert resp.status_code == 422


def test_signature_required_when_secret_configured(monkeypatch):
    monkeypatch.setattr(webhooks, "settings", Settings(github_webhook_secret="topsecret"))
    monkeypatch.setattr(webhooks.queue, "schedule", lambda fn, delay_seconds: None)
    client = TestClient(webhooks.app)

    resp = client.post("/webhooks/github", json={"zen": "hi"}, headers={"X-GitHub-Event": "ping"})

    assert resp.status_code == 401


def test_valid_signature_is_accepted(monkeypatch):
    secret = "topsecret"
    monkeypatch.setattr(webhooks, "settings", Settings(github_webhook_secret=secret))
    monkeypatch.setattr(webhooks.queue, "schedule", lambda fn, delay_seconds: None)
    client = TestClient(webhooks.app)

    body = json.dumps({"zen": "hi"}).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    resp = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-GitHub-Event": "ping",
            "X-Hub-Signature-256": signature,
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 200


def test_invalid_signature_is_rejected(monkeypatch):
    monkeypatch.setattr(webhooks, "settings", Settings(github_webhook_secret="topsecret"))
    monkeypatch.setattr(webhooks.queue, "schedule", lambda fn, delay_seconds: None)
    client = TestClient(webhooks.app)

    resp = client.post(
        "/webhooks/github",
        json={"zen": "hi"},
        headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": "sha256=deadbeef"},
    )

    assert resp.status_code == 401
