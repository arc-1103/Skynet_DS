import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from src.api import webhooks
from src.core.config import Settings, get_settings

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

PULL_REQUEST_OPENED = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "number": 42,
        "title": "Fix flaky test",
        "body": "Closes #41",
        "state": "open",
        "head": {"sha": "deadbeef", "ref": "fix-flaky-test"},
        "base": {"sha": "cafebabe", "ref": "main"},
    },
    "repository": {
        "full_name": "acme/fraud-clf",
        "clone_url": "https://github.com/acme/fraud-clf.git",
        "default_branch": "main",
    },
    "sender": {"login": "octocat"},
}

PULL_REQUEST_CLOSED = {**PULL_REQUEST_OPENED, "action": "closed"}


@pytest.fixture
def client(monkeypatch):
    scheduled = []
    monkeypatch.setattr(
        webhooks.queue,
        "schedule",
        lambda fn, kwargs=None, *, delay_seconds: scheduled.append((fn, kwargs, delay_seconds)),
    )
    webhooks.app.dependency_overrides[get_settings] = lambda: Settings(github_webhook_secret=None)
    test_client = TestClient(webhooks.app)
    test_client.scheduled = scheduled
    yield test_client
    webhooks.app.dependency_overrides.clear()


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
    fn, kwargs, delay = client.scheduled[0]
    assert fn is webhooks.run_deployment_regression_check
    assert kwargs == {
        "repo_full_name": "acme/fraud-clf",
        "deploy_sha": "abc123",
        "environment": "production",
    }
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


def test_pull_request_opened_schedules_job(client):
    resp = client.post(
        "/webhooks/github",
        json=PULL_REQUEST_OPENED,
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert resp.status_code == 200
    assert resp.json()["scheduled"] is True
    assert len(client.scheduled) == 1
    fn, kwargs, delay = client.scheduled[0]
    assert fn is webhooks.run_pr_code_review
    assert kwargs == {
        "repo_full_name": "acme/fraud-clf",
        "pr_number": 42,
        "head_sha": "deadbeef",
    }
    assert delay == 0


def test_pull_request_closed_is_ignored(client):
    resp = client.post(
        "/webhooks/github",
        json=PULL_REQUEST_CLOSED,
        headers={"X-GitHub-Event": "pull_request"},
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
    webhooks.app.dependency_overrides[get_settings] = lambda: Settings(github_webhook_secret="topsecret")
    monkeypatch.setattr(webhooks.queue, "schedule", lambda fn, kwargs=None, *, delay_seconds: None)
    client = TestClient(webhooks.app)

    resp = client.post("/webhooks/github", json={"zen": "hi"}, headers={"X-GitHub-Event": "ping"})

    assert resp.status_code == 401
    webhooks.app.dependency_overrides.clear()


def test_valid_signature_is_accepted(monkeypatch):
    secret = "topsecret"
    webhooks.app.dependency_overrides[get_settings] = lambda: Settings(github_webhook_secret=secret)
    monkeypatch.setattr(webhooks.queue, "schedule", lambda fn, kwargs=None, *, delay_seconds: None)
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
    webhooks.app.dependency_overrides.clear()


def test_invalid_signature_is_rejected(monkeypatch):
    webhooks.app.dependency_overrides[get_settings] = lambda: Settings(github_webhook_secret="topsecret")
    monkeypatch.setattr(webhooks.queue, "schedule", lambda fn, kwargs=None, *, delay_seconds: None)
    client = TestClient(webhooks.app)

    resp = client.post(
        "/webhooks/github",
        json={"zen": "hi"},
        headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": "sha256=deadbeef"},
    )

    assert resp.status_code == 401
    webhooks.app.dependency_overrides.clear()


def test_pull_request_with_a_path_traversal_sha_is_rejected(client):
    """worker/jobs.py builds a cache filename out of head.sha, and signature
    verification is skipped when no webhook secret is configured, so the sha
    is constrained at the parse boundary."""
    payload = {
        **PULL_REQUEST_OPENED,
        "pull_request": {
            **PULL_REQUEST_OPENED["pull_request"],
            "head": {"sha": "../../../../x", "ref": "evil"},
        },
    }
    resp = client.post(
        "/webhooks/github", json=payload, headers={"X-GitHub-Event": "pull_request"}
    )
    assert resp.status_code == 422
    assert client.scheduled == []


def test_deployment_with_a_non_sha_is_rejected(client):
    """deploy sha reaches a GitHub compare URL in retrieval/evidence.py."""
    payload = {
        **DEPLOYMENT_STATUS_SUCCESS,
        "deployment": {"sha": "main/../../x", "ref": "main", "environment": "production"},
    }
    resp = client.post(
        "/webhooks/github", json=payload, headers={"X-GitHub-Event": "deployment_status"}
    )
    assert resp.status_code == 422
    assert client.scheduled == []


@pytest.mark.parametrize(
    "interval, url, expected",
    [(900, "http://prom", [900]), (0, "http://prom", []), (900, None, [])],
)
def test_monitoring_sweep_starts_with_the_server_only_when_configured(monkeypatch, interval, url, expected):
    started = []
    monkeypatch.setattr(
        webhooks, "get_settings",
        lambda: Settings(monitor_interval_seconds=interval, prometheus_url=url),
    )
    monkeypatch.setattr(
        webhooks.queue, "schedule_every",
        lambda func, kwargs=None, *, interval_seconds: started.append(interval_seconds),
    )

    with TestClient(webhooks.app):
        pass

    assert started == expected
