"""Credential handling and lock ownership for the local clone cache."""

import base64
import subprocess

from src.github_integration import repo_client
from src.github_integration.repo_client import _repo_lock, clone_or_checkout


def test_token_never_reaches_the_clone_url(tmp_path, monkeypatch):
    """An in-URL token is written verbatim into .git/config by git, leaving the
    PAT on disk and pinning the clone to whichever token created it."""
    calls = []
    monkeypatch.setattr(
        repo_client.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    clone_or_checkout("acme/demo", "deadbeef", token="ghp_secret", dest_root=tmp_path)

    flat = " ".join(arg for cmd in calls for arg in cmd)
    assert "ghp_secret" not in flat
    assert "https://github.com/acme/demo.git" in flat

    expected = base64.b64encode(b"x-access-token:ghp_secret").decode()
    network_calls = [cmd for cmd in calls if "clone" in cmd or "fetch" in cmd]
    assert len(network_calls) == 2, calls
    for cmd in network_calls:
        assert f"http.extraHeader=Authorization: Basic {expected}" in cmd


def test_lock_is_not_deleted_once_another_holder_has_reclaimed_it(tmp_path):
    """The old finally block unlinked the lock file unconditionally, so a job
    whose stale lock had been reclaimed deleted the new owner's lock."""
    dest = tmp_path / "repo"
    lock_path = dest.parent / f"{dest.name}.lock"

    with _repo_lock(dest):
        assert lock_path.exists()
        lock_path.write_text("another-holders-token")

    assert lock_path.exists()
    assert lock_path.read_text() == "another-holders-token"


def test_lock_is_released_by_its_own_holder(tmp_path):
    dest = tmp_path / "repo"
    lock_path = dest.parent / f"{dest.name}.lock"

    with _repo_lock(dest):
        pass

    assert not lock_path.exists()
