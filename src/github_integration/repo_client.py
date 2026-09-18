# src/github_integration/repo_client.py
"""GitHub-specific integration (Architecture.md github/): clones a repo (or
updates an existing local clone) and checks out a given ref, so a PR's head
branch can be inspected as a real working tree, not just the diff patches
cached by pr_client.fetch_pr. Shells out to the system `git` - no GitPython
dependency needed for clone/fetch/checkout."""

import base64
import os
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_HEARTBEAT_INTERVAL = 30.0


def _clone_url(repo_full_name: str) -> str:
    return f"https://github.com/{repo_full_name}.git"


def _auth_args(token: str | None) -> list[str]:
    """Per-invocation credentials as `git -c http.extraHeader=...`.

    Deliberately not `https://x-access-token:<token>@github.com/...`: git
    writes a clone URL verbatim into .git/config, so that leaves the PAT in
    plaintext on disk and in `git remote -v` output, and pins the clone to
    whichever token first created it (a rotated token then never reaches an
    existing clone). A header applies to this one command only.

    ponytail: the token is still visible in this process's argv for the
    duration of the call; move to a credential helper if local processes
    are part of the threat model."""
    if not token:
        return []
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.extraHeader=Authorization: Basic {basic}"]


@contextmanager
def _repo_lock(dest: Path, timeout: float = 300.0, poll_interval: float = 0.5, stale_after: float = 120.0):
    """File-based lock so two concurrent PR-review jobs for the same repo
    can't fetch/checkout/reset the same on-disk working tree at once - a
    plain O_CREAT|O_EXCL lock file works across both processes and threads,
    on Windows and POSIX alike, with no extra dependency.

    The holder touches the lock file every _HEARTBEAT_INTERVAL seconds, so
    "older than stale_after" means the holder stopped running, not merely
    that its clone is taking a while - a first clone of a big repo can
    easily outlive any fixed timeout, and reclaiming the lock from a job
    that is still mid-clone hands two jobs the same working tree. Each
    holder also writes a random token and only unlinks a lock file still
    carrying its own, so a reclaimed lock is never deleted out from under
    its new owner. (No pid-liveness check: os.kill(pid, 0) terminates the
    process on Windows.)"""
    lock_path = dest.parent / f"{dest.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    token = uuid.uuid4().hex
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_after:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise TimeoutError(f"timed out waiting for repo lock: {lock_path}")
            time.sleep(poll_interval)

    os.write(fd, token.encode())
    released = threading.Event()

    def _heartbeat() -> None:
        while not released.wait(_HEARTBEAT_INTERVAL):
            try:
                os.utime(lock_path, None)
            except OSError:
                return

    threading.Thread(target=_heartbeat, daemon=True).start()
    try:
        yield
    finally:
        released.set()
        os.close(fd)
        try:
            if lock_path.read_text() == token:
                lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def clone_or_checkout(
    repo_full_name: str,
    ref: str,
    token: str | None = None,
    dest_root: str | Path = "data/repos",
    extra_refs: tuple[str, ...] = (),
) -> Path:
    """Ensure repo_full_name is cloned under dest_root and checked out at
    ref (a branch name or commit SHA). Returns the local repo path.

    extra_refs are fetched alongside ref but not checked out - e.g. a PR's
    base_sha, so a caller that only needs `ref` in the working tree (like
    this function's own checkout) can still `git show <extra_ref>:<path>`
    against a ref that's actually present locally. ref is always listed
    first in the fetch, so `reset --hard FETCH_HEAD` still resolves to ref
    itself, not to one of the extra refs, regardless of how many are given."""
    dest = Path(dest_root) / repo_full_name.split("/")[-1]

    with _repo_lock(dest):
        auth = _auth_args(token)
        if (dest / ".git").exists():
            subprocess.run(["git", *auth, "fetch", "origin", ref, *extra_refs], cwd=dest, check=True)
            subprocess.run(["git", "checkout", ref], cwd=dest, check=True)
            subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=dest, check=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", *auth, "clone", _clone_url(repo_full_name), str(dest)], check=True)
            subprocess.run(["git", *auth, "fetch", "origin", ref, *extra_refs], cwd=dest, check=True)
            subprocess.run(["git", "checkout", ref], cwd=dest, check=True)

        return dest


if __name__ == "__main__":
    import argparse

    from src.core.config import settings

    parser = argparse.ArgumentParser(description="Clone/checkout a GitHub repo locally")
    parser.add_argument("repo_full_name", help="e.g. arc-1103/ML-Model-for-SIH-2025")
    parser.add_argument("ref", help="branch name or commit SHA to check out")
    args = parser.parse_args()

    path = clone_or_checkout(args.repo_full_name, args.ref, settings.github_token)
    print(f"Repo ready at: {path}")
