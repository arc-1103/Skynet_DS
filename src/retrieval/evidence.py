# src/retrieval/evidence.py
import logging
import re
from datetime import datetime

from src.github_integration.pr_client import fetch_commit_diff, fetch_path_commits
from src.memory.store import Store
from src.observability.base import TelemetryAdapter
from src.schemas.telemetry import NormalizedRun
from src.schemas.webhook import SHA_PATTERN

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Query layer over observability/ + memory/ + github_integration/ -
    'get me run X', 'get me the last 5 runs for model Y', 'get me the code
    diff between run A and run B'. agents/ and tools/ should go through
    this, never the adapter/store directly (Architecture.md section 2,
    retrieval/)."""

    def __init__(
        self,
        adapter: TelemetryAdapter,
        store: Store,
        github_repo: str | None = None,
        github_token: str | None = None,
    ) -> None:
        self._adapter = adapter
        self._store = store
        self._github_repo = github_repo
        self._github_token = github_token

    def get_run(self, run_id: str) -> NormalizedRun:
        return self._adapter.get_run(run_id)

    def get_recent_runs(self, model_id: str, limit: int = 5) -> list[NormalizedRun]:
        return self._adapter.get_recent_runs(model_id, limit)

    def get_baseline(self, model_id: str) -> NormalizedRun | None:
        cached = self._store.get_baseline(model_id)
        if cached is not None:
            return cached
        baseline = self._adapter.get_baseline(model_id)
        if baseline is not None and hasattr(self._store, "set_baseline"):
            self._store.set_baseline(model_id, baseline)
        return baseline

    def get_diagnosis_history(self, model_id: str):
        return self._store.get_diagnosis_history(model_id)

    def get_code_diff(self, base_ref: str | None, head_ref: str | None) -> str | None:
        """Summary of the commit range base_ref..head_ref, fetched from
        GitHub. Degrades to a plain range label (never raises) when no repo
        is configured or the fetch fails: a missing diff weakens the code
        candidate's evidence, it shouldn't abort the whole diagnosis."""
        if not base_ref or not head_ref or base_ref == head_ref:
            return None
        label = f"commit range {base_ref}..{head_ref}"
        # code_ref comes from run metadata (W&B), which anyone logging a run
        # controls - only a real SHA may reach the GitHub compare URL
        if not (re.fullmatch(SHA_PATTERN, base_ref) and re.fullmatch(SHA_PATTERN, head_ref)):
            return f"{label}, diff not fetched - refs are not commit SHAs"
        if not self._github_repo:
            return f"{label}, diff not fetched - no GITHUB_REPO configured"
        try:
            return f"{label}: " + fetch_commit_diff(self._github_repo, base_ref, head_ref, self._github_token)
        except Exception as exc:  # PyGithub/network errors
            logger.warning("code diff fetch failed for %s %s: %s", self._github_repo, label, exc)
            return f"{label}, diff fetch failed"

    def get_pipeline_commits(self, paths: tuple[str, ...], since: datetime) -> list[str] | None:
        """Commits touching the data pipeline since `since`. None means
        "could not check" (no repo/paths configured, or GitHub failed) -
        distinct from [] ("checked, nothing changed"), which is real
        evidence against a pipeline cause."""
        if not self._github_repo or not paths:
            return None
        try:
            return fetch_path_commits(self._github_repo, paths, since, self._github_token)
        except Exception as exc:  # PyGithub/network errors
            logger.warning("pipeline commit fetch failed for %s: %s", self._github_repo, exc)
            return None
