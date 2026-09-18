# src/drift/pr_drift.py
"""Applies the drift detectors to a pull request.

The question this answers is 'does this PR change the data?' - for every
data file the PR touches, the base branch's version is the reference and
the PR head's version is the current sample, so a PR that quietly reshapes
a fixture, a seed dataset, or a checked-in feature table shows up as drift
per column instead of as an unreadable diff of a few thousand rows.

Reads both versions straight out of git (`git show <sha>:<path>`), so no
working-tree checkout of either side is needed beyond the clone itself."""

from __future__ import annotations

import io
import logging
import subprocess
from pathlib import Path

import pandas as pd

from src.drift.detector import detect_dataset_drift
from src.drift.tables import read_table
from src.schemas.drift import DatasetDriftReport
from src.schemas.pr import PRPayload

logger = logging.getLogger(__name__)

DATA_SUFFIXES = frozenset({".csv", ".parquet"})


def data_files_in_pr(pr: PRPayload) -> list[str]:
    """Changed files that hold tabular data and have both sides to compare.

    Added and deleted files are excluded on purpose: drift is a comparison,
    and a file that exists on only one side has no baseline."""
    return [
        f.filename
        for f in pr.changed_files
        if Path(f.filename).suffix.lower() in DATA_SUFFIXES and f.status == "modified"
    ]


def load_at_ref(repo_path: str | Path, ref: str, file_path: str) -> pd.DataFrame:
    """Read one tabular file as it exists at a given git ref."""
    completed = subprocess.run(
        ["git", "show", f"{ref}:{file_path}"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    return read_table(io.BytesIO(completed.stdout), suffix=Path(file_path).suffix)


def check_pr_drift(
    pr: PRPayload,
    repo_path: str | Path,
    *,
    share_threshold: float = 0.5,
    vote: str = "majority",
) -> list[DatasetDriftReport]:
    """Compare every modified data file in the PR, base version vs head.

    Returns one report per comparable file - an empty list means the PR
    touches no tabular data, which is the common case and not an error."""
    if not pr.base_sha:
        raise ValueError("PR payload has no base_sha; cannot determine what to compare against")

    reports: list[DatasetDriftReport] = []
    for file_path in data_files_in_pr(pr):
        try:
            reference = load_at_ref(repo_path, pr.base_sha, file_path)
            current = load_at_ref(repo_path, pr.commit_sha, file_path)
        except (subprocess.CalledProcessError, ValueError) as exc:
            logger.warning("skipping %s - could not load both versions: %s", file_path, exc)
            continue

        try:
            report = detect_dataset_drift(
                reference,
                current,
                share_threshold=share_threshold,
                vote=vote,
                label=file_path,
            )
        except Exception as exc:  # noqa: BLE001 - any computation failure is per-file, not fatal
            logger.warning("skipping %s - drift detection failed: %s", file_path, exc)
            continue

        reports.append(report)
    return reports
