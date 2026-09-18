# demo/evidently_dashboard.py
"""Generates a standalone, interactive Evidently HTML dashboard
(distribution curves, K-S test statistic, p-values, PSI) for the same
reference/current data pairs drift/detector.py already compares - so a
demo can show off a polished, pre-built visualization instead of hand-
rolling charts with Chart.js/D3.

The CLI/webhook path keeps using drift/ (numpy/scipy only, no Evidently
dependency) for the actual pass/fail decision; this module is a second,
optional renderer over the same base data, purely for presentation. See
demo/README.md and demo/requirements-demo.txt - evidently is not a
dependency of the core CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from src.drift.pr_drift import data_files_in_pr, load_at_ref
from src.schemas.pr import PRPayload

DEFAULT_OUT_DIR = Path(__file__).parent / "out" / "evidently"


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Runs Evidently's DataDriftPreset (K-S test for numeric columns,
    chi-squared for categorical, PSI, share-of-drifted-columns) over one
    reference/current pair and returns the run result."""
    report = Report(metrics=[DataDriftPreset()])
    return report.run(
        reference_data=Dataset.from_pandas(reference),
        current_data=Dataset.from_pandas(current),
    )


def save_dashboard(reference: pd.DataFrame, current: pd.DataFrame, out_path: str | Path) -> Path:
    """Runs the dashboard and writes it as a standalone HTML file -
    open it directly in a browser, no server needed."""
    result = build_drift_report(reference, current)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out_path))
    return out_path


def dashboards_for_pr(
    pr: PRPayload,
    repo_path: str | Path,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> list[Path]:
    """One dashboard per tabular data file the PR touches. Mirrors
    drift.pr_drift.check_pr_drift's file selection (same
    data_files_in_pr/load_at_ref calls) so the Evidently dashboard and
    the CLI's own drift verdict are always comparing the same two
    versions of the same file - only the renderer differs."""
    if not pr.base_sha:
        raise ValueError("PR payload has no base_sha; cannot determine what to compare against")

    out_dir = Path(out_dir)
    paths: list[Path] = []
    for file_path in data_files_in_pr(pr):
        try:
            reference = load_at_ref(repo_path, pr.base_sha, file_path)
            current = load_at_ref(repo_path, pr.commit_sha, file_path)
        except (subprocess.CalledProcessError, ValueError):
            continue
        safe_name = file_path.replace("/", "__")
        paths.append(save_dashboard(reference, current, out_dir / f"{safe_name}.html"))
    return paths
