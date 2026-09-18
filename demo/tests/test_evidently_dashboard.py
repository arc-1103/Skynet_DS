"""Smoke tests only - Evidently's own report math is out of scope here
(that's the library's job); this confirms our wiring (Dataset construction,
Report.run, save_html) actually produces a usable file."""

import numpy as np
import pandas as pd
import pytest

from demo.evidently_dashboard import build_drift_report, save_dashboard

SEED = 20260916


@pytest.fixture
def frames():
    rng = np.random.default_rng(SEED)
    n = 100
    reference = pd.DataFrame({"a": rng.normal(0, 1, n)})
    current = pd.DataFrame({"a": rng.normal(5, 1, n)})
    return reference, current


def test_build_drift_report_runs_without_error(frames):
    reference, current = frames
    result = build_drift_report(reference, current)
    assert result is not None


def test_save_dashboard_writes_an_html_file(frames, tmp_path):
    reference, current = frames
    out_path = tmp_path / "nested" / "report.html"

    result_path = save_dashboard(reference, current, out_path)

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert "<html" in out_path.read_text(encoding="utf-8").lower()
