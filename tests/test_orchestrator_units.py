from datetime import datetime, timezone

from src.agents.orchestrator import (
    _explain,
    _hypothesize_code,
    _hypothesize_config,
    _hypothesize_data,
)
from src.schemas.telemetry import NormalizedRun


def _run(**overrides) -> NormalizedRun:
    defaults = dict(
        run_id="current",
        model_id="m1",
        metrics={"accuracy": 0.8},
        config={"lr": 1e-3},
        code_ref="sha2",
        data_version="v1",
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return NormalizedRun(**defaults)


def test_hypothesize_config_detects_single_change():
    baseline = _run(run_id="baseline", config={"lr": 3e-4})
    current = _run(config={"lr": 1e-3})

    candidate = _hypothesize_config(current, baseline)

    assert candidate is not None
    assert candidate.kind == "config"
    assert candidate.confidence == 0.8
    assert "lr" in candidate.description


def test_hypothesize_config_lowers_confidence_for_multiple_changes():
    baseline = _run(run_id="baseline", config={"lr": 3e-4, "batch_size": 32})
    current = _run(config={"lr": 1e-3, "batch_size": 64})

    candidate = _hypothesize_config(current, baseline)

    assert candidate.confidence == 0.6


def test_hypothesize_config_no_change_returns_none():
    baseline = _run(run_id="baseline", config={"lr": 1e-3})
    current = _run(config={"lr": 1e-3})

    assert _hypothesize_config(current, baseline) is None


def test_hypothesize_data_detects_version_change():
    baseline = _run(run_id="baseline", data_version="v1")
    current = _run(data_version="v2")

    candidate = _hypothesize_data(current, baseline)

    assert candidate.kind == "data"
    assert candidate.confidence == 0.5


def test_hypothesize_data_no_change_returns_low_confidence_candidate():
    baseline = _run(run_id="baseline", data_version="v1")
    current = _run(data_version="v1")

    candidate = _hypothesize_data(current, baseline)

    assert candidate.confidence == 0.05


def test_hypothesize_code_detects_change():
    baseline = _run(run_id="baseline", code_ref="sha1")
    current = _run(code_ref="sha2")

    candidate = _hypothesize_code(current, baseline, diff_summary="some diff")

    assert candidate is not None
    assert candidate.kind == "code"
    assert "some diff" in candidate.description


def test_hypothesize_code_no_change_returns_none():
    baseline = _run(run_id="baseline", code_ref="sha1")
    current = _run(code_ref="sha1")

    assert _hypothesize_code(current, baseline, diff_summary=None) is None


def test_explain_no_baseline():
    current = _run()
    text = _explain(current, None, [])
    assert "No baseline found" in text


def test_explain_no_candidates():
    current = _run()
    baseline = _run(run_id="baseline")
    text = _explain(current, baseline, [])
    assert "no meaningful delta" in text


def test_explain_includes_top_and_secondary_candidates():
    from src.graph.causal import CauseCandidate

    current = _run(metrics={"accuracy": 0.8})
    baseline = _run(run_id="baseline", metrics={"accuracy": 0.9})
    ranked = [
        CauseCandidate(kind="config", description="top cause", confidence=0.9),
        CauseCandidate(kind="data", description="secondary cause", confidence=0.4),
        CauseCandidate(kind="code", description="ignored, too low", confidence=0.01),
    ]

    text = _explain(current, baseline, ranked)

    assert "Most likely cause" in text
    assert "top cause" in text
    assert "Secondary candidate" in text
    assert "secondary cause" in text
    assert "ignored, too low" not in text
