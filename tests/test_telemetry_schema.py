from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.schemas.telemetry import NormalizedRun


def test_normalized_run_minimal_construction():
    run = NormalizedRun(
        run_id="r1",
        model_id="m1",
        timestamp=datetime.now(timezone.utc),
        
    )
    assert run.metrics == {}
    assert run.config == {}
    assert run.code_ref is None
    assert run.data_version is None


def test_normalized_run_full_construction():
    ts = datetime.now(timezone.utc)
    run = NormalizedRun(
        run_id="r1",
        model_id="m1",
        metrics={"accuracy": 0.9},
        config={"lr": 1e-3},
        code_ref="sha123",
        data_version="v2",
        timestamp=ts,
    )
    assert run.metrics["accuracy"] == 0.9
    assert run.code_ref == "sha123"
    assert run.timestamp == ts


def test_normalized_run_requires_run_id_model_id_timestamp():
    with pytest.raises(ValidationError):
        NormalizedRun(model_id="m1", timestamp=datetime.now(timezone.utc),run_id="r1")
    with pytest.raises(ValidationError):
        NormalizedRun(run_id="r1", timestamp=datetime.now(timezone.utc),model_id="m2")
    with pytest.raises(ValidationError):
        NormalizedRun(run_id="r1", model_id="m1",timestamp=datetime.now(timezone.utc))


def test_normalized_run_json_roundtrip():
    run = NormalizedRun(
        run_id="r1",
        model_id="m1",
        metrics={"accuracy": 0.9},
        config={"lr": 1e-3},
        timestamp=datetime.now(timezone.utc),
    )
    restored = NormalizedRun.model_validate_json(run.model_dump_json())
    assert restored == run
