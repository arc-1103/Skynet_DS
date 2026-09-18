"""Dataset-level roll-up: which methods apply to which column kind, how the
per-column verdicts are combined, and how 'Share of Drifted Components'
turns those into a single dataset answer."""

import numpy as np
import pandas as pd
import pytest

from src.drift.detector import (
    ALL_METHODS,
    detect_column_drift,
    detect_dataset_drift,
    format_report,
    infer_kind,
)
from src.schemas.drift import ColumnKind, DriftMethod

SEED = 20260915


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def frames(rng):
    """Four columns: two stable, two drifted (one numeric, one categorical)."""
    n = 300
    reference = pd.DataFrame(
        {
            "stable_num": rng.normal(10.0, 2.0, n),
            "drifted_num": rng.normal(10.0, 2.0, n),
            "stable_cat": rng.choice(["x", "y"], n, p=[0.5, 0.5]),
            "drifted_cat": rng.choice(["x", "y"], n, p=[0.9, 0.1]),
        }
    )
    current = pd.DataFrame(
        {
            "stable_num": rng.normal(10.0, 2.0, n),
            "drifted_num": rng.normal(16.0, 2.0, n),
            "stable_cat": rng.choice(["x", "y"], n, p=[0.5, 0.5]),
            "drifted_cat": rng.choice(["x", "y"], n, p=[0.1, 0.9]),
        }
    )
    return reference, current


def test_infer_kind_separates_numeric_from_everything_else():
    assert infer_kind(pd.Series([1.0, 2.0, 3.0])) is ColumnKind.NUMERICAL
    assert infer_kind(pd.Series(["a", "b"])) is ColumnKind.CATEGORICAL
    # Booleans are numeric to pandas but are categories here.
    assert infer_kind(pd.Series([True, False])) is ColumnKind.CATEGORICAL


def test_numeric_column_runs_every_method(frames):
    reference, current = frames
    report = detect_column_drift(reference["stable_num"], current["stable_num"])
    assert len(report.results) == len(ALL_METHODS)
    assert report.skipped == {}


def test_categorical_column_skips_the_numeric_only_methods(frames):
    reference, current = frames
    report = detect_column_drift(reference["stable_cat"], current["stable_cat"])
    assert set(report.skipped) == {
        DriftMethod.KS.value,
        DriftMethod.PAGE_HINKLEY.value,
        DriftMethod.MMD.value,
    }
    assert {r.method for r in report.results} == {
        DriftMethod.PSI,
        DriftMethod.EUCLIDEAN,
        DriftMethod.COSINE,
        DriftMethod.MODEL_BASED,
    }


def test_stable_columns_are_not_flagged(frames):
    reference, current = frames
    assert detect_column_drift(reference["stable_num"], current["stable_num"]).drifted is False
    assert detect_column_drift(reference["stable_cat"], current["stable_cat"]).drifted is False


def test_drifted_columns_are_flagged(frames):
    reference, current = frames
    assert detect_column_drift(reference["drifted_num"], current["drifted_num"]).drifted is True
    assert detect_column_drift(reference["drifted_cat"], current["drifted_cat"]).drifted is True


def test_any_vote_is_more_sensitive_than_majority(rng):
    """A shift small enough to split the detectors flips the verdict between
    the two voting rules: at ~0.2 sigma only the sensitive hypothesis tests
    fire, which is exactly the false-positive majority voting exists to
    absorb."""
    reference = pd.Series(rng.normal(0, 1, 400))
    current = pd.Series(rng.normal(0.2, 1, 400))
    majority = detect_column_drift(reference, current, vote="majority")
    sensitive = detect_column_drift(reference, current, vote="any")
    assert 0 < majority.n_drifted_methods <= len(majority.results) / 2
    assert majority.drifted is False
    assert sensitive.drifted is True


def test_unknown_vote_is_rejected(frames):
    reference, current = frames
    with pytest.raises(ValueError, match="vote must be"):
        detect_column_drift(reference["stable_num"], current["stable_num"], vote="loudest")


def test_per_method_options_override_thresholds(frames):
    """A threshold tight enough to flag anything proves options reach the
    detector rather than being silently dropped."""
    reference, current = frames
    report = detect_column_drift(
        reference["stable_num"],
        current["stable_num"],
        methods_to_run=(DriftMethod.PSI,),
        options={DriftMethod.PSI: {"threshold": 0.0}},
    )
    assert report.results[0].threshold == 0.0
    assert report.results[0].drifted is True


def test_share_of_drifted_components(frames):
    reference, current = frames
    report = detect_dataset_drift(reference, current)
    assert report.n_columns == 4
    assert report.n_drifted == 2
    assert report.share_drifted == pytest.approx(0.5)
    # Exactly at the threshold is not "more than", so this is not drifted.
    assert report.dataset_drifted is False


def test_dataset_is_drifted_once_the_share_passes_the_threshold(frames):
    reference, current = frames
    report = detect_dataset_drift(reference, current, share_threshold=0.25)
    assert report.dataset_drifted is True


def test_columns_missing_from_either_frame_are_ignored(frames):
    reference, current = frames
    report = detect_dataset_drift(reference, current.drop(columns=["drifted_cat"]))
    assert [c.column for c in report.columns] == ["stable_num", "drifted_num", "stable_cat"]


def test_columns_argument_restricts_the_comparison(frames):
    reference, current = frames
    report = detect_dataset_drift(reference, current, columns=["drifted_num"])
    assert report.n_columns == 1
    assert report.share_drifted == pytest.approx(1.0)


def test_missing_values_are_dropped_before_comparing(rng):
    reference = pd.Series(rng.normal(0, 1, 300))
    current = pd.Series(np.concatenate([rng.normal(0, 1, 300), np.full(50, np.nan)]))
    report = detect_column_drift(reference, current, column="with_nans")
    assert report.n_current == 300
    assert report.drifted is False


def test_empty_report_is_not_drifted():
    report = detect_dataset_drift(pd.DataFrame(), pd.DataFrame())
    assert report.n_columns == 0
    assert report.share_drifted == 0.0
    assert report.dataset_drifted is False


def test_format_report_mentions_every_column_and_the_share(frames):
    reference, current = frames
    text = format_report(detect_dataset_drift(reference, current, label="fixture.csv"))
    for column in ["stable_num", "drifted_num", "stable_cat", "drifted_cat"]:
        assert column in text
    assert "Share of drifted components" in text
    assert "fixture.csv" in text


def test_report_serializes_to_json(frames):
    """The report is written to disk by the worker job, so it has to survive
    a round trip - no NaN or numpy scalars leaking into the payload."""
    reference, current = frames
    payload = detect_dataset_drift(reference, current).model_dump_json()
    assert '"drifted"' in payload
    assert "NaN" not in payload


def test_dataset_report_serializes_its_verdict():
    """The roll-up fields were plain @property, so model_dump dropped them -
    worker/jobs.py then cached, and ml_debugger handed the LLM, a report with
    no verdict in it at all."""
    frame = pd.DataFrame({"x": np.arange(50.0)})
    report = detect_dataset_drift(frame, frame + 100.0)

    dumped = report.model_dump(mode="json")
    assert dumped["dataset_drifted"] == report.dataset_drifted
    assert dumped["n_drifted"] == report.n_drifted
    assert dumped["n_columns"] == report.n_columns
    assert dumped["share_drifted"] == pytest.approx(report.share_drifted)


def test_explicit_columns_are_filtered_to_those_present_in_both_frames():
    """`columns` reaches here from the LLM (ml_debugger_tools.run_drift_analysis),
    naming columns it read off the diff - an unknown one used to raise KeyError
    instead of simply being skipped like any other unshared column."""
    reference = pd.DataFrame({"x": np.arange(50.0), "y": np.arange(50.0)})
    current = pd.DataFrame({"x": np.arange(50.0) + 100.0})

    report = detect_dataset_drift(reference, current, columns=["x", "y", "not_a_column"])
    assert [c.column for c in report.columns] == ["x"]
