"""The SHAP root-cause engine has one job: given a table that drifted, name
the feature responsible. These tests plant a known culprit and check it
comes back top - and, just as importantly, check that nothing is named when
nothing drifted.

Seeded throughout: the surrogate is a trained model, so unseeded runs would
be flaky by construction."""

import numpy as np
import pandas as pd
import pytest

from src.drift.detector import detect_column_drift
from src.drift.root_cause import (
    RootCauseInputError,
    explain_drift,
    format_root_cause,
    to_cause_candidates,
)
from src.schemas.root_cause import ShiftDirection

SEED = 20260915
N = 600


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def stable_frames(rng):
    """Two draws from the same generating process - nothing drifted."""
    build = lambda: pd.DataFrame(
        {
            "latitude": rng.normal(12.97, 0.02, N),
            "longitude": rng.normal(77.59, 0.02, N),
            "velocity_kmh": rng.normal(28.0, 6.0, N),
            "hdop": rng.gamma(2.0, 0.5, N),
        }
    )
    return build(), build()


@pytest.fixture
def velocity_drift_frames(stable_frames):
    """Exactly one feature moves: velocity climbs by 3 sigma."""
    reference, current = stable_frames
    current = current.copy()
    current["velocity_kmh"] = np.random.default_rng(SEED + 1).normal(46.0, 6.0, N)
    return reference, current


def test_isolates_the_single_drifted_feature(velocity_drift_frames):
    reference, current = velocity_drift_frames
    report = explain_drift(reference, current)

    assert report.separable is True
    assert report.attributions[0].feature == "velocity_kmh"
    # The culprit should dominate, not merely edge ahead.
    assert report.attributions[0].share > 0.5


def test_isolates_two_drifted_features(stable_frames, rng):
    reference, current = stable_frames
    current = current.copy()
    current["velocity_kmh"] = rng.normal(46.0, 6.0, N)
    current["hdop"] = rng.gamma(2.0, 1.6, N)

    report = explain_drift(reference, current)

    assert {a.feature for a in report.attributions[:2]} == {"velocity_kmh", "hdop"}


def test_ranks_the_driver_above_a_noisy_correlated_copy(rng):
    """The payoff over testing columns one at a time: when a derived feature
    drifts only because its source did, independent per-column tests flag
    both and cannot say which came first. The surrogate sees them together
    and puts the blame on the driver."""
    velocity_ref = rng.normal(28.0, 6.0, N)
    velocity_cur = rng.normal(46.0, 6.0, N)
    reference = pd.DataFrame(
        {"velocity": velocity_ref, "echo": velocity_ref * 0.5 + rng.normal(0, 8, N)}
    )
    current = pd.DataFrame(
        {"velocity": velocity_cur, "echo": velocity_cur * 0.5 + rng.normal(0, 8, N)}
    )

    # Both columns genuinely drifted, so a per-column test cannot separate them.
    assert detect_column_drift(reference["velocity"], current["velocity"]).drifted
    assert detect_column_drift(reference["echo"], current["echo"]).drifted

    report = explain_drift(reference, current)
    assert report.attributions[0].feature == "velocity"
    assert report.attributions[0].share > 2 * report.attributions[1].share


def test_reports_nothing_when_nothing_drifted(stable_frames):
    reference, current = stable_frames
    report = explain_drift(reference, current)
    assert report.separable is False
    assert report.surrogate_auc < report.auc_threshold


def test_attribution_is_held_out_so_wide_small_data_does_not_fake_drift(rng):
    """50 rows and 30 noise features is where a boosted tree memorizes the
    training split and in-sample SHAP invents a confident ranking."""
    reference = pd.DataFrame({f"f{i}": rng.normal(0, 1, 50) for i in range(30)})
    current = pd.DataFrame({f"f{i}": rng.normal(0, 1, 50) for i in range(30)})
    report = explain_drift(reference, current)
    assert report.separable is False
    assert to_cause_candidates(report) == []


def test_a_small_sample_auc_must_also_be_significant(rng):
    """On a tiny test set the AUC of pure noise swings far past 0.55 on its
    own - across random states this same noise scores anywhere from 0.19 to
    0.56 - so the effect-size threshold alone would wave it through."""
    reference = pd.DataFrame({f"f{i}": rng.normal(0, 1, 50) for i in range(30)})
    current = pd.DataFrame({f"f{i}": rng.normal(0, 1, 50) for i in range(30)})
    report = explain_drift(reference, current, random_state=0)

    assert report.surrogate_auc > report.auc_threshold
    assert report.auc_p_value > report.alpha
    assert report.separable is False


def test_a_real_shift_clears_both_halves_of_the_gate(velocity_drift_frames):
    reference, current = velocity_drift_frames
    report = explain_drift(reference, current)
    assert report.surrogate_auc > report.auc_threshold
    assert report.auc_p_value < report.alpha
    assert report.separable is True


def test_direction_is_reported_for_level_shifts(stable_frames, rng):
    reference, current = stable_frames
    current = current.copy()
    current["velocity_kmh"] = rng.normal(46.0, 6.0, N)
    current["hdop"] = rng.gamma(2.0, 0.12, N)

    by_feature = {a.feature: a for a in explain_drift(reference, current).attributions}
    assert by_feature["velocity_kmh"].direction is ShiftDirection.INCREASED
    assert by_feature["hdop"].direction is ShiftDirection.DECREASED
    assert by_feature["velocity_kmh"].current_mean > by_feature["velocity_kmh"].reference_mean


def test_variance_only_shift_is_reported_as_reshaped(stable_frames, rng):
    """Same centre, wider spread - calling that 'increased' would be wrong."""
    reference, current = stable_frames
    current = current.copy()
    current["velocity_kmh"] = rng.normal(28.0, 20.0, N)

    by_feature = {a.feature: a for a in explain_drift(reference, current).attributions}
    assert by_feature["velocity_kmh"].direction is ShiftDirection.RESHAPED


def test_categorical_drift_is_attributed(rng):
    reference = pd.DataFrame(
        {
            "device": rng.choice(["A", "B", "C"], N, p=[0.8, 0.1, 0.1]),
            "inert": rng.normal(0, 1, N),
        }
    )
    current = pd.DataFrame(
        {
            "device": rng.choice(["A", "B", "C"], N, p=[0.1, 0.1, 0.8]),
            "inert": rng.normal(0, 1, N),
        }
    )
    report = explain_drift(reference, current)
    assert report.separable is True
    assert report.attributions[0].feature == "device"
    assert report.attributions[0].direction is ShiftDirection.RESHAPED
    # Category means are meaningless, so they are left unset rather than faked.
    assert report.attributions[0].reference_mean is None


def test_shares_are_a_distribution(velocity_drift_frames):
    reference, current = velocity_drift_frames
    report = explain_drift(reference, current)
    assert sum(a.share for a in report.attributions) == pytest.approx(1.0)
    assert all(0.0 <= a.share <= 1.0 for a in report.attributions)


def test_attributions_are_sorted_by_magnitude(velocity_drift_frames):
    reference, current = velocity_drift_frames
    values = [a.mean_abs_shap for a in explain_drift(reference, current).attributions]
    assert values == sorted(values, reverse=True)


def test_results_are_deterministic_for_a_fixed_seed(velocity_drift_frames):
    reference, current = velocity_drift_frames
    first = explain_drift(reference, current, random_state=7)
    second = explain_drift(reference, current, random_state=7)
    assert first.surrogate_auc == second.surrogate_auc
    assert first.top_features == second.top_features


def test_cause_candidates_carry_discounted_confidence(velocity_drift_frames):
    reference, current = velocity_drift_frames
    report = explain_drift(reference, current)
    candidates = to_cause_candidates(report, top_n=2)

    assert len(candidates) == 2
    assert all(c.kind == "data" for c in candidates)
    assert all(0.0 <= c.confidence <= 1.0 for c in candidates)
    assert "velocity_kmh" in candidates[0].description
    # Confidence is the share discounted by separability, never the raw share.
    assert candidates[0].confidence <= report.attributions[0].share
    assert candidates[0].confidence > candidates[1].confidence


def test_cause_candidates_cite_their_evidence(velocity_drift_frames):
    reference, current = velocity_drift_frames
    report = explain_drift(reference, current, label="data/gps.csv")
    evidence = to_cause_candidates(report, top_n=1)[0].evidence_refs
    assert any("shap:mean_abs" in ref for ref in evidence)
    assert any("surrogate_auc" in ref for ref in evidence)
    assert "data/gps.csv" in evidence


def test_weak_separability_yields_weak_confidence(stable_frames, rng):
    """A barely-separable surrogate must not emit a confident cause just
    because one feature dominates its ranking."""
    reference, current = stable_frames
    current = current.copy()
    current["velocity_kmh"] = rng.normal(29.5, 6.0, N)

    report = explain_drift(reference, current, auc_threshold=0.5)
    candidates = to_cause_candidates(report, top_n=1)
    if candidates:
        assert candidates[0].confidence < 0.5


def test_too_few_rows_is_rejected(rng):
    tiny = pd.DataFrame({"a": rng.normal(0, 1, 10)})
    with pytest.raises(RootCauseInputError, match="at least"):
        explain_drift(tiny, tiny)


def test_frames_with_no_shared_columns_are_rejected(rng):
    left = pd.DataFrame({"a": rng.normal(0, 1, 50)})
    right = pd.DataFrame({"b": rng.normal(0, 1, 50)})
    with pytest.raises(RootCauseInputError, match="share no columns"):
        explain_drift(left, right)


def test_only_shared_columns_are_attributed(velocity_drift_frames, rng):
    reference, current = velocity_drift_frames
    current = current.copy()
    current["added_in_production"] = rng.normal(0, 1, N)
    report = explain_drift(reference, current)
    assert "added_in_production" not in report.top_features


def test_missing_values_do_not_break_attribution(velocity_drift_frames):
    reference, current = velocity_drift_frames
    current = current.copy()
    current.loc[current.index[:50], "hdop"] = np.nan
    report = explain_drift(reference, current)
    assert report.attributions[0].feature == "velocity_kmh"


def test_format_hides_the_ranking_when_there_is_nothing_to_explain(stable_frames):
    reference, current = stable_frames
    text = format_root_cause(explain_drift(reference, current))
    assert "No drift to attribute" in text
    assert "velocity_kmh" not in text


def test_format_shows_the_culprit_when_there_is_one(velocity_drift_frames):
    reference, current = velocity_drift_frames
    text = format_root_cause(explain_drift(reference, current, label="gps.csv"))
    assert "velocity_kmh" in text
    assert "gps.csv" in text
    assert "increased" in text


def test_report_serializes_to_json(velocity_drift_frames):
    reference, current = velocity_drift_frames
    payload = explain_drift(reference, current).model_dump_json()
    assert '"velocity_kmh"' in payload
    assert "NaN" not in payload
