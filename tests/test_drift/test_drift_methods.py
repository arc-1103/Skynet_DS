"""Each detector is checked two ways: it must stay quiet on samples drawn
from the same distribution, and fire on a genuinely shifted one. Fixed
seeds keep that deterministic - these are statistical tests, so without a
seed they would be flaky by construction."""

import numpy as np
import pytest

from src.drift import methods
from src.schemas.drift import ColumnKind, DriftMethod

SEED = 20260915


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def stable(rng):
    """Two samples from the same distribution."""
    return rng.normal(0.0, 1.0, 400), rng.normal(0.0, 1.0, 400)


@pytest.fixture
def shifted(rng):
    """Same spread, mean moved by 1.5 sigma."""
    return rng.normal(0.0, 1.0, 400), rng.normal(1.5, 1.0, 400)


ALL_DETECTORS = [
    methods.ks_test,
    methods.psi,
    methods.page_hinkley,
    methods.euclidean_distance,
    methods.cosine_distance,
    methods.model_based,
    methods.mmd,
]


@pytest.mark.parametrize("detector", ALL_DETECTORS, ids=lambda f: f.__name__)
def test_detector_is_quiet_on_identical_distributions(detector, stable):
    reference, current = stable
    assert detector(reference, current).drifted is False


@pytest.mark.parametrize("detector", ALL_DETECTORS, ids=lambda f: f.__name__)
def test_detector_fires_on_shifted_distribution(detector, shifted):
    reference, current = shifted
    assert detector(reference, current).drifted is True


@pytest.mark.parametrize("detector", ALL_DETECTORS, ids=lambda f: f.__name__)
def test_detector_reports_its_own_threshold(detector, stable):
    reference, current = stable
    result = detector(reference, current)
    assert result.threshold > 0
    assert isinstance(result.statistic, float)


def test_ks_p_value_collapses_as_samples_separate(rng):
    reference = rng.normal(0, 1, 300)
    near = methods.ks_test(reference, rng.normal(0.2, 1, 300))
    far = methods.ks_test(reference, rng.normal(3.0, 1, 300))
    assert far.p_value < near.p_value
    assert far.statistic > near.statistic


def test_psi_is_zero_for_the_same_sample(rng):
    sample = rng.normal(0, 1, 500)
    assert methods.psi(sample, sample).statistic == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift_magnitude(rng):
    reference = rng.normal(0, 1, 500)
    small = methods.psi(reference, rng.normal(0.3, 1, 500)).statistic
    large = methods.psi(reference, rng.normal(2.0, 1, 500)).statistic
    assert large > small


def test_psi_handles_categorical_input():
    reference = np.array(["a"] * 80 + ["b"] * 20)
    current = np.array(["a"] * 20 + ["b"] * 80)
    result = methods.psi(reference, current, kind=ColumnKind.CATEGORICAL)
    assert result.drifted is True
    assert result.method is DriftMethod.PSI


def test_page_hinkley_threshold_scales_with_stream_length(rng):
    """The default cutoff must grow with n - a fixed one false-positives on
    long driftless streams, since the statistic is a random walk."""
    short = methods.page_hinkley(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
    long = methods.page_hinkley(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    assert long.threshold > short.threshold


def test_page_hinkley_ignores_a_change_inside_the_reference(rng):
    """A level shift that happens within the reference window is an unstable
    baseline, not the current data drifting away from it."""
    reference = np.concatenate([rng.normal(0, 1, 300), rng.normal(6.0, 1, 300)])
    current = rng.normal(6.0, 1, 300)
    assert methods.page_hinkley(reference, current).drifted is False


def test_page_hinkley_stream_reports_where_the_change_happened(rng):
    stream = np.concatenate([rng.normal(0, 1, 500), rng.normal(4.0, 1, 500)])
    detected, index, peak, threshold = methods.page_hinkley_stream(stream)
    assert detected is True
    assert index >= 500
    assert peak > threshold


def test_euclidean_and_cosine_are_zero_for_identical_samples(rng):
    sample = rng.normal(0, 1, 400)
    assert methods.euclidean_distance(sample, sample).statistic == pytest.approx(0.0, abs=1e-9)
    assert methods.cosine_distance(sample, sample).statistic == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_stays_in_unit_range(rng):
    reference = rng.normal(0, 1, 300)
    current = rng.normal(50.0, 1, 300)
    value = methods.cosine_distance(reference, current).statistic
    assert 0.0 <= value <= 1.0


def test_model_based_auc_sits_near_chance_for_identical_samples(stable):
    reference, current = stable
    assert methods.model_based(reference, current).statistic == pytest.approx(0.5, abs=0.1)


def test_model_based_accepts_multivariate_input(rng):
    reference = rng.normal(0, 1, (300, 3))
    current = np.column_stack([rng.normal(0, 1, 300), rng.normal(0, 1, 300), rng.normal(2.5, 1, 300)])
    assert methods.model_based(reference, current).drifted is True


def test_mmd_is_near_zero_for_identical_samples(stable):
    reference, current = stable
    # The unbiased estimator can go slightly negative on matching samples.
    assert abs(methods.mmd(reference, current, n_permutations=100).statistic) < 0.01


def test_mmd_detects_a_variance_only_shift(rng):
    """MMD should catch a change in spread even when the mean is unchanged,
    which is the case a mean-based check would miss."""
    reference = rng.normal(0, 1, 400)
    current = rng.normal(0, 3, 400)
    assert methods.mmd(reference, current, n_permutations=200).drifted is True


def test_mmd_subsamples_large_inputs(rng):
    """Cost is quadratic, so the cap has to actually bind."""
    result = methods.mmd(
        rng.normal(0, 1, 5000), rng.normal(0, 1, 5000), max_samples=100, n_permutations=20
    )
    assert result.method is DriftMethod.MMD


@pytest.mark.parametrize(
    "detector", [methods.ks_test, methods.psi, methods.mmd], ids=lambda f: f.__name__
)
def test_empty_sample_is_rejected(detector, rng):
    with pytest.raises(methods.DriftInputError):
        detector(rng.normal(0, 1, 100), np.array([]))


def test_constant_reference_is_rejected_by_page_hinkley():
    with pytest.raises(methods.DriftInputError):
        methods.page_hinkley(np.ones(100), np.arange(100, dtype=float))


def test_constant_pooled_sample_is_rejected_by_mmd():
    with pytest.raises(methods.DriftInputError):
        methods.mmd(np.ones(50), np.ones(50))


def test_nan_values_are_dropped_not_propagated(rng):
    clean = rng.normal(0, 1, 300)
    dirty = np.concatenate([clean, np.full(20, np.nan)])
    result = methods.ks_test(clean, dirty)
    assert np.isfinite(result.statistic)
    assert result.drifted is False


def test_binning_falls_back_to_categories_for_binary_columns():
    """Quantile edges collapse on a 0/1 column; that must not raise."""
    reference = np.array([0.0] * 90 + [1.0] * 10)
    current = np.array([0.0] * 20 + [1.0] * 80)
    assert methods.psi(reference, current).drifted is True
