# src/drift/methods.py
"""Data-drift detectors (Workflow.md Workflow 2: 'compares current
input/output distribution to baseline, flags drift').

Every public detector has the same shape - f(reference, current, **opts) ->
MethodResult - so drift/detector.py can run all of them over a column
without special-casing. Implemented directly on numpy/scipy rather than
through evidently so the thresholds and estimators stay inspectable; the
statistic's scale and direction differ per method, which is exactly why
MethodResult carries its own threshold.

Two groups:
  - Hypothesis tests (KS, MMD): statistic + p-value, drift when p < alpha.
  - Scores (PSI, Euclidean, Cosine, Page-Hinkley, model-based): drift when
    the score rises above a magnitude threshold.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.schemas.drift import ColumnKind, DriftMethod, MethodResult

# Guards log(0) in PSI and zero-norm vectors in cosine distance.
_EPS = 1e-6


class DriftInputError(ValueError):
    """Raised when a sample cannot be tested at all (empty, or constant in a
    way that makes the statistic undefined). detector.py turns this into a
    'skipped' entry rather than failing the whole report."""


def _numeric(values) -> np.ndarray:
    """Flatten to a 1-D float array, dropping NaN/inf - every detector here
    is undefined on missing values."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise DriftInputError("sample is empty after dropping non-finite values")
    return arr


def _probability_vectors(
    reference, current, kind: ColumnKind, buckets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned probability vectors for reference and current.

    Numerical columns are binned on quantiles of the reference (so bins
    carry equal reference mass), with the outer edges opened to +/-inf so
    current values outside the reference range still land somewhere.
    Categorical columns use the union of observed categories. A numerical
    column whose quantiles collapse (constant, or near-constant like a 0/1
    flag) falls back to the categorical path instead of erroring.
    """
    if kind is ColumnKind.CATEGORICAL:
        return _category_vectors(reference, current)

    ref, cur = _numeric(reference), _numeric(current)
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, buckets + 1)))
    if edges.size < 2:
        return _category_vectors(ref, cur)

    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / ref.size
    cur_pct = np.histogram(cur, bins=edges)[0] / cur.size
    return ref_pct, cur_pct


def _category_vectors(reference, current) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference).ravel()
    cur = np.asarray(current).ravel()
    if ref.size == 0 or cur.size == 0:
        raise DriftInputError("sample is empty")
    categories = np.union1d(ref, cur)
    ref_pct = np.array([np.count_nonzero(ref == c) for c in categories], dtype=float) / ref.size
    cur_pct = np.array([np.count_nonzero(cur == c) for c in categories], dtype=float) / cur.size
    return ref_pct, cur_pct


def ks_test(reference, current, *, threshold: float = 0.05) -> MethodResult:
    """Two-sample Kolmogorov-Smirnov test.

    The statistic is the largest gap between the two empirical CDFs;
    the p-value asks how likely a gap that big is if both samples came from
    the same distribution. Numerical columns only."""
    ref, cur = _numeric(reference), _numeric(current)
    result = stats.ks_2samp(ref, cur)
    p_value = float(result.pvalue)
    return MethodResult(
        method=DriftMethod.KS,
        statistic=float(result.statistic),
        p_value=p_value,
        threshold=threshold,
        drifted=p_value < threshold,
        detail="max CDF gap; drift when p-value < threshold",
    )


def psi(
    reference,
    current,
    *,
    kind: ColumnKind = ColumnKind.NUMERICAL,
    buckets: int = 10,
    threshold: float = 0.2,
) -> MethodResult:
    """Population Stability Index.

    Sums (current% - reference%) * ln(current% / reference%) over bins, so
    it punishes mass moving between bins in either direction. Conventional
    reading: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant."""
    ref_pct, cur_pct = _probability_vectors(reference, current, kind, buckets)
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return MethodResult(
        method=DriftMethod.PSI,
        statistic=value,
        threshold=threshold,
        drifted=value >= threshold,
        detail=f"{len(ref_pct)} bins; <0.1 stable, 0.1-0.2 moderate, >0.2 significant",
    )


def page_hinkley_stream(
    values,
    *,
    delta: float = 0.005,
    threshold: float | None = None,
    alpha: float = 0.9999,
    sensitivity: float = 3.0,
) -> tuple[bool, int | None, float, float]:
    """Page-Hinkley change detection over a single ordered stream.

    Tracks the cumulative deviation of each point from the running mean
    (minus a small tolerance `delta`), and fires when that cumulative sum
    departs from its own running extreme by more than `threshold`. Two-sided:
    catches both upward and downward shifts. `alpha` is a forgetting factor
    that keeps the sum from being dominated by ancient history.

    `threshold=None` scales the cutoff as sensitivity * sqrt(len(stream)).
    That matters: on a stream with no change at all, the cumulative sum is a
    random walk whose peak excursion grows like sqrt(n), so any fixed
    threshold false-positives once the stream is long enough. Measured over
    driftless standard-normal streams (n from 100 to 5000), the 99th
    percentile of the peak statistic sits at ~2.5-2.9 * sqrt(n), so the
    default sensitivity of 3.0 targets a false-alarm rate around 1%. Pass an
    explicit threshold for classic fixed-lambda streaming use.

    Returns (detected, index_of_detection, peak_statistic, threshold_used)."""
    stream = _numeric(values)
    if threshold is None:
        threshold = sensitivity * float(np.sqrt(stream.size))
    running_mean = 0.0
    cumulative = 0.0
    min_sum = 0.0
    max_sum = 0.0
    peak = 0.0
    detected_at: int | None = None

    for i, x in enumerate(stream):
        running_mean += (x - running_mean) / (i + 1)
        cumulative = alpha * cumulative + (x - running_mean - delta)
        min_sum = min(min_sum, cumulative)
        max_sum = max(max_sum, cumulative)
        statistic = max(cumulative - min_sum, max_sum - cumulative)
        peak = max(peak, statistic)
        if statistic > threshold and detected_at is None:
            detected_at = i

    return detected_at is not None, detected_at, peak, threshold


def page_hinkley(
    reference,
    current,
    *,
    delta: float = 0.005,
    threshold: float | None = None,
    alpha: float = 0.9999,
    sensitivity: float = 3.0,
) -> MethodResult:
    """Page-Hinkley applied to reference-then-current as one stream.

    Unlike the other detectors here, Page-Hinkley is sequential: it cares
    *when* the level shifted, not just that two bags of numbers differ. The
    stream is standardized by the reference's mean and standard deviation
    first, so `delta` and `threshold` are in reference-sigma units and the
    same defaults work across columns of wildly different scale.

    Drift is only reported when the change point falls in the current
    window - a change detected inside the reference itself is the reference
    being unstable, not the current data drifting away from it."""
    ref, cur = _numeric(reference), _numeric(current)
    scale = float(ref.std())
    if scale < _EPS:
        raise DriftInputError("reference is constant; Page-Hinkley needs a scale to normalize by")

    centered = (np.concatenate([ref, cur]) - float(ref.mean())) / scale
    detected, index, peak, threshold_used = page_hinkley_stream(
        centered, delta=delta, threshold=threshold, alpha=alpha, sensitivity=sensitivity
    )
    drifted = detected and index is not None and index >= ref.size
    where = "no change point" if index is None else f"change point at stream index {index}"
    return MethodResult(
        method=DriftMethod.PAGE_HINKLEY,
        statistic=peak,
        threshold=threshold_used,
        drifted=bool(drifted),
        detail=f"{where}; reference ends at index {ref.size - 1} (sigma units)",
    )


def euclidean_distance(
    reference,
    current,
    *,
    kind: ColumnKind = ColumnKind.NUMERICAL,
    buckets: int = 10,
    threshold: float = 0.1,
) -> MethodResult:
    """Euclidean distance between the two binned distributions.

    Straight L2 distance over probability vectors - bounded by sqrt(2), and
    insensitive to *which* bins moved, unlike PSI."""
    ref_pct, cur_pct = _probability_vectors(reference, current, kind, buckets)
    value = float(np.linalg.norm(cur_pct - ref_pct))
    return MethodResult(
        method=DriftMethod.EUCLIDEAN,
        statistic=value,
        threshold=threshold,
        drifted=value >= threshold,
        detail="L2 distance between binned distributions; 0 identical, sqrt(2) disjoint",
    )


def cosine_distance(
    reference,
    current,
    *,
    kind: ColumnKind = ColumnKind.NUMERICAL,
    buckets: int = 10,
    threshold: float = 0.1,
) -> MethodResult:
    """Cosine distance (1 - cosine similarity) between binned distributions.

    Compares the *shape* of the distributions rather than their magnitudes,
    so it reacts to mass changing bins but not to both histograms being
    uniformly scaled."""
    ref_pct, cur_pct = _probability_vectors(reference, current, kind, buckets)
    denominator = float(np.linalg.norm(ref_pct) * np.linalg.norm(cur_pct))
    if denominator < _EPS:
        raise DriftInputError("a distribution vector has zero norm")
    similarity = float(np.dot(ref_pct, cur_pct) / denominator)
    value = 1.0 - min(1.0, max(-1.0, similarity))
    return MethodResult(
        method=DriftMethod.COSINE,
        statistic=value,
        threshold=threshold,
        drifted=value >= threshold,
        detail="1 - cosine similarity of binned distributions; 0 identical, 1 orthogonal",
    )


def model_based(
    reference,
    current,
    *,
    threshold: float = 0.55,
    n_splits: int = 5,
    random_state: int = 0,
) -> MethodResult:
    """Domain-classifier drift detection.

    Labels reference rows 0 and current rows 1 and asks a classifier to tell
    them apart, scoring out-of-fold ROC AUC. If the two samples are drawn
    from the same distribution the classifier cannot beat chance and AUC sits
    at ~0.5; an AUC meaningfully above that means the samples are separable,
    i.e. something about the data changed. Accepts 2-D input, so it also
    works as a whole-dataset detector over several columns at once."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref.reshape(len(ref), -1)
    cur = cur.reshape(len(cur), -1)
    if ref.shape[1] != cur.shape[1]:
        raise DriftInputError("reference and current have different numbers of features")

    features = np.vstack([ref, cur])
    if not np.isfinite(features).all():
        raise DriftInputError("input contains non-finite values")
    labels = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])

    # Every fold needs both classes present, so the split count cannot
    # exceed the size of the smaller sample.
    splits = min(n_splits, len(ref), len(cur))
    if splits < 2:
        raise DriftInputError("need at least 2 rows per sample to cross-validate")

    classifier = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=1
    )
    probabilities = cross_val_predict(
        classifier,
        features,
        labels,
        cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state),
        method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(labels, probabilities))
    return MethodResult(
        method=DriftMethod.MODEL_BASED,
        statistic=auc,
        threshold=threshold,
        drifted=auc >= threshold,
        detail=f"out-of-fold ROC AUC over {splits} folds; 0.5 means indistinguishable",
    )


def _rbf_kernel(matrix: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * cdist(matrix, matrix, metric="sqeuclidean"))


def _mmd_squared(kernel: np.ndarray, n_ref: int) -> float:
    """Unbiased MMD^2 estimate from a pooled kernel matrix whose first
    n_ref rows/columns are the reference sample."""
    k_xx = kernel[:n_ref, :n_ref]
    k_yy = kernel[n_ref:, n_ref:]
    k_xy = kernel[:n_ref, n_ref:]
    m, n = n_ref, kernel.shape[0] - n_ref
    # Diagonals are dropped: k(x_i, x_i) carries no information about the
    # distributions and would bias the estimate upward.
    term_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    term_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    return float(term_xx + term_yy - 2.0 * k_xy.mean())


def mmd(
    reference,
    current,
    *,
    threshold: float = 0.05,
    n_permutations: int = 200,
    max_samples: int = 500,
    random_state: int = 0,
) -> MethodResult:
    """Maximum Mean Discrepancy with an RBF kernel and a permutation test.

    MMD embeds both samples in a reproducing kernel Hilbert space and
    measures the distance between their mean embeddings, which is zero if
    and only if the distributions match - so unlike KS it is sensitive to
    differences beyond the marginal CDF, and it extends to multivariate
    input. The kernel bandwidth uses the median heuristic (median pairwise
    distance of the pooled sample).

    MMD^2 has no closed-form null distribution, so significance comes from
    shuffling the labels `n_permutations` times and seeing how often chance
    reproduces the observed value. Cost is quadratic in sample size, hence
    `max_samples` subsampling."""
    rng = np.random.default_rng(random_state)
    ref = _numeric(reference)
    cur = _numeric(current)
    if ref.size < 2 or cur.size < 2:
        raise DriftInputError("MMD needs at least 2 points per sample")

    if ref.size > max_samples:
        ref = rng.choice(ref, max_samples, replace=False)
    if cur.size > max_samples:
        cur = rng.choice(cur, max_samples, replace=False)

    pooled = np.concatenate([ref, cur]).reshape(-1, 1)
    distances = cdist(pooled, pooled, metric="euclidean")
    median = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 0.0
    if median < _EPS:
        raise DriftInputError("pooled sample is constant; MMD bandwidth undefined")
    gamma = 1.0 / (2.0 * median**2)

    kernel = _rbf_kernel(pooled, gamma)
    observed = _mmd_squared(kernel, ref.size)

    # Permuting row/column order of the pooled kernel is equivalent to
    # reassigning which points belong to which sample, without recomputing
    # the (expensive) kernel each time.
    total = kernel.shape[0]
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        order = rng.permutation(total)
        permuted = kernel[np.ix_(order, order)]
        if _mmd_squared(permuted, ref.size) >= observed:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_permutations + 1)

    return MethodResult(
        method=DriftMethod.MMD,
        statistic=observed,
        p_value=p_value,
        threshold=threshold,
        drifted=p_value < threshold,
        detail=f"RBF kernel (median heuristic), {n_permutations} permutations",
    )
