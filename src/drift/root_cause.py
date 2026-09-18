# src/drift/root_cause.py
"""SHAP root-cause engine: which feature is behind the drift?

drift/methods.py answers whether a column moved. This answers the question
that comes next - given a whole table that drifted, which columns actually
account for it - by training a lightweight surrogate (an XGBoost classifier)
to tell reference rows from production rows, then reading the model with
shap.TreeExplainer to see which features it leaned on.

Why a surrogate rather than ranking the per-column drift scores: the
surrogate sees the columns together, so it discounts a feature that only
looks shifted because it is correlated with the one that really moved, and
it can pick up joint shifts that no single-column test would register.

Two cautions are built into the output rather than left to the caller:

  - SHAP is only worth reading if the surrogate can separate the samples at
    all. A classifier at chance level has found nothing, and its feature
    ranking is noise sorted by magnitude - hence the `separable` gate.
  - Attribution is computed on held-out rows. A boosted tree will happily
    memorize a few hundred rows and report confident attributions for
    features that carry no signal, so in-sample SHAP overstates its case.

What this does and does not establish: it isolates the features that
*shifted*, which is the candidate root cause for a regression that followed.
It does not by itself prove those features caused a metric to drop - that
link is what graph/ is for, which is why to_cause_candidates() hands these
over as CauseCandidates to be ranked against code and config changes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.drift.tables import read_table
from src.graph.causal import CauseCandidate
from src.schemas.root_cause import DriftRootCause, FeatureAttribution, ShiftDirection

# Below this fraction of a reference standard deviation, a change in mean is
# not a shift in level - the distribution was reshaped some other way.
_LEVEL_SHIFT_SIGMA = 0.1

# Enough rows per side that a stratified split leaves a usable test set.
_MIN_ROWS_PER_SAMPLE = 20


class RootCauseInputError(ValueError):
    """Raised when the comparison cannot be attributed at all."""


def _encode(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align the two frames and ordinal-encode non-numeric columns.

    Codes are ordinal, which a tree ensemble can still carve up one category
    at a time. Missing values become NaN rather than a sentinel code, so
    XGBoost routes them with its own missing-value handling instead of
    treating 'absent' as just another category."""
    shared = [c for c in reference.columns if c in current.columns]
    if not shared:
        raise RootCauseInputError("reference and current share no columns")

    ref, cur = reference[shared].copy(), current[shared].copy()
    categorical: list[str] = []
    for column in shared:
        is_numeric = pd.api.types.is_numeric_dtype(ref[column]) and pd.api.types.is_numeric_dtype(
            cur[column]
        )
        if is_numeric and not pd.api.types.is_bool_dtype(ref[column]):
            continue
        categorical.append(column)
        levels = pd.Index(
            sorted(set(ref[column].dropna().unique()) | set(cur[column].dropna().unique()))
        )
        for frame in (ref, cur):
            codes = pd.Categorical(frame[column], categories=levels).codes.astype(float)
            frame[column] = np.where(codes < 0, np.nan, codes)

    return ref, cur, categorical


def _mean_abs_shap(explainer: shap.TreeExplainer, features: pd.DataFrame) -> np.ndarray:
    """Per-feature mean |SHAP|, flattened across whichever layout the
    explainer returns (binary classifiers have shipped both a 2-D array and
    a 3-D one with a trailing class axis across shap versions)."""
    values = np.asarray(explainer.shap_values(features))
    if values.ndim == 3:
        values = values[..., -1]
    return np.abs(values).mean(axis=0)


def _direction(
    reference: pd.Series, current: pd.Series, is_categorical: bool
) -> tuple[ShiftDirection, float | None, float | None]:
    """Label the shift as a change in level, or in shape.

    The mean difference is measured against the wider of the two spreads,
    not the reference's alone. If production is three times as dispersed,
    its sample mean wanders by that much too, and scoring the gap against
    the tighter reference would read that wobble as a level shift when what
    actually changed is the shape."""
    if is_categorical:
        return ShiftDirection.RESHAPED, None, None

    ref_mean, cur_mean = float(reference.mean()), float(current.mean())
    spread = max(float(reference.std()), float(current.std()))
    if spread == 0 or abs(cur_mean - ref_mean) < _LEVEL_SHIFT_SIGMA * spread:
        return ShiftDirection.RESHAPED, ref_mean, cur_mean
    direction = ShiftDirection.INCREASED if cur_mean > ref_mean else ShiftDirection.DECREASED
    return direction, ref_mean, cur_mean


def explain_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    auc_threshold: float = 0.55,
    alpha: float = 0.05,
    test_size: float = 0.3,
    n_estimators: int = 150,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    random_state: int = 0,
    label: str | None = None,
) -> DriftRootCause:
    """Train a reference-vs-production surrogate and rank features by SHAP.

    The surrogate is kept deliberately shallow: the goal is to find which
    features carry the shift, not to squeeze out the last point of AUC, and
    a deep model would just memorize the training split."""
    ref, cur, categorical = _encode(reference, current)
    if len(ref) < _MIN_ROWS_PER_SAMPLE or len(cur) < _MIN_ROWS_PER_SAMPLE:
        raise RootCauseInputError(
            f"need at least {_MIN_ROWS_PER_SAMPLE} rows per sample to hold out a test set; "
            f"got {len(ref)} reference and {len(cur)} current"
        )

    features = pd.concat([ref, cur], ignore_index=True)
    labels = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    surrogate = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )
    surrogate.fit(x_train, y_train)
    scores = surrogate.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, scores))

    # ROC AUC is the Mann-Whitney U statistic rescaled, so the same test
    # gives the probability of scoring this well by chance at this test-set
    # size - the check a bare AUC threshold cannot make.
    auc_p_value = float(
        mannwhitneyu(scores[y_test == 1], scores[y_test == 0], alternative="greater").pvalue
    )

    mean_abs = _mean_abs_shap(shap.TreeExplainer(surrogate), x_test)
    total = float(mean_abs.sum())

    attributions: list[FeatureAttribution] = []
    for position, column in enumerate(features.columns):
        direction, ref_mean, cur_mean = _direction(
            reference[column], current[column], column in categorical
        )
        attributions.append(
            FeatureAttribution(
                feature=str(column),
                mean_abs_shap=float(mean_abs[position]),
                share=float(mean_abs[position] / total) if total > 0 else 0.0,
                direction=direction,
                reference_mean=ref_mean,
                current_mean=cur_mean,
            )
        )
    attributions.sort(key=lambda a: a.mean_abs_shap, reverse=True)

    return DriftRootCause(
        surrogate_auc=auc,
        auc_threshold=auc_threshold,
        auc_p_value=auc_p_value,
        alpha=alpha,
        separable=auc >= auc_threshold and auc_p_value < alpha,
        n_reference=len(ref),
        n_current=len(cur),
        attributions=attributions,
        label=label,
    )


def to_cause_candidates(report: DriftRootCause, *, top_n: int = 3) -> list[CauseCandidate]:
    """Hand the top attributions to graph/ as ranked candidate causes.

    Confidence is the feature's attribution share discounted by how well the
    surrogate separated the samples at all, so a weak surrogate cannot emit
    a confident cause no matter how lopsided its SHAP ranking looks. A
    surrogate that found nothing emits nothing."""
    if not report.separable:
        return []

    strength = report.separability_strength
    candidates: list[CauseCandidate] = []
    for attribution in report.attributions[:top_n]:
        movement = attribution.direction.value
        if attribution.reference_mean is not None and attribution.current_mean is not None:
            movement += f" ({attribution.reference_mean:.4g} -> {attribution.current_mean:.4g})"
        candidates.append(
            CauseCandidate(
                kind="data",
                description=(
                    f"feature '{attribution.feature}' {movement}; "
                    f"{attribution.share:.0%} of drift attribution"
                ),
                confidence=round(attribution.share * strength, 4),
                evidence_refs=[
                    f"shap:mean_abs={attribution.mean_abs_shap:.4g}",
                    f"surrogate_auc={report.surrogate_auc:.3f}",
                ]
                + ([report.label] if report.label else []),
            )
        )
    return candidates


def format_root_cause(report: DriftRootCause, *, top_n: int = 10) -> str:
    """Human-readable summary for CLI/log output."""
    header = f"Drift root cause{f' for {report.label}' if report.label else ''}"
    lines = [header, "=" * len(header)]
    shown_p = "<0.0001" if report.auc_p_value < 0.0001 else f"{report.auc_p_value:.4g}"
    lines.append(
        f"Surrogate AUC: {report.surrogate_auc:.3f} (threshold {report.auc_threshold}, "
        f"p={shown_p}, n_ref={report.n_reference}, n_cur={report.n_current})"
    )

    if not report.separable:
        reason = (
            "it scores no better than chance"
            if report.surrogate_auc < report.auc_threshold
            else f"its lead over chance is not significant at this sample size (p={report.auc_p_value:.3g})"
        )
        lines.append("")
        lines.append(
            f"No drift to attribute: the surrogate cannot separate reference from "
            f"production data - {reason}. The SHAP ranking would be noise, so it is not shown."
        )
        return "\n".join(lines)

    lines.append("")
    lines.append(f"{'feature':<24} {'share':>8} {'mean|SHAP|':>12}  shift")
    lines.append("-" * 64)
    for attribution in report.attributions[:top_n]:
        movement = attribution.direction.value
        if attribution.reference_mean is not None and attribution.current_mean is not None:
            movement += f" {attribution.reference_mean:.4g} -> {attribution.current_mean:.4g}"
        lines.append(
            f"{attribution.feature:<24} {attribution.share:>7.1%} "
            f"{attribution.mean_abs_shap:>12.5f}  {movement}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Isolate which features are behind a reference -> production data shift"
    )
    parser.add_argument("reference", help="Baseline CSV/Parquet file")
    parser.add_argument("current", help="Production CSV/Parquet file")
    parser.add_argument("--auc-threshold", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for the AUC")
    parser.add_argument("--top", type=int, default=10, help="How many features to show")
    parser.add_argument("--json", dest="json_out", help="Also write the full report as JSON here")
    args = parser.parse_args()

    root_cause = explain_drift(
        read_table(args.reference),
        read_table(args.current),
        auc_threshold=args.auc_threshold,
        alpha=args.alpha,
        label=f"{args.reference} -> {args.current}",
    )
    print(format_root_cause(root_cause, top_n=args.top))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(root_cause.model_dump_json(indent=2))
        print(f"\nJSON report written to {args.json_out}")

    raise SystemExit(1 if root_cause.separable else 0)
