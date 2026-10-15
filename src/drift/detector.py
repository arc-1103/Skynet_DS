# src/drift/detector.py
"""Runs the drift/methods.py detectors over columns and rolls the results
up into a dataset verdict.

The roll-up metric is 'Share of Drifted Components': each column is judged
by its own applicable detectors, and the dataset counts as drifted when the
fraction of drifted columns exceeds share_threshold. That is deliberately
not 'any column drifted' - on a wide table, at a 5% significance level, a
handful of columns will trip on noise alone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift import methods
from src.drift.tables import read_table
from src.schemas.drift import (
    ColumnDriftReport,
    ColumnKind,
    DatasetDriftReport,
    DriftMethod,
    MethodResult,
)

ALL_METHODS: tuple[DriftMethod, ...] = tuple(DriftMethod)

# Sequential/continuous detectors that are meaningless on unordered labels.
_NUMERIC_ONLY = frozenset({DriftMethod.KS, DriftMethod.PAGE_HINKLEY, DriftMethod.MMD})

# Detectors that bin the data themselves and so need to know the column kind.
_KIND_AWARE = frozenset({DriftMethod.PSI, DriftMethod.EUCLIDEAN, DriftMethod.COSINE})

MethodOptions = dict[DriftMethod, dict]


def infer_kind(*series: pd.Series) -> ColumnKind:
    """Numeric dtypes are treated as numerical, everything else (strings,
    booleans, categoricals) as categorical. A numeric column that turns out
    to be effectively discrete is handled downstream by the binning
    fallback, so it does not need to be declared here."""
    numeric = all(
        pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) for s in series
    )
    return ColumnKind.NUMERICAL if numeric else ColumnKind.CATEGORICAL


def _encode_for_model(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map categories to shared integer codes so the domain classifier can
    consume them. Codes are ordinal, which is fine for a tree ensemble - it
    can split out individual codes - but would not be for a linear model.

    Cast to str first: a column can be int64 in reference and gain a
    string value in current, and np.unique sorts its input, which raises
    TypeError across mixed dtypes. Only category identity matters here,
    so stringifying first is lossless."""
    reference, current = np.asarray(reference).astype(str), np.asarray(current).astype(str)
    categories, codes = np.unique(np.concatenate([reference, current]), return_inverse=True)
    return codes[: len(reference)].astype(float), codes[len(reference) :].astype(float)


def _run_method(
    method: DriftMethod,
    reference: np.ndarray,
    current: np.ndarray,
    kind: ColumnKind,
    options: MethodOptions,
) -> MethodResult:
    opts = dict(options.get(method, {}))
    if method in _KIND_AWARE:
        opts.setdefault("kind", kind)

    if method is DriftMethod.KS:
        return methods.ks_test(reference, current, **opts)
    if method is DriftMethod.PSI:
        return methods.psi(reference, current, **opts)
    if method is DriftMethod.PAGE_HINKLEY:
        return methods.page_hinkley(reference, current, **opts)
    if method is DriftMethod.EUCLIDEAN:
        return methods.euclidean_distance(reference, current, **opts)
    if method is DriftMethod.COSINE:
        return methods.cosine_distance(reference, current, **opts)
    if method is DriftMethod.MMD:
        return methods.mmd(reference, current, **opts)
    if method is DriftMethod.MODEL_BASED:
        if kind is ColumnKind.CATEGORICAL:
            reference, current = _encode_for_model(reference, current)
        return methods.model_based(reference, current, **opts)
    raise ValueError(f"unknown drift method: {method}")


def detect_column_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    column: str | None = None,
    kind: ColumnKind | None = None,
    methods_to_run: tuple[DriftMethod, ...] = ALL_METHODS,
    vote: str = "majority",
    options: MethodOptions | None = None,
) -> ColumnDriftReport:
    """Run every applicable detector against one column.

    vote='majority' (default) calls the column drifted when more than half
    the applicable detectors agree, which stops a single trigger-happy
    method from carrying the verdict. vote='any' is the sensitive setting:
    one detector is enough."""
    if vote not in {"majority", "any"}:
        raise ValueError("vote must be 'majority' or 'any'")

    reference = reference.dropna()
    current = current.dropna()
    name = column or str(reference.name or current.name or "column")
    kind = kind or infer_kind(reference, current)
    options = options or {}

    ref_values = reference.to_numpy()
    cur_values = current.to_numpy()

    results: list[MethodResult] = []
    skipped: dict[str, str] = {}
    for method in methods_to_run:
        if method in _NUMERIC_ONLY and kind is ColumnKind.CATEGORICAL:
            skipped[method.value] = "not applicable to categorical columns"
            continue
        try:
            results.append(_run_method(method, ref_values, cur_values, kind, options))
        except methods.DriftInputError as exc:
            skipped[method.value] = str(exc)

    n_drifted = sum(r.drifted for r in results)
    if not results:
        drifted = False
    elif vote == "any":
        drifted = n_drifted > 0
    else:
        drifted = n_drifted > len(results) / 2

    return ColumnDriftReport(
        column=name,
        kind=kind,
        n_reference=int(reference.size),
        n_current=int(current.size),
        results=results,
        skipped=skipped,
        drifted=drifted,
        vote=vote,
    )


def detect_dataset_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    share_threshold: float = 0.5,
    methods_to_run: tuple[DriftMethod, ...] = ALL_METHODS,
    vote: str = "majority",
    options: MethodOptions | None = None,
    label: str | None = None,
) -> DatasetDriftReport:
    """Compare every shared column and roll up into a share-of-drifted
    verdict. Columns present in only one of the two frames are skipped: a
    column that was added or removed is a schema change, not drift, and
    belongs in a different signal."""
    in_both = [c for c in reference.columns if c in current.columns]
    # An explicit `columns` is filtered the same way rather than trusted:
    # ml_debugger_tools.run_drift_analysis passes column names the LLM read
    # off the diff, which need not exist in either frame, and indexing a
    # missing one raises KeyError instead of the intended DriftInputError.
    shared = [c for c in columns if c in in_both] if columns is not None else in_both

    reports = [
        detect_column_drift(
            reference[name],
            current[name],
            column=name,
            methods_to_run=methods_to_run,
            vote=vote,
            options=options,
        )
        for name in shared
    ]
    return DatasetDriftReport(columns=reports, share_threshold=share_threshold, label=label)


def format_report(report: DatasetDriftReport) -> str:
    """Human-readable summary for CLI/log output."""
    lines: list[str] = []
    header = f"Drift report{f' for {report.label}' if report.label else ''}"
    lines.append(header)
    lines.append("=" * len(header))
    for column in report.columns:
        flag = "DRIFT" if column.drifted else "ok   "
        lines.append(
            f"\n[{flag}] {column.column} ({column.kind.value}, "
            f"n_ref={column.n_reference}, n_cur={column.n_current})"
        )
        for result in column.results:
            mark = "x" if result.drifted else " "
            p_value = "" if result.p_value is None else f"  p={result.p_value:.4f}"
            lines.append(
                f"    [{mark}] {result.method.value:<13} "
                f"stat={result.statistic:>9.4f}  thr={result.threshold:<8.4g}{p_value}"
            )
        for method, reason in column.skipped.items():
            lines.append(f"    [-] {method:<13} skipped: {reason}")

    lines.append("")
    lines.append(
        f"Share of drifted components: {report.n_drifted}/{report.n_columns} "
        f"= {report.share_drifted:.2%} (threshold {report.share_threshold:.0%})"
    )
    lines.append(f"Dataset drifted: {report.dataset_drifted}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check two datasets for data drift using all supported methods"
    )
    parser.add_argument("reference", help="Baseline CSV/Parquet file")
    parser.add_argument("current", help="Current CSV/Parquet file to compare against baseline")
    parser.add_argument("--share-threshold", type=float, default=0.5)
    parser.add_argument("--vote", choices=["majority", "any"], default="majority")
    parser.add_argument("--json", dest="json_out", help="Also write the full report as JSON here")
    args = parser.parse_args()

    dataset_report = detect_dataset_drift(
        read_table(args.reference),
        read_table(args.current),
        share_threshold=args.share_threshold,
        vote=args.vote,
        label=f"{args.reference} -> {args.current}",
    )
    print(format_report(dataset_report))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(dataset_report.model_dump_json(indent=2))
        print(f"\nJSON report written to {args.json_out}")

    raise SystemExit(1 if dataset_report.dataset_drifted else 0)
