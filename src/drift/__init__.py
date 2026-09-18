# src/drift/__init__.py
from .methods import (
    DriftInputError,
    cosine_distance,
    euclidean_distance,
    ks_test,
    mmd,
    model_based,
    page_hinkley,
    page_hinkley_stream,
    psi,
)
from .detector import (
    ALL_METHODS,
    MethodOptions,
    detect_column_drift,
    detect_dataset_drift,
    format_report,
    infer_kind,
)
from .pr_drift import check_pr_drift, data_files_in_pr, load_at_ref
from .root_cause import (
    RootCauseInputError,
    explain_drift,
    format_root_cause,
    to_cause_candidates,
)

__all__ = [
    # Detector orchestration (drift/detector.py)
    "infer_kind",
    "detect_column_drift",
    "detect_dataset_drift",
    "format_report",
    "ALL_METHODS",
    "MethodOptions",
    # Individual statistical methods (drift/methods.py)
    "DriftInputError",
    "ks_test",
    "psi",
    "page_hinkley",
    "page_hinkley_stream",
    "euclidean_distance",
    "cosine_distance",
    "model_based",
    "mmd",
    # PR-level drift (drift/pr_drift.py)
    "data_files_in_pr",
    "load_at_ref",
    "check_pr_drift",
    # SHAP root-cause attribution (drift/root_cause.py)
    "RootCauseInputError",
    "explain_drift",
    "to_cause_candidates",
    "format_root_cause",
]
