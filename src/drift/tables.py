# src/drift/tables.py
"""One tabular reader for the whole project.

This was reimplemented four times - drift/detector.py, drift/root_cause.py,
drift/pr_drift.py and agents/ml_debugger_tools.py - and the copies disagreed
about the interesting case: the two drift CLIs silently fed an unknown
suffix to read_csv, while ml_debugger_tools raised ValueError for the same
input. Raising is the behaviour worth keeping, since a .json or .xlsx read
as CSV produces a garbage DataFrame rather than an error, and the drift
detectors downstream will happily report drift on it."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pandas as pd

SUPPORTED_SUFFIXES = (".csv", ".parquet")


def read_table(source: str | Path | IO[Any], *, suffix: str | None = None) -> pd.DataFrame:
    """Read a CSV or Parquet table.

    `source` is a path, or an already-open binary buffer - pr_drift reads
    blobs straight out of `git show`, which never touch the filesystem - in
    which case `suffix` says which format it holds."""
    if suffix is None:
        if not isinstance(source, (str, Path)):
            raise ValueError("suffix is required when reading from a buffer")
        suffix = Path(source).suffix

    suffix = suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"unsupported file type {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")
