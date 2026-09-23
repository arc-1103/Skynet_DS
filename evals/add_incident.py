# evals/add_incident.py
"""Capture a real regression into incidents.json, labeled with what
actually caused it - how the eval set grows from things that really broke
instead of cases written to fit the heuristic (PRD.md section 6).

Run from the repo root once the root cause is known (post-mortem, fix PR):
    python -m evals.add_incident --run-id acme/fraud-clf/abc123 --truth data \\
        --note "vendor feed switched amount to cents"

The run and its baseline are read from the configured TELEMETRY_BACKEND
and frozen into the file, so the case keeps working after the runs age
out of W&B. Label from the post-mortem, not from what mlcop said - a label
copied from the tool's own answer measures nothing."""

import argparse
import json
from pathlib import Path
from typing import get_args

from src.core.config import settings
from src.graph.causal import Kind
from src.memory.sqlite_store import SQLiteStore
from src.observability.base import TelemetryAdapter, build_adapter
from src.retrieval.evidence import EvidenceRetriever

from evals.run_eval import INCIDENTS


def add_incident(
    adapter: TelemetryAdapter,
    run_id: str,
    truth: str,
    *,
    baseline_run_id: str | None = None,
    note: str = "",
    incident_id: str | None = None,
    path: Path = INCIDENTS,
) -> dict:
    if truth not in get_args(Kind):
        raise ValueError(f"truth must be one of {get_args(Kind)}, got {truth!r}")
    incidents = json.loads(path.read_text(encoding="utf-8"))
    incident_id = incident_id or run_id
    if any(i["id"] == incident_id for i in incidents):
        raise ValueError(f"incident {incident_id!r} is already in {path.name}")

    # in-memory store: read the backend's tagged baseline, not whatever a
    # local memory.db happens to have cached
    retriever = EvidenceRetriever(adapter, SQLiteStore(":memory:"))
    current = retriever.get_run(run_id)
    # --baseline-run-id when the post-mortem found the tagged baseline was
    # the wrong reference
    baseline = retriever.get_run(baseline_run_id) if baseline_run_id else retriever.get_baseline(current.model_id)
    if baseline is None:
        raise ValueError(f"no baseline for model {current.model_id!r} - pass --baseline-run-id")

    incident = {
        "id": incident_id,
        "source": "real",
        "truth": truth,
        "note": note,
        # timestamps are left out: run_eval assigns baseline-before-current
        "baseline": baseline.model_dump(mode="json", exclude={"timestamp"}),
        "current": current.model_dump(mode="json", exclude={"timestamp"}),
    }
    path.write_text(json.dumps([*incidents, incident], indent=2) + "\n", encoding="utf-8")
    return incident


def main() -> None:
    parser = argparse.ArgumentParser(prog="add_incident", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True, help="the regressed run")
    parser.add_argument("--truth", required=True, choices=get_args(Kind), help="the cause the post-mortem found")
    parser.add_argument("--baseline-run-id", help="override the model's tagged baseline")
    parser.add_argument("--note", default="", help="one line: what actually broke")
    parser.add_argument("--id", dest="incident_id", help="incident id (default: the run id)")
    args = parser.parse_args()

    try:
        incident = add_incident(
            build_adapter(settings.telemetry_backend, settings.wandb_entity),
            args.run_id,
            args.truth,
            baseline_run_id=args.baseline_run_id,
            note=args.note,
            incident_id=args.incident_id,
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"added {incident['id']!r} ({incident['truth']}) - run `python -m evals.run_eval` to score it")


if __name__ == "__main__":
    main()
