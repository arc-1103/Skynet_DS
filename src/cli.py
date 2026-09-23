# src/cli.py
"""CLI entry point (Pull path, Architecture.md section 5).
Run from the repo root:
    python -m src.cli debug --run-id run-004
    python -m src.cli compare --runs A,B,C [--metric eval_accuracy]
    python -m src.cli drift --model-id fraud-clf
The telemetry backend comes from TELEMETRY_BACKEND (fake | wandb).
"""

import argparse

from src.agents.orchestrator import compare, diagnose
from src.core.config import settings
from src.memory.sqlite_store import SQLiteStore
from src.observability.base import build_adapter
from src.retrieval.evidence import EvidenceRetriever


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlcop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    debug_cmd = subparsers.add_parser("debug", help="Diagnose why a run regressed vs. its baseline")
    debug_cmd.add_argument("--run-id", required=True)

    compare_cmd = subparsers.add_parser("compare", help="Explain what drives a metric across 2+ runs")
    compare_cmd.add_argument("--runs", required=True, help="comma-separated run ids")
    compare_cmd.add_argument("--metric", help="metric to explain (default: first metric of the first run)")

    drift_cmd = subparsers.add_parser("drift", help="Check a model's live features for drift (needs PROMETHEUS_URL)")
    drift_cmd.add_argument("--model-id", required=True)

    args = parser.parse_args()

    if args.command == "drift":
        # lazy: the worker module pulls in the whole PR-review graph
        from src.worker.jobs import check_drift

        print(check_drift(args.model_id) or f"No drift diagnosed for '{args.model_id}' (see log for why).")
        return

    adapter = build_adapter(settings.telemetry_backend, settings.wandb_entity)
    store = SQLiteStore(settings.memory_db_path)
    retriever = EvidenceRetriever(adapter, store, settings.github_repo, settings.github_token)

    if args.command == "debug":
        result = diagnose(args.run_id, retriever, store)
    else:
        run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
        try:
            result = compare(run_ids, retriever, store, args.metric)
        except ValueError as exc:
            parser.error(str(exc))
    print(result.explanation)


if __name__ == "__main__":
    main()
