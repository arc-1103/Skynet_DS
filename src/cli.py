# src/cli.py
"""CLI entry point (Pull path, Architecture.md section 5).
Run from the repo root: python -m src.cli debug --run-id run-004
"""

import argparse

from src.agents.orchestrator import diagnose
from src.core.config import settings
from src.memory.sqlite_store import SQLiteStore
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.retrieval.evidence import EvidenceRetriever


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlcop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    debug_cmd = subparsers.add_parser("debug", help="Diagnose why a run regressed vs. its baseline")
    debug_cmd.add_argument("--run-id", required=True)

    args = parser.parse_args()

    # Phase 0: fake adapter, no real MLflow/W&B wired yet (BluePrint.md Phase 0).
    adapter = FakeTelemetryAdapter()
    store = SQLiteStore(settings.memory_db_path)
    retriever = EvidenceRetriever(adapter, store)

    if args.command == "debug":
        result = diagnose(args.run_id, retriever, store)
        print(result.explanation)


if __name__ == "__main__":
    main()
