# evals/run_eval.py
"""Labeled-incident eval (PRD.md section 6, BluePrint.md Phase 1 gate).

Runs every incident in incidents.json through the same diagnose() the CLI
and webhook use, and reports:
- top-1 precision: share of incidents whose top-ranked cause kind matches
  the hand label (v1 bar: >= 60%)
- time-to-root-cause: slowest single diagnosis (v1 bar: < 2 minutes)

Run from the repo root: python -m evals.run_eval
Add a real incident:     python -m evals.add_incident --run-id <id> --truth <kind>

Precision is reported per `source`. The "synthetic" cases were written by
the same team that wrote the heuristic, so they are a sanity check, not
evidence. Once there are MIN_REAL_INCIDENTS "real" ones (captured from
actual regressions with add_incident), the precision bar gates on those
alone."""

import json
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agents import orchestrator
from src.agents.orchestrator import diagnose
from src.memory.sqlite_store import SQLiteStore
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.retrieval.evidence import EvidenceRetriever
from src.schemas.telemetry import NormalizedRun

INCIDENTS = Path(__file__).with_name("incidents.json")
PRECISION_BAR = 0.6
TIME_BAR_SECONDS = 120.0
# Below this many real incidents one miss swings precision by 20+ points,
# so the gate stays on the whole set until then.
MIN_REAL_INCIDENTS = 5


def run_eval(path: Path = INCIDENTS) -> dict:
    # What's scored is the ranking; an LLM explanation would only add paid,
    # slow, nondeterministic network calls. Forced off even if
    # MLCOP_LLM_EXPLAIN is set, and restored since tests call this in-process.
    saved = orchestrator.settings
    orchestrator.settings = replace(saved, llm_explain=False)
    try:
        return _score(path)
    finally:
        orchestrator.settings = saved


def _score(path: Path) -> dict:
    now = datetime.now(timezone.utc)
    results = []
    for incident in json.loads(path.read_text(encoding="utf-8")):
        baseline = NormalizedRun(timestamp=now - timedelta(days=1), **incident["baseline"])
        current = NormalizedRun(timestamp=now, **incident["current"])
        adapter = FakeTelemetryAdapter(runs=[baseline, current], baselines={baseline.model_id: baseline.run_id})
        # fresh in-memory store per incident: no history leaks between cases
        store = SQLiteStore(":memory:")
        started = time.perf_counter()
        ranked = diagnose(current.run_id, EvidenceRetriever(adapter, store), store).ranked_candidates
        elapsed = time.perf_counter() - started
        predicted = ranked[0].kind if ranked else None
        results.append(
            {
                "id": incident["id"],
                "source": incident.get("source", "synthetic"),
                "truth": incident["truth"],
                "predicted": predicted,
                "seconds": elapsed,
            }
        )

    by_source = {
        source: _precision([r for r in results if r["source"] == source])
        for source in dict.fromkeys(r["source"] for r in results)
    }
    n_real = sum(r["source"] == "real" for r in results)
    return {
        "precision": _precision(results),
        "by_source": by_source,
        "n_real": n_real,
        "gate_precision": by_source["real"] if n_real >= MIN_REAL_INCIDENTS else _precision(results),
        "max_seconds": max(r["seconds"] for r in results),
        "results": results,
    }


def _precision(results: list[dict]) -> float:
    return sum(r["truth"] == r["predicted"] for r in results) / len(results)


def main() -> None:
    report = run_eval()
    for r in report["results"]:
        mark = "ok  " if r["truth"] == r["predicted"] else "MISS"
        print(f"{mark} [{r['source']:<9}] {r['id']:<26} truth={r['truth']:<7} predicted={r['predicted']}")
    print()
    for source, precision in report["by_source"].items():
        n = sum(r["source"] == source for r in report["results"])
        print(f"top-1 precision, {source}: {precision:.0%} over {n} incident(s)")
    if report["n_real"] >= MIN_REAL_INCIDENTS:
        gated_on = "real incidents only"
    else:
        gated_on = f"all incidents - {MIN_REAL_INCIDENTS - report['n_real']} more real one(s) until it gates on real only"
    print(
        f"gate: {report['gate_precision']:.0%} (bar {PRECISION_BAR:.0%}, on {gated_on}); "
        f"slowest diagnosis {report['max_seconds']:.3f}s (bar {TIME_BAR_SECONDS:.0f}s)"
    )


if __name__ == "__main__":
    main()
