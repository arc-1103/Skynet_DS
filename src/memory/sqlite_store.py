# src/memory/sqlite_store.py
import sqlite3
from pathlib import Path

from src.graph.causal import CauseCandidate
from src.schemas.telemetry import FeatureSample, NormalizedRun

_SCHEMA = """
CREATE TABLE IF NOT EXISTS baselines (
    model_id TEXT PRIMARY KEY,
    run_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    verdict_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS baseline_distributions (
    model_id TEXT PRIMARY KEY,
    sample_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS active_drifts (
    model_id TEXT PRIMARY KEY,
    signature TEXT NOT NULL
);
"""


class SQLiteStore:
    """Local-disk Store implementation (BluePrint.md: SQLite behind a Store
    interface). Swapping to Postgres later means writing a new class against
    the same Protocol - nothing in agents/, graph/, or tools/ should import
    sqlite3 directly."""

    def __init__(self, db_path: str = "src/memory/memory.db") -> None:
        self._path = Path(db_path)
        if db_path != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_baseline(self, model_id: str) -> NormalizedRun | None:
        row = self._conn.execute(
            "SELECT run_json FROM baselines WHERE model_id = ?", (model_id,)
        ).fetchone()
        return NormalizedRun.model_validate_json(row[0]) if row else None

    def set_baseline(self, model_id: str, run: NormalizedRun) -> None:
        self._conn.execute(
            "INSERT INTO baselines (model_id, run_json) VALUES (?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET run_json = excluded.run_json",
            (model_id, run.model_dump_json()),
        )
        self._conn.commit()

    def save_diagnosis(self, model_id: str, verdict: CauseCandidate) -> None:
        self._conn.execute(
            "INSERT INTO diagnoses (model_id, verdict_json) VALUES (?, ?)",
            (model_id, verdict.model_dump_json()),
        )
        self._conn.commit()

    def get_diagnosis_history(self, model_id: str) -> list[CauseCandidate]:
        # order by id, not created_at: CURRENT_TIMESTAMP has only
        # second-resolution, so same-second inserts would otherwise tie.
        rows = self._conn.execute(
            "SELECT verdict_json FROM diagnoses WHERE model_id = ? ORDER BY id DESC",
            (model_id,),
        ).fetchall()
        return [CauseCandidate.model_validate_json(r[0]) for r in rows]

    def get_baseline_distribution(self, model_id: str) -> FeatureSample | None:
        row = self._conn.execute(
            "SELECT sample_json FROM baseline_distributions WHERE model_id = ?", (model_id,)
        ).fetchone()
        return FeatureSample.model_validate_json(row[0]) if row else None

    def set_baseline_distribution(self, model_id: str, sample: FeatureSample) -> None:
        self._conn.execute(
            "INSERT INTO baseline_distributions (model_id, sample_json) VALUES (?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET sample_json = excluded.sample_json",
            (model_id, sample.model_dump_json()),
        )
        self._conn.commit()

    def get_active_drift(self, model_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT signature FROM active_drifts WHERE model_id = ?", (model_id,)
        ).fetchone()
        return row[0] if row else None

    def set_active_drift(self, model_id: str, signature: str | None) -> None:
        if signature is None:
            self._conn.execute("DELETE FROM active_drifts WHERE model_id = ?", (model_id,))
        else:
            self._conn.execute(
                "INSERT INTO active_drifts (model_id, signature) VALUES (?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET signature = excluded.signature",
                (model_id, signature),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
