# src/memory/sqlite_store.py
import sqlite3
from pathlib import Path

from src.graph.causal import CauseCandidate
from src.schemas.telemetry import NormalizedRun

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
"""


class SQLiteStore:
    """Local-disk Store implementation (BluePrint.md: SQLite behind a Store
    interface). Swapping to Postgres later means writing a new class against
    the same Protocol - nothing in agents/, graph/, or tools/ should import
    sqlite3 directly."""

    def __init__(self, db_path: str = "memory.db") -> None:
        self._path = Path(db_path)
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

    def close(self) -> None:
        self._conn.close()
