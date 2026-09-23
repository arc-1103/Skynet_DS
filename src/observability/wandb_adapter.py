# src/observability/wandb_adapter.py
"""Weights & Biases TelemetryAdapter - the v1 real adapter (Architecture.md
section 4).

Mapping into schemas/telemetry.py:
- run_id       <- full run path "entity/project/run_id", so any id this
                  adapter hands out can be passed straight back to get_run
- model_id     <- W&B project name
- metrics      <- run.summary_metrics (final value per key), numeric only
- config       <- run.config, minus W&B-internal "_"-prefixed keys and
                  `data_version`
- code_ref     <- run.commit (W&B records it when a run starts in a git repo)
- data_version <- config key `data_version`
- baseline     <- newest finished run tagged `mlcop-baseline`

This is the only module allowed to import wandb."""

from datetime import datetime, timezone
from itertools import islice
from typing import Any

from wandb.apis.public import Api
from wandb.errors import CommError

from src.schemas.telemetry import NormalizedRun

BASELINE_TAG = "mlcop-baseline"
_DATA_VERSION_KEY = "data_version"
_NEWEST_FIRST = "-created_at"


def _timestamp(created_at: str) -> datetime:
    parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class WandbTelemetryAdapter:
    def __init__(self, entity: str | None = None, api: Any = None) -> None:
        # api is injectable for tests; the real one reads WANDB_API_KEY.
        self._api = api if api is not None else Api()
        self._entity = entity

    def _path(self, *parts: str) -> str:
        return "/".join((self._entity, *parts) if self._entity else parts)

    def _normalize(self, run: Any) -> NormalizedRun:
        config = {k: v for k, v in dict(run.config).items() if not k.startswith("_")}
        data_version = config.pop(_DATA_VERSION_KEY, None)
        return NormalizedRun(
            run_id=f"{run.entity}/{run.project}/{run.id}",
            model_id=run.project,
            timestamp=_timestamp(run.created_at),
            metrics={
                k: float(v)
                for k, v in dict(run.summary_metrics).items()
                if not k.startswith("_") and _is_number(v)
            },
            config=config,
            code_ref=getattr(run, "commit", None) or None,
            data_version=str(data_version) if data_version is not None else None,
        )

    def _runs(self, model_id: str, limit: int, filters: dict | None = None) -> list[NormalizedRun]:
        try:
            runs = self._api.runs(self._path(model_id), filters=filters, order=_NEWEST_FIRST, per_page=limit)
            return [self._normalize(r) for r in islice(runs, limit)]
        except (CommError, ValueError):  # W&B's answer for a project that doesn't exist
            return []

    def get_run(self, run_id: str) -> NormalizedRun:
        # "project/id" is completed with the configured entity; a full
        # "entity/project/id" path is used as-is.
        path = run_id if run_id.count("/") >= 2 else self._path(run_id)
        try:
            run = self._api.run(path)
        except (CommError, ValueError) as exc:
            # KeyError to match FakeTelemetryAdapter - callers shouldn't need
            # to know W&B's own exception types.
            raise KeyError(f"No such run: {run_id}") from exc
        return self._normalize(run)

    def get_recent_runs(self, model_id: str, limit: int = 5) -> list[NormalizedRun]:
        return self._runs(model_id, limit)

    def get_baseline(self, model_id: str) -> NormalizedRun | None:
        runs = self._runs(model_id, 1, {"tags": {"$in": [BASELINE_TAG]}, "state": "finished"})
        return runs[0] if runs else None
