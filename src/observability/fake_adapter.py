# src/observability/fake_adapter.py
from datetime import datetime, timedelta, timezone

from src.schemas.telemetry import NormalizedRun


class FakeTelemetryAdapter:
    """Canned TelemetryAdapter with a seeded regression (config change causing
    an accuracy drop). Lets Phase 0 build and exercise the diagnostic core
    end-to-end before a real MLflow/W&B adapter exists (BluePrint.md Phase 0).

    Pass `runs` + `baselines` (model_id -> baseline run_id) to serve any
    other canned set instead - the labeled-incident eval (eval/) does."""

    def __init__(
        self,
        runs: list[NormalizedRun] | None = None,
        baselines: dict[str, str] | None = None,
    ) -> None:
        if runs is not None:
            self._runs = {r.run_id: r for r in runs}
            self._baseline_by_model = dict(baselines or {})
            return
        now = datetime.now(timezone.utc)
        self._runs: dict[str, NormalizedRun] = {
            "run-baseline": NormalizedRun(
                run_id="run-baseline",
                model_id="fraud-clf",
                metrics={"eval_accuracy": 0.91, "eval_loss": 0.24},
                config={"learning_rate": 3e-4, "batch_size": 64},
                code_ref="abc111",
                data_version="data-v3",
                timestamp=now - timedelta(days=3),
            ),
            "run-004": NormalizedRun(
                run_id="run-004",
                model_id="fraud-clf",
                metrics={"eval_accuracy": 0.83, "eval_loss": 0.41},
                config={"learning_rate": 1e-3, "batch_size": 64},
                code_ref="abc123",
                data_version="data-v3",
                timestamp=now,
            ),
        }
        self._baseline_by_model: dict[str, str] = {"fraud-clf": "run-baseline"}

    def get_run(self, run_id: str) -> NormalizedRun:
        if run_id not in self._runs:
            raise KeyError(f"No such run: {run_id}")
        return self._runs[run_id]

    def get_recent_runs(self, model_id: str, limit: int = 5) -> list[NormalizedRun]:
        runs = [r for r in self._runs.values() if r.model_id == model_id]
        return sorted(runs, key=lambda r: r.timestamp, reverse=True)[:limit]

    def get_baseline(self, model_id: str) -> NormalizedRun | None:
        run_id = self._baseline_by_model.get(model_id)
        return self._runs.get(run_id) if run_id else None
