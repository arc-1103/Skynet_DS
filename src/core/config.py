# src/core/config.py
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

_DEFAULT_GROQ_MODEL = "qwen/qwen3-32b"


@dataclass(frozen=True)
class Settings:
    """Process configuration.

    The field defaults are plain values, not `os.environ.get(...)` calls: a
    default evaluated in the class body runs once, at import, which froze
    the configuration to whatever the environment looked like the first
    time anything imported this module, and made a bare `Settings()` in a
    test silently pick up the developer's real .env. Reading the
    environment lives in from_env() instead, so `Settings(...)` builds
    exactly what it is passed and nothing else."""

    # Relative to the working directory (the repo root, /app in Docker) so a
    # test that chdirs to tmp_path gets its own copy.
    memory_db_path: str = "src/memory/memory.db"
    # HMAC secret for verifying GitHub webhook payloads (X-Hub-Signature-256).
    # If unset, signature verification is skipped - fine for local dev, not
    # for anything internet-reachable.
    github_webhook_secret: str | None = None
    # Personal access token (or GitHub App installation token) used to fetch
    # PR metadata/diffs via the GitHub REST API in worker/jobs.py.
    github_token: str | None = None
    # Redis (or any Celery broker) URL. Set -> jobs run on Celery workers
    # (src/worker/celery_app.py) and survive a restart of the API process;
    # unset -> the in-process asyncio queue (src/worker/dispatcher.py).
    celery_broker_url: str | None = None
    # Local directory PRs are cached to after being pulled from GitHub.
    pr_cache_dir: str = "data/prs"
    # Where the monitoring sweep leaves its latest drift report / root cause
    # per model, for the dashboard (demo-ui) to read off the shared volume.
    monitoring_dir: str = "data/monitoring"
    # Bearer token for the read-only /results API the dashboard calls. Unset
    # -> the API is open, like an unset webhook secret: fine for local dev,
    # not for anything internet-reachable (it serves PR findings).
    results_api_token: str | None = None
    # Groq API key for the specialist agents' LLM-backed analysis pass
    # (src/agents/llm_client.py). Unset -> agents silently fall back to
    # their deterministic regex checks only.
    groq_api_key: str | None = None
    groq_model: str = _DEFAULT_GROQ_MODEL
    # Which observability/ adapter the diagnostic core reads runs from:
    # "fake" (canned seeded regression) or "wandb" (Architecture.md section 4).
    telemetry_backend: str = "fake"
    # W&B entity (user/team) that owns the model projects; None -> the API
    # key's default entity. WANDB_API_KEY itself is read by wandb directly.
    wandb_entity: str | None = None
    # Prometheus base URL for live feature distributions (Workflows 2 and 5).
    # Unset -> drift checks and the monitoring sweep are skipped.
    prometheus_url: str | None = None
    # How much recent traffic one drift check compares against the baseline.
    drift_window_seconds: int = 3600
    # Workflow 5 sweep period; 0 disables the always-on monitor.
    monitor_interval_seconds: int = 0
    # Models the sweep watches; empty -> every model in repo_models.
    monitored_models: tuple[str, ...] = ()
    # Repo paths holding the data pipeline - commits touching them are the
    # "pipeline code change" evidence for a drift (Workflow 2 step 4).
    pipeline_paths: tuple[str, ...] = ()
    # LLM-written diagnosis explanations (prompts/explain.md). Off by default:
    # it sends run configs, metrics and commit SHAs to the Groq API, a
    # separate decision from GROQ_API_KEY enabling the PR-review agents.
    llm_explain: bool = False
    # Slack incoming-webhook URL that drift and deploy-regression verdicts
    # are posted to. Unset -> verdicts go to the log only.
    slack_webhook_url: str | None = None
    # "owner/repo" the diagnosed model's code lives in - where code diffs
    # between two runs' commits are fetched from (Workflow 1 step 4).
    github_repo: str | None = None
    # GitHub repo -> model_id, so a deploy webhook knows which model to
    # triage (Workflow 4). Parsed from "owner/a=model-a,owner/b=model-b".
    repo_models: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build from the current environment, loading .env first."""
        load_dotenv()
        return cls(
            memory_db_path=os.environ.get("MLCOP_MEMORY_DB_PATH") or cls.memory_db_path,
            celery_broker_url=os.environ.get("CELERY_BROKER_URL") or None,
            github_webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET"),
            github_token=os.environ.get("GITHUB_TOKEN"),
            results_api_token=os.environ.get("RESULTS_API_TOKEN") or None,
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            # `or` not a get() default: a copied .env with an empty GROQ_MODEL=
            # line sets the variable to "", which get() treats as present.
            groq_model=os.environ.get("GROQ_MODEL") or _DEFAULT_GROQ_MODEL,
            telemetry_backend=os.environ.get("TELEMETRY_BACKEND") or "fake",
            wandb_entity=os.environ.get("WANDB_ENTITY") or None,
            prometheus_url=os.environ.get("PROMETHEUS_URL") or None,
            drift_window_seconds=int(os.environ.get("MLCOP_DRIFT_WINDOW_SECONDS") or 3600),
            monitor_interval_seconds=int(os.environ.get("MLCOP_MONITOR_INTERVAL_SECONDS") or 0),
            monitored_models=_parse_list(os.environ.get("MLCOP_MONITORED_MODELS", "")),
            pipeline_paths=_parse_list(os.environ.get("MLCOP_PIPELINE_PATHS", "")),
            llm_explain=os.environ.get("MLCOP_LLM_EXPLAIN", "").strip().lower() in {"1", "true", "yes"},
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL") or None,
            github_repo=os.environ.get("GITHUB_REPO") or None,
            repo_models=_parse_pairs(os.environ.get("MLCOP_REPO_MODELS", "")),
        )


def _parse_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_pairs(raw: str) -> dict[str, str]:
    pairs = (item.split("=", 1) for item in raw.split(",") if "=" in item)
    return {k.strip(): v.strip() for k, v in pairs if k.strip() and v.strip()}


settings = Settings.from_env()


def get_settings() -> Settings:
    """FastAPI dependency provider - inject via Depends(get_settings)."""
    return settings
