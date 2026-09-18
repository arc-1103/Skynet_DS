# src/core/config.py
import os
from dataclasses import dataclass

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

    memory_db_path: str = "memory.db"
    # HMAC secret for verifying GitHub webhook payloads (X-Hub-Signature-256).
    # If unset, signature verification is skipped - fine for local dev, not
    # for anything internet-reachable.
    github_webhook_secret: str | None = None
    # Personal access token (or GitHub App installation token) used to fetch
    # PR metadata/diffs via the GitHub REST API in worker/jobs.py.
    github_token: str | None = None
    # Local directory PRs are cached to after being pulled from GitHub.
    pr_cache_dir: str = "data/prs"
    # Groq API key for the specialist agents' LLM-backed analysis pass
    # (src/agents/llm_client.py). Unset -> agents silently fall back to
    # their deterministic regex checks only.
    groq_api_key: str | None = None
    groq_model: str = _DEFAULT_GROQ_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        """Build from the current environment, loading .env first."""
        load_dotenv()
        return cls(
            github_webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET"),
            github_token=os.environ.get("GITHUB_TOKEN"),
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            # `or` not a get() default: a copied .env with an empty GROQ_MODEL=
            # line sets the variable to "", which get() treats as present.
            groq_model=os.environ.get("GROQ_MODEL") or _DEFAULT_GROQ_MODEL,
        )


settings = Settings.from_env()


def get_settings() -> Settings:
    """FastAPI dependency provider - inject via Depends(get_settings)."""
    return settings
