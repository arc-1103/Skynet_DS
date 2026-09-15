# src/core/config.py
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    memory_db_path: str = "memory.db"
    # HMAC secret for verifying GitHub webhook payloads (X-Hub-Signature-256).
    # If unset, signature verification is skipped - fine for local dev, not
    # for anything internet-reachable.
    github_webhook_secret: str | None = os.environ.get("GITHUB_WEBHOOK_SECRET")


settings = Settings()


def get_settings() -> Settings:
    """FastAPI dependency provider - inject via Depends(get_settings)."""
    return settings
