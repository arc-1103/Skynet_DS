"""Langfuse client for the whole app. Import `langfuse` / `propagate_attributes`
from here, never from the SDK directly: this is where the masking client is
registered, after src.core.config has loaded .env (the SDK reads LANGFUSE_*
once, at client construction)."""

from __future__ import annotations

import re
from typing import Any

from langfuse import Langfuse, propagate_attributes

from src.core import config as _config  # noqa: F401  (loads .env before Langfuse() below)

__all__ = ["langfuse", "propagate_attributes"]

_FINE_GRAINED_PAT = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")


def _mask(*, data: Any, **_: Any) -> Any:
    """PR diffs and tool results go into trace input/output, and the security
    agent exists precisely because diffs contain leaked secrets - redact the
    same shapes it detects before anything leaves the process."""
    if isinstance(data, str):
        # lazy: security_agent_tools -> llm_client -> this module
        from src.agents.security_agent_tools import _SECRET_ASSIGNMENT, _SECRET_PATTERNS

        for label, pattern in _SECRET_PATTERNS:
            data = pattern.sub(f"[REDACTED {label}]", data)
        data = _FINE_GRAINED_PAT.sub("[REDACTED GitHub fine-grained token]", data)
        return _SECRET_ASSIGNMENT.sub(lambda m: m.group(0).replace(m.group(2), "[REDACTED]"), data)
    if isinstance(data, dict):
        return {k: _mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(data=v) for v in data]
    return data


langfuse = Langfuse(mask=_mask)
