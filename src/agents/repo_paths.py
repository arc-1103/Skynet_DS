# src/agents/repo_paths.py
"""Shared helper for anchoring an LLM-supplied, repo-relative path to a
local PR checkout - used by every specialist whose Groq tool-calling pass
reads real files (ml_debugger_tools.py, security_agent_tools.py). The
model only ever sees paths as they appear in the diff, so without this a
relative path resolves against the server process's cwd instead of the
PR's actual files - the same role repo_path plays for drift_agent.py's
`git show` calls.

It is also the trust boundary for those paths. The model's only input is
the PR diff, which is attacker-controlled text, so a prompt-injected diff
can ask it to scan `/home/user/.ssh/id_rsa` or `../../secrets.csv` - and
the scanners quote source lines back into the transcript. Every path is
therefore confined to the checkout."""

from __future__ import annotations

from pathlib import Path


def resolve_repo_path(base_path: str | Path | None, file_path: str) -> str:
    """Resolve file_path inside base_path's checkout.

    Raises ValueError if the result escapes base_path (via `..`, an
    absolute path, or a symlink pointing out of the tree). llm_client's
    tool-call loop turns that into a tool-result message for the model
    rather than crashing the review."""
    if base_path is None:
        return str(Path(file_path))

    base = Path(base_path).resolve()
    resolved = (base / file_path).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(
            f"refusing to read {file_path!r}: outside the PR checkout at {base}"
        )
    return str(resolved)
