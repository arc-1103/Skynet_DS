# src/agents/llm_client.py
"""Shared Groq client for the specialist agents' LLM-backed analysis pass.

Every specialist agent (security, architecture, ml_debugger) runs its
deterministic regex checks unconditionally first - those are free, instant,
and what the existing test suite exercises - then, only if GROQ_API_KEY is
configured, adds a second pass where the same diff is handed to Groq's
qwen/qwen3.8-27b for the nuanced issues no regex can catch (data leakage,
SSRF, missing seeds, tight coupling, poor naming, etc.).

If the key is absent, the SDK isn't installed, or the call fails or
returns unusable output for any reason, `call_llm_findings` returns None
and the caller silently falls back to regex-only findings for that file -
the LLM pass is additive, never a hard dependency of the review.

`call_llm_with_tools` is the same contract extended to real function
calling: ml_debugger_agent hands it a set of tools (ml_debugger_tools.py)
the model may invoke - inspecting a dataset's schema, running drift
analysis, explaining drift root cause, checking a doc page for a
deprecated API - and the loop executes whatever the model asks for and
feeds the result back until it settles on a final findings JSON, exactly
like call_llm_findings produces today."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

from src.core.config import settings

if TYPE_CHECKING:
    # Only for type-checking - importing groq's TypedDicts here (instead of
    # a runtime import) keeps groq an optional dependency, same reason
    # _client() below imports the Groq client lazily inside the function
    # body rather than at module load time.
    from groq.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCallParam,
        ChatCompletionToolMessageParam,
        ChatCompletionToolParam,
    )

logger = logging.getLogger(__name__)

# Guards against a model that never stops requesting tool calls.
_MAX_TOOL_ROUNDS = 6


def llm_enabled() -> bool:
    return bool(settings.groq_api_key)


@lru_cache(maxsize=1)
def _client():
    from groq import Groq  # imported lazily - optional dependency when no key is set

    return Groq(api_key=settings.groq_api_key)


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON output in ```json ... ``` even when asked not
    to - strip that before parsing rather than failing the whole pass."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def call_llm_findings(system_prompt: str, user_prompt: str) -> list[dict[str, Any]] | None:
    """Runs one Groq chat completion and returns the parsed `findings` list,
    or None if the pass could not run or produced unusable output. Callers
    treat None exactly like "no additional findings" - never as a reason to
    fail the review."""
    if not llm_enabled():
        return None

    try:
        response = _client().chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="none",
            temperature=0.1,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
    except Exception:  # Groq/network/SDK errors must never take the review down
        logger.warning("Groq call failed; falling back to regex-only findings", exc_info=True)
        return None

    return _parse_findings_payload(content)


def _parse_findings_payload(content: str | None) -> list[dict[str, Any]] | None:
    try:
        parsed = json.loads(_strip_code_fence(content or ""))
    except json.JSONDecodeError:
        logger.warning("Groq returned non-JSON output; discarding: %r", content)
        return None

    findings = parsed.get("findings") if isinstance(parsed, dict) else None
    if not isinstance(findings, list):
        logger.warning("Groq JSON missing a 'findings' list; discarding: %r", parsed)
        return None
    return findings


def _run_tool_call(
    call: Any, tool_implementations: dict[str, Callable[..., Any]]
) -> ChatCompletionToolMessageParam:
    """Executes one requested tool call and packages it as a 'tool' message.
    Never raises - a bad tool name, unparsable arguments, or an exception
    from the tool itself all become an {"error": ...} result handed back to
    the model, which can react to it (try something else, or give up on
    that avenue) instead of crashing the whole review pass.

    No `name` field: Groq/OpenAI's tool-message schema only carries
    `role`/`content`/`tool_call_id` - `name` belongs to the older,
    deprecated `function`-role message, and including it here would send
    the API a field its schema for this role doesn't define."""
    name = call.function.name
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        result: Any = {"error": f"could not parse arguments for tool {name!r}"}
    else:
        implementation = tool_implementations.get(name)
        if implementation is None:
            result = {"error": f"unknown tool {name!r}"}
        else:
            try:
                result = implementation(**arguments)
            except Exception as exc:  # tool implementations must never crash the loop
                logger.warning("tool %r raised", name, exc_info=True)
                result = {"error": f"{name} raised: {exc}"}

    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result, default=str),
    }


def call_llm_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: list[ChatCompletionToolParam],
    tool_implementations: dict[str, Callable[..., Any]],
    *,
    max_rounds: int = _MAX_TOOL_ROUNDS,
) -> list[dict[str, Any]] | None:
    """Runs a Groq tool-calling loop: the model may request any number of
    tool calls (each executed locally via `tool_implementations` and fed
    back as a 'tool' message) across up to `max_rounds` turns, until it
    settles on a final `{"findings": [...]}` response - the same shape
    `call_llm_findings` returns.

    Same fail-open contract as `call_llm_findings`: no API key, an SDK/
    network error anywhere in the loop, a tool-call loop that never
    terminates, or unusable final output all return None, and the caller
    falls back to regex-only findings for that file."""
    if not llm_enabled():
        return None

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        for _ in range(max_rounds):
            response = _client().chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return _parse_findings_payload(message.content)

            tool_call_params: list[ChatCompletionMessageToolCallParam] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_call_params,
                }
            )
            for call in tool_calls:
                messages.append(_run_tool_call(call, tool_implementations))
    except Exception:  # Groq/network/SDK errors must never take the review down
        logger.warning("Groq tool-calling pass failed; falling back to regex-only findings", exc_info=True)
        return None

    logger.warning("Groq tool-call loop exceeded max_rounds=%d without a final answer", max_rounds)
    return None
