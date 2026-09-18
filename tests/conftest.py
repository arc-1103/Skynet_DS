import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.agents import llm_client


@pytest.fixture(autouse=True)
def _disable_llm_by_default(monkeypatch):
    """Forces the specialist agents' LLM-backed pass off for every test,
    regardless of whether a real GROQ_API_KEY is set in the developer's
    local .env - the suite must stay deterministic and free of live API
    calls by default. A test that wants to exercise the LLM path mocks
    `call_llm_findings` directly instead of relying on this key."""
    monkeypatch.setattr(llm_client, "settings", replace(llm_client.settings, groq_api_key=None))
