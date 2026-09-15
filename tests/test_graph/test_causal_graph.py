import pytest
from pydantic import ValidationError

from src.graph.causal import CauseCandidate, CausalGraph


def test_cause_candidate_confidence_bounds():
    CauseCandidate(kind="config", description="ok", confidence=0.0)
    CauseCandidate(kind="config", description="ok", confidence=1.0)
    with pytest.raises(ValidationError):
        CauseCandidate(kind="config", description="bad", confidence=1.1)
    with pytest.raises(ValidationError):
        CauseCandidate(kind="config", description="bad", confidence=-0.1)


def test_cause_candidate_kind_must_be_literal():
    with pytest.raises(ValidationError):
        CauseCandidate(kind="not-a-kind", description="bad", confidence=0.5)


def test_causal_graph_rank_orders_by_confidence_descending():
    graph = CausalGraph()
    low = CauseCandidate(kind="data", description="low", confidence=0.1)
    high = CauseCandidate(kind="config", description="high", confidence=0.9)
    mid = CauseCandidate(kind="code", description="mid", confidence=0.5)
    graph.add_candidate(low)
    graph.add_candidate(high)
    graph.add_candidate(mid)

    ranked = graph.rank()

    assert [c.description for c in ranked] == ["high", "mid", "low"]


def test_causal_graph_rank_empty_returns_empty_list():
    assert CausalGraph().rank() == []


def test_causal_graph_rank_is_stable_for_ties():
    graph = CausalGraph()
    first = CauseCandidate(kind="config", description="first", confidence=0.5)
    second = CauseCandidate(kind="data", description="second", confidence=0.5)
    graph.add_candidate(first)
    graph.add_candidate(second)

    ranked = graph.rank()

    assert [c.description for c in ranked] == ["first", "second"]
