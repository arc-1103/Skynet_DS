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


def _runs(rows):
    from datetime import datetime, timezone

    from src.schemas.telemetry import NormalizedRun

    return [
        NormalizedRun(
            run_id=f"r{i}",
            model_id="m",
            timestamp=datetime.now(timezone.utc),
            metrics={"acc": acc},
            config=config,
            code_ref=code,
            data_version="d1",
        )
        for i, (acc, config, code) in enumerate(rows)
    ]


def test_rank_correlates_finds_the_variable_that_splits_the_metric():
    from src.graph.causal import rank_correlates

    # batch size cleanly splits acc; lr and code vary but don't track it
    runs = _runs([
        (0.91, {"batch_size": 64, "lr": 0.1}, "c1"),
        (0.92, {"batch_size": 64, "lr": 0.2}, "c2"),
        (0.90, {"batch_size": 64, "lr": 0.1}, "c3"),
        (0.87, {"batch_size": 32, "lr": 0.2}, "c4"),
        (0.88, {"batch_size": 32, "lr": 0.1}, "c5"),
    ])

    ranked = rank_correlates(runs, "acc")

    assert ranked[0].kind == "config"
    assert "`batch_size`" in ranked[0].description
    assert "64 -> mean 0.910 (3 run(s))" in ranked[0].description
    assert all(0.0 <= c.confidence <= 1.0 for c in ranked)
    # code differs in every run: it "explains" all variance by construction,
    # so it must not beat a variable that actually groups runs
    code = next(c for c in ranked if c.kind == "code")
    assert code.confidence < ranked[0].confidence
    # data_version never varies, so it isn't a candidate at all
    assert not any(c.kind == "data" for c in ranked)


def test_rank_correlates_nothing_to_explain():
    from src.graph.causal import rank_correlates

    same = _runs([(0.9, {"lr": 0.1}, "c1"), (0.9, {"lr": 0.2}, "c2")])
    assert rank_correlates(same, "acc") == []
    assert rank_correlates(same[:1], "acc") == []
    assert rank_correlates(same, "missing-metric") == []


def test_rank_without_history_is_unchanged():
    graph = CausalGraph()
    graph.add_candidate(CauseCandidate(kind="code", description="x", confidence=0.3))
    graph.add_candidate(CauseCandidate(kind="config", description="y", confidence=0.8))
    assert [c.kind for c in graph.rank(history=[])] == ["config", "code"]
