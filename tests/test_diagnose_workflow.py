from src.agents.orchestrator import diagnose
from src.memory.sqlite_store import SQLiteStore
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.retrieval.evidence import EvidenceRetriever


def test_workflow1_ranks_config_change_as_top_cause(tmp_path):
    adapter = FakeTelemetryAdapter()
    store = SQLiteStore(str(tmp_path / "test_memory.db"))
    retriever = EvidenceRetriever(adapter, store)

    result = diagnose("run-004", retriever, store)

    assert result.baseline is not None
    assert result.ranked_candidates, "expected at least one causal candidate"
    assert result.ranked_candidates[0].kind == "config"
    assert "learning_rate" in result.ranked_candidates[0].description
    assert "Most likely cause" in result.explanation

    history = store.get_diagnosis_history("fraud-clf")
    assert len(history) == 1
    assert history[0].kind == "config"


def test_unknown_run_raises_key_error(tmp_path):
    adapter = FakeTelemetryAdapter()
    store = SQLiteStore(str(tmp_path / "test_memory2.db"))
    retriever = EvidenceRetriever(adapter, store)

    try:
        diagnose("does-not-exist", retriever, store)
        assert False, "expected KeyError"
    except KeyError:
        pass
