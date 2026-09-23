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


def test_deploy_scoped_diagnosis_uses_the_deployed_commit(tmp_path):
    """Workflow 4: code evidence covers baseline..deploy_sha, not the run's ref."""
    store = SQLiteStore(str(tmp_path / "m.db"))
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), store)

    result = diagnose("run-004", retriever, store, deploy_sha="fff999")

    code = next(c for c in result.ranked_candidates if c.kind == "code")
    assert "abc111 -> fff999" in code.description
    assert "abc111..fff999" in code.description


def _compare_retriever(tmp_path):
    from datetime import datetime, timezone

    from src.schemas.telemetry import NormalizedRun

    now = datetime.now(timezone.utc)
    rows = [("a", 64, 0.91), ("b", 32, 0.88), ("c", 64, 0.92), ("d", 32, 0.87)]
    runs = [
        NormalizedRun(run_id=rid, model_id="m", timestamp=now, metrics={"acc": acc, "loss": 1 - acc},
                      config={"batch_size": bs}, code_ref="same")
        for rid, bs, acc in rows
    ]
    store = SQLiteStore(str(tmp_path / "c.db"))
    return EvidenceRetriever(FakeTelemetryAdapter(runs=runs), store), store


def test_compare_explains_the_strongest_correlate_and_persists_it(tmp_path):
    from src.agents.orchestrator import compare

    retriever, store = _compare_retriever(tmp_path)

    result = compare(["a", "b", "c", "d"], retriever, store)

    assert result.metric == "acc"  # defaults to the first run's first metric
    assert result.ranked_candidates[0].kind == "config"
    assert "Strongest correlate" in result.explanation
    assert "`batch_size`" in result.explanation
    assert store.get_diagnosis_history("m")[0] == result.ranked_candidates[0]


def test_compare_rejects_bad_input(tmp_path):
    import pytest

    from src.agents.orchestrator import compare

    retriever, store = _compare_retriever(tmp_path)
    with pytest.raises(ValueError, match="at least 2"):
        compare(["a"], retriever, store)
    with pytest.raises(ValueError, match="not logged on every run"):
        compare(["a", "b"], retriever, store, metric="f1")


def _llm_on(monkeypatch):
    from dataclasses import replace

    from src.agents import orchestrator

    monkeypatch.setattr(orchestrator, "settings", replace(orchestrator.settings, llm_explain=True))
    return orchestrator


def test_llm_explanation_is_used_when_opted_in(tmp_path, monkeypatch):
    orchestrator = _llm_on(monkeypatch)
    prompts = []

    def fake_llm(system, user):
        prompts.append(user)
        return "Run 'run-004' lost 8pts of accuracy; the learning_rate bump is the likely cause."

    monkeypatch.setattr(orchestrator, "call_llm_text", fake_llm)
    store = SQLiteStore(str(tmp_path / "m.db"))

    result = diagnose("run-004", EvidenceRetriever(FakeTelemetryAdapter(), store), store)

    assert result.explanation.startswith("Run 'run-004' lost 8pts")
    # prompts/explain.md was filled with the real evidence, no placeholder left
    assert "learning_rate" in prompts[0] and "{ranked_candidates}" not in prompts[0]


def test_llm_answer_about_the_wrong_run_falls_back_to_the_template(tmp_path, monkeypatch):
    orchestrator = _llm_on(monkeypatch)
    monkeypatch.setattr(orchestrator, "call_llm_text", lambda s, u: "Everything looks fine.")
    store = SQLiteStore(str(tmp_path / "m.db"))

    result = diagnose("run-004", EvidenceRetriever(FakeTelemetryAdapter(), store), store)

    assert "Most likely cause (config" in result.explanation


def test_llm_explanation_is_off_by_default(tmp_path, monkeypatch):
    """Review finding: GROQ_API_KEY was set for the PR agents; it must not
    by itself start shipping run configs and metrics to Groq."""
    import pytest

    from src.agents import orchestrator

    monkeypatch.setattr(orchestrator, "call_llm_text", lambda s, u: pytest.fail("called the LLM"))
    store = SQLiteStore(str(tmp_path / "m.db"))

    result = diagnose("run-004", EvidenceRetriever(FakeTelemetryAdapter(), store), store)

    assert "Most likely cause (config" in result.explanation


def test_a_brace_in_the_prompt_file_does_not_break_diagnose(tmp_path, monkeypatch):
    """Review finding: str.format() raised on any literal "{" in
    prompts/explain.md, crashing every diagnose()."""
    orchestrator = _llm_on(monkeypatch)
    prompt = tmp_path / "explain.md"
    prompt.write_text('Reply like {"cause": "..."}.\nRun: {current_run}\n', encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_EXPLAIN_PROMPT", prompt)
    seen = []
    monkeypatch.setattr(orchestrator, "call_llm_text", lambda s, u: seen.append(u) or "Run 'run-004': lr.")
    store = SQLiteStore(str(tmp_path / "m.db"))

    result = diagnose("run-004", EvidenceRetriever(FakeTelemetryAdapter(), store), store)

    assert result.explanation == "Run 'run-004': lr."
    assert '{"cause": "..."}' in seen[0] and '"run_id":"run-004"' in seen[0]


def test_no_metric_change_means_nothing_to_explain(tmp_path):
    """Review finding: config/code differences were reported as the "most
    likely cause" of a regression even when no metric moved."""
    from datetime import datetime, timezone

    from src.schemas.telemetry import NormalizedRun

    now = datetime.now(timezone.utc)
    base = NormalizedRun(run_id="b", model_id="m", timestamp=now, metrics={"acc": 0.910}, config={"lr": 0.1}, code_ref="aaa1")
    cur = NormalizedRun(run_id="c", model_id="m", timestamp=now, metrics={"acc": 0.912}, config={"lr": 0.3}, code_ref="bbb2")
    store = SQLiteStore(str(tmp_path / "m.db"))
    retriever = EvidenceRetriever(FakeTelemetryAdapter(runs=[base, cur], baselines={"m": "b"}), store)

    result = diagnose("c", retriever, store)

    assert result.ranked_candidates == []
    assert result.explanation == "Run 'c' shows no meaningful delta vs. baseline 'b'."
    assert store.get_diagnosis_history("m") == []


def _drift_inputs():
    """amount shifts by 3 sigma, age is stable -> half the features drift."""
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from src.drift.detector import detect_dataset_drift
    from src.drift.tables import frame_from_features
    from src.schemas.telemetry import FeatureSample

    rng = np.random.default_rng(0)
    now = datetime.now(timezone.utc)
    baseline = FeatureSample(
        model_id="fraud-clf", start=now - timedelta(days=2), end=now - timedelta(days=1),
        features={"amount": rng.normal(0, 1, 300).tolist(), "age": rng.normal(40, 5, 300).tolist()},
    )
    current = FeatureSample(
        model_id="fraud-clf", start=now - timedelta(hours=1), end=now,
        features={"amount": rng.normal(3, 1, 300).tolist(), "age": rng.normal(40, 5, 300).tolist()},
    )
    report = detect_dataset_drift(
        frame_from_features(baseline.features), frame_from_features(current.features), share_threshold=0.4
    )
    return baseline, current, report


def test_drift_with_a_pipeline_commit_blames_the_pipeline(tmp_path, monkeypatch):
    from src.agents.orchestrator import diagnose_drift
    from src.retrieval import evidence

    calls = []

    def fake_commits(repo, paths, since, token):
        calls.append((repo, paths, since))
        return ["abc1234 switch amount to cents"]

    monkeypatch.setattr(evidence, "fetch_path_commits", fake_commits)
    baseline, current, report = _drift_inputs()
    store = SQLiteStore(str(tmp_path / "m.db"))
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), store, "acme/fraud")

    result = diagnose_drift(baseline, current, report, retriever, store, ("etl/",))

    assert calls == [("acme/fraud", ("etl/",), baseline.end)]  # recent baseline: search from its end
    assert result.ranked_candidates[0].kind == "code"
    assert "abc1234 switch amount to cents" in result.explanation
    data = next(c for c in result.ranked_candidates if c.kind == "data")
    assert "Upstream data shift in amount (50% of features)" in data.description
    assert "feature 'amount'" in data.description  # SHAP named the driver
    assert store.get_diagnosis_history("fraud-clf")[0].kind == "code"


def test_pipeline_commits_are_searched_a_bounded_time_before_the_drift(tmp_path, monkeypatch):
    """Review finding: "since the baseline" grows forever, so an old baseline
    ends up blaming every drift on whatever the pipeline last touched."""
    from datetime import timedelta

    from src.agents.orchestrator import PIPELINE_LOOKBACK, diagnose_drift
    from src.retrieval import evidence

    calls = []
    monkeypatch.setattr(evidence, "fetch_path_commits", lambda repo, paths, since, token: calls.append(since) or [])
    baseline, current, report = _drift_inputs()
    old_baseline = baseline.model_copy(update={"end": current.start - timedelta(days=90)})
    store = SQLiteStore(str(tmp_path / "m.db"))
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), store, "acme/fraud")

    diagnose_drift(old_baseline, current, report, retriever, store, ("etl/",))

    assert calls == [current.start - PIPELINE_LOOKBACK]


def test_a_drift_seen_in_two_earlier_episodes_reads_as_seasonal(tmp_path):
    """Each call is one episode - worker/jobs.check_drift guarantees that
    (see test_a_persistent_drift_is_one_episode_diagnosed_and_paged_once)."""
    from src.agents.orchestrator import diagnose_drift

    baseline, current, report = _drift_inputs()
    store = SQLiteStore(str(tmp_path / "m.db"))
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), store)  # no repo: commits unchecked

    first = diagnose_drift(baseline, current, report, retriever, store)
    assert "not checked" in first.explanation
    assert not any("seasonal" in c.description for c in first.ranked_candidates)

    diagnose_drift(baseline, current, report, retriever, store)
    third = diagnose_drift(baseline, current, report, retriever, store)

    assert third.ranked_candidates[0].description.startswith("Expected/seasonal variation")
    assert "2 earlier episodes" in third.ranked_candidates[0].description
