"""PRD.md section 6 success metrics as a regression gate: if a change to
the hypothesis heuristics or graph/ ranking drops top-1 precision on the
labeled incidents below the v1 bar, this fails."""

from evals.run_eval import PRECISION_BAR, TIME_BAR_SECONDS, run_eval


def test_labeled_incidents_meet_the_v1_bars():
    report = run_eval()
    assert report["gate_precision"] >= PRECISION_BAR, report["results"]
    assert report["max_seconds"] < TIME_BAR_SECONDS


def _incident_file(tmp_path):
    import shutil

    from evals.run_eval import INCIDENTS

    path = tmp_path / "incidents.json"
    shutil.copy(INCIDENTS, path)
    return path


def test_add_incident_freezes_the_run_and_its_baseline(tmp_path):
    import json

    from evals.add_incident import add_incident
    from src.observability.fake_adapter import FakeTelemetryAdapter

    path = _incident_file(tmp_path)

    add_incident(FakeTelemetryAdapter(), "run-004", "config", note="lr bump", path=path)

    added = json.loads(path.read_text(encoding="utf-8"))[-1]
    assert (added["id"], added["source"], added["truth"], added["note"]) == ("run-004", "real", "config", "lr bump")
    assert added["baseline"]["run_id"] == "run-baseline"
    assert added["current"]["config"]["learning_rate"] == 0.001
    assert "timestamp" not in added["current"]
    # and it scores like any other case
    report = run_eval(path)
    assert report["by_source"]["real"] == 1.0


def test_add_incident_rejects_bad_input(tmp_path):
    import pytest

    from evals.add_incident import add_incident
    from src.observability.fake_adapter import FakeTelemetryAdapter

    path = _incident_file(tmp_path)
    adapter = FakeTelemetryAdapter()
    with pytest.raises(ValueError, match="truth must be one of"):
        add_incident(adapter, "run-004", "vibes", path=path)
    with pytest.raises(ValueError, match="already in"):
        add_incident(adapter, "run-004", "config", incident_id="lr-bump", path=path)
    with pytest.raises(KeyError):
        add_incident(adapter, "no-such-run", "config", path=path)


def test_gate_moves_to_real_incidents_once_there_are_enough(tmp_path):
    import json

    from evals import run_eval as eval_module

    path = _incident_file(tmp_path)
    incidents = json.loads(path.read_text(encoding="utf-8"))
    # real incidents the heuristic gets wrong: code is the truth, config changed too
    miss = next(i for i in incidents if i["id"] == "logging-noise-hides-code")
    real = [{**miss, "id": f"real-{n}", "source": "real"} for n in range(eval_module.MIN_REAL_INCIDENTS)]

    path.write_text(json.dumps(incidents + real[:-1]), encoding="utf-8")
    below = eval_module.run_eval(path)
    assert below["gate_precision"] == below["precision"]  # not enough real ones yet

    path.write_text(json.dumps(incidents + real), encoding="utf-8")
    report = eval_module.run_eval(path)
    assert report["by_source"] == {"synthetic": 0.8, "real": 0.0}
    assert report["gate_precision"] == 0.0  # synthetic hits no longer paper over real misses


def test_eval_never_calls_the_llm_and_restores_the_setting(monkeypatch):
    from dataclasses import replace

    import pytest

    from src.agents import orchestrator

    opted_in = replace(orchestrator.settings, llm_explain=True)
    monkeypatch.setattr(orchestrator, "settings", opted_in)
    monkeypatch.setattr(orchestrator, "call_llm_text", lambda s, u: pytest.fail("eval called the LLM"))

    run_eval()

    assert orchestrator.settings is opted_in
