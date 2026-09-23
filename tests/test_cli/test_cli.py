import sys

from src import cli


def test_debug_command_prints_explanation(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # so memory.db lands in tmp_path, not the repo
    monkeypatch.setattr(sys, "argv", ["mlcop", "debug", "--run-id", "run-004"])

    cli.main()

    captured = capsys.readouterr()
    assert "Most likely cause" in captured.out
    assert "run-004" in captured.out


def test_debug_command_missing_run_id_exits_nonzero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mlcop", "debug"])
    try:
        cli.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code != 0


def test_compare_command_prints_comparison(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mlcop", "compare", "--runs", "run-baseline,run-004"])

    cli.main()

    out = capsys.readouterr().out
    assert "across 2 runs" in out
    assert "Strongest correlate" in out


def test_compare_command_with_one_run_exits_nonzero(tmp_path, monkeypatch):
    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mlcop", "compare", "--runs", "run-004"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0


def test_drift_command_without_prometheus_says_so(tmp_path, monkeypatch, capsys):
    from dataclasses import replace

    from src.worker import jobs

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "settings", replace(jobs.settings, prometheus_url=None))
    monkeypatch.setattr(sys, "argv", ["mlcop", "drift", "--model-id", "fraud-clf"])

    cli.main()

    assert "No drift diagnosed for 'fraud-clf'" in capsys.readouterr().out
