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
