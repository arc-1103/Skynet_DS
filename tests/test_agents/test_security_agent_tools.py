from src.agents import security_agent_tools


def test_scan_secrets_regex_flags_aws_key():
    text = 'aws_key = "AKIAABCDEFGHIJKLMNOP"\n'
    result = security_agent_tools.scan_secrets_regex(text)
    kinds = [m["kind"] for m in result["matches"]]
    assert "AWS access key ID" in kinds


def test_scan_secrets_regex_flags_openai_key():
    text = 'key = "sk-abcdefghijklmnopqrstuvwxyz123456"\n'
    result = security_agent_tools.scan_secrets_regex(text)
    kinds = [m["kind"] for m in result["matches"]]
    assert "OpenAI API key" in kinds


def test_scan_secrets_regex_flags_private_key_block():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n-----END RSA PRIVATE KEY-----\n"
    result = security_agent_tools.scan_secrets_regex(text)
    kinds = [m["kind"] for m in result["matches"]]
    assert "Private key block" in kinds


def test_scan_secrets_regex_flags_jwt():
    text = (
        "token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U'\n"
    )
    result = security_agent_tools.scan_secrets_regex(text)
    kinds = [m["kind"] for m in result["matches"]]
    assert "JSON Web Token" in kinds


def test_scan_secrets_regex_flags_high_entropy_assignment_with_no_vendor_prefix():
    text = 'db_password = "qX7$vL2#mK9pZ4wR8tY1nJ6"\n'
    result = security_agent_tools.scan_secrets_regex(text)
    kinds = [m["kind"] for m in result["matches"]]
    assert "High-entropy value assigned to a secret-like name" in kinds


def test_scan_secrets_regex_does_not_flag_low_entropy_assignment():
    text = 'password = "aaaaaaaaaaaaaaaaaaaa"\n'
    result = security_agent_tools.scan_secrets_regex(text)
    assert result["matches"] == []


def test_scan_secrets_regex_redacts_the_matched_line():
    text = 'aws_key = "AKIAABCDEFGHIJKLMNOP"\n'
    result = security_agent_tools.scan_secrets_regex(text)
    snippet = result["matches"][0]["snippet"]
    assert "AKIAABCDEFGHIJKLMNOP" not in snippet
    assert "redacted" in snippet


def test_scan_secrets_regex_reports_line_numbers():
    text = "x = 1\ny = 2\naws_key = \"AKIAABCDEFGHIJKLMNOP\"\n"
    result = security_agent_tools.scan_secrets_regex(text)
    assert result["matches"][0]["line"] == 3


def test_scan_secrets_regex_clean_text_has_no_matches():
    result = security_agent_tools.scan_secrets_regex("x = compute_value()\n")
    assert result["matches"] == []
    assert result["n_matches"] == 0


def test_scan_secrets_regex_notes_when_detect_secrets_cli_missing(monkeypatch):
    monkeypatch.setattr(security_agent_tools.shutil, "which", lambda name: None)
    result = security_agent_tools.scan_secrets_regex("x = 1\n")
    assert "detect_secrets_note" in result
    assert "not installed" in result["detect_secrets_note"]


def test_scan_secrets_regex_uses_detect_secrets_cli_when_available(monkeypatch):
    monkeypatch.setattr(security_agent_tools.shutil, "which", lambda name: "/usr/bin/detect-secrets")

    class _FakeCompleted:
        stdout = '{"results": {}}'

    monkeypatch.setattr(
        security_agent_tools.subprocess, "run", lambda *a, **k: _FakeCompleted()
    )
    result = security_agent_tools.scan_secrets_regex("x = 1\n")
    assert result["detect_secrets_report"] == {"results": {}}


def test_scan_secrets_regex_notes_when_detect_secrets_cli_fails(monkeypatch):
    monkeypatch.setattr(security_agent_tools.shutil, "which", lambda name: "/usr/bin/detect-secrets")

    def _boom(*a, **k):
        raise TimeoutError("took too long")

    monkeypatch.setattr(security_agent_tools.subprocess, "run", _boom)
    result = security_agent_tools.scan_secrets_regex("x = 1\n")
    assert "detect_secrets_note" in result
    assert "failed" in result["detect_secrets_note"]


def test_run_ast_security_scan_flags_eval(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text("result = eval(user_input)\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    kinds = [f["kind"] for f in result["findings"]]
    assert "Use of eval()" in kinds


def test_run_ast_security_scan_flags_pickle_loads(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text("import pickle\nobj = pickle.loads(payload)\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    kinds = [f["kind"] for f in result["findings"]]
    assert "Untrusted deserialization via pickle" in kinds


def test_run_ast_security_scan_flags_os_system(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text("import os\nos.system(cmd)\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    kinds = [f["kind"] for f in result["findings"]]
    assert "Use of os.system()" in kinds


def test_run_ast_security_scan_flags_subprocess_shell_true(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text("import subprocess\nsubprocess.run(cmd, shell=True)\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    kinds = [f["kind"] for f in result["findings"]]
    assert "subprocess call with shell=True" in kinds


def test_run_ast_security_scan_flags_dynamic_sql_via_fstring(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text('cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")\n')
    result = security_agent_tools.run_ast_security_scan(str(path))
    kinds = [f["kind"] for f in result["findings"]]
    assert "Possible unparameterized SQL query" in kinds


def test_run_ast_security_scan_flags_dynamic_sql_via_concatenation(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text('cursor.execute("SELECT * FROM users WHERE id = " + user_id)\n')
    result = security_agent_tools.run_ast_security_scan(str(path))
    kinds = [f["kind"] for f in result["findings"]]
    assert "Possible unparameterized SQL query" in kinds


def test_run_ast_security_scan_does_not_flag_parameterized_sql(tmp_path):
    path = tmp_path / "handler.py"
    path.write_text('cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))\n')
    result = security_agent_tools.run_ast_security_scan(str(path))
    assert result["findings"] == []


def test_run_ast_security_scan_returns_error_for_missing_file():
    result = security_agent_tools.run_ast_security_scan("does/not/exist.py")
    assert "error" in result


def test_run_ast_security_scan_returns_error_for_invalid_python(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def f(:\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    assert "error" in result


def test_run_ast_security_scan_clean_file_has_no_findings(tmp_path):
    path = tmp_path / "clean.py"
    path.write_text("x = compute_value()\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    assert result["findings"] == []


def test_run_ast_security_scan_notes_when_bandit_cli_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(security_agent_tools.shutil, "which", lambda name: None)
    path = tmp_path / "clean.py"
    path.write_text("x = 1\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    assert "bandit_note" in result
    assert "not installed" in result["bandit_note"]


def test_run_ast_security_scan_uses_bandit_cli_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(security_agent_tools.shutil, "which", lambda name: "/usr/bin/bandit")

    class _FakeCompleted:
        stdout = '{"results": []}'

    monkeypatch.setattr(security_agent_tools.subprocess, "run", lambda *a, **k: _FakeCompleted())
    path = tmp_path / "clean.py"
    path.write_text("x = 1\n")
    result = security_agent_tools.run_ast_security_scan(str(path))
    assert result["bandit_report"] == {"results": []}


def test_tool_schemas_names_match_implementations():
    schema_names = {tool["function"]["name"] for tool in security_agent_tools.TOOL_SCHEMAS}
    assert schema_names == set(security_agent_tools.TOOL_IMPLEMENTATIONS)


def test_make_tool_implementations_with_no_repo_path_returns_bare_tools():
    implementations = security_agent_tools.make_tool_implementations(None)
    assert implementations["scan_secrets_regex"] is security_agent_tools.scan_secrets_regex
    assert implementations["run_ast_security_scan"] is security_agent_tools.run_ast_security_scan


def test_make_tool_implementations_resolves_relative_paths_against_repo_path(tmp_path):
    (tmp_path / "handler.py").write_text("eval(x)\n")
    implementations = security_agent_tools.make_tool_implementations(tmp_path)
    result = implementations["run_ast_security_scan"]("handler.py")
    assert "error" not in result
    assert result["n_findings"] == 1


def test_make_tool_implementations_still_errors_for_a_real_missing_relative_path(tmp_path):
    implementations = security_agent_tools.make_tool_implementations(tmp_path)
    result = implementations["run_ast_security_scan"]("does_not_exist.py")
    assert "error" in result
