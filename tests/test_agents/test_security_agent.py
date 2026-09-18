from src.agents import security_agent
from src.agents.security_agent import make_security_agent, run_security_agent
from src.schemas.pr import ChangedFile, PRPayload
from src.schemas.review import Severity


def _state(patch: str, filename: str = "src/handler.py") -> dict:
    pr = PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha="head",
        author="octocat",
        title="Test PR",
        changed_files=[ChangedFile(filename=filename, status="modified", patch=patch)],
    )
    return {"pr_payload": pr}


def _hunk(*added_lines: str) -> str:
    n = len(added_lines) + 1
    body = "".join(f"+{line}\n" for line in added_lines)
    return f"@@ -1,1 +1,{n} @@\n context\n{body}"


def test_flags_eval_usage():
    result = run_security_agent(_state(_hunk("result = eval(user_input)")))
    titles = [f.title for f in result["security_findings"]]
    assert "Use of eval()" in titles


def test_flags_pickle_loads():
    result = run_security_agent(_state(_hunk("obj = pickle.loads(payload)")))
    assert result["security_findings"][0].severity == Severity.HIGH


def test_flags_yaml_load_without_safe_loader():
    result = run_security_agent(_state(_hunk("config = yaml.load(f)")))
    titles = [f.title for f in result["security_findings"]]
    assert "yaml.load() without SafeLoader" in titles


def test_does_not_flag_yaml_safe_loader():
    result = run_security_agent(_state(_hunk("config = yaml.load(f, Loader=yaml.SafeLoader)")))
    assert result["security_findings"] == []


def test_flags_shell_true_subprocess():
    result = run_security_agent(_state(_hunk('subprocess.run(cmd, shell=True)')))
    titles = [f.title for f in result["security_findings"]]
    assert "subprocess call with shell=True" in titles


def test_flags_hardcoded_credential():
    result = run_security_agent(_state(_hunk('api_key = "sk-abcdef1234567890"')))
    assert result["security_findings"][0].severity == Severity.CRITICAL


def test_line_number_matches_added_line_position():
    result = run_security_agent(_state(_hunk("safe_line = 1", "os.system(cmd)")))
    finding = result["security_findings"][0]
    assert finding.line_number == 3  # context=1, safe_line=2, os.system=3


def test_clean_diff_produces_no_findings():
    result = run_security_agent(_state(_hunk("x = compute_value()")))
    assert result["security_findings"] == []


def test_ignores_removed_lines():
    patch = "@@ -1,1 +1,1 @@\n-eval(x)\n+safe_call(x)\n"
    result = run_security_agent(_state(patch))
    assert result["security_findings"] == []


def test_merges_llm_findings_when_llm_enabled(monkeypatch):
    monkeypatch.setattr(
        security_agent,
        "call_llm_with_tools",
        lambda system_prompt, user_prompt, tools, tool_implementations: [
            {
                "line_number": 2,
                "title": "SSRF via unsanitized URL",
                "severity": "HIGH",
                "description": "URL is built from request input without validation.",
                "suggestion": "Validate against an allowlist of hosts.",
            }
        ],
    )
    result = run_security_agent(_state(_hunk("requests.get(user_supplied_url)")))
    titles = [f.title for f in result["security_findings"]]
    assert "SSRF via unsanitized URL" in titles


def test_ignores_malformed_llm_finding(monkeypatch):
    monkeypatch.setattr(
        security_agent,
        "call_llm_with_tools",
        lambda system_prompt, user_prompt, tools, tool_implementations: [
            {"title": "missing required fields"}
        ],
    )
    result = run_security_agent(_state(_hunk("x = 1")))
    assert result["security_findings"] == []


def test_llm_disabled_by_default_in_tests():
    # No GROQ_API_KEY configured in the test environment, so call_llm_with_tools
    # returns None internally and only regex findings are produced.
    result = run_security_agent(_state(_hunk("os.system(cmd)")))
    assert len(result["security_findings"]) == 1


def test_make_security_agent_binds_ast_scan_tool_to_repo_path(monkeypatch, tmp_path):
    (tmp_path / "handler.py").write_text("eval(x)\n")

    captured_tool_implementations = {}

    def _fake_call_llm_with_tools(system_prompt, user_prompt, tools, tool_implementations):
        captured_tool_implementations.update(tool_implementations)
        return [{"title": "noop", "severity": "INFO", "description": "n/a"}]

    monkeypatch.setattr(security_agent, "call_llm_with_tools", _fake_call_llm_with_tools)

    agent = make_security_agent(repo_path=tmp_path)
    agent(_state(_hunk("result = eval(user_input)"), filename="handler.py"))

    scan_result = captured_tool_implementations["run_ast_security_scan"]("handler.py")
    assert "error" not in scan_result
    assert scan_result["n_findings"] == 1


def test_does_not_flag_eval_and_exec_as_method_calls():
    r"""The old `\beval\(` matched after a dot, so every model.eval() - the exact
    call ml_debugger hunts for - was reported HIGH and auto-failed the PR."""
    state = _state(_hunk("model.eval()", 'df.eval("a + b")', "self.exec(stmt)"))
    assert run_security_agent(state)["security_findings"] == []


def test_still_flags_bare_eval_and_exec():
    findings = run_security_agent(_state(_hunk("eval(user_input)", "exec(src)")))["security_findings"]
    assert {f.title for f in findings} == {"Use of eval()", "Use of exec()"}
