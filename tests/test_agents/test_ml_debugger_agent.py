from src.agents import ml_debugger_agent
from src.agents.ml_debugger_agent import make_ml_debugger_agent, run_ml_debugger_agent
from src.schemas.pr import ChangedFile, PRPayload


def _state(patch: str, filename: str = "train.py") -> dict:
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


def test_flags_iterrows():
    result = run_ml_debugger_agent(_state(_hunk("for i, row in df.iterrows():")))
    titles = [f.title for f in result["ml_debugger_findings"]]
    assert "Row-wise DataFrame iteration via .iterrows()" in titles


def test_flags_eval_without_no_grad():
    patch = _hunk("model.eval()", "output = model(batch)")
    result = run_ml_debugger_agent(_state(patch))
    titles = [f.title for f in result["ml_debugger_findings"]]
    assert "Inference call without torch.no_grad()" in titles


def test_does_not_flag_eval_with_no_grad_present():
    patch = _hunk("model.eval()", "with torch.no_grad():", "    output = model(batch)")
    result = run_ml_debugger_agent(_state(patch))
    titles = [f.title for f in result["ml_debugger_findings"]]
    assert "Inference call without torch.no_grad()" not in titles


def test_flags_fit_inside_loop():
    patch = _hunk("for x, y in samples:", "    model.fit(x, y)")
    result = run_ml_debugger_agent(_state(patch))
    titles = [f.title for f in result["ml_debugger_findings"]]
    assert "Model .fit() called inside a loop" in titles


def test_does_not_flag_fit_without_loop():
    patch = _hunk("model.fit(X_train, y_train)")
    result = run_ml_debugger_agent(_state(patch))
    titles = [f.title for f in result["ml_debugger_findings"]]
    assert "Model .fit() called inside a loop" not in titles


def test_clean_diff_produces_no_findings():
    result = run_ml_debugger_agent(_state(_hunk("x = df['col'].sum()")))
    assert result["ml_debugger_findings"] == []


def test_merges_llm_findings_when_llm_enabled(monkeypatch):
    monkeypatch.setattr(
        ml_debugger_agent,
        "call_llm_with_tools",
        lambda system_prompt, user_prompt, tools, tool_implementations: [
            {
                "line_number": 2,
                "title": "Scaler fit before train/test split",
                "severity": "HIGH",
                "description": "Fitting the scaler on the full dataset leaks test-set statistics into training.",
                "suggestion": "Fit the scaler only on the training split.",
            }
        ],
    )
    result = run_ml_debugger_agent(_state(_hunk("scaler.fit(full_df)")))
    titles = [f.title for f in result["ml_debugger_findings"]]
    assert "Scaler fit before train/test split" in titles


def test_make_ml_debugger_agent_binds_dataset_tools_to_repo_path(monkeypatch, tmp_path):
    (tmp_path / "train.csv").write_text("x,y\n1,2\n3,4\n")

    captured_tool_implementations = {}

    def _fake_call_llm_with_tools(system_prompt, user_prompt, tools, tool_implementations):
        captured_tool_implementations.update(tool_implementations)
        return [{"title": "noop", "severity": "INFO", "description": "n/a"}]

    monkeypatch.setattr(ml_debugger_agent, "call_llm_with_tools", _fake_call_llm_with_tools)

    agent = make_ml_debugger_agent(repo_path=tmp_path)
    agent(_state(_hunk("df = pd.read_csv('train.csv')")))

    schema_result = captured_tool_implementations["inspect_dataset_schema"]("train.csv")
    assert "error" not in schema_result
    assert schema_result["n_rows"] == 2
