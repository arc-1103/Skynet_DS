from src.agents import architecture_agent
from src.agents.architecture_agent import run_architecture_agent
from src.schemas.pr import ChangedFile, PRPayload
from src.schemas.review import Severity


def _state(*files: ChangedFile) -> dict:
    pr = PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha="head",
        author="octocat",
        title="Test PR",
        changed_files=list(files),
    )
    return {"pr_payload": pr}


def _hunk(*added_lines: str) -> str:
    n = len(added_lines) + 1
    body = "".join(f"+{line}\n" for line in added_lines)
    return f"@@ -1,1 +1,{n} @@\n context\n{body}"


def test_flags_bare_except():
    patch = _hunk("try:", "    risky()", "except:", "    pass")
    findings = run_architecture_agent(
        _state(ChangedFile(filename="a.py", status="modified", patch=patch))
    )["architecture_findings"]
    titles = [f.title for f in findings]
    assert "Bare except clause" in titles


def test_flags_print_statement():
    patch = _hunk('print("debug")')
    findings = run_architecture_agent(
        _state(ChangedFile(filename="a.py", status="modified", patch=patch))
    )["architecture_findings"]
    assert findings[0].title == "print() used instead of logging"
    assert findings[0].severity == Severity.LOW


def test_flags_deeply_nested_line():
    deeply_nested = " " * 16 + "do_the_thing()"
    patch = _hunk(deeply_nested)
    findings = run_architecture_agent(
        _state(ChangedFile(filename="a.py", status="modified", patch=patch))
    )["architecture_findings"]
    assert findings[0].title == "Deeply nested logic"


def test_flags_large_diff_by_additions_field():
    big_file = ChangedFile(filename="big.py", status="modified", additions=301, patch=None)
    findings = run_architecture_agent(_state(big_file))["architecture_findings"]
    assert findings[0].title == "Large diff in a single file"
    assert findings[0].line_number is None


def test_clean_small_diff_produces_no_findings():
    patch = _hunk("x = compute()")
    findings = run_architecture_agent(
        _state(ChangedFile(filename="a.py", status="modified", additions=1, patch=patch))
    )["architecture_findings"]
    assert findings == []


def test_merges_llm_findings_when_llm_enabled(monkeypatch):
    monkeypatch.setattr(
        architecture_agent,
        "call_llm_findings",
        lambda system_prompt, user_prompt: [
            {
                "line_number": 2,
                "title": "God function",
                "severity": "MEDIUM",
                "description": "This function handles parsing, validation, and persistence.",
                "suggestion": "Split into three smaller functions.",
            }
        ],
    )
    patch = _hunk("def do_everything(): pass")
    findings = run_architecture_agent(
        _state(ChangedFile(filename="a.py", status="modified", patch=patch))
    )["architecture_findings"]
    titles = [f.title for f in findings]
    assert "God function" in titles
