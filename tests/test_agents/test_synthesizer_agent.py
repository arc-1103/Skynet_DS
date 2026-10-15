from src.agents.synthesizer import run_synthesizer_agent
from src.schemas.drift import ColumnDriftReport, ColumnKind, DatasetDriftReport
from src.schemas.pr import PRPayload
from src.schemas.review import PRLabel, ReviewFinding, Severity


def _pr() -> PRPayload:
    return PRPayload(
        pr_id=42,
        repo_full_name="acme/demo",
        commit_sha="head",
        author="octocat",
        title="Add feature",
        changed_files=[],
    )


def _finding(severity: Severity, title: str = "issue", line: int = 1) -> ReviewFinding:
    return ReviewFinding(
        file_path="a.py", line_number=line, title=title, severity=severity, description="desc"
    )


def _base_state(**overrides) -> dict:
    state = {
        "pr_payload": _pr(),
        "security_findings": [],
        "architecture_findings": [],
        "ml_debugger_findings": [],
        "drift_report": [],
    }
    state.update(overrides)
    return state


def test_no_findings_passes_and_has_no_labels():
    result = run_synthesizer_agent(_base_state())
    review = result["final_review"]
    assert review.overall_passed is True
    assert review.labels == []
    assert review.inline_findings == []


def test_high_severity_security_finding_fails_and_labels_security():
    result = run_synthesizer_agent(
        _base_state(security_findings=[_finding(Severity.HIGH)])
    )
    review = result["final_review"]
    assert review.overall_passed is False
    assert PRLabel.SECURITY in review.labels


def test_low_severity_architecture_finding_does_not_fail_but_labels_refactor():
    result = run_synthesizer_agent(
        _base_state(architecture_findings=[_finding(Severity.LOW)])
    )
    review = result["final_review"]
    assert review.overall_passed is True
    assert PRLabel.REFACTOR in review.labels


def test_ml_debugger_findings_label_performance():
    result = run_synthesizer_agent(
        _base_state(ml_debugger_findings=[_finding(Severity.MEDIUM)])
    )
    assert PRLabel.PERFORMANCE in result["final_review"].labels


def test_drifted_report_labels_config_and_appears_in_summary():
    column = ColumnDriftReport(
        column="a", kind=ColumnKind.NUMERICAL, n_reference=10, n_current=10,
        results=[], drifted=True, vote="majority",
    )
    report = DatasetDriftReport(columns=[column], share_threshold=0.5, label="data/x.csv")
    result = run_synthesizer_agent(_base_state(drift_report=[report]))
    review = result["final_review"]
    assert PRLabel.CONFIG in review.labels
    assert "DRIFTED" in review.markdown_summary


def test_inline_findings_aggregate_all_three_specialists():
    result = run_synthesizer_agent(
        _base_state(
            security_findings=[_finding(Severity.HIGH, "sec")],
            architecture_findings=[_finding(Severity.LOW, "arch", line=2)],
            ml_debugger_findings=[_finding(Severity.MEDIUM, "ml", line=3)],
        )
    )
    titles = {f.title for f in result["final_review"].inline_findings}
    assert titles == {"sec", "arch", "ml"}


def test_flagged_drift_fails_the_review():
    """overall_passed was computed from inline findings alone, so a PR whose
    data files all drifted rendered "PASSED" right next to its CONFIG label."""
    drifted = DatasetDriftReport(
        columns=[
            ColumnDriftReport(
                column="amount",
                kind=ColumnKind.NUMERICAL,
                n_reference=100,
                n_current=100,
                drifted=True,
                vote="majority",
            )
        ],
        share_threshold=0.5,
        label="data/train.csv",
    )
    review = run_synthesizer_agent(_base_state(drift_report=[drifted]))["final_review"]

    assert drifted.dataset_drifted is True
    assert review.overall_passed is False
    assert "CHANGES REQUESTED" in review.markdown_summary


def test_same_line_flagged_by_security_is_not_repeated_by_architecture():
    sec = ReviewFinding(file_path="m.py", line_number=11, title="Use of eval()", severity=Severity.HIGH, description="d")
    arch = ReviewFinding(file_path="m.py", line_number=11, title="Use of eval()", severity=Severity.CRITICAL, description="d")
    other_line = ReviewFinding(file_path="m.py", line_number=3, title="God Function", severity=Severity.MEDIUM, description="d")
    file_level = ReviewFinding(file_path="m.py", line_number=None, title="Large diff", severity=Severity.MEDIUM, description="d")

    review = run_synthesizer_agent(
        _base_state(security_findings=[sec], architecture_findings=[arch, other_line, file_level])
    )["final_review"]

    assert review.inline_findings == [sec, other_line, file_level]
