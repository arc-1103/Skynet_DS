from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest
from github import GithubException

from src.agents.architecture_agent import run_architecture_agent
from src.agents.security_agent import run_security_agent
from src.core.config import _parse_fix_pr_mode
from src.github_integration import fix_pr
from src.github_integration.fix_pr import apply_line_fixes, open_fix_pr
from src.schemas.pr import ChangedFile, PRPayload
from src.schemas.review import FinalPRReview, LineFix, ReviewFinding, Severity
from src.worker import jobs

HEAD = "deadbeefcafe"
SOURCE = "import yaml\n\ndef load(f):\n    return yaml.load(f)\n"
FIX = LineFix(old_line="    return yaml.load(f)", new_line="    return yaml.safe_load(f)")


def _review(*findings):
    return FinalPRReview(overall_passed=True, markdown_summary="", inline_findings=list(findings))


def _finding(path="app.py", line=4, fix=FIX):
    return ReviewFinding(
        file_path=path, line_number=line, title="t", severity=Severity.MEDIUM, description="d", fix=fix
    )


# --- fake GitHub ------------------------------------------------------------


def _missing():
    return GithubException(404, {}, {})


@dataclass
class FakeRepo:
    files: dict[str, str]
    refs: set[str] = field(default_factory=set)
    calls: list[tuple] = field(default_factory=list)
    pull: SimpleNamespace = None
    fail_on_update: bool = False
    full_name: str = "o/r"
    open_pulls: list = field(default_factory=list)

    def get_pull(self, number):
        return self.pull

    def get_pulls(self, state):
        return self.open_pulls

    def get_contents(self, path, ref):
        if path not in self.files:
            raise _missing()
        text = self.files[path]
        return SimpleNamespace(sha=f"blob-{path}", decoded_content=text.encode())

    def get_git_ref(self, name):
        if name not in self.refs:
            raise _missing()
        return SimpleNamespace(delete=lambda: self.calls.append(("delete_ref", name)) or self.refs.discard(name))

    def create_git_ref(self, ref, sha):
        self.refs.add(ref.removeprefix("refs/"))
        self.calls.append(("create_ref", ref, sha))

    def update_file(self, path, message, content, sha, branch):
        if self.fail_on_update:
            raise GithubException(500, {}, {})
        self.calls.append(("update_file", path, content, sha, branch))

    def create_pull(self, title, body, base, head):
        self.calls.append(("create_pull", base, head, body))
        return SimpleNamespace(html_url="https://github.com/o/r/pull/99")

    def writes(self):
        return [c for c in self.calls if c[0] != "delete_ref"]


def _pull(**over):
    base = {"state": "open", "head": SimpleNamespace(ref="feature", sha=HEAD, repo=SimpleNamespace(full_name="o/r"))}
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def repo(monkeypatch):
    r = FakeRepo(files={"app.py": SOURCE}, pull=_pull())
    monkeypatch.setattr(fix_pr, "Github", lambda token: SimpleNamespace(get_repo=lambda name: r))
    return r


# --- apply_line_fixes -------------------------------------------------------


def test_apply_line_fixes_patches_only_matching_lines():
    stale = LineFix(old_line="something else", new_line="x")
    text, applied, skipped = apply_line_fixes(SOURCE, [(4, FIX), (2, stale), (99, FIX)])
    assert text == SOURCE.replace("yaml.load(f)", "yaml.safe_load(f)")
    assert applied == [4]
    assert [line for line, _ in skipped] == [2, 99]


def test_apply_line_fixes_keeps_crlf():
    crlf = SOURCE.replace("\n", "\r\n")
    fix = LineFix(old_line=FIX.old_line + "\r", new_line=FIX.new_line + "\r")
    text, applied, _ = apply_line_fixes(crlf, [(4, fix)])
    assert applied == [4] and text.count("\r\n") == crlf.count("\r\n")


# --- open_fix_pr ------------------------------------------------------------


def test_opens_fix_pr_against_the_pr_branch(repo):
    result = open_fix_pr("o/r", 7, _review(_finding()), "tok")

    branch = f"skynet/fix-pr7-{HEAD[:7]}"
    assert result.url == "https://github.com/o/r/pull/99" and result.branch == branch
    assert repo.calls[0] == ("create_ref", f"refs/heads/{branch}", HEAD)  # branched from the PR head, not main
    _, path, content, blob, on = repo.calls[1]
    assert (path, blob, on) == ("app.py", "blob-app.py", branch)
    assert "yaml.safe_load(f)" in content and "yaml.load(f)" not in content
    _, base, head, body = repo.calls[2]
    assert (base, head) == ("feature", branch)  # targets the PR's branch, never the default branch
    assert "yaml.safe_load" in body


def test_dry_run_writes_nothing(repo):
    result = open_fix_pr("o/r", 7, _review(_finding()), "tok", dry_run=True)
    assert result.url is None and len(result.applied) == 1
    assert repo.calls == []


def test_nothing_to_fix_opens_nothing(repo):
    assert open_fix_pr("o/r", 7, _review(_finding(fix=None), _finding(line=None)), "tok") is None
    assert repo.writes() == []


@pytest.mark.parametrize(
    "pull",
    [
        _pull(state="closed"),
        _pull(head=SimpleNamespace(ref="skynet/fix-pr3-abc1234", sha=HEAD, repo=SimpleNamespace(full_name="o/r"))),
        _pull(head=SimpleNamespace(ref="feature", sha=HEAD, repo=SimpleNamespace(full_name="someone/fork"))),
        _pull(head=SimpleNamespace(ref="feature", sha=HEAD, repo=None)),  # fork whose repo was deleted
    ],
    ids=["closed", "is-a-fix-pr", "fork", "deleted-fork"],
)
def test_ineligible_prs_are_skipped(repo, pull):
    repo.pull = pull
    assert open_fix_pr("o/r", 7, _review(_finding()), "tok") is None
    assert repo.calls == []


def test_repo_name_comparison_ignores_case(repo):
    repo.pull = _pull(head=SimpleNamespace(ref="feature", sha=HEAD, repo=SimpleNamespace(full_name="O/R")))
    assert open_fix_pr("o/r", 7, _review(_finding()), "tok") is not None


def test_line_changed_since_review_is_skipped(repo):
    repo.files["app.py"] = SOURCE.replace("yaml.load(f)", "yaml.load(f, x)")
    assert open_fix_pr("o/r", 7, _review(_finding()), "tok") is None
    assert repo.calls == []


def test_unreadable_file_is_skipped_not_fatal(repo):
    result = open_fix_pr("o/r", 7, _review(_finding(path="gone.py"), _finding()), "tok")
    assert [p for p, _, _ in result.applied] == ["app.py"]
    assert [p for p, _, _ in result.skipped] == ["gone.py"]


def test_existing_fix_branch_is_not_duplicated(repo):
    repo.refs.add(f"heads/skynet/fix-pr7-{HEAD[:7]}")
    assert open_fix_pr("o/r", 7, _review(_finding()), "tok") is None
    assert repo.writes() == []


def test_failed_commit_removes_the_half_written_branch(repo):
    repo.fail_on_update = True
    with pytest.raises(GithubException):
        open_fix_pr("o/r", 7, _review(_finding()), "tok")
    assert repo.refs == set()  # a retry isn't blocked by the leftover branch


# --- where fixes come from --------------------------------------------------


def _pr(path, *added):
    patch = f"@@ -0,0 +1,{len(added)} @@\n" + "\n".join("+" + a for a in added)
    return PRPayload(
        pr_id=1, repo_full_name="o/r", commit_sha=HEAD, author="a", title="t",
        changed_files=[ChangedFile(filename=path, status="added", additions=len(added), patch=patch)],
    )


# The eval/os.system/yaml.load/verify=False strings below are fixtures: source text handed to
# the scanner as data. None of it is ever executed.
def _security(*added):
    return run_security_agent({"pr_payload": _pr("a.py", *added)})["security_findings"]


def test_mechanical_security_checks_carry_an_exact_fix():
    yaml_f, verify_f = _security("cfg = yaml.load(f)", "requests.get(u, verify=False)")
    assert yaml_f.fix == LineFix(old_line="cfg = yaml.load(f)", new_line="cfg = yaml.safe_load(f)")
    assert verify_f.fix == LineFix(old_line="requests.get(u, verify=False)", new_line="requests.get(u, verify=True)")


def test_checks_without_a_safe_rewrite_carry_no_fix():
    findings = _security("x = eval(s)", "cfg = yaml.load(f, Loader=yaml.Loader)", "os.system(c)")
    assert findings and all(f.fix is None for f in findings)


def test_bare_except_fix():
    findings = run_architecture_agent({"pr_payload": _pr("a.py", "try:", "    f()", "except:", "    pass")})["architecture_findings"]
    bare = next(f for f in findings if f.title == "Bare except clause")
    assert bare.fix == LineFix(old_line="except:", new_line="except Exception:")


# --- settings and the job hook ----------------------------------------------


@pytest.mark.parametrize(
    "raw,mode",
    [("", "off"), ("true", "on"), ("ON", "on"), ("dry-run", "dry-run"), ("tru", "off"), ("0", "off")],
)
def test_fix_pr_mode_parsing_fails_closed(raw, mode):
    assert _parse_fix_pr_mode(raw) == mode


def test_job_hook_does_nothing_when_off(monkeypatch):
    # set explicitly: the default comes from the environment, which a developer's .env may flip on
    monkeypatch.setattr(jobs, "settings", replace(jobs.settings, fix_pr_mode="off", github_token="tok"))
    monkeypatch.setattr(jobs, "open_fix_pr", lambda *a, **k: pytest.fail("must not run when off"))
    jobs._maybe_open_fix_pr("o/r", 7, _review(_finding()))


def test_job_hook_passes_mode_and_swallows_failures(monkeypatch):
    seen = []
    monkeypatch.setattr(jobs, "settings", replace(jobs.settings, fix_pr_mode="dry-run", github_token="tok"))
    monkeypatch.setattr(jobs, "open_fix_pr", lambda *a, **k: seen.append(k) or (_ for _ in ()).throw(RuntimeError("boom")))
    jobs._maybe_open_fix_pr("o/r", 7, _review(_finding()))  # must not raise
    assert seen == [{"dry_run": True}]


def test_job_hook_needs_a_token(monkeypatch):
    monkeypatch.setattr(jobs, "settings", replace(jobs.settings, fix_pr_mode="on", github_token=None))
    monkeypatch.setattr(jobs, "open_fix_pr", lambda *a, **k: pytest.fail("no token, no call"))
    jobs._maybe_open_fix_pr("o/r", 7, _review(_finding()))


# --- superseded fix PRs -----------------------------------------------------


def _fix_pull(repo, number, ref, full_name="o/r"):
    log = repo.calls
    return SimpleNamespace(
        number=number,
        head=SimpleNamespace(ref=ref, repo=SimpleNamespace(full_name=full_name)),
        create_issue_comment=lambda body: log.append(("comment", number)),
        edit=lambda state: log.append(("edit", number, state)),
    )


def test_new_push_retires_the_old_fix_pr_and_its_branch(repo):
    repo.refs.add("heads/skynet/fix-pr7-aaaaaaa")
    repo.open_pulls = [_fix_pull(repo, 11, "skynet/fix-pr7-aaaaaaa")]
    open_fix_pr("o/r", 7, _review(_finding()), "tok")
    assert ("edit", 11, "closed") in repo.calls and ("delete_ref", "heads/skynet/fix-pr7-aaaaaaa") in repo.calls


def test_stale_fix_pr_is_retired_even_when_nothing_new_needs_fixing(repo):
    repo.refs.add("heads/skynet/fix-pr7-aaaaaaa")
    repo.open_pulls = [_fix_pull(repo, 11, "skynet/fix-pr7-aaaaaaa")]
    assert open_fix_pr("o/r", 7, _review(), "tok") is None
    assert ("edit", 11, "closed") in repo.calls


def test_only_this_prs_own_fix_branches_are_ever_closed(repo):
    repo.open_pulls = [
        _fix_pull(repo, 20, f"skynet/fix-pr7-{HEAD[:7]}"),        # current head: keep
        _fix_pull(repo, 21, "skynet/fix-pr8-aaaaaaa"),            # another PR's fix
        _fix_pull(repo, 22, "feature-x"),                          # a human's PR
        _fix_pull(repo, 23, "skynet/fix-pr7-bbbbbbb", "evil/fork"),  # same name, from a fork
    ]
    open_fix_pr("o/r", 7, _review(), "tok")
    assert [c for c in repo.calls if c[0] == "edit"] == []


def test_dry_run_closes_nothing(repo):
    repo.open_pulls = [_fix_pull(repo, 11, "skynet/fix-pr7-aaaaaaa")]
    open_fix_pr("o/r", 7, _review(_finding()), "tok", dry_run=True)
    assert repo.calls == []
