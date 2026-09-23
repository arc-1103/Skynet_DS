import json
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.agents import drift_agent
from src.core.config import settings
from src.retrieval import evidence
from src.schemas.drift import DatasetDriftReport
from src.schemas.pr import ChangedFile, PRPayload
from src.worker import jobs


@pytest.fixture
def slack(monkeypatch):
    """Captures Slack posts instead of sending them."""
    posts = []
    monkeypatch.setattr(jobs, "notify_slack", lambda url, title, body: posts.append((url, title, body)))
    return posts


@pytest.fixture
def mapped_repo(tmp_path, monkeypatch, slack):
    """Maps acme/fraud-clf to the fake adapter's seeded model and stubs the
    GitHub compare call; returns the list of diagnose() calls."""
    monkeypatch.chdir(tmp_path)  # memory.db lands in tmp_path, not the repo
    monkeypatch.setattr(
        jobs, "settings",
        replace(settings, repo_models={"acme/fraud-clf": "fraud-clf"}, slack_webhook_url="https://hooks.test/x"),
    )
    monkeypatch.setattr(evidence, "fetch_commit_diff", lambda repo, base, head, token: f"{repo} diff")
    calls = []
    real_diagnose = jobs.diagnose

    def spy(*args, **kwargs):
        result = real_diagnose(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(jobs, "diagnose", spy)
    return calls


@pytest.mark.asyncio
async def test_run_deployment_regression_check_diagnoses_the_mapped_model(mapped_repo, slack):
    await jobs.run_deployment_regression_check(
        repo_full_name="acme/fraud-clf",
        deploy_sha="abc999",
        environment="production",
    )

    [(args, kwargs, result)] = mapped_repo
    assert args[0] == "run-004"  # latest run of fraud-clf
    assert kwargs["deploy_sha"] == "abc999"
    code = next(c for c in result.ranked_candidates if c.kind == "code")
    # code evidence is scoped to the deploy and fetched from the deploy's repo
    assert "abc111..abc999: acme/fraud-clf diff" in code.description
    [(url, title, body)] = slack
    assert url == "https://hooks.test/x"
    assert title == "Deploy abc999 of acme/fraud-clf to production - fraud-clf post-deploy check"
    assert body == result.explanation


@pytest.mark.asyncio
async def test_run_deployment_regression_check_skips_an_unmapped_repo(mapped_repo, slack):
    await jobs.run_deployment_regression_check(
        repo_full_name="acme/other",
        deploy_sha="def456",
        environment="staging",
    )
    assert mapped_repo == []
    assert slack == []  # nothing diagnosed, nothing posted


@pytest.mark.asyncio
async def test_run_deployment_regression_check_handles_a_model_with_no_runs(mapped_repo, monkeypatch):
    monkeypatch.setattr(jobs, "settings", replace(settings, repo_models={"acme/fraud-clf": "no-such-model"}))
    await jobs.run_deployment_regression_check(
        repo_full_name="acme/fraud-clf",
        deploy_sha="def456",
        environment="staging",
    )
    assert mapped_repo == []


@pytest.mark.asyncio
async def test_a_deploy_with_no_metric_change_does_not_page(mapped_repo, slack, monkeypatch):
    """Review finding: with W&B a deploy usually has no new training run, so
    the latest run is the baseline itself - that used to post a "most
    likely cause" for a regression that never happened, on every deploy."""
    from src.observability.fake_adapter import FakeTelemetryAdapter
    from src.schemas.telemetry import NormalizedRun

    only = NormalizedRun(
        run_id="r1", model_id="fraud-clf", timestamp=datetime.now(timezone.utc),
        metrics={"acc": 0.91}, config={"lr": 0.1}, code_ref="abc111",
    )
    monkeypatch.setattr(
        jobs, "build_adapter", lambda *a: FakeTelemetryAdapter(runs=[only], baselines={"fraud-clf": "r1"})
    )

    await jobs.run_deployment_regression_check(
        repo_full_name="acme/fraud-clf", deploy_sha="bbb222", environment="production"
    )

    [(_, _, result)] = mapped_repo
    assert result.ranked_candidates == []
    assert "no meaningful delta" in result.explanation
    assert slack == []


def _sample(model_id, shift, n=300):
    """`shift` moves amount and age; tenure never moves. So a shift drifts
    2/3 features - over the detector's 50% share_threshold, while keeping
    the drift's own confidence below a seasonal verdict's."""
    import numpy as np

    from src.schemas.telemetry import FeatureSample

    rng = np.random.default_rng(1)
    now = datetime.now(timezone.utc)
    return FeatureSample(
        model_id=model_id, start=now - timedelta(hours=1), end=now,
        features={
            "amount": rng.normal(shift, 1, n).tolist(),
            "age": rng.normal(40 + 5 * shift, 5, n).tolist(),
            "tenure": rng.normal(3, 1, n).tolist(),
        },
    )


@pytest.fixture
def prometheus(tmp_path, monkeypatch, slack):
    """Points check_drift at a fake Prometheus whose next window is
    whatever the test puts in `windows`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jobs, "settings",
        replace(
            settings, prometheus_url="http://prom", repo_models={"acme/fraud": "fraud-clf"},
            slack_webhook_url="https://hooks.test/x",
        ),
    )
    windows = []

    class FakePrometheus:
        def __init__(self, url):
            assert url == "http://prom"

        def get_feature_samples(self, model_id, start, end):
            return windows.pop(0)

    monkeypatch.setattr(jobs, "PrometheusAdapter", FakePrometheus)
    return windows


def test_check_drift_skips_without_prometheus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "settings", replace(settings, prometheus_url=None))
    monkeypatch.setattr(jobs, "PrometheusAdapter", lambda url: pytest.fail("should not query"))
    assert jobs.check_drift("fraud-clf") is None


def test_check_drift_captures_a_baseline_then_diagnoses_drift(prometheus, slack):
    from src.memory.sqlite_store import SQLiteStore

    prometheus += [_sample("fraud-clf", 0), _sample("fraud-clf", 0.02), _sample("fraud-clf", 3)]

    assert jobs.check_drift("fraud-clf") is None  # first window becomes the baseline
    assert SQLiteStore(jobs.settings.memory_db_path).get_baseline_distribution("fraud-clf") is not None
    assert jobs.check_drift("fraud-clf") is None  # same distribution: no drift, no diagnosis
    assert slack == []  # neither a baseline capture nor a clean check pages anyone

    explanation = jobs.check_drift("fraud-clf")  # amount and age shifted 3 sigma

    assert explanation.startswith("Drift on 'fraud-clf': 2/3 features drifted")
    assert "Upstream data shift in age, amount" in explanation
    assert slack == [("https://hooks.test/x", "Drift detected on fraud-clf", explanation)]
    assert SQLiteStore(jobs.settings.memory_db_path).get_diagnosis_history("fraud-clf")


def test_check_drift_does_not_baseline_an_empty_window(prometheus):
    from src.memory.sqlite_store import SQLiteStore
    from src.schemas.telemetry import FeatureSample

    now = datetime.now(timezone.utc)
    prometheus.append(FeatureSample(model_id="fraud-clf", start=now, end=now))

    assert jobs.check_drift("fraud-clf") is None
    assert SQLiteStore(jobs.settings.memory_db_path).get_baseline_distribution("fraud-clf") is None


def test_a_persistent_drift_is_one_episode_diagnosed_and_paged_once(prometheus, slack):
    """Review finding: every sweep used to re-diagnose and re-page the same
    ongoing drift, and count each sweep as a recurrence - by the fourth
    sweep a pipeline-caused drift was reported as seasonal variation."""
    from src.memory.sqlite_store import SQLiteStore

    prometheus += [_sample("fraud-clf", 0)] + [_sample("fraud-clf", 3) for _ in range(5)]

    results = [jobs.check_drift("fraud-clf") for _ in range(6)]  # baseline + 5 sweeps

    assert results[0] is None and results[1] is not None
    assert results[2:] == [None] * 4  # still the same episode: already reported
    assert len(slack) == 1
    assert len(SQLiteStore(jobs.settings.memory_db_path).get_diagnosis_history("fraud-clf")) == 1


def test_seasonality_counts_episodes_separated_by_clean_checks(prometheus, slack):
    drift, clean = _sample("fraud-clf", 3), _sample("fraud-clf", 0.02)
    prometheus += [_sample("fraud-clf", 0), drift, drift, clean, drift, clean, clean, drift]

    results = [jobs.check_drift("fraud-clf") for _ in range(8)]

    episodes = [r for r in results if r is not None]
    assert len(episodes) == len(slack) == 3  # episode starts only
    assert "Most likely cause (data, confidence 0.57): Upstream data shift" in episodes[1]
    # the third episode of the same features reads as recurring, not new
    assert "Most likely cause (data, confidence 0.70): Expected/seasonal variation: "         "the same features drifted in 2 earlier episodes" in episodes[2]


@pytest.mark.asyncio
async def test_monitoring_sweep_checks_every_model_and_isolates_failures(monkeypatch):
    monkeypatch.setattr(jobs, "settings", replace(settings, monitored_models=(), repo_models={"a/x": "m1", "a/y": "m2", "a/z": "m1"}))
    checked = []

    def fake_check(model_id):
        checked.append(model_id)
        if model_id == "m1":
            raise RuntimeError("prometheus down")

    monkeypatch.setattr(jobs, "check_drift", fake_check)

    await jobs.run_monitoring_sweep()

    assert checked == ["m1", "m2"]  # deduplicated, and m1 failing didn't stop m2


def test_monitored_models_prefers_the_explicit_list(monkeypatch):
    monkeypatch.setattr(jobs, "settings", replace(settings, monitored_models=("only",), repo_models={"a/x": "m1"}))
    assert jobs.monitored_models() == ("only",)


@pytest.mark.asyncio
async def test_run_pr_code_review_caches_fetched_pr_locally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fake_pr = PRPayload(
        pr_id=42,
        repo_full_name="acme/fraud-clf",
        commit_sha="deadbeef",
        author="octocat",
        title="Fix flaky test",
        body="Closes #41",
        changed_files=[ChangedFile(filename="tests/test_x.py", status="modified", patch="@@ -1 +1 @@")],
    )
    captured = {}

    def fake_fetch_pr(repo_full_name, pr_number, token=None):
        captured["args"] = (repo_full_name, pr_number, token)
        return fake_pr

    monkeypatch.setattr(jobs, "fetch_pr", fake_fetch_pr)

    await jobs.run_pr_code_review(
        repo_full_name="acme/fraud-clf",
        pr_number=42,
        head_sha="deadbeef",
    )

    assert captured["args"] == ("acme/fraud-clf", 42, settings.github_token)

    out_path = tmp_path / settings.pr_cache_dir / "acme_fraud-clf" / "pr_42_deadbeef.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["pr_id"] == 42
    assert saved["changed_files"][0]["filename"] == "tests/test_x.py"


@pytest.mark.asyncio
async def test_run_pr_code_review_skips_drift_for_a_code_only_pr(tmp_path, monkeypatch):
    """Drift checking needs a clone, so a PR touching no data files must not
    trigger one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jobs,
        "fetch_pr",
        lambda *a, **k: PRPayload(
            pr_id=7,
            repo_full_name="acme/fraud-clf",
            commit_sha="cafe",
            base_sha="beef",
            author="octocat",
            title="Code only",
            changed_files=[ChangedFile(filename="src/model.py", status="modified")],
        ),
    )
    monkeypatch.setattr(
        jobs, "clone_or_checkout", lambda *a, **k: pytest.fail("should not clone")
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=7, head_sha="cafe")


@pytest.mark.asyncio
async def test_run_pr_code_review_writes_a_drift_report_per_data_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jobs,
        "fetch_pr",
        lambda *a, **k: PRPayload(
            pr_id=8,
            repo_full_name="acme/fraud-clf",
            commit_sha="cafe",
            base_sha="beef",
            author="octocat",
            title="Refresh data",
            changed_files=[ChangedFile(filename="data/train.csv", status="modified")],
        ),
    )
    monkeypatch.setattr(jobs, "clone_or_checkout", lambda *a, **k: tmp_path / "clone")
    # drift now reaches the job through the graph's drift branch, so the stub
    # goes where that branch imported it
    monkeypatch.setattr(
        drift_agent,
        "check_pr_drift",
        lambda pr, repo_path: [
            DatasetDriftReport(columns=[], share_threshold=0.5, label="data/train.csv")
        ],
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=8, head_sha="cafe")

    cache_dir = tmp_path / settings.pr_cache_dir / "acme_fraud-clf"
    drift_path = cache_dir / "pr_8_cafe_drift_data_train.csv.json"
    assert drift_path.exists()
    assert json.loads(drift_path.read_text(encoding="utf-8"))["label"] == "data/train.csv"

    review = json.loads((cache_dir / "pr_8_cafe_review.json").read_text(encoding="utf-8"))
    assert review["overall_passed"] is True


@pytest.mark.asyncio
async def test_run_pr_code_review_writes_the_multi_agent_review(tmp_path, monkeypatch):
    """The graph used to be unreachable from the webhook path entirely."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jobs,
        "fetch_pr",
        lambda *a, **k: PRPayload(
            pr_id=9,
            repo_full_name="acme/fraud-clf",
            commit_sha="cafe",
            author="octocat",
            title="Add handler",
            changed_files=[
                ChangedFile(
                    filename="src/handler.py",
                    status="modified",
                    patch="@@ -1,1 +1,2 @@\n context\n+eval(user_input)\n",
                )
            ],
        ),
    )
    monkeypatch.setattr(
        jobs, "clone_or_checkout", lambda *a, **k: pytest.fail("no data file - should not clone")
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=9, head_sha="cafe")

    review_path = tmp_path / settings.pr_cache_dir / "acme_fraud-clf" / "pr_9_cafe_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assert review["overall_passed"] is False
    assert [f["title"] for f in review["inline_findings"]] == ["Use of eval()"]
    assert review["inline_findings"][0]["line_number"] == 2


@pytest.mark.asyncio
async def test_run_pr_code_review_rejects_a_head_sha_that_is_not_a_sha(tmp_path, monkeypatch):
    """head_sha comes straight off the webhook payload and is interpolated
    into the cache filename, so it must not be able to carry a path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "fetch_pr", lambda *a, **k: pytest.fail("should not fetch"))

    with pytest.raises(ValueError, match="non-SHA head_sha"):
        await jobs.run_pr_code_review(
            repo_full_name="acme/fraud-clf", pr_number=9, head_sha="../../../../x"
        )


def _data_pr(pr_id: int):
    return PRPayload(
        pr_id=pr_id,
        repo_full_name="acme/fraud-clf",
        commit_sha="cafe",
        base_sha="beef",
        author="octocat",
        title="Refresh data",
        changed_files=[ChangedFile(filename="data/train.csv", status="modified")],
    )


@pytest.mark.asyncio
async def test_run_pr_code_review_still_reviews_when_the_clone_fails(tmp_path, monkeypatch):
    """A transient git failure shouldn't cost the whole review - the PR is
    already cached and the static branches work off the diff alone."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "fetch_pr", lambda *a, **k: _data_pr(10))

    def _git_exploded(*a, **k):
        raise subprocess.CalledProcessError(128, ["git", "clone"])

    monkeypatch.setattr(jobs, "clone_or_checkout", _git_exploded)

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=10, head_sha="cafe")

    cache_dir = tmp_path / settings.pr_cache_dir / "acme_fraud-clf"
    review = json.loads((cache_dir / "pr_10_cafe_review.json").read_text(encoding="utf-8"))
    assert review["overall_passed"] is True
    assert not list(cache_dir.glob("*_drift_*.json"))


@pytest.mark.asyncio
async def test_drift_report_filename_survives_a_backslash_in_the_label(tmp_path, monkeypatch):
    """git tracks paths containing a literal backslash, and with_name() raises
    on an OS separator - which on Windows silently lost every remaining drift
    report for that PR."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "fetch_pr", lambda *a, **k: _data_pr(11))
    monkeypatch.setattr(jobs, "clone_or_checkout", lambda *a, **k: tmp_path / "clone")
    monkeypatch.setattr(
        drift_agent,
        "check_pr_drift",
        lambda pr, repo_path: [
            DatasetDriftReport(columns=[], share_threshold=0.5, label="data\\odd:name.csv")
        ],
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=11, head_sha="cafe")

    cache_dir = tmp_path / settings.pr_cache_dir / "acme_fraud-clf"
    written = list(cache_dir.glob("pr_11_cafe_drift_*.json"))
    assert len(written) == 1
    assert written[0].name == "pr_11_cafe_drift_data_odd_name.csv.json"
