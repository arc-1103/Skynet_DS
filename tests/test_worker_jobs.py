import pytest

from src.worker import jobs


@pytest.mark.asyncio
async def test_run_deployment_regression_check_diagnoses_seeded_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # memory.db lands in tmp_path, not the repo

    await jobs.run_deployment_regression_check(
        repo_full_name="acme/fraud-clf",
        deploy_sha="abc123",
        environment="production",
    )
    # No assertion beyond "it completes without raising" - this job is a
    # documented stub (see TODO in src/worker/jobs.py) until Phase 1 wires a
    # real repo -> model_id mapping and a real adapter.


@pytest.mark.asyncio
async def test_run_deployment_regression_check_handles_unknown_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "_STUB_MODEL_ID", "no-such-model")

    await jobs.run_deployment_regression_check(
        repo_full_name="acme/other",
        deploy_sha="def456",
        environment="staging",
    )
    # Should log a warning and return cleanly, not raise.
