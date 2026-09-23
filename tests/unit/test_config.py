"""Settings construction: env reading is in from_env(), not in the defaults."""

from src.core.config import _DEFAULT_GROQ_MODEL, Settings


def test_bare_settings_does_not_read_the_environment(monkeypatch):
    """Field defaults used to be `os.environ.get(...)` evaluated once at
    import, so a bare Settings() in a test picked up the developer's real
    .env and could never be overridden after the first import."""
    monkeypatch.setenv("GITHUB_TOKEN", "from-the-environment")

    assert Settings().github_token is None
    assert Settings(github_token="explicit").github_token == "explicit"


def test_from_env_reads_the_current_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "shh")
    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.setenv("GROQ_MODEL", "some/model")

    settings = Settings.from_env()

    assert settings.github_token == "tok"
    assert settings.github_webhook_secret == "shh"
    assert settings.groq_api_key == "key"
    assert settings.groq_model == "some/model"


def test_from_env_falls_back_when_groq_model_is_set_but_empty(monkeypatch):
    """A copied .env with a bare `GROQ_MODEL=` line sets the variable to "",
    which os.environ.get treats as present."""
    monkeypatch.setenv("GROQ_MODEL", "")
    assert Settings.from_env().groq_model == _DEFAULT_GROQ_MODEL


def test_from_env_reads_diagnostic_core_settings(monkeypatch):
    monkeypatch.setenv("TELEMETRY_BACKEND", "wandb")
    monkeypatch.setenv("WANDB_ENTITY", "acme")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prom:9090")
    monkeypatch.setenv("MLCOP_MONITOR_INTERVAL_SECONDS", "900")
    monkeypatch.setenv("MLCOP_MONITORED_MODELS", "fraud-clf, churn,")
    monkeypatch.setenv("MLCOP_PIPELINE_PATHS", "pipelines/,etl/load.py")
    monkeypatch.setenv("GITHUB_REPO", "acme/models")
    monkeypatch.setenv("MLCOP_REPO_MODELS", " acme/a = model-a ,acme/b=model-b,broken,=x,")

    settings = Settings.from_env()

    assert settings.telemetry_backend == "wandb"
    assert settings.wandb_entity == "acme"
    assert settings.prometheus_url == "http://prom:9090"
    assert settings.monitor_interval_seconds == 900
    assert settings.drift_window_seconds == 3600
    assert settings.monitored_models == ("fraud-clf", "churn")
    assert settings.pipeline_paths == ("pipelines/", "etl/load.py")
    assert settings.github_repo == "acme/models"
    assert settings.repo_models == {"acme/a": "model-a", "acme/b": "model-b"}


def test_from_env_defaults_to_the_fake_backend(monkeypatch):
    monkeypatch.setenv("TELEMETRY_BACKEND", "")
    monkeypatch.delenv("MLCOP_REPO_MODELS", raising=False)
    settings = Settings.from_env()
    assert settings.telemetry_backend == "fake"
    assert settings.repo_models == {}


def test_llm_explain_is_an_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key")  # a key alone doesn't opt in
    monkeypatch.delenv("MLCOP_LLM_EXPLAIN", raising=False)
    assert Settings.from_env().llm_explain is False
    for value, expected in [("1", True), ("true", True), ("Yes", True), ("0", False), ("", False)]:
        monkeypatch.setenv("MLCOP_LLM_EXPLAIN", value)
        assert Settings.from_env().llm_explain is expected
