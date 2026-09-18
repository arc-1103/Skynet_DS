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
