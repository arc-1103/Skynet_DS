import numpy as np
import pandas as pd
import pytest

from src.agents import ml_debugger_tools


@pytest.fixture
def csv_pair(tmp_path):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {
            "amount": rng.normal(loc=10.0, scale=1.0, size=200),
            "category": rng.choice(["a", "b", "c"], size=200),
        }
    )
    shifted = pd.DataFrame(
        {
            "amount": rng.normal(loc=25.0, scale=1.0, size=200),
            "category": rng.choice(["a", "b", "c"], size=200),
        }
    )
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    reference.to_csv(reference_path, index=False)
    shifted.to_csv(current_path, index=False)
    return str(reference_path), str(current_path)


def test_inspect_dataset_schema_reports_shape_and_dtypes(tmp_path):
    frame = pd.DataFrame({"x": [1, 2, 3, None], "y": ["a", "b", "b", "c"]})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)

    result = ml_debugger_tools.inspect_dataset_schema(str(path))

    assert result["n_rows"] == 4
    assert result["n_columns"] == 2
    assert result["columns"]["x"]["n_null"] == 1
    assert result["columns"]["y"]["n_unique"] == 3
    assert "x" in result["numeric_summary"]


def test_inspect_dataset_schema_returns_error_for_missing_file():
    result = ml_debugger_tools.inspect_dataset_schema("does/not/exist.csv")
    assert "error" in result


def test_inspect_dataset_schema_returns_error_for_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("not tabular")
    result = ml_debugger_tools.inspect_dataset_schema(str(path))
    assert "error" in result


def test_run_drift_analysis_flags_shifted_column(csv_pair):
    reference_path, current_path = csv_pair
    result = ml_debugger_tools.run_drift_analysis(reference_path, current_path)

    assert "error" not in result
    assert result["scipy_report"]["label"] == f"{reference_path} -> {current_path}"
    columns = {c["column"]: c for c in result["scipy_report"]["columns"]}
    assert columns["amount"]["drifted"] is True
    assert "summary" in result
    # evidently is an optional dependency (see demo/requirements-demo.txt) -
    # the tool must degrade gracefully either way, so accept both outcomes.
    assert "evidently_summary" in result
    if result["evidently_summary"] is None:
        assert "evidently_note" in result


def test_run_drift_analysis_returns_error_for_bad_path(csv_pair):
    _, current_path = csv_pair
    result = ml_debugger_tools.run_drift_analysis("missing.csv", current_path)
    assert "error" in result


def test_explain_drift_root_cause_ranks_the_shifted_feature(csv_pair):
    reference_path, current_path = csv_pair
    result = ml_debugger_tools.explain_drift_root_cause(reference_path, current_path)

    assert "error" not in result
    report = result["report"]
    assert report["separable"] is True
    assert report["attributions"][0]["feature"] == "amount"
    assert "summary" in result


def test_explain_drift_root_cause_returns_error_for_too_few_rows(tmp_path):
    small = tmp_path / "small.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(small, index=False)
    result = ml_debugger_tools.explain_drift_root_cause(str(small), str(small))
    assert "error" in result


class _FakeResponse:
    is_redirect = False
    next_request = None

    def __init__(self, text=""):
        self.text = text

    def raise_for_status(self):
        return None


def _stub_doc_fetch(monkeypatch, get):
    """check_deprecated_api drives redirects by hand through an httpx.Client,
    so the client is stubbed rather than httpx.get - and the SSRF screen is
    stubbed with it, since these tests cover the parsing, not the guard."""

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url):
            return get(url)

    monkeypatch.setattr(ml_debugger_tools, "_unsafe_url_reason", lambda url: None)
    monkeypatch.setattr(ml_debugger_tools.httpx, "Client", _FakeClient)


def test_check_deprecated_api_flags_symbol_near_deprecation_language(monkeypatch):
    html = "<html><body><p>DataFrame.append is deprecated, use pandas.concat instead.</p></body></html>"
    _stub_doc_fetch(monkeypatch, lambda url: _FakeResponse(html))

    result = ml_debugger_tools.check_deprecated_api("DataFrame.append", "https://example.com/docs")
    assert result["likely_deprecated"] is True
    assert result["evidence_snippets"]


def test_check_deprecated_api_no_match_when_symbol_absent(monkeypatch):
    _stub_doc_fetch(
        monkeypatch, lambda url: _FakeResponse("<html><body>Nothing relevant here.</body></html>")
    )

    result = ml_debugger_tools.check_deprecated_api("DataFrame.append", "https://example.com/docs")
    assert result["likely_deprecated"] is False
    assert result["evidence_snippets"] == []


def test_check_deprecated_api_returns_error_on_fetch_failure(monkeypatch):
    import httpx

    def _boom(url):
        raise httpx.ConnectError("no network")

    _stub_doc_fetch(monkeypatch, _boom)

    result = ml_debugger_tools.check_deprecated_api("DataFrame.append", "https://example.com/docs")
    assert "error" in result


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/docs",
        "https://169.254.169.254/latest/meta-data/",
        "https://127.0.0.1/docs",
        "file:///etc/passwd",
    ],
)
def test_check_deprecated_api_refuses_non_public_urls(url):
    """doc_url comes from the model, whose only input is the PR diff, so an
    injected diff must not be able to aim the worker at plaintext HTTP, cloud
    metadata, or anything else inside the network."""
    result = ml_debugger_tools.check_deprecated_api("anything", url)
    assert "refusing to fetch" in result["error"]


def test_check_deprecated_api_screens_every_redirect_hop(monkeypatch):
    """A single allowed URL that 302s inward must not walk past the guard."""

    class _Redirect(_FakeResponse):
        is_redirect = True

        class next_request:
            url = "https://127.0.0.1/internal"

    _stub_doc_fetch(monkeypatch, lambda url: _Redirect())
    monkeypatch.setattr(
        ml_debugger_tools,
        "_unsafe_url_reason",
        lambda url: None if url == "https://example.com/docs" else "not public",
    )

    result = ml_debugger_tools.check_deprecated_api("x", "https://example.com/docs")
    assert "refusing to fetch https://127.0.0.1/internal" in result["error"]


def test_tool_schemas_names_match_implementations():
    schema_names = {tool["function"]["name"] for tool in ml_debugger_tools.TOOL_SCHEMAS}
    assert schema_names == set(ml_debugger_tools.TOOL_IMPLEMENTATIONS)


def test_make_tool_implementations_with_no_repo_path_drops_file_reading_tools():
    """Without a checkout there's no tree to confine a model-supplied path
    to, so the file-reading tools (which would then open any path on the
    server) are omitted entirely rather than handed out unconfined."""
    implementations = ml_debugger_tools.make_tool_implementations(None)
    assert implementations == {"check_deprecated_api": ml_debugger_tools.check_deprecated_api}


def test_make_tool_implementations_resolves_relative_paths_against_repo_path(tmp_path):
    frame_dir = tmp_path / "data"
    frame_dir.mkdir()
    (frame_dir / "train.csv").write_text("x,y\n1,2\n3,4\n")

    implementations = ml_debugger_tools.make_tool_implementations(tmp_path)
    result = implementations["inspect_dataset_schema"]("data/train.csv")

    assert "error" not in result
    assert result["n_rows"] == 2


def test_make_tool_implementations_still_errors_for_a_real_missing_relative_path(tmp_path):
    implementations = ml_debugger_tools.make_tool_implementations(tmp_path)
    result = implementations["inspect_dataset_schema"]("data/does_not_exist.csv")
    assert "error" in result


@pytest.mark.parametrize("escape", ["{outside}", "../abs.csv", "../../abs.csv"])
def test_make_tool_implementations_refuses_paths_outside_the_checkout(tmp_path, escape):
    """The model's only input is the PR diff, so an injected diff must not be
    able to aim a file-reading tool anywhere outside the PR checkout - the
    scanners quote source lines straight back into the transcript."""
    (tmp_path / "abs.csv").write_text("x\n1\n")
    other_base = tmp_path / "unrelated"
    other_base.mkdir()

    implementations = ml_debugger_tools.make_tool_implementations(other_base)
    with pytest.raises(ValueError, match="outside the PR checkout"):
        implementations["inspect_dataset_schema"](escape.format(outside=tmp_path / "abs.csv"))
