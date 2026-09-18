from dataclasses import replace

from src.agents import llm_client


def _with_api_key(api_key):
    return replace(llm_client.settings, groq_api_key=api_key)


def test_llm_disabled_when_no_api_key(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key(None))
    assert llm_client.llm_enabled() is False
    assert llm_client.call_llm_findings("system", "user") is None


def test_llm_enabled_when_api_key_present(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    assert llm_client.llm_enabled() is True


def test_call_llm_findings_returns_none_on_client_error(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))

    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(llm_client, "_client", _boom)
    assert llm_client.call_llm_findings("system", "user") is None


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


def test_call_llm_findings_parses_json_object(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    monkeypatch.setattr(
        llm_client,
        "_client",
        lambda: _FakeClient('{"findings": [{"title": "x"}]}'),
    )
    assert llm_client.call_llm_findings("system", "user") == [{"title": "x"}]


def test_call_llm_findings_strips_code_fence(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    monkeypatch.setattr(
        llm_client,
        "_client",
        lambda: _FakeClient('```json\n{"findings": []}\n```'),
    )
    assert llm_client.call_llm_findings("system", "user") == []


def test_call_llm_findings_returns_none_on_malformed_json(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    monkeypatch.setattr(llm_client, "_client", lambda: _FakeClient("not json"))
    assert llm_client.call_llm_findings("system", "user") is None


def test_call_llm_findings_returns_none_when_findings_key_missing(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    monkeypatch.setattr(llm_client, "_client", lambda: _FakeClient('{"other": []}'))
    assert llm_client.call_llm_findings("system", "user") is None


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


class _FakeToolMessage:
    def __init__(self, content: str | None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeToolResponse:
    def __init__(self, message: _FakeToolMessage):
        self.choices = [type("Choice", (), {"message": message})]


class _FakeToolCompletions:
    """Returns one canned response per call, in order - lets a test script
    a multi-round tool-call conversation the same way the real Groq SDK
    would produce one across successive .create() calls."""

    def __init__(self, responses: list[_FakeToolResponse]):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeToolChat:
    def __init__(self, responses: list[_FakeToolResponse]):
        self.completions = _FakeToolCompletions(responses)


class _FakeToolClient:
    def __init__(self, responses: list[_FakeToolResponse]):
        self.chat = _FakeToolChat(responses)


def test_call_llm_with_tools_disabled_when_no_api_key(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key(None))
    assert llm_client.call_llm_with_tools("system", "user", [], {}) is None


def test_call_llm_with_tools_returns_findings_without_any_tool_call(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    responses = [_FakeToolResponse(_FakeToolMessage('{"findings": [{"title": "x"}]}'))]
    monkeypatch.setattr(llm_client, "_client", lambda: _FakeToolClient(responses))
    result = llm_client.call_llm_with_tools("system", "user", [], {})
    assert result == [{"title": "x"}]


def test_call_llm_with_tools_executes_requested_tool_and_returns_final_findings(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    tool_call = _FakeToolCall("call_1", "my_tool", '{"x": 2}')
    responses = [
        _FakeToolResponse(_FakeToolMessage(None, tool_calls=[tool_call])),
        _FakeToolResponse(_FakeToolMessage('{"findings": [{"title": "used tool"}]}')),
    ]
    fake_client = _FakeToolClient(responses)
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    calls: list[int] = []

    def my_tool(x: int) -> dict:
        calls.append(x)
        return {"doubled": x * 2}

    result = llm_client.call_llm_with_tools(
        "system", "user", [{"type": "function"}], {"my_tool": my_tool}
    )
    assert result == [{"title": "used tool"}]
    assert calls == [2]


def test_call_llm_with_tools_reports_unknown_tool_without_crashing(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    tool_call = _FakeToolCall("call_1", "does_not_exist", "{}")
    responses = [
        _FakeToolResponse(_FakeToolMessage(None, tool_calls=[tool_call])),
        _FakeToolResponse(_FakeToolMessage('{"findings": []}')),
    ]
    fake_client = _FakeToolClient(responses)
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)
    result = llm_client.call_llm_with_tools("system", "user", [{"type": "function"}], {})
    assert result == []


def test_call_llm_with_tools_reports_tool_exception_without_crashing(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    tool_call = _FakeToolCall("call_1", "boom_tool", "{}")
    responses = [
        _FakeToolResponse(_FakeToolMessage(None, tool_calls=[tool_call])),
        _FakeToolResponse(_FakeToolMessage('{"findings": []}')),
    ]
    fake_client = _FakeToolClient(responses)
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    def boom_tool():
        raise RuntimeError("kaboom")

    result = llm_client.call_llm_with_tools(
        "system", "user", [{"type": "function"}], {"boom_tool": boom_tool}
    )
    assert result == []


def test_call_llm_with_tools_returns_none_when_loop_never_terminates(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))
    tool_call = _FakeToolCall("call_1", "my_tool", "{}")
    responses = [
        _FakeToolResponse(_FakeToolMessage(None, tool_calls=[tool_call]))
        for _ in range(llm_client._MAX_TOOL_ROUNDS)
    ]
    fake_client = _FakeToolClient(responses)
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)
    result = llm_client.call_llm_with_tools(
        "system", "user", [{"type": "function"}], {"my_tool": lambda: {}}
    )
    assert result is None


def test_call_llm_with_tools_returns_none_on_client_error(monkeypatch):
    monkeypatch.setattr(llm_client, "settings", _with_api_key("fake-key"))

    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(llm_client, "_client", _boom)
    assert llm_client.call_llm_with_tools("system", "user", [], {}) is None
