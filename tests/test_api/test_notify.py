import httpx

from src.api import notify


def test_no_webhook_configured_is_a_quiet_no_op(monkeypatch):
    monkeypatch.setattr(notify.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("posted")))
    assert notify.notify_slack(None, "t", "b") is False


def test_posts_escaped_text(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(notify.httpx, "post", fake_post)

    assert notify.notify_slack("https://hooks.slack.test/T/B/x", "Drift on m", "lr 0.1 -> 0.3 & <more>") is True
    assert sent["url"] == "https://hooks.slack.test/T/B/x"
    # "->" must not reach Slack as a raw ">" (it would be parsed as markup)
    assert sent["json"] == {"text": "*Drift on m*\nlr 0.1 -&gt; 0.3 &amp; &lt;more&gt;"}


def test_slack_failure_never_raises_and_does_not_log_the_url(monkeypatch, caplog):
    url = "https://hooks.slack.test/T/B/secret-token"
    monkeypatch.setattr(
        notify.httpx, "post", lambda u, json, timeout: httpx.Response(500, request=httpx.Request("POST", u))
    )

    assert notify.notify_slack(url, "t", "b") is False
    assert "secret-token" not in caplog.text

    def down(*a, **k):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(notify.httpx, "post", down)
    assert notify.notify_slack(url, "t", "b") is False
