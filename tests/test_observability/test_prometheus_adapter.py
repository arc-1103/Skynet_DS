from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.observability.prometheus_adapter import PrometheusAdapter

END = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
START = END - timedelta(hours=1)


def _adapter(handler):
    return PrometheusAdapter("http://prom:9090/", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_range_query_groups_series_by_feature():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [
                        {"metric": {"feature": "amount", "instance": "a"}, "values": [[1, "10.5"], [2, "NaN"]]},
                        {"metric": {"feature": "amount", "instance": "b"}, "values": [[1, "11"]]},
                        {"metric": {"feature": "age"}, "values": [[1, "40"], [2, "+Inf"]]},
                        {"metric": {"instance": "unlabelled"}, "values": [[1, "1"]]},
                    ],
                },
            },
        )

    sample = _adapter(handler).get_feature_samples('fraud"clf', START, END)

    assert seen["path"] == "/api/v1/query_range"
    # a quote in the model id can't break out of the label matcher
    assert seen["query"] == 'mlcop_feature_value{model="fraud\\"clf"}'
    assert seen["step"] == "15"
    # NaN/Inf points and series without a feature label are dropped
    assert sample.features == {"amount": [10.5, 11.0], "age": [40.0]}
    assert (sample.start, sample.end) == (START, END)


def test_query_error_is_raised():
    handler = lambda request: httpx.Response(200, json={"status": "error", "error": "parse error"})  # noqa: E731
    with pytest.raises(RuntimeError, match="parse error"):
        _adapter(handler).get_feature_samples("m", START, END)


def test_http_error_is_raised():
    with pytest.raises(httpx.HTTPStatusError):
        _adapter(lambda request: httpx.Response(503)).get_feature_samples("m", START, END)
