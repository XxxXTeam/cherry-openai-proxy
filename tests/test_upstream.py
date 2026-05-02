import httpx
import pytest

from app.config import Settings
from app.upstream import CherryClient, CherryUpstreamError, create_http_client


def test_stream_error_response_is_read_before_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(
                b'{"error":{"message":"bad signature","type":"authentication_error","code":"invalid_signature"}}'
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        cherry_base_url="https://api.cherry-ai.com",
        cherry_models={"qwen": "qwen"},
    )
    upstream = CherryClient(settings, http_client=client)

    try:
        upstream.stream_chat_completion(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        )
        assert False, "expected CherryUpstreamError"
    except CherryUpstreamError as exc:
        assert exc.status_code == 401
        assert exc.error_type == "authentication_error"
        assert exc.code == "invalid_signature"
        assert exc.message == "bad signature"


def test_create_http_client_falls_back_without_h2(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        calls.append(bool(kwargs.get("http2", False)))
        if kwargs.get("http2"):
            raise ImportError("missing h2")
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)

    client = create_http_client(30.0)

    try:
        assert calls == [True, False]
        assert isinstance(client, real_client)
    finally:
        client.close()
