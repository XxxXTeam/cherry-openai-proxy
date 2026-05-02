import json

import pytest

from app.factory import create_app
from app.config import Settings


class FakeCherryStream:
    def __init__(
        self,
        lines: list[str] | None = None,
        raw_chunks: list[bytes] | None = None,
        content_type: str = "text/event-stream",
    ) -> None:
        self.lines = lines
        self.raw_chunks = raw_chunks
        self.content_type = content_type
        self.closed = False
        self.close_calls = 0

    @property
    def is_sse(self) -> bool:
        return self.content_type.startswith("text/event-stream")

    def __iter__(self):
        for line in self.lines or []:
            yield line

    def iter_bytes(self):
        for chunk in self.raw_chunks or []:
            yield chunk

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class FakeCherryClient:
    def __init__(self, response: dict | None = None, stream_lines: list[str] | None = None) -> None:
        self.response = response or {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        }
        self.stream = FakeCherryStream(
            lines=stream_lines
            or [
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"he"},"index":0}]}',
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"llo"},"index":0}]}',
            ]
        )
        self.last_payload = None

    def create_chat_completion(self, payload: dict) -> dict:
        self.last_payload = payload
        return self.response

    def stream_chat_completion(self, payload: dict) -> FakeCherryStream:
        self.last_payload = payload
        return self.stream


@pytest.fixture
def settings() -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8000,
        openai_api_key=None,
        cherry_base_url="https://api.cherry-ai.com",
        cherry_models={"qwen": "qwen"},
        request_timeout=30.0,
    )


def test_health_and_models(settings: Settings) -> None:
    app = create_app(settings, FakeCherryClient())
    client = app.test_client()

    health_response = client.get("/health")
    models_response = client.get("/v1/models")

    assert health_response.status_code == 200
    assert health_response.get_json() == {"status": "ok"}
    assert models_response.status_code == 200
    assert models_response.get_json()["data"] == [
        {
            "id": "qwen",
            "object": "model",
            "owned_by": "cherry-proxy",
        }
    ]


def test_chat_completion_maps_request_and_response(settings: Settings) -> None:
    fake_client = FakeCherryClient(
        response={
            "id": "chatcmpl-1",
            "model": "qwen-upstream",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    app = create_app(settings, fake_client)
    client = app.test_client()

    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "fetchJson"}}],
        "tool_choice": "auto",
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "extra_body": {"passthrough": True},
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert fake_client.last_payload == {
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "fetchJson"}}],
        "tool_choice": "auto",
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "extra_body": {"passthrough": True},
    }
    assert response.get_json() == fake_client.response


def test_chat_completion_rejects_missing_model(settings: Settings) -> None:
    app = create_app(settings, FakeCherryClient())
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_model"


def test_models_can_map_public_name_to_upstream_name() -> None:
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        openai_api_key=None,
        cherry_base_url="https://api.cherry-ai.com",
        cherry_models={"qwen-cherry": "qwen", "glm": "GLM-4.5-Air"},
        request_timeout=30.0,
    )
    fake_client = FakeCherryClient()
    app = create_app(settings, fake_client)
    client = app.test_client()

    models_response = client.get("/v1/models")
    chat_response = client.post(
        "/v1/chat/completions",
        json={
            "model": "glm",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert models_response.status_code == 200
    assert models_response.get_json()["data"] == [
        {"id": "qwen-cherry", "object": "model", "owned_by": "cherry-proxy"},
        {"id": "glm", "object": "model", "owned_by": "cherry-proxy"},
    ]
    assert chat_response.status_code == 200
    assert fake_client.last_payload["model"] == "GLM-4.5-Air"
    assert chat_response.get_json() == fake_client.response


def test_unmapped_model_is_forwarded_as_is(settings: Settings) -> None:
    fake_client = FakeCherryClient()
    app = create_app(settings, fake_client)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert fake_client.last_payload["model"] == "Qwen/Qwen3-8B"


def test_streaming_chat_returns_sse_and_done(settings: Settings) -> None:
    fake_client = FakeCherryClient(
        stream_lines=['{"id":"chatcmpl-1","choices":[{"delta":{"content":"hi"},"index":0}]}']
    )
    fake_client.stream.content_type = "application/x-ndjson"
    app = create_app(settings, fake_client)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert 'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hi"},"index":0}]}' in body
    assert body.strip().endswith("data: [DONE]")
    assert fake_client.stream.closed is True
    assert fake_client.stream.close_calls == 1


def test_streaming_chat_preserves_sse_metadata_lines(settings: Settings) -> None:
    fake_client = FakeCherryClient(
        stream_lines=[
            ": keep-alive",
            'event: message',
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hi"},"index":0}]}',
            "data: [DONE]",
        ]
    )
    fake_client.stream.content_type = "application/x-ndjson"
    app = create_app(settings, fake_client)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert ": keep-alive\n" in body
    assert "event: message\n" in body
    assert body.count("data: [DONE]") == 1


def test_streaming_tool_call_sse_is_passthrough(settings: Settings) -> None:
    tool_chunks = [
        b'data: {"id":"1","object":"chat.completion.chunk","model":"Qwen/Qwen3-8B","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"mcp__CherryFetch__fetchJson"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"1","object":"chat.completion.chunk","model":"Qwen/Qwen3-8B","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"","function":{"arguments":"{\\"url\\":\\"https://example.com\\"}"}}]},"finish_reason":null}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    fake_client = FakeCherryClient()
    fake_client.stream = FakeCherryStream(raw_chunks=tool_chunks)
    app = create_app(settings, fake_client)
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "fetchJson"}}],
        },
    )

    body = response.get_data()

    assert response.status_code == 200
    assert body == b"".join(tool_chunks)
    assert fake_client.stream.close_calls == 1


def test_bearer_auth_is_optional_or_enforced() -> None:
    protected_settings = Settings(
        host="127.0.0.1",
        port=8000,
        openai_api_key="secret-key",
        cherry_base_url="https://api.cherry-ai.com",
        cherry_models={"qwen": "qwen"},
    )
    app = create_app(protected_settings, FakeCherryClient())
    client = app.test_client()

    missing = client.get("/v1/models")
    wrong = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    ok = client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200


def test_error_response_is_logged(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    app = create_app(settings, FakeCherryClient())
    client = app.test_client()

    with caplog.at_level("ERROR"):
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    assert any("API request failed" in message for message in caplog.messages)
    assert any("invalid_model" in message for message in caplog.messages)


def test_upstream_json_error_is_returned_as_is(settings: Settings) -> None:
    class ErrorCherryClient(FakeCherryClient):
        def create_chat_completion(self, payload: dict) -> dict:
            from app.upstream import CherryUpstreamError

            raise CherryUpstreamError(
                "upstream bad request",
                status_code=400,
                error_type="invalid_request_error",
                code="bad_request",
                body={"error": {"message": "upstream bad request", "type": "invalid_request_error"}},
                content_type="application/json; charset=utf-8",
            )

    app = create_app(settings, ErrorCherryClient())
    client = app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": {"message": "upstream bad request", "type": "invalid_request_error"}
    }


def test_sse_stream_is_logged_when_enabled(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        openai_api_key=None,
        cherry_base_url="https://api.cherry-ai.com",
        cherry_models={"qwen": "qwen"},
        request_timeout=30.0,
        log_sse_stream=True,
    )
    fake_client = FakeCherryClient()
    fake_client.stream = FakeCherryStream(raw_chunks=[b"data: hello\n\n", b"data: [DONE]\n\n"])
    app = create_app(settings, fake_client)
    client = app.test_client()

    with caplog.at_level("INFO"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        _ = response.get_data()

    assert response.status_code == 200
    assert any("SSE upstream raw chunk=" in message for message in caplog.messages)
