import hashlib
import hmac
import pytest

from app.signatures import (
    CHERRY_CLIENT_ID,
    CHERRY_SIGNING_SECRET,
    generate_signature_headers,
    serialize_body,
)


def test_serialize_body_matches_expected_json_shape() -> None:
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}
    assert serialize_body(body) == '{"model":"qwen","messages":[{"role":"user","content":"hello"}]}'


def test_serialize_body_normalizes_json_string_before_signing() -> None:
    body = '{\n  "model": "qwen",\n  "messages": []\n}'
    assert serialize_body(body) == '{"model":"qwen","messages":[]}'


def test_generate_signature_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.signatures.CHERRY_SIGNING_SECRET", "test-secret")

    headers = generate_signature_headers(
        method="POST",
        path="/chat/completions",
        body={"model": "qwen", "messages": []},
        timestamp=1700000000,
    )

    expected_signature = hmac.new(
        b"test-secret",
        b'POST\n/chat/completions\n\ncherry-studio\n1700000000\n{"model":"qwen","messages":[]}',
        hashlib.sha256,
    ).hexdigest()

    assert headers == {
        "X-Client-ID": CHERRY_CLIENT_ID,
        "X-Timestamp": "1700000000",
        "X-Signature": expected_signature,
    }


def test_generate_signature_headers_uses_embedded_secret() -> None:
    headers = generate_signature_headers(
        method="POST",
        path="/chat/completions",
        body="",
        timestamp=1700000000,
    )

    assert headers["X-Client-ID"] == CHERRY_CLIENT_ID
    assert "." in CHERRY_SIGNING_SECRET
