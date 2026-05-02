from __future__ import annotations

import hashlib
import hmac
import json
import time

CHERRY_CLIENT_ID = "cherry-studio"
CHERRY_SIGNING_SECRET = "K3RNPFx19hPh1AHr5E1wBEFfi4uYUjoCFuzjDzvS9cAWD8KuKJR8FOClwUpGqRRX.GvI6I5ZrEHcGOWjO5AKhJKGmnwwGfM62XKpWqkjhvzRU2NZIinM77aTGIqhqys0g"


def _normalize_body_for_signature(body: dict | str | bytes | None) -> dict | list | str | None:
    if body is None:
        return None

    if isinstance(body, bytes):
        body = body.decode("utf-8")

    if isinstance(body, str):
        stripped = body.strip()
        if not stripped:
            return ""

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return body

    return body


def serialize_body(body: dict | str | bytes | None) -> str:
    normalized = _normalize_body_for_signature(body)
    if normalized is None:
        return ""

    if isinstance(normalized, str):
        return normalized

    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def generate_signature_headers(
    *,
    method: str,
    path: str,
    query: str = "",
    body: dict | str | bytes | None = None,
    timestamp: int | str | None = None,
) -> dict[str, str]:
    resolved_timestamp = str(timestamp or int(time.time()))
    body_string = serialize_body(body)
    raw = "\n".join(
        [
            method.upper(),
            path,
            query,
            CHERRY_CLIENT_ID,
            resolved_timestamp,
            body_string,
        ]
    )

    signature = hmac.new(
        CHERRY_SIGNING_SECRET.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "X-Client-ID": CHERRY_CLIENT_ID,
        "X-Timestamp": resolved_timestamp,
        "X-Signature": signature,
    }
