from __future__ import annotations


def to_cherry_payload(request_payload: dict, upstream_model: str) -> dict:
    messages = request_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty array.")

    payload = dict(request_payload)
    payload["model"] = upstream_model
    return payload


def format_stream_line_as_sse(line: str | bytes) -> str | None:
    if isinstance(line, bytes):
        line = line.decode("utf-8")

    raw_line = line.strip("\r\n")
    if not raw_line:
        return None

    if raw_line.startswith((":", "event:", "id:", "retry:", "data:")):
        if raw_line == "data: [DONE]":
            return "data: [DONE]\n\n"
        if raw_line.startswith("data:"):
            return f"{raw_line}\n\n"
        return f"{raw_line}\n"

    if raw_line == "[DONE]":
        return "data: [DONE]\n\n"

    return f"data: {raw_line}\n\n"
