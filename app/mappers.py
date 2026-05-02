from __future__ import annotations

import json


ALLOWED_OPTIONAL_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
)


def to_cherry_payload(request_payload: dict, upstream_model: str) -> dict:
    messages = request_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty array.")

    payload = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(request_payload.get("stream", False)),
    }

    for key in ALLOWED_OPTIONAL_FIELDS:
        if key in request_payload and request_payload[key] is not None:
            payload[key] = request_payload[key]

    if "tools" in request_payload:
        payload["tools"] = request_payload["tools"]

    if "tool_choice" in request_payload:
        payload["tool_choice"] = request_payload["tool_choice"]

    return payload


def normalize_chat_completion_response(response_body: dict, public_model_name: str) -> dict:
    if not isinstance(response_body, dict):
        return response_body

    normalized = dict(response_body)
    normalized["model"] = public_model_name
    normalized.setdefault("object", "chat.completion")
    return normalized


def normalize_stream_line(line: str | bytes, public_model_name: str) -> str | None:
    if isinstance(line, bytes):
        line = line.decode("utf-8")

    raw_line = line.strip("\r\n")
    if not raw_line:
        return None

    if raw_line.startswith(":") or raw_line.startswith("event:") or raw_line.startswith("id:") or raw_line.startswith("retry:"):
        return f"{raw_line}\n"

    if raw_line.startswith("data:"):
        payload = raw_line[5:].strip()
    else:
        payload = raw_line

    if payload == "[DONE]":
        return "data: [DONE]\n\n"

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return f"data: {payload}\n\n"

    if isinstance(parsed, dict):
        parsed["model"] = public_model_name
        parsed.setdefault("object", "chat.completion.chunk")

    return f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
