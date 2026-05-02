from __future__ import annotations

import json

from flask import Flask, Response, jsonify, request, stream_with_context

from .auth import is_authorized
from .config import Settings
from .mappers import (
    format_stream_line_as_sse,
    to_cherry_payload,
)
from .upstream import CherryClient, CherryUpstreamError


def create_app(
    settings: Settings | None = None,
    cherry_client: CherryClient | None = None,
) -> Flask:
    settings = settings or Settings.from_env()
    cherry_client = cherry_client or CherryClient(settings)

    app = Flask(__name__)
    app.config["SETTINGS"] = settings
    app.config["CHERRY_CLIENT"] = cherry_client

    @app.before_request
    def require_api_key() -> Response | None:
        if not request.path.startswith("/v1/"):
            return None

        if is_authorized(request, settings.openai_api_key):
            return None

        return openai_error(
            "Invalid or missing API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    @app.after_request
    def log_error_response(response: Response) -> Response:
        if request.path.startswith("/v1/") and response.status_code >= 400:
            response_body = response.get_json(silent=True)
            app.logger.error(
                "API request failed remote_addr=%s method=%s path=%s status=%s request=%s response=%s",
                request.remote_addr,
                request.method,
                request.path,
                response.status_code,
                summarize_request_body(),
                safe_log_json(response_body),
            )
        return response

    @app.get("/")
    def index() -> Response:
        return jsonify(
            {
                "name": "cherry-openai-proxy",
                "object": "service",
                "message": "OpenAI-compatible proxy for Cherry chat completions.",
                "endpoints": [
                    "/v1/chat/completions",
                    "/v1/models",
                    "/health",
                ],
                "model": settings.default_public_model,
                "models": list(settings.public_models.keys()),
            }
        )

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/v1/models")
    def list_models() -> Response:
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": public_model,
                        "object": "model",
                        "owned_by": settings.model_owner,
                    }
                    for public_model in settings.public_models
                ],
            }
        )

    @app.post("/v1/chat/completions")
    def chat_completions() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return openai_error(
                "Request body must be a JSON object.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_json",
            )

        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            return openai_error(
                "'model' must be a non-empty string.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_model",
            )
        upstream_model = settings.resolve_upstream_model(model) or model

        try:
            upstream_payload = to_cherry_payload(payload, upstream_model)
        except ValueError as exc:
            return openai_error(
                str(exc),
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_request",
            )

        stream = bool(payload.get("stream", False))

        try:
            if stream:
                upstream_stream = cherry_client.stream_chat_completion(upstream_payload)

                if getattr(upstream_stream, "is_sse", False) and hasattr(upstream_stream, "iter_bytes"):
                    def generate_passthrough() -> bytes:
                        seen_done = False
                        tail = b""
                        done_marker = b"data: [DONE]"

                        try:
                            for chunk in upstream_stream.iter_bytes():
                                if not chunk:
                                    continue
                                if settings.log_sse_stream:
                                    app.logger.info(
                                        "SSE upstream raw chunk=%s",
                                        safe_log_text(chunk.decode("utf-8", errors="replace")),
                                    )
                                combined = tail + chunk
                                if done_marker in combined:
                                    seen_done = True
                                tail = combined[-64:]
                                yield chunk
                        except GeneratorExit:
                            app.logger.info("Client closed streaming response path=%s", request.path)
                            raise
                        except Exception:
                            app.logger.exception(
                                "Streaming passthrough failed path=%s upstream_request=%s",
                                request.path,
                                safe_log_json(upstream_payload),
                            )
                        finally:
                            close = getattr(upstream_stream, "close", None)
                            if callable(close):
                                close()

                        if not seen_done:
                            yield b"data: [DONE]\n\n"

                    return Response(
                        stream_with_context(generate_passthrough()),
                        content_type="text/event-stream; charset=utf-8",
                        direct_passthrough=True,
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )

                def generate() -> bytes:
                    seen_done = False
                    try:
                        for line in upstream_stream:
                            if settings.log_sse_stream:
                                raw_line = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
                                app.logger.info("SSE upstream line=%s", safe_log_text(raw_line))
                            normalized = format_stream_line_as_sse(line)
                            if normalized is None:
                                continue
                            if settings.log_sse_stream:
                                app.logger.info("SSE proxy line=%s", safe_log_text(normalized))
                            if normalized.strip() == "data: [DONE]":
                                seen_done = True
                            yield normalized.encode("utf-8")
                    except GeneratorExit:
                        app.logger.info("Client closed streaming response path=%s", request.path)
                        raise
                    except Exception:
                        app.logger.exception(
                            "Streaming proxy failed path=%s upstream_request=%s",
                            request.path,
                            safe_log_json(upstream_payload),
                        )
                    finally:
                        close = getattr(upstream_stream, "close", None)
                        if callable(close):
                            close()

                    if not seen_done:
                        yield b"data: [DONE]\n\n"

                return Response(
                    stream_with_context(generate()),
                    content_type="text/event-stream; charset=utf-8",
                    direct_passthrough=True,
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            response_body = cherry_client.create_chat_completion(upstream_payload)
            return json_response(response_body)
        except CherryUpstreamError as exc:
            app.logger.error(
                "Cherry upstream error status=%s type=%s code=%s message=%s upstream_request=%s",
                exc.status_code,
                exc.error_type,
                exc.code,
                exc.message,
                safe_log_json(upstream_payload),
            )
            if exc.body is not None:
                return upstream_error_response(exc)
            return openai_error(
                exc.message,
                status_code=exc.status_code,
                error_type=exc.error_type,
                code=exc.code,
            )

    return app


def openai_error(
    message: str,
    *,
    status_code: int,
    error_type: str,
    code: str | None = None,
) -> Response:
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                }
            }
        ),
        status_code,
    )


def json_response(payload: object, status_code: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False),
        status=status_code,
        content_type="application/json; charset=utf-8",
    )


def upstream_error_response(error: CherryUpstreamError) -> Response:
    if isinstance(error.body, dict):
        return Response(
            json.dumps(error.body, ensure_ascii=False),
            status=error.status_code,
            content_type=error.content_type or "application/json; charset=utf-8",
        )

    return Response(
        error.body,
        status=error.status_code,
        content_type=error.content_type or "text/plain; charset=utf-8",
    )


def summarize_request_body() -> str:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return "<non-json>"

    summary = {
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "messages": len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else None,
        "tools": len(payload.get("tools", [])) if isinstance(payload.get("tools"), list) else None,
        "tool_choice": payload.get("tool_choice"),
    }
    return safe_log_json(summary)


def safe_log_json(value: object, limit: int = 1500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = repr(value)

    return safe_log_text(text, limit=limit)


def safe_log_text(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text

    return f"{text[:limit]}...<truncated>"
