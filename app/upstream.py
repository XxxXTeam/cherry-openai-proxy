from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

import httpx

from .config import Settings
from .signatures import generate_signature_headers, serialize_body


class CherryUpstreamError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        error_type: str = "api_error",
        code: str | None = None,
        body: dict | str | None = None,
        content_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.body = body
        self.content_type = content_type


@dataclass
class CherryStream:
    response: httpx.Response
    context_manager: object
    closed: bool = field(default=False, init=False)

    @property
    def content_type(self) -> str:
        return self.response.headers.get("content-type", "")

    @property
    def is_sse(self) -> bool:
        return self.content_type.startswith("text/event-stream")

    def __iter__(self) -> Iterator[str]:
        try:
            for line in self.response.iter_lines():
                yield line
        finally:
            self.close()

    def iter_bytes(self) -> Iterator[bytes]:
        try:
            for chunk in self.response.iter_bytes():
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.context_manager.__exit__(None, None, None)


class CherryClient:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.http_client = http_client or create_http_client(settings.request_timeout)

    def create_chat_completion(self, payload: dict) -> dict:
        body_string = serialize_body(payload)
        response = self._request("POST", self.settings.chat_completions_path, payload, body_string)
        return self._parse_json_response(response)

    def stream_chat_completion(self, payload: dict) -> CherryStream:
        body_string = serialize_body(payload)
        path = self.settings.chat_completions_path
        url = f"{self.settings.cherry_base_url}{path}"
        headers = self._build_headers(path, payload)

        try:
            context_manager = self.http_client.stream(
                "POST",
                url,
                content=body_string.encode("utf-8"),
                headers=headers,
            )
            response = context_manager.__enter__()
        except httpx.TimeoutException as exc:
            raise CherryUpstreamError(
                "Upstream request timed out.",
                status_code=504,
                error_type="timeout_error",
                code="upstream_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise CherryUpstreamError(
                "Failed to connect to Cherry upstream.",
                status_code=502,
                error_type="api_connection_error",
                code="upstream_connection_error",
            ) from exc

        if response.status_code >= 400:
            error = self._error_from_response(response)
            context_manager.__exit__(None, None, None)
            raise error

        return CherryStream(response=response, context_manager=context_manager)

    def _request(self, method: str, path: str, payload: dict, body_string: str) -> httpx.Response:
        url = f"{self.settings.cherry_base_url}{path}"
        headers = self._build_headers(path, payload)
        try:
            response = self.http_client.request(
                method,
                url,
                content=body_string.encode("utf-8"),
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise CherryUpstreamError(
                "Upstream request timed out.",
                status_code=504,
                error_type="timeout_error",
                code="upstream_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise CherryUpstreamError(
                "Failed to connect to Cherry upstream.",
                status_code=502,
                error_type="api_connection_error",
                code="upstream_connection_error",
            ) from exc

        if response.status_code >= 400:
            raise self._error_from_response(response)

        return response

    def _build_headers(self, path: str, payload: dict | str | bytes | None) -> dict[str, str]:
        signed_headers = generate_signature_headers(
            method="POST",
            path=path,
            body=payload,
        )

        return {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": self.settings.cherry_user_agent,
            "X-Title": self.settings.cherry_title,
            "HTTP-Referer": self.settings.cherry_referer,
            **signed_headers,
        }

    def _parse_json_response(self, response: httpx.Response) -> dict:
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise CherryUpstreamError(
                "Cherry upstream returned invalid JSON.",
                status_code=502,
                error_type="api_error",
                code="invalid_upstream_json",
            ) from exc

    def _error_from_response(self, response: httpx.Response) -> CherryUpstreamError:
        message = f"Cherry upstream returned HTTP {response.status_code}."
        error_type = "api_error"
        code = None
        body: dict | str | None = None

        try:
            response.read()
        except httpx.ResponseNotRead:
            response.read()
        except httpx.StreamError:
            pass

        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            body = payload
            if isinstance(payload.get("error"), dict):
                error = payload["error"]
                message = error.get("message", message)
                error_type = error.get("type", error_type)
                code = error.get("code", code)
            else:
                message = payload.get("message", payload.get("error", message))
        else:
            try:
                if response.text:
                    message = response.text
                    body = response.text
            except httpx.ResponseNotRead:
                pass

        return CherryUpstreamError(
            message,
            status_code=response.status_code,
            error_type=error_type,
            code=code,
            body=body,
            content_type=response.headers.get("content-type"),
        )


def create_http_client(timeout: float) -> httpx.Client:
    resolved_timeout = httpx.Timeout(timeout=timeout, connect=min(timeout, 10.0))
    try:
        return httpx.Client(timeout=resolved_timeout, http2=True)
    except ImportError:
        return httpx.Client(timeout=resolved_timeout)
