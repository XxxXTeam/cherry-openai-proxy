from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) CherryStudio/1.9.4 Chrome/146.0.7680.188 "
    "Electron/41.2.1 Safari/537.36"
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_model_aliases(raw_value: str | None, fallback_model: str) -> dict[str, str]:
    if not raw_value or not raw_value.strip():
        return {fallback_model: fallback_model}

    aliases: dict[str, str] = {}
    normalized = raw_value.replace("\r", "\n").replace(",", "\n").replace(";", "\n")
    for entry in normalized.split("\n"):
        item = entry.strip()
        if not item:
            continue

        if "=" in item:
            public_name, upstream_name = item.split("=", 1)
            public_name = public_name.strip()
            upstream_name = upstream_name.strip()
        else:
            public_name = item
            upstream_name = item

        if not public_name or not upstream_name:
            continue

        aliases[public_name] = upstream_name

    return aliases or {fallback_model: fallback_model}


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    openai_api_key: str | None = None
    cherry_base_url: str = "https://api.cherry-ai.com"
    cherry_models: Mapping[str, str] | None = None
    cherry_user_agent: str = DEFAULT_USER_AGENT
    cherry_referer: str = "https://cherry-ai.com"
    cherry_title: str = "Cherry Studio"
    request_timeout: float = 60.0
    log_sse_stream: bool = False

    @property
    def chat_completions_path(self) -> str:
        return "/chat/completions"

    @property
    def model_owner(self) -> str:
        return "cherry-proxy"

    @property
    def public_models(self) -> dict[str, str]:
        return dict(self.cherry_models or {"qwen": "qwen"})

    @property
    def default_public_model(self) -> str:
        return next(iter(self.public_models))

    def resolve_upstream_model(self, public_model: str | None) -> str | None:
        if public_model is None:
            return None
        return self.public_models.get(public_model)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        fallback_model = os.getenv("CHERRY_MODEL", "qwen")
        return cls(
            host=os.getenv("HOST", cls.host),
            port=int(os.getenv("PORT", cls.port)),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            cherry_base_url=os.getenv("CHERRY_BASE_URL", cls.cherry_base_url).rstrip("/"),
            cherry_models=parse_model_aliases(os.getenv("CHERRY_MODELS"), fallback_model),
            cherry_user_agent=os.getenv("CHERRY_USER_AGENT", cls.cherry_user_agent),
            cherry_referer=os.getenv("CHERRY_REFERER", cls.cherry_referer),
            cherry_title=os.getenv("CHERRY_TITLE", cls.cherry_title),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", cls.request_timeout)),
            log_sse_stream=env_flag("LOG_SSE_STREAM", cls.log_sse_stream),
        )
