"""Client for the resilient-llm-gateway. Generation is delegated to it so all
provider fallback and retry logic lives in one service, not here."""

from __future__ import annotations

from typing import Any

import httpx

from core.config import get_settings


class GatewayError(RuntimeError):
    """The gateway call failed (unreachable, timed out, or returned non-2xx)."""


def generate(system: str, user: str, timeout: float = 60.0) -> dict[str, Any]:
    """POST /v1/chat and return {content, provider, model}."""
    settings = get_settings()
    payload = {"system": system, "messages": [{"role": "user", "content": user}]}
    try:
        resp = httpx.post(f"{settings.gateway_url}/v1/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GatewayError(str(e)) from e
    data = resp.json()
    return {"content": data["content"], "provider": data.get("provider"), "model": data.get("model")}