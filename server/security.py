"""Protections for a public deployment, configured from environment variables.

SANTORINI_API_KEYS        Comma-separated keys; when set, every /api request (except /api/health) needs a
                          matching X-API-Key header, and WebSocket clients send the key in their first message.
                          A key shipped inside a public web page is visible to anyone, so treat it as an
                          identifier you can rotate, not as a password.
SANTORINI_ALLOWED_ORIGINS Comma-separated sites hosted elsewhere (for example https://you.github.io) allowed by
                          CORS and the WebSocket origin check.
SANTORINI_RATE_LIMITS     "off" disables per-IP rate limits (on by default).
SANTORINI_TRUST_PROXY     "1" behind a hosting reverse proxy: client IPs come from X-Forwarded-For.
SANTORINI_API_DOCS        "1" publishes /docs and /openapi.json (off by default).
"""
from __future__ import annotations

import hmac
import math
import os
import time
import unicodedata
from dataclasses import dataclass, field

from fastapi import HTTPException, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _list(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = field(default_factory=lambda: tuple(_list("SANTORINI_API_KEYS")))
    allowed_origins: tuple[str, ...] = field(default_factory=lambda: tuple(o.rstrip("/") for o in _list("SANTORINI_ALLOWED_ORIGINS")))
    rate_limits: bool = field(default_factory=lambda: os.getenv("SANTORINI_RATE_LIMITS", "on").lower() not in {"off", "0", "false"})
    api_docs: bool = field(default_factory=lambda: os.getenv("SANTORINI_API_DOCS", "") == "1")
    max_rooms: int = field(default_factory=lambda: int(os.getenv("SANTORINI_MAX_ROOMS", "10000")))
    max_body: int = 4096
    max_sockets_per_ip: int = 16
    max_sockets_per_room: int = 8

    @property
    def public_key(self) -> str:
        """The key a page served by this server should send (the first configured key)."""
        return self.api_keys[0] if self.api_keys else ""


def key_matches(candidate: str | None, keys: tuple[str, ...] | list[str]) -> bool:
    """Constant-time comparison against every configured key."""
    if not candidate:
        return False
    found = False
    for key in keys:
        found |= hmac.compare_digest(candidate.encode(), key.encode())
    return found


def client_ip(scope_or_request) -> str:
    client = scope_or_request.client if hasattr(scope_or_request, "client") else scope_or_request.get("client")
    return (client.host if hasattr(client, "host") else client[0]) if client else "unknown"


class RateLimiter:
    """Token buckets per (client IP, action). In memory: right for the single-process server."""

    # Requests allowed per minute, per client IP.
    LIMITS = {"create": 10, "join": 20, "read": 120, "write": 120, "socket": 30}

    def __init__(self, enabled: bool = True, clock=time.monotonic) -> None:
        self.enabled, self.clock = enabled, clock
        self.buckets: dict[tuple[str, str], tuple[float, float]] = {}

    def check(self, ip: str, action: str) -> None:
        if not self.enabled:
            return
        rate = self.LIMITS[action]
        now = self.clock()
        tokens, last = self.buckets.get((ip, action), (float(rate), now))
        tokens = min(float(rate), tokens + (now - last) * rate / 60)
        if tokens < 1:
            retry = math.ceil((1 - tokens) * 60 / rate)
            raise HTTPException(429, "Too many requests. Wait a moment and try again.", headers={"Retry-After": str(retry)})
        self.buckets[(ip, action)] = (tokens - 1, now)
        if len(self.buckets) > 50_000:
            self.prune(now)

    def prune(self, now: float) -> None:
        """Forget buckets that have refilled completely."""
        self.buckets = {key: value for key, value in self.buckets.items() if now - value[1] < 60}


def clean_name(name: str, fallback: str) -> str:
    """Display names: no control or formatting characters, collapsed whitespace, at most 24 characters."""
    spaced = "".join(" " if ch.isspace() else ch for ch in name)
    visible = "".join(ch for ch in spaced if unicodedata.category(ch) not in {"Cc", "Cf", "Cs", "Co", "Cn"})
    return " ".join(visible.split())[:24] or fallback


class BodyLimit:
    """Reject request bodies over a byte limit, including chunked uploads without a Content-Length.

    API bodies are tiny, so the body is read (up to the limit) before the app sees it, then replayed.
    """

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.limit:
            await self._reject(send)
            return
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunks.append(message.get("body", b""))
            size += len(chunks[-1])
            if size > self.limit:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break
        body, replayed = b"".join(chunks), False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"detail":"Request body too large."}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def security_headers(request: Request, headers, is_page: bool) -> None:
    """Defence-in-depth headers; the page gets a strict Content-Security-Policy."""
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()")
    if request.url.scheme == "https":
        headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    if is_page:
        host = request.headers.get("host", "")
        headers["Content-Security-Policy"] = "; ".join([
            "default-src 'self'", "script-src 'self'", "style-src 'self'", "img-src 'self' data: blob:", "font-src 'self'",
            f"connect-src 'self' wss://{host} ws://{host}", "worker-src 'self'", "object-src 'none'", "base-uri 'self'",
            "form-action 'self'", "frame-ancestors 'none'",
        ])
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
