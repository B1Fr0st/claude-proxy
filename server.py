import asyncio
import json
import os
import time
from collections import deque

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_CODE_VERSION = "2.1.148"

_STRIP_HEADERS = {"x-api-key", "authorization"}

# Anthropic gates Pro-tier models behind this exact system prompt as the
# first system block — OAuth tokens from `claude setup-token` only get
# haiku without it.
CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

# Fallback: after >2 upstream 429s within 30 min, switch to a Haiku API key
# and run a background probe that flips back when the OAuth token recovers.
FALLBACK_MODEL = "claude-haiku-4-5"
RATE_LIMIT_THRESHOLD = 2  # strictly more than this many events triggers fallback
RATE_LIMIT_WINDOW = 30 * 60
RECOVERY_CHECK_INTERVAL = 60

_token: str = ""
_haiku_api_key: str = ""

_rate_limit_events: deque[float] = deque()
_fallback_mode: bool = False
_state_lock: asyncio.Lock | None = None
_recovery_task: asyncio.Task | None = None


def _get_lock() -> asyncio.Lock:
    global _state_lock
    if _state_lock is None:
        _state_lock = asyncio.Lock()
    return _state_lock


def load_token() -> None:
    global _token, _haiku_api_key
    _token = os.environ.get("CLAUDE_TOKEN", "").strip()
    if not _token:
        raise RuntimeError(
            "CLAUDE_TOKEN not set. Generate one with 'claude setup-token'."
        )
    _haiku_api_key = os.environ.get("HAIKU_API_KEY", "").strip()


def _oauth_headers() -> dict[str, str]:
    return {
        "authorization": f"Bearer {_token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
        "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
    }


def _api_key_headers() -> dict[str, str]:
    return {
        "x-api-key": _haiku_api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": "fine-grained-tool-streaming-2025-05-14",
    }


def _build_upstream_headers(incoming: dict[str, str], fallback: bool) -> dict[str, str]:
    headers = _api_key_headers() if fallback else _oauth_headers()

    if "anthropic-version" in incoming:
        headers["anthropic-version"] = incoming["anthropic-version"]

    if "anthropic-beta" in incoming:
        ours = set(headers["anthropic-beta"].split(","))
        theirs = set(incoming["anthropic-beta"].split(","))
        headers["anthropic-beta"] = ",".join(b for b in ours | theirs if b)

    headers["content-type"] = "application/json"
    headers["accept"] = "application/json, text/event-stream"
    return headers


def _incoming_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in _STRIP_HEADERS}


def _inject_claude_code_system(body: bytes) -> bytes:
    """Ensure the Claude Code identifier is the first system block so the
    OAuth token can access Pro-tier models. Anthropic does exact-match on
    the first block's text, so we must use array form."""
    try:
        data = json.loads(body)
    except Exception:
        return body

    identifier_block = {"type": "text", "text": CLAUDE_CODE_SYSTEM}
    system = data.get("system")

    if system is None:
        data["system"] = [identifier_block]
    elif isinstance(system, str):
        data["system"] = [identifier_block, {"type": "text", "text": system}]
    elif isinstance(system, list):
        first = system[0] if system else None
        already_present = (
            isinstance(first, dict)
            and first.get("type") == "text"
            and first.get("text") == CLAUDE_CODE_SYSTEM
        )
        if not already_present:
            data["system"] = [identifier_block, *system]
    else:
        return body

    return json.dumps(data).encode()


def _force_haiku_model(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body
    data["model"] = FALLBACK_MODEL
    return json.dumps(data).encode()


async def _record_rate_limit() -> None:
    """Record an upstream 429. Activates fallback once we cross the threshold."""
    global _fallback_mode, _recovery_task
    async with _get_lock():
        now = time.time()
        cutoff = now - RATE_LIMIT_WINDOW
        while _rate_limit_events and _rate_limit_events[0] < cutoff:
            _rate_limit_events.popleft()
        _rate_limit_events.append(now)

        if (
            not _fallback_mode
            and _haiku_api_key
            and len(_rate_limit_events) > RATE_LIMIT_THRESHOLD
        ):
            _fallback_mode = True
            print(
                f"[fallback] activated — {len(_rate_limit_events)} 429s in last "
                f"{RATE_LIMIT_WINDOW // 60}min; routing to {FALLBACK_MODEL} via API key"
            )
            if _recovery_task is None or _recovery_task.done():
                _recovery_task = asyncio.create_task(_recovery_loop())


async def _recovery_loop() -> None:
    """While in fallback, periodically probe the OAuth token until it returns 200."""
    global _fallback_mode
    probe_body = json.dumps(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4,
            "system": [{"type": "text", "text": CLAUDE_CODE_SYSTEM}],
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode()

    while _fallback_mode:
        await asyncio.sleep(RECOVERY_CHECK_INTERVAL)
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                resp = await c.post(
                    f"{ANTHROPIC_BASE_URL}/v1/messages",
                    headers=_oauth_headers(),
                    content=probe_body,
                )
            if resp.status_code == 200:
                async with _get_lock():
                    _fallback_mode = False
                    _rate_limit_events.clear()
                print("[fallback] deactivated — OAuth token restored")
                return
        except Exception as e:
            print(f"[fallback] recovery probe error: {e}")


@app.on_event("startup")
async def startup() -> None:
    load_token()


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.body()

    is_stream = False
    try:
        is_stream = json.loads(body).get("stream", False)
    except Exception:
        pass

    fallback = _fallback_mode
    headers = _build_upstream_headers(dict(_incoming_headers(request)), fallback)
    body = _force_haiku_model(body) if fallback else _inject_claude_code_system(body)
    proxy_headers = {"x-claude-proxy-fallback": "true"} if fallback else {}

    if is_stream:

        async def generate():
            saw_429 = False
            async with httpx.AsyncClient(timeout=600) as c:
                async with c.stream(
                    "POST",
                    f"{ANTHROPIC_BASE_URL}/v1/messages",
                    headers=headers,
                    content=body,
                ) as resp:
                    if resp.status_code == 429 and not fallback:
                        saw_429 = True
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            if saw_429:
                await _record_rate_limit()

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=proxy_headers
        )

    async with httpx.AsyncClient(timeout=120) as c:
        resp = await c.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages", headers=headers, content=body
        )

    if resp.status_code == 429 and not fallback:
        await _record_rate_limit()

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
        headers=proxy_headers,
    )


@app.get("/v1/models")
@app.get("/v1/models/{model_id}")
async def models(request: Request, model_id: str = ""):
    headers = _oauth_headers()
    headers["accept"] = "application/json"
    path = f"/v1/models/{model_id}" if model_id else "/v1/models"
    async with httpx.AsyncClient(timeout=30) as c:
        resp = await c.get(f"{ANTHROPIC_BASE_URL}{path}", headers=headers)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fallback": _fallback_mode,
        "rate_limit_events": len(_rate_limit_events),
    }
