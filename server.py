import json
import os

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

_token: str = ""


def load_token() -> None:
    global _token
    _token = os.environ.get("CLAUDE_TOKEN", "").strip()
    if not _token:
        raise RuntimeError(
            "CLAUDE_TOKEN not set. Generate one with 'claude setup-token'."
        )


def _auth_headers() -> dict[str, str]:
    return {
        "authorization": f"Bearer {_token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
        "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
    }


def _build_upstream_headers(incoming: dict[str, str]) -> dict[str, str]:
    headers = _auth_headers()

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

    headers = _build_upstream_headers(dict(_incoming_headers(request)))
    body = _inject_claude_code_system(body)

    if is_stream:

        async def generate():
            async with httpx.AsyncClient(timeout=600) as c:
                async with c.stream(
                    "POST",
                    f"{ANTHROPIC_BASE_URL}/v1/messages",
                    headers=headers,
                    content=body,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(generate(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as c:
        resp = await c.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages", headers=headers, content=body
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/v1/models")
@app.get("/v1/models/{model_id}")
async def models(request: Request, model_id: str = ""):
    headers = _auth_headers()
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
    return {"status": "ok"}
