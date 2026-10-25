import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.background import BackgroundTask

from auth import PROVIDERS, Accounts, NotConnected, OAuthError, ReauthRequired
from codex import (
    CODEX_BASE_URL,
    CODEX_CLIENT_VERSION,
    chat_chunks,
    chat_to_responses,
    codex_headers,
    collect_chat_completion,
    prepare_responses_body,
    sse_events,
)

log = logging.getLogger("claude-proxy")

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_CODE_VERSION = "2.1.148"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
# Codex's OAuth client only accepts http://localhost:1455/auth/callback, so
# this port is fixed in practice. Claude's callback is served here too.
CALLBACK_PORT = int(os.environ.get("CALLBACK_PORT", "1455"))
PUBLIC_URL = os.environ.get("PUBLIC_URL", f"http://localhost:{PORT}").rstrip("/")
TOKEN_STORE = Path(os.environ.get("TOKEN_STORE", Path(__file__).parent / "data" / "tokens.json"))
ENABLED_PROVIDERS = [
    p.strip() for p in os.environ.get("PROVIDERS", "claude,openai").split(",") if p.strip()
]
# "auto" opens the setup page on startup when an account is missing, unless
# we are inside a container (there is no browser to open there).
OPEN_BROWSER = os.environ.get("OPEN_BROWSER", "auto").lower()

STATIC_DIR = Path(__file__).parent / "static"

_STRIP_HEADERS = {"x-api-key", "authorization"}

# Anthropic gates Pro-tier models behind this exact system prompt as the
# first system block — OAuth tokens only get haiku without it.
CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

# Server-side tool types are versioned like "web_search_20250305"; their
# matching anthropic-beta header is "web-search-2025-03-05". We derive it.
_SERVER_TOOL_TYPE_RE = re.compile(r"^([a-z_]+)_(\d{4})(\d{2})(\d{2})$")

accounts: Accounts | None = None
http: httpx.AsyncClient | None = None


def _setup_url() -> str:
    return f"{PUBLIC_URL}/"


def _should_open_browser() -> bool:
    if OPEN_BROWSER in ("0", "false", "no", "never"):
        return False
    if OPEN_BROWSER in ("1", "true", "yes", "always"):
        return True
    return not Path("/.dockerenv").exists()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global accounts, http
    if accounts is None:
        accounts = Accounts(
            TOKEN_STORE, ENABLED_PROVIDERS, os.environ.get("CLAUDE_TOKEN", "").strip()
        )
    if http is None:
        http = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=15))

    if accounts.setup_required():
        missing = [e["label"] for e in accounts.status().values() if not e["connected"]]
        log.warning(
            "Not signed in to %s. Open %s to connect your accounts.",
            " and ".join(missing),
            _setup_url(),
        )
        if _should_open_browser():
            asyncio.get_running_loop().call_later(1, webbrowser.open, _setup_url())
    yield
    await http.aclose()


app = FastAPI(lifespan=lifespan)


# ── errors ────────────────────────────────────────────────────────────────


def _anthropic_error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": kind, "message": message}}, status_code=status
    )


def _openai_error(status: int, message: str, kind: str = "upstream_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def _not_connected_message(name: str) -> str:
    return f"{PROVIDERS[name].label} account is not connected. Sign in at {_setup_url()}"


def _upstream_detail(raw: bytes) -> str:
    try:
        body = json.loads(raw)
    except Exception:
        return raw.decode(errors="replace")[:500]
    err = body.get("error")
    if isinstance(err, dict) and err.get("message"):
        return err["message"]
    return body.get("detail") or str(err or body)[:500]


def _relay(resp: httpx.Response) -> StreamingResponse:
    """Stream an upstream response through with its real status code."""
    return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
        background=BackgroundTask(resp.aclose),
    )


# ── Claude ────────────────────────────────────────────────────────────────


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
        "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
    }


def _build_upstream_headers(
    token: str, incoming: dict[str, str], extra_betas: set[str] = frozenset()
) -> dict[str, str]:
    headers = _auth_headers(token)

    if "anthropic-version" in incoming:
        headers["anthropic-version"] = incoming["anthropic-version"]

    ours = set(headers["anthropic-beta"].split(","))
    theirs: set[str] = set()
    if "anthropic-beta" in incoming:
        theirs = set(incoming["anthropic-beta"].split(","))
    merged = (ours | theirs | set(extra_betas)) - {""}
    headers["anthropic-beta"] = ",".join(sorted(merged))

    headers["content-type"] = "application/json"
    headers["accept"] = "application/json, text/event-stream"
    return headers


def _detect_tool_betas(body: bytes) -> set[str]:
    """Find server-side tools declared in the request and return the
    anthropic-beta headers Anthropic requires for them (web_search,
    code_execution, web_fetch, etc.)."""
    try:
        data = json.loads(body)
    except Exception:
        return set()
    tools = data.get("tools")
    if not isinstance(tools, list):
        return set()
    betas: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        t = tool.get("type")
        if not isinstance(t, str):
            continue
        m = _SERVER_TOOL_TYPE_RE.match(t)
        if m:
            name, y, mo, d = m.groups()
            betas.add(f"{name.replace('_', '-')}-{y}-{mo}-{d}")
    return betas


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


async def _send_with_refresh(provider: str, build) -> httpx.Response:
    """Send ``build(credentials)`` upstream, retrying once with a forced
    token refresh if the provider answers 401."""
    creds = await accounts.credentials(http, provider)
    resp = await http.send(build(creds), stream=True)
    if resp.status_code == 401 and creds.get("source") != "env":
        await resp.aclose()
        creds = await accounts.credentials(http, provider, force_refresh=True)
        resp = await http.send(build(creds), stream=True)
    return resp


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.body()
    incoming = _incoming_headers(request)
    tool_betas = _detect_tool_betas(body)
    body = _inject_claude_code_system(body)

    def build(creds: dict) -> httpx.Request:
        return http.build_request(
            "POST",
            f"{ANTHROPIC_BASE_URL}/v1/messages",
            headers=_build_upstream_headers(creds["access_token"], incoming, tool_betas),
            content=body,
        )

    try:
        resp = await _send_with_refresh("claude", build)
    except NotConnected:
        return _anthropic_error(401, "authentication_error", _not_connected_message("claude"))
    except ReauthRequired as e:
        return _anthropic_error(401, "authentication_error", f"{e} {_setup_url()}")
    except httpx.HTTPError as e:
        return _anthropic_error(502, "api_error", f"could not reach Anthropic: {e}")
    return _relay(resp)


@app.get("/v1/models")
@app.get("/v1/models/{model_id}")
async def models(model_id: str = ""):
    try:
        creds = await accounts.credentials(http, "claude")
    except NotConnected:
        return _anthropic_error(401, "authentication_error", _not_connected_message("claude"))
    except ReauthRequired as e:
        return _anthropic_error(401, "authentication_error", f"{e} {_setup_url()}")
    headers = _auth_headers(creds["access_token"])
    headers["accept"] = "application/json"
    path = f"/v1/models/{model_id}" if model_id else "/v1/models"
    resp = await http.get(f"{ANTHROPIC_BASE_URL}{path}", headers=headers, timeout=30)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ── ChatGPT (OpenAI-compatible, under /openai) ─────────────────────────────


async def _open_codex(payload: dict) -> httpx.Response | JSONResponse:
    def build(creds: dict) -> httpx.Request:
        return http.build_request(
            "POST",
            f"{CODEX_BASE_URL}/responses",
            headers=codex_headers(creds["access_token"], creds["account_id"]),
            json=payload,
        )

    try:
        resp = await _send_with_refresh("openai", build)
    except NotConnected:
        return _openai_error(401, _not_connected_message("openai"), "authentication_error")
    except ReauthRequired as e:
        return _openai_error(401, f"{e} {_setup_url()}", "authentication_error")
    except httpx.HTTPError as e:
        return _openai_error(502, f"could not reach ChatGPT: {e}")

    if resp.status_code != 200:
        raw = await resp.aread()
        await resp.aclose()
        return _openai_error(resp.status_code, _upstream_detail(raw))
    return resp


@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    resp = await _open_codex(chat_to_responses(body))
    if isinstance(resp, JSONResponse):
        return resp

    chunks = chat_chunks(sse_events(resp.aiter_lines()), body.get("model", ""))

    if body.get("stream"):
        include_usage = (body.get("stream_options") or {}).get("include_usage")

        async def generate():
            async for c in chunks:
                if not include_usage:
                    c.pop("usage", None)
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(), media_type="text/event-stream", background=BackgroundTask(resp.aclose)
        )

    try:
        return JSONResponse(await collect_chat_completion(chunks))
    except RuntimeError as e:
        return _openai_error(502, str(e))
    finally:
        await resp.aclose()


@app.post("/openai/v1/responses")
async def responses(request: Request):
    body = await request.json()
    resp = await _open_codex(prepare_responses_body(body))
    if isinstance(resp, JSONResponse):
        return resp
    if body.get("stream"):
        return _relay(resp)

    # The backend only streams, so fold the events into the final object.
    try:
        output = []
        async for event in sse_events(resp.aiter_lines()):
            kind = event.get("type")
            if kind == "response.output_item.done":
                output.append(event.get("item"))
            elif kind in ("response.completed", "response.incomplete"):
                result = event.get("response") or {}
                if not result.get("output"):
                    result["output"] = output
                return JSONResponse(result)
            elif kind in ("response.failed", "error"):
                err = event.get("error") or (event.get("response") or {}).get("error") or {}
                return _openai_error(502, err.get("message") or "the model returned an error")
        return _openai_error(502, "stream ended before the response completed")
    finally:
        await resp.aclose()


@app.get("/openai/v1/models")
async def openai_models():
    try:
        creds = await accounts.credentials(http, "openai")
    except NotConnected:
        return _openai_error(401, _not_connected_message("openai"), "authentication_error")
    except ReauthRequired as e:
        return _openai_error(401, f"{e} {_setup_url()}", "authentication_error")
    headers = codex_headers(creds["access_token"], creds["account_id"])
    headers["accept"] = "application/json"
    resp = await http.get(
        f"{CODEX_BASE_URL}/models",
        params={"client_version": CODEX_CLIENT_VERSION},
        headers=headers,
        timeout=30,
    )
    if resp.status_code != 200:
        return _openai_error(resp.status_code, _upstream_detail(resp.content))
    return {
        "object": "list",
        "data": [
            {"id": m["slug"], "object": "model", "created": 0, "owned_by": "openai"}
            for m in resp.json().get("models", [])
            if m.get("visibility") != "hide"
        ],
    }


# ── setup page & sign-in ──────────────────────────────────────────────────


def _require_json(request: Request) -> None:
    """State-changing setup calls must be JSON so a cross-site form post
    can't trigger them (browsers preflight JSON, and we allow no CORS)."""
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(415, "expected application/json")


def _status() -> dict:
    return {
        "providers": accounts.status(),
        "setup_required": accounts.setup_required(),
        "setup_url": _setup_url(),
    }


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status():
    return _status()


@app.get("/login/{provider}")
async def login(provider: str):
    if provider not in accounts.enabled:
        raise HTTPException(404, "unknown provider")
    redirect_uri = f"http://localhost:{CALLBACK_PORT}{PROVIDERS[provider].callback_path}"
    return RedirectResponse(accounts.begin_login(provider, redirect_uri))


async def _finish_login(request: Request, expected: str):
    params = request.query_params
    if params.get("error"):
        message = params.get("error_description") or params["error"]
        return RedirectResponse(f"{PUBLIC_URL}/?error={quote(message)}")
    try:
        name = await accounts.complete_login(
            http, params.get("state", ""), params.get("code", ""), expected
        )
    except (OAuthError, httpx.HTTPError) as e:
        return RedirectResponse(f"{PUBLIC_URL}/?error={quote(str(e))}")
    return RedirectResponse(f"{PUBLIC_URL}/?connected={name}")


@app.get("/callback")
async def claude_callback(request: Request):
    return await _finish_login(request, "claude")


@app.get("/auth/callback")
async def openai_callback(request: Request):
    return await _finish_login(request, "openai")


@app.post("/api/login/claude/manual")
async def claude_manual_start(request: Request):
    """Start a Claude login that ends on Anthropic's "copy this code" page,
    for when the browser can't reach the callback port."""
    _require_json(request)
    if "claude" not in accounts.enabled:
        raise HTTPException(404, "unknown provider")
    return {"url": accounts.begin_login("claude", PROVIDERS["claude"].manual_redirect_uri)}


@app.post("/api/login/claude/code")
async def claude_manual_finish(request: Request):
    _require_json(request)
    pasted = str((await request.json()).get("code", "")).strip()
    code, _, state = pasted.partition("#")
    if not code or not state:
        return _openai_error(400, "paste the whole code, including the part after '#'", "invalid_request_error")
    try:
        await accounts.complete_login(http, state, code, "claude")
    except (OAuthError, httpx.HTTPError) as e:
        return _openai_error(400, str(e), "invalid_request_error")
    return _status()


@app.post("/api/logout/{provider}")
async def logout(provider: str, request: Request):
    _require_json(request)
    if provider not in accounts.enabled:
        raise HTTPException(404, "unknown provider")
    accounts.logout(provider)
    return _status()


@app.get("/health")
async def health():
    return {"status": "ok", **_status()}


# ── entry point ───────────────────────────────────────────────────────────


class _Server(uvicorn.Server):
    # Two servers share one event loop; signals are wired up once below
    # instead of each server overwriting the other's handlers.
    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def _serve() -> None:
    main = _Server(uvicorn.Config(app, host=HOST, port=PORT))
    servers = [main]
    if CALLBACK_PORT != PORT:
        servers.append(
            _Server(uvicorn.Config(app, host=HOST, port=CALLBACK_PORT, lifespan="off"))
        )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: [setattr(s, "should_exit", True) for s in servers])
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    asyncio.run(_serve())
