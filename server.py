import asyncio
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

# ── OAuth constants (from Claude Code binary) ─────────────────────────────────
OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
KEYCHAIN_SERVICE = "Claude Code-credentials"
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"

# ── Anthropic API constants ───────────────────────────────────────────────────
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_CODE_VERSION = "2.1.148"

# Headers the client sends for auth — we strip and replace with ours.
_STRIP_HEADERS = {"x-api-key", "authorization"}

# ── Credential state ──────────────────────────────────────────────────────────
_cred: dict = {"access": None, "refresh": None, "expires": 0.0}
_refresh_lock: asyncio.Lock | None = None


def _get_refresh_lock() -> asyncio.Lock:
    global _refresh_lock
    if _refresh_lock is None:
        _refresh_lock = asyncio.Lock()
    return _refresh_lock


# ── Credential reading / writing ──────────────────────────────────────────────


def _parse_claude_oauth(raw: dict | None) -> dict | None:
    if not raw or not isinstance(raw, dict):
        return None
    access = raw.get("accessToken")
    if not isinstance(access, str) or not access:
        return None
    refresh = raw.get("refreshToken")
    expires_ms = raw.get("expiresAt", 0)
    return {
        "access": access,
        "refresh": refresh if isinstance(refresh, str) else None,
        "expires": expires_ms / 1000 if isinstance(expires_ms, (int, float)) else 0.0,
    }


def _read_keychain() -> dict | None:
    if platform.system() != "Darwin":
        return None
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        return _parse_claude_oauth(json.loads(r.stdout.strip()).get("claudeAiOauth"))
    except Exception:
        return None


def _read_file() -> dict | None:
    try:
        return _parse_claude_oauth(
            json.loads(CREDENTIALS_FILE.read_text()).get("claudeAiOauth")
        )
    except Exception:
        return None


def _write_keychain(cred: dict) -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return False
        existing = json.loads(r.stdout.strip())
        oauth = existing.get("claudeAiOauth", {})
        oauth["accessToken"] = cred["access"]
        if cred["refresh"]:
            oauth["refreshToken"] = cred["refresh"]
        oauth["expiresAt"] = int(cred["expires"] * 1000)
        existing["claudeAiOauth"] = oauth
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                "Claude Code",
                "-w",
                json.dumps(existing),
            ],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except Exception:
        return False


def _write_file(cred: dict) -> bool:
    try:
        if not CREDENTIALS_FILE.exists():
            return False
        data = json.loads(CREDENTIALS_FILE.read_text())
        oauth = data.get("claudeAiOauth", {})
        oauth["accessToken"] = cred["access"]
        if cred["refresh"]:
            oauth["refreshToken"] = cred["refresh"]
        oauth["expiresAt"] = int(cred["expires"] * 1000)
        data["claudeAiOauth"] = oauth
        CREDENTIALS_FILE.write_text(json.dumps(data))
        return True
    except Exception:
        return False


def _persist(cred: dict) -> None:
    if not _write_keychain(cred):
        _write_file(cred)


def load_credentials() -> None:
    global _cred
    token = os.environ.get("CLAUDE_TOKEN", "").strip()
    if token:
        _cred = {"access": token, "refresh": None, "expires": float("inf")}
        return
    cred = _read_keychain() or _read_file()
    if cred:
        _cred = cred
        return
    raise RuntimeError(
        "No Claude credentials found. "
        "Set CLAUDE_TOKEN, run 'claude auth login', or place credentials at "
        f"{CREDENTIALS_FILE}"
    )


# ── Token refresh ─────────────────────────────────────────────────────────────

REFRESH_BEFORE_SEC = 300  # refresh 5 min before expiry


def _is_fresh() -> bool:
    return bool(_cred.get("access")) and (
        _cred["expires"] == float("inf")
        or time.time() < _cred["expires"] - REFRESH_BEFORE_SEC
    )


async def _do_refresh() -> None:
    refresh_token = _cred.get("refresh")
    if not refresh_token:
        raise RuntimeError(
            "No refresh token — re-authenticate with 'claude auth login'"
        )
    async with httpx.AsyncClient(timeout=30) as c:
        resp = await c.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
            },
            headers={"content-type": "application/json"},
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"OAuth refresh failed: HTTP {resp.status_code} — {resp.text[:200]}"
        )
    data = resp.json()
    _cred["access"] = data["access_token"]
    if "refresh_token" in data:
        _cred["refresh"] = data["refresh_token"]
    _cred["expires"] = time.time() + float(data.get("expires_in", 3600))
    _persist(_cred)


async def get_token() -> str:
    if _is_fresh():
        return _cred["access"]  # type: ignore[return-value]
    async with _get_refresh_lock():
        if _is_fresh():
            return _cred["access"]  # type: ignore[return-value]
        # Another process may have refreshed — re-read before hitting the network.
        fresh = _read_keychain() or _read_file()
        if fresh and (
            fresh["expires"] == float("inf")
            or time.time() < fresh["expires"] - REFRESH_BEFORE_SEC
        ):
            _cred.update(fresh)
            return _cred["access"]  # type: ignore[return-value]
        await _do_refresh()
        return _cred["access"]  # type: ignore[return-value]


# ── Auth header construction ──────────────────────────────────────────────────


def _is_oauth(token: str) -> bool:
    return "sk-ant-oat" in token


def _auth_headers(token: str) -> dict[str, str]:
    betas = ["fine-grained-tool-streaming-2025-05-14"]
    if _is_oauth(token):
        return {
            "authorization": f"Bearer {token}",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": f"claude-code-20250219,oauth-2025-04-20,{','.join(betas)}",
            "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
            "x-app": "cli",
            "anthropic-dangerous-direct-browser-access": "true",
        }
    return {
        "x-api-key": token,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": ",".join(betas),
        "anthropic-dangerous-direct-browser-access": "true",
    }


def _build_upstream_headers(token: str, incoming: dict[str, str]) -> dict[str, str]:
    """Merge our auth headers with safe passthrough headers from the client."""
    headers = _auth_headers(token)

    # Let the client override anthropic-version.
    if "anthropic-version" in incoming:
        headers["anthropic-version"] = incoming["anthropic-version"]

    # Merge extra anthropic-beta values the client requests.
    if "anthropic-beta" in incoming:
        ours = set(headers["anthropic-beta"].split(","))
        theirs = set(incoming["anthropic-beta"].split(","))
        headers["anthropic-beta"] = ",".join(b for b in ours | theirs if b)

    # Always use our content-type and accept.
    headers["content-type"] = "application/json"
    headers["accept"] = "application/json, text/event-stream"

    return headers


# ── Proxy helpers ─────────────────────────────────────────────────────────────


def _incoming_headers(request: Request) -> dict[str, str]:
    """Client headers minus the auth ones we're replacing."""
    return {k: v for k, v in request.headers.items() if k.lower() not in _STRIP_HEADERS}


# ── Routes ────────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup() -> None:
    load_credentials()


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.body()

    is_stream = False
    try:
        is_stream = json.loads(body).get("stream", False)
    except Exception:
        pass

    token = await get_token()
    headers = _build_upstream_headers(token, dict(_incoming_headers(request)))

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
    token = await get_token()
    headers = _auth_headers(token)
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
    return {"status": "ok", "token_fresh": _is_fresh()}
