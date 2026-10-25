"""OAuth sign-in and token storage for the Claude and ChatGPT accounts.

Both providers use OAuth 2.0 authorization code + PKCE with the public client
IDs of their official CLIs (Claude Code and Codex), so the redirect URIs we
can use are the ones those clients have registered:

- Claude Code accepts ``http://localhost:<any port>/callback`` and a manual
  "copy this code" page for headless machines.
- Codex only accepts ``http://localhost:1455/auth/callback``.

Tokens are persisted to a JSON file (mode 0600) and refreshed shortly before
they expire. Refresh tokens rotate, so every refresh is written back.
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

# Refresh this many seconds before the access token actually expires.
REFRESH_MARGIN = 300
# Abandon half-finished logins after this long.
PENDING_TTL = 600


class NotConnected(Exception):
    """No credentials are stored for the provider."""


class ReauthRequired(Exception):
    """Stored credentials were rejected and the user must sign in again."""


class OAuthError(Exception):
    """The provider rejected a token exchange or refresh."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def pkce_pair() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def jwt_claims(token: str) -> dict:
    """Decode a JWT payload without verifying it. We only read our own
    tokens straight from the provider's token endpoint, so the claims are
    used for display and expiry, never for trust decisions."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _expires_at(data: dict, access_token: str) -> float:
    if isinstance(data.get("expires_in"), (int, float)):
        return time.time() + data["expires_in"]
    exp = jwt_claims(access_token).get("exp")
    if isinstance(exp, (int, float)):
        return float(exp)
    # Unknown lifetime: assume an hour so we refresh conservatively.
    return time.time() + 3600


async def _post_token(http: httpx.AsyncClient, url: str, **kwargs) -> dict:
    resp = await http.post(url, timeout=30, **kwargs)
    if resp.status_code != 200:
        try:
            body = resp.json()
        except Exception:
            body = {}
        err = body.get("error") if isinstance(body, dict) else None
        detail = (
            (body.get("error_description") if isinstance(body, dict) else None)
            or (err.get("message") if isinstance(err, dict) else err)
            or resp.text[:200]
        )
        raise OAuthError(f"{resp.status_code}: {detail or 'token request failed'}")
    return resp.json()


class ClaudeOAuth:
    name = "claude"
    label = "Claude"
    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    authorize_endpoint = "https://claude.com/cai/oauth/authorize"
    token_endpoint = "https://platform.claude.com/v1/oauth/token"
    manual_redirect_uri = "https://platform.claude.com/oauth/code/callback"
    callback_path = "/callback"
    # Same scope sets Claude Code requests for a claude.ai login and refresh.
    scopes = [
        "org:create_api_key",
        "user:profile",
        "user:inference",
        "user:sessions:claude_code",
        "user:mcp_servers",
        "user:file_upload",
    ]
    refresh_scopes = [s for s in scopes if s != "org:create_api_key"]

    def authorize_url(self, redirect_uri: str, challenge: str, state: str) -> str:
        return f"{self.authorize_endpoint}?" + urlencode(
            {
                "code": "true",
                "client_id": self.client_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": " ".join(self.scopes),
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
            }
        )

    async def exchange(
        self, http: httpx.AsyncClient, code: str, verifier: str, redirect_uri: str, state: str
    ) -> dict:
        data = await _post_token(
            http,
            self.token_endpoint,
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "code_verifier": verifier,
                "state": state,
            },
        )
        account = data.get("account") or {}
        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_at": _expires_at(data, data["access_token"]),
            "email": account.get("email_address") or account.get("email"),
        }

    async def refresh(self, http: httpx.AsyncClient, record: dict) -> dict:
        data = await _post_token(
            http,
            self.token_endpoint,
            json={
                "grant_type": "refresh_token",
                "refresh_token": record["refresh_token"],
                "client_id": self.client_id,
                "scope": " ".join(self.refresh_scopes),
            },
        )
        return {
            **record,
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token") or record["refresh_token"],
            "expires_at": _expires_at(data, data["access_token"]),
        }


class OpenAIOAuth:
    name = "openai"
    label = "ChatGPT"
    client_id = "app_EMoamEEZ73f0CkXaXp7hrann"
    authorize_endpoint = "https://auth.openai.com/oauth/authorize"
    token_endpoint = "https://auth.openai.com/oauth/token"
    manual_redirect_uri = None
    callback_path = "/auth/callback"
    scopes = [
        "openid",
        "profile",
        "email",
        "offline_access",
        "api.connectors.read",
        "api.connectors.invoke",
    ]

    def authorize_url(self, redirect_uri: str, challenge: str, state: str) -> str:
        return f"{self.authorize_endpoint}?" + urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "scope": " ".join(self.scopes),
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
                "state": state,
                "originator": "codex_cli_rs",
            }
        )

    @staticmethod
    def _identity(id_token: str) -> dict:
        claims = jwt_claims(id_token)
        auth = claims.get("https://api.openai.com/auth") or {}
        return {
            "email": claims.get("email"),
            "account_id": auth.get("chatgpt_account_id"),
            "plan": auth.get("chatgpt_plan_type"),
        }

    async def exchange(
        self, http: httpx.AsyncClient, code: str, verifier: str, redirect_uri: str, state: str
    ) -> dict:
        data = await _post_token(
            http,
            self.token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "code_verifier": verifier,
            },
        )
        identity = self._identity(data.get("id_token", ""))
        if not identity["account_id"]:
            raise OAuthError("sign-in succeeded but no ChatGPT account id was returned")
        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "id_token": data.get("id_token"),
            "expires_at": _expires_at(data, data["access_token"]),
            **identity,
        }

    async def refresh(self, http: httpx.AsyncClient, record: dict) -> dict:
        data = await _post_token(
            http,
            self.token_endpoint,
            json={
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": record["refresh_token"],
                "scope": "openid profile email",
            },
        )
        updated = {
            **record,
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token") or record["refresh_token"],
            "expires_at": _expires_at(data, data["access_token"]),
        }
        if data.get("id_token"):
            updated["id_token"] = data["id_token"]
            identity = self._identity(data["id_token"])
            updated.update({k: v for k, v in identity.items() if v})
        return updated


PROVIDERS = {p.name: p for p in (ClaudeOAuth(), OpenAIOAuth())}


class Accounts:
    """Stored credentials for every provider, plus in-flight logins."""

    def __init__(self, path: Path, enabled: list[str], claude_env_token: str = ""):
        self.path = path
        self.enabled = [name for name in enabled if name in PROVIDERS]
        self.claude_env_token = claude_env_token
        self._records: dict[str, dict] = {}
        self._pending: dict[str, dict] = {}
        self._locks = {name: asyncio.Lock() for name in PROVIDERS}
        self._load()

    # ── persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            self._records = json.loads(self.path.read_text())
        except FileNotFoundError:
            self._records = {}
        except Exception as e:
            raise RuntimeError(f"could not read token store {self.path}: {e}") from e

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self._records, f, indent=2)
        os.replace(tmp, self.path)

    # ── login flow ────────────────────────────────────────────────────────

    def begin_login(self, name: str, redirect_uri: str) -> str:
        provider = PROVIDERS[name]
        now = time.time()
        self._pending = {
            s: p for s, p in self._pending.items() if now - p["created"] < PENDING_TTL
        }
        verifier, challenge = pkce_pair()
        state = _b64url(secrets.token_bytes(24))
        self._pending[state] = {
            "provider": name,
            "verifier": verifier,
            "redirect_uri": redirect_uri,
            "created": now,
        }
        return provider.authorize_url(redirect_uri, challenge, state)

    async def complete_login(
        self, http: httpx.AsyncClient, state: str, code: str, expected: str | None = None
    ) -> str:
        pending = self._pending.pop(state, None)
        if not pending or time.time() - pending["created"] > PENDING_TTL:
            raise OAuthError("this sign-in link expired or was already used; start again")
        name = pending["provider"]
        if expected and name != expected:
            raise OAuthError("sign-in callback does not match the provider that started it")
        record = await PROVIDERS[name].exchange(
            http, code, pending["verifier"], pending["redirect_uri"], state
        )
        record["connected_at"] = time.time()
        self._records[name] = record
        self._save()
        return name

    def logout(self, name: str) -> None:
        if self._records.pop(name, None) is not None:
            self._save()

    # ── credentials ───────────────────────────────────────────────────────

    async def credentials(
        self, http: httpx.AsyncClient, name: str, force_refresh: bool = False
    ) -> dict:
        """Return a usable record for ``name``, refreshing it if needed."""
        async with self._locks[name]:
            record = self._records.get(name)
            if record is None:
                if name == "claude" and self.claude_env_token:
                    return {"access_token": self.claude_env_token, "source": "env"}
                raise NotConnected(name)
            if record.get("error"):
                raise ReauthRequired(record["error"])

            stale = record.get("expires_at", 0) - REFRESH_MARGIN < time.time()
            if (stale or force_refresh) and record.get("refresh_token"):
                try:
                    record = await PROVIDERS[name].refresh(http, record)
                except OAuthError as e:
                    # A rejected refresh token never recovers; make the UI
                    # say so instead of failing every request silently.
                    record = {**record, "error": f"Sign-in expired ({e}). Reconnect."}
                    self._records[name] = record
                    self._save()
                    raise ReauthRequired(record["error"]) from e
                self._records[name] = record
                self._save()
            return record

    def status(self) -> dict:
        out = {}
        for name in self.enabled:
            provider = PROVIDERS[name]
            record = self._records.get(name)
            entry = {"label": provider.label, "connected": False}
            if record:
                entry.update(
                    connected=not record.get("error"),
                    source="oauth",
                    email=record.get("email"),
                    plan=record.get("plan"),
                    error=record.get("error"),
                )
            elif name == "claude" and self.claude_env_token:
                entry.update(connected=True, source="env")
            out[name] = entry
        return out

    def setup_required(self) -> bool:
        return any(not entry["connected"] for entry in self.status().values())
