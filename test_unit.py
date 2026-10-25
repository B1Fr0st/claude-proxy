"""Offline tests: upstream OAuth and model endpoints are mocked.

    pip install -r requirements.txt -r requirements-dev.txt
    pytest test_unit.py
"""

import base64
import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

import server
from auth import Accounts, ReauthRequired
from codex import chat_to_responses


def _jwt(claims: dict) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{enc({'alg': 'none'})}.{enc(claims)}.sig"


ID_TOKEN = _jwt(
    {
        "email": "me@example.com",
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_1", "chatgpt_plan_type": "pro"},
    }
)


def _sse(*events: dict) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)


class Upstream:
    """A fake for every upstream the proxy talks to."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.messages_status = [200]
        self.refresh_ok = True
        self.codex_events = [
            {"type": "response.output_text.delta", "delta": "hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}},
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url == "https://platform.claude.com/v1/oauth/token":
            body = json.loads(request.content)
            if body["grant_type"] == "refresh_token" and not self.refresh_ok:
                return httpx.Response(400, json={"error": "invalid_grant", "error_description": "revoked"})
            return httpx.Response(
                200,
                json={
                    "access_token": f"claude-{body['grant_type']}",
                    "refresh_token": "claude-rt-2",
                    "expires_in": 28800,
                    "account": {"email_address": "me@example.com"},
                },
            )
        if url == "https://auth.openai.com/oauth/token":
            return httpx.Response(
                200,
                json={"access_token": "oa-at", "refresh_token": "oa-rt", "id_token": ID_TOKEN, "expires_in": 3600},
            )
        if url == "https://api.anthropic.com/v1/messages":
            status = self.messages_status.pop(0) if self.messages_status else 200
            return httpx.Response(status, json={"id": "msg_1", "content": [{"type": "text", "text": "ok"}]})
        if url.startswith("https://chatgpt.com/backend-api/codex/responses"):
            return httpx.Response(200, content=_sse(*self.codex_events), headers={"content-type": "text/event-stream"})
        if url.startswith("https://chatgpt.com/backend-api/codex/models"):
            return httpx.Response(200, json={"models": [{"slug": "gpt-a", "visibility": "list"}, {"slug": "hidden", "visibility": "hide"}]})
        return httpx.Response(404)


@pytest.fixture
def upstream():
    return Upstream()


@pytest.fixture
def client(tmp_path, upstream, monkeypatch):
    monkeypatch.setattr(server, "accounts", Accounts(tmp_path / "tokens.json", ["claude", "openai"]))
    monkeypatch.setattr(server, "http", httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    monkeypatch.setattr(server, "OPEN_BROWSER", "never")
    with TestClient(server.app) as c:
        yield c


def _login(client: TestClient, provider: str) -> None:
    r = client.get(f"/login/{provider}", follow_redirects=False)
    assert r.status_code == 307
    query = parse_qs(urlparse(r.headers["location"]).query)
    callback = server.PROVIDERS[provider].callback_path
    r = client.get(callback, params={"code": "abc", "state": query["state"][0]}, follow_redirects=False)
    assert r.headers["location"].endswith(f"/?connected={provider}")


def test_status_requires_setup_until_both_connected(client):
    assert client.get("/health").json()["setup_required"] is True
    _login(client, "claude")
    _login(client, "openai")
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["setup_required"] is False
    assert health["providers"]["openai"]["email"] == "me@example.com"
    assert health["providers"]["openai"]["plan"] == "pro"


def test_login_uses_pkce_and_registered_redirects(client):
    r = client.get("/login/openai", follow_redirects=False)
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert q["code_challenge_method"] == ["S256"]
    r = client.get("/login/claude", follow_redirects=False)
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["redirect_uri"] == ["http://localhost:1455/callback"]
    assert "user:inference" in q["scope"][0]


def test_callback_rejects_unknown_state(client):
    r = client.get("/auth/callback", params={"code": "x", "state": "nope"}, follow_redirects=False)
    assert "error=" in r.headers["location"]


def test_tokens_persist_with_private_permissions(client, tmp_path):
    _login(client, "openai")
    path = tmp_path / "tokens.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["openai"]["account_id"] == "acct_1"


def test_messages_401_when_not_connected(client):
    r = client.post("/v1/messages", json={"model": "m", "messages": []})
    assert r.status_code == 401
    assert "Sign in at" in r.json()["error"]["message"]


def test_messages_uses_oauth_token_and_injects_system(client, upstream):
    _login(client, "claude")
    r = client.post("/v1/messages", json={"model": "m", "system": "be brief", "messages": []})
    assert r.status_code == 200
    sent = upstream.requests[-1]
    assert sent.headers["authorization"] == "Bearer claude-authorization_code"
    assert json.loads(sent.content)["system"][0]["text"] == server.CLAUDE_CODE_SYSTEM


def test_messages_refreshes_once_on_401(client, upstream):
    _login(client, "claude")
    upstream.messages_status = [401, 200]
    r = client.post("/v1/messages", json={"model": "m", "messages": []})
    assert r.status_code == 200
    assert upstream.requests[-1].headers["authorization"] == "Bearer claude-refresh_token"


def test_upstream_error_status_is_preserved_when_streaming(client, upstream):
    _login(client, "claude")
    upstream.messages_status = [429]
    r = client.post("/v1/messages", json={"model": "m", "stream": True, "messages": []})
    assert r.status_code == 429


def test_env_token_fallback(tmp_path):
    accounts = Accounts(tmp_path / "t.json", ["claude"], claude_env_token="sk-ant-oat01-x")
    assert accounts.status()["claude"] == {"label": "Claude", "connected": True, "source": "env"}
    assert accounts.setup_required() is False


@pytest.mark.anyio
async def test_rejected_refresh_requires_reauth(tmp_path, upstream):
    accounts = Accounts(tmp_path / "t.json", ["claude"])
    accounts._records["claude"] = {"access_token": "old", "refresh_token": "rt", "expires_at": time.time() - 1}
    upstream.refresh_ok = False
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        with pytest.raises(ReauthRequired):
            await accounts.credentials(http, "claude")
    assert accounts.status()["claude"]["connected"] is False
    assert "Reconnect" in accounts.status()["claude"]["error"]


def test_chat_to_responses_translation():
    out = chat_to_responses(
        {
            "model": "gpt-a",
            "temperature": 0.2,
            "max_tokens": 50,
            "messages": [
                {"role": "system", "content": "be terse"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "low"}},
                        {"type": "text", "text": "what is this"},
                    ],
                },
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "42"},
            ],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        }
    )
    assert out["instructions"] == "be terse"
    assert "temperature" not in out
    assert out["max_output_tokens"] == 50
    assert out["stream"] is True and out["store"] is False
    assert out["input"][0]["content"][0] == {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "low"}
    assert out["input"][1] == {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
    assert out["input"][2] == {"type": "function_call_output", "call_id": "c1", "output": "42"}
    assert out["tools"][0]["name"] == "f"


def test_chat_completions_stream(client, upstream):
    _login(client, "openai")
    r = client.post(
        "/openai/v1/chat/completions",
        json={"model": "gpt-a", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    lines = [l[6:] for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(l) for l in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == "hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    sent = next(r for r in upstream.requests if "codex/responses" in str(r.url))
    assert sent.headers["chatgpt-account-id"] == "acct_1"


def test_chat_completions_tool_call_non_stream(client, upstream):
    _login(client, "openai")
    upstream.codex_events = [
        {"type": "response.output_item.added", "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_time"}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"tz":'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '"UTC"}'},
        {"type": "response.completed", "response": {}},
    ]
    r = client.post("/openai/v1/chat/completions", json={"model": "gpt-a", "messages": [{"role": "user", "content": "time?"}]})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "get_time", "arguments": '{"tz":"UTC"}'}


def test_chat_completions_reports_midstream_failure(client, upstream):
    _login(client, "openai")
    upstream.codex_events = [{"type": "response.failed", "response": {"error": {"message": "quota exceeded"}}}]
    r = client.post("/openai/v1/chat/completions", json={"model": "gpt-a", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert r.json()["error"]["message"] == "quota exceeded"


def test_openai_models_hides_hidden(client):
    _login(client, "openai")
    assert [m["id"] for m in client.get("/openai/v1/models").json()["data"]] == ["gpt-a"]


def test_logout_requires_json(client):
    _login(client, "openai")
    assert client.post("/api/logout/openai", data={"x": "1"}).status_code == 415
    assert client.post("/api/logout/openai", json={}).json()["providers"]["openai"]["connected"] is False


def test_manual_claude_code_flow(client):
    url = client.post("/api/login/claude/manual", json={}).json()["url"]
    q = parse_qs(urlparse(url).query)
    assert q["redirect_uri"] == ["https://platform.claude.com/oauth/code/callback"]
    r = client.post("/api/login/claude/code", json={"code": f"abc#{q['state'][0]}"})
    assert r.json()["providers"]["claude"]["connected"] is True
