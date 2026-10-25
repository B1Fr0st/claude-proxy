"""Translation between OpenAI-style requests and the ChatGPT Codex backend.

The Codex backend (``chatgpt.com/backend-api/codex/responses``) speaks the
Responses API with a few restrictions observed against the live service:

- ``stream`` must be true and ``store`` must be false.
- ``temperature`` (and other sampling knobs) are rejected outright.
- Only the models listed by ``/codex/models`` are accepted.

Chat Completions callers are translated to Responses and the SSE stream is
translated back into ``chat.completion.chunk`` events.
"""

import json
import time
import uuid
from typing import AsyncIterator

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_CLIENT_VERSION = "0.164.0"

# Sampling parameters the backend rejects with "Unsupported parameter".
_UNSUPPORTED = ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed")


def codex_headers(access_token: str, account_id: str) -> dict[str, str]:
    return {
        "authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "originator": "codex_cli_rs",
        "user-agent": f"codex_cli_rs/{CODEX_CLIENT_VERSION}",
        "session_id": str(uuid.uuid4()),
        "content-type": "application/json",
        "accept": "text/event-stream",
    }


def prepare_responses_body(body: dict) -> dict:
    """Coerce a client's Responses request into what the backend accepts."""
    out = {k: v for k, v in body.items() if k not in _UNSUPPORTED}
    out["stream"] = True
    out["store"] = False
    if isinstance(out.get("input"), str):
        out["input"] = [
            {"role": "user", "content": [{"type": "input_text", "text": out["input"]}]}
        ]
    return out


# ── chat completions → responses ──────────────────────────────────────────


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
    )


def _user_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts = []
    for part in content or []:
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            converted = {"type": "input_image", "image_url": url}
            if isinstance(image, dict) and image.get("detail"):
                converted["detail"] = image["detail"]
            parts.append(converted)
    return parts


def chat_to_responses(body: dict) -> dict:
    instructions: list[str] = []
    items: list[dict] = []

    for message in body.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            instructions.append(_text_of(content))
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": _text_of(content),
                }
            )
        elif role == "assistant":
            text = _text_of(content)
            if text:
                items.append(
                    {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
                )
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
        else:
            items.append({"role": "user", "content": _user_parts(content)})

    out: dict = {"model": body.get("model"), "input": items, "store": False, "stream": True}
    joined = "\n\n".join(i for i in instructions if i)
    if joined:
        out["instructions"] = joined

    tools = [
        {
            "type": "function",
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in body.get("tools") or []
        if t.get("type") == "function"
    ]
    if tools:
        out["tools"] = tools
        choice = body.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            out["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
        elif choice:
            out["tool_choice"] = choice
        if "parallel_tool_calls" in body:
            out["parallel_tool_calls"] = body["parallel_tool_calls"]

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_tokens:
        out["max_output_tokens"] = max_tokens
    if body.get("reasoning_effort"):
        out["reasoning"] = {"effort": body["reasoning_effort"]}
    return out


# ── responses SSE → chat completion chunks ────────────────────────────────


async def sse_events(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    async for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _error_message(event: dict) -> str:
    err = event.get("error") or (event.get("response") or {}).get("error") or {}
    if isinstance(err, dict):
        return err.get("message") or "the model returned an error"
    return str(err)


def _usage(response: dict) -> dict | None:
    usage = response.get("usage")
    if not usage:
        return None
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


async def chat_chunks(events: AsyncIterator[dict], model: str) -> AsyncIterator[dict]:
    """Yield ``chat.completion.chunk`` objects (and a final ``{"error": …}``
    object if the response fails mid-stream)."""
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }

    def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
        out = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            out["usage"] = usage
        return out

    yield chunk({"role": "assistant", "content": ""})
    tool_index: dict[str, int] = {}

    async for event in events:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            yield chunk({"content": event.get("delta", "")})
        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                tool_index[item.get("id")] = len(tool_index)
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_index[item.get("id")],
                                "id": item.get("call_id"),
                                "type": "function",
                                "function": {"name": item.get("name"), "arguments": ""},
                            }
                        ]
                    }
                )
        elif kind == "response.function_call_arguments.delta":
            index = tool_index.get(event.get("item_id"))
            if index is not None:
                yield chunk(
                    {"tool_calls": [{"index": index, "function": {"arguments": event.get("delta", "")}}]}
                )
        elif kind == "response.completed":
            finish = "tool_calls" if tool_index else "stop"
            yield chunk({}, finish, _usage(event.get("response") or {}))
            return
        elif kind == "response.incomplete":
            yield chunk({}, "length", _usage(event.get("response") or {}))
            return
        elif kind in ("response.failed", "error"):
            yield {"error": {"message": _error_message(event), "type": "upstream_error"}}
            return


async def collect_chat_completion(chunks: AsyncIterator[dict]) -> dict:
    """Fold a chunk stream into a single non-streaming chat completion.
    Raises ``RuntimeError`` if the stream reported an error."""
    content: list[str] = []
    calls: dict[int, dict] = {}
    finish = None
    usage = None
    first = None

    async for c in chunks:
        if "error" in c:
            raise RuntimeError(c["error"]["message"])
        first = first or c
        choice = c["choices"][0]
        delta = choice["delta"]
        content.append(delta.get("content") or "")
        for call in delta.get("tool_calls") or []:
            slot = calls.setdefault(
                call["index"],
                {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
            )
            if call.get("id"):
                slot["id"] = call["id"]
            fn = call.get("function", {})
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            slot["function"]["arguments"] += fn.get("arguments", "")
        finish = choice["finish_reason"] or finish
        usage = c.get("usage") or usage

    message: dict = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    out = {
        "id": first["id"] if first else f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": first["created"] if first else int(time.time()),
        "model": first["model"] if first else "",
        "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}],
    }
    if usage:
        out["usage"] = usage
    return out
