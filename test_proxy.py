"""
Smoke tests for the Anthropic API proxy.

Usage:
    pip install anthropic httpx
    uvicorn server:app --port 8000   # in another terminal
    python test_proxy.py
"""

import sys

import anthropic
import httpx

BASE_URL = "http://localhost:8000"
MODEL = "claude-haiku-4-5"

client = anthropic.Anthropic(base_url=BASE_URL, api_key="unused")

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def run(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        return False


# ── tests ─────────────────────────────────────────────────────────────────────


def test_health():
    r = httpx.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "token_fresh" in r.json()


def test_models():
    models = client.models.list()
    ids = {m.id for m in models.data}
    assert any("claude" in mid for mid in ids), f"no claude models in {ids}"


def test_completion():
    msg = client.messages.create(
        model=MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    )
    assert msg.content, "no content blocks"
    text = msg.content[0].text
    assert text.strip(), "empty response"
    assert msg.usage.input_tokens > 0
    assert msg.usage.output_tokens > 0
    assert msg.stop_reason == "end_turn"


def test_system_prompt():
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16,
        system="You are a calculator. Output only the numeric answer.",
        messages=[{"role": "user", "content": "2 + 2"}],
    )
    assert "4" in msg.content[0].text


def test_streaming():
    chunks = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "Count to 3, one number per line."}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)

    full = "".join(chunks)
    assert full.strip(), "empty streamed response"
    assert "1" in full and "2" in full and "3" in full


def test_multi_turn():
    msg = client.messages.create(
        model=MODEL,
        max_tokens=32,
        messages=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ],
    )
    assert "Alice" in msg.content[0].text


def test_haiku():
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=16,
        messages=[{"role": "user", "content": "Say hi."}],
    )
    assert msg.content[0].text.strip()


# ── runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ("health endpoint", test_health),
    ("model listing", test_models),
    ("non-streaming completion", test_completion),
    ("system prompt", test_system_prompt),
    ("streaming completion", test_streaming),
    ("multi-turn conversation", test_multi_turn),
    ("haiku model", test_haiku),
]

if __name__ == "__main__":
    print(f"\nTesting proxy at {BASE_URL}  (model: {MODEL})\n")
    results = [run(name, fn) for name, fn in TESTS]
    total, passed = len(results), sum(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
