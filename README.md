# claude-proxy

A FastAPI proxy that turns your Claude and ChatGPT subscriptions into local API endpoints:

- **Anthropic-compatible** at `http://localhost:8000` (`/v1/messages`, `/v1/models`), backed by a Claude Pro/Max login.
- **OpenAI-compatible** at `http://localhost:8000/openai/v1` (`/chat/completions`, `/responses`, `/models`), backed by a ChatGPT Plus/Pro login through the Codex backend.

Is this probably against TOS? Yeah.
Is this gonna get you banned? Maybe!

## Run

```bash
./start.sh
```

This builds and starts the container, waits for it to be healthy, and opens
<http://localhost:8000/> if an account still needs connecting. Click **Sign in**
for each provider; tokens are saved in the `tokens` Docker volume and refreshed
automatically.

Or bare (opens the sign-in page itself on first start):

```bash
pip install -r requirements.txt   # Python 3.10+
python server.py
```

Run it with `python server.py`, not `uvicorn server:app`: the server also listens on
port **1455**, the only OAuth callback Codex's sign-in accepts. Port 1455 must be free
while you sign in (a `codex login` in progress uses it too).

### Headless / remote machines

The OAuth callbacks redirect to `localhost`, so sign in from a browser on the same
machine as the proxy, or forward the ports (`ssh -L 8000:localhost:8000 -L 1455:localhost:1455 host`).
For Claude, the page also offers **"Sign in with a code instead"**, which needs no callback.

### Legacy token

`CLAUDE_TOKEN` (from `claude setup-token`) still works and is used for Claude until you
sign in on the page.

## Configuration

| Variable | Default | |
|---|---|---|
| `PORT` | `8000` | API and setup page |
| `CALLBACK_PORT` | `1455` | OAuth callbacks (Codex requires 1455) |
| `PUBLIC_URL` | `http://localhost:$PORT` | Where callbacks send the browser afterwards |
| `TOKEN_STORE` | `./data/tokens.json` (`/data/tokens.json` in Docker) | Saved credentials, mode 0600 |
| `PROVIDERS` | `claude,openai` | Which accounts the setup page asks for |
| `OPEN_BROWSER` | `auto` | Open the setup page on start when sign-in is needed (`auto` skips inside containers) |

`GET /health` returns `setup_required` and per-provider status, so callers can send users
to the setup page.

## Usage

```python
anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")
openai.OpenAI(base_url="http://localhost:8000/openai/v1", api_key="unused")
```

The ChatGPT backend only accepts the models listed at `/openai/v1/models`, always
streams (non-streaming requests are assembled by the proxy), and rejects sampling
parameters like `temperature`, which the proxy drops.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest test_unit.py      # offline, upstreams mocked
python test_proxy.py     # live smoke test against a running proxy
```
