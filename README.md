# claude-api

A FastAPI proxy that turns a Claude Code OAuth token (from `claude setup-token`) into a local Anthropic-compatible API endpoint.

Is this probably against TOS? Yeah.
Is this gonna get you banned? Maybe!

## Run

With Docker Compose:

```bash
export CLAUDE_TOKEN="sk-ant-oat01-..."   # from `claude setup-token`
docker compose up -d
```

Or bare:

```bash
pip install -r requirements.txt
CLAUDE_TOKEN=... uvicorn server:app --port 8000
```

## Usage

```python
client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")
```
