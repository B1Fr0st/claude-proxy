#!/usr/bin/env sh
# Start the proxy with Docker Compose and open the sign-in page if any
# account still needs connecting.
set -eu
cd "$(dirname "$0")"

# macOS writes AppleDouble `._*` sidecars on non-APFS drives; BuildKit fails
# reading their xattrs before .dockerignore applies, so merge them away.
if command -v dot_clean >/dev/null 2>&1; then
  dot_clean -m . 2>/dev/null || true
fi

docker compose up -d --build

url="http://localhost:8000"
for _ in $(seq 1 30); do
  if health=$(curl -fsS "$url/health" 2>/dev/null); then
    break
  fi
  sleep 1
done

if [ -z "${health:-}" ]; then
  echo "Proxy did not become healthy; check: docker compose logs" >&2
  exit 1
fi

case "$health" in
  *'"setup_required":true'*)
    echo "Opening $url/ to connect your accounts..."
    if command -v open >/dev/null 2>&1; then open "$url/"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url/"
    else echo "Visit $url/ in your browser."
    fi
    ;;
  *) echo "Proxy is running at $url (accounts connected)." ;;
esac
