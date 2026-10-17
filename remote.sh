#!/usr/bin/env bash
# Expose the local DM server to the internet so you can play from ANY network.
# Gives a public https URL with a real cert -> the phone mic works, no LAN needed.
#
# One-time install (Windows, in an admin PowerShell):  winget install Cloudflare.cloudflared
# Then run the DM server (python server/app.py) and, in another terminal:  bash remote.sh
set -e
PORT="${PORT:-8080}"
SCHEME="http"; [ -f certs/cert.pem ] && SCHEME="https"
if [ -x ./bin/cloudflared.exe ]; then           # vendored copy wins -- no install needed
  CF=./bin/cloudflared.exe
elif command -v cloudflared >/dev/null 2>&1; then
  CF=cloudflared
else
  echo "cloudflared not found. Install it, then re-run."
  echo "  Windows: winget install Cloudflare.cloudflared"
  echo "  macOS:   brew install cloudflared"
  exit 1
fi
echo "Tunneling $SCHEME://localhost:$PORT to a public https URL (open it on your phone)…"
# --no-tls-verify lets the tunnel accept the local self-signed cert
exec "$CF" tunnel --url "$SCHEME://localhost:$PORT" --no-tls-verify
