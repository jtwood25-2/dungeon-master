#!/usr/bin/env bash
# Self-signed cert so the phones can use their microphones. iOS requires HTTPS for getUserMedia,
# and without a cert the only way to get it is a tunnel.
#
# This matters far more than it looks. A whole seven-player session ran through a free Cloudflare
# quick tunnel -- the path of least resistance when there is no cert -- while every player sat in
# the same room on the same wifi. That evening opened 181 websockets with a median lifetime of 54
# seconds and 105 abnormal closures, and utterances kept landing in sockets that were gone before
# the DM could answer. On the LAN there is no tunnel to drop: the phones talk straight to this
# machine, and the round trip stops going out to Cloudflare and back.
#
# Run once from git-bash:  bash certs/make_cert.sh
set -e
here="$(cd "$(dirname "$0")" && pwd)"

# Two separate git-bash hazards, and getting one right breaks the other:
#   - the subject "/CN=..." is rewritten into a Windows path unless MSYS_NO_PATHCONV is set
#   - but openssl here is a Windows binary, so the FILE paths must be Windows paths
# So: convert the paths ourselves, and switch the rewriting off for the subject.
win="$(cygpath -w "$here" 2>/dev/null || echo "$here")"

lan="$(python -c 'import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(("8.8.8.8",80));print(s.getsockname()[0]);s.close()' 2>/dev/null)"
[ -z "$lan" ] && lan="127.0.0.1"
echo "including this machine's LAN address: $lan"

# Every address a phone might type has to be in the cert, or Safari refuses it and the microphone
# stays blocked -- which is exactly what sends everyone back to the tunnel.
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
  -keyout "${win}\\key.pem" -out "${win}\\cert.pem" \
  -subj "/CN=dungeon-master.local" \
  -addext "subjectAltName=DNS:dungeon-master.local,DNS:localhost,IP:127.0.0.1,IP:${lan}" \
  2>/dev/null

echo
echo "wrote $here/cert.pem and key.pem"
echo "restart the server, then open on each phone:   https://${lan}:8080"
echo "Safari warns once about the certificate. Accept it -- it is yours."
