"""Keep the public tunnel up, and always know its address.

  python tunnel.py            # start it, print the URL, restart it if it dies
  python tunnel.py --show     # just print the current URL and exit

A free Cloudflare quick tunnel gets a random hostname, and a new one every time it starts.
That is the price of not owning a domain — but losing the tunnel silently, or having to dig
the URL out of a log, is not. The URL is written to state/tunnel-url.txt the moment it
appears, so --show always answers "what do I open on my phone".
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
URL_FILE = ROOT / "state" / "tunnel-url.txt"
CLOUDFLARED = ROOT / "bin" / "cloudflared.exe"
PORT = 8080
_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def show() -> int:
    if not URL_FILE.exists():
        print("no tunnel recorded — start one with: python tunnel.py")
        return 1
    print(URL_FILE.read_text(encoding="utf-8").strip())
    return 0


def _https() -> bool:
    return (pathlib.Path(__file__).parent / "certs" / "cert.pem").exists()


def _binary() -> str:
    if CLOUDFLARED.exists():
        return str(CLOUDFLARED)
    return "cloudflared"                      # fall back to one on PATH


def run_once() -> int:
    """Run cloudflared, capturing the URL it prints. Returns its exit code."""
    proc = subprocess.Popen(
        # https when there is a cert, because that is what the server is then speaking. Left
        # pointing at http it tunnels to a port that no longer answers and every phone gets a
        # 502 -- with the LAN working fine, which is a confusing way to find out.
        [_binary(), "tunnel", "--url",
         f"{'https' if _https() else 'http'}://localhost:{PORT}", "--no-tls-verify"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", bufsize=1)
    found = False
    for line in proc.stderr:
        if not found and (m := _URL.search(line)):
            found = True
            URL_FILE.parent.mkdir(parents=True, exist_ok=True)
            URL_FILE.write_text(m.group(0), encoding="utf-8")
            print("\n" + "=" * 62)
            print("  open this on your phone:")
            print("  " + m.group(0))
            print("=" * 62 + "\n", flush=True)
    return proc.wait()


def main() -> int:
    if "--show" in sys.argv:
        return show()
    delay = 2
    while True:
        try:
            code = run_once()
        except KeyboardInterrupt:
            return 0
        except FileNotFoundError:
            print("cloudflared not found. Put it in bin/ or install it, then re-run.")
            return 1
        # A quick tunnel gets a NEW hostname on restart, so say so plainly rather than
        # letting the old link fail silently in someone's browser.
        print(f"\ntunnel exited ({code}) — restarting in {delay}s; the URL will change\n",
              flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 30)


if __name__ == "__main__":
    sys.exit(main())
