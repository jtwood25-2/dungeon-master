"""One command to bring the whole thing up, and to put it on a schedule.

Nothing here auto-started on boot, so every restart meant remembering four separate launches
in the right order and then hunting for the new tunnel URL. Worse, the failures were quiet:
the app runs perfectly well with the voice server missing, and just sounds wrong.

  python start.py              start whatever is not already running, wait, report
  python start.py --status     what is up, what is not, and the current public URL
  python start.py --stop       stop the things this script manages
  python start.py --install    run it automatically at logon (Windows Task Scheduler)
  python start.py --uninstall  remove that task

Idempotent by design: it checks the PORT rather than a pidfile, so running it twice is
harmless and it recovers correctly if something was started by hand.

Ordering matters in one place only -- ollama must answer before the app finishes warming, or
the DM's quick-lookup path logs a failure on first use. Everything else is independent.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable
STATE = ROOT / "state"
OLLAMA = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
TASK = "DungeonMasterStart"

# name -> (port, how to start it, how long it may take to answer)
SERVICES = [
    ("ollama", 11434, lambda: _spawn([str(OLLAMA), "serve"], "ollama"), 40),
    # run.py supervises BOTH the app and the voice server, and restarts the app on exit
    # code 3 so the in-app Update button works. Starting app.py directly breaks that.
    ("app", 8080, lambda: _spawn([PY, str(ROOT / "run.py")], "supervisor"), 180),
    ("voice", 8123, None, 180),          # comes up with the supervisor; never started alone
    ("tunnel", None, lambda: _spawn([PY, str(ROOT / "tunnel.py")], "tunnel"), 60),
]


def _tts_engine() -> str:
    """Which narrator engine is configured, read from .env without importing the server."""
    import os
    val = os.environ.get("TTS_ENGINE", "")
    if not val:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("TTS_ENGINE=") and not line.startswith("#"):
                    val = line.split("=", 1)[1].strip()
                    break
    return val or ("chatterbox" if (ROOT / ".venv-tts" / "Scripts" / "python.exe").exists()
                   else "kokoro")


def _up(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _spawn(cmd: list, log: str) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    out = (STATE / f"{log}.out").open("a", encoding="utf-8")
    err = (STATE / f"{log}.err").open("a", encoding="utf-8")
    flags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: survive this script exiting, which is
        # the whole point -- a child that dies with its parent is not a startup.
        flags = 0x00000008 | 0x00000200
    p = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=str(ROOT),
                         creationflags=flags, close_fds=True)
    return p.pid


def lan_url() -> str:
    """Where the phones should actually point when everyone is in the same room.

    A seven-player session ran entirely through a Cloudflare quick tunnel while the whole table
    sat on one wifi: 181 sockets, median life 54 seconds, 105 abnormal closures. There was
    nothing wrong with the tunnel that being unnecessary would not fix.
    """
    import socket
    scheme = "https" if (ROOT / "certs" / "cert.pem").exists() else "http"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return f"{scheme}://{ip}:8080"


def tunnel_url() -> str:
    p = STATE / "tunnel-url.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def status() -> dict:
    d = {"services": {}, "lan": lan_url(), "url": tunnel_url()}
    for name, port, _start, _wait in SERVICES:
        if name == "voice" and _tts_engine() != "chatterbox":
            d["services"][name] = "not needed (TTS_ENGINE=" + _tts_engine() + ")"
            continue
        d["services"][name] = ("up" if _up(port) else "down") if port else (
            "up" if tunnel_url() else "down")
    return d


def cmd_start(a) -> int:
    started = []
    for name, port, start, wait in SERVICES:
        if port and _up(port):
            print(f"  {name:8} already up"); continue
        if name == "tunnel" and tunnel_url() and a.keep_url:
            print(f"  {name:8} already up"); continue
        if start is None:
            continue                       # voice comes up with the supervisor
        if name == "ollama" and not OLLAMA.exists():
            print(f"  {name:8} SKIPPED (not installed at {OLLAMA})"); continue
        pid = start()
        started.append((name, port, wait))
        print(f"  {name:8} starting (pid {pid})")

    deadline = {n: time.time() + w for n, _p, w in started}
    pending = {n: p for n, p, _w in started if p}
    # the tunnel has no port; it is ready when it has written its URL
    want_url = any(n == "tunnel" for n, _p, _w in started)
    before = tunnel_url()
    while pending or want_url:
        time.sleep(2)
        for n in list(pending):
            if _up(pending[n]):
                print(f"  {n:8} up"); pending.pop(n)
            elif time.time() > deadline[n]:
                print(f"  {n:8} DID NOT COME UP -- see state/*.err"); pending.pop(n)
        if want_url:
            u = tunnel_url()
            if u and u != before:
                print(f"  tunnel   up"); want_url = False
            elif time.time() > deadline.get("tunnel", 0):
                print("  tunnel   DID NOT COME UP -- see state/tunnel.err"); want_url = False

    # The voice server used to be the one that failed quietly: the DM still speaks, just in
    # the fallback voice, and nothing at the table says why. It is only worth warning about
    # when the narrator is actually meant to be using it.
    if _tts_engine() == "chatterbox" and not _up(8123):
        print("  voice    NOT RUNNING -- the narrator will fall back to kokoro")
    s = status()
    print(json.dumps(s, indent=2))
    return 0 if all(v == "up" or v.startswith("not needed")
                    for v in s["services"].values()) else 1


def cmd_stop(a) -> int:
    if sys.platform != "win32":
        print(json.dumps({"error": "stop is implemented for Windows only"})); return 1
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
          "'run.py|tunnel.py|server\\\\app.py|voice_server.py' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    if a.all:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-Process ollama,'ollama app' -ErrorAction SilentlyContinue | "
                        "Stop-Process -Force"], capture_output=True)
    time.sleep(2)
    print(json.dumps(status(), indent=2))
    return 0


def cmd_install(a) -> int:
    """Register a logon task. ONLOGON rather than ONSTART deliberately: the tunnel and the
    GPU models want a real user session, and a boot-time service would race the graphics
    driver."""
    if sys.platform != "win32":
        print(json.dumps({"error": "install is Windows-only"})); return 1
    cmd = f'"{PY}" "{ROOT / "start.py"}"'
    base = ["schtasks", "/Create", "/TN", TASK, "/SC", "ONLOGON",
            "/TR", cmd, "/F", "/DELAY", "0000:30"]
    # /RL HIGHEST needs an elevated shell. Nothing here requires elevation -- it binds user
    # ports and spawns user processes -- so fall back rather than refuse to install.
    r = subprocess.run(base + ["/RL", "HIGHEST"], capture_output=True, text=True)
    elevated = r.returncode == 0
    if not elevated:
        r = subprocess.run(base, capture_output=True, text=True)
    ok = r.returncode == 0
    if ok:
        print(json.dumps({"installed": True, "via": "scheduled task", "task": TASK,
                          "runs": cmd, "run_level": "highest" if elevated else "normal",
                          "note": "30s delay after logon so the network and GPU driver are ready"},
                         indent=2))
        return 0

    # Creating ONLOGON tasks needs administrator on this machine, at any run level. The user
    # Startup folder achieves the same thing with no elevation at all, so use it rather than
    # telling someone to re-run an install from an admin shell they may not have.
    shortcut = _startup_dir() / "DungeonMaster.cmd"
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    shortcut.write_text(
        "@echo off\r\n"
        "rem Started at logon. Delete this file (shell:startup) to stop that.\r\n"
        f'cd /d "{ROOT}"\r\n'
        "timeout /t 30 /nobreak >nul\r\n"          # let the network and GPU driver settle
        f'"{PY}" "{ROOT / "start.py"}" >> "{STATE / "autostart.log"}" 2>&1\r\n',
        encoding="utf-8")
    print(json.dumps({"installed": True, "via": "startup folder",
                      "path": str(shortcut), "runs": cmd,
                      "why": "schtasks needs admin for ONLOGON; the Startup folder does not",
                      "remove_with": "python start.py --uninstall"}, indent=2))
    return 0


def _startup_dir() -> Path:
    return (Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows"
            / "Start Menu" / "Programs" / "Startup")


def cmd_uninstall(a) -> int:
    """Remove BOTH install routes -- whichever one was used, this should undo it."""
    r = subprocess.run(["schtasks", "/Delete", "/TN", TASK, "/F"],
                       capture_output=True, text=True)
    shortcut = _startup_dir() / "DungeonMaster.cmd"
    had_shortcut = shortcut.exists()
    shortcut.unlink(missing_ok=True)
    print(json.dumps({"task_removed": r.returncode == 0,
                      "startup_entry_removed": had_shortcut,
                      "detail": (r.stdout or r.stderr).strip()[:160]}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--all", action="store_true", help="with --stop, also stop ollama")
    p.add_argument("--install", action="store_true")
    p.add_argument("--uninstall", action="store_true")
    p.add_argument("--keep-url", action="store_true",
                   help="do not start a new tunnel if one is already recorded")
    a = p.parse_args()
    if a.status:
        print(json.dumps(status(), indent=2)); return 0
    if a.stop:
        return cmd_stop(a)
    if a.install:
        return cmd_install(a)
    if a.uninstall:
        return cmd_uninstall(a)
    return cmd_start(a)


if __name__ == "__main__":
    sys.exit(main())
