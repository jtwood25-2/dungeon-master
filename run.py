"""Run the Dungeon Master server so it can pick up new code ON DEMAND.

  python run.py

Edits to the server sit inert on disk while you play. When you press ⟳ Update in the app the
server exits with RESTART_CODE and this supervisor starts it again on the new code — so a
change lands the moment you ask for it and never a moment before.

Ctrl-C stops for real.
"""
import atexit
import subprocess
import sys
from pathlib import Path

RESTART_CODE = 3
ROOT = Path(__file__).parent
APP = ROOT / "server" / "app.py"
VOICE = ROOT / "voice_server.py"
VOICE_PY = ROOT / ".venv-tts" / "Scripts" / "python.exe"


def _engine() -> str:
    """Read TTS_ENGINE the same way server/config.py does, without importing it.

    run.py is the supervisor and deliberately keeps no dependency on the server package, so
    the value is read straight from .env with the same default config.py applies.
    """
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
    if val:
        return val
    return "chatterbox" if (ROOT / ".venv-tts" / "Scripts" / "python.exe").exists() else "kokoro"


def start_voice():
    """Start the Chatterbox voice server, but ONLY when something is going to use it.

    It is a 0.5B model that sits resident on the GPU, and with TTS_ENGINE=kokoro nothing ever
    calls it -- it was holding several GB of VRAM to serve no requests, on a 16GB card shared
    with Whisper, Kokoro and the quick-answer model. Loading it costs ~40s of startup as well.
    """
    engine = _engine()
    if engine != "chatterbox":
        print(f"(TTS_ENGINE={engine} — not starting the chatterbox voice server)", flush=True)
        return None
    if not VOICE_PY.exists():
        print("(no .venv-tts — narrator will use kokoro)", flush=True)
        return None
    log = (ROOT / "state" / "voice.log").open("a", encoding="utf-8")
    p = subprocess.Popen([str(VOICE_PY), str(VOICE)], stdout=log, stderr=log)
    atexit.register(lambda: p.terminate())
    print(f"voice server started (pid {p.pid}) -> state/voice.log", flush=True)
    return p


def main() -> int:
    start_voice()
    while True:
        try:
            # Write the server's log HERE rather than inheriting whatever handles the
            # supervisor was launched with. Relying on inheritance left state/supervisor.err
            # frozen at a timestamp hours old while the server was running and logging
            # normally -- the log looked dead, and losing visibility is worst exactly when
            # something is going wrong at the table. -u because Python block-buffers stderr
            # when it is a file rather than a terminal.
            log = (ROOT / "state" / "server.log").open("a", encoding="utf-8", buffering=1)
            try:
                code = subprocess.call([sys.executable, "-u", str(APP)],
                                       stdout=log, stderr=subprocess.STDOUT)
            finally:
                log.close()
        except KeyboardInterrupt:
            return 0
        if code == RESTART_CODE:
            print("\n--- update requested: restarting on the new code ---\n", flush=True)
            continue
        return code


if __name__ == "__main__":
    sys.exit(main())
