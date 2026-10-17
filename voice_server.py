"""High-quality neural voice, kept loaded in its own process.

  .venv-tts/Scripts/python voice_server.py        # start it (or let run.py do it)

Chatterbox needs Python 3.12 and its own heavy dependency tree, and it cannot build on the
3.14 the app runs on. So it lives here: a tiny HTTP service in a separate venv, holding the
model resident on the GPU. That isolation is also why the app can fall back to the local
Kokoro voice without this process running at all.

Resident matters: loading the model costs ~7s, and doing that per line would be unusable.
Warmed once, generation runs at roughly 0.65x realtime, so after the first line synthesis
stays ahead of playback.
"""
import io
import json
import os
import sys
import threading
import time
import wave
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

PORT = int(os.environ.get("VOICE_PORT", "8123"))
DEVICE = os.environ.get("VOICE_DEVICE", "cuda")
# Chatterbox itself has one voice, but it clones from a reference clip — so every Kokoro
# voice becomes available at Chatterbox quality. The clips are generated locally from Kokoro
# (see tools that write state/models/voiceprints), which keeps the whole cast offline and
# free of any question about where a voice sample came from.
PRINTS = Path(__file__).parent / "state" / "models" / "voiceprints"
_model = None
_lock = threading.Lock()          # one GPU, one generation at a time


def model():
    global _model
    if _model is None:
        import warnings
        warnings.filterwarnings("ignore")
        from chatterbox.tts import ChatterboxTTS
        t = time.perf_counter()
        _model = ChatterboxTTS.from_pretrained(device=DEVICE)
        print(f"chatterbox loaded in {time.perf_counter() - t:.1f}s on {DEVICE}", flush=True)
    return _model


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    peak = float(np.abs(pcm).max()) or 1.0
    pcm = (pcm / peak * 0.85 * 32767).astype("<i2")     # even level between lines
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                            # the app logs what matters

    def do_GET(self):
        if self.path == "/health":
            ready = _model is not None
            voices = sorted(p.stem for p in PRINTS.glob("*.wav")) if PRINTS.exists() else []
            self._send(200, json.dumps({"ready": ready, "voices": voices}).encode(),
                       "application/json")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self):
        if self.path != "/speak":
            return self._send(404, b"", "text/plain")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            text = (body.get("text") or "").strip()
            if not text:
                return self._send(400, b"empty", "text/plain")
            m = model()
            kw = {}
            voice = (body.get("voice") or "").strip()
            if voice:
                clip = PRINTS / f"{voice}.wav"
                if clip.exists():
                    kw["audio_prompt_path"] = str(clip)
                else:
                    print(f"no voiceprint for {voice!r}; using the default", flush=True)
            with _lock:
                wav = m.generate(
                    text,
                    exaggeration=float(body.get("exaggeration", 0.5)),
                    cfg_weight=float(body.get("cfg_weight", 0.5)),
                    **kw,
                )
            audio = wav.squeeze(0).cpu().numpy()
            self._send(200, to_wav(audio, m.sr), "audio/wav")
        except Exception as e:
            print(f"speak failed: {e}", flush=True)
            self._send(500, str(e).encode(), "text/plain")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


def main() -> int:
    model()                                             # load before accepting traffic
    with _lock:
        model().generate("Ready.")                      # and warm the graph
    print(f"voice server listening on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
