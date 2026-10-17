"""Compare TTS engines on the same lines: latency, realtime factor, and a sample to listen to.

Quality is not something this script can judge and not something I can hear, so it writes one
wav per engine per line into data/tts_samples/ and reports only what is measurable. The
listening is yours; the numbers are mine.

What matters for this system, in order:
  1. time to FIRST audio        the player is sitting in silence until this lands
  2. realtime factor            seconds of compute per second of speech; under 1.0 or the
                                narrator can never keep up with a long description
  3. fixed cost per call        Chatterbox is ~4.2s of it, which is why batching sentences
                                mattered more than shortening them

Architecture is the thing to watch. Autoregressive models (Chatterbox) generate one token at
a time at batch size 1 and are launch-bound, so they do not benefit from wider math -- measured
bf16 at 0.87x and tf32 at 0.93x, both SLOWER. Non-autoregressive models (Kokoro, StyleTTS2,
Piper) emit the whole utterance in a few passes and are an order of magnitude faster for the
same GPU.

Usage
  python tools/tts_bakeoff.py list
  python tools/tts_bakeoff.py run --engines kokoro,piper --runs 3
  python tools/tts_bakeoff.py run --all --out data/tts_samples
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

LINES = [
    "Roll me a perception check.",
    "The door gives way to a low stone hall, and the air that comes out of it is colder than it should be.",
    "Three of you are in the water now, and it is not slowing down for any of you. "
    "Below the surface the light goes thin fast, and something pale drifts past where it should be black.",
]


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def wav_seconds(b: bytes) -> float:
    try:
        import io
        with wave.open(io.BytesIO(b)) as w:
            return w.getnframes() / (w.getframerate() or 1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------- engines
def engine_kokoro():
    import config
    config.TTS_ENGINE = "kokoro"
    import tts

    async def run(text):
        return await tts.synth(text, "am_michael")
    return run, "kokoro-82M (non-AR, local ONNX)"


def engine_chatterbox():
    import config
    config.TTS_ENGINE = "chatterbox"
    import tts

    async def run(text):
        return await tts.synth(text, "am_michael")
    return run, "chatterbox-0.5B (autoregressive, voice clone)"


def engine_piper():
    """Piper: non-autoregressive VITS, ONNX, built for speed on modest hardware."""
    import asyncio
    from piper import PiperVoice          # raises if not installed
    import config
    model = next((ROOT / "state" / "models" / "piper").glob("*.onnx"), None)
    if model is None:
        raise RuntimeError("no piper voice in state/models/piper (see `list`)")
    v = PiperVoice.load(str(model))

    async def run(text):
        def go():
            import io
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                v.synthesize_wav(text, w)
            return buf.getvalue()
        return await asyncio.to_thread(go)
    return run, f"piper ({model.stem}, non-AR VITS)"


ENGINES = {"kokoro": engine_kokoro, "chatterbox": engine_chatterbox, "piper": engine_piper}


def cmd_list(a) -> int:
    avail = {}
    for name, factory in ENGINES.items():
        try:
            factory()
            avail[name] = "ready"
        except Exception as e:
            avail[name] = f"unavailable: {type(e).__name__}: {str(e)[:90]}"
    print(json.dumps({"engines": avail,
                      "piper_voices": "download a .onnx + .onnx.json into state/models/piper/ "
                                      "from https://huggingface.co/rhasspy/piper-voices"},
                     indent=2))
    return 0


def cmd_run(a) -> int:
    import asyncio
    names = list(ENGINES) if a.all else [n.strip() for n in a.engines.split(",") if n.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        if name not in ENGINES:
            rows.append({"engine": name, "error": "unknown"}); continue
        try:
            run, label = ENGINES[name]()
        except Exception as e:
            rows.append({"engine": name, "error": f"{type(e).__name__}: {str(e)[:110]}"})
            continue

        async def bench():
            per = []
            for i, line in enumerate(LINES):
                times, audio = [], b""
                for r in range(a.runs):
                    # a unique suffix defeats the cache; a repeat line is not a measurement
                    t0 = time.perf_counter()
                    audio = await run(line if r == a.runs - 1 else f"{line} {time.time_ns()}")
                    times.append(time.perf_counter() - t0)
                secs = wav_seconds(audio)
                (out / f"{name}_{i}.wav").write_bytes(audio or b"")
                per.append({"chars": len(line), "speech_s": round(secs, 2),
                            "p50_s": round(sorted(times)[len(times) // 2], 2),
                            "min_s": round(min(times), 2),
                            "realtime_factor": round(min(times) / secs, 3) if secs else None})
            return per
        try:
            per = asyncio.run(bench())
        except Exception as e:
            rows.append({"engine": name, "error": f"{type(e).__name__}: {str(e)[:110]}"})
            continue
        short, long_ = per[0], per[-1]
        # fixed cost from two points: the intercept of time against character count
        span = long_["chars"] - short["chars"]
        slope = (long_["min_s"] - short["min_s"]) / span if span else 0.0
        rows.append({"engine": name, "label": label, "per_line": per,
                     "fixed_cost_s": round(short["min_s"] - slope * short["chars"], 2),
                     "per_100_chars_s": round(slope * 100, 3),
                     "best_realtime_factor": min(
                         (p["realtime_factor"] for p in per if p["realtime_factor"]), default=None)})
    print(json.dumps({"samples": str(out), "results": rows,
                      "note": "listen to the samples; the numbers cannot judge quality"},
                     indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.set_defaults(fn=cmd_list)
    s = sub.add_parser("run")
    s.add_argument("--engines", default="kokoro")
    s.add_argument("--all", action="store_true")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--out", default=str(ROOT / "data" / "tts_samples"))
    s.set_defaults(fn=cmd_run)
    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
