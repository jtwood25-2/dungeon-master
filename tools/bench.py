"""Where the p90 actually goes.

A table feels the p90, not the median. Tonight's live DM sat at p50 3.0s and p90 7.3s, and
"the model was too slow" is not actionable until that 7.3s is attributed to a component. This
times each stage in isolation, repeatedly, and reports the distribution rather than an
average -- an average hides exactly the tail being complained about.

Stages, each independently switchable so a change can be measured on its own:

  stt     faster-whisper on a fixed clip, across beam sizes
  llm     first-token latency from the live backend (the number the player waits on)
  tts     narrator synthesis, cold and warm
  e2e     the three chained, as a turn actually runs

Usage
  python tools/bench.py stt  --runs 10
  python tools/bench.py tts  --runs 8
  python tools/bench.py llm  --runs 6
  python tools/bench.py all  --runs 5 --json
"""
import argparse
import asyncio
import json
import statistics as st
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

SR = 16000
LINES = [
    "The door gives way to a low stone hall, and the air that comes out of it is colder than it should be.",
    "Roll me a perception check.",
    "Three of you are in the water now, and it is not slowing down for any of you.",
]


def dist(name: str, samples: list) -> dict:
    if not samples:
        return {"stage": name, "n": 0}
    s = sorted(samples)
    q = lambda f: s[min(int(len(s) * f), len(s) - 1)]
    return {"stage": name, "n": len(s), "p50": round(q(0.5), 3), "p90": round(q(0.9), 3),
            "max": round(s[-1], 3), "mean": round(st.mean(s), 3),
            "spread_p90_over_p50": round(q(0.9) / max(q(0.5), 1e-6), 2)}


def _clip(seconds=4.0) -> np.ndarray:
    """A speech-shaped clip. Real words would be better, but the point here is to hold the
    INPUT constant so run-to-run variation is the system and not the audio."""
    t = np.arange(int(SR * seconds)) / SR
    sig = sum(np.sin(2 * np.pi * f * t) * a
              for f, a in ((120, .5), (240, .3), (480, .15), (1400, .05)))
    env = (np.sin(2 * np.pi * 2.5 * t) * 0.5 + 0.5) ** 2      # syllable-rate envelope
    return (sig * env * 0.3).astype(np.float32)


def bench_stt(runs: int, beams: list) -> list:
    import stt
    stt.warmup()
    pcm = _clip()
    out = []
    for beam in beams:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            stt.transcribe(pcm, "", beam_size=beam) if _accepts_beam(stt) else stt.transcribe(pcm, "")
            times.append(time.perf_counter() - t0)
        out.append({**dist(f"stt(beam={beam})", times),
                    "audio_seconds": round(len(pcm) / SR, 1),
                    "realtime_factor": round(dist("", times)["p50"] / (len(pcm) / SR), 3)})
    return out


def _accepts_beam(mod) -> bool:
    import inspect
    return "beam_size" in inspect.signature(mod.transcribe).parameters


def bench_tts(runs: int) -> list:
    import tts
    out = []
    asyncio.run(tts.warmup())

    async def one(line, fresh):
        if fresh:                        # defeat the cache; a repeat line is not a measurement
            line = line + " " + str(time.time_ns())
        t0 = time.perf_counter()
        await tts.synth(line)
        return time.perf_counter() - t0

    for label, fresh in (("tts(cached line)", False), ("tts(new line)", True)):
        times = [asyncio.run(one(LINES[i % len(LINES)], fresh)) for i in range(runs)]
        out.append({**dist(label, times)})
    return out


def bench_llm(runs: int) -> list:
    """First-token latency: the only part of generation the player actually waits through."""
    import llm, game_state
    state = game_state.GameState("_bench")
    state.add("Oktah", max_hp=24, is_pc=True)
    firsts, totals = [], []

    async def one(prompt):
        t0 = time.perf_counter()
        first = None
        async for _s in llm.respond(prompt, state):
            if first is None:
                first = time.perf_counter() - t0
        return first, time.perf_counter() - t0

    for i in range(runs):
        try:
            f, tot = asyncio.run(one(f"I look around the room. (bench {i})"))
        except Exception as e:
            return [{"stage": "llm", "error": f"{type(e).__name__}: {e}"}]
        if f is not None:
            firsts.append(f); totals.append(tot)
    return [dist("llm(first token)", firsts), dist("llm(full reply)", totals)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["stt", "tts", "llm", "all"])
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--beams", default="1,5")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    rows = []
    beams = [int(b) for b in a.beams.split(",") if b.strip()]
    try:
        if a.stage in ("stt", "all"):
            rows += bench_stt(a.runs, beams)
        if a.stage in ("tts", "all"):
            rows += bench_tts(a.runs)
        if a.stage in ("llm", "all"):
            rows += bench_llm(max(3, a.runs // 2))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1

    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    head = f"{'stage':<24}{'n':>4}{'p50':>9}{'p90':>9}{'max':>9}{'p90/p50':>9}"
    print(head); print("-" * len(head))
    for r in rows:
        if r.get("error"):
            print(f"{r['stage']:<24} {r['error']}")
            continue
        print(f"{r['stage']:<24}{r['n']:>4}{r.get('p50',0):>8.2f}s{r.get('p90',0):>8.2f}s"
              f"{r.get('max',0):>8.2f}s{r.get('spread_p90_over_p50',0):>9.2f}")
    print("\n  p90/p50 above ~2 means the tail, not the average, is what the table feels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
