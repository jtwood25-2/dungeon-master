"""Compare speech detectors on real table audio.

The energy detector this replaced found 38% of the speech and quietly destroyed a whole
corpus, so "which VAD" is not a detail. What matters here, in order:

  coverage      fraction of a continuously-talking recording marked as speech. A podcast
                table is speaking ~90% of the time; anything far below that is dropping turns.
  boundaries    how many regions it produces. Too few means turns are being welded together,
                which loses exactly the pauses turn-taking is learned from.
  speed         it runs over hours of audio, so seconds per minute of audio matters.

Coverage alone is not the goal -- a detector that returns "all speech, one region" scores 100%
and is useless. Read the two together.

Usage
  python tools/vad_bakeoff.py --audio data/clean/<file>.wav --minutes 12
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
SR = 16000


def load(path: Path, minutes: float) -> np.ndarray:
    with wave.open(str(path)) as w:
        n = w.getnframes() if not minutes else min(w.getnframes(), int(minutes * 60 * SR))
        return np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0


# each returns [(start_s, end_s)]
def vad_energy(pcm):
    import corpus
    return corpus._energy_regions(pcm)


def vad_faster_whisper(pcm):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(pcm, VadOptions(min_speech_duration_ms=350,
                                               min_silence_duration_ms=300))
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def vad_silero_standalone(pcm):
    """The pip package tracks upstream; faster-whisper vendors an older snapshot."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(pcm), model, sampling_rate=SR,
                               min_speech_duration_ms=350, min_silence_duration_ms=300,
                               return_seconds=True)
    return [(t["start"], t["end"]) for t in ts]


def vad_ten(pcm):
    """TEN VAD: frame-wise probabilities, so the hysteresis is ours to apply."""
    from ten_vad import TenVad
    hop = 256                                   # 16ms at 16k, the model's frame
    v = TenVad(hop_size=hop)
    i16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    probs = []
    for i in range(0, len(i16) - hop, hop):
        p, _flag = v.process(i16[i:i + hop])
        probs.append(p)
    probs = np.array(probs)
    on, off = 0.5, 0.35                         # hysteresis: enter loud, leave quiet
    out, start, speaking = [], None, False
    for i, p in enumerate(probs):
        t = i * hop / SR
        if not speaking and p >= on:
            speaking, start = True, t
        elif speaking and p < off:
            if t - start >= 0.35:
                out.append((start, t))
            speaking = False
    if speaking and (len(probs) * hop / SR) - start >= 0.35:
        out.append((start, len(probs) * hop / SR))
    merged = []
    for s, e in out:
        if merged and s - merged[-1][1] < 0.30:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


ENGINES = {
    "energy (old)": vad_energy,
    "silero (bundled)": vad_faster_whisper,
    "silero 6.x": vad_silero_standalone,
    "ten-vad": vad_ten,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True)
    p.add_argument("--minutes", type=float, default=12.0)
    a = p.parse_args()

    src = Path(a.audio)
    if not src.exists():
        print(json.dumps({"error": f"no audio at {src}"})); return 1
    pcm = load(src, a.minutes)
    total = len(pcm) / SR
    rows = []
    for name, fn in ENGINES.items():
        try:
            t0 = time.perf_counter()
            regs = fn(pcm)
            el = time.perf_counter() - t0
            speech = sum(e - s for s, e in regs)
            gaps = [regs[i + 1][0] - regs[i][1] for i in range(len(regs) - 1)]
            rows.append({"engine": name, "regions": len(regs),
                         "coverage": round(speech / total, 3),
                         "median_region_s": round(float(np.median([e - s for s, e in regs])), 2) if regs else 0,
                         "median_gap_s": round(float(np.median(gaps)), 2) if gaps else 0,
                         "seconds_per_audio_minute": round(el / (total / 60), 3)})
        except Exception as e:
            rows.append({"engine": name, "error": f"{type(e).__name__}: {str(e)[:110]}"})
    print(json.dumps({"audio": src.name, "minutes": round(total / 60, 1), "results": rows,
                      "note": "coverage near 0.9 is expected on a table that talks constantly; "
                              "very few regions means turns are being welded together"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
