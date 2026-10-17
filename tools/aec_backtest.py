"""Measure the echo canceller against every session we have recorded.

Sixty-two minutes of session audio sit in state/audio, and not one second of it has the
narrator's own signal alongside it -- only the microphone was ever written down, so there is
nothing to subtract. Recording is stereo from now on (mic left, speaker right), and once real
sessions accumulate this reads them directly.

Until then it MAKES the pair, which is a fair test and in one way a harsher one. It takes real
room audio from a real session, renders real narration through the real voice engine, pushes
that narration through a synthetic echo path -- delay, attenuation, a couple of reflections,
a low-pass for the speaker cone -- and adds it to the room. The canceller then gets exactly
what it will get live, and because the clean room audio is known, the score can be checked
against it rather than eyeballed.

Two numbers matter and they pull in opposite directions:

  ERLE           how much of the narrator is gone, in dB, measured where ONLY the narrator was
                 speaking. Higher is better.
  speech damage  how much of the ROOM survived, measured where the narrator was silent. This
                 is the one that catches a canceller that scores brilliantly by deleting
                 everything, which is the failure mode that actually hurts a table.

Usage
  python tools/aec_backtest.py                     # every recording
  python tools/aec_backtest.py --limit 8 --seconds 30
  python tools/aec_backtest.py --file "state/audio/.../s001-....wav"
"""
import argparse
import glob
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))


def read_wav(path: str, seconds: float = 0.0):
    """Returns (near, far). far is empty for the mono recordings made before the change."""
    with wave.open(path) as w:
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        if seconds:
            n = min(n, int(seconds * sr))
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch == 2:
        return a[0::2], a[1::2]
    return a, np.zeros(0, dtype=np.float32)


def echo_path(x: np.ndarray, delay: int, gain: float = 0.55) -> np.ndarray:
    """What a phone speaker in a room does to a signal on its way back to the microphone.

    Direct path plus two reflections plus a one-pole low-pass: a small speaker loses the top
    and the room smears the rest. Deliberately not a clean delay -- a canceller that only
    copes with a pure delay would score well here and fail in the room.
    """
    y = np.zeros(len(x) + delay + 2000, dtype=np.float32)
    y[delay:delay + len(x)] += gain * x
    y[delay + 700:delay + 700 + len(x)] += 0.22 * gain * x        # first reflection
    y[delay + 1900:delay + 1900 + len(x)] += 0.10 * gain * x      # second, later and quieter
    b = 0.0
    out = np.empty_like(y)
    for i, v in enumerate(y):                                      # speaker cone roll-off
        b = 0.72 * b + 0.28 * v
        out[i] = b
    return out[:len(x)]


def narration(seconds: float, sr: int = 16000) -> np.ndarray:
    """Real narration if the voice engine is up, a speech-shaped stand-in if it is not."""
    try:
        import asyncio
        import config
        config.TTS_ENGINE = "kokoro"
        import tts
        import audiolog

        async def go():
            await tts.warmup()
            parts, lines = [], [
                "The shaft cuts down at a shallow grade, packed earth underfoot.",
                "Three drowned dig-crew hold the passage ahead of you.",
                "Cold air comes out of the dark, and something is moving in it.",
                "Marn raises a waterlogged hand. He does not strike.",
            ]
            while sum(len(p) for p in parts) < seconds * sr * 2:
                for line in lines:
                    a = await tts.synth(line + " ")
                    if a:
                        parts.append(audiolog.wav_to_pcm16k(a))
                    if sum(len(p) for p in parts) >= seconds * sr * 2:
                        break
                if not parts:
                    break
            return b"".join(parts)

        pcm = asyncio.run(go())
        if pcm:
            a = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            return a[:int(seconds * sr)]
    except Exception as e:
        print(f"  (voice engine unavailable: {e}; using a speech-shaped stand-in)",
              file=sys.stderr)
    # fallback: noise shaped and gated like speech, so the test still means something
    rng = np.random.default_rng(0)
    n = int(seconds * sr)
    x = rng.standard_normal(n).astype(np.float32)
    b = 0.0
    for i in range(n):
        b = 0.9 * b + 0.1 * x[i]
        x[i] = b
    env = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:                                    # bursts and gaps, like sentences
        on = rng.integers(sr // 2, sr * 3)
        env[i:i + on] = 1.0
        i += on + rng.integers(sr // 4, sr)
    return x * env * 3.0


def score_one(path: str, seconds: float) -> dict:
    import aec

    near_clean, far = read_wav(path, seconds)
    if len(near_clean) < 16000:
        return {"file": os.path.basename(path), "skipped": "too short"}

    made_up = len(far) == 0
    if made_up:
        far = narration(len(near_clean) / aec.SR)[:len(near_clean)]
        if len(far) < len(near_clean):
            far = np.pad(far, (0, len(near_clean) - len(far)))
        true_delay = int(0.18 * aec.SR)
        near = near_clean + echo_path(far, true_delay)
    else:
        near, true_delay = near_clean, None
        near_clean = None                            # unknown for a real recording

    est = aec.estimate_delay(near, far)
    c = aec.Canceller()
    c.set_delay(est)
    out = np.concatenate([c.process(near[i:i + 1024], far[i:i + 1024])
                          for i in range(0, len(near), 1024)])

    # where was the narrator actually talking? energy in the far end says so
    win = 1024
    fe = np.array([float(np.mean(far[i:i + win] ** 2)) for i in range(0, len(far) - win, win)])
    thr = max(1e-8, float(np.mean(fe)) * 0.5)
    loud = fe > thr
    def gather(mask):
        idx = [i for i, m in enumerate(mask) if m]
        return ([np.concatenate([near[i * win:(i + 1) * win] for i in idx]),
                 np.concatenate([out[i * win:(i + 1) * win] for i in idx])] if idx else None)

    r = {"file": os.path.basename(path), "seconds": round(len(near) / aec.SR, 1),
         "reference": "synthesised" if made_up else "recorded",
         "delay_true": true_delay, "delay_found": est}
    if (pair := gather(loud)) is not None:
        r["erle_db"] = round(aec.erle_db(pair[0], pair[1]), 1)
    if (pair := gather(~loud)) is not None:
        # narrator silent: the output should be almost untouched. A big number here is a
        # canceller eating the room, not a canceller working.
        r["room_loss_db"] = round(aec.erle_db(pair[0], pair[1]), 1)
    if near_clean is not None and (pair := gather(loud)) is not None:
        idx = [i for i, m in enumerate(loud) if m]
        clean = np.concatenate([near_clean[i * win:(i + 1) * win] for i in idx])
        resid = pair[1] - clean
        num, den = float(np.mean(clean ** 2)), float(np.mean(resid ** 2))
        r["recovered_snr_db"] = round(10 * np.log10(num / den), 1) if den > 0 and num > 0 else None
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="")
    p.add_argument("--limit", type=int, default=0, help="0 = every recording")
    p.add_argument("--seconds", type=float, default=25.0, help="per file")
    a = p.parse_args()

    files = [a.file] if a.file else sorted(
        glob.glob(str(ROOT / "state" / "audio" / "**" / "*.wav"), recursive=True))
    files = [f for f in files if os.path.getsize(f) > 44 + 16000 * 2]
    if a.limit:
        files = files[-a.limit:]
    if not files:
        print(json.dumps({"error": "no recordings with audio in them"})); return 1

    rows = [score_one(f, a.seconds) for f in files]
    ok = [r for r in rows if "erle_db" in r]
    out = {"files": len(rows), "scored": len(ok), "detail": rows[-12:]}
    if ok:
        out["median_erle_db"] = round(float(np.median([r["erle_db"] for r in ok])), 1)
        loss = [r["room_loss_db"] for r in ok if r.get("room_loss_db") is not None]
        if loss:
            out["median_room_loss_db"] = round(float(np.median(loss)), 1)
        snr = [r["recovered_snr_db"] for r in ok if r.get("recovered_snr_db") is not None]
        if snr:
            out["median_recovered_snr_db"] = round(float(np.median(snr)), 1)
        found = [r for r in ok if r.get("delay_true") is not None]
        if found:
            out["delay_within_2ms"] = sum(
                1 for r in found if abs(r["delay_found"] - r["delay_true"]) < 32) / len(found)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
