"""Prosodic features: the signal humans actually use to know a turn is over.

Every text and timing feature we had scored near a coin flip on held-out data -- `gap` itself
managed 0.517 -- and a learning curve showed test AUC flat at ~0.61 from half the corpus
onward, so more episodes would only add more uninformative rows.

What is missing is how the speech SOUNDS as it ends. Turn-taking research is consistent about
which cues carry it, and none of them are in a transcript:

  falling pitch        a completed statement drops in F0 across its final syllables; a
                       continuation or a question holds level or rises
  final lengthening    the last syllable before a turn boundary is drawn out, and it is one of
                       the strongest boundary cues in conversational speech
  energy decay         speakers taper off when yielding the floor and hold level when they
                       intend to carry on
  voicing tail         trailing off into breath rather than stopping cleanly

Computed over the final window of each region, and reported RELATIVE to that speaker's own
baseline where it matters -- absolute pitch says who is talking, not whether they are finished,
so an un-normalised F0 feature would mostly encode speaker identity.

Adds a `prosody` block to an existing segments.jsonl; no re-transcription, so this runs over
the whole corpus in minutes rather than re-doing the pipeline.

Usage
  python tools/prosody.py add --session data/raw/<name> --audio data/clean/<name>.wav
  python tools/prosody.py all --raw data/raw --clean data/clean
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
SR = 16000
TAIL = 0.60                 # seconds at the end of a turn that carry the boundary cue
FMIN, FMAX = 60.0, 400.0    # human speech F0 range; outside this is noise or octave errors


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def f0_track(x: np.ndarray, sr: int = SR, hop: int = 160) -> np.ndarray:
    """F0 per frame via librosa's YIN. NaN where unvoiced."""
    import librosa
    if len(x) < int(sr * 0.1):
        return np.array([])
    try:
        f0 = librosa.yin(x, fmin=FMIN, fmax=FMAX, sr=sr,
                         frame_length=1024, hop_length=hop)
    except Exception:
        return np.array([])
    # YIN always returns a number; gate on aperiodicity via energy so unvoiced frames do not
    # contribute an arbitrary pitch to the slope. The strided RMS and YIN's own framing round
    # differently, so trim both to the shorter rather than assuming they line up.
    rms = np.sqrt(np.convolve(x ** 2, np.ones(1024) / 1024, mode="same"))[::hop]
    n = min(len(rms), len(f0))
    if n == 0:
        return np.array([])
    rms, f0 = rms[:n], f0[:n]
    f0 = np.where(rms > max(np.median(rms) * 0.35, 1e-5), f0, np.nan)
    return np.where((f0 > FMIN) & (f0 < FMAX), f0, np.nan)


def _slope(y: np.ndarray) -> float:
    """Least-squares slope per second, ignoring NaNs."""
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return 0.0
    t = np.arange(len(y))[ok]
    return float(np.polyfit(t, y[ok], 1)[0])


def features(seg: np.ndarray, sr: int = SR) -> dict:
    """Prosody for one speech region. All values are safe on short or silent input."""
    out = {"f0_slope": 0.0, "f0_end_rel": 0.0, "f0_range": 0.0, "voiced_frac": 0.0,
           "energy_slope": 0.0, "energy_end_rel": 0.0, "final_lengthening": 0.0,
           "tail_silence": 0.0}
    if seg.size < int(sr * 0.2):
        return out

    hop = 160
    rms = np.sqrt(np.convolve(seg ** 2, np.ones(400) / 400, mode="same"))[::hop]
    if rms.size < 4:
        return out
    n_tail = max(3, int(TAIL * sr / hop))
    tail_rms = rms[-n_tail:]
    mean_rms = float(rms.mean()) or 1e-9

    out["energy_slope"] = _slope(tail_rms) / mean_rms * (sr / hop)
    out["energy_end_rel"] = float(tail_rms[-3:].mean() / mean_rms)
    # how much of the very end is effectively silent -- trailing off vs stopping mid-word
    out["tail_silence"] = float((tail_rms < mean_rms * 0.15).mean())

    f0 = f0_track(seg, sr, hop)
    if f0.size >= 4:
        tail = f0[-n_tail:]
        voiced = f0[~np.isnan(f0)]
        out["voiced_frac"] = float((~np.isnan(tail)).mean())
        if voiced.size >= 4:
            base = float(np.median(voiced))
            out["f0_range"] = float(np.nanpercentile(voiced, 90) - np.nanpercentile(voiced, 10))
            tv = tail[~np.isnan(tail)]
            if tv.size >= 2:
                # semitones relative to this speaker's own median: pitch height identifies the
                # person, pitch MOVEMENT identifies the boundary
                out["f0_end_rel"] = float(12 * np.log2(max(tv[-1], 1e-6) / max(base, 1e-6)))
                out["f0_slope"] = _slope(tail) / max(base, 1e-6) * (sr / hop)

    # Final lengthening: the last voiced stretch drawn out relative to typical voiced runs
    # earlier in the same turn. Measured within the turn so it does not encode speaking rate.
    if f0.size >= 8:
        voiced_mask = ~np.isnan(f0)
        runs, cur = [], 0
        for v in voiced_mask:
            if v:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        if len(runs) >= 3:
            out["final_lengthening"] = float(runs[-1] / max(np.median(runs[:-1]), 1e-6))
    return out


KEYS = sorted(features(np.zeros(SR)))


def cmd_add(a) -> int:
    sess, wav = Path(a.session), Path(a.audio)
    segf = sess / "segments.jsonl"
    if not segf.exists():
        return _fail(f"no segments.jsonl in {sess}")
    if not wav.exists():
        return _fail(f"no audio at {wav}")
    rows = [json.loads(l) for l in segf.read_text(encoding="utf-8").splitlines() if l.strip()]
    pcm = load_wav(wav)
    done = 0
    for r in rows:
        if "prosody" in r and not a.force:
            continue
        seg = pcm[int(r["start"] * SR):int(r["end"] * SR)]
        r["prosody"] = {k: round(float(v), 5) for k, v in features(seg).items()}
        done += 1
    segf.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(json.dumps({"session": sess.name, "segments": len(rows), "computed": done}))
    return 0


def cmd_all(a) -> int:
    raw, clean = Path(a.raw), Path(a.clean)
    out = []
    for sess in sorted(p for p in raw.iterdir() if (p / "segments.jsonl").exists()):
        wav = clean / f"{sess.name}.wav"
        if not wav.exists():
            out.append({"session": sess.name, "error": "audio not found"}); continue
        ns = argparse.Namespace(session=str(sess), audio=str(wav), force=a.force)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_add(ns)
        try:
            out.append(json.loads(buf.getvalue()))
        except Exception:
            out.append({"session": sess.name, "error": buf.getvalue()[:120]})
        print(f"  {sess.name[:52]:54} {out[-1].get('computed', out[-1].get('error'))}",
              file=sys.stderr)
    print(json.dumps({"sessions": len(out), "detail": out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add"); s.add_argument("--session", required=True)
    s.add_argument("--audio", required=True); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_add)
    s = sub.add_parser("all")
    s.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    s.add_argument("--clean", default=str(ROOT / "data" / "clean"))
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_all)
    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
