"""Replay a recorded session offline and score the listening decisions.

Every tuning change so far has needed five people in a room to evaluate, which means it gets
evaluated once, badly, in the middle of a game. A recorded session plus its decision index is
enough to re-run the whole listening path offline: the VAD, the transcription, and the
endpoint call -- as fast as the machine will go, and as many times as we like.

What this answers:

  replay   what WOULD the current settings decide on this audio?
  sweep    which VAD / endpoint settings interrupt least while still answering cues?
  diff     where do two configurations disagree, and on which utterances?

Scoring uses the recorded index as ground truth where it exists (`answered` / `held` /
`ignored` / `echo` written live at the exact sample offset), and falls back to corpus labels
otherwise. The metric that matters is INTERRUPTIONS -- decisions to speak while the DM already
had the floor -- because that is what made the session unpleasant.

Usage
  python tools/replay.py replay --session state/audio/<camp>/<day>/<stem>.wav
  python tools/replay.py sweep  --session <wav> --silence 700,1100,1500 --hold 1200,1600,2400
  python tools/replay.py diff   --session <wav> --a silence=1100 --b silence=1500
"""
import argparse
import itertools
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

SR = 16000


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def load_session(wav_path: Path) -> tuple:
    with wave.open(str(wav_path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise SystemExit(json.dumps({"error": "expected 16k mono"}))
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    idx = wav_path.with_suffix(".jsonl")
    marks = []
    if idx.exists():
        marks = [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pcm, marks


def run_config(pcm: np.ndarray, cfg: dict, transcribe=True) -> list:
    """Feed the recording through the real VAD and the real endpoint rule.

    Imports the live modules rather than reimplementing them, so a replay result is a
    statement about the shipped code and not about a copy of it that has drifted.
    """
    import config as cfg_mod
    import endpoint
    from vad import VAD

    for k, v in cfg.items():
        setattr(cfg_mod, k, v)

    vad = VAD()
    decisions = []
    frame = int(SR * 0.02)
    speaking = False                       # replayed cold: nothing knows when the DM had the floor
    stt_mod = None
    if transcribe:
        import stt as stt_mod
        stt_mod.warmup()

    for i in range(0, len(pcm) - frame, frame):
        chunk = pcm[i:i + frame].tobytes()
        utt = vad.feed(chunk)
        if utt is None:
            continue
        text = ""
        if stt_mod is not None:
            audio = np.frombuffer(utt, dtype="<i2").astype(np.float32) / 32768.0
            text = (stt_mod.transcribe(audio, "") or "").strip()
        call, why = endpoint.assess(text, None, speaking)
        decisions.append({"sec": round(i / SR, 2), "text": text, "call": call, "why": why,
                          "samples": len(utt) // 2})
    return decisions


def score(decisions: list, marks: list) -> dict:
    """Compare against what the live system actually did, where that was recorded."""
    import endpoint
    go = sum(d["call"] == endpoint.GO for d in decisions)
    wait = sum(d["call"] == endpoint.WAIT for d in decisions)
    ign = sum(d["call"] == endpoint.IGNORE for d in decisions)
    n = max(len(decisions), 1)
    truth = {m["kind"] for m in marks}
    return {"utterances": len(decisions),
            "go": go, "wait": wait, "ignore": ign,
            "go_rate": round(go / n, 3), "ignore_rate": round(ign / n, 3),
            "recorded_marks": len(marks),
            "recorded_kinds": sorted(truth) if truth else []}


def cmd_replay(a) -> int:
    wav = Path(a.session)
    if not wav.exists():
        return _fail(f"no recording at {wav}")
    pcm, marks = load_session(wav)
    d = run_config(pcm, {}, transcribe=not a.no_stt)
    out = {"session": wav.name, "seconds": round(len(pcm) / SR, 1), **score(d, marks)}
    if a.verbose:
        out["decisions"] = d[:80]
    print(json.dumps(out, indent=2))
    return 0


def cmd_sweep(a) -> int:
    """Grid over the settings that govern when it speaks.

    Reports go_rate per configuration. A lower go_rate is not automatically better -- a system
    that never speaks is useless -- so the cue count is reported beside it and the choice
    stays a judgement rather than an argmax.
    """
    wav = Path(a.session)
    if not wav.exists():
        return _fail(f"no recording at {wav}")
    pcm, marks = load_session(wav)
    sil = [int(x) for x in a.silence.split(",") if x.strip()]
    hold = [int(x) for x in a.hold.split(",") if x.strip()]
    rows = []
    for s, h in itertools.product(sil, hold):
        d = run_config(pcm, {"VAD_SILENCE_MS": s, "VAD_HOLD_MS": float(h)},
                       transcribe=not a.no_stt)
        rows.append({"VAD_SILENCE_MS": s, "VAD_HOLD_MS": h, **score(d, marks)})
    rows.sort(key=lambda r: (r["go_rate"], -r["utterances"]))
    print(json.dumps({"session": wav.name, "seconds": round(len(pcm) / SR, 1),
                      "grid": rows,
                      "note": "lower go_rate interrupts less; check utterances has not collapsed"},
                     indent=2))
    return 0


def _kv(pairs: str) -> dict:
    out = {}
    for part in pairs.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            key = {"silence": "VAD_SILENCE_MS", "hold": "VAD_HOLD_MS",
                   "noise": "VAD_NOISE_MULT", "barge": "VAD_BARGE_FLOOR"}.get(k.strip(), k.strip())
            out[key] = float(v) if "." in v else int(v)
    return out


def cmd_diff(a) -> int:
    wav = Path(a.session)
    if not wav.exists():
        return _fail(f"no recording at {wav}")
    pcm, marks = load_session(wav)
    A, B = run_config(pcm, _kv(a.a)), run_config(pcm, _kv(a.b))
    by_a = {round(d["sec"], 1): d for d in A}
    changed = []
    for d in B:
        prev = by_a.get(round(d["sec"], 1))
        if prev and prev["call"] != d["call"]:
            changed.append({"sec": d["sec"], "text": d["text"][:70],
                            "a": prev["call"], "a_why": prev["why"],
                            "b": d["call"], "b_why": d["why"]})
    print(json.dumps({"a": _kv(a.a), "b": _kv(a.b),
                      "a_summary": score(A, marks), "b_summary": score(B, marks),
                      "changed": changed[:60], "n_changed": len(changed)}, indent=2))
    return 0


def cmd_list(a) -> int:
    import audiolog
    print(json.dumps(audiolog.sessions(a.campaign), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="recorded sessions available to replay")
    s.add_argument("--campaign", default="")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("replay", help="what the current settings decide on this audio")
    s.add_argument("--session", required=True)
    s.add_argument("--no-stt", action="store_true", help="timings only, skip transcription")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("sweep", help="grid over the settings that govern when it speaks")
    s.add_argument("--session", required=True)
    s.add_argument("--silence", default="700,1100,1500,2000")
    s.add_argument("--hold", default="1200,1600,2400")
    s.add_argument("--no-stt", action="store_true")
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("diff", help="where two configurations disagree")
    s.add_argument("--session", required=True)
    s.add_argument("--a", default="silence=1100,hold=1600")
    s.add_argument("--b", default="silence=1500,hold=2400")
    s.set_defaults(fn=cmd_diff)

    a = p.parse_args()
    try:
        return a.fn(a)
    except SystemExit:
        raise
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
