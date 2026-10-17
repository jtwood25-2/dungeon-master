"""Cut episode audio down to the parts that are actually a game being run.

A Critical Role episode is ~4 hours, and a large slice of it is not a table playing: a
pre-show of announcements and sponsor reads, a mid-episode break with music and adverts, and
a post-show. Training turn-taking on that teaches the model the rhythm of two hosts reading
copy, which is the opposite of the behaviour wanted.

The transcripts carry the structure explicitly -- Pre-Show / Part I / Break / Part II /
Post-Show -- so the boundaries are known semantically. What is missing is WHERE they fall in
time, because wiki transcripts have no timestamps. So:

  inspect  find long non-speech stretches; intro music, the break and the outro all show up
           as sustained gaps, and they are the candidate cut points
  align    anchor the transcript's part boundaries to real times, by transcribing short
           windows around each candidate and fuzzy-matching them against the first lines of
           that part. Anchoring on TEXT rather than on a fixed offset means an episode that
           runs long, or opens with an extra announcement, still cuts correctly.
  crop     write Part I and Part II as clean 16k mono wav, the format the corpus tools read

Everything prints JSON. `align` reports a confidence per boundary and refuses low-confidence
matches rather than cutting in the wrong place -- a bad cut silently poisons every label built
from the audio after it.

Usage
  python tools/prep_audio.py inspect --audio "critical_role_sessions/....mp3"
  python tools/prep_audio.py align   --audio <mp3> --transcript data/transcripts/<same>.md
  python tools/prep_audio.py crop    --all --out data/clean
"""
import argparse
import difflib
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))
SR = 16000

WORDS_PER_SEC = 2.6      # conversational English; sets how much anchor a window can hold
PARTS = ["Pre-Show", "Part I", "Break", "Part II", "Post-Show"]


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def decode(path: Path, start_s: float = 0.0, dur_s: float | None = None) -> np.ndarray:
    """Mono float32 @16k from any container, seeking rather than decoding the lot.

    A four-hour mp3 at 44.1kHz stereo is ~2.5GB decoded; alignment only ever needs a handful
    of thirty-second windows, so seeking is the difference between a minute and an hour.
    """
    import av
    out = []
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        if start_s:
            c.seek(int(start_s / st.time_base), stream=st)
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        want = int(dur_s * SR) if dur_s else None
        got = 0
        for frame in c.decode(st):
            for r in rs.resample(frame):
                a = r.to_ndarray().reshape(-1)
                out.append(a); got += len(a)
            if want and got >= want:
                break
    if not out:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(out).astype(np.float32) / 32768.0
    return pcm[:want] if want else pcm


def duration(path: Path) -> float:
    import av
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        return float(st.duration * st.time_base) if st.duration else float(c.duration / av.time_base)


def energy_profile(path: Path, step_s: float = 1.0) -> np.ndarray:
    """One RMS value per second across the whole episode, decoded once, cheaply."""
    import av
    vals, buf = [], []
    need = int(SR * step_s)
    with av.open(str(path)) as c:
        st = c.streams.audio[0]
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for frame in c.decode(st):
            for r in rs.resample(frame):
                buf.append(r.to_ndarray().reshape(-1))
                total = sum(len(b) for b in buf)
                while total >= need:
                    chunk = np.concatenate(buf)
                    vals.append(float(np.sqrt((chunk[:need].astype(np.float32) / 32768.0) ** 2).mean()))
                    rest = chunk[need:]
                    buf = [rest] if len(rest) else []
                    total = len(rest)
    return np.array(vals)


def quiet_runs(rms: np.ndarray, min_len_s: int = 6) -> list:
    """Sustained low-energy stretches: the seams between segments of the show."""
    thr = max(float(np.quantile(rms, 0.15)) * 1.6, float(np.quantile(rms, 0.4)) * 0.35)
    runs, start = [], None
    for i, v in enumerate(rms):
        if v < thr and start is None:
            start = i
        elif v >= thr and start is not None:
            if i - start >= min_len_s:
                runs.append((start, i))
            start = None
    if start is not None and len(rms) - start >= min_len_s:
        runs.append((start, len(rms)))
    return runs


def part_words(md_path: Path) -> dict:
    """Total spoken words under each heading.

    This is the prior that makes alignment affordable. Searching 40% of a four-hour episode at
    the resolution needed costs ~1,200 transcriptions per boundary; nine episodes would run
    for hours. But the transcript already says how much talking precedes each section, and
    speech rate is near enough constant, so each boundary can be ESTIMATED to within a few
    minutes and only that neighbourhood searched. Seams in the audio can't do this job -- the
    break is music, not silence, so it doesn't show up as a quiet run at all.
    """
    out, cur = {}, None
    for line in md_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        h = re.match(r"^#{2,}\s*\**\s*(.+?)\s*\**\s*$", s)
        if h:
            name = h.group(1).strip()
            cur = next((p for p in PARTS if p.lower() == name.lower()), None)
            if cur:
                out.setdefault(cur, 0)
            continue
        if cur is None:
            continue
        m = re.match(r"^\*{0,2}([A-Z][A-Za-z' .-]{1,24}):\*{0,2}\s+(.*\S)\s*$", s)
        if m:
            out[cur] += len(m.group(2).split())
    return out


def estimate_starts(md_path: Path, total_s: float) -> dict:
    """Predicted start time of each section, from cumulative word counts."""
    counts = part_words(md_path)
    present = [p for p in PARTS if p in counts]
    total_words = sum(counts[p] for p in present) or 1
    # The break is mostly music and adverts with little transcribed speech, so word count
    # under-represents its real duration; give it a floor rather than letting Part II drift early.
    est, acc = {}, 0.0
    for p in present:
        est[p] = acc / total_words * total_s
        acc += counts[p]
    return {"starts": est, "words": counts, "words_per_sec": total_words / max(total_s, 1)}


def part_text(md_path: Path) -> dict:
    """First lines of dialogue under each heading, for anchoring."""
    out, cur = {}, None
    for line in md_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        h = re.match(r"^#{2,}\s*\**\s*(.+?)\s*\**\s*$", s)
        if h:
            name = h.group(1).strip()
            cur = next((p for p in PARTS if p.lower() == name.lower()), None)
            if cur:
                out.setdefault(cur, [])
            continue
        if cur is None:
            continue
        m = re.match(r"^\*{0,2}([A-Z][A-Za-z' .-]{1,24}):\*{0,2}\s+(.*\S)\s*$", s)
        if m and len(out[cur]) < 12:
            out[cur].append(m.group(2))
    return out


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z']+", (s or "").lower()))


def _bigrams(s: str) -> set:
    w = s.split()
    return set(zip(w, w[1:]))


def _containment(anchor: str, heard: str) -> float:
    """How much of the anchor appears in the window, by word bigram.

    Character-level similarity was the wrong measure and scored real matches at 0.2: the
    window is a fixed 30 seconds and the anchor is a few lines, so the two strings differ
    wildly in length and SequenceMatcher punishes that regardless of content. Containment asks
    the question that actually matters -- is the anchor text IN here -- and is unbothered by
    the window carrying extra speech either side of it. Bigrams rather than single words so
    that common function words cannot manufacture a match.
    """
    a, h = _bigrams(anchor), _bigrams(heard)
    if not a:
        return 0.0
    return len(a & h) / len(a)


def find_anchor(path: Path, anchor: str, lo_s: float, hi_s: float,
                stt, window_s: float = 30.0, step_s: float = 45.0) -> tuple:
    """Scan a range for the window containing `anchor`, then refine around the best hit.

    Coarse-to-fine: a wide sweep to find the neighbourhood, then a tight one to place the
    boundary. Scanning the whole range at fine resolution costs many times more for the same
    answer, and the coarse pass is what makes searching 45 minutes of audio affordable.
    """
    target = _norm(anchor)
    if not target:
        return None, 0.0

    # A step wider than the window leaves gaps: at window=30 and step=90 two thirds of the
    # timeline was never looked at, so the anchor was usually skipped over entirely and the
    # best score came from whatever unrelated window happened to be sampled. Overlap the
    # windows instead, so every second of audio is covered by at least one of them.
    step_s = min(step_s, window_s * 0.5)

    def sweep(a, b, step):
        best, best_t = 0.0, None
        t = a
        while t < b:
            pcm = decode(path, t, window_s)
            if len(pcm) < SR:
                break
            r = _containment(target, _norm(stt.transcribe(pcm, "")))
            if r > best:
                best, best_t = r, t
            t += step
        return best_t, best

    # The anchor must be no longer than the window can physically contain. Part I opens with a
    # 349-word recap monologue; matched against a 30s window holding ~75 words, containment
    # could never exceed ~25% no matter how well it lined up, and every boundary was rejected
    # for looking like a bad match when it was actually a length mismatch.
    target = " ".join(target.split()[:int(window_s * WORDS_PER_SEC)])

    t0, s0 = sweep(lo_s, hi_s, step_s)
    if t0 is None:
        return None, 0.0
    # refine within one coarse step either side, at a third of the resolution
    t1, s1 = sweep(max(lo_s, t0 - step_s), min(hi_s, t0 + step_s), max(step_s / 3, 8.0))
    return (t1, round(s1, 3)) if s1 >= s0 else (t0, round(s0, 3))


def cmd_inspect(a) -> int:
    src = Path(a.audio)
    if not src.exists():
        return _fail(f"no audio at {src}")
    total = duration(src)
    rms = energy_profile(src)
    runs = quiet_runs(rms, a.min_gap)
    runs.sort(key=lambda r: (r[1] - r[0]), reverse=True)
    print(json.dumps({"audio": src.name, "minutes": round(total / 60, 1),
                      "longest_quiet_runs": [{"start_min": round(s / 60, 2),
                                              "end_min": round(e / 60, 2),
                                              "length_s": e - s} for s, e in runs[:12]],
                      "note": "intro, break and outro usually appear among these"}, indent=2))
    return 0


def cmd_align(a) -> int:
    src, md = Path(a.audio), Path(a.transcript)
    if not src.exists():
        return _fail(f"no audio at {src}")
    if not md.exists():
        return _fail(f"no transcript at {md}")
    import stt
    stt.warmup()
    total = duration(src)
    parts = part_text(md)

    # Search a band around the word-count estimate rather than a fixed fraction of the episode.
    # The estimate is only approximate -- speech rate varies and the break carries little
    # transcribed speech for its real duration -- so the band is wide enough to absorb that,
    # while still being a tenth of the audio a blind sweep would transcribe.
    pred = estimate_starts(md, total)
    band = a.band * 60
    plan = [(name, max(0.0, t - band), min(total - 30, t + band))
            for name, t in pred["starts"].items() if name != "Pre-Show"]
    found = {}
    for name, lo, hi in plan:
        lines = parts.get(name) or []
        if not lines:
            found[name] = {"seconds": None, "confidence": 0.0, "reason": "no such section"}
            continue
        t, conf = find_anchor(src, " ".join(lines[:4]), lo, hi, stt,
                              window_s=a.window, step_s=a.step)
        found[name] = {"seconds": round(t, 1) if t is not None else None,
                       "minutes": round(t / 60, 2) if t is not None else None,
                       "confidence": conf,
                       "accepted": bool(t is not None and conf >= a.min_confidence)}
    out = {"audio": src.name, "minutes": round(total / 60, 1),
           "sections_in_transcript": sorted(parts), "boundaries": found,
           "min_confidence": a.min_confidence}
    dest = Path(a.out or (ROOT / "data" / "align")) / (src.stem + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["saved"] = str(dest)
    print(json.dumps(out, indent=2))
    return 0


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())


def cmd_crop(a) -> int:
    aligns = sorted(Path(a.align_dir).glob("*.json")) if a.all else [Path(a.align)]
    if not aligns or not aligns[0].exists():
        return _fail("no alignment files — run `align` first")
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for af in aligns:
        info = json.loads(af.read_text(encoding="utf-8"))
        src = next((p for p in Path(a.audio_dir).glob("*.mp3") if p.stem == af.stem), None)
        if src is None:
            skipped.append({"align": af.name, "why": "audio not found"}); continue
        b = info["boundaries"]
        spans = []
        # The break is music and adverts carrying almost no transcribed speech, so it has
        # nothing to anchor on and is routinely rejected. That must not cost us Part I, whose
        # END it defines. Fall back to Part II's confident start, pulled back by the break's
        # own estimated length — cutting Part I slightly short is harmless; running it into
        # the break would feed sponsor reads to a turn-taking model as if they were play.
        p1_end = None
        if b.get("Break", {}).get("accepted"):
            p1_end = b["Break"]["seconds"]
        elif b.get("Part II", {}).get("accepted") and b.get("Break", {}).get("seconds"):
            gap = b["Part II"]["seconds"] - b["Break"]["seconds"]
            p1_end = b["Part II"]["seconds"] - max(gap, 5 * 60)
        if b.get("Part I", {}).get("accepted") and p1_end:
            spans.append(("partI", b["Part I"]["seconds"], p1_end))
        if b.get("Part II", {}).get("accepted"):
            end = b["Post-Show"]["seconds"] if b.get("Post-Show", {}).get("accepted") \
                else info["minutes"] * 60
            spans.append(("partII", b["Part II"]["seconds"], end))
        if not spans:
            skipped.append({"align": af.name, "why": "no confident boundaries"}); continue
        for label, s, e in spans:
            if e is None or s is None or e - s < 60:
                skipped.append({"align": af.name, "why": f"{label} span too short"}); continue
            pcm = decode(src, s, e - s)
            dest = out_root / f"{af.stem}__{label}.wav"
            write_wav(dest, pcm)
            written.append({"file": dest.name, "minutes": round(len(pcm) / SR / 60, 1),
                            "from_min": round(s / 60, 2), "to_min": round(e / 60, 2)})
    print(json.dumps({"written": written, "skipped": skipped, "out": str(out_root)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect"); s.add_argument("--audio", required=True)
    s.add_argument("--min-gap", type=int, default=6); s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("align")
    s.add_argument("--audio", required=True); s.add_argument("--transcript", required=True)
    s.add_argument("--out", default=""); s.add_argument("--window", type=float, default=30.0)
    s.add_argument("--step", type=float, default=45.0)
    s.add_argument("--min-confidence", type=float, default=0.45)
    s.add_argument("--band", type=float, default=6.0,
                   help="minutes either side of the word-count estimate to search")
    s.set_defaults(fn=cmd_align)

    s = sub.add_parser("crop")
    s.add_argument("--align", default=""); s.add_argument("--all", action="store_true")
    s.add_argument("--align-dir", default=str(ROOT / "data" / "align"))
    s.add_argument("--audio-dir", default=str(ROOT / "critical_role_sessions"))
    s.add_argument("--out", default=str(ROOT / "data" / "clean"))
    s.set_defaults(fn=cmd_crop)

    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
