"""Can we rebuild the transcript from the audio alone? Measure it, don't assume it.

Every model in this project rests on two claims that have never been tested against ground
truth: that Whisper hears the words, and that the speaker clustering knows who said them. A
published transcript with speaker labels makes both falsifiable.

  wer       transcribe audio, score words against the reference       -> word error rate
  diarize   cluster speakers, map clusters to real names, score       -> attribution accuracy
  recreate  do both and emit a reconstructed transcript + a scorecard

Reported separately on purpose. A system can hear every word and still attribute half of them
to the wrong person, and that failure -- which is what the table actually complained about --
is invisible in WER.

Mapping clusters to names uses the Hungarian assignment over a cluster x speaker overlap
matrix, so "cluster 3 is Matt" is chosen globally rather than greedily. Greedy assignment
inflates the score by letting the dominant speaker absorb ambiguous clusters.

Usage
  python tools/eval_stt.py wer      --audio data/clean/<ep>__partI.wav --transcript <ep>.md
  python tools/eval_stt.py recreate --audio <wav> --transcript <md> --minutes 10
"""
import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
SR = 16000


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def words(s: str) -> list:
    return re.findall(r"[a-z']+", (s or "").lower())


def wer(ref: list, hyp: list) -> dict:
    """Levenshtein over word sequences. O(len(ref) * len(hyp)) with a rolling row."""
    if not ref:
        return {"wer": 1.0 if hyp else 0.0, "ref_words": 0, "hyp_words": len(hyp)}
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    d = prev[-1]
    return {"wer": round(d / len(ref), 4), "errors": d,
            "ref_words": len(ref), "hyp_words": len(hyp)}


def load_wav(path: Path, minutes: float = 0.0) -> np.ndarray:
    with wave.open(str(path)) as w:
        n = w.getnframes()
        if minutes:
            n = min(n, int(minutes * 60 * w.getframerate()))
        pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return pcm


def reference_text(md: Path, part: str = "") -> list:
    """Reference words, optionally limited to one section of the show."""
    import fetch_transcripts as ft
    lines, cur, keep = [], None, True
    for raw in md.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        h = re.match(r"^#{2,}\s*\**\s*(.+?)\s*\**\s*$", s)
        if h:
            cur = h.group(1).strip().lower()
            keep = (not part) or (cur == part.lower())
            continue
        if not keep:
            continue
        m = re.match(r"^\*{0,2}([A-Z][A-Za-z' .-]{1,24}):\*{0,2}\s+(.*\S)\s*$", s)
        if m:
            lines.append({"speaker": m.group(1).strip().upper(), "text": m.group(2)})
    return lines


def transcribe_segments(pcm: np.ndarray, want_embeddings: bool = False) -> list:
    """Segment -> transcribe -> embed, the same path the corpus builder uses."""
    import corpus
    import speakers
    import stt
    stt.warmup()
    out = []
    for s, e, _synth in corpus.speech_regions(pcm):
        clip = pcm[int(s * SR):int(e * SR)]
        text = (stt.transcribe(clip, "") or "").strip()
        if not text:
            continue
        row = {"start": round(s, 2), "end": round(e, 2), "text": text}
        if want_embeddings:
            emb = speakers.embed(clip)
            row["emb"] = emb.tolist() if emb is not None else None
        out.append(row)
    return out


def align_words(hyp: list, ref: list) -> list:
    """Pair hypothesis words with reference words by edit-distance alignment.

    Returns [(hyp_index, ref_index)] for MATCHED words only.

    Pairing by position was wrong and quietly understated everything built on it. At a word
    error rate of 0.27 an insertion early on shifts every later word by one, so comparing
    hyp[i] against ref[i] across seventeen hundred words measures drift far more than it
    measures attribution. Two model comparisons were run against that metric before this was
    noticed; both verdicts had to be re-checked.

    Only matched positions are scored. A word Whisper never heard says nothing about whether
    the speaker was identified correctly, so it is excluded rather than counted as an error.
    """
    n, m = len(hyp), len(ref)
    if not n or not m:
        return []
    # Hirschberg would be tidier, but n*m here is ~1.7k x 1.7k which is nothing; keep the
    # simple table so the backtrace is obvious.
    import numpy as _np
    d = _np.zeros((n + 1, m + 1), dtype=_np.int32)
    d[:, 0] = _np.arange(n + 1)
    d[0, :] = _np.arange(m + 1)
    for i in range(1, n + 1):
        hi = hyp[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (hi != ref[j - 1]))
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if hyp[i - 1] == ref[j - 1] and d[i][j] == d[i - 1][j - 1]:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i -= 1; j -= 1                     # substitution: not a reliable pairing
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def assign_clusters(pred_labels: list, true_labels: list, n_clusters: int) -> dict:
    """Globally optimal cluster -> speaker map (Hungarian on the overlap matrix)."""
    from scipy.optimize import linear_sum_assignment
    names = sorted({t for t in true_labels if t})
    if not names:
        return {}
    M = np.zeros((n_clusters, len(names)))
    for c, t in zip(pred_labels, true_labels):
        if t and 0 <= c < n_clusters:
            M[c, names.index(t)] += 1
    r, col = linear_sum_assignment(-M)
    return {int(i): names[j] for i, j in zip(r, col)}


def cmd_wer(a) -> int:
    wav, md = Path(a.audio), Path(a.transcript)
    if not wav.exists():
        return _fail(f"no audio at {wav}")
    if not md.exists():
        return _fail(f"no transcript at {md}")
    pcm = load_wav(wav, a.minutes)
    segs = transcribe_segments(pcm)
    hyp = words(" ".join(s["text"] for s in segs))
    ref_rows = reference_text(md, a.part)
    ref = words(" ".join(r["text"] for r in ref_rows))
    # Only comparable when the same span is being compared; a 10-minute slice against a
    # 2-hour reference is meaningless, so scale the reference to the audio actually scored.
    if a.minutes and ref:
        ref = ref[:int(len(ref) * min(1.0, (a.minutes * 60) / max(_span(segs), 1)))] or ref
    print(json.dumps({"audio": wav.name, "minutes_scored": round(len(pcm) / SR / 60, 2),
                      "segments": len(segs), **wer(ref, hyp)}, indent=2))
    return 0


def _span(segs: list) -> float:
    return (segs[-1]["end"] - segs[0]["start"]) if segs else 0.0


def cmd_recreate(a) -> int:
    """Rebuild 'SPEAKER: line' from audio only, and score it against the real transcript."""
    import corpus
    wav, md = Path(a.audio), Path(a.transcript)
    if not wav.exists() or not md.exists():
        return _fail("need both --audio and --transcript")
    pcm = load_wav(wav, a.minutes)
    segs = transcribe_segments(pcm, want_embeddings=True)
    if not segs:
        return _fail("no speech found")

    rows = corpus.cluster_speakers([dict(s) for s in segs], a.speakers)
    labels = [r.get("speaker", -1) for r in rows]

    ref_rows = reference_text(md, a.part)
    # Align on the text so cluster->name mapping is scored against the right reference line
    ref_words_all = [(r["speaker"], w) for r in ref_rows for w in words(r["text"])]
    hyp_words = [(labels[i], w) for i, s in enumerate(segs) for w in words(s["text"])]
    # Align on the actual word sequences, not by position -- see align_words.
    pairs = align_words([w for _c, w in hyp_words], [w for _t, w in ref_words_all])
    n = len(pairs)
    mapping = assign_clusters([hyp_words[i][0] for i, _j in pairs],
                              [ref_words_all[j][0] for _i, j in pairs], a.speakers)
    correct = sum(1 for i, j in pairs
                  if mapping.get(hyp_words[i][0]) == ref_words_all[j][0])

    recon = [{"start": s["start"], "speaker": mapping.get(labels[i], f"SPK{labels[i]}"),
              "text": s["text"]} for i, s in enumerate(segs)]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(
            "\n".join(f"{r['speaker']}: {r['text']}" for r in recon), encoding="utf-8")

    w = wer(words(" ".join(r["text"] for r in ref_rows))[:n or None],
            words(" ".join(s["text"] for s in segs)))
    print(json.dumps({
        "audio": wav.name, "minutes_scored": round(len(pcm) / SR / 60, 2),
        "segments": len(segs), "clusters": a.speakers,
        "cluster_to_speaker": {str(k): v for k, v in mapping.items()},
        "word_error_rate": w["wer"],
        "speaker_attribution_accuracy": round(correct / max(n, 1), 4),
        "words_compared": n,
        "wrote": a.out or None,
        "note": "WER and attribution fail independently; a perfect WER can still name the "
                "wrong person for half the lines",
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("wer", cmd_wer), ("recreate", cmd_recreate)):
        s = sub.add_parser(name)
        s.add_argument("--audio", required=True)
        s.add_argument("--transcript", required=True)
        s.add_argument("--part", default="", help="e.g. 'Part I' to score one section")
        s.add_argument("--minutes", type=float, default=0.0, help="0 = all")
        s.add_argument("--speakers", type=int, default=8)
        s.add_argument("--out", default="")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
