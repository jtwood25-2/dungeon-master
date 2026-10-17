"""Measure narration style, so the prompt rules come from Matt's actual prose rather than
from my impression of it.

The complaint is "it sounds like AI" and "too much fluff that drags on". Both are measurable.
This compares MATT's in-game narration in the Critical Role corpus against our own DM output
in state/transcripts, and reports the handful of dimensions that actually separate them.

The pre-show, the ad reads and the table chatter are not narration and would poison every
number here, so Matt's lines are filtered to the ones that are actually description: after the
game starts, not a question to a player, not a one-word answer.

Usage
  python tools/style.py                          # corpus vs our latest session
  python tools/style.py --ours state/transcripts/'Beyond the Grove 2026-08-23.jsonl'
  python tools/style.py --samples 12             # print example lines from each side
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Openers that mark a line as table-talk rather than narration: Matt answering a rules
# question or asking for a roll is not the prose we are trying to imitate.
_TABLE = re.compile(
    r"^\s*(?:yeah|yep|no|nope|okay|ok|sure|right|correct|exactly|so|and|but|well|"
    r"go ahead|make a|roll|what do you|what are you|as you|hold on|one sec|"
    r"that(?:'s| is)|it(?:'s| is) a|you can|you could|does that|did that|how many|"
    r"which|who(?:'s| is)|all right|alright)\b", re.I)
_ASKS = re.compile(r"\?\s*$")
_SENT = re.compile(r"(?<=[.!?])\s+")

# The tells. Each is a pattern that shows up far more in generated prose than in a person's,
# and each is separately fixable in a prompt, which is the point of counting them apart.
TELLS = {
    # "not a whisper, but a roar" -- the antithesis reflex
    "not_x_but_y": re.compile(r"\bnot\s+(?:a|an|the)?\s*\w+[^.!?]{0,40}?,?\s+but\s+", re.I),
    # "something moves", "somewhere ahead" -- vagueness standing in for detail
    "vague_some": re.compile(r"\b(?:something|somewhere|somehow|someone)\b", re.I),
    # "cold, wet, and wrong" -- the rule of three
    "triad": re.compile(r"\b\w+,\s+\w+,\s+and\s+\w+\b"),
    # "-- wrong for a hole in the ground" -- the trailing em-dash gloss
    "em_dash": re.compile(r"\s[-—–]{1,2}\s"),
    # "like a hand at your back" -- simile
    "simile": re.compile(r"\b(?:like|as if|as though)\s+(?:a|an|the|it|something)\b", re.I),
    # "gone gray-white", "gone to slick mud" -- the literary "gone X" construction
    "gone_x": re.compile(r"\bgone\s+(?:to\s+)?\w+", re.I),
    # participial scene-setting opener: "Standing there, you see..."
    "participle_open": re.compile(r"^\s*\w+ing\b[^.!?]{0,30},", re.I),
    # "the way the spring runs on a cold morning" -- the explanatory tail
    "the_way_that": re.compile(r"\bthe way (?:the|a|an|it|you|he|she|they)\b", re.I),
    # doesn't/don't belong, shouldn't be -- the ominous negation
    "wrongness": re.compile(r"\b(?:doesn't|does not|don't|shouldn't|didn't)\s+"
                            r"(?:belong|move|stop|look|feel|sound)\b", re.I),
}

# words that carry no information in narration -- pure texture
FLUFF = re.compile(
    r"\b(?:slowly|softly|faintly|gently|quietly|barely|somehow|almost|nearly|just|"
    r"utterly|deeply|impossibly|unnaturally|strangely|oddly|eerily|dimly|"
    r"ancient|eldritch|primal|profound|palpable|tangible|unmistakable)\b", re.I)


def _words(s):
    return re.findall(r"[A-Za-z']+", s or "")


def _sentences(s):
    return [x.strip() for x in _SENT.split((s or "").strip()) if x.strip()]


def measure(lines: list[str]) -> dict:
    """Per-line and per-sentence shape, plus the rate of each tell."""
    lines = [l for l in lines if l and l.strip()]
    if not lines:
        return {}
    sents, wps = [], []
    for l in lines:
        ss = _sentences(l)
        sents.extend(ss)
        wps.append(len(_words(l)))
    slen = [len(_words(s)) for s in sents] or [0]
    body = "\n".join(lines)
    nw = max(1, len(_words(body)))
    out = {
        "lines": len(lines),
        "sentences": len(sents),
        "words": nw,
        "words_per_line": round(st.mean(wps), 1),
        "words_per_sentence": round(st.mean(slen), 1),
        "median_words_per_sentence": round(st.median(slen), 1),
        "p90_words_per_sentence": round(sorted(slen)[int(len(slen) * .9)], 1),
        "sentences_per_line": round(len(sents) / len(lines), 2),
        # per 1000 words so the two corpora are comparable despite very different sizes
        "fluff_per_1k": round(1000 * len(FLUFF.findall(body)) / nw, 1),
    }
    for name, pat in TELLS.items():
        out[f"{name}_per_1k"] = round(1000 * len(pat.findall(body)) / nw, 1)
    return out


def matt_narration(min_words: int = 12) -> list[str]:
    """Matt's in-game descriptive prose, with table-talk and the pre-show filtered out."""
    out = []
    for p in sorted((ROOT / "data" / "transcripts").glob("*.speakers.jsonl")):
        rows = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        # The pre-show and ad read run until the first long Matt block after someone says the
        # show is starting; simplest robust proxy is to drop the first 8% of each episode.
        rows = rows[int(len(rows) * .08):]
        for r in rows:
            if r.get("speaker") != "MATT":
                continue
            t = (r.get("text") or "").strip()
            if len(_words(t)) < min_words or _ASKS.search(t) or _TABLE.match(t):
                continue
            out.append(t)
    return out


def ours(path: Path | None) -> list[str]:
    """Our DM's spoken lines. Consecutive dm rows are one delivered turn, so they are joined
    -- measuring them apart would make our sentences look shorter than they are."""
    if path is None:
        tr = sorted((ROOT / "state" / "transcripts").glob("*.jsonl"),
                    key=lambda p: p.stat().st_mtime)
        if not tr:
            return []
        path = tr[-1]
    turns, cur = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("role") or r.get("speaker")) == "dm":
            cur.append((r.get("text") or "").strip())
        elif cur:
            turns.append(" ".join(cur)); cur = []
    if cur:
        turns.append(" ".join(cur))
    # strip the NPC dialogue cue lines -- those are voice-acted, not narration
    return [re.sub(r"^\s*\w[\w' ]{0,20}:\s*\".*?\"\s*$", "", t).strip() for t in turns]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", default=None, help="one of our session transcripts")
    p.add_argument("--samples", type=int, default=0, help="print N example lines per side")
    p.add_argument("--min-words", type=int, default=12)
    a = p.parse_args()

    matt = matt_narration(a.min_words)
    mine = ours(Path(a.ours) if a.ours else None)
    m, o = measure(matt), measure(mine)
    if not m:
        print(json.dumps({"error": "no corpus narration found"})); return 1

    keys = [k for k in m if k not in ("lines", "sentences", "words")]
    delta = {k: round(o.get(k, 0) - m[k], 2) for k in keys} if o else {}
    # the tells worth acting on: ours is at least 50% above Matt AND the gap is material
    worst = sorted(((k, o.get(k, 0), m[k]) for k in keys if k.endswith("_per_1k")),
                   key=lambda t: -(t[1] - t[2]))
    out = {"matt": m, "ours": o, "delta_ours_minus_matt": delta,
           "biggest_gaps": [{"metric": k, "ours": v, "matt": mv, "ratio":
                             round(v / mv, 2) if mv else None}
                            for k, v, mv in worst[:6] if v > mv]}
    if a.samples:
        out["samples"] = {"matt": matt[:a.samples], "ours": mine[:a.samples]}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
