"""Pick DRIFT_THRESHOLD from evidence instead of from feel.

drift.py decides whether the party has wandered far enough off the planned arc to be worth an
Opus pass. That decision is a single number, and a number chosen by intuition is exactly the
kind of thing that quietly costs money or quietly drops the story layer.

So: score real turns against their OWN campaign's arc (these should read as on-arc) and against
every OTHER campaign's arc (these are off-arc by construction, since the events belong to a
different story entirely). The gap between the two distributions is the signal, and a threshold
that separates them is the one to ship.

If the two overlap badly there is no good threshold and the honest answer is to say so rather
than to pick the midpoint anyway.

Usage
  python tools/drift_calibrate.py
  python tools/drift_calibrate.py --window 8
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))


def _turns(path: Path, window: int) -> list[str]:
    """Rolling windows of what actually happened, the same shape drift.score() sees."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = (r.get("text") or "").strip()
        if t and (r.get("role") or r.get("speaker")) in ("dm", "player") or r.get("speaker"):
            rows.append(t)
    return [" ".join(rows[i:i + window]) for i in range(0, max(1, len(rows) - window), window)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=8, help="turns per measured window")
    a = p.parse_args()

    import campaign
    import drift
    from game_state import GameState

    # campaign -> (its arc beats, its real session windows)
    corpora = {}
    for slug in campaign.list_campaigns():
        name = slug if isinstance(slug, str) else slug.get("name") or slug.get("slug")
        try:
            g = GameState.load(name)
        except Exception:
            continue
        beats = drift.beats(getattr(g, "plan", "") or "")
        tr = sorted((ROOT / "state" / "transcripts").glob(f"{name} *.jsonl"))
        wins = [w for t in tr for w in _turns(t, a.window) if len(w.split()) > 15]
        if beats and wins:
            corpora[name] = (beats, wins)
    if len(corpora) < 1:
        print(json.dumps({"error": "no campaign has both a plan and a transcript"})); return 1

    vecs = {}
    for name, (beats, _w) in corpora.items():
        v = drift._embed(beats, timeout=60)
        if v:
            vecs[name] = list(zip(beats, v))

    same, cross = [], []
    for name, (_b, wins) in corpora.items():
        wv = drift._embed(wins, timeout=120)
        if not wv:
            continue
        for w in wv:
            for other, pairs in vecs.items():
                best = max(drift._cos(w, bv) for _bt, bv in pairs)
                (same if other == name else cross).append(round(1 - best, 3))

    # With only one planned campaign there are no cross-campaign negatives, and a threshold
    # fitted on positives alone is not a threshold at all. Critical Role is a different story
    # entirely, which is exactly what "off the arc" has to look like.
    if not cross:
        neg = []
        for p in sorted((ROOT / "data" / "transcripts").glob("*.speakers.jsonl"))[:4]:
            neg += _turns(p, a.window)[:40]
        nv = drift._embed(neg[:120], timeout=180) if neg else None
        for w in (nv or []):
            for _other, pairs in vecs.items():
                cross.append(round(1 - max(drift._cos(w, bv) for _bt, bv in pairs), 3))
        if cross:
            out_note = "negatives are Critical Role transcripts (a different story)"
        else:
            out_note = ""
    else:
        out_note = "negatives are other campaigns' real sessions"

    out = {"campaigns": {k: {"beats": len(v[0]), "windows": len(v[1])}
                         for k, v in corpora.items()},
           "on_arc": _dist(same), "off_arc": _dist(cross), "negatives": out_note}
    if same and cross:
        # the threshold that best separates them: sweep and take the best accuracy
        best = max(((t / 100,
                     (sum(1 for s in same if s < t / 100) + sum(1 for c in cross if c >= t / 100))
                     / (len(same) + len(cross)))
                    for t in range(5, 96)), key=lambda x: x[1])
        out["recommended_threshold"] = best[0]
        out["separation_accuracy"] = round(best[1], 3)
        out["verdict"] = (
            "usable" if best[1] >= .75 else
            "WEAK -- the two distributions overlap; gate on structural events instead")
        if same and cross:
            out["gap_medians"] = round(st.median(cross) - st.median(same), 3)
    print(json.dumps(out, indent=2))
    return 0


def _dist(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "min": s[0], "p25": s[len(s) // 4], "median": st.median(s),
            "p75": s[3 * len(s) // 4], "max": s[-1], "mean": round(st.mean(s), 3)}


if __name__ == "__main__":
    sys.exit(main())
