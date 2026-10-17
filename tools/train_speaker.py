"""Train and measure the speaker model instead of trusting it.

The live book was built from ONE utterance per player and claimed a name on a 0.06 margin
while the enrolled profiles sat up to 0.749 similar to each other -- so it named the wrong
person confidently, and then trained on its own mistakes. The margin is fixed; what is still
missing is any measurement of how well it can possibly do.

  enroll    build profiles from several clips per speaker, not one
  eval      score attribution on held-out clips: accuracy, and how often it declines
  margin    sweep the decision margin, so the operating point is chosen rather than guessed
  separation  how far apart the current profiles actually are -- the ceiling on everything else

DECLINING TO ANSWER IS A VALID OUTPUT and is scored as such. An unlabelled line is visibly
unattributed; a wrongly-named one reads as fact and is worse than silence.

Usage
  python tools/train_speaker.py separation --campaign "The Drowned Crown"
  python tools/train_speaker.py enroll --clips data/voices --out state/voices/trained.json
  python tools/train_speaker.py eval   --book state/voices/trained.json --clips data/voices
  python tools/train_speaker.py margin --book state/voices/trained.json --clips data/voices
"""
import argparse
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


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def clips_by_speaker(root: Path) -> dict:
    """data/voices/<speaker>/*.wav — one directory per person."""
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.wav"))
        if files:
            out[d.name] = files
    return out


def embed_all(files: list) -> list:
    import speakers
    out = []
    for f in files:
        e = speakers.embed(load_wav(f))
        if e is not None:
            out.append(np.asarray(e, dtype=float))
    return out


def cmd_separation(a) -> int:
    """How distinguishable the enrolled voices are. This bounds every other number here."""
    book = ROOT / "state" / "voices" / f"{a.campaign}.json"
    if not book.exists():
        return _fail(f"no voice book at {book}")
    d = json.loads(book.read_text(encoding="utf-8"))
    names = list(d)
    if len(names) < 2:
        return _fail("need at least two enrolled voices")
    E = {}
    for n in names:
        v = np.array(d[n]["emb"], dtype=float)
        E[n] = v / (np.linalg.norm(v) or 1)
    pairs = [{"a": a_, "b": b_, "similarity": round(float(np.dot(E[a_], E[b_])), 3)}
             for i, a_ in enumerate(names) for b_ in names[i + 1:]]
    worst = max(pairs, key=lambda p: p["similarity"])
    sims = [p["similarity"] for p in pairs]
    print(json.dumps({
        "campaign": a.campaign, "speakers": names,
        "clips_per_speaker": {n: len(d[n].get("clips", [])) for n in names},
        "mean_similarity": round(float(np.mean(sims)), 3),
        "closest_pair": worst,
        "safe_margin_floor": round(worst["similarity"], 3),
        "pairs": sorted(pairs, key=lambda p: -p["similarity"]),
        "note": "a decision margin below the closest pair's similarity cannot separate them; "
                "the live default was 0.06 against a closest pair of 0.749",
    }, indent=2))
    return 0


def cmd_enroll(a) -> int:
    root = Path(a.clips)
    if not root.is_dir():
        return _fail(f"no clip directory at {root} (expected <speaker>/*.wav)")
    groups = clips_by_speaker(root)
    if not groups:
        return _fail("no per-speaker subdirectories with wavs")
    book, report = {}, []
    for name, files in groups.items():
        keep = files[:-a.holdout] if a.holdout and len(files) > a.holdout else files
        embs = embed_all(keep)
        if not embs:
            report.append({"speaker": name, "error": "no usable clips"}); continue
        # Mean of unit vectors, renormalised: a single clip captures one moment of one mood,
        # which is why the live profiles were so close together.
        v = np.mean([e / (np.linalg.norm(e) or 1) for e in embs], axis=0)
        v /= (np.linalg.norm(v) or 1)
        book[name.lower()] = {"emb": v.tolist(), "label": name,
                              "clips": [f.name for f in keep]}
        report.append({"speaker": name, "enrolled_from": len(embs),
                       "held_out": len(files) - len(keep)})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(book), encoding="utf-8")
    print(json.dumps({"out": str(out), "speakers": len(book), "detail": report}, indent=2))
    return 0


def _score(book: dict, groups: dict, threshold: float, margin: float, holdout: int) -> dict:
    profs = {k: np.array(v["emb"], dtype=float) for k, v in book.items()}
    for k in profs:
        profs[k] /= (np.linalg.norm(profs[k]) or 1)
    right = wrong = declined = 0
    for name, files in groups.items():
        test = files[-holdout:] if holdout and len(files) > holdout else files
        for e in embed_all(test):
            e = e / (np.linalg.norm(e) or 1)
            scored = sorted(((float(np.dot(e, p)), k) for k, p in profs.items()), reverse=True)
            top = scored[0]
            second = scored[1][0] if len(scored) > 1 else -1.0
            if top[0] >= threshold and (top[0] - second) >= margin:
                right += (book[top[1]]["label"].lower() == name.lower())
                wrong += (book[top[1]]["label"].lower() != name.lower())
            else:
                declined += 1
    n = max(right + wrong + declined, 1)
    named = max(right + wrong, 1)
    return {"threshold": threshold, "margin": margin, "clips": n,
            "correct": right, "wrong": wrong, "declined": declined,
            "precision_when_named": round(right / named, 3),
            "coverage": round((right + wrong) / n, 3),
            "wrong_rate": round(wrong / n, 3)}


def cmd_eval(a) -> int:
    bp, cp = Path(a.book), Path(a.clips)
    if not bp.exists():
        return _fail(f"no book at {bp}")
    book = json.loads(bp.read_text(encoding="utf-8"))
    groups = clips_by_speaker(cp)
    if not groups:
        return _fail("no clips to evaluate")
    print(json.dumps({"book": bp.name,
                      **_score(book, groups, a.threshold, a.margin, a.holdout),
                      "note": "declining is not an error; a wrong name is worse than no name"},
                     indent=2))
    return 0


def cmd_margin(a) -> int:
    bp, cp = Path(a.book), Path(a.clips)
    book = json.loads(bp.read_text(encoding="utf-8"))
    groups = clips_by_speaker(cp)
    rows = [_score(book, groups, a.threshold, m, a.holdout)
            for m in [round(x, 2) for x in np.arange(0.02, 0.41, 0.03)]]
    ok = [r for r in rows if r["precision_when_named"] >= a.target_precision]
    print(json.dumps({"sweep": rows, "target_precision": a.target_precision,
                      "recommended": max(ok, key=lambda r: r["coverage"]) if ok else None,
                      "live_default_margin": 0.15}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("separation"); s.add_argument("--campaign", default="The Drowned Crown")
    s.set_defaults(fn=cmd_separation)
    s = sub.add_parser("enroll")
    s.add_argument("--clips", default=str(ROOT / "data" / "voices"))
    s.add_argument("--out", default=str(ROOT / "state" / "voices" / "trained.json"))
    s.add_argument("--holdout", type=int, default=1)
    s.set_defaults(fn=cmd_enroll)
    for name, fn in (("eval", cmd_eval), ("margin", cmd_margin)):
        s = sub.add_parser(name)
        s.add_argument("--book", default=str(ROOT / "state" / "voices" / "trained.json"))
        s.add_argument("--clips", default=str(ROOT / "data" / "voices"))
        s.add_argument("--threshold", type=float, default=0.55)
        s.add_argument("--margin", type=float, default=0.15)
        s.add_argument("--holdout", type=int, default=1)
        s.add_argument("--target-precision", type=float, default=0.95)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
