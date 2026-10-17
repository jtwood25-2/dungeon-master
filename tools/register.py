"""Was that the DM, or just Matt? Was that the character, or the player?

The training target so far was "did the DM speak within 2.5s", and it is noisier than it
looks. Matt laughing, arguing a rule, or saying "yeah exactly" all counted as the DM
responding -- 37% of his lines are under six words, which is not narration by any reading.
Meanwhile a player asking "do I still have my bonus action" is a real cue that needs an
answer, but it is a different KIND of cue from "I kick the door in".

So each line gets a register:

  DM_ROLE   describing the world, voicing an NPC, calling for a roll, making a ruling
  DM_TABLE  the DM as a person - joking, logistics, reacting
  PC_ROLE   a player in character, or declaring what their character does
  PC_TABLE  a player out of character - rules questions, table talk

The DM must answer both PC registers, so this is not a filter on the input. It sharpens the
TARGET: "the DM took the floor IN ROLE" is a much cleaner thing to predict than "Matt made a
noise", and the register of the preceding line is itself a feature.

HOW IT IS LABELLED. A local model (ollama) labels a sample, and a fast classifier is trained
on those labels and applied to everything. The local model is ~0.6s per line, which is eight
hours for the full corpus; the trained classifier is microseconds and runs live too. Crucially
the label choice is CONSTRAINED BY SPEAKER -- the DM can only be DM_ROLE or DM_TABLE - because
the unconstrained prompt put a DM's "make a perception check" into a player class.

Usage
  python tools/register.py label --n 600            # local LLM labels a sample
  python tools/register.py train                    # fast classifier from those labels
  python tools/register.py apply --root data/raw    # tag every segment
  python tools/register.py check                    # agreement + class balance
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))
LABELS_PATH = ROOT / "data" / "register_labels.jsonl"
MODEL_PATH = ROOT / "state" / "models" / "register"

DM_LABELS = ["DM_ROLE", "DM_TABLE"]
PC_LABELS = ["PC_ROLE", "PC_TABLE"]

# Few-shot, because the instruction alone was not enough: "make a perception check for me"
# came back DM_TABLE even though calling for a roll is named as DM_ROLE in the same prompt.
# Small local models follow examples far more reliably than definitions, and the procedural
# half of a DM's job is exactly what the definition kept losing.
_SYS_DM = (
    "You label one line spoken by the game master in a tabletop RPG recording.\n"
    "Answer with exactly one word: DM_ROLE or DM_TABLE.\n\n"
    "DM_ROLE  - running the game: describing the world, voicing an NPC, calling for a roll,\n"
    "           stating a result, or making a ruling\n"
    "DM_TABLE - speaking as themselves: joking, logistics, agreeing, reacting\n\n"
    "Examples:\n"
    "The corridor opens into a low chamber. -> DM_ROLE\n"
    "Go ahead and make a perception check for me. -> DM_ROLE\n"
    "That hits, roll your damage. -> DM_ROLE\n"
    "He looks up and says he has not seen anyone in weeks. -> DM_ROLE\n"
    "No, advantage does not stack with that. -> DM_ROLE\n"
    "Yeah, exactly. -> DM_TABLE\n"
    "Okay hang on, let me find it. -> DM_TABLE\n"
    "That was amazing. -> DM_TABLE\n"
    "We will pick this up after the break. -> DM_TABLE")
_SYS_PC = (
    "You label one line spoken by a player in a tabletop RPG recording.\n"
    "Answer with exactly one word: PC_ROLE or PC_TABLE.\n\n"
    "PC_ROLE  - in character, or declaring what their character does in the world\n"
    "PC_TABLE - out of character: rules questions, table talk, reacting\n\n"
    "Examples:\n"
    "I want to check the door for traps. -> PC_ROLE\n"
    "I draw my sword and step in front of her. -> PC_ROLE\n"
    "Please, we do not want any trouble. -> PC_ROLE\n"
    "I cast guidance on him. -> PC_ROLE\n"
    "Wait, do I still have my bonus action? -> PC_TABLE\n"
    "Hold on, let me find my sheet. -> PC_TABLE\n"
    "That was a terrible roll. -> PC_TABLE\n"
    "What page is that on? -> PC_TABLE")


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def ask_local(line: str, is_dm: bool, model: str, timeout: float = 60) -> str:
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": _SYS_DM if is_dm else _SYS_PC},
                     {"role": "user", "content": line[:600]}],
        "options": {"temperature": 0, "num_predict": 8},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    allowed = DM_LABELS if is_dm else PC_LABELS
    up = (out or "").strip().upper()
    for lab in allowed:
        if lab in up:
            return lab
    return allowed[1]              # unparseable -> the "table" class, the safer default


def transcript_lines() -> list:
    """Speaker-labelled lines from the wiki transcripts: the only place the SPEAKER is known
    for certain, so it is where labels should be learned from."""
    out = []
    for f in sorted((ROOT / "data" / "transcripts").glob("*.speakers.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            spk, txt = r.get("speaker", ""), (r.get("text") or "").strip()
            if not txt or len(txt) < 2:
                continue
            # only single named speakers; "LAURA AND SAM" and "ALL" are crosstalk
            if not re.fullmatch(r"[A-Z][A-Z' .-]{1,20}", spk) or " AND " in spk:
                continue
            out.append({"speaker": spk, "text": txt, "is_dm": spk == "MATT"})
    return out


def cmd_label(a) -> int:
    rows = transcript_lines()
    if not rows:
        return _fail("no speaker-labelled transcripts; run fetch_transcripts pair first")
    random.seed(a.seed)
    # balance DM and player lines, and over-sample short lines: they are where the register
    # actually differs and where a length-based shortcut would otherwise fool the classifier
    dm = [r for r in rows if r["is_dm"]]
    pc = [r for r in rows if not r["is_dm"]]
    # Half DM, half players, each at its NATURAL length distribution. The first pass drew a
    # third of the sample from lines of six words or fewer; those are ~2% in-role, so it
    # pushed the measured DM_ROLE rate down to 6% as an artefact of sampling, not of the data.
    pick = (random.sample(dm, min(len(dm), a.n // 2))
            + random.sample(pc, min(len(pc), a.n // 2)))
    seen, sample = set(), []
    for r in pick:
        k = (r["speaker"], r["text"])
        if k not in seen:
            seen.add(k); sample.append(r)

    done = {}
    if LABELS_PATH.exists() and not a.force:
        for l in LABELS_PATH.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l); done[(d["speaker"], d["text"])] = d
    todo = [r for r in sample if (r["speaker"], r["text"]) not in done]
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with LABELS_PATH.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            try:
                lab = ask_local(r["text"], r["is_dm"], a.model)
            except Exception as e:
                print(f"  stopped at {i}: {type(e).__name__}: {e}", file=sys.stderr)
                break
            f.write(json.dumps({**r, "label": lab}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  {(time.time()-t0)/i:.2f}s/line", file=sys.stderr)
    total = sum(1 for l in LABELS_PATH.read_text(encoding="utf-8").splitlines() if l.strip())
    print(json.dumps({"labelled_now": len(todo), "labels_on_disk": total,
                      "out": str(LABELS_PATH)}))
    return 0


def _feats(text: str, is_dm: bool) -> dict:
    w = text.split()
    return {"n_words": len(w), "is_dm": int(is_dm),
            "has_quote": int('"' in text), "has_you": int(bool(re.search(r"\byou\b", text, re.I))),
            "has_i": int(bool(re.search(r"\bI\b", text))),
            "mechanics": int(bool(re.search(
                r"\b(roll|dc|check|damage|hit points|hp|initiative|saving throw|advantage|"
                r"turn|action|spell slot|bonus action)\b", text, re.I))),
            "question": int(text.strip().endswith("?"))}


FEATS = sorted(_feats("x", True))


def cmd_train(a) -> int:
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from scipy.sparse import hstack, csr_matrix
    if not LABELS_PATH.exists():
        return _fail("no labels; run `label` first")
    rows = [json.loads(l) for l in LABELS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 80:
        return _fail(f"only {len(rows)} labels; run `label --n 600`")
    # one model per speaker kind: the decision boundary is genuinely different, and it keeps
    # the classifier from having to relearn "who is talking" from the words
    out = {}
    report = {}
    for kind, is_dm in (("dm", True), ("pc", False)):
        sub = [r for r in rows if r["is_dm"] == is_dm]
        if len(sub) < 40:
            report[kind] = f"only {len(sub)} examples"; continue
        y = np.array([1 if r["label"].endswith("ROLE") else 0 for r in sub])
        texts = [r["text"] for r in sub]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=6000, sublinear_tf=True)
        T = vec.fit_transform(texts)
        N = csr_matrix(np.array([[_feats(r["text"], is_dm)[k] for k in FEATS] for r in sub], float))
        X = hstack([T, N]).tocsr()
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        cv = cross_val_score(clf, X, y, cv=5, scoring="roc_auc") if len(set(y)) > 1 else [0.0]
        clf.fit(X, y)
        out[kind] = {"vec": vec, "clf": clf}
        report[kind] = {"n": len(sub), "in_role_rate": round(float(y.mean()), 3),
                        "cv_auc": round(float(np.mean(cv)), 3)}
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": out, "feats": FEATS}, MODEL_PATH / "register.joblib")
    print(json.dumps({"trained": report, "out": str(MODEL_PATH)}, indent=2))
    return 0


def load_classifier():
    import joblib
    p = MODEL_PATH / "register.joblib"
    return joblib.load(p) if p.exists() else None


def classify(bundle, text: str, is_dm: bool) -> str:
    """Fast path used by `apply` and, later, by the live server."""
    import numpy as np
    from scipy.sparse import hstack, csr_matrix
    kind = "dm" if is_dm else "pc"
    m = (bundle or {}).get("models", {}).get(kind)
    if not m:
        return ("DM_TABLE" if is_dm else "PC_TABLE")
    T = m["vec"].transform([text])
    N = csr_matrix(np.array([[_feats(text, is_dm)[k] for k in bundle["feats"]]], float))
    p = m["clf"].predict_proba(hstack([T, N]).tocsr())[0, 1]
    role = p >= 0.5
    return ("DM_ROLE" if role else "DM_TABLE") if is_dm else ("PC_ROLE" if role else "PC_TABLE")


def cmd_apply(a) -> int:
    bundle = load_classifier()
    if bundle is None:
        return _fail("no trained classifier; run `train` first")
    done = []
    for sess in sorted(p for p in Path(a.root).iterdir() if (p / "segments.jsonl").exists()):
        f = sess / "segments.jsonl"
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        # segments.jsonl carries no speaker: clustering runs inside `label` and only reaches
        # examples.jsonl. Reading r["speaker"] here compared None to -1, so every line was
        # treated as a player and not one DM_ROLE tag was produced. Cluster here, the same way
        # corpus.cmd_label does, so both agree on who the DM is.
        import corpus
        rows = corpus.cluster_speakers(rows, a.speakers)
        talk = {}
        for r in rows:
            talk[r.get("speaker", -1)] = talk.get(r.get("speaker", -1), 0.0) + r.get("dur", 0)
        dm = max(talk, key=talk.get) if talk else -1
        for r in rows:
            r["register"] = classify(bundle, r.get("text") or "", r.get("speaker") == dm)
        f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        done.append({"session": sess.name, "segments": len(rows),
                     "dm_in_role": sum(1 for r in rows if r["register"] == "DM_ROLE")})
    print(json.dumps({"sessions": len(done), "detail": done}, indent=2))
    return 0


def cmd_check(a) -> int:
    if not LABELS_PATH.exists():
        return _fail("no labels yet")
    rows = [json.loads(l) for l in LABELS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    short = [r for r in rows if len(r["text"].split()) <= 6]
    print(json.dumps({
        "labels": len(rows), "distribution": dict(c),
        "short_lines": len(short),
        "short_in_role_rate": round(sum(1 for r in short if r["label"].endswith("ROLE"))
                                    / max(len(short), 1), 3),
        "note": "if short lines are almost never in-role, the register is doing its job: "
                "those are the DM's 'yeah exactly' that used to count as a response",
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("label"); s.add_argument("--n", type=int, default=600)
    s.add_argument("--model", default="qwen2.5:7b-instruct")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_label)
    s = sub.add_parser("train"); s.set_defaults(fn=cmd_train)
    s = sub.add_parser("apply"); s.add_argument("--root", default=str(ROOT / "data" / "raw"))
    s.add_argument("--speakers", type=int, default=8)
    s.set_defaults(fn=cmd_apply)
    s = sub.add_parser("check"); s.set_defaults(fn=cmd_check)
    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
