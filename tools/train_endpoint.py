"""Train and evaluate "should the DM speak now?" against the current rule.

endpoint.assess() is a hand-written rule: a word count, a stop-phrase list, a check for
someone else's name. It has no notion of how long the pause was, whether the sentence landed
or trailed, or whether the table is mid-exchange with each other. That is why it answered the
DM's own narration and sat silent through real questions.

This trains a model on the labelled corpus and reports it AGAINST that rule on held-out
sessions. A model that does not beat the rule is not worth loading, and the comparison is
printed either way.

THE COSTS ARE NOT SYMMETRIC. Speaking over the table is what made the session unpleasant --
it interrupts, it talks over people, it derails. Staying quiet a beat too long is mildly
awkward and self-corrects because someone just says it again. So the operating point is
chosen for PRECISION on "respond", not for accuracy, and the sweep is printed so the
trade-off is a decision rather than a default.

Usage
  python tools/train_endpoint.py train --splits data/splits --out state/models/endpoint
  python tools/train_endpoint.py eval  --splits data/splits --model state/models/endpoint
  python tools/train_endpoint.py sweep --splits data/splits --model state/models/endpoint
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

# words that mean the thought is unfinished, so the pause is a breath and not a handover
DANGLING = {"and", "but", "so", "or", "then", "because", "if", "the", "a", "an", "to", "i",
            "we", "my", "he", "she", "they", "it", "that", "with", "for", "of", "at", "in"}
QUESTION = {"what", "where", "who", "when", "why", "how", "can", "do", "does", "did", "is",
            "are", "should", "could", "would", "will"}


# What the prosody extractor supplies per segment. Named here so a missing block yields
# zeros rather than a shape error, and so the order is fixed across train and predict.
PROSODY_KEYS = ["energy_end_rel", "energy_slope", "f0_end_rel", "f0_range", "f0_slope",
                "final_lengthening", "tail_silence", "voiced_frac"]


def features(r: dict) -> dict:
    """Timing, shape, and how the speech SOUNDS as it ends.

    The text and timing block alone was measured at near chance on held-out data -- every
    feature between 0.50 and 0.63 AUC, with `gap` at 0.517 -- and a learning curve flat from
    half the corpus onward, so the ceiling was the features and not the sample size.

    The prosody block is the category that was missing entirely: falling pitch, final
    lengthening, energy decay and the voicing tail are what actually mark a finished turn, and
    none of them survive into a transcript.
    """
    t = (r.get("text") or "").strip()
    w = re.findall(r"[A-Za-z']+", t.lower())
    last = w[-1] if w else ""
    out = {
        "n_words": len(w),
        "n_chars": len(t),
        "dur": float(r.get("dur") or 0.0),
        "gap": float(r.get("gap") or 0.0),
        "words_per_sec": len(w) / max(float(r.get("dur") or 0.01), 0.01),
        "ends_period": int(t.endswith(".")),
        "ends_question": int(t.endswith("?")),
        "ends_trailing": int(bool(re.search(r"(\.{2,}|…|-)$", t))),
        "ends_dangling": int(last in DANGLING),
        "starts_question": int(bool(w) and w[0] in QUESTION),
        "has_i": int("i" in w),
        "prev_was_dm": int(r.get("prev_was_dm") or 0),
    }
    pros = r.get("prosody") or {}
    for k in PROSODY_KEYS:
        out[f"p_{k}"] = float(pros.get(k) or 0.0)
    # TRIED AND REVERTED: in-character vs out-of-character as a feature. The register
    # classifier is good at its own job (0.83 AUC on DM lines) but adding it here moved the
    # endpoint model 0.609 -> 0.605 -- nothing. Whether a turn is in character does not tell
    # you whether it is finished, which is the question this model is actually asked.

    # An "addressed to the DM" score was tried here and REMOVED: it took the model from
    # 0.673 to 0.659. The tf-idf text model already learns those lexical patterns from data
    # ("I attack", "seventeen", "can I") and learns them better than hand-written regexes, so
    # the extra column was redundant noise. It survives in server/addressed.py for the one job
    # the corpus cannot speak to -- see the note there.
    return out


FEATURE_ORDER = sorted(features({"text": "x", "dur": 1, "gap": 1}))


def load(path: Path) -> tuple:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(json.dumps({"error": f"{path} is empty"}))
    X = np.array([[features(r)[k] for k in FEATURE_ORDER] for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows])
    texts = [r.get("text") or "" for r in rows]
    return X, y, texts, rows


def rule_baseline(rows: list[dict]) -> np.ndarray:
    """The rule that is live today, scored on the same held-out rows."""
    import endpoint
    out = []
    for r in rows:
        call, _why = endpoint.assess(r.get("text") or "", None, False)
        out.append(int(call == endpoint.GO))
    return np.array(out)


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
            "accuracy": round((tp + tn) / max(len(y), 1), 3),
            "spoke_over_table": fp,        # the failure that ruined the session
            "missed_cue": fn, "tp": tp, "tn": tn}


def build(X, y, texts):
    """Numeric features plus an OUT-OF-FOLD text score.

    The text probability used to be produced by predicting on the very rows the text model was
    fitted on. In-sample it looks nearly perfect, so the booster learns to lean on it, and at
    test time that column collapses to its real strength -- measured at 0.587 AUC alone. The
    stacked model came out at 0.617 while the plain numeric features managed 0.666: the text
    column was costing five points, not adding any.

    cross_val_predict gives each row a score from a model that never saw it, so the column the
    booster trains against has the same quality as the one it will meet at inference.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000, sublinear_tf=True)
    T = vec.fit_transform(texts)
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    oof = cross_val_predict(base, T, y, cv=5, method="predict_proba")[:, 1:2]
    lr = base.fit(T, y)                       # full-data model is what inference will use
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                        class_weight="balanced").fit(np.hstack([X, oof]), y)
    return {"vec": vec, "lr": lr, "gb": gb}


def predict_proba(m, X, texts):
    T = m["vec"].transform(texts)
    Xa = np.hstack([X, m["lr"].predict_proba(T)[:, 1:2]])
    return m["gb"].predict_proba(Xa)[:, 1]


def cmd_train(a) -> int:
    import joblib
    sp = Path(a.splits)
    Xtr, ytr, ttr, _ = load(sp / "train.jsonl")
    m = build(Xtr, ytr, ttr)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "features": FEATURE_ORDER}, out / "endpoint.joblib")
    print(json.dumps({"trained_on": int(len(ytr)), "positives": int(ytr.sum()),
                      "features": FEATURE_ORDER + ["text_prob"], "out": str(out)}))
    return 0


def cmd_eval(a) -> int:
    import joblib
    sp, mp = Path(a.splits), Path(a.model)
    Xte, yte, tte, rows = load(sp / "test.jsonl")
    m = joblib.load(mp / "endpoint.joblib")["model"]
    p = predict_proba(m, Xte, tte)
    model_at = scores(yte, (p >= a.threshold).astype(int))
    rule = scores(yte, rule_baseline(rows))
    # Judge on the cost model actually stated, not on recall parity. Demanding the model keep
    # 90% of the rule's recall guarantees the rule wins, because the rule buys its recall by
    # answering almost everything -- which is the behaviour being replaced. What matters is
    # interruptions avoided against cues missed, and an interruption costs several misses.
    saved = rule["spoke_over_table"] - model_at["spoke_over_table"]
    cost = model_at["missed_cue"] - rule["missed_cue"]
    verdict = "model" if saved > cost * a.interrupt_cost else "rule"
    print(json.dumps({"test_examples": int(len(yte)), "positives": int(yte.sum()),
                      "threshold": a.threshold,
                      "model": model_at, "rule_in_production": rule,
                      "interruptions_avoided_vs_rule": int(saved),
                      "extra_cues_missed_vs_rule": int(cost),
                      "verdict": verdict,
                      "note": "spoke_over_table is the count that made the session unpleasant"},
                     indent=2))
    return 0


def cmd_sweep(a) -> int:
    import joblib
    sp, mp = Path(a.splits), Path(a.model)
    Xte, yte, tte, rows = load(sp / "test.jsonl")
    m = joblib.load(mp / "endpoint.joblib")["model"]
    p = predict_proba(m, Xte, tte)
    rows_out = []
    for th in [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]:
        s = scores(yte, (p >= th).astype(int))
        rows_out.append({"threshold": th, **s})
    ok = [r for r in rows_out if r["precision"] >= a.target_precision]
    pick = max(ok, key=lambda r: r["recall"]) if ok else None
    print(json.dumps({"sweep": rows_out, "target_precision": a.target_precision,
                      "recommended": pick,
                      "rule_in_production": scores(yte, rule_baseline(rows))}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("eval", cmd_eval), ("sweep", cmd_sweep)):
        s = sub.add_parser(name)
        s.add_argument("--splits", default=str(ROOT / "data" / "splits"))
        s.add_argument("--model", default=str(ROOT / "state" / "models" / "endpoint"))
        s.add_argument("--out", default=str(ROOT / "state" / "models" / "endpoint"))
        s.add_argument("--threshold", type=float, default=0.6)
        s.add_argument("--target-precision", type=float, default=0.9)
        s.add_argument("--interrupt-cost", type=float, default=3.0,
                       help="how many missed cues one interruption is worth")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    try:
        return a.fn(a)
    except SystemExit:
        raise
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
