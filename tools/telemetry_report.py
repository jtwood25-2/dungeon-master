#!/usr/bin/env python
"""Summarise per-turn model telemetry: prompt size vs latency, by backend/model.

Reads state/telemetry.jsonl (written by server/telemetry.py) and reports the relationship
you'd want before tuning anything: how long turns take, how big the prompts are, and
whether latency actually tracks prompt size (correlation + latency by prompt-size bucket).

This only describes what happened -- it changes nothing and tunes nothing.

    python tools/telemetry_report.py --days 7
    python tools/telemetry_report.py --backend claude_code
"""
import argparse
import json
import statistics as st
import sys
from datetime import datetime, timedelta
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "state" / "telemetry.jsonl"


def _load(days: int, backend: str | None) -> list[dict]:
    if not LOG.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days)
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            if datetime.fromisoformat(r.get("ts", "")) < cutoff:
                continue
        except ValueError:
            pass
        if backend and r.get("backend") != backend:
            continue
        rows.append(r)
    return rows


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(num / (dx * dy), 3) if dx and dy else None


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[min(int(len(s) * p), len(s) - 1)], 1)


def _summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    tot = [r["ms_total"] for r in ok if r.get("ms_total") is not None]
    ftt = [r["ms_first_token"] for r in ok if r.get("ms_first_token") is not None]
    pch = [r["chars_prompt"] for r in ok if r.get("chars_prompt") is not None]
    out = {
        "turns": len(rows),
        "errors": len(rows) - len(ok),
        "ms_total": {"p50": _pct(tot, .5), "p90": _pct(tot, .9), "max": round(max(tot), 1) if tot else 0},
        "ms_first_token": {"p50": _pct(ftt, .5), "p90": _pct(ftt, .9)},
        "prompt_chars": {"p50": _pct(pch, .5), "p90": _pct(pch, .9), "max": max(pch) if pch else 0},
        "prompt_tokens_est": {"p50": _pct([r.get("tokens_prompt_est", 0) for r in ok], .5)},
        "reply_chars_p50": _pct([r.get("chars_reply", 0) for r in ok], .5),
    }
    # the question this data exists to answer
    pairs = [(r["chars_prompt"], r["ms_first_token"]) for r in ok
             if r.get("chars_prompt") and r.get("ms_first_token")]
    if len(pairs) >= 3:
        out["corr_prompt_chars_vs_first_token"] = _pearson([p[0] for p in pairs],
                                                           [p[1] for p in pairs])
    pairs_t = [(r["chars_prompt"], r["ms_total"]) for r in ok
               if r.get("chars_prompt") and r.get("ms_total")]
    if len(pairs_t) >= 3:
        out["corr_prompt_chars_vs_total"] = _pearson([p[0] for p in pairs_t],
                                                     [p[1] for p in pairs_t])
    # latency by prompt-size bucket -- the shape a correlation alone would hide
    if pairs_t:
        buckets: dict[str, list[float]] = {}
        for chars, ms in pairs_t:
            k = "0-2k" if chars < 2000 else "2-5k" if chars < 5000 else \
                "5-10k" if chars < 10000 else "10k+"
            buckets.setdefault(k, []).append(ms)
        out["ms_total_by_prompt_size"] = {
            k: {"n": len(v), "p50": _pct(v, .5)}
            for k, v in sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)}
    # which part of the prompt is actually big
    parts = ("chars_system", "chars_state", "chars_message", "chars_player", "chars_prepared")
    out["prompt_composition_p50"] = {p.replace("chars_", ""): _pct([r[p] for r in ok if p in r], .5)
                                     for p in parts if any(p in r for r in ok)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise DM model telemetry")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--backend", default=None, help="filter: claude_code | ollama | api")
    ap.add_argument("--by-model", action="store_true", help="break the summary out per model")
    args = ap.parse_args()

    rows = _load(args.days, args.backend)
    if not rows:
        print(json.dumps({"error": f"no telemetry in the last {args.days} days "
                                   f"({LOG} — play a session first)"}))
        sys.exit(1)

    out = {"log": str(LOG), "days": args.days, "overall": _summary(rows)}
    if args.by_model:
        models: dict[str, list[dict]] = {}
        for r in rows:
            models.setdefault(f'{r.get("backend","?")}/{r.get("model","?")}', []).append(r)
        out["by_model"] = {k: _summary(v) for k, v in models.items()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
