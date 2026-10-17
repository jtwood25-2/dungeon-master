"""What is in the DM's prompt, and what is being paid for on turns that never use it.

DM_SYS is ~24,000 characters. It is sent once per session rather than once per turn, so it is
not the runaway cost the old telemetry implied -- but it is still the single largest thing in
the model's context for the whole session, and a long context costs latency on every turn even
when cached.

The question this answers is not "how big is it" but "which parts earn their place". A block
of grappling rules is dead weight across a two-hour session with no grapple in it, and the
transcripts say how often each block is actually exercised.

  sections  split the prompt into blocks and size them
  usage     cross-reference each block's trigger words against real transcript history
  state     what state.summary() costs per turn, by field, on a real campaign

Usage
  python tools/prompt_audit.py sections
  python tools/prompt_audit.py usage --campaign "The Drowned Crown"
  python tools/prompt_audit.py state --campaign "The Drowned Crown"
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

CHARS_PER_TOKEN = 3.7          # matches server/telemetry.py so the numbers are comparable


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def sections(text: str) -> list:
    """Split on blank-line-separated blocks headed by an ALL-CAPS or bracketed line."""
    out, cur = [], {"title": "(preamble)", "body": []}
    for line in text.splitlines():
        s = line.strip()
        head = (s and (s.isupper() and len(s) > 3 and len(s) < 80)) or re.match(r"^\[[A-Z]+", s)
        if head and cur["body"]:
            out.append(cur)
            cur = {"title": s[:70], "body": []}
        elif head:
            cur["title"] = s[:70]
        else:
            cur["body"].append(line)
    out.append(cur)
    return [{"title": c["title"], "chars": len("\n".join(c["body"])),
             "lines": len(c["body"])} for c in out if c["body"]]


def cmd_sections(a) -> int:
    import backend
    secs = sorted(sections(backend.DM_SYS), key=lambda s: -s["chars"])
    total = len(backend.DM_SYS)
    print(json.dumps({
        "total_chars": total, "total_tokens_est": int(total / CHARS_PER_TOKEN),
        "sections": len(secs),
        "largest": [{**s, "pct": round(s["chars"] / total * 100, 1)} for s in secs[:20]],
    }, indent=2))
    return 0


# blocks worth measuring: the trigger words that mean this rule was needed
TOPICS = {
    "combat": ["attack", "initiative", "damage", "hit points", "ac ", "roll to hit"],
    "death saves": ["death save", "dying", "stabilize", "unconscious"],
    "concentration": ["concentration", "concentrating"],
    "rests": ["long rest", "short rest", "hit die", "hit dice"],
    "spells/slots": ["spell slot", "cast", "prepared", "cantrip"],
    "loot/gold": ["gold", "gp", "loot", "treasure"],
    "maps": ["map", "square", "grid", "move to"],
    "level up": ["level up", "levelled", "leveled"],
    "inspiration": ["inspiration", "inspire"],
    "downtime": ["downtime", "between sessions"],
    "skill checks": ["perception", "investigation", "insight", "athletics", "stealth"],
}


def cmd_usage(a) -> int:
    """How often each rule block is actually exercised, from real transcripts."""
    import transcript as tr
    files = sorted(Path(tr.DIR).glob("*.jsonl"))
    if a.campaign:
        files = [f for f in files if f.name.startswith(a.campaign)]
    files = [f for f in files if "unittest" not in f.name]
    if not files:
        return _fail("no transcripts to measure against")
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    blob = " ".join((r.get("text") or "").lower() for r in rows)
    turns = sum(1 for r in rows if r.get("role") == "dm")
    out = []
    for topic, keys in TOPICS.items():
        hits = sum(blob.count(k) for k in keys)
        out.append({"topic": topic, "mentions": hits,
                    "per_dm_turn": round(hits / max(turns, 1), 3)})
    out.sort(key=lambda r: r["mentions"])
    print(json.dumps({"transcript_files": len(files), "rows": len(rows), "dm_turns": turns,
                      "topics_least_used_first": out,
                      "note": "topics with ~0 mentions are candidates for loading only when "
                              "the situation arises, rather than sitting in context all session"},
                     indent=2))
    return 0


def cmd_state(a) -> int:
    """What the per-turn state block costs, by field — this one IS sent every turn."""
    import campaign as camp
    import game_state
    g = game_state.GameState.load(camp._safe(a.campaign))
    summary = g.summary()
    fields = {}
    for line in summary.splitlines():
        key = line.split(":", 1)[0].strip()[:28] if ":" in line else "(continuation)"
        fields[key] = fields.get(key, 0) + len(line) + 1
    big = sorted(fields.items(), key=lambda kv: -kv[1])
    per_char = {c.name: len(f"{c.name} {c.status} {c.mods}") for c in g.combatants.values()}
    print(json.dumps({
        "campaign": a.campaign, "state_chars": len(summary),
        "state_tokens_est": int(len(summary) / CHARS_PER_TOKEN),
        "largest_lines": [{"field": k, "chars": v} for k, v in big[:14]],
        "per_character_chars": per_char,
        "note": "this block is rebuilt and sent on EVERY turn, unlike DM_SYS",
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sections"); s.set_defaults(fn=cmd_sections)
    s = sub.add_parser("usage"); s.add_argument("--campaign", default="")
    s.set_defaults(fn=cmd_usage)
    s = sub.add_parser("state"); s.add_argument("--campaign", default="The Drowned Crown")
    s.set_defaults(fn=cmd_state)
    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
