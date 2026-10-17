"""Build the per-turn prompt deliberately, instead of sending everything every time.

The system prompt is fixed at spawn (`--append-system-prompt`), so it cannot vary per turn.
What DOES go out on every single turn is the state block, and on a real campaign that is 8,039
characters — roughly 2,200 tokens per turn to carry a 32-character player line.

Two things make it that big, and only one of them is legitimate:

  the world bible   append-only, written by the director every session, sent in full forever.
                    By session 10 it is larger than the rules. Nothing ever expired.
  bookkeeping       the director writes SYSTEM notes into it — "seat-join duplicates",
                    "sessions 4-7 opened with no in-world events". That is debugging output
                    living in the model's context and costing tokens on every turn, and it
                    actively distracts from the fiction.

So: keep the bible entries that bear on the scene in front of us, keep the most recent
regardless, drop the bookkeeping, and cap the whole thing. Everything dropped is still on
disk — this trims what is SENT, never what is stored.
"""
import re

# Notes that are about the software rather than the world. These are the director explaining
# itself, and they should never have been in the fiction's memory.
_BOOKKEEPING = re.compile(
    r"seat[- ]join|duplicate|joined twice|no in-world events|session \d+ opens|"
    r"opened with no|roster|voice id|placeholder|prep, not table history|unplayed prep",
    re.I)

_TAG = re.compile(r"^\[(\w+)\]\s*")


def _entries(bible: str) -> list:
    """Split the bible into entries. Written as one blob with CANON markers, not a list."""
    if not bible:
        return []
    parts = re.split(r"\n(?=\[|CANON\b)", bible)
    return [p.strip() for p in parts if p.strip()]


def _keywords(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())}


def relevant_bible(bible: str, scene: str, location: str, quests=(),
                   keep_recent: int = 4, budget: int = 2500) -> tuple:
    """Choose which canon to carry into this turn. Returns (text, manifest).

    Recency alone is wrong — a fact established in session 1 can be the point of session 6 —
    and relevance alone is wrong, because a brand-new scene shares no words with anything.
    So: always keep the newest few, then fill the remaining budget with whatever overlaps
    what is actually happening.
    """
    entries = _entries(bible)
    if not entries:
        return "", {"kept": 0, "dropped": 0, "bookkeeping_dropped": 0, "chars": 0}

    kept_book = [e for e in entries if _BOOKKEEPING.search(e)]
    story = [e for e in entries if not _BOOKKEEPING.search(e)]

    context = _keywords(f"{scene} {location} {' '.join(quests)}")
    recent = story[-keep_recent:] if keep_recent else []
    rest = story[:-keep_recent] if keep_recent else story
    scored = sorted(
        ((len(_keywords(e) & context) / max(len(_keywords(e)), 1), i, e)
         for i, e in enumerate(rest)),
        key=lambda t: (-t[0], -t[1]))

    chosen, used = list(recent), sum(len(e) + 1 for e in recent)
    for score, _i, e in scored:
        if score <= 0 or used + len(e) > budget:
            continue
        chosen.append(e); used += len(e) + 1
    # restore original order so the world reads chronologically, not by score
    order = {e: i for i, e in enumerate(story)}
    chosen.sort(key=lambda e: order.get(e, 0))
    return "\n".join(chosen), {
        "kept": len(chosen), "dropped": len(story) - len(chosen),
        "bookkeeping_dropped": len(kept_book), "chars": used,
        "total_entries": len(entries)}


def compact_party(combatants, focus: str = "") -> str:
    """One line each, with detail only where it is needed this turn.

    Spell lists and inventories for five characters are most of the party block, and almost
    all of it is irrelevant to any given turn. The character acting gets their numbers; the
    rest get enough to be described and addressed.
    """
    lines = []
    for c in combatants:
        if not c.is_pc:
            continue
        head = f"  {c.name} (L{c.level}): {c.status} AC {c.ac}"
        if focus and c.name.lower() == focus.lower():
            prepared = [s["name"] for s in (c.spells or []) if s.get("prepared")][:12]
            worn = [i["name"] for i in (c.inventory or []) if i.get("equipped")]
            head += f" | {c.mods}"
            if worn:
                head += f" | equipped: {', '.join(worn)}"
            if prepared:
                head += f" | prepared: {', '.join(prepared)}"
        lines.append(head)
    return "\n".join(lines)


def budget_report(state, scene: str = "", location: str = "") -> dict:
    """What the per-turn block costs now, and what it would cost trimmed."""
    full = state.summary()
    trimmed, manifest = relevant_bible(getattr(state, "bible", ""),
                                       scene or getattr(state, "scene", ""),
                                       location or getattr(state, "location", ""),
                                       getattr(state, "quests", ()))
    saved = len(getattr(state, "bible", "")) - len(trimmed)
    return {"state_chars": len(full), "bible_chars": len(getattr(state, "bible", "")),
            "bible_kept_chars": len(trimmed), "bible_saved_chars": saved,
            "manifest": manifest,
            "est_tokens_saved_per_turn": int(saved / 3.7)}
