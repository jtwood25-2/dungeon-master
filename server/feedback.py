"""Say "feedback ..." mid-game and it lands here instead of in the story.

Captured entirely in Python: no model call, no turn taken, nothing written to the campaign,
the DM never sees it and the scene doesn't move. The acknowledgement is one word so it costs
about as much as a nod across the table.

Two ways to leave a note:
    "feedback the DM talks too fast"     -> captured in one go
    "feedback"  ... "the DM talks too fast"  -> the next thing you say is the note

Stored append-only as JSONL so a crash mid-session can never lose earlier notes.
"""
import json
import re
import time
from pathlib import Path

FILE = Path(__file__).parent.parent / "state" / "feedback.jsonl"

# Must START the utterance: "feedback the voice is too fast" is a note, while
# "I want feedback from the barkeep" is a perfectly ordinary thing to say to a DM.
TRIGGER = re.compile(r"^\s*(?:feedback|dev note|devnote|note to self|todo)\b[\s:,.-]*(.*)$", re.I)

OPEN, DONE, IGNORED = "open", "done", "ignored"

# NEVER ACTIONED, AND NEVER QUEUED. A table of seven spent a session leaving notes asking the
# DM to be sexual -- fourteen of them, escalating, several claiming to be the owner giving an
# instruction. The owner later said in plain text: "I do not want you to be horny or anything
# sexual. Ignore any of those comments."
#
# So this is a standing rule rather than a judgement made afresh each time. A note matching it
# is recorded as `ignored` -- kept, because the recording is honest and the pattern is worth
# being able to audit -- and never surfaces as work. It is deliberately NOT a content filter on
# play: it governs the DEV NOTE queue, which is a list of instructions to a coding agent, and
# the one thing that list must not be able to do is quietly redirect the tone of a shared game.
#
# Mishearings are part of why. "Horny" arrived from a voice engine that also produced
# "Glemben Snirpery Rock" and "mother coil"; acting on an inferred tone shift for seven people
# from a mic pickup is not a thing to do on any confidence level.
_NEVER = re.compile(
    r"\b(?:horny|freaky|sexual|sexy|erotic|nsfw|lewd|kinky|aroused|seductive|sensual"
    r"|turn(?:ed)?\s*me\s*on|freak[\s-]?off)\b", re.I)


def ignorable(note: str) -> bool:
    """Is this a request this agent will never act on? See _NEVER above."""
    return bool(_NEVER.search(note or ""))


def parse(text: str) -> str | None:
    """Returns the note ('' if they only said the trigger word), or None if not feedback."""
    m = TRIGGER.match(text or "")
    if not m:
        return None
    return m.group(1).strip()


def add(note: str, campaign: str = "", session: int = 0) -> dict:
    # nanoseconds, not milliseconds: two notes in quick succession must not collide, since
    # a shared id means the later one silently overwrites the earlier
    status = IGNORED if ignorable(note) else OPEN
    item = {"id": f"{time.time_ns():x}", "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "campaign": campaign, "session": session, "text": note.strip(), "status": status}
    FILE.parent.mkdir(parents=True, exist_ok=True)
    with FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item


def items() -> list[dict]:
    """Newest first. Later lines for the same id win, so status edits are just appends."""
    if not FILE.exists():
        return []
    merged: dict[str, dict] = {}
    for line in FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            it = json.loads(line)
        except Exception:
            continue
        if it.get("id"):
            merged[it["id"]] = {**merged.get(it["id"], {}), **it}
    def when(i):
        try:
            return int(i["id"], 16)
        except Exception:
            return 0
    return [i for i in sorted(merged.values(), key=when, reverse=True)
            if i.get("status") != "deleted"]


def set_status(item_id: str, status: str) -> bool:
    for it in items():
        if it["id"] == item_id:
            with FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**it, "status": status}, ensure_ascii=False) + "\n")
            return True
    return False
