"""Opus writes the beats ahead of time; Sonnet plays between them.

The arc already exists — `state.plan` is an Opus-written campaign, and drift.py embeds it into
beats and measures how far the table has wandered from them. What was missing is the middle: a
beat was a *line of plan*, so when the party finally walked into a scene Opus had designed
months ago, Sonnet improvised it cold, at speed, with no more to go on than one summary line.

The expensive thinking is done when nobody is waiting:

  ahead-of-time   for the beats the party is approaching, Opus writes what is actually THERE —
                  the anchors of the scene, who is in it and what they want, the one feature
                  that makes it playable, and what it costs to walk away. Once per beat, ever.
  in the moment   Sonnet is handed that material and plays it against whatever the party
                  actually did. It interpolates; it does not recite.
  on departure    when drift says the table has left the arc, the prep for the beats ahead is
                  void, and Opus is asked again — which is the existing director path.

Material, deliberately, and not a script. A beat can be reached in an order nobody planned,
with the NPC already dead and the door already open, and prose written before that happened
would be confidently wrong. Recorded scene *facts* survive being arrived at sideways; recorded
scene *sentences* do not. The one thing voiced in advance is the recap, in recap.py, because a
recap is the only narration whose content cannot be invalidated by what the players do next.
"""
import hashlib
import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger("prepared")

ROOT = Path(__file__).parent.parent / "state" / "prepared"

# How many beats ahead to write. Two is the useful window: the beat they are walking into and
# the one after it. Further ahead is mostly wasted -- a party that has taken two unplanned turns
# has usually invalidated everything past that anyway.
LOOK_AHEAD = 2

SYS = (
    "You are the story architect for a live tabletop RPG, preparing a scene the party is about "
    "to reach. Another Dungeon Master will PLAY this scene from your notes, live, at speed — so "
    "write what is TRUE about it, never the words they should say.\n"
    "Give exactly these, each on its own line, no markdown and no headings:\n"
    "HERE: the place in one concrete sentence — what it physically is, and the one detail that "
    "makes it itself.\n"
    "WHO: anyone present, what each of them wants, and what they are doing when the party walks "
    "in. Named. If nobody is here, say so.\n"
    "PLAYABLE: the one feature that gives the players something to do with their hands — a "
    "thing to climb, open, break, bargain for, or refuse.\n"
    "PRESSURE: what happens if they do nothing, and roughly how soon.\n"
    "COSTS: what it costs them to leave without resolving this.\n"
    "SECRET: what is true here that they cannot yet perceive, and the specific thing that would "
    "reveal it.\n"
    "Concrete and physical throughout. Names, objects, distances. Never 'something', 'somewhere' "
    "or 'somehow'. No atmosphere standing in for detail. This is a briefing, not prose."
)


def _safe(campaign: str) -> str:
    return "".join(c for c in str(campaign) if c.isalnum() or c in " -_").strip() or "untitled"


def _path(campaign: str) -> Path:
    return ROOT / f"{_safe(campaign)}.json"


def key(beat: str) -> str:
    """Beats are identified by their text, so a re-planned arc invalidates only what changed."""
    return hashlib.sha1((beat or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def _load(campaign: str) -> dict:
    p = _path(campaign)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"prepared scenes unreadable ({e}); starting empty")
        return {}


def _save(campaign: str, data: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _path(campaign).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


SECTION = re.compile(r"^#+\s*(.+?)\s*$")
ITEM = re.compile(r"^\s*[-*\d.]+\s*\**(?P<name>[^*:—-]{2,60}?)\**\s*[—:-]\s*(?P<rest>.+)$")


def scenes(plan: str) -> list:
    """The places and beats worth preparing, as (name, description) pairs.

    NOT drift.beats(). That takes any line of six words or more, which is right for measuring
    similarity -- noise costs nothing there -- and wrong here, where every item becomes an Opus
    call. Asked for the next two beats of a real campaign it returned "Themes: innocence vs. a
    compromised world", and a briefing for a theme statement is a scene nobody can walk into.

    Locations are the reliable half. A place is knowable in advance in a way an event is not:
    the party may arrive having burned the bridge and killed the guide, and the room is still
    the same room. Story beats come second, and only the ones naming a session.
    """
    locations, beats_, section = [], [], ""
    for raw in (plan or "").splitlines():
        line = raw.rstrip()
        if m := SECTION.match(line.strip()):
            section = m.group(1).lower()
            continue
        low = section
        if not line.strip():
            continue
        if "location" in low or "place" in low:
            if m := ITEM.match(line):
                locations.append((m.group("name").strip(), line.strip()))
        elif "beat" in low or "session" in low or "act" in low:
            t = line.strip().strip("*")
            if t.lower().startswith("session") and len(t) < 80:
                beats_.append((t, t))
            elif len(t.split()) >= 8 and beats_:
                name, prev = beats_[-1]
                beats_[-1] = (name, (prev + " " + t)[:600])
    return locations + beats_


def upcoming(state, n: int = LOOK_AHEAD) -> list:
    """What to prepare next: everywhere they have not been prepared for, nearest first.

    Everywhere, not the next two. There are a handful of locations in a campaign, they are each
    prepared exactly once, and which one the party walks into next is precisely the thing nobody
    can predict -- so guessing wrong costs a scene improvised cold, while preparing them all
    costs a few Opus calls once per campaign.
    """
    items = scenes(getattr(state, "plan", "") or "")
    if not items:
        import drift
        return [(b[:60], b) for b in drift.beats(getattr(state, "plan", "") or "")][:n]
    where = f"{getattr(state, 'scene', '')} {getattr(state, 'map_loc', '')}".lower()
    items.sort(key=lambda it: 0 if it[0].lower() in where else 1)
    return items


def here(state) -> str:
    """The name of the prepared place the party is actually standing in, or ''.

    Matched by NAME against the live scene rather than by embedding: 'are they in Millhollow'
    has a right answer, and a similarity score is the wrong tool for a question that does.
    """
    where = (f"{getattr(state, 'scene', '')} {getattr(state, 'map_loc', '')} "
             + " ".join(list(getattr(state, "log", []) or [])[-4:])).lower()
    best = ""
    for name, _desc in scenes(getattr(state, "plan", "") or ""):
        n = name.lower().strip()
        if len(n) > 3 and n in where and len(n) > len(best):
            best = name
    return best


def brief(state, limit: int = 2) -> str:
    """Prepared material for where the party is now, for the DM's prompt. Never blocks."""
    have = _load(getattr(state, "campaign", "default"))
    out = []
    want = here(state)
    for name, desc in upcoming(state, limit):
        if want and name != want:
            continue
        if rec := have.get(key(desc)):
            if text := (rec.get("text") or "").strip():
                out.append(text)
    if not out:
        return ""
    return ("PREPARED SCENE (facts worked out in advance — play them against what the party "
            "actually did, in your own words; never read this out):\n" + "\n\n".join(out))


def missing(state, n: int = LOOK_AHEAD) -> list:
    """Beats ahead that have no prep yet."""
    have = _load(getattr(state, "campaign", "default"))
    return [(nm, d) for nm, d in upcoming(state, n) if key(d) not in have]


async def build(state, n: int = LOOK_AHEAD, force: bool = False) -> int:
    """Write prep for the beats ahead. Call when nobody is waiting. Returns how many were written."""
    camp = getattr(state, "campaign", "default")
    have = _load(camp)
    todo = upcoming(state, n) if force else missing(state, n)
    if not todo:
        return 0
    written = 0
    for name, desc in todo:
        text = await _write(state, desc)
        if not text:
            continue
        have[key(desc)] = {"beat": desc, "name": name, "text": text, "written": time.time()}
        written += 1
    if written:
        _save(camp, have)
        logger.info(f"prepared {written} scene(s) ahead for {camp!r}")
    return written


async def _write(state, beat: str) -> str:
    """Opus, once per beat, with nobody waiting on it."""
    import asyncio
    import config
    plan = getattr(state, "plan", "") or ""
    world = "\n".join(list(getattr(state, "bible", []) or [])[-25:])
    prompt = (f"{SYS}\n\n--- the campaign arc ---\n{plan}\n\n"
              f"--- established canon, which you must not contradict ---\n{world}\n\n"
              f"--- the beat to prepare ---\n{beat}\n\n"
              "Write the briefing now. Nothing else, no preamble.")
    try:
        proc = await asyncio.create_subprocess_exec(
            config.CLAUDE_BIN, "-p", "--model", config.PREPARE_CLAUDE_MODEL,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=300)
    except Exception as e:
        logger.warning(f"could not prepare {beat[:50]!r} ({e})")
        return ""
    text = (out or b"").decode("utf-8", "replace").strip()
    if not usable(text):
        logger.warning(f"prep for {beat[:40]!r} looked wrong ({text[:60]!r}); discarding")
        return ""
    return text


def usable(text: str) -> bool:
    """A briefing, or the model talking about itself?

    Pure, because the failure it guards against is silent: an apology stored as prep would be
    handed to the DM as fact for every turn the party spends in that scene.
    """
    t = (text or "").strip()
    if len(t) < 60:
        return False
    if t.lower().startswith(("i can", "i cannot", "i'm sorry", "im sorry", "i don't", "sorry",
                             "as an ai", "could you", "i would need")):
        return False
    # It must actually be the briefing that was asked for, not prose wearing its name.
    return sum(tag in t.upper() for tag in ("HERE:", "WHO:", "PLAYABLE:")) >= 2


def forget(campaign: str) -> int:
    """Drop everything prepared for a campaign — used when the arc itself is rewritten."""
    p = _path(campaign)
    n = len(_load(campaign))
    if p.exists():
        p.unlink()
    return n
