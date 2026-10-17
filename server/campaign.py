"""Campaign registry & session continuity.

Each campaign is a GameState persisted at state/<campaign>.json (world bible, roster,
plan, recap, quests, log). This module lists them, starts/loads one, and assembles the
"what has happened so far" memory the DM gets at the top of a session.
"""
import json
import shutil
import time
from pathlib import Path

import game_state
import maps

STATE = game_state.STATE_DIR


def list_campaigns() -> list[dict]:
    out = []
    if not STATE.exists():
        return out
    for p in sorted(STATE.glob("*.json")):
        if p.stem.startswith(("bench", "unittest", "smoketest", "betest", "rolltest", "_")):
            continue
        if p.stem.startswith("."):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        out.append({"name": p.stem, "premise": d.get("premise", ""),
                    "session": d.get("session", 0),
                    "party": [r.get("character") for r in d.get("roster", [])]})
    return out


def _path(name: str) -> Path:
    return STATE / f"{_safe(name)}.json"


def _safe(name: str) -> str:
    """Campaign names become filenames, so keep them to something that can't escape state/."""
    clean = "".join(ch for ch in str(name).strip() if ch.isalnum() or ch in " -_'")
    return clean.strip()[:60] or "untitled"


def detail(name: str) -> dict | None:
    """Everything the manage screen shows about one campaign."""
    p = _path(name)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    log = d.get("log") or []
    return {
        "name": p.stem,
        "premise": d.get("premise", ""),
        "session": d.get("session", 0),
        "recap": d.get("recap", ""),
        "party": [{"player": r.get("player"), "character": r.get("character")}
                  for r in (d.get("roster") or [])],
        "quests": d.get("quests") or [],
        "location": d.get("location", ""),
        "scene": d.get("scene", ""),
        "notes": len((d.get("bible") or "").splitlines()),
        # the phone must not re-derive this; both ends read one answer
        "unplayed": is_unplayed(type("S", (), d)),
        # one entry per session marker, newest first -- this is the "sessions" list
        "sessions": [ln for ln in log if ln.startswith("=== Session")][::-1][:40],
        "events": log[-12:][::-1],
    }


def create(name: str, premise: str = "") -> dict:
    g = game_state.GameState(_safe(name))
    g.premise = (premise or "").strip()
    g.save()
    return {"name": g.campaign, "premise": g.premise, "session": 0, "party": []}


def delete(name: str) -> bool:
    """Archive rather than destroy -- a campaign is a lot of play to lose to a mis-tap."""
    p = _path(name)
    if not p.exists():
        return False
    arc = STATE / "_archive" / time.strftime("%Y%m%d-%H%M%S")
    arc.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(arc / p.name))
    for sub in ("cast", "voices", "prep"):
        for ext in (".json", ".html"):
            side = STATE / sub / f"{p.stem}{ext}"
            if side.exists():
                (arc / sub).mkdir(parents=True, exist_ok=True)
                shutil.move(str(side), str(arc / sub / side.name))
    # the map library lives in its own directory and was being left behind, so a NEW
    # campaign created with the same name inherited the dead one's maps
    if (mdir := maps.ROOT / maps.slug(p.stem)).exists():
        shutil.move(str(mdir), str(arc / "maps"))
    return True


def reset(name: str) -> bool:
    """Clear what HAPPENED; keep what the campaign IS.

    This used to keep only the premise and the roster, which meant "clear history" quietly
    destroyed the imported character sheets — leaving a roster that named Oktah with no Oktah
    to go with it, so the DM had no AC, no slots and no skills to answer from. It also threw
    away the campaign arc, which is prep rather than history, and the table's lines and veils,
    which must never be lost to a button labelled "clear history".
    """
    p = _path(name)
    if not p.exists():
        return False
    old = game_state.GameState.load(_safe(name))
    g = game_state.GameState(_safe(name))
    g.premise, g.roster, g.plan = old.premise, old.roster, old.plan
    g.lines, g.veils = list(old.lines), list(old.veils)

    for key, c in old.combatants.items():
        if not c.is_pc:
            continue                                  # monsters and NPCs belong to the story
        c.hp = c.max_hp                               # they start the new run whole
        c.conditions, c.initiative, c.pos = [], None, []
        c.death_saves = {"success": 0, "fail": 0}
        c.concentrating, c.hit_dice_used = "", 0
        c.inspiration, c.pending_level = False, 0
        for s in c.slots.values():
            s["used"] = 0
        g.combatants[key] = c
    # the live map state is cleared by starting from a fresh GameState; the stored library
    # has to go with it, or a "cleared" campaign still remembers rooms it never visited
    maps.clear(_safe(name))
    g.save(force=True)                 # intentional wipe; save() backs the old copy up first
    return True


def new_session(name: str) -> bool:
    """End the current session so the next connect opens a genuinely NEW one.

    Sessions otherwise only roll over after 45 idle minutes, so "we're done for tonight" and
    "pick up where we left off" were the same thing to the app. This keeps everything and only
    closes the book: the next connect bumps the counter and opens with 'previously on'.
    """
    p = _path(name)
    if not p.exists():
        return False
    g = game_state.GameState.load(_safe(name))
    g.last_played = 0.0                # puts the next start_session outside the resume window
    g.in_combat, g.round, g.turn_idx = False, 0, 0
    g.order, g.pending_init = [], []
    g.save(touch=False)                # touch=True would undo the very thing we just did
    return True


def is_unplayed(g) -> bool:
    """Has anything actually happened in this campaign yet?

    The one question both ends need, and they used to answer it separately: the phone judged
    by the scene text, the server judged by whether the socket had resumed. Reload inside the
    resume window and they disagreed — the button offered to BEGIN the campaign while the
    server asked the DM to recap a session that never happened.

    The world bible is deliberately NOT a signal. The director runs at session_start and
    writes canon before a word is spoken, so a brand-new campaign that has merely been opened
    twice already has a bible — and judging by it sent the party into "recap and continue" for
    a session that never happened. A recap, a location, or a scene that has moved on are all
    consequences of play; prepared lore is not.
    """
    scene = (getattr(g, "scene", "") or "")
    return not (getattr(g, "recap", "") or getattr(g, "location", "")
                or (scene and not scene.startswith("Session not yet")))


def best_default() -> str:
    """Which campaign to open in: the one most recently PLAYED.

    This used to prefer any campaign with a roster, which meant a newly created campaign — no
    party yet, by definition — could never win. Selecting it, then reloading, snapped straight
    back to the old campaign, and it looked like the app was stuck on one game. last_played is
    stamped on every save, so switching is now sticky.
    """
    named = list_campaigns()
    if not named:
        return config_default()

    def played_at(c) -> float:
        try:
            d = json.loads(_path(c["name"]).read_text())
            return float(d.get("last_played") or 0)
        except Exception:
            return 0.0

    named.sort(key=lambda c: (played_at(c), _path(c["name"]).stat().st_mtime), reverse=True)
    return named[0]["name"]


def config_default() -> str:
    import config
    return config.DM_CAMPAIGN


RESUME_WINDOW = 45 * 60      # seconds


def start_session(name: str) -> game_state.GameState:
    """Load a campaign and continue it.

    Reconnecting is not the same as sitting down to play again. A dropped socket, a phone
    lock, or applying an update would each have started a "new session" — bumping the counter
    and cutting the thread mid-scene. If the last activity was recent we resume where we were;
    a real gap starts a genuine new session.
    """
    if not _path(name).exists():
        raise FileNotFoundError(
            f"campaign '{name}' does not exist — refusing to recreate it as an empty one. "
            "A deleted campaign must stay deleted; use create() to make a new one.")
    g = game_state.GameState.load(name)
    gap = time.time() - float(getattr(g, "last_played", 0) or 0)
    if g.session and gap < RESUME_WINDOW:
        g.resumed = True                      # picking the thread back up mid-scene
    else:
        g.session = int(getattr(g, "session", 0)) + 1
        g.resumed = False
        g._log(f"=== Session {g.session} begins ===")
    g.last_played = time.time()
    g.fresh = not g.resumed        # a genuinely new session opens with "previously on…"
    return g


def memory_brief(g: game_state.GameState) -> str:
    """The continuity block handed to the DM at session start so it 'remembers'.

    This was computed and never delivered — which is exactly why each session opened in a new
    place with a history nobody recognised. WHERE THEY ARE is stated first and as fact,
    because that is the thing the DM was most prone to inventing.
    """
    if is_unplayed(g):
        # A brand new campaign has nothing to remember. Asking for a "previously on" here
        # makes it invent a history on turn one, which is the opposite of continuity.
        parts = [f"This is the FIRST session of '{g.campaign}'. Nothing has happened yet — "
                 f"do not imply otherwise or reference events that never occurred."]
        if g.premise:
            parts.append(f"Premise: {g.premise}")
        if g.roster:
            parts.append("Party: " + "; ".join(f"{r['player']} plays {r['character']}"
                                               for r in g.roster))
        if g.plan:
            parts.append("YOUR CAMPAIGN ARC — where this is meant to go. Steer toward it "
                         "without railroading; if they go elsewhere, bend it to them:\n" + g.plan)
        parts.append(
            "OPEN THE CAMPAIGN properly — this is the cold open of an episode, not a summary. "
            "Take 6-10 sentences: where they are and what it smells and sounds like, who else "
            "is there and what they are doing, what is WRONG, and why it falls to them. Land "
            "on one concrete thing happening in front of them right now. Then set the scene — the place, "
            "what's wrong with it, and one concrete thing happening in front of them right "
            "now. Use [SCENE] to record where they are, then hand over.")
        return "\n".join(parts)

    parts = [f"You are CONTINUING the campaign '{g.campaign}'. This is session {g.session}."]
    if g.location or g.scene:
        parts.append(f"WHERE THEY ARE RIGHT NOW (do not move them, do not invent somewhere "
                     f"else): {g.location or '?'} — {g.scene}")
    if g.premise:
        parts.append(f"Premise: {g.premise}")
    if g.roster:
        parts.append("Party: " + "; ".join(f"{r['player']} plays {r['character']}"
                                           for r in g.roster))
    if g.recap:
        parts.append(f"What happened last time: {g.recap}")
    if g.bible:
        parts.append("Established canon — treat as true:\n" + g.bible)
    if g.plan:
        parts.append("YOUR CAMPAIGN ARC — steer toward it without railroading:\n" + g.plan)
    if g.quests:
        parts.append("Open threads: " + "; ".join(g.quests))
    npcs = [c.name for c in g.combatants.values() if not c.is_pc and c.hp > 0]
    if npcs:
        parts.append("Still present: " + ", ".join(npcs))
    parts.append(
        "OPEN THE SESSION with a 'previously on' recap in your own words — 3-5 sentences "
        "covering where they are, who's around, what they were doing and what's unresolved — "
        "then put them back exactly where they left off and hand over. Do NOT relocate them, "
        "invent a new scene, or contradict any of the above.")
    return "\n".join(parts)
