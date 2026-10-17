"""Structured actions — how the Claude Code (Pro) DM drives real mechanics.

The API backend uses native tool-calls; the CLI can't, so instead the DM writes bracketed
commands on their own lines and we parse+execute them against the real game state (via
tools.dispatch). Roll/lookup results are fed back so the DM reacts to real numbers.
"""
import re

import config
import tools as tool_mod

ACTION_HELP = """You control the game with ACTIONS. Put each action on its OWN line in square
brackets; write spoken narration on normal lines (1-3 short sentences). Actions:
  [ROLL 1d20+5]                  roll ONLY for your monsters/NPCs or the environment (2d6+3,
                                 4d6kh3, '1d20 adv'). NEVER roll for a player character.
  [DAMAGE Goblin A 7]            deal damage        [HEAL Thorin 10]     restore HP
  [CONDITION Elara poisoned on]  add/remove a condition (on/off)
  [SPAWN goblin 3]               add SRD monsters   [ADD Kael 30 16 pc]  add combatant name hp ac [pc]
  [NAME River Wraith A Marn]     give a creature the name you are CALLING it. Do this the
                                 moment you name one out loud, or the board and the story
                                 drift apart and turns get handed to the wrong creature.
  [COMBAT]                       start combat — rolls for YOUR monsters only, then STOP and
                                 ask each player for their initiative
  [INIT Oktah 18]                record a player's own initiative roll
  [LEVELUP]                      milestone level-up: tell the party to level on
                                 D&D Beyond (their numbers do not change yet)
  [VERIFY]                       re-read their D&D Beyond sheets to confirm they did it
  [NEXTTURN]   [ENDCOMBAT]
  [REST long]  [REST short]      a long rest restores HP and all slots; a short one offers hit dice
  [HITDIE Oktah 5]               they spent a hit die and rolled 5 (CON is added for you)
  [DEATHSAVE Oktah]              only if they roll their own — [DEATHSAVE Oktah 14]. Otherwise a
                                 dying character saves automatically when their turn comes round.
  [CONC Oktah Entangle]          they cast a concentration spell. ALWAYS log it — the CON save
                                 when they take damage then happens by itself. [CONC Oktah -] to drop.
  [INSPIRE Oktah great line]     grant inspiration    [USEINSP Oktah]  they spend it
  [LOOT 40gp, a silvered dagger] record treasure; they add it on D&D Beyond
  [DOWNTIME Oktah trained with the hunters]
                                 what they did between sessions — offer this at the
                                 start of a session, then pay it off concretely
  [CHECK owlbear 3, goblin 4]    weigh a fight against the party BEFORE running it —
                                 returns trivial/easy/medium/hard/deadly/LETHAL
  [SFX thunder]                  one-shot sound on their device at a dramatic beat — a few times
                                 a scene at most, never on routine attacks
  [SCENE a ruined chapel]        set the scene      [REMEMBER the bell is cursed]
  [MAP the drowned chapel]       draw a battle map of where the party is — use it when a
                                 fight starts or they enter somewhere tactical. It appears
                                 on their screen a few seconds later; keep narrating.
  [MOVE Wolf A F7]               move a token to a map square (column letter + row number)
  [REVEAL 2]                     uncover map area N as they explore into it. The map starts
                                 fogged except where they stand — never describe a room they
                                 have not entered.
  [LOOKUP monster owlbear]       look up monster/spell/condition/item
  [VARIANTS wolf]                other books' versions of a creature — the Level Up ones have
                                 real tactics. Use before a fight to pick a better stat block.
  [IMPORT 12345678]              import a player's D&D Beyond character (sheet must be Public)
  [ADDPLAYER Alex as Thorn]      record who plays which character (party roster)
Never invent dice results or HP — use [ROLL]/[DAMAGE]. Roll & lookup results come back to you."""

# The looping soundscape is opt-in. Left in the prompt while switched off, the DM keeps
# emitting [AMBIENCE …] every scene change and every one of them is silently discarded.
if config.AMBIENCE:
    ACTION_HELP = ACTION_HELP.replace("  [SFX thunder]", """  [AMBIENCE cave]                looping background soundscape under the whole scene. SET IT
                                 EVERY TIME the party changes surroundings, right after [SCENE].
                                 Moods: tavern/inn/market/town, forest/wind, cave/dungeon/crypt,
                                 battle/tense, sea/ship, rain/storm. It keeps playing until you
                                 change it — [AMBIENCE off] when silence is the point.
  [SFX thunder]""", 1)

_ACTION = re.compile(r"^\s*\[([A-Za-z]+)\s*(.*?)\]\s*$")
_FEEDBACK = {"roll_dice", "lookup_rule", "spawn_monster", "add_combatant",
             "start_combat", "advance_turn", "import_character", "set_initiative",
             "level_up", "verify_levels", "rest", "spend_hit_die", "death_save",
             "concentrate", "spend_inspiration", "check_encounter", "downtime",
             "monster_variants", "rename",
             # damage now carries the automatic concentration check and death-save result
             # inside its message, so the DM has to see it or a broken spell goes unnarrated
             "apply_damage", "inspire", "loot"}   # DM should see the result


def is_action_line(line: str) -> bool:
    return bool(_ACTION.match(line))


def _to_call(verb: str, rest: str):
    """Map a bracket action to (tool_name, args) for tools.dispatch."""
    v = verb.upper()
    if v == "ROLL":
        return "roll_dice", {"notation": rest}
    if v in ("DAMAGE", "HEAL"):
        m = re.match(r"(.+?)\s+(\d+)$", rest)
        if not m: return None
        return ("apply_damage" if v == "DAMAGE" else "heal"), {"name": m.group(1), "amount": int(m.group(2))}
    if v == "CONDITION":
        m = re.match(r"(.+?)\s+(\w+)(?:\s+(on|off))?$", rest, re.I)
        if not m: return None
        return "set_condition", {"name": m.group(1), "condition": m.group(2),
                                 "active": (m.group(3) or "on").lower() == "on"}
    if v == "NAME":
        m = re.match(r"(.+?)\s+(\S.*)$", rest)
        if not m: return None
        return "rename", {"old": m.group(1).strip(), "new": m.group(2).strip()}
    if v == "SPAWN":
        # an empty [SPAWN] used to raise here and take the whole turn down with it
        m = re.match(r"(.+?)(?:\s+(\d+))?$", rest)
        if not m:
            return None
        return "spawn_monster", {"name": m.group(1), "count": int(m.group(2) or 1)}
    if v == "ADD":
        m = re.match(r"(.+?)\s+(\d+)\s+(\d+)(\s+pc)?$", rest, re.I)
        if not m: return None
        return "add_combatant", {"name": m.group(1), "max_hp": int(m.group(2)),
                                 "ac": int(m.group(3)), "is_pc": bool(m.group(4))}
    if v == "REST":       return "rest", {"kind": (rest or "long").strip().lower()}
    if v == "HITDIE":
        m = re.match(r"(.+?)\s+(\d+)$", rest)
        if not m: return None
        return "spend_hit_die", {"name": m.group(1), "roll": int(m.group(2))}
    if v == "DEATHSAVE":
        m = re.match(r"(.+?)(?:\s+(\d{1,2}))?$", rest)
        if not m: return None
        return "death_save", {"name": m.group(1).strip(), "roll": int(m.group(2)) if m.group(2) else None}
    if v == "CONC":
        m = re.match(r"(\S+)\s+(.+)$", rest)
        if not m: return None
        spell = m.group(2).strip()
        return "concentrate", {"name": m.group(1), "spell": "" if spell in ("-", "off", "none") else spell}
    if v == "INSPIRE":
        m = re.match(r"(\S+)\s*(.*)$", rest)
        if not m: return None
        return "inspire", {"name": m.group(1), "reason": m.group(2).strip()}
    if v == "USEINSP":    return "spend_inspiration", {"name": rest.strip()}
    if v == "LOOT":
        gold = 0
        g = re.search(r"\b(\d+)\s*(?:gp|gold|gold pieces)\b", rest, re.I)
        if g:
            gold = int(g.group(1))
            rest = re.sub(r"\s{2,}", " ", rest[:g.start()] + rest[g.end():])
            rest = re.sub(r"^\s*(?:and|of|,)\s+|\s+(?:and|of)\s*$", " ", rest)
        return "loot", {"items": rest.strip(" ,"), "gold": gold}
    if v == "DOWNTIME":
        m = re.match(r"(\S+)\s+(.+)$", rest)
        if not m: return None
        return "downtime", {"name": m.group(1), "activity": m.group(2)}
    if v == "SFX":        return "sfx", {"name": rest.strip().lower()}
    if v == "CHECK":
        mons = []
        for part in rest.split(","):
            m = re.match(r"(.+?)(?:\s+(\d+))?$", part.strip())
            if m and m.group(1):
                mons.append({"name": m.group(1).strip(), "count": int(m.group(2) or 1)})
        return "check_encounter", {"monsters": mons}
    if v == "LEVELUP":    return "level_up", {}
    if v == "VERIFY":     return "verify_levels", {}
    if v == "COMBAT":     return "start_combat", {}
    if v == "INIT":
        m = re.match(r"(.+?)\s+(\d{1,2})$", rest)
        if not m:
            return None
        return "set_initiative", {"name": m.group(1), "value": int(m.group(2))}
    if v == "NEXTTURN":   return "advance_turn", {}
    if v == "ENDCOMBAT":  return "end_combat", {}
    if v == "SCENE":      return "set_scene", {"scene": rest}
    if v == "REMEMBER":   return "remember", {"note": rest}
    if v == "LOOKUP":
        m = re.match(r"(\w+)\s+(.+)$", rest)
        if not m: return None
        return "lookup_rule", {"category": m.group(1), "name": m.group(2)}
    if v == "IMPORT":
        return "import_character", {"ddb_id": rest}
    if v == "ADDPLAYER":
        m = re.split(r"\s+as\s+", rest, maxsplit=1)
        if len(m) != 2:
            return None
        return "add_player", {"player": m[0].strip(), "character": m[1].strip()}
    if v == "RESOURCE":                              # [RESOURCE Thorn slot1 -1] or =3
        m = re.match(r"(.+?)\s+(\w+)\s+(=?[+-]?\d+)$", rest)
        if not m:
            return None
        val = m.group(3)
        return "set_resource", {"name": m.group(1), "key": m.group(2),
                                "value": int(val.lstrip("=")), "absolute": val.startswith("=")}
    if v == "AMBIENCE":
        return "ambience", {"mood": rest}
    if v == "VARIANTS":
        return "monster_variants", {"name": rest}
    if v == "MAP":
        return "draw_map", {"location": rest}
    if v == "MOVE":
        m = re.match(r"(.+?)\s+([A-Za-z])\s*(\d{1,2})$", rest)
        if not m:
            return None
        return "move_token", {"name": m.group(1), "col": m.group(2), "row": int(m.group(3))}
    if v == "REVEAL":
        return "reveal_area", {"area": rest}
    return None


def run_action(line: str, state) -> tuple[bool, str | None]:
    """Execute one line if it's an action. Returns (was_action, feedback_for_DM_or_None)."""
    m = _ACTION.match(line)
    if not m:
        return False, None
    call = _to_call(m.group(1), m.group(2).strip())
    if not call:
        # A malformed action used to vanish silently, and if it was the whole turn the player
        # got nothing at all -- one playtest turn emitted an [INIT] it could not fill in and
        # said not a word. Hand the mistake back so it can be fixed on the spot.
        return True, (f"[THAT ACTION DID NOT PARSE: {line.strip()} — check the spelling and "
                      f"the arguments, then either run it correctly or just say what happens.]")
    tool, args = call
    result = tool_mod.dispatch(tool, args, state)
    return True, (result if tool in _FEEDBACK else None)
