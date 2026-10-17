"""Run the monsters' side of a fight in Python, because none of it is a judgement call.

Session 9 is the argument for this file. Three "River Wraiths" were on the board with
`actions: []` and `profile: {}` -- no attack bonus, no damage dice, no CR -- so when the DM
narrated "Wraith C grabs for you" there was nothing underneath it to resolve. The player had
to ask "was there an attack roll?", the answer was "rolling now, 16 beats your AC" invented on
the spot, no damage was ever applied, and the turn never advanced past the first monster, so
the player never got a turn at all.

Every one of those is arithmetic a model should not be doing mid-sentence:

  who does it attack      the lowest-HP conscious PC it can reach
  does it hit             d20 + attack bonus vs AC
  how much                the damage dice, doubled on a natural 20
  what happens next       apply it, then advance the turn

So the engine resolves a monster's whole turn and hands the DM the FACTS. The model's only
remaining job is to say those facts in one sentence, which is the thing it is actually good at.
A PC's turn takes no model call at all -- the engine says whose turn it is and stops.

Monsters invented on the fly (the wraiths were added by name, HP and AC alone, not from the
SRD) get a stat block derived from their CR rather than an empty one, so an improvised
creature still fights by real numbers.
"""
import logging
import re

import dice
import rules

logger = logging.getLogger("combat")

# DMG "Monster Statistics by Challenge Rating": CR -> (attack bonus, damage per round, AC,
# top of the hit-point band). The HP column is what lets an improvised creature be placed on
# the table by the only number the DM actually gave it.
_BY_CR = {
    0: (3, 1, 13, 6), 0.125: (3, 3, 13, 35), 0.25: (3, 5, 13, 49), 0.5: (3, 8, 13, 70),
    1: (3, 14, 13, 85), 2: (3, 20, 13, 100), 3: (4, 26, 13, 115), 4: (5, 32, 14, 130),
    5: (6, 38, 15, 145), 6: (6, 44, 15, 160), 7: (6, 50, 15, 175), 8: (7, 56, 16, 190),
    9: (7, 62, 16, 205), 10: (7, 68, 17, 220), 11: (8, 74, 17, 235), 12: (8, 80, 17, 250),
    13: (8, 86, 18, 265), 14: (8, 92, 18, 280), 15: (8, 98, 18, 295), 16: (9, 104, 18, 310),
    17: (10, 110, 19, 325), 18: (10, 116, 19, 340), 19: (10, 122, 19, 355),
    20: (10, 140, 19, 400),
}

_DMG = re.compile(r"(\d+)d(\d+)\s*(?:([+-])\s*(\d+))?")
_HIT = re.compile(r"([+-]\d+)\s+to hit", re.I)
_MULTI = re.compile(r"\b(two|three|four|2|3|4)\b[^.]{0,40}?attacks?", re.I)
_WORD_N = {"two": 2, "three": 3, "four": 4, "2": 2, "3": 3, "4": 4}
# Actions that are not a weapon swing: recharge breath, spellcasting, utility. Kept out of the
# default attack pick so a dragon does not open every round with its once-per-fight breath.
_NOT_ATTACK = re.compile(r"recharge|spellcasting|change shape|teleport|summon|invisib", re.I)


def _cr_row(cr) -> tuple:
    from encounter import _cr
    c = _cr(cr)
    key = min(_BY_CR, key=lambda k: abs(k - c))
    return _BY_CR[key]


def stat_block(name: str, cr=None, hp: int = 0) -> dict:
    """Attacks for a creature that has none: look it up, else derive from CR.

    An improvised monster is a legitimate DM move -- "River Wraith" is not in any book -- but
    it must still have numbers. Without a CR the hit points imply one, which is rough but far
    better than an empty list.
    """
    if m := rules.monster(name):
        if m.get("actions"):
            return {"actions": m["actions"], "stats": m.get("stats") or {},
                    "cr": m.get("cr"), "source": "srd"}
    if cr is None:
        # Place it by hit points, against the HP band -- the only stat an improvised monster
        # reliably has. Reading it off the damage column instead inferred CR 3 from 22 HP and
        # handed a level-1 character a 6d6 attack that killed it outright from full health.
        cr = min(_BY_CR, key=lambda k: abs(_BY_CR[k][3] - max(1, hp)))
    atk, dpr, _ac, _hp = _cr_row(cr)
    # Damage as a die plus a modifier rather than a fistful of dice: a single 6d6 swing has a
    # range of 6-36, so the same creature either tickles or one-shots. Half from the die, half
    # flat, keeps the average honest and the spread survivable.
    avg_die = max(1, round(dpr / 2))
    n, faces = max(1, round(avg_die / 3.5)), 6
    bonus = max(0, dpr - int(n * 3.5))
    return {"actions": [{"name": "Attack", "attack_bonus": atk,
                         "damage_dice": f"{n}d{faces}",
                         "damage_bonus": bonus}],
            "stats": {}, "cr": cr, "source": "derived"}


def _attacks(c) -> list:
    """The creature's usable attacks, with the numbers parsed out of the SRD prose."""
    out = []
    for a in (c.actions or []):
        name = a.get("name") or "Attack"
        desc = a.get("desc") or ""
        if name.lower() == "multiattack" or _NOT_ATTACK.search(name + " " + desc):
            continue
        bonus = a.get("attack_bonus")
        if bonus is None and (h := _HIT.search(desc)):
            bonus = int(h.group(1))
        dd, db = a.get("damage_dice"), a.get("damage_bonus") or 0
        if not dd and (d := _DMG.search(desc)):
            dd = f"{d.group(1)}d{d.group(2)}"
            db = int(d.group(4) or 0) * (-1 if d.group(3) == "-" else 1)
        if bonus is None or not dd:
            continue
        out.append({"name": name, "bonus": int(bonus), "dice": dd, "dmg_bonus": int(db)})
    return out


def _multiattack_count(c) -> int:
    for a in (c.actions or []):
        if (a.get("name") or "").lower() == "multiattack":
            if m := _MULTI.search(a.get("desc") or ""):
                return _WORD_N.get(m.group(1).lower(), 1)
    return 1


def choose_target(state, attacker):
    """Who it goes for. Monsters press an advantage; they do not finish off the unconscious.

    Attacking a downed PC is a hard failure at a real table -- it turns a bad round into a
    dead character with no decision in between -- so a conscious target always wins.
    """
    up = [c for c in state.combatants.values()
          if c.is_pc and c.hp > 0 and "dead" not in c.conditions]
    if not up:
        return None
    # Not strictly the weakest: going for the lowest HP every single round means a hurt
    # character is finished off by every monster in turn with no decision in between. The
    # weakest is the usual pick, but a badly hurt one is only worth ganging up on some of
    # the time -- which is roughly how a thinking creature behaves anyway.
    up.sort(key=lambda c: (c.hp, c.name))
    if len(up) > 1 and state.rng.random() < 0.35:
        return up[1]
    return up[0]


def _one_attack(state, attacker, target, atk, rng) -> dict:
    r = dice.roll(f"1d20+{atk['bonus']}", rng)
    nat = r.dice[0] if r.dice else r.total - atk["bonus"]
    crit, fumble = nat == 20, nat == 1
    hit = crit or (not fumble and r.total >= target.ac)
    out = {"attack": atk["name"], "target": target.name, "d20": nat, "total": r.total,
           "ac": target.ac, "hit": hit, "crit": crit, "damage": 0}
    if not hit:
        return out
    d = dice.roll(f"{atk['dice']}+{atk['dmg_bonus']}" if atk["dmg_bonus"] >= 0
                  else f"{atk['dice']}{atk['dmg_bonus']}", rng)
    dmg = d.total
    if crit:                                   # crits double the DICE, not the modifier
        dmg += dice.roll(atk["dice"], rng).total
    out["damage"] = max(1, dmg)
    out["result"] = state.damage(target.name, out["damage"], crit=crit)
    return out


def take_turn(state, name: str) -> dict:
    """Resolve one monster's entire turn. Returns the facts; narration is the model's job."""
    c = state.get(name)
    if c is None or c.is_pc:
        return {"error": f"{name} is not a monster on the board"}
    if c.hp <= 0:
        return {"name": name, "skipped": "down"}
    for cond in ("stunned", "paralyzed", "unconscious", "incapacitated", "petrified"):
        if cond in c.conditions:
            return {"name": name, "skipped": cond}

    # give it a stat block the first time it acts, and keep it for the rest of the fight
    if not _attacks(c):
        blk = stat_block(c.name, (c.profile or {}).get("cr"), c.max_hp)
        c.actions = blk["actions"]
        c.profile = dict(c.profile or {}, cr=blk["cr"], statted=blk["source"])
        if blk["stats"] and not c.stats:
            c.stats = blk["stats"]

    atks = _attacks(c)
    target = choose_target(state, c)
    if target is None:
        return {"name": name, "skipped": "no conscious target"}

    rng = state.rng
    swings = []
    n = _multiattack_count(c)
    for i in range(n):
        atk = atks[i % len(atks)] if n > 1 else atks[0]
        swings.append(_one_attack(state, c, target, atk, rng))
        if target.hp <= 0:                     # stop hitting someone who has gone down
            break
    return {"name": c.name, "target": target.name, "swings": swings,
            "target_status": target.status if target.is_pc else target.vibe}


def describe(turn: dict) -> str:
    """The facts, as one plain line for the DM to voice. Deliberately unadorned."""
    if turn.get("error"):
        return turn["error"]
    if s := turn.get("skipped"):
        return f"{turn['name']} does nothing ({s})."
    bits = []
    for sw in turn["swings"]:
        if not sw["hit"]:
            bits.append(f"{sw['attack']} misses ({sw['total']} vs AC {sw['ac']})")
        else:
            crit = " CRIT" if sw["crit"] else ""
            bits.append(f"{sw['attack']} hits{crit} for {sw['damage']} "
                        f"({sw['total']} vs AC {sw['ac']})")
    return (f"{turn['name']} attacks {turn['target']}: " + "; ".join(bits)
            + f". {turn['target']} is now {turn['target_status']}.")


# A monster doing violence to a player, as the DM actually writes it. Only ever consulted
# when there are living hostiles on the board, which is what keeps it from firing on prose.
_VIOLENCE = re.compile(
    # present AND past. A live run narrated "Goblin A swung its sword at you and missed" and
    # "its sword comes around at you" -- the pattern had "swing" and "swings" and matched
    # neither, so the fight ran with nothing on the board resolving it.
    r"\b(?:lunge|claw|swing|swung|strike|struck|slash|bite|bit|grab|grabb|grasp|seize|"
    r"lash|attack|swipe|stab|stabb|hurl|charge|rake|maul|drag|slam|smash|thrust|jab|"
    r"cleave|hack|pounce|leap)(?:s|es|ed|ing)?\b"
    r"|\b(?:comes?|came|swings?|swung)\s+(?:around\s+|back\s+)?(?:at|for|toward)\b"
    r"|\b(?:hits?|misses|missed|catches|caught|connects?)\s+you\b"
    r"|\byou\s+(?:take|took)\s+\d+\b", re.I)
_AT_YOU = re.compile(r"\b(?:you|your|yours)\b", re.I)
# the DM saying out loud that a fight is running
_FIGHT_TALK = re.compile(
    r"\b(?:top of the round|roll initiative|initiative|combat (?:continues|begins|starts)|"
    r"(?:'s|s) turn\b|next up|is up next|round \d+)\b", re.I)


def sniff(line: str, state) -> str | None:
    """Start combat when the fiction has plainly become a fight and nobody said so.

    Three sessions in a row the DM ran an entire fight in prose and called nothing: no
    [COMBAT], no [ROLL], no [DAMAGE]. The last one narrated "4 damage, you're at 8 out of 12"
    while the character sat at 12/12 in the state and in_combat was False, so every engine
    hook downstream -- which all key off in_combat -- stayed switched off.

    Asking the model to declare combat has now failed at every level it was asked. So the
    server decides: living hostiles on the board plus the DM narrating violence at a player IS
    combat, whatever the model did or did not emit.
    """
    if getattr(state, "in_combat", False):
        return None
    if not (_FIGHT_TALK.search(line) or
            (_VIOLENCE.search(line) and _AT_YOU.search(line))):
        return None
    hostiles = [c for c in state.combatants.values() if not c.is_pc and c.hp > 0]
    if not hostiles:
        # Narrating a fight against creatures that are not on the board. Nothing downstream can
        # work -- no stat blocks to roll, no initiative to take, no turns to advance -- and a
        # playtest spent a whole session in a room with two goblins that existed only in the
        # prose. Say so at the moment it happens, while it is still fixable.
        logger.info(f"fight narrated with nothing on the board: {line[:60]!r}")
        return ("[NOBODY IS ON THE BOARD] You are narrating a fight against creatures that do "
                "not exist in the game yet, so nothing you describe can be rolled or tracked. "
                "Run [SPAWN <creature> <count>] for them NOW -- use a real SRD creature name so "
                "they come with a stat block -- and then [COMBAT]. Do this before you say "
                "anything else.")
    out = state.start_combat()
    logger.info(f"combat auto-started: the DM narrated a fight without declaring one "
                f"({line[:60]!r})")
    return ("[COMBAT AUTO-STARTED — you were narrating a fight that had not begun, so nothing "
            "you described was being rolled or applied. It is running now.] " + out)


def _plain(s: str) -> str:
    """Strip the [BRACKETS] the tools wrap their results in.

    Those brackets exist to catch the model's eye inside a prompt, and they are exactly what
    leaked into the voice when a line was read out as written.
    """
    return re.sub(r"\s{2,}", " ", re.sub(r"[\[\]]", "", s or "")).strip()


def _block(facts: list, note: str) -> str:
    """One shape for everything the engine hands the narrator.

    FACTS and INSTRUCTIONS used to sit on lines that looked alike, and the table heard the
    difference: "Order's set. River Wraith B [dice] 1d20+4 = 24, you're up." Facts are marked,
    one per line, already in plain speakable English; the instruction is separate and never
    looks like something to read out.
    """
    out = ["[COMBAT — these have ALREADY been rolled and applied. They happened.]"]
    out += [f"> {f}" for f in facts if f]
    out.append(f"[HOW TO SAY IT] {note}")
    return "\n".join(out) + "\n"


# A player SAYING their turn is over. This is the only thing that ends it.
#
# Guessing was tried three times and failed three different ways: it took questions as turns,
# then corrections as turns, then movement as turns, and each time the fix was another regex
# and the next session found another phrasing. A turn boundary is not something to infer from
# prose -- it is something the table says out loud, and every real table says it out loud.
# So the list below is the whole rule, it is deterministic, and no model is consulted.
_END_TURN = re.compile(
    r"\b(?:"
    r"(?:that(?:'s| is)|thats)\s+(?:it|all|my turn|the end of my turn|everything|me done)"
    r"|end\s+of\s+(?:my|the)\s+turn"
    r"|end(?:ing)?\s+(?:my|the)\s+turn"
    r"|(?:my|the)\s+turn(?:'s| is)?\s+(?:over|done|finished|up)"
    r"|(?:i(?:'m| am)|im)\s+(?:done|finished|good|out)"
    r"|(?:i|we)\s+(?:end|finish)\s+(?:my|our|the)\s+turn"
    r"|that(?:'s| is)\s+all\s+(?:i can do|for me|i(?:'ve| have) got)"
    r"|end\s+turn|turn\s+over|done\s+with\s+my\s+turn"
    r"|(?:i\s+)?pass\s+my\s+turn"
    r"|go\s+ahead(?:\s+and\s+continue)?"
    r"|next\s+(?:turn|up|person|player)"
    r"|over\s+to\s+you"
    r")\b", re.I)

# Things a player says that are NOT them taking their turn. Asking what happened, querying a
# rule, correcting the DM, thinking out loud -- none of it is an action, and none of it should
# cost them the round.
_NOT_AN_ACTION = re.compile(
    r"^\s*(?:what|when|where|who|why|how|which|whose|can|could|may|might|do|does|did|is|are|"
    r"am|was|were|will|would|should|have|has|had|if)\b"
    r"|\?\s*$"
    r"|\b(?:that(?:'s| is) (?:not right|wrong)|i (?:haven't|hadn't|didn't|never)|"
    r"hold on|wait|hang on|actually|no,|nope|sorry|i mean|i meant|"
    r"should not be|shouldn't be|isn't my|is not my)\b"
    # "My turn's not over yet. I want to walk in our tent." -- the correction and the action
    # arrive in one breath, and taking it as a turn is what the player is objecting to. If
    # they are telling us the turn is not finished, it is not finished.
    r"|\bmy turn(?:'s| is)? (?:not|isn't|ain't) (?:over|done|finished)"
    r"|\b(?:i(?:'m| am) not (?:done|finished)|not done yet|not finished yet|"
    r"still my turn|it(?:'s| is) still my turn)\b", re.I)


# Moving is FREE. A character gets their movement and their action on the same turn, and
# announcing a move must never end the turn -- "I want to then take my full movement" cost the
# player their round and handed all three wraiths a free swing.
_JUST_MOVING = re.compile(
    r"^\s*(?:i(?:'ll| will| want to| wanna| am going to| would like to)?\s*)?"
    r"(?:then\s+|also\s+|first\s+)?"
    r"(?:take\s+(?:my\s+)?(?:full\s+)?(?:movement|move|speed)"
    r"|take\s+a\s+step|step\s+back|step\s+away"
    r"|move|step|back up|back off|walk|run|retreat|withdraw|reposition|disengage"
    r"|get\s+(?:back|clear|away)|pull\s+back|fall\s+back)"
    r"[\w\s,'-]{0,40}\s*[.!]?\s*$", re.I)


# Bare "pass" / "I pass", alone. It used to sit in _END_TURN as a loose alternative and fired
# inside ordinary sentences: at a real table "Is this going to pass an area or is this
# completely new?" ended somebody's turn. A word that common has to be the WHOLE utterance
# before it means anything.
_BARE_PASS = re.compile(r"^\s*(?:i\s+)?pass(?:es|ing)?\s*[.!]?\s*$", re.I)


def ends_turn(text: str) -> bool:
    """Did the player SAY their turn is over? Nothing else ends it.

    The old version of this tried to work the answer out from the shape of the sentence, and
    got it wrong three sessions running -- questions, then corrections, then movement were all
    read as "I am done", and each cost the player their round and gave every monster a free
    swing. Inference is the wrong tool. The table says when a turn is over; we listen for it.
    """
    t = (text or "").strip()
    return bool(_END_TURN.search(t) or _BARE_PASS.match(t))


def is_turn_action(text: str) -> bool:
    """Kept for the callers that still ask "was that table talk?"; the turn boundary is
    ends_turn() and only ends_turn()."""
    t = (text or "").strip()
    if not t:
        return False
    if _JUST_MOVING.match(t):
        return False
    return not bool(_NOT_AN_ACTION.search(t))



def spoken_facts(brief: str) -> str:
    """The engine's own account, for when the narrator says nothing at all.

    A brief is FACTS plus INSTRUCTIONS, deliberately separated so the two never get confused --
    and this only ever takes the facts. Reading "[HOW TO SAY IT] Ask, in a few words, whether
    that is their turn" out loud would be worse than the silence it is replacing.

    Rolls and armour classes are stripped on the way out. They are true, they are in the brief
    for the model's benefit, and the table is never meant to hear them.
    """
    out = []
    for line in (brief or "").splitlines():
        line = line.strip()
        if not line.startswith(">"):
            continue
        t = line.lstrip("> ").strip()
        t = re.sub(r"\s*\((?:[^()]*\bAC\b[^()]*|\s*\d+\s*vs[^()]*)\)", "", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        if t:
            out.append(t if t.endswith((".", "!", "?")) else t + ".")
    return " ".join(out)


def waiting_on_pc(state) -> str:
    """The player whose turn the engine is holding for, or ''.

    The fight stops here and only here: a player's turn ends when they SAY so, which is right
    and was still the single reason a real session never left round 1. 433 utterances, 7 of them
    ending a turn, and ZERO DM lines asking anybody to -- because the ask was written into the
    prompt and the model went silent on 30% of turns, while asked_end was set to True regardless.
    It asked once, into nothing, and never again.
    """
    if not getattr(state, "in_combat", False) or not state.order:
        return ""
    who = state.current_turn()
    c = state.get(who) if who else None
    if c is None or not c.is_pc:
        return ""
    if c.hp <= 0 and "stable" not in c.conditions and "dead" not in c.conditions:
        return ""                     # dying: death saves run themselves, nothing to wait for
    return who if getattr(state, "pc_acted", False) else ""


def nudge_line(state) -> str:
    """What to SAY, verbatim, when the engine is waiting on somebody. Never a prompt.

    Spoken by the server rather than asked of the model, for the same reason the engine's own
    combat facts are: if it has to be heard, something that cannot go silent has to say it.
    """
    who = waiting_on_pc(state)
    if not who:
        return ""
    first = who.split()[0]
    n = int(getattr(state, "end_nudges", 0) or 0)
    if n == 0:
        return f"{first}, that's your action. Anything else, or is that your turn?"
    if n == 1:
        return f"{first}, still your turn. Say 'that's my turn' when you're done."
    return f"Still waiting on {first}. Say 'end my turn' to move on."


def took_over(state, speaker: str) -> bool:
    """Did a DIFFERENT player just act while somebody else's turn was open?

    This is the one place any judgement enters, and it is a fact rather than a guess about
    intent: the table has visibly moved on. Without it the fight waits forever on a player who
    has put their phone down, which is exactly what happened.
    """
    who = waiting_on_pc(state)
    if not who or not speaker:
        return False
    return speaker.strip().lower() != who.strip().lower()

def brief(state, player_text: str = "", limit: int = 12) -> str:
    """Resolve everything the monsters do RIGHT NOW and describe it for the prompt.

    This is called before the model sees the turn, and that placement is the whole point.
    Leaving combat to a [NEXTTURN] the model had to remember to emit meant it simply did not:
    session 10 has the DM writing "That brings us to River Wraith B." as prose and then
    stopping, so the player sat for fifty-eight seconds and finally had to say "River Wraith B
    do." to make the monster take its own turn. The engine was in place and never once ran --
    the wraiths still had actions:[] and turn_idx was still 0 at the end of the fight.

    Nothing is asked of the model here except to say what already happened.
    """
    if not getattr(state, "in_combat", False) or not state.order:
        return ""
    who = state.current_turn()
    c = state.get(who) if who else None
    if c is None:
        return ""
    if c.is_pc:
        dying = (c.hp <= 0 and "dead" not in c.conditions and "stable" not in c.conditions)
        # Their turn ends when they have ACTED, not when the model remembers to say so. The
        # model narrated "that brings us to River Wraith A" and never emitted [NEXTTURN], so
        # the order never moved; tools.py sets acted=True the moment a roll or damage lands on
        # a player's own go, which is the same thing a real table watches for.
        #
        # Unless they are unconscious, in which case waiting for them to act waits forever:
        # their death saves simply never happened and the fight sat still.
        if not dying:
            # THE ONLY THING THAT ENDS A TURN IS THE PLAYER SAYING SO. Not a question, not a
            # correction, not moving, not casting -- they may want to move after they cast, and
            # three sessions were spoiled by assuming otherwise.
            if not ends_turn(player_text):
                # Held. The ASK is no longer put to the model -- app.py speaks nudge_line()
                # itself, because a prompt the model can swallow is a prompt the table never
                # hears. Count the nudges so the wording escalates instead of repeating.
                if getattr(state, "pc_acted", False):
                    state.end_nudges = int(getattr(state, "end_nudges", 0) or 0) + 1
                    state.asked_end = True
                return ""
            state.asked_end = False
            state.end_nudges = 0
        state.pc_acted = False
        state.end_nudges = 0
        if dying:
            res = state.death_save(c.name)
            facts = [_plain(res)]
            if "dead" in c.conditions:
                facts.append(f"{c.name} is dead.")
                note = "Play the death, then stop."
            elif "stable" in c.conditions:
                facts.append(f"{c.name} is stable at 0 hit points, unconscious.")
                note = "Say they have stopped slipping away."
            else:
                d = c.death_saves
                facts.append(f"{d['success']} successes and {d['fail']} failures so far.")
                note = ("Say what the save was and that they are still going. ONE beat -- do "
                        "not roll another and do not skip ahead to whether they live.")
            state.next_turn()
            return _block(facts, note)

        r = run_until_player(state, limit=limit, advance_first=True)
    else:
        r = run_until_player(state, limit=limit)

    lines = [e["line"] for e in r["events"] if e.get("line")]
    lines += [_plain(e["result"]) for e in r["events"] if e.get("result") and not e.get("line")]
    if not lines:
        return ""
    note = ("Narrate ALL of it in one go, briefly. Do not roll any of it again, do not stop "
            "between monsters, and never announce a turn you are not also resolving. Say no "
            "numbers at all -- no hit points, no damage totals, no die rolls, no armour class.")
    if nxt := r.get("player_turn"):
        note += (f" Finish with exactly this handover and nothing fancier: \"{nxt}, you're up.\" "
                 "There is no initiative token and nothing is handed over as an object.")
    elif down := r.get("down"):
        note += (f" {down} is down -- narrate the hit that dropped them and stop."
                 + ("" if r.get("party_standing") else
                    " Nobody is left standing, so play the consequence rather than rolling "
                    "death saves until they die."))
    return _block(lines, note)


_NUM = re.compile(r"\b(\d{1,3})\b")
_NAMED = re.compile(r"\b(?:at|on|the)\s+([A-Z][\w']*(?:\s+[A-Z])?)", re.I)


def _say_it(pc, what: str) -> str:
    """The constraints a small model needs spelled out, every single time.

    Left to infer them it narrated the character in the third person ("Tester's quarterstaff
    strikes"), invented a hit point total the facts never contained ("Goblin A is down to 1
    hit point" against a stated 'badly hurt'), and handed the turn to a character from a
    different campaign entirely.
    """
    return (f"{what} ONE short sentence. Speak to {pc.name} as 'you' -- never '{pc.name}' in "
            f"the third person, and never name anyone who is not in the facts above. Say NO "
            f"numbers at all: no hit points, no damage total, no die roll, no armour class. "
            f"If you hand the turn over, hand it to {pc.name} and nobody else.")


def _with(pc) -> str:
    """What they are actually holding. Left out, the model gives them a sword they do not own."""
    for a in (pc.actions or []):
        if a.get("kind") == "attack" and a.get("equipped"):
            return f" with their {a['name'].lower()}"
    for i in (pc.inventory or []):
        if i.get("equipped") and (i.get("kind") or "").lower() not in ("armor", "armour"):
            return f" with their {i['name'].lower()}"
    for a in (pc.actions or []):
        if a.get("kind") == "attack":
            return f" with their {a['name'].lower()}"
    return ""


def _target_for(state, text: str):
    """Who the player is swinging at. Named if they named one, otherwise the obvious one."""
    live = [c for c in state.combatants.values() if not c.is_pc and c.hp > 0]
    if not live:
        return None
    low = (text or "").lower()
    for c in live:                                 # "the wraith", "goblin b", "at Marn"
        if c.name.lower() in low:
            return c
        short = c.name.split()[-1].lower()
        if len(short) > 1 and re.search(rf"\b{re.escape(short)}\b", low):
            return c
    if re.search(r"\b(?:same|it|him|her|again)\b", low):
        if (last := getattr(state, "last_target", "")) and (c := state.get(last)):
            if c.hp > 0:
                return c
    return live[0] if len(live) == 1 else min(live, key=lambda c: (c.hp, c.name))


def resolve_player(state, text: str) -> str:
    """Resolve the player's own mechanical turn, so it becomes settled facts like any other.

    Without this the local model was asked to DECIDE -- did a 17 hit, what did it do -- and it
    is measurably bad at deciding: it narrated the character in the third person, invented "5
    damage" with nothing rolled, read a monster's hit points out ("Goblin A: 7/7 -> 0/7") and
    gave them a weapon they do not own. Resolved here, all of that is arithmetic again.
    """
    if not getattr(state, "in_combat", False):
        return ""
    who = state.current_turn()
    pc = state.get(who) if who else None
    if pc is None or not pc.is_pc:
        return ""
    n = _NUM.search(text or "")
    pending = getattr(state, "pending_attack", None) or {}

    # they owe us damage on a hit we already confirmed
    if pending.get("hit") and n:
        tgt = state.get(pending.get("target") or "")
        amount = int(n.group(1))
        if tgt is not None:
            state.damage(tgt.name, amount)
            state.pending_attack = None
            state.pc_acted = True
            facts = [f"You hit {tgt.name}{_with(pc)}.",
                     f"{tgt.name} is {tgt.vibe}." if tgt.hp > 0 else f"{tgt.name} is down."]
            return _block(facts, _say_it(pc, "Say the blow landing and how they look now."))

    tgt = _target_for(state, text)
    if tgt is None:
        return ""
    if n:                                          # a d20 total against its armour class
        roll = int(n.group(1))
        hit = roll >= tgt.ac
        state.last_target = tgt.name
        if hit:
            state.pending_attack = {"target": tgt.name, "hit": True}
            return _block([f"Your attack{_with(pc)} beats {tgt.name}'s defences."],
                          _say_it(pc, "Say it connects, then ask for the damage roll in a few "
                                      "words. Nothing else, and no handover yet."))
        state.pending_attack = None
        state.pc_acted = True
        return _block([f"Your attack{_with(pc)} misses {tgt.name}."],
                      _say_it(pc, "Say the miss as something the world did -- what the blow "
                                  "hit instead."))
    # an attack declared with no number yet
    state.last_target = tgt.name
    return _block([f"You are attacking {tgt.name}. No roll given yet."],
                  _say_it(pc, "Ask for their attack roll and nothing else -- 'Roll to hit.' is "
                              "enough. Do not roll it for them and do not narrate the swing."))


def run_until_player(state, limit: int = 12, advance_first: bool = False) -> dict:
    """Advance combat, resolving every monster turn, and stop when it is a player's go.

    The whole point: a monster's turn needs no model call, and a player's turn needs no model
    call either -- it needs the table to be told whose go it is. Only the narration of what the
    monsters did goes to the model, once, with the numbers already settled.

    `advance_first` is the difference between "this turn is finished, move on" (the DM ending a
    player's go) and "resolve from where we are" (the first turn of a new fight). Without it,
    ending a player's turn just re-announced the same player's turn forever.
    """
    events, guard = [], 0
    if advance_first and state.in_combat and state.order:
        state.next_turn()
    # Nobody on the party's side can act, so there is no fight left to run -- only monsters
    # taking turns against a corpse. This spun from round 2 to round 10 in one call, and the
    # DM had to tell the player "combat's just been cycling empty turns since".
    if not any(c.is_pc and c.hp > 0 for c in state.combatants.values()):
        if not any(c.is_pc and "dead" not in c.conditions for c in state.combatants.values()):
            state.end_combat()
            return {"events": [], "over": "no player character is still alive"}
    start_round = state.round
    while guard < limit:
        guard += 1
        # One call must never carry the fight past the end of a round. Resolving several
        # rounds in one go is how a player watched three of their own death saves arrive in a
        # single breath, already decided.
        if state.round > start_round:
            return {"events": events, "round_ended": state.round}
        who = state.current_turn()
        if not who:
            return {"events": events, "waiting": "no initiative order yet"}
        c = state.get(who)
        if c is None:
            state.next_turn(); continue
        if c.is_pc:
            if c.hp <= 0 and "dead" not in c.conditions and "stable" not in c.conditions:
                # ONE save, then hand back. Looping here rolled all three inside a single call
                # and killed a character outright with nobody at the table getting a word in --
                # the log read "Oktah of the Wood DIES" with no beat before it, because
                # _death_fail only logs the third. Each save is its own moment: the party can
                # act between them, and the player has to hear it happen.
                res = state.death_save(who)
                state.next_turn()
                return {"events": events + [{"death_save": who, "result": res}],
                        "dying": who, "round": state.round,
                        "dead": "dead" in c.conditions}
            return {"events": events, "player_turn": who, "round": state.round,
                    "status": c.status}
        turn = take_turn(state, who)
        turn["line"] = describe(turn)
        events.append(turn)
        state.next_turn()
        # A character going down is not a round to be played through -- it is the moment the
        # whole table reacts to. Grinding on to the next monster rolled straight past it and
        # only stopped after three failed death saves had killed them.
        if turn.get("swings") and (t := state.get(turn.get("target") or "")) and t.hp <= 0:
            return {"events": events, "down": t.name, "round": state.round,
                    "party_standing": any(p.is_pc and p.hp > 0
                                          for p in state.combatants.values())}
    return {"events": events, "stopped": "turn limit"}
