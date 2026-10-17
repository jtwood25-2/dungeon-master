"""Deterministic answers — computed in Python, with no model in the loop at all.

Anything that is a field read or plain arithmetic is answered here. Two reasons, both of
which matter more than they sound:

  CORRECTNESS. A language model asked "what's my AC" can state a number that isn't on your
  sheet, and it will say it with complete confidence. `combatant.ac` cannot.
  SPEED. This costs ~0.1 microseconds against ~500ms for the local model.

The hard rule is STRICT MATCH OR FALL THROUGH: every pattern is a fullmatch against the
whole utterance, so "what's my AC" is answered here while "what's my AC against the ogre's
bite" is not matched at all and goes to the model. With no model in the loop there is
nothing to catch a misparse, so anything less than certain must be handed on.
"""
import re

ABILITIES = {"str": "Strength", "dex": "Dexterity", "con": "Constitution",
             "int": "Intelligence", "wis": "Wisdom", "cha": "Charisma"}
_ABIL_WORDS = {
    "strength": "str", "str": "str", "dexterity": "dex", "dex": "dex",
    "constitution": "con", "con": "con", "intelligence": "int", "int": "int",
    "wisdom": "wis", "wis": "wis", "charisma": "cha", "cha": "cha",
}
# 5e skill -> governing ability
SKILLS = {
    "athletics": "str",
    "acrobatics": "dex", "sleight of hand": "dex", "stealth": "dex",
    "arcana": "int", "history": "int", "investigation": "int", "nature": "int",
    "religion": "int",
    "animal handling": "wis", "insight": "wis", "medicine": "wis", "perception": "wis",
    "survival": "wis",
    "deception": "cha", "intimidation": "cha", "performance": "cha", "persuasion": "cha",
}

_FILLER = re.compile(r"^(?:hey |ok |okay |so |um |uh |and )+")
_PUNCT = re.compile(r"[^\w' ]+")


def _norm(text: str) -> str:
    t = _PUNCT.sub(" ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return _FILLER.sub("", t).strip()


def _mod(score: int) -> int:
    return (int(score) - 10) // 2


def _spell(n: int) -> str:
    """Speak a signed modifier the way a person says it."""
    if n > 0:
        return f"Plus {n}"
    if n < 0:
        return f"Minus {abs(n)}"
    return "Zero"


def _list(items: list[str]) -> str:
    """Speak a list the way a person reads it aloud."""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pc(state, speaker=None):
    """Whose sheet is 'my'? Only answer when that is unambiguous."""
    pcs = [c for c in state.combatants.values() if c.is_pc]
    if speaker:
        for c in pcs:
            if c.name.lower() == str(speaker).lower():
                return c
    return pcs[0] if len(pcs) == 1 else None


def prof_bonus(level: int) -> int:
    return 2 + max(0, (int(level) - 1)) // 4


def skill_mod(c, skill: str):
    """Returns (modifier, True) only when proficiency is actually known. Without the sheet's
    proficiency list a skill modifier is a GUESS — a proficient level 1 druid's Perception is
    +5, not the +3 the raw ability score suggests — so we return unknown and let the model
    handle it rather than stating a confident wrong number."""
    abil = SKILLS.get(skill)
    if abil is None or abil not in c.stats:
        return None, False
    skills = getattr(c, "skills", None)
    if not skills:
        return None, False                       # proficiencies unknown -> do not guess
    base = _mod(c.stats[abil])
    rank = skills.get(skill)
    pb = prof_bonus(getattr(c, "level", 1) or 1)
    if rank == "expert":
        return base + 2 * pb, True
    if rank == "prof":
        return base + pb, True
    return base, True                            # known to be non-proficient


# ---------------------------------------------------------------- patterns
_WHATS = r"(?:what(?:'s|s| is| are)?\s+)?"
_MY = r"(?:my|our)\s+"
_TAIL = r"(?:\s+(?:right now|currently|again|now|at the moment))?"


def _p(body: str) -> re.Pattern:
    return re.compile(_WHATS + body + _TAIL)


AC_RE = _p(_MY + r"(?:ac|armou?r class)")
HP_RE = _p(r"(?:how many |how much )?" + _MY + r"(?:hp|hit points|health)"
           r"(?:\s+do i have)?(?:\s+left)?")
HP_RE2 = re.compile(r"how (?:many|much) (?:hp|hit points|health) do i have(?:\s+left)?" + _TAIL)
MAXHP_RE = _p(_MY + r"(?:max|maximum) (?:hp|hit points)")
ABIL_RE = _p(_MY + r"(" + "|".join(_ABIL_WORDS) + r")(?:\s+(?:modifier|mod|score|bonus))?")
SKILL_RE = _p(_MY + r"(" + "|".join(SKILLS) + r")(?:\s+(?:modifier|mod|bonus|check))?")
INIT_RE = _p(_MY + r"initiative(?:\s+(?:modifier|mod|bonus))?")
COND_RE = re.compile(_WHATS + r"(?:conditions?(?:\s+am i (?:under|suffering))?"
                     r"|am i affected by anything|any conditions on me)" + _TAIL)
# "Can you hear us?" is a question about the EQUIPMENT, and it was answered in character:
#   "Can you hear us?"  ->  Voice: "Hear you fine. Still waiting on the rest of your names."
#   "Can you hear us?"  ->  Voice: "Still hearing you. Names?"
#   "Can you hear us?"  ->  Voice: "Yes. Still here."
# Three times in three minutes, while the table sat there still unable to tell whether the
# microphone was working. A question about the app gets an answer from the app, in plain
# words, out of the fiction, and without waiting on a model.
MIC_CHECK_RE = re.compile(
    r"^\s*(?:hey\s+|ok(?:ay)?\s+|so\s+)?(?:dm[,\s]+|dungeon master[,\s]+)?"
    r"(?:can|do|are)\s+you\s+(?:still\s+)?"
    # only US, never a thing in the world -- "can you hear the wraiths?" is a real question
    # about the scene and belongs to the DM, not to the microphone
    r"(?:hear|hearing|reading|getting)\s+(?:me|us|us all|anything|that|this)\b"
    r"|^\s*(?:hey\s+|ok(?:ay)?\s+)?(?:dm[,\s]+|dungeon master[,\s]+)?"
    r"(?:can|do|are)\s+you\s+(?:still\s+)?(?:there|working|on|listening|awake|alive)\b",
    re.I)
MIC_CHECK_RE2 = re.compile(
    r"^\s*(?:is\s+(?:this|it|the mic|the microphone)\s+(?:on|working|live|recording)"
    r"|(?:are\s+you|you)\s+there"
    r"|mic\s*check|testing(?:\s*,?\s*testing)?"
    r"|(?:is|are)\s+(?:this|you|it)\s+working)"
    r"[\w\s']{0,20}\??\s*$", re.I)

TURN_RE = re.compile(r"(?:whose|who's|whos) turn(?:\s+is it)?" + _TAIL)
NEXT_TURN_RE = re.compile(r"(?:next turn|end (?:my |the )?turn|i end my turn|pass(?:\s+my turn)?)")
EQUIPPED_RE = re.compile(_WHATS + r"(?:am i (?:wearing|carrying|holding)"
                         r"|(?:do i have )?equipped"
                         r"|(?:what(?:'s|s| is) )?(?:my )?(?:gear|inventory|equipment)"
                         r"|what am i (?:wearing|carrying|holding))" + _TAIL)
HAVE_RE = re.compile(r"do i have (?:an? |any |my )?(.{2,30}?)" + _TAIL)
OPTIONS_RE = re.compile(r"(?:what (?:can|could) i do|what are my options|what can i do "
                        r"(?:on|for) my turn|options)" + _TAIL)
SLOTS_RE = _p(r"(?:how many |what )?" + r"(?:spell )?slots?(?: do i have)?(?: (?:left|remaining))?")
INSP_RE = _p(r"(?:do i have )?inspiration")
HITDICE_RE = _p(r"(?:how many )?hit dice(?: do i have)?(?: (?:left|remaining))?")
# "What dice do I use for this?" cost sixteen seconds through Sonnet, and the answer was
# sitting on the character sheet: the imported spell text for Create Bonfire literally reads
# "take 1d8 fire damage". Nothing about it needs a language model.
DICE_RE = re.compile(
    r"(?:what|which|how much)\s+"
    r"(?:dice|die|damage|dmg|healing)\s*"
    r"(?:do i|should i|does it|does that|do we)?\s*"
    r"(?:roll|use|deal|do|does|for)?\s*"
    r"(?:for\s+)?(?:the\s+|my\s+|a\s+|an\s+)?(.{0,40}?)"
    r"(?:\s+(?:do|does|deal|deals))?\s*\??$")
# "how much damage does create bonfire do" -- the name sits in the middle instead
DICE_RE2 = re.compile(
    r"how much (?:damage|healing) does (?:the |my |a |an )?(.{2,40}?) (?:do|deal|cause)\s*\??$")
# the words people use to mean "the thing I just did"
_THIS = re.compile(r"^(?:this|that|it|the spell|my spell|the attack|the cantrip)?$")
# damage/healing dice inside a spell or feature description
_DICE_IN_TEXT = re.compile(
    r"(\d+d\d+)(?:\s*\+\s*\d+)?\s+(?:points? of\s+)?(\w+)?\s*(?:damage|hit points)"
    # healing puts the dice AFTER the thing restored: "regains Hit Points equal to 2d4"
    r"|(?:hit points|hp)\s+equal to\s+(\d+d\d+)()", re.I)
_SCALES = re.compile(
    r"(?:increases|increase)s?\s+by\s+(\d+d\d+)[^.]*?(?:level|levels)\s*\(?(\d+)", re.I)
DAMAGE_RE = re.compile(r"i (?:take|took|takes) (\d{1,3}) (?:points? of )?damage")
HEAL_RE = re.compile(r"i (?:heal|healed|regain|regained|gain) (\d{1,3})(?: hit points?| hp)?")


def _cantrip_dice(base: str, level: int) -> str:
    """Cantrips scale with CHARACTER level. Reporting 1d8 to a level-11 druid is just wrong."""
    n, _, faces = base.partition("d")
    steps = sum(1 for t in (5, 11, 17) if level >= t)
    return f"{int(n) + steps}d{faces}" if steps else base


def find_dice(c, want: str) -> tuple[str, str] | None:
    """(what it is called, what to roll) for a spell or weapon on their own sheet.

    Their sheet is the authority, not the SRD -- Create Bonfire is not in the open content at
    all (rules.spell returns None) but the D&D Beyond import carries the full text.
    """
    want = (want or "").strip().lower()
    if not want:
        return None
    pools = [(s.get("name", ""), s.get("text", ""), s.get("level", 1)) for s in (c.spells or [])]
    pools += [(a.get("name", ""), a.get("damage", ""), None) for a in (c.actions or [])
              if a.get("damage")]
    hit = None
    for name, text, lvl in pools:
        nm = name.lower()
        if nm == want or want in nm or (len(want) > 4 and nm in want):
            hit = (name, text, lvl)
            break
    if hit is None:
        return None
    name, text, lvl = hit
    if not (m := _DICE_IN_TEXT.search(text)):
        # a weapon action stores the dice bare: "1d4 Slashing"
        if m2 := re.match(r"\s*(\d+d\d+)\s*(\w+)?", text or ""):
            kind = f" {m2.group(2).lower()}" if m2.group(2) else ""
            return name, f"{m2.group(1)}{kind}"
        return name, ""
    dice = m.group(1) or m.group(3)                # damage form, or the healing form
    kind = (m.group(2) or "").lower() if m.group(1) else "healing"
    if lvl == 0:                                   # a cantrip: scale it to who is casting
        dice = _cantrip_dice(dice, c.level or 1)
    kind = "" if kind in ("", "of", "a", "the") else f" {kind}"
    return name, f"{dice}{kind}"


def remember_subject(text: str, state) -> None:
    """Note which spell or weapon was just named, so 'what dice for THIS' can be answered.

    Called on both sides of the exchange. Without it the commonest phrasing at the table --
    'what do I roll for that?' -- has no referent and has to go to a model.
    """
    low = (text or "").lower()
    for c in state.combatants.values():
        if not c.is_pc:
            continue
        for name in [s.get("name", "") for s in (c.spells or [])] + \
                    [a.get("name", "") for a in (c.actions or []) if a.get("damage")]:
            if len(name) > 3 and name.lower() in low:
                state.last_subject = name
                return


def answer(text: str, state, speaker=None) -> str | None:
    """Return a spoken answer, or None to let a model handle it."""
    t = _norm(text)
    if not t:
        return None
    c = _pc(state, speaker)

    # answered before anything else, and never in character
    if MIC_CHECK_RE.match(t) or MIC_CHECK_RE2.match(t):
        who = (speaker or "").strip()
        return (f"Yes, I can hear you{', ' + who if who else ''}." if who
                else "Yes, I can hear you.")

    if TURN_RE.fullmatch(t):
        if not state.in_combat:
            return None                          # "whose turn" out of combat is a real question
        if not state.order:
            return None                          # order not settled yet -> let the DM explain
        # f"{...}" around a None yields the word "None", and it gets SPOKEN. That happened:
        # "Whose turn is it now?" was answered with the single word "None" out loud.
        who = state.current_turn()
        return f"{who}." if who else None

    if NEXT_TURN_RE.fullmatch(t):
        if not state.in_combat or not state.order:
            return None
        state.next_turn()
        return f"{state.current_turn()}."

    if c is None:
        return None                              # ambiguous whose sheet -> let the model ask

    if AC_RE.fullmatch(t):
        return f"{c.ac}."
    if HP_RE.fullmatch(t) or HP_RE2.fullmatch(t):
        return f"{c.hp} of {c.max_hp}."
    if MAXHP_RE.fullmatch(t):
        return f"{c.max_hp}."
    if INIT_RE.fullmatch(t):
        return f"{_spell(c.init_mod)}."
    if COND_RE.fullmatch(t):
        return ", ".join(c.conditions).capitalize() + "." if c.conditions else "Nothing."

    if (m := ABIL_RE.fullmatch(t)):
        key = _ABIL_WORDS[m.group(1)]
        if key not in c.stats:
            return None
        score = c.stats[key]
        if re.search(r"\bscore\b", t):
            return f"{score}."
        return f"{_spell(_mod(score))}."

    if (m := SKILL_RE.fullmatch(t)):
        mod, known = skill_mod(c, m.group(1))
        return f"{_spell(mod)}." if known else None

    # these must be tested BEFORE HAVE_RE, which would otherwise read "do I have inspiration"
    # as a question about an item called "inspiration"
    if OPTIONS_RE.fullmatch(t):
        return _options(c)

    if SLOTS_RE.fullmatch(t):
        if not c.slots:
            return None                          # not a caster, or we never imported slots
        got = [f"{n} at level {lvl}" for lvl, s in sorted(c.slots.items())
               if (n := s["max"] - s.get("used", 0)) > 0]
        return _list(got) + "." if got else "No slots left."

    if INSP_RE.fullmatch(t):
        return "Yes, you have inspiration." if c.inspiration else "No inspiration right now."

    if HITDICE_RE.fullmatch(t):
        total = c.level or 1
        left = total - c.hit_dice_used
        if not (die := (c.profile or {}).get("hit_die")):
            return None
        return f"{left} of {total}, d{die} each." if left else "None left until a long rest."

    if EQUIPPED_RE.fullmatch(t):
        if not c.inventory:
            return None
        worn = [i["name"] for i in c.inventory if i.get("equipped")]
        carried = [i["name"] for i in c.inventory if not i.get("equipped")]
        parts = []
        if worn:
            parts.append("You've got " + _list(worn) + " equipped")
        if carried:
            parts.append(("and you're carrying " if parts else "You're carrying ") + _list(carried))
        return ". ".join(parts).replace(". and", ", and") + "." if parts else "Nothing."

    if (m := HAVE_RE.fullmatch(t)):
        if not c.inventory:
            return None
        want = m.group(1).strip().rstrip("s")
        for i in c.inventory:
            if want and want in i["name"].lower():
                n = i.get("qty", 1)
                return (f"Yes, {n} of them." if n > 1 else "Yes.") + \
                       (" Equipped." if i.get("equipped") else "")
        return "No."

    if (m := DICE_RE.fullmatch(t)) or (m := DICE_RE2.fullmatch(t)):
        want = (m.group(1) or "").strip()
        if _THIS.fullmatch(want):                # "what dice for this" -> the last one named
            want = getattr(state, "last_subject", "") or ""
        if found := find_dice(c, want):
            name, dice = found
            return f"{dice} for {name}." if dice else None
        return None                              # not on their sheet -> let the model answer

    if (m := DAMAGE_RE.fullmatch(t)):
        state.damage(c.name, int(m.group(1)))
        if c.hp <= 0:
            return "You're down."
        return f"{c.hp} of {c.max_hp}."
    if (m := HEAL_RE.fullmatch(t)):
        state.heal(c.name, int(m.group(1)))
        return f"{c.hp} of {c.max_hp}."

    return None


def _options(c) -> str | None:
    """What this character can actually do on their turn, read off their own sheet.

    Newer players freeze here and the DM ends up coaching them, which kills the pace of a
    fight. Everything below is a field read — no model, no invented abilities.
    """
    attacks = [a for a in (c.actions or []) if a.get("equipped")] or (c.actions or [])[:3]
    ready = [s for s in (c.spells or []) if s.get("prepared")]
    clauses = []
    if attacks:
        names = [a["name"] + (f" for {a['damage']}" if a.get("damage") else "")
                 for a in attacks[:3]]
        clauses.append("attack with " + _list(names))
    if ready:
        cantrips = [s["name"] for s in ready if not s.get("level")][:3]
        levelled = [s["name"] for s in ready if s.get("level")][:4]
        free = sum(s["max"] - s.get("used", 0) for s in (c.slots or {}).values())
        if cantrips:
            clauses.append("cast " + _list(cantrips) + " for free")
        if levelled and free:
            clauses.append(f"or spend one of your {free} slots on " + _list(levelled))
        elif levelled:
            clauses.append("though you're out of spell slots")
    if not clauses:
        return None                              # nothing imported -> don't pretend to know
    speed = (c.profile or {}).get("speed") or 30
    out = ["You can " + ", ".join(clauses) + "."]
    out.append(f"You've got {speed} feet of movement, and you can always dash, disengage, "
               "dodge, hide, help, or shove.")
    if c.inspiration:
        out.append("You still have inspiration.")
    return " ".join(out)
