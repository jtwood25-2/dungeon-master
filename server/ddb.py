"""D&D Beyond character import (unofficial).

DDB has no official API, but its character-service returns a character as JSON *if the
sheet's sharing is set to public*. This is the same endpoint Beyond20/importers use. Give
it the character id (the number in the sheet URL: dndbeyond.com/characters/<ID>).

Caveats: unofficial (could change), requires public sharing, and final Armor Class is
DDB-computed from gear (not stored) -- we approximate it and let you override in-game.
"""
import re
import urllib.request
import json

ENDPOINT = "https://character-service.dndbeyond.com/character/v5/character/{}"
_STAT_NAMES = {1: "str", 2: "dex", 3: "con", 4: "int", 5: "wis", 6: "cha"}
_ABIL_SUB = {"strength-score": 1, "dexterity-score": 2, "constitution-score": 3,
             "intelligence-score": 4, "wisdom-score": 5, "charisma-score": 6}
# always-on ability sources. Item/condition bonuses are situational (attunement, equipped,
# temporary) so we leave those out — matches the sheet for the common case.
_STAT_SOURCES = ("race", "class", "background", "feat")


def _mod(score: int) -> int:
    return (score - 10) // 2


def _ability_bonuses(d: dict) -> dict:
    """Racial/feat/ASI/background ability bumps live in `modifiers`, NOT in bonusStats."""
    out = {i: 0 for i in range(1, 7)}
    mods = d.get("modifiers") or {}
    for bucket in _STAT_SOURCES:
        for m in (mods.get(bucket) or []):
            sid = _ABIL_SUB.get(m.get("subType"))
            if sid and m.get("type") == "bonus":
                out[sid] += (m.get("value") or m.get("fixedValue") or 0)
    return out


def _skill_proficiencies(d: dict) -> dict:
    """Which skills the character is proficient (or expert) in. Without this a skill modifier
    can only be guessed — a proficient level 1 druid's Perception is +5, not the +3 the raw
    Wisdom score implies — and a guess stated confidently is worse than no answer."""
    import quick
    out: dict[str, str] = {}
    mods = d.get("modifiers") or {}
    for bucket in ("race", "class", "background", "feat", "item", "condition"):
        for m in (mods.get(bucket) or []):
            sub = (m.get("subType") or "").replace("-", " ").lower()
            if sub not in quick.SKILLS:
                continue
            kind = m.get("type")
            if kind == "expertise":
                out[sub] = "expert"
            elif kind == "proficiency" and out.get(sub) != "expert":
                out[sub] = "prof"
    return out


# armorTypeId in the DDB item definitions
_LIGHT, _MEDIUM, _HEAVY, _SHIELD = 1, 2, 3, 4


def _inventory(d: dict) -> list[dict]:
    """Everything carried, with what's currently worn/held flagged."""
    out = []
    for it in (d.get("inventory") or []):
        de = it.get("definition") or {}
        if not de.get("name"):
            continue
        out.append({"name": de["name"], "equipped": bool(it.get("equipped")),
                    "qty": it.get("quantity") or 1,
                    "ac": de.get("armorClass"), "armor_type": de.get("armorTypeId"),
                    "kind": de.get("filterType") or de.get("type") or ""})
    return out


# Unarmored Defense: class -> the second ability added to 10 + DEX, and whether a shield is
# still allowed. The monk's version says "not wielding a shield"; the barbarian's permits one.
_UNARMORED = {"barbarian": ("con", True), "monk": ("wis", False)}


def _unarmored_defense(d: dict, scores: dict) -> tuple[int, str, bool] | None:
    """The class feature, if this character actually has it.

    Without this every barbarian and monk came in badly under-armoured -- a level 4 monk with
    DEX 20 / WIS 19 was read as AC 15 instead of 19, so the DM had him hit on rolls that
    should have missed. It went unseen while the only imported character wore leather.
    """
    for cls in (d.get("classes") or []):
        name = ((cls.get("definition") or {}).get("name") or "").lower()
        if name not in _UNARMORED:
            continue
        # the feature is level 1, but only trust it if the sheet really lists it
        feats = [(f.get("definition") or {}).get("name") for f in (cls.get("classFeatures") or [])]
        if "Unarmored Defense" not in feats:
            continue
        stat, shield_ok = _UNARMORED[name]
        return _mod(scores[stat]), f"{name.title()} Unarmored Defense", shield_ok
    return None


def _armor_class(inv: list[dict], scores: dict, d: dict) -> tuple[int, str]:
    """Compute AC from what's actually equipped.

    DDB doesn't hand out a final AC number, but it does give the whole inventory with
    equipped flags and armour values, so this is derived rather than guessed:
      light armour  = base + full DEX
      medium armour = base + DEX capped at +2
      heavy armour  = base, no DEX
      no armour     = 10 + DEX, or the class's Unarmored Defense when it has one
    plus every equipped shield, plus any armor-class bonus modifiers (rings, fighting styles).
    """
    dex_mod = _mod(scores["dex"])
    worn = [i for i in inv if i["equipped"] and i.get("ac")]
    body = [i for i in worn if i["armor_type"] in (_LIGHT, _MEDIUM, _HEAVY)]
    shields = [i for i in worn if i["armor_type"] == _SHIELD]

    if body:
        armor = max(body, key=lambda i: i["ac"])     # if several, the best one is being worn
        base, t = armor["ac"], armor["armor_type"]
        if t == _LIGHT:
            ac, why = base + dex_mod, f"{armor['name']} {base} + DEX {dex_mod:+d}"
        elif t == _MEDIUM:
            capped = min(dex_mod, 2)
            ac, why = base + capped, f"{armor['name']} {base} + DEX {capped:+d} (capped)"
        else:
            ac, why = base, f"{armor['name']} {base}"
    # a monk wielding a shield loses the feature outright and drops to plain 10 + DEX (the
    # shield's own +2 still applies), so the shield check has to gate the branch, not follow it
    elif (ud := _unarmored_defense(d, scores)) and (ud[2] or not shields):
        extra, label, _ = ud
        ac = 10 + dex_mod + extra
        why = f"{label} 10 + DEX {dex_mod:+d} {extra:+d}"
    else:
        ac, why = 10 + dex_mod, f"unarmored 10 + DEX {dex_mod:+d}"

    for s in shields:
        ac += s["ac"]
        why += f" + {s['name']} {s['ac']:+d}"

    bonus = 0
    for bucket in ("race", "class", "background", "feat", "item", "condition"):
        for m in ((d.get("modifiers") or {}).get(bucket) or []):
            if m.get("subType") == "armor-class" and m.get("type") == "bonus":
                bonus += (m.get("value") or m.get("fixedValue") or 0)
    if bonus:
        ac += bonus
        why += f" + {bonus:+d}"
    return ac, why


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")


def _text(html_str: str, limit: int = 1600) -> str:
    """DDB descriptions are HTML. Keep the paragraph breaks, drop everything else.

    Tables are removed rather than flattened: stripped of markup, the Languages table becomes
    thirty orphan lines ("1d12 / Language / Origin / Common / Sigil / 1 / …") that bury the
    prose around them. A reference sheet is worse for containing them.
    """
    if not html_str:
        return ""
    t = re.sub(r"<table[\s\S]*?</table>", " ", html_str, flags=re.I)
    t = re.sub(r"</(p|div|li|tr)>", "\n", t, flags=re.I)
    t = re.sub(r"<li[^>]*>", "• ", t, flags=re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = _TAG.sub("", t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
          .replace("&gt;", ">").replace("&rsquo;", "'").replace("&mdash;", "—")
          .replace("&quot;", '"').replace("&#39;", "'"))
    t = _WS.sub(" ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t).strip()
    return t[:limit].rsplit(" ", 1)[0] + "…" if len(t) > limit else t


def _hp_bonus(d: dict, level: int) -> int:
    """Hit points granted by feats and race, which are NOT in baseHitPoints.

    Tough (and the Farmer/Hill Dwarf style origins that grant it) adds hit points through a
    `hit-points-per-level` modifier. Summing only base + bonus + CON silently understated the
    max HP of every character who had one — the sheet said 12 and we said 10.
    """
    per_level = flat = 0
    for bucket in _STAT_SOURCES:            # always-on sources only, as with ability scores
        for m in ((d.get("modifiers") or {}).get(bucket) or []):
            if m.get("type") != "bonus":
                continue
            v = m.get("value") or m.get("fixedValue") or 0
            if m.get("subType") == "hit-points-per-level":
                per_level += v
            elif m.get("subType") == "hit-points":
                flat += v
    return per_level * max(1, level) + flat


def _traits(d: dict, level: int) -> list[dict]:
    """Racial traits, class features they've actually reached, and feats — with their text.

    Level-gated deliberately: a level 1 druid's sheet should not describe Wild Shape at 2.
    """
    out = []
    race = d.get("race") or {}
    for t in (race.get("racialTraits") or []):
        de = t.get("definition") or {}
        name = de.get("name")
        # skip what's already a number on page one, and the generic rules boilerplate that
        # is identical for every character and tells you nothing about this one
        if not name or name in ("Creature Type", "Size", "Speed", "Ability Score Increase",
                                "Ability Score Increases", "Languages", "Age", "Alignment"):
            continue
        out.append({"name": name, "source": race.get("fullName") or "Race", "level": 0,
                    "text": _text(de.get("description") or de.get("snippet") or "")})
    for cls in (d.get("classes") or []):
        cname = (cls.get("definition") or {}).get("name") or "Class"
        for f in (cls.get("classFeatures") or []):
            de = f.get("definition") or {}
            req = de.get("requiredLevel") or 1
            if req > level or not de.get("name"):
                continue
            out.append({"name": de["name"], "source": cname, "level": req,
                        "text": _text(de.get("description") or "")})
    for f in (d.get("feats") or []):
        de = f.get("definition") or {}
        if de.get("name"):
            out.append({"name": de["name"], "source": "Feat", "level": 0,
                        "text": _text(de.get("description") or de.get("snippet") or "")})
    return out


_ACTIVATION = {1: "Action", 2: "—", 3: "Bonus", 4: "Reaction", 5: "Special",
               6: "Minute", 7: "Hour", 8: "Special"}
_ABIL_BY_ID = {1: "str", 2: "dex", 3: "con", 4: "int", 5: "wis", 6: "cha"}


def _cast_time(de: dict) -> str:
    a = de.get("activation") or {}
    kind = _ACTIVATION.get(a.get("activationType"), "")
    n = a.get("activationTime") or 1
    if kind in ("Minute", "Hour"):
        return f"{n} {kind.lower()}{'s' if n > 1 else ''}"
    return kind


def _duration(de: dict) -> str:
    du = de.get("duration") or {}
    if du.get("durationType") == "Instantaneous":
        return "Instant"
    n, unit = du.get("durationInterval"), (du.get("durationUnit") or "").lower()
    if n and unit:
        return f"{n} {unit}{'s' if n > 1 else ''}"
    return du.get("durationType") or ""


def _spell_range(de: dict) -> str:
    r = de.get("range") or {}
    if r.get("aoeSize"):
        return f"{r.get('rangeValue') or r.get('origin') or ''} ({r['aoeSize']}ft {r.get('aoeType') or ''})".strip()
    if r.get("rangeValue"):
        return f"{r['rangeValue']} ft"
    return r.get("origin") or ""


def _spells(d: dict) -> list[dict]:
    """Everything the character can cast, from every source (class list, race, feats, items).

    Without this the DM has no idea what's on the sheet — it fumbled a Create Bonfire ruling
    for a druid who has Create Bonfire prepared.
    """
    out, seen = [], set()

    def add(s, source):
        de = s.get("definition") or {}
        name = de.get("name")
        if not name or name.lower() in seen:
            return
        seen.add(name.lower())
        level = de.get("level", 0)
        out.append({"name": name, "level": level,
                    "school": de.get("school") or "",
                    # cantrips never need preparing; DDB reports them as prepared=False, which
                    # would read on the sheet as "can't cast this"
                    "prepared": level == 0 or bool(s.get("prepared") or s.get("alwaysPrepared")),
                    "always": bool(s.get("alwaysPrepared")), "source": source,
                    # the details you'd otherwise stop play to look up
                    "cast": _cast_time(de), "range": _spell_range(de),
                    "conc": bool(de.get("concentration")), "ritual": bool(de.get("ritual")),
                    "duration": _duration(de), "components": de.get("componentsDescription") or "",
                    "text": _text(de.get("description") or de.get("snippet") or "", 1100)})

    for bucket, lst in (d.get("spells") or {}).items():
        for s in (lst or []):
            add(s, bucket)
    for cs in (d.get("classSpells") or []):
        for s in (cs.get("spells") or []):
            add(s, "class")
    out.sort(key=lambda s: (s["level"], s["name"]))
    return out


def _slots(d: dict) -> dict:
    """Spell slots by level: {"1": {"max": 2, "used": 0}}."""
    out = {}
    for s in (d.get("spellSlots") or []) + (d.get("pactMagic") or []):
        # "available" is what's LEFT, not the maximum: a level 1 druid mid-session reported
        # available=0, used=2. The total is the sum either way round, so this reads correctly
        # whether the sheet is fresh or half spent.
        used = s.get("used") or 0
        total = (s.get("available") or 0) + used
        if total:
            out[str(s.get("level"))] = {"max": total, "used": used}
    return out


def _actions(d: dict, inv: list[dict]) -> list[dict]:
    """Things they can DO on a turn: weapon attacks plus class/race/feat actions."""
    out = []
    for it in (d.get("inventory") or []):
        de = it.get("definition") or {}
        if de.get("filterType") != "Weapon":
            continue
        dmg = (de.get("damage") or {}).get("diceString")
        out.append({"name": de.get("name"), "kind": "attack",
                    "damage": f"{dmg} {de.get('damageType') or ''}".strip() if dmg else "",
                    "equipped": bool(it.get("equipped"))})
    for bucket, lst in (d.get("actions") or {}).items():
        for a in (lst or []):
            if a.get("name"):
                out.append({"name": a["name"], "kind": bucket, "damage": "", "equipped": False})
    return out


def _char_id(s: str) -> str | None:
    m = re.search(r"(\d{5,})", str(s))
    return m.group(1) if m else None


def fetch_character(id_or_url: str) -> dict:
    cid = _char_id(id_or_url)
    if not cid:
        return {"error": "give the character id (number in the D&D Beyond sheet URL)"}
    try:
        req = urllib.request.Request(ENDPOINT.format(cid), headers={"User-Agent": "dungeon-master/0.1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return {"error": "character not found or not public — set the sheet's sharing to "
                             "'Public' in D&D Beyond, then try again"}
        return {"error": f"D&D Beyond HTTP {e.code}"}
    except Exception as e:
        return {"error": f"fetch failed: {e}"}

    d = payload.get("data") or {}
    if not d.get("name"):
        return {"error": "empty character data (is the sheet public?)"}

    # ability scores: base + manual bonus + racial/feat/ASI modifiers, or a hard override
    base = {s["id"]: s["value"] for s in d.get("stats", []) if s.get("value") is not None}
    bonus = {s["id"]: (s.get("value") or 0) for s in d.get("bonusStats", [])}
    override = {s["id"]: s["value"] for s in d.get("overrideStats", []) if s.get("value") is not None}
    modbonus = _ability_bonuses(d)
    scores = {}
    for sid, nm in _STAT_NAMES.items():
        if sid in override:
            scores[nm] = override[sid]
        else:
            scores[nm] = base.get(sid, 10) + bonus.get(sid, 0) + modbonus.get(sid, 0)

    total_level = sum(c.get("level", 0) for c in d.get("classes", []))
    classes = ", ".join(f"{(c.get('definition') or {}).get('name','?')} {c.get('level','')}"
                        for c in d.get("classes", [])) or "?"

    con_mod = _mod(scores["con"])
    if d.get("overrideHitPoints") is not None:
        max_hp = d["overrideHitPoints"]
    else:
        max_hp = ((d.get("baseHitPoints") or 0) + (d.get("bonusHitPoints") or 0)
                  + con_mod * total_level + _hp_bonus(d, total_level))
    cur_hp = max(0, max_hp - (d.get("removedHitPoints") or 0))

    inv = _inventory(d)
    ac, ac_why = _armor_class(inv, scores, d)
    cdef = ((d.get("classes") or [{}])[0].get("definition") or {})
    race = d.get("race") or {}
    profile = {
        # D&D Beyond has carried this all along and we never read it, so the DM guessed at
        # pronouns and got them wrong. Prefer explicit pronouns when the sheet sets them.
        "gender": (d.get("gender") or "").strip(),
        "pronouns": (d.get("pronouns") or "").strip(),
        "race": race.get("fullName") or race.get("baseName") or "",
        "speed": ((race.get("weightSpeeds") or {}).get("normal") or {}).get("walk") or 30,
        "hit_die": cdef.get("hitDice") or 8,
        # which ability powers their spells -> save DC and attack bonus can be computed
        "cast_stat": _ABIL_BY_ID.get(cdef.get("spellCastingAbilityId") or 0, ""),
        "subclass": ((d.get("classes") or [{}])[0].get("subclassDefinition") or {}).get("name") or "",
    }

    return {"id": str(cid), "name": d["name"], "max_hp": max_hp, "hp": cur_hp, "ac": ac, "ac_why": ac_why,
            "level": total_level, "classes": classes, "stats": scores,
            "skills": _skill_proficiencies(d), "inventory": inv,
            "spells": _spells(d), "slots": _slots(d), "actions": _actions(d, inv),
            "profile": profile, "traits": _traits(d, total_level),
            "notes": f"L{total_level} {classes} (D&D Beyond {cid}; AC {ac} = {ac_why})"}


def _apply(c: dict, combatant):
    """Copy a freshly fetched sheet onto a combatant. D&D Beyond is authoritative for
    everything here — level, HP, slots, feats, ASIs, subclass features — which is why we
    re-read it rather than trying to apply a level-up ourselves."""
    combatant.hp = c["hp"]
    combatant.max_hp = c["max_hp"]
    combatant.ac = c["ac"]
    combatant.stats = c["stats"]
    combatant.init_mod = _mod(c["stats"]["dex"])
    combatant.notes = c["notes"]
    combatant.skills = c.get("skills") or {}      # lets skill modifiers be computed, not guessed
    combatant.level = c.get("level") or 1
    combatant.inventory = c.get("inventory") or []
    combatant.spells = c.get("spells") or []
    combatant.slots = c.get("slots") or {}
    combatant.actions = c.get("actions") or []
    combatant.profile = c.get("profile") or {}
    combatant.traits = c.get("traits") or []


def resync(combatant) -> dict:
    """Re-read this character's sheet from D&D Beyond. Returns what changed."""
    if not combatant.ddb_id:
        # characters imported before ddb_id existed still carry it in their notes line
        m = re.search(r"D&D Beyond (\d+)", combatant.notes or "")
        if not m:
            return {"error": f"{combatant.name} wasn't imported from D&D Beyond."}
        combatant.ddb_id = m.group(1)
    before = {"level": combatant.level, "max_hp": combatant.max_hp, "ac": combatant.ac,
              "spells": len(combatant.spells)}
    c = fetch_character(combatant.ddb_id)
    if c.get("error"):
        return {"error": c["error"]}
    _apply(c, combatant)
    return {"before": before, "level": combatant.level, "max_hp": combatant.max_hp,
            "ac": combatant.ac, "spells": len(combatant.spells), "classes": c["classes"]}


def import_to_state(id_or_url: str, state) -> str:
    c = fetch_character(id_or_url)
    if c.get("error"):
        return f"Import failed: {c['error']}"
    combatant = state.add(c["name"], max_hp=c["max_hp"], ac=c["ac"], is_pc=True,
                          init_mod=_mod(c["stats"]["dex"]), stats=c["stats"], notes=c["notes"])
    _apply(c, combatant)
    # A sheet carries the damage from whatever game it was last used in. Joining a campaign
    # is not a wound, so an IMPORT arrives whole -- three of a new crew turned up at 36/44,
    # 19/26 and 24/31 from someone else's session. A RESYNC keeps the sheet's HP, because
    # mid-campaign that number is the live one.
    combatant.hp = combatant.max_hp
    combatant.ddb_id = c["id"]   # so we can re-read it later
    worn = ", ".join(i["name"] for i in combatant.inventory if i["equipped"]) or "nothing"
    prepared = sum(1 for s in combatant.spells if s.get("prepared"))
    return (f"Imported {c['name']} ({c['classes']}): {combatant.status}, "
            f"AC {c['ac']} ({c['ac_why']}). Equipped: {worn}. "
            f"{len(combatant.spells)} spells ({prepared} prepared).")
