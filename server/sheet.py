"""Character sheets, rendered from the authoritative game state.

Everything is COMPUTED, never guessed: ability modifiers from the scores, skill modifiers from
the sheet's own proficiencies plus the proficiency bonus, spell save DC and spell attack from
the class's casting ability, weapon attacks from proficiency and the right ability. Anything
that can't be derived honestly shows "?" rather than a confident wrong number.

Designed for the two places it's actually read: a phone mid-session, and paper. It leads with
the numbers you call out loud — AC, HP, save DC, to-hit — because looking those up is what
interrupts play.
"""
import html

import quick

_ORDER = ["str", "dex", "con", "int", "wis", "cha"]
_FINESSE = {"sickle", "dagger", "rapier", "shortsword", "scimitar", "whip", "dart"}
_RANGED = {"shortbow", "longbow", "sling", "dart", "crossbow", "light crossbow",
           "hand crossbow", "heavy crossbow"}

CSS = """
*{box-sizing:border-box}
:root{
  --ink:#1a1613; --dim:#6d6257; --faint:#a99e90; --rule:#ded3c0;
  --paper:#fdfaf3; --panel:#f6f0e2; --gold:#9a7b3f; --blood:#8c3b34; --leaf:#4a6b48;
}
body{font-family:"Iowan Old Style",Georgia,"Times New Roman",serif;color:var(--ink);
  background:var(--paper);max-width:860px;margin:0 auto;padding:26px 20px 40px;line-height:1.4;
  -webkit-text-size-adjust:100%}

/* ---- name plate ---- */
.plate{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap}
h1{margin:0;font-size:34px;letter-spacing:.4px;line-height:1.05}
.who{color:var(--dim);font-style:italic;font-size:15px;margin-top:3px}
.badge{font-size:11px;text-transform:uppercase;letter-spacing:1.4px;color:var(--gold);
  border:1px solid var(--gold);border-radius:999px;padding:3px 10px;white-space:nowrap}
.rule{border:0;border-top:2px solid var(--gold);margin:12px 0 18px}

/* ---- the numbers you say out loud ---- */
.vitals{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:9px;margin-bottom:16px}
.v{border:1.5px solid var(--ink);border-radius:10px;padding:9px 6px;text-align:center;background:#fff}
.v .k{font-size:9.5px;text-transform:uppercase;letter-spacing:1.1px;color:var(--dim)}
.v .n{font-size:27px;font-weight:700;line-height:1.15;font-variant-numeric:tabular-nums}
.v .s{font-size:11px;color:var(--dim)}
.v.hero{background:var(--ink);color:#fff;border-color:var(--ink)}
.v.hero .k,.v.hero .s{color:#c9bfae}

/* ---- abilities ---- */
.abils{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:0 0 20px}
.ab{border:1.5px solid var(--ink);border-radius:10px;text-align:center;padding:7px 2px;background:#fff}
.ab .n{font-size:9.5px;text-transform:uppercase;letter-spacing:1.1px;color:var(--dim)}
.ab .m{font-size:23px;font-weight:700;font-variant-numeric:tabular-nums}
.ab .s{font-size:11px;color:var(--faint)}

h2{font-size:11.5px;text-transform:uppercase;letter-spacing:1.6px;color:var(--gold);
  border-bottom:1px solid var(--rule);padding-bottom:4px;margin:22px 0 9px;font-weight:700}
h3{font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim);
  margin:0 0 4px;font-weight:700}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:640px){.cols{grid-template-columns:1fr;gap:6px}
  .abils{grid-template-columns:repeat(3,1fr)}h1{font-size:28px}}

/* ---- lists ---- */
ul{list-style:none;margin:0;padding:0}
li.row{display:flex;justify-content:space-between;gap:10px;align-items:baseline;
  padding:3px 0;border-bottom:1px dotted var(--rule);font-size:13.5px}
li.row:last-child{border-bottom:0}
.row .m{font-variant-numeric:tabular-nums;font-weight:700;white-space:nowrap}
.row .sub{color:var(--faint);font-size:11px;font-weight:400;margin-left:5px}
.dot{display:inline-block;width:8px;height:8px;border:1.4px solid var(--dim);border-radius:50%;
  margin-right:7px;vertical-align:middle}
.dot.on{background:var(--ink);border-color:var(--ink)}
.dot.exp{background:var(--gold);border-color:var(--gold)}
.skills{columns:2;column-gap:24px}
.skills li.row{break-inside:avoid}

/* ---- attacks ---- */
.atk li.row .m{color:var(--blood)}
.tag{font-size:9.5px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px;margin-left:6px}

/* ---- spells ---- */
.slots{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 12px}
.slot{border:1px solid var(--rule);border-radius:8px;padding:4px 9px;font-size:12px;background:#fff}
.slot b{color:var(--gold);margin-right:4px}
.tick{display:inline-block;width:13px;height:13px;border:1.2px solid var(--dim);
  border-radius:3px;margin-right:2px;vertical-align:-2px}
.tick.used{background:var(--dim)}
.spellcols{columns:2;column-gap:24px}
.grp{break-inside:avoid;margin-bottom:12px}
.conc{color:var(--blood);font-weight:700}

.note{font-size:11.5px;color:var(--faint);margin-top:7px}
.blank{border-bottom:1px solid var(--rule);height:20px}
.gear li{font-size:13.5px;padding:2.5px 0}
.gear .eq{font-weight:700}
.gear .qty{color:var(--faint);font-size:11px}

/* ---- page two: the full text ---- */
.ref{margin-top:34px;padding-top:8px}
.entry{break-inside:avoid;margin:0 0 11px;padding-left:10px;border-left:2px solid var(--rule)}
.entry .eh{font-size:13.5px}
.entry .eh b{font-weight:700}
.entry .meta{color:var(--faint);font-size:11px;margin-left:8px;font-style:italic}
.entry p{margin:2px 0 0;font-size:12.5px;color:#332c25;line-height:1.42}
h3.lvl{font-size:10.5px;text-transform:uppercase;letter-spacing:1.3px;color:var(--gold);
  margin:14px 0 6px;font-weight:700}
@media(min-width:800px){.ref .entry{break-inside:avoid}
  .ref > h2 + *{columns:2;column-gap:26px}}

@media print{
  body{max-width:none;padding:0;background:#fff;font-size:11pt}
  .page{page-break-after:always}
  .page:last-child{page-break-after:auto}
  .ref{margin-top:0;page-break-before:always}
  .v,.ab{border-width:1px}
  h2{margin-top:14px}
  .entry p{font-size:9.5pt}
}
"""


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _sign(n) -> str:
    return f"{n:+d}" if isinstance(n, int) else "?"


def _prof(c) -> int:
    return quick.prof_bonus(getattr(c, "level", 1) or 1)


def _mods(c) -> dict:
    return {k: quick._mod(v) for k, v in (c.stats or {}).items()}


def _class_line(c) -> str:
    note = (c.notes or "").split("(")[0].strip()
    prof = getattr(c, "profile", None) or {}
    bits = [b for b in (prof.get("race"), note or (f"Level {c.level}" if c.level else "")) if b]
    return " · ".join(bits)


def _attacks(c) -> list[dict]:
    """To-hit and damage, worked out. Looking these up is what stops play."""
    m, pb = _mods(c), _prof(c)
    out = []
    for a in (getattr(c, "actions", None) or []):
        if a.get("kind") != "attack":
            continue
        name = a.get("name", "")
        low = name.lower()
        # finesse and ranged weapons use DEX; everything else STR
        use_dex = low in _FINESSE or low in _RANGED or "bow" in low or "crossbow" in low
        ability = "dex" if (use_dex and m.get("dex", 0) >= m.get("str", 0)) else "str"
        if low in _RANGED or "bow" in low:
            ability = "dex"
        mod = m.get(ability, 0)
        dmg = a.get("damage") or ""
        parts = dmg.split(" ", 1)
        dice, dtype = (parts[0], parts[1] if len(parts) > 1 else "") if dmg else ("", "")
        out.append({
            "name": name, "hit": pb + mod, "ability": ability.upper(),
            "damage": f"{dice}{mod:+d}".replace("+0", "") if dice else "",
            "type": dtype, "equipped": a.get("equipped"),
        })
    return out


def _spellcasting(c) -> dict | None:
    prof = getattr(c, "profile", None) or {}
    stat = prof.get("cast_stat")
    if not stat or stat not in (c.stats or {}):
        return None
    mod, pb = quick._mod(c.stats[stat]), _prof(c)
    return {"stat": stat.upper(), "dc": 8 + pb + mod, "attack": pb + mod, "mod": mod}


def render(c, campaign: str = "") -> str:
    m, pb = _mods(c), _prof(c)
    sc = _spellcasting(c)
    prof = getattr(c, "profile", None) or {}
    passive, known = quick.skill_mod(c, "perception")

    vitals = [("Armor Class", c.ac, "", True), ("Hit Points", f"{c.hp}", f"/ {c.max_hp}", True)]
    vitals.append(("Initiative", _sign(c.init_mod), "", False))
    vitals.append(("Speed", prof.get("speed", 30), "ft", False))
    vitals.append(("Prof. Bonus", _sign(pb), "", False))
    vitals.append(("Passive Perc.", 10 + passive if known else "—", "", False))
    if sc:
        vitals.append(("Spell Save DC", sc["dc"], sc["stat"], True))
        vitals.append(("Spell Attack", _sign(sc["attack"]), "", False))
    if c.level and prof.get("hit_die"):
        vitals.append(("Hit Dice", f"{c.level}d{prof['hit_die']}", "", False))

    vit_html = "".join(
        f'<div class="v{" hero" if hero else ""}"><div class="k">{_e(k)}</div>'
        f'<div class="n">{_e(v)}</div><div class="s">{_e(s)}</div></div>'
        for k, v, s, hero in vitals)

    abils = "".join(
        f'<div class="ab"><div class="n">{k.upper()}</div>'
        f'<div class="m">{_sign(m[k])}</div><div class="s">{c.stats[k]}</div></div>'
        for k in _ORDER if k in (c.stats or {}))

    saves = "".join(
        f'<li class="row"><span><i class="dot"></i>{_e(quick.ABILITIES[k])}</span>'
        f'<span class="m">{_sign(m[k])}</span></li>' for k in _ORDER if k in m)

    skills = []
    for name in sorted(quick.SKILLS):
        mod, ok = quick.skill_mod(c, name)
        rank = (getattr(c, "skills", None) or {}).get(name)
        dot = "dot on" if rank == "prof" else ("dot exp" if rank == "expert" else "dot")
        skills.append(
            f'<li class="row"><span><i class="{dot}"></i>{_e(name.title())}'
            f'<span class="sub">{quick.SKILLS[name][:3].upper()}</span></span>'
            f'<span class="m">{_sign(mod) if ok else "?"}</span></li>')

    return f"""<div class="page">
  <div class="plate">
    <div>
      <h1>{_e(c.name)}</h1>
      <div class="who">{_e(_class_line(c))}</div>
    </div>
    {f'<span class="badge">{_e(campaign)}</span>' if campaign else ""}
  </div>
  <hr class="rule">

  <div class="vitals">{vit_html}</div>
  <div class="abils">{abils}</div>

  <div class="cols">
    <div>
      <h2>Skills</h2>
      <ul class="skills">{"".join(skills)}</ul>
      <div class="note">● proficient &nbsp; ◆ expertise &nbsp; ? not on the imported sheet</div>
    </div>
    <div>
      <h2>Saving Throws</h2>
      <ul>{saves}</ul>
      <h2>Condition</h2>
      <div style="font-size:13.5px">{_e(", ".join(c.conditions)) if c.conditions
        else '<span class="note">None</span>'}</div>
      <h2>Death Saves</h2>
      <div class="note">Successes {"".join('<i class="tick"></i>' for _ in range(3))}
        &nbsp;&nbsp; Failures {"".join('<i class="tick"></i>' for _ in range(3))}</div>
    </div>
  </div>

  {_attacks_block(c)}
  {_spells_block(c, sc)}

  <div class="cols">
    <div>
      <h2>Equipment</h2>
      {_gear_block(c)}
    </div>
    <div>
      {_traits_summary(c)}
      <h2>Notes</h2>
      {"".join('<div class="blank"></div>' for _ in range(5))}
    </div>
  </div>
</div>
{_reference(c)}"""


def _traits_summary(c) -> str:
    """Page one names them; page two explains them."""
    traits = getattr(c, "traits", None) or []
    if not traits:
        return ""
    rows = "".join(
        f'<li class="row"><span>{_e(t["name"])}</span>'
        f'<span class="sub">{_e(t["source"])}</span></li>' for t in traits)
    return f'<h2>Features &amp; Traits</h2><ul>{rows}</ul>'


def _entry(name: str, meta: str, text: str) -> str:
    if not text:
        return ""
    paras = "".join(f"<p>{_e(p)}</p>" for p in text.split("\n") if p.strip())
    return (f'<div class="entry"><div class="eh"><b>{_e(name)}</b>'
            f'{f"<span class=meta>{_e(meta)}</span>" if meta else ""}</div>{paras}</div>')


def _reference(c) -> str:
    """Page two: the full text, so nobody reaches for a phone mid-turn."""
    spells = [s for s in (getattr(c, "spells", None) or []) if s.get("text")]
    traits = [t for t in (getattr(c, "traits", None) or []) if t.get("text")]
    if not spells and not traits:
        return ""

    trait_html = "".join(_entry(t["name"], t["source"], t["text"]) for t in traits)

    by_level: dict = {}
    for s in spells:
        by_level.setdefault(s.get("level", 0), []).append(s)
    spell_html = ""
    for lvl in sorted(by_level):
        label = "Cantrips" if lvl == 0 else f"Level {lvl}"
        spell_html += f'<h3 class="lvl">{label}</h3>'
        for s in sorted(by_level[lvl], key=lambda x: x["name"]):
            meta = " · ".join(x for x in (
                s.get("cast"), s.get("range"), s.get("duration"),
                "Concentration" if s.get("conc") else "",
                "Ritual" if s.get("ritual") else "",
                s.get("components")) if x)
            spell_html += _entry(s["name"], meta, s["text"])

    return f"""<div class="page ref">
  <div class="plate">
    <div><h1>{_e(c.name)}</h1><div class="who">Reference — traits, features and spells</div></div>
  </div>
  <hr class="rule">
  {f'<h2>Features &amp; Traits</h2>{trait_html}' if trait_html else ''}
  {f'<h2>Spells</h2>{spell_html}' if spell_html else ''}
</div>"""


def _attacks_block(c) -> str:
    atks = _attacks(c)
    others = [a for a in (getattr(c, "actions", None) or []) if a.get("kind") != "attack"]
    if not atks and not others:
        return ""
    rows = "".join(
        f'<li class="row"><span>{"★ " if a["equipped"] else ""}{_e(a["name"])}'
        f'<span class="sub">{a["ability"]}</span></span>'
        f'<span class="m">{_sign(a["hit"])} <span class="sub">to hit</span> '
        f'{_e(a["damage"])} <span class="sub">{_e(a["type"].lower())}</span></span></li>'
        for a in atks)
    other = "".join(f'<li class="row"><span>{_e(a["name"])}</span></li>' for a in others)
    return (f'<h2>Attacks &amp; Actions</h2><ul class="atk">{rows}</ul>'
            + (f'<ul style="margin-top:6px">{other}</ul>' if other else ""))


def _spells_block(c, sc) -> str:
    spells = getattr(c, "spells", None) or []
    if not spells:
        return ""
    slots = getattr(c, "slots", None) or {}
    slot_html = ""
    if slots:
        cells = []
        for lvl in sorted(slots, key=lambda k: int(k)):
            s = slots[lvl]
            used, mx = s.get("used", 0), s.get("max", 0)
            ticks = "".join(f'<i class="tick{" used" if i < used else ""}"></i>'
                            for i in range(min(mx, 9)))
            cells.append(f'<span class="slot"><b>L{_e(lvl)}</b>{ticks}</span>')
        slot_html = f'<div class="slots">{"".join(cells)}</div>'

    by_level: dict = {}
    for s in spells:
        by_level.setdefault(s.get("level", 0), []).append(s)
    groups = []
    for lvl in sorted(by_level):
        label = "Cantrips" if lvl == 0 else f"Level {lvl}"
        items = []
        for s in sorted(by_level[lvl], key=lambda x: x["name"]):
            tags = []
            if s.get("conc"):
                tags.append('<span class="conc">C</span>')
            if s.get("ritual"):
                tags.append("R")
            meta = " · ".join(x for x in (s.get("cast"), s.get("range")) if x)
            items.append(
                f'<li class="row"><span><i class="dot{" on" if s.get("prepared") else ""}"></i>'
                f'{_e(s["name"])} {" ".join(tags)}'
                f'<span class="sub">{_e(meta)}</span></span></li>')
        groups.append(f'<div class="grp"><h3>{label}</h3><ul>{"".join(items)}</ul></div>')

    head = "Spells"
    if sc:
        head += (f' <span class="sub" style="text-transform:none;letter-spacing:0">'
                 f'save DC {sc["dc"]} · attack {_sign(sc["attack"])} · {sc["stat"]}</span>')
    return (f'<h2>{head}</h2>{slot_html}<div class="spellcols">{"".join(groups)}</div>'
            f'<div class="note">● prepared &nbsp; <b class="conc">C</b> concentration '
            f'&nbsp; R ritual</div>')


def _gear_block(c) -> str:
    inv = getattr(c, "inventory", None) or []
    if not inv:
        return "".join('<div class="blank"></div>' for _ in range(8))
    worn = [i for i in inv if i.get("equipped")]
    rest = [i for i in inv if not i.get("equipped")]
    def li(i, eq=False):
        qty = f' <span class="qty">×{i["qty"]}</span>' if i.get("qty", 1) > 1 else ""
        tag = ' <span class="tag">worn</span>' if eq else ""
        return f'<li class="{"eq" if eq else ""}">{_e(i["name"])}{qty}{tag}</li>'
    return ('<ul class="gear">' + "".join(li(i, True) for i in worn)
            + "".join(li(i) for i in rest) + "</ul>")


def page(chars, campaign: str = "") -> str:
    """A standalone document — one sheet per character, each on its own printed page."""
    title = f"{campaign} — character sheets" if campaign else "Character sheets"
    body = "".join(render(c, campaign) for c in chars) or "<p>No characters yet.</p>"
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(title)}</title><style>{CSS}</style></head><body>{body}</body></html>')
