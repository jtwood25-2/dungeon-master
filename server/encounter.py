"""Encounter difficulty, computed rather than guessed.

"Is three owlbears too much for a level 3 party?" is arithmetic, and a language model
answering it from feel is how a table gets accidentally wiped in session two. The DMG gives
an XP budget per character per level and a multiplier for fighting several things at once;
both are tables, so both live here.

Deliberately advisory: it reports the difficulty and never blocks the DM from running the
fight. A deadly encounter on purpose is good DMing; a deadly encounter by accident is not.
"""
import json
from pathlib import Path

import rules

_PREFS = Path(__file__).parent.parent / "reference" / "difficulty.json"


def prefs() -> dict:
    """Table difficulty targets, read fresh so retuning is editing data, not code."""
    try:
        return json.loads(_PREFS.read_text(encoding="utf-8"))
    except Exception:
        return {}

# XP by challenge rating (DMG). Fractional CRs are the common low-level cases.
XP_BY_CR = {
    0: 10, 0.125: 25, 0.25: 50, 0.5: 100, 1: 200, 2: 450, 3: 700, 4: 1100, 5: 1800,
    6: 2300, 7: 2900, 8: 3900, 9: 5000, 10: 5900, 11: 7200, 12: 8400, 13: 10000,
    14: 11500, 15: 13000, 16: 15000, 17: 18000, 18: 20000, 19: 22000, 20: 25000,
    21: 33000, 22: 41000, 23: 50000, 24: 62000, 25: 75000, 26: 90000, 27: 105000,
    28: 120000, 29: 135000, 30: 155000,
}

# per-character XP thresholds: level -> (easy, medium, hard, deadly)
THRESHOLDS = {
    1: (25, 50, 75, 100), 2: (50, 100, 150, 200), 3: (75, 150, 225, 400),
    4: (125, 250, 375, 500), 5: (250, 500, 750, 1100), 6: (300, 600, 900, 1400),
    7: (350, 750, 1100, 1700), 8: (450, 900, 1400, 2100), 9: (550, 1100, 1600, 2400),
    10: (600, 1200, 1900, 2800), 11: (800, 1600, 2400, 3600), 12: (1000, 2000, 3000, 4500),
    13: (1100, 2200, 3400, 5100), 14: (1250, 2500, 3800, 5700), 15: (1400, 2800, 4300, 6400),
    16: (1600, 3200, 4800, 7200), 17: (2000, 3900, 5900, 8800), 18: (2100, 4200, 6300, 9500),
    19: (2400, 4900, 7300, 10900), 20: (2800, 5700, 8500, 12700),
}


def _cr(value) -> float:
    """CRs arrive as 1, '1', '1/4', or '0.25' depending on the source book."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if "/" in s:
        a, _, b = s.partition("/")
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def xp_for(cr) -> int:
    c = _cr(cr)
    if c in XP_BY_CR:
        return XP_BY_CR[c]
    # unlisted CR: take the nearest listed one rather than returning zero, which would
    # silently make a hard fight look trivial
    return XP_BY_CR[min(XP_BY_CR, key=lambda k: abs(k - c))]


def multiplier(count: int, party_size: int) -> float:
    """Several weaker enemies are harder than their raw XP suggests — more attack rolls per
    round. The band shifts for very small or very large parties."""
    for cap, m in ((1, 1.0), (2, 1.5), (6, 2.0), (10, 2.5), (14, 3.0)):
        if count <= cap:
            mult = m
            break
    else:
        mult = 4.0
    if party_size < 3:                    # a duo feels every hit
        mult *= 1.5
    elif party_size > 5:
        mult /= 1.5
    return mult


def assess(levels: list[int], monsters: list[dict]) -> dict:
    """levels: each PC's level. monsters: [{"cr": ..., "count": n, "name": str}]."""
    levels = [int(l or 1) for l in levels] or [1]
    budget = [0, 0, 0, 0]
    for lv in levels:
        for i, v in enumerate(THRESHOLDS.get(min(20, max(1, lv)), THRESHOLDS[1])):
            budget[i] += v
    raw = sum(xp_for(m.get("cr")) * int(m.get("count", 1)) for m in monsters)
    total = sum(int(m.get("count", 1)) for m in monsters)
    adj = raw * multiplier(total, len(levels))

    if adj < budget[0]:
        band, advice = "trivial", "They will walk through this. Fine as a warm-up or a mook wave."
    elif adj < budget[1]:
        band, advice = "easy", "A speed bump. Good for pacing, not for tension."
    elif adj < budget[2]:
        band, advice = "medium", "A real fight they should win. The standard encounter."
    elif adj < budget[3]:
        band, advice = "hard", "Someone will probably go down. Good for a set piece."
    else:
        over = adj / max(1, budget[3])
        band = "deadly" if over < 2 else "LETHAL"
        advice = ("A character could die. Run it only if you mean to." if over < 2 else
                  f"This is {over:.1f}x the deadly threshold — expect a TPK. "
                  "Cut the numbers, or give them a way out that isn't a fight.")
    return {"band": band, "advice": advice, "adjusted_xp": int(adj), "raw_xp": int(raw),
            "party": f"{len(levels)} PCs, levels {'/'.join(str(l) for l in levels)}",
            "thresholds": dict(zip(("easy", "medium", "hard", "deadly"), budget))}


_ORDER = ["trivial", "easy", "medium", "hard", "deadly", "LETHAL"]


def verdict(r: dict) -> str:
    """Judge the fight against the TABLE'S target, not against neutral 5e defaults.

    This table wants combat very difficult but survivable, so a medium encounter is a miss
    that needs fixing — the same as an accidentally lethal one. Saying so here means the DM
    is told to adjust with a number in hand, rather than being asked to aim by feel.
    """
    c = prefs().get("combat", {})
    floor = c.get("floor_band", "hard")
    ceiling = float(c.get("ceiling_multiple", 1.6))
    deadly = max(1, r["thresholds"]["deadly"])
    over = r["adjusted_xp"] / deadly

    if over > ceiling:
        return (f"ABOVE THE TABLE'S CEILING ({over:.1f}x deadly, limit {ceiling:g}x). This "
                "stops being hard and becomes a coin flip on initiative. Cut the numbers or "
                "split it into two waves.")
    if _ORDER.index(r["band"]) < _ORDER.index(floor):
        need = int(deadly * 0.75) - r["adjusted_xp"]
        return (f"BELOW TARGET — this table wants {floor} or above and this is {r['band']}. "
                f"Add roughly {max(1, need)} more adjusted XP: more bodies, a second wave, or "
                "a nastier [VARIANTS] stat block. Do not run it as-is unless it's a "
                "deliberate mook wave to burn resources.")
    fair = ("Make it survivable by playing it well, not by softening it: telegraph the big "
            "hits, give the enemies morale, leave them an out that isn't winning.")
    if over < 0.85 * float(c.get("aim_multiple", 1.0)):
        return (f"LOW END OF TARGET ({over:.1f}x deadly — the table aims at "
                f"{c.get('aim_multiple', 1.0):g}x). It qualifies as {r['band']}, but one more "
                f"body or a nastier variant would put it where they want it. {fair}")
    return (f"ON TARGET for this table ({over:.1f}x the deadly threshold, ceiling {ceiling:g}x). "
            + fair)


def summary(levels: list[int], monsters: list[dict]) -> str:
    r = assess(levels, monsters)
    who = ", ".join(f"{m.get('count', 1)}x {m.get('name', 'monster')} (CR {m.get('cr', '?')})"
                    for m in monsters) or "nothing"
    return (f"{who} against {r['party']} is {r['band'].upper()} "
            f"({r['adjusted_xp']} adjusted XP vs deadly threshold "
            f"{r['thresholds']['deadly']}). {verdict(r)}")


def for_state(state, proposed: list[dict] | None = None) -> str:
    """Assess what is actually on the board, or a proposed line-up of SRD monster names."""
    levels = [c.level or 1 for c in state.combatants.values() if c.is_pc]
    if not levels:
        return "No player characters to weigh this against."
    if proposed:
        mons = []
        for p in proposed:
            m = rules.monster(p["name"])
            mons.append({"name": p["name"], "count": int(p.get("count", 1)),
                         "cr": (m or {}).get("cr")})
    else:
        mons = [{"name": c.name, "count": 1, "cr": (c.profile or {}).get("cr")}
                for c in state.combatants.values() if not c.is_pc and c.hp > 0]
    if not mons:
        return "No monsters on the board."
    return summary(levels, mons)
