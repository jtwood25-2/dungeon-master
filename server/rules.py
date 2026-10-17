"""Rules lookup for the DM. Cache-first: the local open-content index (srd_cache) is
checked first (in-memory, instant, offline); the live SRD API is only a fallback.

All content is open-licensed (SRD / OGL / CC via Open5e + dnd5eapi). Copyrighted books
are never used. Outputs are concise mechanical summaries.
"""
import hashlib
import json
import urllib.request
import urllib.parse
from pathlib import Path

import srd_cache

CACHE = Path(__file__).parent.parent / "state" / "cache" / "rules"
DND5E = "https://www.dnd5eapi.co/api"

_idx: srd_cache.Index | None = None


def index() -> srd_cache.Index:
    global _idx
    if _idx is None:
        _idx = srd_cache.Index().load()
    return _idx


def _cap(s, n=500):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _norm(cat: str, rec: dict) -> dict:
    """Normalize an open5e/SRD record to a concise, model-friendly summary."""
    name = rec.get("name", "?")
    if cat == "monsters":
        ac = rec.get("armor_class")
        ac = ac[0].get("value") if isinstance(ac, list) and ac else ac
        # actions and ability scores used to be dropped here, which left every spawned monster
        # with an empty stat block -- no attack bonus, no damage dice, nothing to roll. The DM
        # then invented results ("16 beats your AC") because there was genuinely nothing on the
        # board to resolve. Combat cannot be deterministic without these.
        stats = {k[:3]: rec.get(k) for k in ("strength", "dexterity", "constitution",
                                             "intelligence", "wisdom", "charisma")
                 if rec.get(k) is not None}
        return {"kind": "monster", "name": name, "hp": rec.get("hit_points"), "ac": ac,
                "cr": rec.get("challenge_rating"), "size": rec.get("size"), "type": rec.get("type"),
                "speed": rec.get("speed"), "stats": stats,
                "actions": rec.get("actions") or [],
                "special_abilities": rec.get("special_abilities") or []}
    if cat == "spells":
        lvl = rec.get("level_int", rec.get("level"))
        return {"kind": "spell", "name": name, "level": lvl,
                "school": rec.get("school"), "casting_time": rec.get("casting_time"),
                "range": rec.get("range"), "duration": rec.get("duration"),
                "components": rec.get("components"), "desc": _cap(rec.get("desc"))}
    if cat == "conditions":
        return {"kind": "condition", "name": name, "desc": _cap(rec.get("desc"), 400)}
    if cat == "magicitems":
        return {"kind": "item", "name": name, "type": rec.get("type"), "rarity": rec.get("rarity"),
                "attunement": rec.get("requires_attunement"), "desc": _cap(rec.get("desc"))}
    if cat == "sections":
        return {"kind": "rule", "name": name, "desc": _cap(rec.get("desc"), 700)}
    return {"kind": cat, "name": name, "desc": _cap(rec.get("desc"))}


_CAT_ALIAS = {"spell": "spells", "monster": "monsters", "condition": "conditions",
              "item": "magicitems", "magicitem": "magicitems", "rule": "sections",
              "feat": "feats", "race": "races", "class": "classes"}


def lookup(category: str, name: str, live: bool = False) -> dict | None:
    """Cache-first, offline, ~microseconds. If not in the local open-content cache it's
    almost certainly Product Identity (not in any open source) -> return None instantly so
    the DM falls back to its own knowledge, rather than stalling on a network miss. Pass
    live=True to also try the SRD API."""
    cat = _CAT_ALIAS.get(category.lower().rstrip("s"), category.lower())
    hit = index().get(name, cat if cat in srd_cache.CATEGORIES else None)
    if hit:
        return _norm(hit[0], hit[1])
    return _live(category, name) if live else None


def variants(name: str, category: str = "monsters") -> list[dict]:
    """Every open book's take on the same creature. The alternative stat blocks are where the
    more interesting versions live — Level Up's Monstrous Menagerie gives ordinary monsters
    real tactics — and they were being discarded by the name-keyed index."""
    out = []
    for rec in index().variants(name, category):
        n = _norm(category, rec)
        n["source"] = rec.get("document__title") or rec.get("document__slug") or "?"
        out.append(n)
    return out


def spell(name):     return lookup("spell", name)
def monster(name):   return lookup("monster", name)
def condition(name): return lookup("condition", name)
def item(name):      return lookup("item", name)
def search(query, limit=8): return index().search(query, limit)


# -- live fallback (only when a name isn't in the local cache) -----------------
def _live(category: str, name: str) -> dict | None:
    slug = name.strip().lower().replace("'", "").replace(" ", "-")
    cat = {"spell": "spells", "monster": "monsters", "condition": "conditions"}.get(
        category.lower().rstrip("s"))
    if not cat:
        return None
    d = _get(f"{DND5E}/{cat}/{slug}")
    if d.get("error") or not d.get("name"):
        return None
    if cat == "monsters":
        ac = d.get("armor_class")
        ac = ac[0]["value"] if isinstance(ac, list) and ac else ac
        return {"kind": "monster", "name": d["name"], "hp": d.get("hit_points"), "ac": ac,
                "cr": d.get("challenge_rating")}
    if cat == "spells":
        return {"kind": "spell", "name": d["name"], "level": d.get("level"),
                "school": (d.get("school") or {}).get("name"), "casting_time": d.get("casting_time"),
                "range": d.get("range"), "duration": d.get("duration"),
                "components": ",".join(d.get("components", [])), "desc": _cap(" ".join(d.get("desc", [])))}
    return {"kind": "condition", "name": d["name"], "desc": _cap(" ".join(d.get("desc", [])), 400)}


def _get(url: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if key.exists():
        return json.loads(key.read_text())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dungeon-master/0.1"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        key.write_text(json.dumps(data))
        return data
    except Exception as e:
        return {"error": str(e)}
