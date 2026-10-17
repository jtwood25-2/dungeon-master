"""Persistent map library — a place once drawn is a place that stays drawn.

The engine previously kept exactly one map (state.map_svg). Walking into a second
location overwrote the first, and coming back later redrew it from scratch: a new
layout, freshly invented room names, and the fog reset, so the same cave was a
different cave every visit.

Here each location gets a file under state/maps/<campaign>/<slug>.json holding its
grid, room names and explored squares. Revisiting restores that exact map — including
what the party had already uncovered — and costs no model call. Maps are kept until
explicitly cleared.
"""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent / "state" / "maps"


# The DM re-words a place every time it mentions it — "the mine", "The Mine, dark and wet",
# "the old mine" — and each spelling was a separate map, so one place quietly became three.
# Trim the parts that are description rather than identity, and keep the rest exact: "the
# mine entrance" really is a different battle map from "the mine", and must stay one.
_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)
_TRAILING_DESC = re.compile(r"\s*[,;(—–-]\s.*$")       # "the mine, dark and wet" -> "the mine"


def normalise(location: str) -> str:
    s = (location or "").strip()
    s = _TRAILING_DESC.sub("", s)
    s = _ARTICLE.sub("", s).strip()
    return s or (location or "").strip()


def slug(location: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", normalise(location).lower()).strip("-")
    return s[:60] or "unnamed"


def _dir(campaign: str) -> Path:
    return ROOT / slug(campaign or "default")


def path(campaign: str, location: str) -> Path:
    return _dir(campaign) / f"{slug(location)}.json"


def save(campaign: str, location: str, *, title: str, grid: str,
         rooms: list, seen: list) -> None:
    p = path(campaign, location)
    prev = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    row = {"location": location, "title": title, "grid": grid, "rooms": rooms,
           "seen": sorted(set(seen or [])), "slug": slug(location),
           "created": prev.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S"),
           "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           # this added 0 whenever the map already existed, so it read 1 forever
           "visits": int(prev.get("visits") or 0) + 1}
    _write(p, row)


def load(campaign: str, location: str) -> dict | None:
    p = path(campaign, location)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def visit(campaign: str, location: str) -> int:
    """Count a return to a place already in the library.

    save() only runs when a map is first DRAWN — a revisit restores and returns early — so
    incrementing there alone left this reading 1 forever however often they came back.
    """
    row = load(campaign, location)
    if not row:
        return 0
    row["visits"] = int(row.get("visits") or 0) + 1
    row["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(path(campaign, location), row)
    return row["visits"]


def _write(p: Path, row: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(row), encoding="utf-8")
    tmp.replace(p)                       # atomic: never truncate a map on a crash


def touch_seen(campaign: str, location: str, seen: list) -> None:
    """Persist exploration progress so fog-of-war survives a revisit."""
    row = load(campaign, location)
    if not row:
        return
    merged = sorted(set(row.get("seen") or []) | set(seen or []))
    if merged != row.get("seen"):
        row["seen"] = merged
        row["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write(path(campaign, location), row)


def listing(campaign: str) -> list[dict]:
    d = _dir(campaign)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"slug": r.get("slug") or p.stem, "title": r.get("title") or p.stem,
                    "location": r.get("location", ""), "updated": r.get("updated", ""),
                    "rooms": len(r.get("rooms") or []), "seen": len(r.get("seen") or [])})
    return sorted(out, key=lambda r: r["updated"], reverse=True)


def by_slug(campaign: str, s: str) -> dict | None:
    p = _dir(campaign) / f"{slug(s)}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def svg_of(campaign: str, s: str, render) -> str:
    """Render a stored map by slug. `render` takes a state-like object -- the caller
    passes its own renderer so this module stays free of drawing concerns."""
    row = by_slug(campaign, s)
    if not row:
        return ""
    class _M:                       # minimal shape _render_map reads
        map_grid = row["grid"]
        map_rooms = row["rooms"]
        map_seen = row.get("seen") or [0]
        map_title = row.get("title") or s
        location = row.get("location") or ""
        combatants: dict = {}
        map_svg = ""
    return render(_M())


def clear(campaign: str, s: str | None = None) -> int:
    """Delete one map (by slug) or every map in the campaign. Returns how many went."""
    d = _dir(campaign)
    if not d.exists():
        return 0
    targets = [d / f"{slug(s)}.json"] if s else list(d.glob("*.json"))
    n = 0
    for p in targets:
        try:
            if p.exists():
                p.unlink()
                n += 1
        except Exception:
            pass
    return n
