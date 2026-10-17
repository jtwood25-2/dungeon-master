"""Draw the map of the place the fiction describes, instead of a dungeon shaped like it.

The old generator is procedural and seeded by the location STRING, so it cannot know anything
about the place. Asked for Sixpence Reach — which Opus had already written as "a quarter-mile
stone quay ending in a gibbet with three nooses, the ropes kept glossy with whale oil" — it
produced rooms and corridors:

    ########################
    #.......###....##.*~~*.#
    #.......###.......~~~..#
    #.......+.+o......*..*.#

A nautical campaign about a drowned city, drawn as a dungeon crawl. The description existed the
whole time; nothing read it.

THE MODEL DOES NOT DRAW THE GRID. That was the first attempt and it failed exactly where you
would expect: asked for twenty-four characters across, qwen2.5:14b produced sixty-four, in a
repeating tile pattern. Counting is not what it is for.

So the work is split along the line each side is actually good at:

  the model   says WHAT IS THERE and ROUGHLY WHERE, in normalised 0-1 coordinates: water along
              the east edge, a quay running north to south, a gibbet at the south end, a loft
              in the north-west. This is a reading-comprehension task and it is good at it.
  this file   rasterises that onto an exact grid, seals the border, guarantees every walkable
              square is reachable, and throws the whole thing away for the procedural layout if
              any of it fails. Counting, geometry and verification have right answers.

A wrong map is worse than a generic one, because the table plans around it.
"""
import json
import logging
import re

logger = logging.getLogger("cartograph")

TILES = "#.+~o*,^T="

# What the model is allowed to place, and the tile each becomes. Deliberately small: a handful
# of things that read clearly on a battle map at a glance, rather than a vocabulary nobody can
# tell apart once it is drawn in two colours.
WHAT = {
    "water": "~", "sea": "~", "river": "~", "canal": "~", "flood": "~",
    "wall": "#", "building": "#", "rock": "#", "cliff": "#", "hull": "#",
    "rubble": ",", "scrub": ",", "debris": ",", "wreck": ",",
    "pillar": "*", "post": "*", "mast": "*", "stone": "*", "gibbet": "*",
    "table": "T", "stall": "T", "tent": "T", "crate": "o", "cargo": "o",
    "furniture": "o", "shelf": "o", "barrel": "o",
    "stairs": "^", "ramp": "^", "rise": "^", "ladder": "^",
    "bridge": "=", "jetty": "=", "gangplank": "=", "pier": "=",
    "door": "+", "gate": "+",
}

SYS = (
    "You read a description of a place and list what is physically in it, as JSON. You are NOT "
    "drawing anything; another program does the drawing.\n"
    "Reply with ONLY a JSON object:\n"
    '{"features":[{"what":"...","shape":"band|rect|blob|line","x":0.0,"y":0.0,'
    '"w":0.0,"h":0.0,"side":"north|south|east|west","name":"..."}]}\n'
    "x, y, w and h are fractions of the map from 0 to 1. x=0 is west, y=0 is north.\n"
    "\"what\" must be one of: water, wall, building, rock, cliff, rubble, scrub, wreck, pillar, "
    "post, mast, stone, gibbet, table, stall, tent, crate, cargo, shelf, barrel, stairs, ramp, "
    "bridge, jetty, pier, door, gate.\n"
    "shape: \"band\" runs along a named side (give side and w as its thickness). \"rect\" is a "
    "block (give x, y, w, h). \"blob\" is a small cluster at a point (give x, y). \"line\" is a "
    "run of one tile between x,y and x+w,y+h.\n"
    "Give 4 to 10 features. Name the two or three that matter, using the names in the text.\n"
    "MOST OF THE MAP MUST STAY OPEN GROUND so a fight can happen on it. Do not fill it."
)


def describe(state, location: str) -> str:
    """The best description of this place we have, without calling anything.

    The prepared briefing first: it was written by Opus before the session, it is already on
    disk, and it names what is physically there. Everything else falls back to the scene the DM
    just described, which is what makes this work somewhere nobody planned.
    """
    try:
        import prepared
        have = prepared._load(getattr(state, "campaign", "default"))
        want = (location or "").strip().lower()
        for rec in have.values():
            nm = (rec.get("name") or "").strip().lower()
            if nm and (nm in want or want in nm):
                return rec.get("text") or ""
    except Exception:
        pass
    bits = [getattr(state, "scene", ""), location]
    return "\n".join(b for b in bits if b and b.strip())[:1500]


def _blank(w: int, h: int) -> list:
    return [["." if 0 < x < w - 1 and 0 < y < h - 1 else "#" for x in range(w)] for y in range(h)]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rasterise(features: list, w: int, h: int) -> tuple:
    """Features to a grid. Every coordinate is clamped, so bad numbers cannot escape this."""
    g = _blank(w, h)
    named = []

    def put(x, y, ch):
        if 0 < x < w - 1 and 0 < y < h - 1:
            g[y][x] = ch

    for f in features:
        if not isinstance(f, dict):
            continue
        ch = WHAT.get(str(f.get("what", "")).strip().lower())
        if not ch:
            continue
        shape = str(f.get("shape", "blob")).strip().lower()
        try:
            fx = float(f.get("x", 0.5)); fy = float(f.get("y", 0.5))
            fw = float(f.get("w", 0.2)); fh = float(f.get("h", 0.2))
        except (TypeError, ValueError):
            continue
        cx, cy = int(_clamp(fx, 0, 1) * (w - 1)), int(_clamp(fy, 0, 1) * (h - 1))
        if shape == "band":
            side = str(f.get("side", "east")).strip().lower()
            thick = max(1, int(_clamp(fw or 0.18, 0.04, 0.4) * (w if side in ("east", "west") else h)))
            for y in range(h):
                for x in range(w):
                    hit = (side == "west" and x <= thick) or (side == "east" and x >= w - 1 - thick) \
                        or (side == "north" and y <= thick) or (side == "south" and y >= h - 1 - thick)
                    if hit:
                        put(x, y, ch)
        elif shape == "rect":
            rw = max(1, int(_clamp(fw, 0.05, 0.6) * w)); rh = max(1, int(_clamp(fh, 0.05, 0.6) * h))
            for y in range(cy, cy + rh):
                for x in range(cx, cx + rw):
                    put(x, y, ch)
            if f.get("name"):
                named.append((_clamp(cx + rw // 2, 1, w - 2), _clamp(cy, 1, h - 2), str(f["name"])[:28]))
            continue
        elif shape == "line":
            steps = max(abs(int(fw * w)), abs(int(fh * h)), 1)
            for i in range(steps + 1):
                put(cx + int(fw * w * i / steps), cy + int(fh * h * i / steps), ch)
        else:                                            # blob
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):
                put(cx + dx, cy + dy, ch)
        if f.get("name"):
            named.append((_clamp(cx, 1, w - 2), _clamp(cy, 1, h - 2), str(f["name"])[:28]))
    return "\n".join("".join(r) for r in g), named


def open_up(grid: str) -> str:
    """Reconnect anything the features walled off, rather than discarding an otherwise good map.

    The model places a building across a quay without meaning to seal the far end; carving one
    square through is what a cartographer would do and is far better than falling back to a
    generic dungeon over a single wall.
    """
    import planner
    rows = [list(r) for r in grid.splitlines()]
    h, w = len(rows), len(rows[0])
    walk = lambda x, y: rows[y][x] not in "#~"

    def regions():
        seen, out = set(), []
        for y in range(h):
            for x in range(w):
                if walk(x, y) and (x, y) not in seen:
                    stack, cur = [(x, y)], set()
                    while stack:
                        a, b = stack.pop()
                        if (a, b) in cur or not (0 <= a < w and 0 <= b < h) or not walk(a, b):
                            continue
                        cur.add((a, b))
                        stack += [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
                    seen |= cur; out.append(cur)
        return sorted(out, key=len, reverse=True)

    for _ in range(12):
        rs = regions()
        if len(rs) <= 1:
            break
        main, other = rs[0], rs[1]
        best = min(((abs(ax-bx)+abs(ay-by), (ax, ay), (bx, by))
                    for ax, ay in list(other)[:400] for bx, by in list(main)[:400]),
                   default=None)
        if not best:
            break
        _d, (ax, ay), (bx, by) = best
        x, y = ax, ay
        while (x, y) != (bx, by):
            x += (bx > x) - (bx < x)
            if 0 < x < w - 1 and 0 < y < h - 1 and rows[y][x] in "#~":
                rows[y][x] = "="                          # a plank across, not a hole in a wall
            y += (by > y) - (by < y)
            if 0 < x < w - 1 and 0 < y < h - 1 and rows[y][x] in "#~":
                rows[y][x] = "="
    return "\n".join("".join(r) for r in rows)


def valid(grid: str, w: int, h: int) -> tuple:
    """(ok, why). Everything with a right answer is checked, not hoped for."""
    import planner
    rows = grid.splitlines()
    if len(rows) != h or any(len(r) != w for r in rows):
        return False, "wrong size"
    if any(c not in TILES for r in rows for c in r):
        return False, "unknown tile"
    floor = sum(r.count(".") for r in rows)
    if floor < w * h * 0.20:
        return False, f"only {floor} open squares of {w*h} — no room to fight"
    if not planner.reachable(grid):
        return False, "a sealed-off area"
    return True, "ok"


async def draw(state, location: str, title: str, w: int = 24, h: int = 16):
    """(grid, named_rooms) for this place, or (None, []) to use the procedural layout."""
    desc = describe(state, location)
    if not (desc or "").strip():
        return None, []
    raw = await _ask(f"PLACE: {title}\n\nDESCRIPTION:\n{desc}\n\nList its features as JSON now.")
    feats = _features(raw)
    if not feats:
        logger.info(f"map: no usable features for {title!r}; procedural layout")
        return None, []
    grid, named = rasterise(feats, w, h)
    grid = open_up(grid)
    ok, why = valid(grid, w, h)
    if not ok:
        logger.info(f"map: layout for {title!r} rejected ({why}); procedural layout")
        return None, []
    logger.info(f"map: drawn from the description of {title!r} ({len(feats)} features)")
    return grid, named


def _features(raw: str) -> list:
    """Pull the feature list out of whatever came back. Never raises."""
    t = (raw or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
    except Exception:
        return []
    f = d.get("features") if isinstance(d, dict) else None
    return [x for x in f if isinstance(x, dict)][:12] if isinstance(f, list) else []


async def _ask(prompt: str) -> str:
    """The local model: free, resident, and this is comprehension rather than composition."""
    import config
    import httpx
    try:
        async with httpx.AsyncClient(timeout=float(getattr(config, "OLLAMA_TIMEOUT", 60))) as c:
            r = await c.post(f"{config.OLLAMA_URL}/api/chat", json={
                "model": config.OLLAMA_MODEL, "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE, "format": "json",
                "messages": [{"role": "system", "content": SYS},
                             {"role": "user", "content": prompt}],
                "options": {"temperature": 0.2, "num_predict": 700}})
            r.raise_for_status()
            return r.json()["message"]["content"] or ""
    except Exception as e:
        logger.info(f"map: local model unavailable ({type(e).__name__}); procedural layout")
        return ""
