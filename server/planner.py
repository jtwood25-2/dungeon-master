"""Offline campaign planning -> printable prep sheets + battle-map grids.

Generates ORIGINAL content (never reproduces published adventures) via the configured
backend, then renders a print-friendly HTML pack: the campaign plan, the party roster,
and square-grid battle maps players can drop minis onto. Latency doesn't matter here
(it's prep), so the Claude Code path uses a simple one-shot call.
"""
import asyncio
import random
import re
from pathlib import Path

import config

PREP = Path(__file__).parent.parent / "state" / "prep"

CAMPAIGN_PROMPT = (
    'Design an ORIGINAL {n}-session tabletop RPG campaign from this premise: "{premise}". '
    'Do NOT copy any published adventure or copyrighted setting — invent your own. Output '
    'markdown with these sections: ## Overview, ## Key NPCs (name — one-line motive & secret), '
    '## Locations, ## Story Beats (session by session), ## Encounters (2-4; foes with rough '
    'HP/AC + a twist). Concise and table-ready.')

TILES = {
    "#": "wall", ".": "floor", "+": "door", "~": "water", "o": "furniture",
    "*": "pillar", ",": "rubble", "^": "stairs", "T": "table", "=": "bridge",
}

# The SAME tile means different things indoors and out. The outdoor generator was already
# laying down tents, spoil heaps and standing stones -- and the legend called them "table",
# "furniture" and "pillar", all drawn the same grey, so a riverbank read as a furnished room.
TILE_LABELS = {
    "dungeon": TILES,
    "outdoor": {"#": "edge", ".": "open ground", "+": "gate", "~": "water",
                "o": "spoil heap", "*": "standing stone", ",": "scrub",
                "^": "rise", "T": "tent", "=": "bridge"},
}
# and they should not all be the same colour, or nothing on the map is distinguishable
TILE_FILL = {
    "dungeon": {},
    "outdoor": {"*": "#6f8f5e", ",": "#8aa473", "T": "#c9a87c", "o": "#a8916b", "^": "#b9a06a"},
}


def _fill(kind: str, ch: str, default: str) -> str:
    return TILE_FILL.get(kind, {}).get(ch, default)


def map_kind(location: str) -> str:
    """Indoors wins. A location that names an enclosed space IS one, whatever else it says.

    "the Long Cut tunnel" came out as an open-air map with trees on it, because the outdoor
    pattern matched "cut" and "dig" -- both from "the Long Cut", which is the PROPER NAME of a
    canal works. The word actually describing the space was "tunnel", and it lost. The party
    was underground in a mine and got a map of a wood.
    """
    if _INDOOR.search(location or ""):
        return "dungeon"
    return "outdoor" if _OUTDOOR.search(location or "") else "dungeon"

MAP_PROMPT = (
    'Draw a top-down BATTLE MAP for this location: "{loc}". Output ONLY a character grid, '
    'exactly {w} columns by {h} rows, using these characters:\n'
    '  # wall      . floor     + door      ~ water     o furniture/crate\n'
    '  * pillar    , rubble    ^ stairs    T table     = bridge\n'
    'Make it TACTICALLY INTERESTING: distinct rooms connected by doors, cover to hide behind, '
    'a hazard or choke point, and at least one alternate route. Enclose the outside with walls. '
    'Output exactly {h} lines of exactly {w} characters and nothing else — no labels, no prose.')

ROOMS_PROMPT = (
    'A battle map of "{loc}" has {n} distinct areas. Invent a short evocative name for each, '
    'fitting that location. AT MOST TWO WORDS each — they are printed inside small squares, '
    'so "Flooded Cellar" works and "Chamber of the Forgotten Sentinel" does not. One per '
    'line, no numbering, no other text. Exactly {n} lines.')


async def _generate_local(prompt: str, max_tokens=1200, timeout=90) -> str:
    """The resident local model. Drawing a grid is a structural job rather than a creative
    one, and a Claude Code one-shot cold-starts the CLI every time -- which timed out at 120s
    and made mid-session maps impossible."""
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{config.OLLAMA_URL}/api/generate", json={
            "model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0.6, "num_predict": max_tokens}})
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


async def _generate(prompt: str, max_tokens=2200, timeout=200, model: str = "") -> str:
    if config.DM_BACKEND == "api" and config.ANTHROPIC_API_KEY:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = await client.messages.create(model=config.DM_MODEL, max_tokens=max_tokens,
                                           messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")
    use = model or config.PLANNER_CLAUDE_MODEL
    import telemetry
    with telemetry.Turn("claude_code", use, route="planner") as tel:
        tel.prompt(message=prompt)
        proc = await asyncio.create_subprocess_exec(   # one-shot; cold start is fine for prep
            config.CLAUDE_BIN, "-p", prompt, "--model", use,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        text = out.decode(errors="replace").strip()
        tel.first_token(); tel.chunk(text)
        return text


async def plan_campaign(state, premise: str, sessions: int = 3) -> str:
    # the arc is deep, rare work — worth opus, unlike a per-session recap
    plan = await _generate(CAMPAIGN_PROMPT.format(n=sessions, premise=premise),
                           model=config.DIRECTOR_CLAUDE_MODEL)
    state.premise = premise
    state.plan = plan
    return plan


_SETUP = re.compile(r"^(?:===|.+ enters \(|Player .+ joins as |.+ is here \(|Scene: Session not yet started)", re.I)


def _session_script(state, limit: int = 240) -> str:
    """What was actually SAID this session, from the transcript.

    state.log only records tool calls — damage, scene changes, rolls. A session of pure
    roleplay generates almost none, so a night spent in town talking to people produced a log
    of one line and a recap that skipped the whole visit. The next session's DM then narrated
    the party going "off the path, away from the smoke of the town" — it had no idea they had
    ever been there. The transcript has every word; that is what a recap should be built from.
    """
    try:
        import transcript
        # transcripts are one file per DAY, and path_for() only gives today's — a session
        # that started yesterday, or ran past midnight, lived in a file we never opened. Scan
        # every file this campaign has and take the rows belonging to this session.
        stem = transcript.path_for(state.campaign).stem.rsplit(" ", 1)[0]
        entries = []
        for p in sorted(transcript.DIR.glob(f"{stem} *.jsonl")):
            entries += [e for e in transcript.load(p)
                        if e.get("session") == state.session
                        and e.get("role") in ("player", "dm")]
    except Exception:
        return ""
    lines = [f"{'Player' if e['role'] == 'player' else 'DM'}: {e['text']}" for e in entries]
    return chr(10).join(lines[-limit:])


async def recap_session(state) -> str:
    """Summarize this session's events into state.recap so the DM 'remembers' next time."""
    # Setting a campaign up is not playing it. The old guard only caught a totally empty log,
    # so importing a character and opening a session was enough to produce a recap of nothing
    # ("the story is only just beginning") — which then made memory_brief treat a brand new
    # campaign as one already in progress, the exact thing its first-session branch prevents.
    play = [l for l in state.log[-80:]
            if not _SETUP.match(l.strip())]
    script = _session_script(state)
    if len(play) < 2 and not script:
        return state.recap
    events = "\n".join(play)
    if script:
        events = "WHAT WAS SAID:\n" + script + "\n\nMECHANICS:\n" + events
    prompt = (
        "Write a 'previously on…' recap of this tabletop session, 4-8 sentences, for the DM to "
        "read back next time.\n"
        "Cover, in this order: WHERE the party ended up, WHO they met or fought, WHAT changed, "
        "and what is UNRESOLVED.\n"
        "Cover the WHOLE session in order, including scenes that were only conversation — an "
        "evening spent talking to people in a town IS the session, not a gap in it. Name every "
        "place they went and every person they spoke to.\n"
        "Write it as story, not as a log. Never mention hit points, AC, or any other number — "
        "'the wolf was left bloodied' not 'the wolf at 3/14 HP'. Name places and people.\n"
        "Events:\n" + events)
    try:
        state.recap = (await _generate(prompt, max_tokens=300, timeout=90)).strip()
    except Exception:
        pass
    return state.recap


# An interior dungeon and an open-air site want completely different maps. The BSP builder
# below makes a fine crypt or keep, but rendering a riverside dig camp as a sealed stone maze
# of boxy rooms and doors reads as wrong -- most of this campaign happens outdoors, following
# water through wilderness. These decide which builder a location gets.
# Words that name an ENCLOSED space. These beat the outdoor list outright -- see map_kind.
_INDOOR = re.compile(
    r"\b(tunnel|tunnels|shaft|mine|mines|adit|gallery|cave|cavern|grotto|cellar|basement|"
    r"crypt|catacomb|catacombs|tomb|vault|dungeon|chamber|chambers|hall|halls|corridor|"
    r"passage|passageway|room|rooms|temple|chapel|shrine|keep|tower|barracks|prison|cell|"
    r"cells|forge|smithy|library|kitchen|stair|stairs|stairwell|attic|loft|hold|brig|"
    r"warehouse|storeroom|granary|inn|tavern|house|manor|mansion|castle|fort|fortress|"
    # "below" and "beneath" are NOT here: they are prepositions describing where something
    # sits, and "the river road below the hills" is as open-air as it gets.
    r"lodge|hut|cabin|shack|mill|underground)\b", re.I)
_OUTDOOR = re.compile(
    r"\b(camp|encampment|dig|cut|canal|road|trail|path|grove|glade|wood|woods|forest|"
    r"clearing|field|meadow|pasture|orchard|garden|riverbank|river|bank|shore|coast|beach|"
    r"ford|crossing|bridge|dock|docks|wharf|quay|marsh|fen|moor|bog|swamp|ridge|hill|hills|"
    r"slope|ravine|gorge|pass|yard|market|graveyard|cemetery|ruins|palisade|gate|gates|"
    r"spring|well|sump|pit|quarry|hollow|dell|vale|valley|stones|standing)\b", re.I)
_WATERY = re.compile(
    r"\b(cut|canal|river|riverbank|bank|ford|crossing|bridge|dock|docks|wharf|quay|marsh|"
    r"fen|bog|swamp|shore|coast|beach|spring|well|sump|pit|dig|flood|flooded|water|lake|"
    r"pond|stream|brook|creek|mill|drowned)\b", re.I)


def generate_map(location: str, w: int = 24, h: int = 16, seed=None, with_rooms=False):
    """Build the map in Python rather than asking a model to draw one.

    Layout is a structural problem, not a creative one. Asked for a grid, the local model
    returned an alternating "#+#+#+" checkerboard -- walls and doors in a chequer, no rooms,
    no route through it. This produces a connected, tactically sensible map every time, in
    about a millisecond, and the same location always yields the same map.
    """
    rng = random.Random(seed if seed is not None else location.strip().lower())
    if _OUTDOOR.search(location or ""):
        return _generate_outdoor(location, w, h, rng, bool(_WATERY.search(location or "")),
                                 with_rooms)
    g = [["#"] * w for _ in range(h)]

    # --- split the space into rooms (BSP) ---
    def split(x, y, ww, hh, depth):
        min_side = 6
        if depth <= 0 or (ww < min_side * 2 and hh < min_side * 2):
            return [(x, y, ww, hh)]
        horizontal = ww < hh if abs(ww - hh) > 3 else rng.random() < 0.5
        if horizontal and hh >= min_side * 2:
            cut = rng.randint(min_side, hh - min_side)
            return split(x, y, ww, cut, depth - 1) + split(x, y + cut, ww, hh - cut, depth - 1)
        if not horizontal and ww >= min_side * 2:
            cut = rng.randint(min_side, ww - min_side)
            return split(x, y, cut, hh, depth - 1) + split(x + cut, y, ww - cut, hh, depth - 1)
        return [(x, y, ww, hh)]

    leaves = split(0, 0, w, h, 3)
    rooms = []
    for (x, y, ww, hh) in leaves:
        if ww < 5 or hh < 5:
            continue
        # fill most of the leaf: small rooms in a big block leave the map mostly dead wall,
        # which reads as an empty page rather than a place
        rw = max(4, ww - rng.randint(2, 3)); rh = max(3, hh - rng.randint(2, 3))
        rx = x + rng.randint(1, max(1, ww - rw - 1)); ry = y + rng.randint(1, max(1, hh - rh - 1))
        rooms.append((rx, ry, rw, rh))
        for yy in range(ry, min(ry + rh, h - 1)):
            for xx in range(rx, min(rx + rw, w - 1)):
                g[yy][xx] = "."
    if not rooms:                                     # degenerate size -> one open chamber
        for yy in range(1, h - 1):
            for xx in range(1, w - 1):
                g[yy][xx] = "."
        rooms = [(1, 1, w - 2, h - 2)]

    # --- connect them, so there is always a route ---
    def centre(r):
        return (r[0] + r[2] // 2, r[1] + r[3] // 2)

    def carve_h(x1, x2, y):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < x < w - 1 and 0 < y < h - 1 and g[y][x] == "#":
                g[y][x] = "."

    def carve_v(y1, y2, x):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < x < w - 1 and 0 < y < h - 1 and g[y][x] == "#":
                g[y][x] = "."

    ordered = sorted(rooms, key=lambda r: centre(r)[0] + centre(r)[1])
    for a, b in zip(ordered, ordered[1:]):
        (ax, ay), (bx, by) = centre(a), centre(b)
        if rng.random() < 0.5:
            carve_h(ax, bx, ay); carve_v(ay, by, bx)
        else:
            carve_v(ay, by, ax); carve_h(ax, bx, by)
    if len(ordered) > 2 and rng.random() < 0.8:       # a loop, so there's an alternate route
        (ax, ay), (bx, by) = centre(ordered[0]), centre(ordered[-1])
        carve_v(ay, by, ax); carve_h(ax, bx, by)

    # --- doors where a corridor pierces a room wall ---
    for (rx, ry, rw, rh) in rooms:
        for x in range(rx, min(rx + rw, w)):
            for y in (ry - 1, ry + rh):
                if 0 < y < h - 1 and g[y][x] == "." and g[y - 1][x] == "." and g[y + 1][x] == "." \
                        and g[y][max(0, x - 1)] == "#" and g[y][min(w - 1, x + 1)] == "#":
                    g[y][x] = "+"
        for y in range(ry, min(ry + rh, h)):
            for x in (rx - 1, rx + rw):
                if 0 < x < w - 1 and g[y][x] == "." and g[y][x - 1] == "." and g[y][x + 1] == "." \
                        and g[max(0, y - 1)][x] == "#" and g[min(h - 1, y + 1)][x] == "#":
                    g[y][x] = "+"

    # --- features: cover to fight around, a hazard, a way down ---
    big = [r for r in rooms if r[2] >= 5 and r[3] >= 4]
    if big:
        r = rng.choice(big)                            # pillared hall
        for dx in (1, r[2] - 2):
            for dy in (1, r[3] - 2):
                x, y = r[0] + dx, r[1] + dy
                if g[y][x] == ".":
                    g[y][x] = "*"
    if len(big) > 1:
        r = rng.choice([b for b in big if b is not big[0]] or big)   # flooded corner
        for yy in range(r[1] + 1, min(r[1] + 3, h - 1)):
            for xx in range(r[0] + 1, min(r[0] + 4, w - 1)):
                if g[yy][xx] == ".":
                    g[yy][xx] = "~"
    for r in rooms:
        for _ in range(rng.randint(0, 2)):             # crates and tables for cover
            x = rng.randint(r[0], r[0] + r[2] - 1); y = rng.randint(r[1], r[1] + r[3] - 1)
            if 0 < x < w - 1 and 0 < y < h - 1 and g[y][x] == ".":
                g[y][x] = rng.choice("oT")
    for _ in range(rng.randint(3, 7)):                 # rubble / difficult ground
        x, y = rng.randint(1, w - 2), rng.randint(1, h - 2)
        if g[y][x] == ".":
            g[y][x] = ","
    if rooms:                                          # stairs out
        r = rng.choice(rooms)
        x, y = r[0] + r[2] // 2, r[1] + r[3] // 2
        if g[y][x] == ".":
            g[y][x] = "^"

    grid = "\n".join("".join(row) for row in g)
    # Rectangles, not just centres: fog needs to know exactly which squares belong to a room.
    # Flood-filling from a centre leaks down every corridor and reveals half the level.
    rects = [(rx, ry, min(rw, w - rx), min(rh, h - ry)) for (rx, ry, rw, rh) in rooms]
    return (grid, rects) if with_rooms else grid


def _generate_outdoor(location, w, h, rng, watery, with_rooms):
    """An open-air site: not rooms-and-corridors but ground you move freely across, broken by
    terrain. A watercourse (the whole campaign follows one) cuts across with bridges for the
    only crossings -- the natural choke and alternate route -- and cover comes from spoil
    heaps, tents and standing stones rather than walls. Reads as a place, not a maze."""
    g = [["." if 0 < x < w - 1 and 0 < y < h - 1 else "#" for x in range(w)] for y in range(h)]

    def put(x, y, ch):
        if 0 < x < w - 1 and 0 < y < h - 1:
            g[y][x] = ch

    # --- the watercourse runs the long way across and splits the ground into two banks. A
    # dug "cut" or canal is straight and clean; a natural river is allowed to wander. If the
    # place is not wet it is a dry ravine of rubble instead. ---
    vertical = rng.random() < 0.5
    span, cross = (h, w) if vertical else (w, h)
    mid = cross // 2 + rng.randint(-2, 2)
    fill = "~" if watery else ","
    straight = bool(re.search(r"\b(cut|canal|dig|quarry)\b", location or "", re.I))
    width = 3 if straight else 2
    channel = {}                                        # position along span -> near edge of cut
    c = mid
    for s in range(1, span - 1):
        if not straight and rng.random() < 0.35:        # a river bends; a canal does not
            c += rng.choice((-1, 1))
        c = max(2, min(cross - 2 - width, c))
        channel[s] = c
        for d in range(width):
            x, y = (c + d, s) if vertical else (s, c + d)
            put(x, y, fill)

    # bridges: the primary crossing (the choke) and one more further off (the alternate route)
    crossings = sorted(rng.sample(range(2, span - 2), min(2, max(1, span // 6))))
    for s in crossings:
        c = channel.get(s, mid)
        for d in range(-1, 3):
            x, y = (c + d, s) if vertical else (s, c + d)
            if g[y][x] in "~,":
                put(x, y, "=")

    # --- banks split into open zones for labels/fog/tokens. Kept clear of the water. ---
    def clear_zone(zx, zy, zw, zh):
        for yy in range(zy, min(zy + zh, h - 1)):
            for xx in range(zx, min(zx + zw, w - 1)):
                if g[yy][xx] in "#,":
                    put(xx, yy, ".")
        return (zx, zy, min(zw, w - 1 - zx), min(zh, h - 1 - zy))

    rects = []
    if vertical:
        left, right = mid - 2, mid + 4
        for zy in (1, h // 2 + 1):
            rects.append(clear_zone(1, zy, left - 1, h // 2 - 2))
            rects.append(clear_zone(right, zy, w - 1 - right, h // 2 - 2))
    else:
        top, bot = mid - 2, mid + 4
        for zx in (1, w // 2 + 1):
            rects.append(clear_zone(zx, 1, w // 2 - 2, top - 1))
            rects.append(clear_zone(zx, bot, w // 2 - 2, h - 1 - bot))
    rects = [r for r in rects if r[2] >= 3 and r[3] >= 3]

    # --- terrain: spoil heaps for cover, a cluster of tents/crates in each camp zone, standing
    # stones or winch posts, and the dig shaft itself. All landed on real ground. ---
    def floor(x, y):
        return 0 < x < w - 1 and 0 < y < h - 1 and g[y][x] == "."

    def cluster(cx, cy, kind, n):
        placed = 0
        for _ in range(n * 4):
            if placed >= n:
                break
            x, y = cx + rng.randint(-2, 2), cy + rng.randint(-1, 1)
            if floor(x, y):
                put(x, y, rng.choice(kind)); placed += 1

    for (rx, ry, rw, rh) in rects:
        cx, cy = rx + rw // 2, ry + rh // 2
        cluster(cx, cy, "oT", rng.randint(2, 4))        # tents & crates
        if rng.random() < 0.6:
            cluster(rx + 1, ry + rh - 1, ",", rng.randint(2, 4))   # a spoil heap
        if rng.random() < 0.5:
            for _ in range(rng.randint(1, 2)):          # a standing stone / winch post
                x, y = rx + rng.randint(0, rw - 1), ry + rng.randint(0, rh - 1)
                if floor(x, y):
                    put(x, y, "*")

    if rects:                                           # the dig goes down somewhere central
        r = max(rects, key=lambda r: r[2] * r[3])
        x, y = r[0] + r[2] // 2, r[1] + r[3] // 2
        for _ in range(9):
            if floor(x, y):
                put(x, y, "^"); break
            x, y = r[0] + rng.randint(0, r[2] - 1), r[1] + rng.randint(0, r[3] - 1)

    grid = "\n".join("".join(row) for row in g)
    if not rects:                                       # degenerate small size: one open field
        rects = [(1, 1, w - 2, h - 2)]
    return (grid, rects) if with_rooms else grid



def rooms_of(grid: str, min_cells: int = 8) -> list:
    """Bounding boxes of the open areas in a grid we did not generate.

    make_map_full() hands back the rectangles it carved, because it knows them. A grid drawn
    from a description has no such record, and the labels and fog-of-war both need somewhere to
    attach -- so the open regions are found by flood fill and boxed.
    """
    rows = grid.splitlines()
    if not rows:
        return []
    h, w = len(rows), len(rows[0])
    walk = lambda x, y: 0 <= x < w and 0 <= y < h and rows[y][x] not in "#~"
    seen, out = set(), []
    for y in range(h):
        for x in range(w):
            if not walk(x, y) or (x, y) in seen:
                continue
            stack, cur = [(x, y)], []
            while stack:
                a, b = stack.pop()
                if (a, b) in seen or not walk(a, b):
                    continue
                seen.add((a, b)); cur.append((a, b))
                stack += [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
            if len(cur) >= min_cells:
                xs = [c[0] for c in cur]; ys = [c[1] for c in cur]
                out.append((min(xs), min(ys), max(xs)-min(xs)+1, max(ys)-min(ys)+1))
    return sorted(out, key=lambda r: -r[2]*r[3])[:8]

def room_centre(rect) -> tuple:
    x, y, rw, rh = rect
    return (x + rw // 2, y + rh // 2)


def reachable(grid: str) -> bool:
    """Every open square must be walkable from every other -- a map with a sealed room is a
    broken map, and it's cheap to prove it isn't."""
    rows = grid.splitlines()
    open_sq = {(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c not in "#"}
    if not open_sq:
        return False
    start = next(iter(open_sq)); seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if (nx, ny) in open_sq and (nx, ny) not in seen:
                seen.add((nx, ny)); stack.append((nx, ny))
    return seen == open_sq


def make_map_full(location: str, w: int = 24, h: int = 16):
    """Deterministic layout; retried with a nudged seed on the rare disconnected result."""
    for attempt in range(6):
        grid, centres = generate_map(location, w, h, with_rooms=True,
                                     seed=f"{location}{'' if not attempt else attempt}")
        if reachable(grid):
            return grid, centres
    return grid, centres


async def make_map(location: str, w: int = 24, h: int = 16) -> str:
    return make_map_full(location, w, h)[0]


async def name_rooms(location: str, centres: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """Names for the areas, placed on centres CODE worked out.

    Asking the model for coordinates put labels on top of walls -- it can't reliably read a
    grid it didn't build. Geometry comes from the generator, which knows exactly where each
    room is; the model only supplies the words. An unlabelled map is still usable, so any
    failure here just yields no labels.
    """
    if not centres:
        return []
    try:
        raw = await _generate_local(ROOMS_PROMPT.format(loc=location, n=len(centres)),
                                    max_tokens=200)
    except Exception:
        return []
    names = []
    for ln in raw.splitlines():
        n = re.sub(r'^\s*[-*\d.)\s]+', "", ln).strip(' "\'')
        if not n:
            continue
        words = n.split()
        while words and len(" ".join(words)) > 20:     # trim whole words, never mid-word
            words.pop()
        if words:
            names.append(" ".join(words))
    return [(x, y, n) for (x, y), n in zip(centres, names) if n]


# ---- deterministic rendering ------------------------------------------------
# Ink-on-parchment, the way a printed battle map reads best: floors are the paper, walls are
# solid ink, everything else is a drawn symbol rather than a coloured square.
INK, PAPER, STONE = "#2c2419", "#efe4cc", "#6b5c43"
FLOOR, FOG = "#f7efdc", "#584f3f"
_COL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Texture and depth, defined once and reused. A flat two-tone grid reads as a spreadsheet;
# grain on the paper and a drop shadow under the walls make it read as a place.
_DEFS = f"""<defs>
<filter id="grain" x="0" y="0" width="100%" height="100%">
  <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="4" seed="7" result="n"/>
  <feColorMatrix in="n" type="saturate" values="0"/>
  <feComponentTransfer><feFuncA type="linear" slope="0.055"/></feComponentTransfer>
  <feComposite operator="over" in2="SourceGraphic"/>
</filter>
<filter id="drop" x="-20%" y="-20%" width="150%" height="150%">
  <feDropShadow dx="0" dy="1.4" stdDeviation="1.4" flood-color="#000" flood-opacity="0.34"/>
</filter>
<linearGradient id="water" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0%" stop-color="#a9c9e2"/><stop offset="100%" stop-color="#89b0d0"/>
</linearGradient>
<radialGradient id="vig" cx="50%" cy="50%" r="75%">
  <stop offset="60%" stop-color="#000" stop-opacity="0"/>
  <stop offset="100%" stop-color="#000" stop-opacity="0.13"/>
</radialGradient>
</defs>"""


def _floorish(ch: str) -> bool:
    return ch not in ("#", " ")


def room_cells(grid: str, rects: list, idx: int) -> set:
    """The squares of one room: its rectangle, plus the wall ring around it so the room reads
    as an enclosed space rather than a floating patch of floor."""
    rows = grid.splitlines()
    if idx >= len(rects):
        return set()
    x, y, rw, rh = rects[idx]
    out = set()
    for yy in range(y - 1, y + rh + 1):
        for xx in range(x - 1, x + rw + 1):
            if 0 <= yy < len(rows) and 0 <= xx < len(rows[yy]):
                out.add((xx, yy))
    return out


def visible_cells(grid: str, rects: list, seen_rooms: list) -> set | None:
    """Squares the party has actually seen. None means 'show everything'."""
    if not seen_rooms:
        return None
    out = set()
    for i in seen_rooms:
        out |= room_cells(grid, rects, i)
    return out


def grid_to_svg(grid: str, cell: int = 30, title: str = "", rooms: list | None = None,
                kind: str = "dungeon",
                pad: int = 26, tokens: list | None = None, visible: set | None = None) -> str:
    """Render the grid as a printable battle map: labelled squares, a legend and a scale bar.

    Coordinates along the edges matter more than they look: running this by VOICE, "the ogre
    is at F7" is the only practical way to place something on a shared map.
    """
    rows = grid.splitlines()
    h = len(rows); w = max((len(r) for r in rows), default=0)
    if not h or not w:
        return ""
    at = lambda x, y: (rows[y][x] if 0 <= y < h and 0 <= x < len(rows[y]) else "#")

    top = pad + (22 if title else 0)
    W, H = w * cell + pad * 2, h * cell + top + pad + 34
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Georgia,serif">']
    o.append(_DEFS)
    o.append(f'<rect width="{W}" height="{H}" fill="{PAPER}" filter="url(#grain)"/>')
    if title:
        o.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-size="17" '
                 f'fill="{INK}" font-style="italic">{_esc(title)}</text>')

    gx, gy = pad, top
    o.append(f'<g transform="translate({gx},{gy})">')
    o.append(f'<rect width="{w*cell}" height="{h*cell}" fill="{FLOOR}"/>')

    # Walls as one merged silhouette with a shadow beneath the floor layer, so rooms look
    # carved out of stone rather than tiled with black squares.
    wall_path = []
    for y in range(h):
        for x in range(w):
            if at(x, y) == "#" and (visible is None or (x, y) in visible or True):
                wall_path.append(f"M{x*cell} {y*cell}h{cell}v{cell}h-{cell}z")
    if wall_path:
        o.append(f'<path d="{"".join(wall_path)}" fill="{INK}" filter="url(#drop)"/>')

    for y in range(h):
        for x in range(w):
            ch = at(x, y)
            X, Y = x * cell, y * cell
            if visible is not None and (x, y) not in visible:
                continue                          # unexplored: drawn as fog further down
            if ch == "#":
                continue                              # already drawn as the wall silhouette
            elif ch == "~":
                o.append(f'<rect x="{X}" y="{Y}" width="{cell}" height="{cell}" fill="url(#water)"/>')
                o.append(f'<path d="M{X+4} {Y+cell*0.62} q{cell*0.18} -7 {cell*0.36} 0 '
                         f'q{cell*0.18} 7 {cell*0.36} 0" fill="none" stroke="#7ba3c4" '
                         f'stroke-width="1.6"/>')
            elif ch == "=":
                o.append(f'<rect x="{X}" y="{Y}" width="{cell}" height="{cell}" fill="#d8c39a"/>')
                for i in (0.3, 0.7):
                    o.append(f'<line x1="{X}" y1="{Y+cell*i}" x2="{X+cell}" y2="{Y+cell*i}" '
                             f'stroke="{STONE}" stroke-width="1.3"/>')
            elif ch == "+":
                o.append(f'<rect x="{X}" y="{Y}" width="{cell}" height="{cell}" fill="#f7efdc"/>')
                horiz = at(x, y - 1) == "#" and at(x, y + 1) == "#"
                if horiz:
                    o.append(f'<rect x="{X+2}" y="{Y+cell*0.28}" width="{cell-4}" '
                             f'height="{cell*0.44}" fill="#c8a24a" stroke="{INK}" stroke-width="1.6"/>')
                else:
                    o.append(f'<rect x="{X+cell*0.28}" y="{Y+2}" width="{cell*0.44}" '
                             f'height="{cell-4}" fill="#c8a24a" stroke="{INK}" stroke-width="1.6"/>')
            elif ch == "*":
                o.append(f'<circle cx="{X+cell/2}" cy="{Y+cell/2}" r="{cell*0.3}" '
                         f'fill="{_fill(kind, "*", STONE)}" stroke="{INK}" stroke-width="1.5"/>')
            elif ch in ("o", "T"):
                s = cell * (0.62 if ch == "T" else 0.5)
                o.append(f'<rect x="{X+(cell-s)/2}" y="{Y+(cell-s)/2}" width="{s}" height="{s}" '
                         f'rx="2" fill="{_fill(kind, ch, "#b79b6e")}" stroke="{INK}" stroke-width="1.4"/>')
            elif ch == ",":
                for dx, dy in ((0.3, 0.35), (0.62, 0.5), (0.42, 0.7)):
                    o.append(f'<circle cx="{X+cell*dx}" cy="{Y+cell*dy}" r="2.1" '
                             f'fill="{_fill(kind, ",", STONE)}"/>')
            elif ch == "^":
                for i in range(4):
                    o.append(f'<line x1="{X+3}" y1="{Y+4+i*(cell-8)/3}" x2="{X+cell-3}" '
                             f'y2="{Y+4+i*(cell-8)/3}" stroke="{STONE}" stroke-width="2"/>')

    # grid lines only over open ground, so walls stay solid
    for y in range(h):
        for x in range(w):
            if visible is not None and (x, y) not in visible:
                continue
            if _floorish(at(x, y)):
                o.append(f'<rect x="{x*cell}" y="{y*cell}" width="{cell}" height="{cell}" '
                         f'fill="none" stroke="{INK}" stroke-opacity="0.13" stroke-width="1"/>')

    # fog over everything unexplored -- the map should reveal as they go, not spoil the level
    if visible is not None:
        for y in range(h):
            for x in range(w):
                if (x, y) not in visible:
                    o.append(f'<rect x="{x*cell}" y="{y*cell}" width="{cell}" height="{cell}" '
                             f'fill="{FOG}"/>')
    o.append(f'<rect width="{w*cell}" height="{h*cell}" fill="none" stroke="{INK}" stroke-width="2.5"/>')

    # labels sit on a parchment plate: without it they collide with doors and stairs and
    # become unreadable exactly where the map is busiest
    for rx, ry, name in (rooms or []):
        cx, cy = rx * cell + cell / 2, ry * cell + cell / 2
        wpx = len(name) * 6.4 + 12
        cx = min(max(cx, wpx / 2 + 2), w * cell - wpx / 2 - 2)    # keep the plate on the map
        o.append(f'<rect x="{cx-wpx/2}" y="{cy-9.5}" width="{wpx}" height="17" rx="8" '
                 f'fill="#f7efdc" fill-opacity="0.88"/>')
        o.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="12" '
                 f'fill="{INK}" fill-opacity="0.85" font-style="italic" '
                 f'letter-spacing="0.4">{_esc(name)}</text>')
    # tokens: who is standing where. Initials on a disc, players in green, enemies in red.
    for t in (tokens or []):
        tx, ty = t.get("pos", (0, 0))
        if visible is not None and (tx, ty) not in visible:
            continue                                   # not seen yet -> not on the map
        cx, cy = tx * cell + cell / 2, ty * cell + cell / 2
        r = cell * 0.38
        fill = "#3f7d52" if t.get("is_pc") else "#9c3b34"
        o.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{PAPER}" '
                 f'stroke-width="2" filter="url(#drop)"/>')
        o.append(f'<text x="{cx}" y="{cy+4.4}" text-anchor="middle" font-size="{cell*0.42:.0f}" '
                 f'font-weight="700" fill="{PAPER}" font-family="Georgia,serif">'
                 f'{_esc(t.get("label", "?"))}</text>')

    o.append(f'<rect width="{w*cell}" height="{h*cell}" fill="url(#vig)" pointer-events="none"/>')
    o.append("</g>")

    # edge coordinates -- "the ogre is at F7" is how you place things by voice
    for x in range(w):
        o.append(f'<text x="{gx+x*cell+cell/2}" y="{gy-6}" text-anchor="middle" font-size="10" '
                 f'fill="{STONE}">{_COL[x % 26]}</text>')
    for y in range(h):
        o.append(f'<text x="{gx-7}" y="{gy+y*cell+cell/2+3.5}" text-anchor="end" font-size="10" '
                 f'fill="{STONE}">{y+1}</text>')

    # legend + scale
    ly = gy + h * cell + 21
    labels = TILE_LABELS.get(kind, TILES)
    seen = {c for r in rows for c in r if c in labels and c != "."}
    items, lx = [], gx
    for ch in ("#", "+", "~", "*", "o", "T", ",", "^", "="):
        if ch in seen:
            items.append((ch, labels[ch]))
    for ch, label in items[:7]:
        swatch = (INK if ch == "#" else "#bcd6e8" if ch == "~" else "#c8a24a" if ch == "+"
                  else TILE_FILL.get(kind, {}).get(ch, STONE))
        o.append(f'<rect x="{lx}" y="{ly-9}" width="11" height="11" rx="1.5" fill="{swatch}"/>')
        o.append(f'<text x="{lx+16}" y="{ly}" font-size="11" fill="{INK}">{label}</text>')
        lx += 22 + len(label) * 6.2
    o.append(f'<line x1="{W-pad-cell*2}" y1="{ly-4}" x2="{W-pad}" y2="{ly-4}" '
             f'stroke="{INK}" stroke-width="2"/>')
    o.append(f'<text x="{W-pad-cell}" y="{ly+11}" text-anchor="middle" font-size="10" '
             f'fill="{STONE}">10 ft</text>')
    o.append("</svg>")
    return "".join(o)


def place_tokens(grid: str, rects: list, state) -> list:
    """Put everyone on the board: the party together in one room, each enemy group elsewhere.
    Positions persist on the combatants so the DM can move them between turns."""
    rows = grid.splitlines()
    free = lambda x, y: (0 <= y < len(rows) and 0 <= x < len(rows[y])
                         and rows[y][x] in ".,^" and (x, y) not in taken)
    taken: set = set()
    pcs = [c for c in state.combatants.values() if c.is_pc]
    foes = [c for c in state.combatants.values() if not c.is_pc and c.hp > 0]

    def spot(near, want):
        """The nearest free square to a room centre."""
        cx, cy = near
        for radius in range(0, 8):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    if free(cx + dx, cy + dy):
                        return cx + dx, cy + dy
        return None

    out = []
    party_room = room_centre(rects[0]) if rects else (1, 1)
    for c in pcs:
        p = tuple(c.pos) if c.pos and free(*c.pos) else spot(party_room, c)
        if p:
            taken.add(p); c.pos = list(p)
            out.append({"pos": p, "label": c.name[:1].upper(), "is_pc": True, "name": c.name})
    for i, c in enumerate(foes):
        home = (room_centre(rects[(i % max(1, len(rects) - 1)) + 1]) if len(rects) > 1
                else party_room)
        p = tuple(c.pos) if c.pos and free(*c.pos) else spot(home, c)
        if p:
            taken.add(p); c.pos = list(p)
            out.append({"pos": p, "label": c.name[:1].upper(), "is_pc": False, "name": c.name})
    return out


def map_title(location: str) -> str:
    """The DM passes whatever it was describing ('the ruined watchtower at Thornwick,
    fog-choked, three wolves and Oktah'); a map wants a name, not a sentence."""
    t = re.split(r"[,;(]|\bwith\b|\bwhere\b", str(location).strip(), 1)[0].strip(" .-")
    t = re.sub(r"\s+", " ", t)
    if len(t) > 46:
        t = t[:46].rsplit(" ", 1)[0] + "…"
    return t.title() if t.islower() else t


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _md_to_html(md: str) -> str:
    html = []
    for line in md.splitlines():
        if line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.strip().startswith(("- ", "* ")):
            html.append(f"<li>{line.strip()[2:]}</li>")
        elif line.strip():
            html.append(f"<p>{line}</p>")
    return "\n".join(html)


def render_prep_html(state, maps: dict[str, str]) -> Path:
    """maps: {location_name: grid_string}. Writes a printable HTML pack; returns its path."""
    import sheet
    PREP.mkdir(parents=True, exist_ok=True)
    roster = "".join(f"<li><b>{r['character']}</b> — {r['player']}"
                     + (f" · DDB {r['ddb_id']}" if r.get('ddb_id') else "") + "</li>"
                     for r in state.roster) or "<li>(no party set)</li>"
    pcs = [c for c in state.combatants.values() if c.is_pc]
    sheets_html = ("".join(sheet.render(c, state.campaign) for c in pcs)
                   if pcs else "<p>(no character sheets — import from D&amp;D Beyond first)</p>")
    map_html = "".join(
        f'<div class="map"><h3>{loc}</h3>{grid_to_svg(g, title=loc)}</div>' for loc, g in maps.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{state.campaign} — prep</title>
<style>
 body{{font-family:Georgia,serif;max-width:820px;margin:24px auto;color:#1a1a1a;line-height:1.4}}
 h1{{border-bottom:2px solid #8a6d3b}} h2{{color:#6b4f1d;margin-top:1.4em}}
 .map{{page-break-inside:avoid;margin:16px 0}} .map svg{{border:1px solid #999;max-width:100%}}
 @media print{{.map{{page-break-before:always}} body{{margin:0}}}}
{sheet.CSS}
</style></head><body>
<h1>{state.campaign}</h1><p><i>{state.premise}</i></p>
<h2>Party</h2><ul>{roster}</ul>
{_md_to_html(state.plan)}
<h2>Battle Maps</h2>{map_html or '<p>(none)</p>'}
<h2>Character Sheets</h2>{sheets_html}
</body></html>"""
    path = PREP / f"{state.campaign}.html"
    path.write_text(doc, encoding="utf-8")
    return path
