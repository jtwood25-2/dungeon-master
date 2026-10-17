"""Battle maps as geometry, drawn deterministically from what kind of place it is.

Two things were wrong with the map and they compounded.

THE GRID WAS THE CEILING. Everything was one character per five-foot square, rendered as one
30-pixel box per character. A shoreline could only ever be a staircase of squares; a round
market could only be a rectangle; a ship's hull could not be curved at all. No amount of better
content fixes a representation that cannot express a diagonal.

THE MODEL WAS THE WRONG TOOL. Asking one to list features gave a different answer every run --
Sixpence Reach drew a quay with a gibbet and no water on it, having drawn perfectly well an
hour earlier -- because nothing forced the answer to contain the things the place is made of.
A quay always has water. A vault always has shelving. That is not a judgement call, it is what
the words mean, and a lookup does it the same way every time.

So: the place is classified by name into an ARCHETYPE, and each archetype lays itself out as
real geometry -- polygons, arcs, a curved shoreline -- seeded by the location so the same place
is the same map forever. Rendering is proper vector art over that geometry: water with depth,
stone with joints, timber with planks, buildings with a cast shadow.

The five-foot grid does not disappear; it stops being the drawing. It is overlaid for measuring
and token placement, which is what it was always for, and walkability is computed from the
polygons underneath -- so "can I get there" still has a real answer.

Nothing here calls a model. It is instant, free and identical on every run.
"""
import hashlib
import math
import re

# Feet per grid square, the D&D convention, and what the scale bar and coordinates mean.
FOOT = 5


def seed_of(name: str) -> int:
    return int(hashlib.sha1((name or "map").encode("utf-8")).hexdigest()[:8], 16)


class R:
    """Small deterministic RNG. random.Random would do, but this keeps a map stable across
    Python versions -- a party returning to a place must find the same place."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF or 1

    def next(self) -> float:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF

    def rng(self, a, b):
        return a + (b - a) * self.next()

    def pick(self, xs):
        return xs[int(self.next() * len(xs)) % len(xs)]

    def chance(self, p) -> bool:
        return self.next() < p


# ---------------------------------------------------------------- what kind of place is this
ARCHETYPES = {
    "quay": ("quay", "dock", "wharf", "pier", "harbour", "harbor", "reach", "jetty", "landing"),
    "market": ("market", "bazaar", "square", "plaza", "fair", "stalls"),
    "vault": ("vault", "registry", "archive", "library", "counting", "warehouse", "store"),
    "reef": ("reef", "shoal", "tally", "atoll", "sandbar", "shallows"),
    "cavern": ("cave", "cavern", "grotto", "mine", "shaft", "tunnel", "undercroft", "crypt"),
    "hall": ("hall", "keep", "manor", "temple", "chapel", "fort", "hold", "vestry"),
    # Towers breaking the surface of a flooded city are islands, not rooms. Classifying them as
    # a hall drew Halvenny's drowned steeples as a keep with corridors in it.
    "spires": ("steeple", "spire", "tower", "belfry", "roofs", "rooftops"),
    "ship": ("ship", "deck", "galley", "hulk", "wreck", "barge", "boat"),
    "street": ("street", "town", "village", "lane", "row", "quarter", "yard", "lampyard"),
    "grove": ("grove", "wood", "forest", "glade", "spring", "thicket"),
}


def archetype(location: str) -> str:
    """Deterministic. The words in a place's name say what it is made of."""
    t = (location or "").lower()
    best, best_at = "hall", 10 ** 9
    for kind, words in ARCHETYPES.items():
        for wd in words:
            i = t.find(wd)
            if i >= 0 and i < best_at:
                best, best_at = kind, i
    return best


# ---------------------------------------------------------------- the scene
class Scene:
    """Geometry in grid units. The renderer knows nothing about what a place is."""

    def __init__(self, w, h, kind, title):
        self.w, self.h, self.kind, self.title = w, h, kind, title
        self.water = []        # polygons
        self.land = []         # polygons drawn as walkable ground
        self.solid = []        # polygons that block: buildings, rock
        self.timber = []       # decking / boardwalk polygons
        self.props = []        # (x, y, kind) small things
        self.labels = []       # (x, y, text)
        self.edges = []        # (x1, y1, x2, y2, kind) railings, rope lines


def _poly_wave(x0, y0, x1, y1, amp, rng, steps=14, vertical=False):
    """A wavy line between two points — a shoreline, not a staircase of squares."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        if vertical:
            y = y0 + (y1 - y0) * t
            x = x0 + (x1 - x0) * t + math.sin(t * math.pi * 2.2 + rng.next() * 0.4) * amp
        else:
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t + math.sin(t * math.pi * 2.2 + rng.next() * 0.4) * amp
        pts.append((round(x, 2), round(y, 2)))
    return pts


def _rect(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _blob(cx, cy, r, rng, n=12):
    """An organic outline — a reef, a rock, a stand of trees."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * rng.rng(0.62, 1.25)
        pts.append((round(cx + math.cos(a) * rr, 2), round(cy + math.sin(a) * rr, 2)))
    return pts


# ---------------------------------------------------------------- archetype layouts
def _quay(s: Scene, rng: R):
    """Stone running along the water, buildings inland, the sea taking one whole side."""
    shore = s.w * rng.rng(0.56, 0.66)
    top = _poly_wave(shore, 0, shore + rng.rng(-1.5, 1.5), s.h, 1.1, rng, vertical=True)
    s.water.append(top + [(s.w, s.h), (s.w, 0)])
    s.land.append([(0, 0), *top, (0, s.h)][::1])
    # the dock itself: timber decking along the whole shoreline
    deck = [(x - rng.rng(2.4, 3.2), y) for x, y in top]
    s.timber.append(top + deck[::-1])
    # bollards down the edge, and crates behind them
    n = int(s.h // 2.4)
    for i in range(n):
        y = 1.2 + i * (s.h - 2.4) / max(1, n - 1)
        x = shore - 1.0 + math.sin(y * 0.7) * 0.8
        s.props.append((round(x, 2), round(y, 2), "bollard"))
        if rng.chance(0.4):
            s.props.append((round(x - rng.rng(1.6, 3.2), 2), round(y + rng.rng(-.6, .6), 2), "crate"))
    # warehouses inland, with a gap between them to fight in
    bx = 0.8
    while bx < shore - 7:
        bw, bh = rng.rng(3.5, 5.5), rng.rng(3.0, 4.6)
        by = rng.rng(0.8, max(1.0, s.h - bh - 1))
        s.solid.append(_rect(round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)))
        bx += bw + rng.rng(2.6, 4.2)
    if rng.chance(0.8):
        s.props.append((round(shore - 1.5, 2), round(s.h * rng.rng(0.15, 0.85), 2), "gibbet"))
    s.edges.append((shore, 0, shore, s.h, "rail"))


def _market(s: Scene, rng: R):
    """An open plaza with stalls in rows, an arcade round the edge, something in the middle."""
    s.land.append(_rect(0, 0, s.w, s.h))
    # arcade: a colonnade, not a wall — you can walk between the pillars
    for i in range(int(s.w // 2.2)):
        x = 1.2 + i * 2.2
        s.props.append((round(x, 2), 1.0, "pillar"))
        s.props.append((round(x, 2), round(s.h - 1.0, 2), "pillar"))
    cx, cy = s.w / 2, s.h / 2
    s.props.append((round(cx, 2), round(cy, 2), "bell"))
    s.labels.append((cx, cy - 1.4, "the bell"))
    # stalls in two banks, with a clear aisle down the middle
    for side in (-1, 1):
        rows = int(rng.rng(2, 4))
        for r0 in range(rows):
            y = cy + side * (2.4 + r0 * 2.2)
            if not (1.5 < y < s.h - 1.5):
                continue
            x = rng.rng(2.0, 3.4)
            while x < s.w - 3.5:
                w = rng.rng(1.6, 2.6)
                s.props.append((round(x + w / 2, 2), round(y, 2), "stall"))
                x += w + rng.rng(1.4, 2.4)


def _vault(s: Scene, rng: R):
    """Long parallel runs of shelving with aisles between them. A corridor, not a cave."""
    s.land.append(_rect(0, 0, s.w, s.h))
    x = rng.rng(2.0, 3.0)
    while x < s.w - 3.0:
        run_w = rng.rng(0.9, 1.3)
        y0 = rng.rng(0.9, 2.2)
        y1 = s.h - rng.rng(0.9, 2.2)
        # a gap partway down so the aisles connect: shelving with a break in it
        gap = rng.rng(y0 + 2, max(y0 + 2.5, y1 - 2))
        s.solid.append(_rect(round(x, 2), round(y0, 2), round(run_w, 2), round(gap - y0 - 0.9, 2)))
        s.solid.append(_rect(round(x, 2), round(gap + 0.9, 2), round(run_w, 2), round(y1 - gap - 0.9, 2)))
        x += run_w + rng.rng(2.0, 2.8)
    s.props.append((round(s.w - 2.0, 2), round(s.h / 2, 2), "desk"))
    s.labels.append((s.w - 2.0, s.h / 2 - 1.3, "the counting desk"))


def _reef(s: Scene, rng: R):
    """Water with land breaking through it, and channels between the banks.

    The first version was 9% walkable, which is a drowning rather than an encounter. A reef
    fight happens ON the exposed rock: the water is the hazard and the shape of the fight, not
    the whole board. Banks are large and deliberately overlapping so there is somewhere to go.
    """
    s.water.append(_rect(0, 0, s.w, s.h))
    # a long spine of rock across the middle, which is what a reef ring actually is
    spine_y = s.h * rng.rng(0.42, 0.58)
    band = _poly_wave(0.5, spine_y - 2.2, s.w - 0.5, spine_y - 2.4, 1.3, rng)
    back = _poly_wave(s.w - 0.5, spine_y + 2.6, 0.5, spine_y + 2.4, 1.3, rng)
    s.land.append(band + back)
    for _ in range(int(rng.rng(3, 5))):
        cx, cy = rng.rng(3.0, s.w - 3.0), rng.rng(1.8, s.h - 1.8)
        s.land.append(_blob(round(cx, 2), round(cy, 2), rng.rng(2.4, 4.2), rng))
    for _ in range(int(rng.rng(3, 6))):
        s.props.append((round(rng.rng(1, s.w - 1), 2), round(rng.rng(1, s.h - 1), 2), "rock"))
    for _ in range(int(rng.rng(1, 3))):
        s.props.append((round(rng.rng(2, s.w - 2), 2), round(rng.rng(2, s.h - 2), 2), "wreck"))


def _cavern(s: Scene, rng: R):
    """Cellular automata, then drawn as one smooth outline instead of a staircase of squares."""
    W, H = int(s.w), int(s.h)
    grid = [[1 if rng.chance(0.44) else 0 for _ in range(W)] for _ in range(H)]
    for _ in range(4):
        nxt = [row[:] for row in grid]
        for y in range(H):
            for x in range(W):
                n = sum(grid[yy][xx]
                        for yy in range(max(0, y - 1), min(H, y + 2))
                        for xx in range(max(0, x - 1), min(W, x + 2))
                        if (xx, yy) != (x, y))
                nxt[y][x] = 1 if (n > 4 or x in (0, W - 1) or y in (0, H - 1)) else 0
        grid = nxt
    s.land.append(_rect(0, 0, s.w, s.h))
    # rock as blobs over the open ground reads far better than square-by-square walls
    for y in range(H):
        x = 0
        while x < W:
            if grid[y][x]:
                run = 0
                while x + run < W and grid[y][x + run]:
                    run += 1
                if run >= 1:
                    s.solid.append(_rect(x - 0.1, y - 0.1, run + 0.2, 1.2))
                x += run
            x += 1
    for _ in range(int(rng.rng(2, 5))):
        s.props.append((round(rng.rng(2, s.w - 2), 2), round(rng.rng(2, s.h - 2), 2), "rubble"))


def _hall(s: Scene, rng: R):
    """Rooms by binary partition, with real doorways rather than gaps in a wall of characters."""
    s.land.append(_rect(0, 0, s.w, s.h))
    rooms = [(0.6, 0.6, s.w - 1.2, s.h - 1.2)]
    for _ in range(int(rng.rng(3, 5))):
        rooms.sort(key=lambda r: -r[2] * r[3])
        x, y, w, h = rooms.pop(0)
        if w < 6 and h < 6:
            rooms.append((x, y, w, h)); break
        if w > h:
            cut = rng.rng(0.38, 0.62) * w
            rooms += [(x, y, cut - 0.35, h), (x + cut + 0.35, y, w - cut - 0.35, h)]
            s.solid.append(_rect(round(x + cut - 0.35, 2), round(y, 2), 0.7, round(h, 2)))
            dy = y + rng.rng(0.25, 0.75) * h
            s.props.append((round(x + cut, 2), round(dy, 2), "door"))
        else:
            cut = rng.rng(0.38, 0.62) * h
            rooms += [(x, y, w, cut - 0.35), (x, y + cut + 0.35, w, h - cut - 0.35)]
            s.solid.append(_rect(round(x, 2), round(y + cut - 0.35, 2), round(w, 2), 0.7))
            dx = x + rng.rng(0.25, 0.75) * w
            s.props.append((round(dx, 2), round(y + cut, 2), "door"))
    for (x, y, w, h) in rooms:
        if w > 2 and h > 2 and rng.chance(0.6):
            s.props.append((round(x + w / 2, 2), round(y + h / 2, 2), rng.pick(["table", "brazier", "crate"])))


def _ship(s: Scene, rng: R):
    """A hull, which is the shape the old grid could not draw at all."""
    s.water.append(_rect(0, 0, s.w, s.h))
    cx, cy = s.w / 2, s.h / 2
    L, B = s.w * 0.78, s.h * 0.52
    hull = []
    for i in range(28):
        t = i / 27
        a = math.pi * t
        x = cx - L / 2 + L * t
        bulge = math.sin(a) ** 0.7
        hull.append((round(x, 2), round(cy - B / 2 * bulge, 2)))
    for i in range(28):
        t = 1 - i / 27
        a = math.pi * t
        x = cx - L / 2 + L * t
        bulge = math.sin(a) ** 0.7
        hull.append((round(x, 2), round(cy + B / 2 * bulge, 2)))
    s.land.append(hull)
    s.timber.append(hull)
    for f in (0.3, 0.62):
        s.props.append((round(cx - L / 2 + L * f, 2), round(cy, 2), "mast"))
    s.props.append((round(cx + L * 0.32, 2), round(cy, 2), "wheel"))
    for _ in range(int(rng.rng(3, 6))):
        s.props.append((round(rng.rng(cx - L / 3, cx + L / 3), 2),
                        round(rng.rng(cy - B / 4, cy + B / 4), 2), "crate"))


def _street(s: Scene, rng: R):
    """Blocks of building with streets between them; the street is the battlefield."""
    s.land.append(_rect(0, 0, s.w, s.h))
    y = rng.rng(1.0, 2.0)
    while y < s.h - 4:
        x = rng.rng(0.8, 2.2)
        bh = rng.rng(2.6, 4.2)
        while x < s.w - 3:
            bw = rng.rng(2.6, 5.0)
            s.solid.append(_rect(round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)))
            if rng.chance(0.45):
                s.props.append((round(x + bw / 2, 2), round(y + bh + 0.7, 2), "door"))
            x += bw + rng.rng(1.8, 3.4)
        y += bh + rng.rng(2.6, 3.8)
    for _ in range(int(rng.rng(2, 5))):
        s.props.append((round(rng.rng(1, s.w - 1), 2), round(rng.rng(1, s.h - 1), 2),
                        rng.pick(["barrel", "cart", "brazier"])))


def _grove(s: Scene, rng: R):
    """Open ground, a spring, and stands of trees you can break line of sight behind."""
    s.land.append(_rect(0, 0, s.w, s.h))
    cx, cy = s.w * rng.rng(0.35, 0.65), s.h * rng.rng(0.35, 0.65)
    s.water.append(_blob(round(cx, 2), round(cy, 2), rng.rng(1.6, 2.6), rng, n=14))
    s.labels.append((cx, cy - 2.4, "the spring"))
    for _ in range(int(rng.rng(7, 13))):
        x, y = rng.rng(1, s.w - 1), rng.rng(1, s.h - 1)
        if (x - cx) ** 2 + (y - cy) ** 2 > 12:
            s.props.append((round(x, 2), round(y, 2), "tree"))
    for _ in range(int(rng.rng(2, 5))):
        s.props.append((round(rng.rng(1, s.w - 1), 2), round(rng.rng(1, s.h - 1), 2), "stone"))



def _spires(s: Scene, rng: R):
    """Tower tops standing out of flood water: separate platforms, linked by planks."""
    s.water.append(_rect(0, 0, s.w, s.h))
    # Roofs, not pinnacles. The first pass gave 11% walkable, which is a swim rather than a
    # fight -- a rooftop battle needs roof to stand on, and the water is the drop off the edge.
    tops = []
    for _ in range(int(rng.rng(4, 6))):
        cx, cy = rng.rng(3.5, s.w - 3.5), rng.rng(3.0, s.h - 3.0)
        r = rng.rng(3.4, 5.0)
        s.land.append(_blob(round(cx, 2), round(cy, 2), r, rng, n=9))
        # the spire itself blocks, but it is a spire and not a blockhouse
        s.solid.append(_rect(round(cx - 0.45, 2), round(cy - 0.45, 2), 0.9, 0.9))
        s.props.append((round(cx, 2), round(cy, 2), "bell"))
        tops.append((cx, cy))
    # gangplanks between the nearest pairs, or the party cannot cross at all
    for i in range(len(tops) - 1):
        (x0, y0), (x1, y1) = tops[i], tops[i + 1]
        s.timber.append([(x0, y0 - 0.45), (x1, y1 - 0.45), (x1, y1 + 0.45), (x0, y0 + 0.45)])
    for _ in range(int(rng.rng(2, 4))):
        s.props.append((round(rng.rng(2, s.w - 2), 2), round(rng.rng(2, s.h - 2), 2), "rubble"))

LAYOUTS = {"quay": _quay, "market": _market, "vault": _vault, "reef": _reef,
           "cavern": _cavern, "hall": _hall, "ship": _ship, "street": _street,
           "grove": _grove, "spires": _spires}


def build(location: str, title: str = "", w: int = 26, h: int = 18) -> Scene:
    """The scene for this place. Same input, same map, forever."""
    kind = archetype(location)
    s = Scene(w, h, kind, title or location)
    LAYOUTS[kind](s, R(seed_of(location)))
    return s
