"""Render a Scene as a battle map somebody would actually want on the table.

The old renderer drew one 30-pixel box per character of an ASCII grid, so every edge in the
world was a staircase and every surface was a flat fill. This draws the geometry instead:
water with depth and a wave line along the shore, stone with joints, timber with planks,
buildings with a cast shadow, trees and stalls as marks rather than letters.

The five-foot grid is still here and still matters -- it is how "the ghoul is at F7" works by
voice -- but it sits OVER the art as a light measuring overlay, which is what a grid on a
printed battle map has always been.

Everything is plain SVG with no external anything: the phone renders it inline, and a map that
needs a font or a library to load is a map that is blank when the tunnel is slow.
"""
import math

CELL = 34          # pixels per five-foot square
PAD = 30


def _pal(dark: bool) -> dict:
    """Two palettes, because the table reads this on seven phones in whatever theme they run."""
    if dark:
        return dict(paper="#101a24", ink="#dbe6f0", soft="#8ba3b8", rule="#27384a",
                    water="#16394f", water2="#1d4c66", foam="#5f95b0",
                    ground="#1a2733", ground2="#223140", stone="#30465a",
                    timber="#3a3026", timber2="#4a3d2f", solid="#2b3a49",
                    solidEdge="#42596f", shadow="#0a1118", prop="#c9b48a", accent="#7fb2d9")
    return dict(paper="#f4f1e9", ink="#1d2b38", soft="#63788a", rule="#c9d3db",
                water="#bcd8e6", water2="#9cc4d8", foam="#ffffff",
                ground="#e8e2d2", ground2="#ded6c2", stone="#cdd5da",
                timber="#cbab7e", timber2="#b8945f", solid="#b9c2c9",
                solidEdge="#8b99a4", shadow="#00000022", prop="#6b5a3e", accent="#2d6f9e")


def _pts(poly, ox, oy):
    return " ".join(f"{ox + x * CELL:.1f},{oy + y * CELL:.1f}" for x, y in poly)


def _smooth(poly, ox, oy, close=True):
    """A path through the points with rounded corners — what stops a shoreline looking sawn."""
    if len(poly) < 3:
        return ""
    p = [(ox + x * CELL, oy + y * CELL) for x, y in poly]
    d = [f"M{p[0][0]:.1f},{p[0][1]:.1f}"]
    for i in range(1, len(p)):
        x0, y0 = p[i - 1]
        x1, y1 = p[i]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        d.append(f"Q{x0:.1f},{y0:.1f} {mx:.1f},{my:.1f}")
    if close:
        d.append("Z")
    return " ".join(d)


def _prop(kind, px, py, c, out):
    """One mark per thing. Drawn, not lettered — a stall should look like a stall."""
    r = CELL * 0.32
    ink, fill = c["prop"], c["ground2"]
    if kind in ("bollard", "pillar", "post", "mast", "stone"):
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r*0.62:.1f}" fill="{c["stone"]}" '
                   f'stroke="{ink}" stroke-width="1.4"/>')
    elif kind in ("crate", "barrel", "cargo"):
        out.append(f'<rect x="{px-r*0.7:.1f}" y="{py-r*0.7:.1f}" width="{r*1.4:.1f}" '
                   f'height="{r*1.4:.1f}" fill="{c["timber"]}" stroke="{ink}" stroke-width="1.2" rx="2"/>')
        out.append(f'<line x1="{px-r*0.7:.1f}" y1="{py:.1f}" x2="{px+r*0.7:.1f}" y2="{py:.1f}" '
                   f'stroke="{ink}" stroke-width="0.8" opacity=".6"/>')
    elif kind in ("stall", "tent"):
        out.append(f'<path d="M{px-r:.1f},{py+r*0.6:.1f} L{px:.1f},{py-r:.1f} '
                   f'L{px+r:.1f},{py+r*0.6:.1f} Z" fill="{c["timber2"]}" stroke="{ink}" stroke-width="1.2"/>')
    elif kind in ("table", "desk"):
        out.append(f'<rect x="{px-r:.1f}" y="{py-r*0.55:.1f}" width="{r*2:.1f}" '
                   f'height="{r*1.1:.1f}" rx="3" fill="{c["timber"]}" stroke="{ink}" stroke-width="1.2"/>')
    elif kind == "door":
        out.append(f'<rect x="{px-r*0.75:.1f}" y="{py-r*0.28:.1f}" width="{r*1.5:.1f}" '
                   f'height="{r*0.56:.1f}" rx="2" fill="{c["accent"]}" stroke="{ink}" stroke-width="1.2"/>')
    elif kind == "tree":
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r*0.95:.1f}" fill="{c["ground2"]}" '
                   f'stroke="{ink}" stroke-width="1.2" opacity=".95"/>')
        for a in range(6):
            ang = a * math.pi / 3
            out.append(f'<line x1="{px:.1f}" y1="{py:.1f}" '
                       f'x2="{px+math.cos(ang)*r*0.9:.1f}" y2="{py+math.sin(ang)*r*0.9:.1f}" '
                       f'stroke="{ink}" stroke-width="0.7" opacity=".45"/>')
    elif kind in ("rock", "rubble", "wreck"):
        out.append(f'<path d="M{px-r:.1f},{py+r*0.5:.1f} L{px-r*0.4:.1f},{py-r*0.7:.1f} '
                   f'L{px+r*0.5:.1f},{py-r*0.3:.1f} L{px+r*0.8:.1f},{py+r*0.6:.1f} Z" '
                   f'fill="{c["stone"]}" stroke="{ink}" stroke-width="1.1"/>')
    elif kind in ("bell", "brazier", "wheel"):
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r*0.85:.1f}" fill="none" '
                   f'stroke="{c["accent"]}" stroke-width="2"/>')
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r*0.3:.1f}" fill="{c["accent"]}"/>')
    elif kind == "gibbet":
        out.append(f'<path d="M{px:.1f},{py+r:.1f} L{px:.1f},{py-r*1.3:.1f} L{px+r*0.9:.1f},{py-r*1.3:.1f}" '
                   f'fill="none" stroke="{ink}" stroke-width="2"/>')
        out.append(f'<line x1="{px+r*0.9:.1f}" y1="{py-r*1.3:.1f}" x2="{px+r*0.9:.1f}" '
                   f'y2="{py-r*0.4:.1f}" stroke="{ink}" stroke-width="1.2"/>')
    elif kind == "cart":
        out.append(f'<rect x="{px-r:.1f}" y="{py-r*0.5:.1f}" width="{r*2:.1f}" height="{r:.1f}" '
                   f'fill="{c["timber"]}" stroke="{ink}" stroke-width="1.2" rx="2"/>')
        out.append(f'<circle cx="{px-r*0.5:.1f}" cy="{py+r*0.6:.1f}" r="{r*0.3:.1f}" fill="none" stroke="{ink}" stroke-width="1.2"/>')
    else:
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r*0.5:.1f}" fill="{fill}" stroke="{ink}" stroke-width="1.1"/>')


def render(scene, tokens=None, dark: bool = False, note: str = "") -> str:
    """The whole map. `tokens` is [(x, y, label, is_pc)] in grid units."""
    c = _pal(dark)
    W = int(scene.w * CELL + PAD * 2)
    H = int(scene.h * CELL + PAD * 2 + 46)
    ox, oy = PAD, PAD + 26
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" role="img" aria-label="battle map of {scene.title}">']
    o.append(
        '<defs>'
        f'<linearGradient id="wg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{c["water"]}"/><stop offset="1" stop-color="{c["water2"]}"/>'
        '</linearGradient>'
        '<filter id="sh" x="-30%" y="-30%" width="170%" height="170%">'
        '<feDropShadow dx="2.5" dy="3" stdDeviation="2.2" flood-opacity="0.28"/></filter>'
        f'<pattern id="plank" width="{CELL}" height="{CELL*0.5:.0f}" patternUnits="userSpaceOnUse">'
        f'<rect width="{CELL}" height="{CELL*0.5:.0f}" fill="{c["timber"]}"/>'
        f'<line x1="0" y1="0" x2="{CELL}" y2="0" stroke="{c["timber2"]}" stroke-width="1.2"/>'
        '</pattern>'
        '</defs>')
    o.append(f'<rect width="{W}" height="{H}" fill="{c["paper"]}"/>')

    # ---- ground, then water, then the built things on top
    for poly in scene.land:
        o.append(f'<path d="{_smooth(poly, ox, oy)}" fill="{c["ground"]}"/>')
    for poly in scene.water:
        o.append(f'<path d="{_smooth(poly, ox, oy)}" fill="url(#wg)"/>')
        o.append(f'<path d="{_smooth(poly, ox, oy)}" fill="none" stroke="{c["foam"]}" '
                 f'stroke-width="1.6" opacity=".55"/>')
    for poly in scene.timber:
        o.append(f'<path d="{_smooth(poly, ox, oy)}" fill="url(#plank)" opacity=".9"/>')
    for poly in scene.solid:
        pts = _pts(poly, ox, oy)
        o.append(f'<polygon points="{pts}" fill="{c["solid"]}" stroke="{c["solidEdge"]}" '
                 f'stroke-width="1.6" filter="url(#sh)"/>')

    # ---- the measuring grid, over the art and deliberately quiet
    for i in range(int(scene.w) + 1):
        x = ox + i * CELL
        o.append(f'<line x1="{x:.0f}" y1="{oy}" x2="{x:.0f}" y2="{oy + scene.h*CELL:.0f}" '
                 f'stroke="{c["ink"]}" stroke-width="0.5" opacity=".13"/>')
    for j in range(int(scene.h) + 1):
        y = oy + j * CELL
        o.append(f'<line x1="{ox}" y1="{y:.0f}" x2="{ox + scene.w*CELL:.0f}" y2="{y:.0f}" '
                 f'stroke="{c["ink"]}" stroke-width="0.5" opacity=".13"/>')

    # ---- coordinates: how a table says where something is, out loud
    for i in range(int(scene.w)):
        lab = chr(65 + i) if i < 26 else chr(65 + i // 26 - 1) + chr(65 + i % 26)
        o.append(f'<text x="{ox + i*CELL + CELL/2:.0f}" y="{oy - 7}" font-size="10" '
                 f'fill="{c["soft"]}" text-anchor="middle" font-family="ui-monospace,monospace">{lab}</text>')
    for j in range(int(scene.h)):
        o.append(f'<text x="{ox - 8}" y="{oy + j*CELL + CELL/2 + 3.5:.0f}" font-size="10" '
                 f'fill="{c["soft"]}" text-anchor="end" font-family="ui-monospace,monospace">{j+1}</text>')

    for x, y, text in scene.labels:
        o.append(f'<text x="{ox + x*CELL:.0f}" y="{oy + y*CELL:.0f}" font-size="12" '
                 f'fill="{c["soft"]}" text-anchor="middle" font-style="italic" '
                 f'font-family="Georgia,serif">{_esc(text)}</text>')
    for kind_ in scene.props:
        x, y, kind = kind_
        _prop(kind, ox + x * CELL, oy + y * CELL, c, o)

    # ---- tokens last: they must never be hidden by scenery
    for (tx, ty, label, is_pc) in (tokens or []):
        px, py = ox + tx * CELL, oy + ty * CELL
        fill = c["accent"] if is_pc else "#a8412f"
        o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{CELL*0.38:.1f}" fill="{fill}" '
                 f'stroke="{c["paper"]}" stroke-width="2.5" filter="url(#sh)"/>')
        o.append(f'<text x="{px:.1f}" y="{py+4:.1f}" font-size="13" font-weight="700" '
                 f'fill="{c["paper"]}" text-anchor="middle" '
                 f'font-family="ui-sans-serif,system-ui">{_esc((label or "?")[:2])}</text>')

    o.append(f'<text x="{PAD}" y="{PAD}" font-size="19" fill="{c["ink"]}" '
             f'font-family="Georgia,serif" font-weight="600">{_esc(scene.title)}</text>')
    sub = note or f"{scene.kind} · {int(scene.w)}×{int(scene.h)} squares · {FOOT_LABEL}"
    o.append(f'<text x="{W - PAD}" y="{PAD}" font-size="11" fill="{c["soft"]}" '
             f'text-anchor="end" font-family="ui-monospace,monospace">{_esc(sub)}</text>')
    # scale bar: four squares, so twenty feet is a thing you can see
    by = H - PAD + 6
    o.append(f'<line x1="{PAD}" y1="{by}" x2="{PAD + 4*CELL}" y2="{by}" stroke="{c["ink"]}" stroke-width="2"/>')
    for k in range(5):
        o.append(f'<line x1="{PAD + k*CELL}" y1="{by-4}" x2="{PAD + k*CELL}" y2="{by+4}" '
                 f'stroke="{c["ink"]}" stroke-width="2"/>')
    o.append(f'<text x="{PAD + 4*CELL + 10}" y="{by+4}" font-size="11" fill="{c["soft"]}" '
             f'font-family="ui-monospace,monospace">20 feet</text>')
    o.append("</svg>")
    return "".join(o)


FOOT_LABEL = "5 ft squares"


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def walkable(scene, x: float, y: float) -> bool:
    """Can something stand here? Computed from the polygons, so movement still has an answer."""
    if not (0 <= x < scene.w and 0 <= y < scene.h):
        return False
    for poly in scene.solid:
        if _inside(poly, x, y):
            return False
    for poly in scene.water:
        if _inside(poly, x, y) and not any(_inside(l, x, y) for l in scene.land):
            return False
    return True


def _inside(poly, x, y) -> bool:
    """Ray casting. Small and exact, and the reason walkability did not have to stay on a grid."""
    n, inside = len(poly), False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside
