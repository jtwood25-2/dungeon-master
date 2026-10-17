"""Dice engine. Parses standard RPG notation and returns a structured result.

Supported: `NdM`, flat modifiers, chained terms, keep-highest/lowest, advantage.
  1d20            2d6+3           1d20-1          2d6+1d4+2
  4d6kh3          (keep highest 3 -- ability scores)
  2d20kl1         (keep lowest 1 -- disadvantage)   1d20 adv / 1d20 dis (shorthand)
"""
import random
import re
from dataclasses import dataclass, field

_TERM = re.compile(r"([+-]?)\s*(\d*)d(\d+)(k[hl]\d+)?|([+-]?\s*\d+)", re.I)


@dataclass
class Roll:
    notation: str
    total: int
    dice: list[int] = field(default_factory=list)   # every die face rolled (incl. dropped)
    kept: list[int] = field(default_factory=list)    # faces that counted
    modifier: int = 0
    detail: str = ""

    def __str__(self):
        return f"{self.notation} = {self.total}  ({self.detail})"


def roll(notation: str, rng: random.Random | None = None) -> Roll:
    r = rng or random
    text = notation.strip().lower()
    # advantage/disadvantage shorthand on a bare d20
    if text.endswith(" adv"):
        text = "2d20kh1" + text[:-4].replace("1d20", "").strip()
    elif text.endswith(" dis"):
        text = "2d20kl1" + text[:-4].replace("1d20", "").strip()

    total, all_dice, all_kept, mod, parts = 0, [], [], 0, []
    for m in _TERM.finditer(text.replace(" ", "")):
        sign_d, n, faces, keep, flat = m.groups()
        if flat is not None:
            v = int(flat.replace(" ", "")); total += v; mod += v
            parts.append(f"{v:+d}")
            continue
        sign = -1 if sign_d == "-" else 1
        n = int(n) if n else 1
        faces = int(faces)
        if faces < 1 or n < 1 or n > 1000:
            continue
        faces_rolled = [r.randint(1, faces) for _ in range(n)]
        all_dice += faces_rolled
        kept = faces_rolled
        if keep:
            k = int(keep[2:]); hi = keep[1] == "h"
            kept = sorted(faces_rolled, reverse=hi)[:k]
        all_kept += kept
        sub = sign * sum(kept)
        total += sub
        drop = "" if kept == faces_rolled else f" keep {kept}"
        parts.append(f"{'-' if sign < 0 else ''}{n}d{faces}{faces_rolled}{drop}")
    return Roll(notation=notation, total=total, dice=all_dice, kept=all_kept,
                modifier=mod, detail=" ".join(parts) or "0")


def check(notation: str, dc: int, rng: random.Random | None = None) -> tuple[Roll, bool]:
    """Roll and compare to a difficulty class. Returns (roll, success)."""
    rr = roll(notation, rng)
    return rr, rr.total >= dc
