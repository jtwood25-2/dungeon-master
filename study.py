"""Learn DM craft from real transcripts, by measuring rather than by impression.

  python study.py                       # re-measure the cached transcripts, rewrite the guide
  python study.py --add <wiki-page>     # fetch another episode and include it

Writes reference/craft.md: the measured numbers plus the rules derived from them. backend.py
loads that file into the DM's instructions, so what the DM is told is traceable to evidence
and improves whenever another episode is added — rather than being whatever I remembered
about how the show sounds.

Only statistics and short illustrative phrases are kept; transcripts stay in state/ and are
never redistributed.
"""
import argparse
import json
import re
import statistics
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "state" / "transcripts-ref"
GUIDE = ROOT / "reference" / "craft.md"
API = "https://criticalrole.fandom.com/api.php"
SPEAKER = re.compile(r"^([A-Z][A-Z ]{1,12}):\s*(.+)$", re.M)
DM = "MATT"


def fetch(page: str) -> Path:
    url = (f"{API}?action=parse&page={urllib.parse.quote(page)}/Transcript"
           f"&format=json&prop=wikitext&formatversion=2")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (dungeon-master)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        wt = json.loads(r.read())["parse"]["wikitext"]
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"{page.replace('/', '_')}.txt"
    p.write_text(wt, encoding="utf-8")
    return p


def measure() -> dict:
    dm_lines, runs, episodes = [], [], 0
    for p in sorted(RAW.glob("*.txt")):
        rows = [(m.group(1).strip(), m.group(2).strip())
                for m in SPEAKER.finditer(p.read_text(encoding="utf-8"))]
        if not rows:
            continue
        episodes += 1
        dm_lines += [t for s, t in rows if s == DM]
        run = 0
        for s, _ in rows:
            if s == DM:
                if run:
                    runs.append(run)
                run = 0
            else:
                run += 1
    if not dm_lines:
        return {}
    w = [len(t.split()) for t in dm_lines]
    total_turns = len(dm_lines) + sum(runs)
    short = [t.lower().strip(" .!?\"") for t in dm_lines if len(t.split()) <= 3]
    blob = "\n".join(dm_lines)

    def count(pat):
        return len(re.findall(pat, blob, re.I))

    return {
        "episodes": episodes, "dm_lines": len(dm_lines),
        "median_words": int(statistics.median(w)),
        "pct_under_10": sum(1 for x in w if x <= 9) * 100 // len(w),
        "pct_under_4": sum(1 for x in w if x <= 3) * 100 // len(w),
        "pct_over_40": sum(1 for x in w if x > 40) * 100 // len(w),
        "dm_share": len(dm_lines) * 100 // max(total_turns, 1),
        "gap_p90": int(statistics.quantiles(runs, n=10)[-1]) if len(runs) > 10 else 0,
        "gap_max": max(runs) if runs else 0,
        "pct_gaps_5plus": sum(1 for r in runs if r >= 5) * 100 // max(len(runs), 1),
        "acks": [a for a, _ in Counter(short).most_common(10)],
        "what_do_you_do": count(r"what do you do\??"),
        "youre_up": count(r"you'?re up"),
        "go_ahead": count(r"go ahead"),
        "make_a_check": count(r"make (?:a|an) [a-z ]+(?:check|saving throw)"),
    }


def render(m: dict) -> str:
    return f"""# DM craft, measured

Derived by `study.py` from {m['episodes']} full transcripts of the show this DM is modelled
on — {m['dm_lines']} DM lines. These are counts, not opinions. Regenerate with `python study.py`.

## Talk less than feels natural
- The DM spoke **{m['dm_share']}% of the time**. The players had the other {100 - m['dm_share']}%.
- Median DM beat: **{m['median_words']} words**. {m['pct_under_10']}% were under 10 words;
  only {m['pct_over_40']}% ran past 40.
- **{m['pct_under_4']}% of DM lines were three words or fewer** — {", ".join(repr(a) for a in m['acks'][:6])}.
  A short acknowledgement keeps play moving without taking it over.

## Let the players talk to each other
- {m['pct_gaps_5plus']}% of the time the players exchanged **five or more turns with no DM at
  all**; the longest unbroken player stretch was **{m['gap_max']} turns**.
- Silence is a legitimate DM move. If a player is talking to another player, or thinking out
  loud, say nothing. Only step in when the WORLD needs to respond.

## Hand back with two words, not a question
- "What do you do?" appeared **{m['what_do_you_do']} times** across all {m['episodes']} episodes.
  "You're up" appeared {m['youre_up']} times and "Go ahead" {m['go_ahead']}.
- End on the thing that is happening, or on "You're up." Do not tack a question onto every beat.

## Name the check in full, always
- Every request was the complete phrase — "Make a Perception check", "Make a Constitution
  saving throw" ({m['make_a_check']} instances). Never a bare skill word.

## Drama comes from pacing, not effects
- The show has no sound effects. Tension is built with short beats, specific sensory detail and
  a willingness to stop talking. Save the long descriptive passage for arriving somewhere new;
  everything else is one or two sentences.

## When he DOES go long ({m['pct_over_40']}% of the time), here is the shape

Read from his actual long passages — the technique, not the content.

**Recaps are a journey, not a summary.** He moves them through geography, naming every place
out loud, then stops dead and hands over:
> "You made your way northward, past the Wuyun Gorge, back into the Dwendalian Empire. Passed
> through Trostenwald and Alfield and met up with Bryce for a bit to catch up, and then made
> your way toward Felderwin."
It ends "That is where we left off. So — what would you like to do?" Chain the places; don't
summarise events in the abstract.

**People are described outside-in, physically, with no interpretation.** Clothing first (colour,
material, how it moves), then one specific ornament, then hair, then an age estimate — never
"she seems dangerous":
> "An elven woman in fine flowing clothes of green and black... billowing sleeves. Jewelry
> around the fingers, a number of rings and a series of necklaces."
Give the players what a camera would see and let them draw the conclusion.

**Name everything, constantly.** Real proper nouns for places, factions, and people — the
Wuyun Gorge, the Cerberus Assembly, Martinet Lunidus Da'leth. A named thing feels like it
existed before the party arrived; "a gorge" and "a wizard" do not. Invent names freely and
reuse them.

**Always second person, always present in the scene.** "You discovered", "you came to
discover", "upon inspecting the vicinity" — the party is the camera; the world is what they
find.

**Long passages are for arrival and recap only.** Entering somewhere new, meeting someone new,
or opening a session. Everything in between is one or two sentences.

## Opening a campaign — two moves, in this order

**1. Set the world, wide to narrow, and date it.** He opens with an era, then zooms:
continent → region → city → who rules it. Every noun is a proper noun.
> "I would like to welcome you to Marquet. The year is 843 P.D., or Post Divergence... The
> epicenter of this ancient destruction created the Rumedam Desert... surrounding the central
> city of Ank'Harel, known as the Jewel of Hope. Here, the long-living ruler, J'mon Sa Ord,
> and their Hands of Ord keep the peace."
Invent: an era name and year, one continent-scale catastrophe that shaped the map, two or
three named regions, one central city with a nickname, and who holds power there. This is the
ONLY place a long monologue is welcome.

**2. Then cut hard to one small moving moment, and hand over.** He does not begin with the
party standing still, and he does not begin with a quest:
> "So... We begin with the rocking sensation of one of the various engineered cable cars...
> The small windows outside are giving a beautiful view of the jungle mists... To that point,
> Laura, if you'd like to describe your character."
Put them somewhere physical and already in motion, give two sensory details, then hand
straight to a player. Ask them to describe their character before anything happens to it.

Keep the two moves separate: the world, then the moment. Do not blend them.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[], help="wiki page name to fetch")
    a = ap.parse_args()
    for page in a.add:
        print(f"fetching {page} …")
        print("  ->", fetch(page))
    m = measure()
    if not m:
        print(f"no transcripts in {RAW} — add some with --add <Wiki_Page_Name>")
        return 1
    GUIDE.parent.mkdir(parents=True, exist_ok=True)
    GUIDE.write_text(render(m), encoding="utf-8")
    print(f"measured {m['dm_lines']} DM lines from {m['episodes']} episodes -> {GUIDE}")
    print(f"  median beat {m['median_words']} words | DM speaks {m['dm_share']}% | "
          f"'what do you do' x{m['what_do_you_do']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
