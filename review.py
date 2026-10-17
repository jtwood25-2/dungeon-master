"""Have a stronger model read a session back and grade the DM.

  python review.py                 # newest transcript, newest session
  python review.py --all           # the whole file, every session
  python review.py --file "state/transcripts/Thornwick 2026-07-31.jsonl"
  python review.py --model sonnet  # default is opus: the reviewer should be smarter than the DM
  python review.py --feedback      # file each finding into the feedback queue

You cannot judge the DM from inside a session — the tells are patterns across a whole evening.
This hands the transcript to a bigger model with the DM's ACTUAL rules (imported, not
restated, so the review can never drift from what the DM was told) and asks for specific
quoted violations rather than vibes.
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))
import backend, config, feedback, transcript  # noqa: E402

RUBRIC = """You are reviewing a Dungeon Master's performance in a live VOICE game, so that its
instructions can be improved. Below are the exact rules it was given, then a transcript.

Judge ONLY against those rules. For each breach, quote the DM's own words. Be specific and
be hard to please — a review that finds nothing is useless, but do not invent problems.

Pay closest attention to the failures that ruin this particular game:
  1. Speaking or acting for a player character, or narrating them in the third person.
  2. Stating a monster's hit points, max HP or AC as numbers.
  3. Resolving an attack without asking for a separate damage roll.
  4. Commenting on the player's dice ("nice roll") or reciting their number back.
  5. Commentary, praise or filler openers ("alright", "okay", "nice idea").
  6. Answering a short mechanical question with narration instead of an answer.
  7. Volunteering things the players should have had to earn by asking, looking or rolling.
  8. Long opening sentences — the first line is what they wait for in silence.
  9. Narrating into a vacuum instead of ending with the ball in their court.
 10. Continuity slips: contradicting earlier facts, forgetting names or what just happened.

Reply in this exact shape, nothing else:

## Verdict
One paragraph: how it actually read as a session, and the single biggest thing to fix.

## Breaches
- **<rule>** — "<exact quote>" — why it's wrong, in one line.
(omit the section entirely if there are none)

## What worked
- <specific thing, with a short quote>

## Change the instructions
- <a concrete, actionable edit to the DM's rules — the wording to add or remove>
"""


async def run(prompt: str, model: str, timeout: float) -> str:
    proc = await asyncio.create_subprocess_exec(
        config.CLAUDE_BIN, "-p", "--model", model,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL)
    out, _ = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=timeout)
    return out.decode("utf-8", errors="replace").strip()


def file_findings(report: str, campaign: str) -> int:
    """Turn the review's suggested instruction changes into feedback items, so they land in
    the same queue as the human's notes rather than in a file nobody opens."""
    section = re.split(r"##\s*Change the instructions", report, flags=re.I)
    if len(section) < 2:
        return 0
    n = 0
    for line in section[1].splitlines():
        item = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if len(item) > 12:
            feedback.add(f"[review] {item}", campaign)
            n += 1
    return n


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file"); ap.add_argument("--model", default="opus")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--session", type=int); ap.add_argument("--timeout", type=float, default=420)
    a = ap.parse_args()

    path = Path(a.file) if a.file else (transcript.files() or [None])[0]
    if not path or not Path(path).exists():
        print("no transcripts yet — play a session first"); return 1
    entries = transcript.load(path)
    if not entries:
        print(f"{path} is empty"); return 1

    sessions = sorted({e.get("session", 0) for e in entries})
    pick = a.session if a.session is not None else (None if a.all else sessions[-1])
    script = transcript.as_script(entries, pick)
    turns = sum(1 for e in entries if e["role"] == "player"
                and (pick is None or e.get("session") == pick))
    if not turns:
        print("nothing the player said in that session"); return 1

    campaign_name = Path(path).stem.rsplit(" ", 1)[0]
    print(f"reviewing {path.name} — session {'all' if pick is None else pick}, "
          f"{turns} player turns, with {a.model}…")

    prompt = (RUBRIC + "\n\n=== THE DM'S RULES ===\n" + backend.DM_SYS
              + "\n\n=== TRANSCRIPT ===\n" + script)
    try:
        report = await run(prompt, a.model, a.timeout)
    except asyncio.TimeoutError:
        print(f"the reviewer timed out after {a.timeout}s"); return 1
    if not report:
        print("the reviewer returned nothing"); return 1

    print("\n" + report)
    out = Path("state/reviews"); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{Path(path).stem} s{pick if pick is not None else 'all'}.md"
    dest.write_text(report, encoding="utf-8")
    print(f"\nsaved -> {dest}")
    if a.feedback:
        print(f"filed {file_findings(report, campaign_name)} suggestion(s) into the feedback queue")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
