"""How much worse is a local model than Sonnet at NARRATION, actually?

local_bench.py answered a narrow question -- can a local model repeat numbers the engine has
already settled -- and the answer was yes, perfectly. This asks the harder one: can it write
the prose. That decides how much more of the table can come off Sonnet.

Scored on the things that have actually gone wrong at this table, all of them measured rather
than judged:

  style      distance from the reference DM's prose, using the counts in tools/style.py --
             the parenthetical dash, "gone X", ominous negation, vague "something", similes
  length     words per reply; the reference table averages 41 and ours ran 59
  obedience  did it break a hard rule -- narrating the player's character, reading out an
             armour class, asking "what do you do?" every single time, markdown
  variety    does it open every reply the same way, which is what a small model does worst

Sonnet is measured through the SAME path the game uses, so the comparison is fair.

Usage
  python tools/narration_bench.py
  python tools/narration_bench.py --models qwen2.5:14b-instruct,phi4:14b --no-sonnet
"""
import argparse
import asyncio
import json
import re
import time
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))

# Ordinary table moments, not combat. Each is a thing a player actually said this month.
PROMPTS = [
    ("scene", "the Long Cut tunnel, timbered walls, water underfoot, three drowned dig-crew "
              "standing across the passage blocking the way down; they have spoken a warning "
              "and have not attacked",
     "What do I see ahead of me?"),
    ("npc", "the dig camp, abandoned mid-shift; Ivo Tanner, a young surveyor with bruised ribs, "
            "is sitting under a tarp and is frightened of you",
     "I ask Ivo what happened to the others."),
    ("action", "a spoil heap of loose shale beside a mine shaft, a broken pickaxe head jammed "
               "upright in it as a marker",
     "I start digging where the pickaxe head is."),
    ("travel", "the river road, half a day upstream of the grove, late afternoon",
     "We keep walking upriver until we find the dig site."),
    ("refusal", "a sealed iron door with no visible hinges or handle",
     "Can I just smash the door open with my staff?"),
    ("emotion", "the mine shaft mouth; the character's birthmark has gone hot for the first "
                "time in his life and he has never left home before",
     "I stop and put my hand over the mark."),
]

# Hard rules. Breaking one of these is a fault regardless of how nice the prose is.
BREAKS = {
    "narrated the player": re.compile(
        r"\byou (?:feel|decide|realise|realize|wonder|remember|know that|understand)\b", re.I),
    "read out a number": re.compile(r"\b(?:AC|DC)\s*\d+|\bd\d+\b|\bdifficulty class\b", re.I),
    "stock prompt": re.compile(r"what (?:do|will) you do\??\s*$", re.I),
    "markdown": re.compile(r"\*\*|^\s*[-*]\s|^#{1,3}\s|__", re.M),
    "stage direction": re.compile(r"\*[^*\n]{3,}\*"),
}


async def _ollama(model: str, scene: str, said: str) -> str:
    import httpx
    import backend
    payload = {"model": model, "stream": False, "keep_alive": "10m",
               "messages": [{"role": "system", "content": backend.STORY_SYS},
                            {"role": "user",
                             "content": f"--- where they are ---\n{scene}\n\n"
                                        f"The player said: {said}"}],
               "options": {"temperature": 0.6, "num_predict": 160}}
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post("http://127.0.0.1:11434/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


async def _sonnet(scene: str, said: str, be=None) -> str:
    """Through the real backend, so the comparison is against what the table actually hears."""
    import backend
    from game_state import GameState
    g = GameState("_narrbench")
    g.add("Oktah", max_hp=12, ac=15, is_pc=True, stats={"dex": 14})
    g.set_scene(scene, scene)
    be = be or backend.ClaudeCodeBackend()
    out = []
    async for s in be.stream_dm(said, g, None, [], combat_brief=""):
        out.append(s)
    return " ".join(out).strip()


async def _thinking_arms(a) -> dict:
    """Same model, same prompts, thinking on and off -- does turning it off cost prose quality?

    Turning off extended thinking made every latency statistic better, and the worst case
    better by 44%. That is only a good trade if the writing survives it, and "the samples read
    fine to me" is not a measurement -- so the two arms go through the identical path and get
    scored by the identical rules.
    """
    import backend
    import config
    out = {}
    for label, budget in (("thinking on", ""), ("thinking off", "0")):
        old = config.DM_THINKING_TOKENS
        config.DM_THINKING_TOKENS = budget
        be = backend.ClaudeCodeBackend()
        replies, first = [], []
        try:
            # Repeat the set rather than lengthen it: six replies could not separate a real
            # style difference from one bad sentence, and the fluff score is the number this
            # decision actually turns on.
            for _ in range(a.repeat):
                for _tag, scene, said in PROMPTS:
                    t = time.perf_counter()
                    replies.append(await _sonnet(scene, said, be))
                    first.append((time.perf_counter() - t) * 1000)
        except Exception as e:
            out[label] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            be._kill()
            config.DM_THINKING_TOKENS = old
        # A run that scored 24 empty replies as words_per_reply: 0 reported silence as a
        # measurement, and the zeros sat in a results table looking exactly like data.
        # Sustained benching is what caused it -- the subscription rate-limits and
        # stream_dm then yields nothing -- so this has to be loud, not averaged.
        said = [r for r in replies if r.strip()]
        if replies and len(said) < len(replies) * 0.8:
            out[label] = {"error": f"{len(replies) - len(said)} of {len(replies)} replies came back EMPTY -- rate limited? not scored"}
        elif said:
            out[label] = score(said)
            out[label]["replies"] = len(said)
            out[label]["whole_reply_ms_median"] = round(st.median(first))
            out[label]["_samples"] = [{"asked": p[2], "said": r[:240]}
                                      for p, r in zip(PROMPTS, replies)][:3]
    return out


def score(replies: list) -> dict:
    import style
    body = "\n".join(replies)
    m = style.measure(replies) or {}
    breaks = {}
    for name, pat in BREAKS.items():
        n = sum(1 for r in replies if pat.search(r))
        if n:
            breaks[name] = n
    # variety: how many distinct opening words across the set
    openers = [(r.split() or [""])[0].lower().strip(".,") for r in replies if r]
    return {
        "words_per_reply": round(st.mean([len(r.split()) for r in replies]) if replies else 0, 1),
        "words_per_sentence": m.get("words_per_sentence"),
        "em_dash_per_1k": m.get("em_dash_per_1k"),
        "fluff_per_1k": m.get("fluff_per_1k"),
        "vague_some_per_1k": m.get("vague_some_per_1k"),
        "simile_per_1k": m.get("simile_per_1k"),
        "gone_x_per_1k": m.get("gone_x_per_1k"),
        "distinct_openers": f"{len(set(openers))}/{len(openers)}",
        "rule_breaks": breaks or "none",
    }


# the reference, measured from the corpus by tools/style.py
REFERENCE = {"words_per_reply": 41, "words_per_sentence": 12.9, "em_dash_per_1k": 0.04,
             "fluff_per_1k": 6.3, "vague_some_per_1k": 1.0, "simile_per_1k": 1.4,
             "gone_x_per_1k": 0.1}


async def run(a) -> dict:
    out = {"reference_matt": REFERENCE, "models": {}, "samples": {}}
    if a.thinking:
        out["thinking"] = await _thinking_arms(a)
        return out
    for model in [x.strip() for x in a.models.split(",") if x.strip()]:
        replies = []
        for _tag, scene, said in PROMPTS:
            try:
                replies.append(await _ollama(model, scene, said))
            except Exception as e:
                out["models"][model] = {"error": str(e)}
                break
        if replies:
            out["models"][model] = score(replies)
            out["samples"][model] = [{"asked": p[2], "said": r[:220]}
                                     for p, r in zip(PROMPTS, replies)][:3]
    if not a.no_sonnet:
        replies = []
        for _tag, scene, said in PROMPTS:
            try:
                replies.append(await _sonnet(scene, said))
            except Exception as e:
                out["models"]["sonnet (claude code)"] = {"error": str(e)}
                break
        if replies:
            out["models"]["sonnet (claude code)"] = score(replies)
            out["samples"]["sonnet (claude code)"] = [
                {"asked": p[2], "said": r[:220]} for p, r in zip(PROMPTS, replies)][:3]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="qwen2.5:14b-instruct")
    p.add_argument("--repeat", type=int, default=1,
                   help="passes over the prompt set per arm (--thinking only)")
    p.add_argument("--thinking", action="store_true",
                   help="compare Sonnet with extended thinking on vs off, nothing else")
    p.add_argument("--no-sonnet", action="store_true")
    a = p.parse_args()
    print(json.dumps(asyncio.run(run(a)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
