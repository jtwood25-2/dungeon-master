"""Does the director's background Opus work slow down the DM turn a player is waiting on?

It was named as a suspect on circumstantial evidence -- "director" appeared in the log near 26
of the 55 turns over eight seconds -- and that evidence is worthless on its own, because both
things cluster around the start of a new campaign. A new campaign fires session_start, which
runs three Opus specialists, AND builds a continuity block, AND has nothing cached. Correlation
there says nothing about causation.

The two are separate OS processes, so they cannot corrupt each other's pipe, and the DM never
waits on the director: pull_prepared() takes whatever is already finished and returns
immediately. So the only channels left are the machine and the account -- CPU, memory, and a
subscription both are spending against.

This measures it directly: identical DM turns with the director idle, then the same turns again
with real Opus specialists running alongside, interleaved so that any drift in the service hits
both arms equally.

Usage
  python tools/director_interference.py
  python tools/director_interference.py --turns 8 --load 3
"""
import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

QUESTIONS = ["I push the tent flap aside.", "I listen at the shaft mouth.",
             "I ask Ivo what he saw.", "I search the crates.", "I look at the spoil heap.",
             "I check the rope.", "I watch the treeline.", "I read the report."]


async def opus_load(n: int, stop: asyncio.Event) -> None:
    """Real director-shaped work: n Opus subprocesses, respawned until told to stop."""
    import config
    async def one():
        while not stop.is_set():
            p = await asyncio.create_subprocess_exec(
                config.CLAUDE_BIN, "-p", "--model", config.DIRECTOR_CLAUDE_MODEL,
                "Name three complications that could follow from a flooded mine shaft. "
                "Two sentences each.",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            try:
                await asyncio.wait_for(p.wait(), timeout=120)
            except asyncio.TimeoutError:
                p.kill()
    await asyncio.gather(*[one() for _ in range(n)])


async def measure(be, state, questions) -> list:
    import backend
    out = []
    for q in questions:
        t = time.perf_counter()
        first = None
        async for _s in be.stream_dm(q, state, None, []):
            if first is None:
                first = (time.perf_counter() - t) * 1000
        if first:
            out.append(first)
    return out


async def run(a) -> int:
    import backend
    import game_state
    state = game_state.GameState.load(a.campaign)
    qs = (QUESTIONS * 4)[:a.turns]
    be = backend.ClaudeCodeBackend()
    await be._ensure()

    quiet = await measure(be, state, qs)

    stop = asyncio.Event()
    load = asyncio.create_task(opus_load(a.load, stop))
    await asyncio.sleep(a.warm)                      # let the specialists actually be running
    busy = await measure(be, state, qs)
    stop.set()
    load.cancel()
    try:
        await load
    except (asyncio.CancelledError, Exception):
        pass
    be._kill()

    def stats(v):
        if not v:
            return {}
        s = sorted(v)
        return {"n": len(v), "median": round(st.median(v)), "mean": round(st.mean(v)),
                "p90": round(s[int(.9 * len(s)) - 1]), "max": round(max(v))}

    out = {"campaign": a.campaign, "director_processes": a.load,
           "director_idle": stats(quiet), "director_busy": stats(busy)}
    if quiet and busy:
        out["median_change_ms"] = round(st.median(busy) - st.median(quiet))
        out["verdict"] = ("the director slows the DM" if st.median(busy) > st.median(quiet) * 1.25
                          else "no measurable interference")
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", default="Beyond the Grove")
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--load", type=int, default=3, help="concurrent Opus specialists")
    p.add_argument("--warm", type=float, default=6.0, help="seconds before measuring under load")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
