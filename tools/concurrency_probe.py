"""Make several players speak at once and measure what the table actually waits.

Every recorded turn over a minute has the same shape: three people speaking inside twenty
seconds during combat. They never ran at the same time -- the Claude Code backend is one
subprocess behind one lock, so they QUEUED, and the third player waited out the two turns in
front of theirs. Nothing in the test suite could see that, because the suite drives one player.

This opens a websocket per player, sends their lines within a few hundred milliseconds of each
other, and reports the two numbers that decide whether a table can be run this way:

  wait        seconds from speaking to the first word back, per player. The last player to be
              answered is the one who decides whether the format works.
  generations how many separate DM replies the utterances produced. Three players answered
              together is one; three answered in sequence is three, and three times the wait.

Usage
  python tools/concurrency_probe.py --players 3
  python tools/concurrency_probe.py --players 4 --gap 0.4 --campaign "probe"
"""
import argparse
import asyncio
import json
import statistics as st
import sys
import time

try:
    import websockets
except ImportError:
    print(json.dumps({"error": "pip install websockets"})); sys.exit(1)

LINES = [
    "I attack the closest one with my staff.",
    "I cast bless on the fighter.",
    "I move behind the pillar and ready my bow.",
    "I check the far door for traps.",
    "I shout for the others to fall back.",
]


async def player(url, campaign, name, line, gap, out, quiet):
    async with websockets.connect(url, max_size=None, ping_interval=None,
                                  open_timeout=30, close_timeout=5) as ws:
        await ws.send(json.dumps({"type": "campaign", "name": campaign}))
        await asyncio.sleep(1.0)                     # let the session open before anybody talks
        await asyncio.sleep(gap)                     # then stagger, so they overlap but differ
        t0 = time.perf_counter()
        out.setdefault("spoke_at", {})[name] = time.time()
        await ws.send(json.dumps({"type": "text", "text": line}))
        said, first, last = [], None, t0
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=quiet)
                if isinstance(raw, bytes):
                    continue                         # audio; the text is what we are scoring
                m = json.loads(raw)
                if m.get("type") == "dm" and m.get("text"):
                    if first is None:
                        first = time.perf_counter() - t0
                    said.append(m["text"]); last = time.perf_counter()
                elif m.get("type") == "status" and "mid-sentence" in (m.get("msg") or ""):
                    out.setdefault("queued", []).append(name)
        except (asyncio.TimeoutError, Exception):
            pass
        out["players"][name] = {"line": line, "wait_s": round(first, 2) if first else None,
                                "sentences": len(said), "said": " ".join(said)[:400]}


def ensure_campaign(base: str, name: str, premise: str) -> None:
    """The scratch campaign has to EXIST before anybody connects.

    Pointing this at a name that does not exist looked like it worked -- three players, one
    reply -- and the measurement was worthless: the server refuses to invent a campaign, so
    each connection took a slow error path and the three players ended up speaking 24 seconds
    apart. They never overlapped, so nothing was being measured but a queue that was never
    formed. A concurrency probe that cannot prove the utterances were concurrent proves nothing.
    """
    import urllib.request
    body = json.dumps({"name": name, "premise": premise}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/campaigns", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception:
        pass                                         # already there is the normal case


async def run(a):
    ensure_campaign(a.url, a.campaign,
                    "A scratch campaign used only by the concurrency probe. Three adventurers "
                    "stand in a lamplit stone corridor. Two ghouls block the way ahead.")
    out = {"players": {}, "queued": []}
    url = a.url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws"
    names = [f"P{i+1}" for i in range(a.players)]
    await asyncio.gather(*[
        player(url, a.campaign, n, LINES[i % len(LINES)], i * a.gap, out, a.quiet)
        for i, n in enumerate(names)])

    waits = [p["wait_s"] for p in out["players"].values() if p["wait_s"]]
    answered = [n for n, p in out["players"].items() if p["sentences"]]
    # Every player whose line came back inside somebody else's reply is NOT a generation of its
    # own -- that is the whole point. Count distinct replies, not distinct speakers.
    bodies = {p["said"] for p in out["players"].values() if p["said"]}
    spoke = sorted(out.get("spoke_at", {}).values())
    out["summary"] = {
        "players": a.players,
        # If this is not small, nothing below is a measurement of concurrency.
        "speech_spread_s": round(spoke[-1] - spoke[0], 2) if len(spoke) > 1 else 0.0,
        "answered": len(answered),
        "generations": len(bodies),
        "worst_wait_s": max(waits) if waits else None,
        "median_wait_s": round(st.median(waits), 2) if waits else None,
        "queued_notice": len(out["queued"]),
    }
    print(json.dumps(out, indent=2))
    return 0 if answered else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--players", type=int, default=3)
    p.add_argument("--gap", type=float, default=0.3, help="seconds between each player speaking")
    p.add_argument("--campaign", default="probe", help="scratch campaign; never a real one")
    p.add_argument("--quiet", type=float, default=25.0, help="seconds of silence that ends a turn")
    a = p.parse_args()
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
