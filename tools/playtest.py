"""Play the game and check it, so the bugs are found here instead of at the table.

Every fault in the last four sessions was findable without a person: raw tool output spoken
aloud, a turn that recorded an initiative and said nothing, three death saves resolved in one
step, HP dropping eight points with no line to explain it. None of them needed a human ear --
they needed something to actually take turns and then look at what came back.

Two modes, because they catch different things:

  engine   drives combat and the game state directly with a scripted DM. No model, no network,
           runs in a second. Catches everything mechanical: turn order, death saves, damage
           application, whether the facts handed to the narrator are even coherent.

  live     opens a real WebSocket against the running server and plays scripted lines through
           the whole stack. Slow and costs model calls, so it is short by default. Catches what
           only appears once a language model is in the loop -- silent turns, unnarrated damage,
           tool output reaching the voice.

Both judge the output against the same rules (RULES below), so a failure means the same thing
in either mode.

Usage
  python tools/playtest.py engine
  python tools/playtest.py engine --rounds 40 --seeds 25
  python tools/playtest.py live --turns 6
  python tools/playtest.py check "state/transcripts/Beyond the Grove 2026-08-23.jsonl"
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

# Things that must never be SPOKEN. Each one was heard at the table.
RULES = [
    ("raw dice notation", re.compile(r"\b\d*d\d+\s*[+-]?\s*\d*\s*=|🎲|\[1d\d")),
    ("a bracketed action", re.compile(r"\[[A-Z]{3,}\b|\[/?[A-Za-z]+\s+[^\]]*\]")),
    ("a difficulty class", re.compile(r"\bDC\s*\d+", re.I)),
    ("tool bookkeeping", re.compile(r"\b(?:hp|max_hp|init_mod|turn_idx|is_pc|in_combat)\b")),
    ("a stray status tag", re.compile(r"\[(?:DEATH SAVE|COMBAT|BALANCE|RESULTS|STATE|PLAYER)")),
    # The CLI reports a spent subscription as an ordinary assistant message, so the DM read
    # "You've hit your session limit - resets 4:30pm" aloud on every turn until the reset.
    ("a plumbing message", re.compile(
        r"session limit|usage limit|rate.?limit|resets? at|\bAPI\b|anthropic|claude code|"
        r"not logged in|try again later", re.I)),
]


# Lines the DM is meant to ignore entirely -- silence is the correct answer.
EXPECT_SILENCE = re.compile(r"did you guys|you all|y'?all|anyone else", re.I)


def spoken_faults(line: str) -> list:
    return [why for why, pat in RULES if pat.search(line or "")]


def _w(s: str) -> list:
    """Content words, for spotting the same beat delivered twice."""
    return [x for x in re.findall(r"[a-z']+", (s or "").lower()) if len(x) > 3]


class Check:
    """Collects violations with enough context to act on."""

    def __init__(self):
        self.bad: list = []

    def fail(self, kind: str, detail: str, ctx: str = ""):
        self.bad.append({"kind": kind, "detail": detail, "context": ctx[:200]})

    def speech(self, line: str):
        for why in spoken_faults(line):
            self.fail("spoke " + why, line[:160])

    def report(self, name: str) -> dict:
        return {"scenario": name, "violations": len(self.bad), "detail": self.bad[:40]}


# ---------------------------------------------------------------- engine mode
def engine_fight(seed: int, rounds: int, chk: Check, monsters=("goblin", 2), pc_hp=12):
    """A whole fight driven through the real engine, with a scripted player."""
    import combat
    import tools as tool_mod
    from game_state import GameState

    g = GameState(f"_playtest{seed}")
    g.rng = random.Random(seed)
    g.add("Tester", max_hp=pc_hp, ac=15, is_pc=True, init_mod=2, stats={"dex": 14})
    tool_mod.dispatch("spawn_monster", {"name": monsters[0], "count": monsters[1]}, g)
    tool_mod.dispatch("start_combat", {}, g)
    tool_mod.dispatch("set_initiative", {"name": "Tester", "value": 12}, g)

    pc = g.get("Tester")
    seen_saves, last_hp, guard = 0, pc.hp, 0
    for _ in range(rounds):
        guard += 1
        before_round, before_hp = g.round, pc.hp
        out = combat.brief(g, "")

        # every line the narrator is handed must be safe to say and must be self-consistent
        # only the fact lines are candidates for speech; anything in [brackets] is an
        # instruction to the narrator and is never read out
        for line in (out or "").splitlines():
            if line.startswith(">"):
                chk.speech(line[1:].strip())

        # A death save must arrive ONE at a time. The counter RESETS on a natural 20 and on
        # stabilising, so comparing running totals reads a reset as a batch. Count what the
        # brief actually REPORTS instead.
        saves = pc.death_saves["success"] + pc.death_saves["fail"]
        reported = len(re.findall(r"death save|natural 20|stabilis|is dead", out or "", re.I))
        # a natural 1 is ONE roll that costs two failures -- correct 5e, not a batch
        reported += len(re.findall(r"fail x2|natural 1\b", out or "", re.I))
        if saves > seen_saves + 1 and reported < (saves - seen_saves):
            chk.fail("death saves resolved in a batch",
                     f"{saves - seen_saves} rolled, {reported} reported", out)
        seen_saves = saves

        # HP must never move without the brief saying so
        if pc.hp != before_hp:
            moved = abs(before_hp - pc.hp)
            if not re.search(r"\bhits?\b|damage|for \d+|is now|back up|stabilis|heal",
                             out or "", re.I):
                chk.fail("HP changed with nothing said",
                         f"{before_hp} -> {pc.hp} ({moved})", out)
        # never more than one round per call
        if g.round > before_round + 1:
            chk.fail("multiple rounds in one call", f"{before_round} -> {g.round}", out)

        if "dead" in pc.conditions:
            if pc.death_saves["fail"] < 3:
                chk.fail("died without three failed saves", str(pc.death_saves), out)
            break
        if not g.in_combat:
            break
        # Asking a question must never cost the player a round. This is what session 12 died
        # of and the harness did not catch it, because the scripted player only ever acted --
        # it never stopped to ask anything, which is most of what a real player does.
        if g.current_turn() == "Tester" and pc.hp > 0 and tool_mod is not None:
            before = (pc.hp, g.round, g.turn_idx)
            for q in ("What was the damage?", "How is it their turn? I haven't gone yet.",
                      "Have any of them taken damage?"):
                if combat.brief(g, q):
                    chk.fail("a question advanced the fight", q, "")
            if (pc.hp, g.round, g.turn_idx) != before:
                chk.fail("the board moved while the player was asking questions",
                         f"{before} -> {(pc.hp, g.round, g.turn_idx)}", "")

        # the scripted player acts on their own turn
        if g.current_turn() == "Tester" and pc.hp > 0:
            tool_mod.dispatch("roll_dice", {"notation": "1d20+4", "reason": "attack"}, g)
            target = next((c for c in g.combatants.values() if not c.is_pc and c.hp > 0), None)
            if target is None:
                break
            tool_mod.dispatch("apply_damage", {"name": target.name, "amount": 5}, g)
            # moving is free and must not hand over; only saying so does
            combat.brief(g, "I take my full movement.")
            if g.current_turn() != "Tester":
                chk.fail("movement ended the turn", "", "")
            combat.brief(g, "That's my turn.")
        last_hp = pc.hp
    # a fight has to actually end
    if guard >= rounds and g.in_combat:
        alive = [c.name for c in g.combatants.values() if not c.is_pc and c.hp > 0]
        if not alive:
            chk.fail("combat never ended", "no monsters left standing but still in combat")
    return g


def run_engine(a) -> dict:
    chk = Check()
    scenarios = []
    for seed in range(a.seeds):
        for label, mon, hp in (("even fight", ("goblin", 2), 12),
                               ("outnumbered", ("goblin", 4), 12),
                               ("one big one", ("ogre", 1), 20),
                               ("nearly dead", ("goblin", 2), 1)):
            before = len(chk.bad)
            try:
                engine_fight(seed, a.rounds, chk, mon, hp)
            except Exception as e:
                import traceback
                chk.fail("engine raised", f"{type(e).__name__}: {e}",
                         traceback.format_exc()[-300:])
            if len(chk.bad) > before:
                scenarios.append(f"{label} seed={seed}")
    out = chk.report("engine")
    out["scenarios_run"] = a.seeds * 4
    out["failing_scenarios"] = scenarios[:20]
    return out


# ---------------------------------------------------------------- transcript mode
def run_check(path: Path) -> dict:
    """Judge a transcript that already happened -- ours or a real session."""
    chk = Check()
    dm_lines, hp_claims = [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            r = json.loads(raw)
        except Exception:
            continue
        text = (r.get("text") or "").strip()
        if (r.get("role") or r.get("speaker")) == "dm":
            dm_lines.append(text)
            chk.speech(text)
            if m := re.search(r"you(?:'re| are) (?:at|on|down to) (\d+)", text, re.I):
                hp_claims.append((r.get("t", ""), int(m.group(1))))
        if (fa := r.get("first_audio_ms")) and fa > 15000:
            chk.fail("turn took too long", f"{fa}ms to first audio", text)
    # HP announced to the table must only ever fall, unless healing was mentioned
    for (t1, a1), (t2, b1) in zip(hp_claims, hp_claims[1:]):
        if b1 > a1 and not any(re.search(r"heal|potion|regain|goodberry", d, re.I)
                               for d in dm_lines):
            chk.fail("HP went UP with no healing", f"{t1}: {a1} -> {t2}: {b1}")
    out = chk.report(str(path.name))
    out["dm_lines"] = len(dm_lines)
    return out


# ---------------------------------------------------------------- live mode
async def _live(a) -> dict:
    import websockets
    chk = Check()
    url = a.url or "ws://127.0.0.1:8080/ws"
    # A fight, played the way a person plays one: declare, roll, ask what happened, ask a rules
    # question mid-turn, and keep going. The mechanical beats are the ones that have broken.
    # A TABLE, not one player: several people acting in turn, chatter between them that the DM
    # must let go, a rules question mid-fight, and turns ended out loud. Everything that has
    # broken combat lives in the gaps between these lines.
    script = [
        "Recap where we left off, then put us back in the scene.",
        "What do I see ahead of me?",
        "I attack the closest one with my staff.",
        "17.",            # four people give initiative, one bare number each
        "12.",
        "19.",
        "8.",
        "I rolled 17.",
        "I rolled 6 damage.",
        "Have any of them taken damage?",              # a question must not cost a turn
        "I take my full movement toward the door.",     # moving must not end it either
        "That's my turn.",                              # only this ends it
        "did you guys see the game last night",         # table talk: must be ignored entirely
        "Bryn attacks the one on the left.",
        "I rolled 19.",
        "I rolled 8 damage.",
        "That's it for Bryn.",
        "Whose turn is it now?",
        "Sella shoots the nearest one.",
        "I rolled 11.",
        "How many hit points do I have left?",
        "That's my turn.",
    ][:a.turns]
    import asyncio
    import urllib.request
    # NEVER play in a real campaign. The first live run of this harness drove the actual save
    # into a fight and killed the character in it -- a test that can lose someone's game is
    # worse than no test. The server deliberately refuses to conjure a missing campaign, so
    # make it here rather than silently landing back in whichever one is current.
    base = (a.url or "ws://127.0.0.1:8080/ws").replace("ws://", "http://").replace("/ws", "")
    try:
        req = urllib.request.Request(
            f"{base}/campaigns",
            data=json.dumps({"name": a.campaign,
                             "premise": "A bare room with a door and two goblins in it. "
                                        "Used only for automated testing."}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=120).read()
    except Exception as e:
        if "409" not in str(e):                    # already there is the normal case
            return {"error": f"could not create the scratch campaign: {e}"}

    # Give it a character. Without one the fight cannot start -- there is nobody to roll
    # initiative for -- and the run measures the DM asking "what's your character's name?"
    # instead of measuring combat. Written straight to the save; we are on the same machine.
    from game_state import GameState
    g = GameState.load(a.campaign)
    if not any(c.is_pc for c in g.combatants.values()):
        # Seat a PARTY, not one character. Every fault so far has been found solo or in the
        # engine alone, and a table is a different machine: turn order interleaves with
        # monsters, the microphone hears people who are not taking a turn, and "whose go is
        # it" stops being obvious.
        crew = [("Tester", 20, 15, "Quarterstaff", "1d8 Bludgeoning"),
                ("Bryn", 26, 16, "Longsword", "1d8 Slashing"),
                ("Sella", 18, 13, "Shortbow", "1d6 Piercing"),
                ("Hob", 22, 14, "Handaxe", "1d6 Slashing")][:max(1, a.party)]
        for nm, hp, ac, wep, dmg in crew:
            pc = g.add(nm, max_hp=hp, ac=ac, is_pc=True, init_mod=2,
                       stats={"str": 12, "dex": 14, "con": 14, "int": 10, "wis": 15, "cha": 11})
            pc.level = 3
            pc.actions = [{"name": wep, "kind": "attack", "damage": dmg, "equipped": True}]
            g.roster.append({"player": nm.lower(), "character": nm,
                             "ddb_id": "", "voice_id": nm.lower()})
        g.save(force=True)

    heard: list = []
    # Opening a session is not instant: the server primes the arc and fires the director's
    # specialists, which is tens of seconds of background work on a real campaign. Connecting
    # and immediately talking raced all of it and the socket was closed under us. No pings
    # from either side, and a real pause before the first line.
    async with websockets.connect(url, max_size=None, ping_interval=None,
                                  close_timeout=10) as ws:
        await ws.send(json.dumps({"type": "campaign", "name": a.campaign}))
        await asyncio.sleep(a.settle)
        for line in script:
            await ws.send(json.dumps({"type": "text", "text": line}))
            said, quiet = [], 0
            while quiet < a.wait:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    quiet += 1
                    continue
                except Exception:
                    break
                if isinstance(msg, bytes):
                    continue                    # audio
                try:
                    ev = json.loads(msg)
                except Exception:
                    continue
                if ev.get("type") == "dm":
                    t = ev.get("text", "")
                    # The same beat twice in one turn: "Your staff cracks into the first
                    # goblin's shoulder. It staggers but stays up." then the same sentence
                    # again about "the nearer goblin". Said aloud it sounds broken.
                    for prev in said:
                        was, now = set(_w(prev)), set(_w(t))
                        if len(was) >= 5 and len(was & now) / max(1, len(was | now)) > 0.6:
                            chk.fail("said the same beat twice", f"{prev[:70]} // {t[:70]}", line)
                            break
                    said.append(t)
                    chk.speech(t)
                elif ev.get("type") == "status" and "⚠" in str(ev.get("msg", "")):
                    chk.fail("silent turn", str(ev.get("msg"))[:120], line)
            # Some lines are SUPPOSED to get nothing: table talk the DM must let go. Scoring
            # its silence as a fault is scoring the feature as a bug.
            if not said and not EXPECT_SILENCE.search(line):
                chk.fail("no answer at all", "nothing spoken", line)
            heard.append({"said": line, "dm": " ".join(said)[:300]})
    out = chk.report("live")
    out["exchanges"] = heard
    return out


def run_live(a) -> dict:
    import asyncio
    try:
        return asyncio.run(_live(a))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "hint": "is the server up? python start.py --status"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("engine", help="scripted fights, no model")
    e.add_argument("--rounds", type=int, default=30)
    e.add_argument("--seeds", type=int, default=12)
    lv = sub.add_parser("live", help="real websocket session against the running server")
    lv.add_argument("--turns", type=int, default=4)
    lv.add_argument("--url", default="")
    lv.add_argument("--party", type=int, default=4,
                    help="how many characters to seat (a table behaves differently)")
    lv.add_argument("--settle", type=int, default=45,
                    help="seconds to let the session finish opening before speaking")
    lv.add_argument("--campaign", default="_playtest",
                    help="scratch campaign to play in; never point this at a real one")
    lv.add_argument("--wait", type=int, default=25, help="seconds of quiet before moving on")
    c = sub.add_parser("check", help="judge a transcript that already happened")
    c.add_argument("path")
    a = p.parse_args()

    if a.mode == "engine":
        out = run_engine(a)
    elif a.mode == "live":
        out = run_live(a)
    else:
        out = run_check(Path(a.path))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 1 if out.get("violations") or out.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
