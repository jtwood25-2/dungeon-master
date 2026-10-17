"""Response-time simulation: end-of-speech -> first DM voice, and how smooth it feels.

Local stages (cache lookup, dice, tools, state) are MEASURED here on this machine.
Remote stages (STT, Claude, TTS, network) are MODELED from typical benchmarks -- clearly
labelled -- since they need the GPU/API/phone to measure for real. Runs a Monte-Carlo over
realistic turns (narration vs combat) for three setups: LAN, remote tunnel, remote+local TTS.
"""
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))
import rules, dice, game_state, tools  # noqa: E402

TARGET_S = 2.5          # a turn feels "real-time" if first voice lands within this


# ---- MEASURED local stages (microseconds) ----------------------------------
def measure_local() -> dict:
    rules.index()                                   # warm the cache
    g = game_state.GameState("bench"); g.rng.seed(1)
    def t(fn, n=2000):
        s = time.perf_counter()
        for _ in range(n): fn()
        return (time.perf_counter() - s) / n * 1e6   # us
    tools.dispatch("spawn_monster", {"name": "goblin", "count": 3}, g)
    tools.dispatch("add_combatant", {"name": "Hero", "max_hp": 30, "ac": 16, "is_pc": True}, g)
    tools.dispatch("start_combat", {}, g)
    m = {
        "rules_lookup": t(lambda: rules.lookup("monster", "goblin")),
        "dice_roll":    t(lambda: dice.roll("1d20+5", g.rng)),
        "apply_damage": t(lambda: tools.dispatch("apply_damage", {"name": "Hero", "amount": 0}, g)),
        "advance_turn": t(lambda: tools.dispatch("advance_turn", {}, g)),
    }
    Path("server/state/bench.json").unlink(missing_ok=True)
    return m


# ---- MODELED remote stages (ms) --------------------------------------------
def clamp(x, lo): return max(lo, x)

def net(one_way_ms, jitter):                        # up or down hop
    return clamp(random.gauss(one_way_ms, jitter), 1)

def stt_ms():    return clamp(random.gauss(280, 70), 90)        # faster-whisper small, GPU, short clip
def llm_ttft():  return clamp(random.gauss(450, 130), 180)      # Claude Haiku, time-to-first-token
def llm_tool():  return clamp(random.gauss(420, 130), 160)      # one tool round-trip (decide+continue)
def tts_edge():  return clamp(random.gauss(380, 100), 150)      # edge-tts first audio (network)
def tts_local(): return clamp(random.gauss(120, 40), 40)        # Piper on GPU, first audio


def sim_turn(setup: str, local_us: dict) -> float:
    """Return end-of-speech -> first-DM-audio, in seconds, for one turn."""
    remote = setup != "LAN"
    up = net(50, 25) if remote else net(12, 6)
    down = net(50, 25) if remote else net(12, 6)
    # combat turns tend to use tools (rolls/damage); narration turns often don't
    is_combat = random.random() < 0.45
    n_tools = random.choice([1, 2, 2, 3]) if is_combat else random.choice([0, 0, 0, 1])
    tts = tts_local() if setup == "remote+localTTS" else tts_edge()
    local_ms = sum(local_us.values()) / 1000 * (n_tools + 1)     # measured, negligible
    ms = up + stt_ms() + llm_ttft() + n_tools * llm_tool() + local_ms + tts + down
    return ms / 1000


def report(setup: str, local_us: dict, n=8000):
    xs = sorted(sim_turn(setup, local_us) for _ in range(n))
    med = statistics.median(xs); p95 = xs[int(n * 0.95)]
    under = sum(x <= TARGET_S for x in xs) / n
    verdict = "smooth" if med <= 2.0 and under >= 0.9 else ("usable" if med <= 2.8 else "laggy")
    print(f"  {setup:<18} median {med:4.2f}s   p95 {p95:4.2f}s   "
          f"<{TARGET_S}s {under*100:4.0f}%   -> {verdict}")


def main():
    print("\n=== MEASURED local stages (this machine) ===")
    local = measure_local()
    for k, v in local.items():
        print(f"  {k:<14} {v:7.1f} us")
    print("  (all local game logic is sub-millisecond -> not the bottleneck)")

    print(f"\n=== SIMULATED end-of-speech -> first DM voice  (target <{TARGET_S}s) ===")
    print("  [remote stages MODELED from typical benchmarks; local stages measured above]")
    for setup in ["LAN", "remote(tunnel)", "remote+localTTS"]:
        report(setup, local)
    print("\n  Takeaways:")
    print("  - The DM's game logic (dice/rules/combat) is ~free; latency = STT+Claude+TTS+network.")
    print("  - Going off-network (tunnel) adds ~0.1s of network — small.")
    print("  - Biggest lever: local TTS (Piper) instead of edge-tts saves ~0.25s and de-jitters.")
    print("  - Combat turns cost more (tool rounds); narration turns are fastest.")


if __name__ == "__main__":
    main()
