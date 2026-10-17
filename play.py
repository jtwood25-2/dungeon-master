"""Text-mode DM — play/test the entire brain with NO audio, NO phone, NO latency pipeline.

Run:  python play.py [campaign]
Type what your character does; the DM responds, rolling real dice and tracking combat.
Commands: /state  /roll 1d20+3  /event scene_change  /save  /quit
Needs ANTHROPIC_API_KEY for real narration; without it you still get the mechanics + mock
director notes (useful for testing the plumbing).
"""
import asyncio
import sys
from pathlib import Path

try:                                   # emoji/dice glyphs on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent / "server"))
import config, dice, game_state, director as director_mod, llm  # noqa: E402
import campaign as campaign_mod, planner  # noqa: E402


async def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "default"
    state = campaign_mod.start_session(name)
    director = director_mod.Director(state)
    history: list = []

    print(f"\n🐉 Dungeon Master — '{name}', session {state.session}  [backend: {config.DM_BACKEND}]")
    if state.recap:
        print(f"   previously: {state.recap}")
    print("   commands: /state  /roll <dice>  /event <name>  /save  /quit\n")
    director.on_event("session_start")

    while True:
        try:
            line = (await asyncio.to_thread(input, "you » ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/state":
            print("\n" + state.summary() + "\n"); continue
        if line == "/save":
            state.save(); print("saved.\n"); continue
        if line.startswith("/roll "):
            print("  " + str(dice.roll(line[6:], state.rng)) + "\n"); continue
        if line.startswith("/event "):
            director.on_event(line[7:].strip()); print("  (director dispatched)\n"); continue

        print("DM » ", end="", flush=True)
        async for sentence in llm.stream_dm(line, state, director, history):
            print(sentence, end=" ", flush=True)
        print("\n")
        state.save()

    await director.drain()
    print("saving session recap for next time…")
    try:
        await planner.recap_session(state)
    except Exception:
        pass
    state.save()
    print("session saved. farewell, adventurer.")


if __name__ == "__main__":
    asyncio.run(main())
