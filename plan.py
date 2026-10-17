"""Plan a campaign and produce a PRINTABLE prep pack (plan + party + battle-map grids).

  python plan.py "<campaign name>" "<premise>" [num_sessions]
Generates original content, extracts key locations, renders maps, and writes a
print-ready HTML to state/prep/<campaign>.html (open it and Ctrl-P / Print to PDF).
"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))
import game_state, planner, campaign  # noqa: E402


def _extract_locations(plan: str, limit: int = 3) -> list[str]:
    locs, in_sec = [], False
    for line in plan.splitlines():
        if line.startswith("## "):
            in_sec = "location" in line.lower()
            continue
        if in_sec and line.strip().lstrip("-*").strip():
            name = re.split(r"[—:\-\(]", line.strip().lstrip("-*").strip(), 1)[0].strip()
            if 2 < len(name) < 50:
                locs.append(name)
    return locs[:limit]


async def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    name, premise = sys.argv[1], sys.argv[2]
    sessions = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    g = game_state.GameState.load(name)
    print(f"planning '{name}' ({sessions} sessions)…")
    plan = await planner.plan_campaign(g, premise, sessions)
    locs = _extract_locations(plan) or ["the main location"]
    print("locations:", locs)
    maps = {}
    for loc in locs:
        print(f"  mapping: {loc}")
        maps[loc] = await planner.make_map(loc)
    path = planner.render_prep_html(g, maps)
    g.save()
    print(f"\nprintable prep -> {path}\n(open it in a browser and Print to PDF)")


if __name__ == "__main__":
    asyncio.run(main())
