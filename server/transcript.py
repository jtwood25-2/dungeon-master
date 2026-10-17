"""Every line of play, written down.

The point isn't archival — it's that you cannot tell whether the DM is any good from inside a
session. Reading a whole session back afterwards is the only way to see the patterns: the
places it spoke for a player, recited a number it shouldn't have, forgot to ask for damage, or
answered a question with scenery.

One JSONL file per campaign per day, appended as play happens, so a crash loses nothing and
`review.py` can hand a whole session to a stronger model for critique.
"""
import json
import time
from pathlib import Path

DIR = Path(__file__).parent.parent / "state" / "transcripts"


def path_for(campaign: str, day: str | None = None) -> Path:
    day = day or time.strftime("%Y-%m-%d")
    safe = "".join(ch for ch in campaign if ch.isalnum() or ch in " -_").strip() or "untitled"
    return DIR / f"{safe} {day}.jsonl"


def log(campaign: str, session: int, role: str, text: str, **meta):
    """role: 'player' | 'dm' | 'system'. meta carries latency, which brain answered, etc."""
    if not (text or "").strip():
        return
    rec = {"t": time.strftime("%H:%M:%S"), "session": session, "role": role,
           "text": text.strip(), **{k: v for k, v in meta.items() if v is not None}}
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        with path_for(campaign).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass                                   # logging must never interrupt play


def files() -> list[Path]:
    if not DIR.exists():
        return []
    return sorted(DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def load(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def recent(campaign: str, n: int = 8) -> str:
    """The last few exchanges, for picking the thread back up after a restart. The DM's own
    memory of the conversation dies with the process; the scene shouldn't die with it."""
    p = path_for(campaign)
    if not p.exists():
        return ""
    lines = []
    for e in load(p)[-40:]:
        if e["role"] == "system":
            continue
        who = "Player" if e["role"] == "player" else "You"
        lines.append(f"{who}: {e['text']}")
    return "\n".join(lines[-n:])


def as_script(entries: list[dict], session: int | None = None) -> str:
    """The transcript as a readable script, which is what a reviewer needs to see."""
    lines = []
    last = None
    for e in entries:
        if session is not None and e.get("session") != session:
            continue
        if e.get("session") != last:
            last = e.get("session")
            lines.append(f"\n--- session {last} ---")
        who = {"player": "PLAYER", "dm": "DM", "system": "SYSTEM"}.get(e["role"], e["role"])
        tag = ""
        if e.get("route"):
            tag = f"  [{e['route']}]"
        if e.get("first_audio_ms"):
            tag += f"  [{e['first_audio_ms']}ms]"
        lines.append(f"{who}: {e['text']}{tag}")
    return "\n".join(lines).strip()
