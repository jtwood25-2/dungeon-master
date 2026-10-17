"""Put open feedback in front of Claude on every turn.

Claude only acts when prompted, so "I'll check the queue" is a promise it cannot keep — it
forgets, and notes sit unread for days. This runs as a UserPromptSubmit hook, which is
deterministic: the queue is injected into context whether or not anyone remembers it.

Silent when there's nothing open, so it costs nothing on a quiet day.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

try:
    import feedback
    items = [i for i in feedback.items() if i.get("status") == "open"]
except Exception:
    sys.exit(0)                       # never break a turn over the reminder

if not items:
    sys.exit(0)

lines = [f"{len(items)} unaddressed feedback note(s) from the player "
         f"(state/feedback.jsonl). Work these in unless the user asks for something else; "
         f"mark each done via feedback.set_status(id, 'done') once it is actually fixed:"]
for i in items:
    lines.append(f"  - [{i['id']}] ({i['ts']}) {i['text']}")

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": "\n".join(lines),
    },
    "suppressOutput": True,
}))
