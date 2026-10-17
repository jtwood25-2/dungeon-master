"""Safety tools — lines, veils, and an X-card that works mid-sentence.

The X-card is the important one. It has to be usable by someone who is not enjoying
themselves *right now*, which means three properties the rest of the app doesn't need:

  IT CUTS IN.       Recognised on the same fast path as feedback, so it lands while the DM
                    is still talking rather than waiting for a turn boundary.
  NO EXPLANATION.   Nobody is asked why. The scene simply moves. Asking "what was wrong
                    with that?" in front of the table is the thing the card exists to avoid.
  NO POST-MORTEM.   The DM is told to cut away and pick up elsewhere, not to apologise at
                    length — which draws exactly the attention the person didn't want.

Lines and veils are set once and then quietly enforced: a line is content that never
appears, a veil is content that happens but off-screen.
"""
import re

# Must START the utterance, like the feedback trigger -- "cut the rope" is a normal thing to
# say to a DM, "cut" alone at the head of an utterance is not.
X_CARD = re.compile(r"^\s*(?:x[- ]?card|cut(?:\s+(?:that|scene))?|hard stop|red card|"
                    r"pause the game|tap(?:ping)? out)\b[\s:,.!-]*$", re.I)

# "line: torture" / "veil: anything with spiders" / "add a line about drowning"
SET_RE = re.compile(r"^\s*(?:add\s+(?:a\s+)?)?(line|veil)\s*(?:about|on|:|,)?\s+(.{2,60}?)\s*$", re.I)
LIST_RE = re.compile(r"^\s*(?:what are (?:our|the) )?(?:lines and veils|lines|veils)"
                     r"(?:\s+do we have)?\s*\??\s*$", re.I)

# Handed to the DM in place of the player's words. Deliberately prescriptive: a model asked
# to "handle this sensitively" tends to produce a paragraph of apology.
PULL_AWAY = (
    "[SAFETY — X-CARD] Someone at the table just used the X-card on what is happening now. "
    "Do NOT ask why. Do NOT apologise, explain, or refer to it even obliquely. Do NOT ask if "
    "everyone is okay. Cut away from the current content immediately: skip forward, change "
    "location, or resolve the moment off-screen in one short line. Then continue the story "
    "somewhere safe, in two sentences or fewer, and hand back to the players. Never return to "
    "the content that was cut."
)


def is_x_card(text: str) -> bool:
    return bool(X_CARD.match(text or ""))


def parse_set(text: str):
    """Returns ('line'|'veil', topic) or None."""
    if m := SET_RE.match(text or ""):
        return m.group(1).lower(), m.group(2).strip().rstrip(".")
    return None


def is_list_request(text: str) -> bool:
    return bool(LIST_RE.match(text or ""))


def add(state, kind: str, topic: str) -> str:
    bucket = state.lines if kind == "line" else state.veils
    if any(topic.lower() == t.lower() for t in bucket):
        return f"Already on the list."
    bucket.append(topic)
    state.save()
    return (f"Noted — {topic} is off the table." if kind == "line"
            else f"Noted — {topic} happens off-screen.")


def describe(state) -> str:
    if not state.lines and not state.veils:
        return "No lines or veils set."
    out = []
    if state.lines:
        out.append("Lines, never in play: " + ", ".join(state.lines))
    if state.veils:
        out.append("Veils, off-screen only: " + ", ".join(state.veils))
    return ". ".join(out) + "."


def prompt_rule(state) -> str:
    """Injected into every DM turn. Lines and veils are worthless if they're only in the
    system prompt at session start and forgotten forty turns later."""
    if not state.lines and not state.veils:
        return ""
    bits = []
    if state.lines:
        bits.append("NEVER include, describe, or allude to: " + ", ".join(state.lines))
    if state.veils:
        bits.append("These may happen but ALWAYS off-screen, never described: "
                    + ", ".join(state.veils))
    return "[SAFETY] " + " ".join(bits) + " This overrides everything else, including realism."
