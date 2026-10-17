"""Is this actually for the Dungeon Master? Decided locally, before a metered call is spent.

A seven-player session settled an argument I had been having with myself in regex. Measured
across 417 turns in one evening:

    417  turns sent to Sonnet, every one a metered call
    126  of them (30%) produced NO AUDIO AT ALL
    190  more were rejected by the regex gate before reaching a model

The silent 30% were almost entirely the room talking to itself — "Dude, I hate it.", "Thank
you.", "What's up, man?", "Fuck you!" — reaching the story model, which correctly declined to
answer. Every one of those cost a metered call, occupied the turn desk while real play queued
behind it, and left the table watching a DM that appeared to be thinking and then said nothing.
That is what "way too slow" turned out to mean: not latency, silence. The median turn that
DID answer took 2.5 seconds.

Both regex gates failed, in opposite directions, and the failures are instructive:

  strict   an allow-list. Rejected almost everything real players said, because play does not
           announce itself as an action. The table concluded it could not hear them.
  loose    a deny-list. Let everything through, so the flood above happened instead.

The missing piece was never a better pattern. Whether a sentence is addressed to the DM is a
judgement about language, and there is a 14B model already resident in VRAM, already warm,
answering in a few hundred milliseconds, costing nothing per call. It should be making this
decision — and the expensive model should only ever see what survives it.

Order matters, because latency does:

  1. cheap YES  a declared action, a roll, an end-of-turn, direct address. No model, no wait.
  2. cheap NO   plainly said to another player by name.
  3. ask qwen   everything else, ~300ms, free, and far better at this than a pattern ever was.

A local model that is down or slow must never silence the table, so every failure path here
answers YES and lets the turn through. Being occasionally too talkative is recoverable; being
deaf is what ruined an evening.
"""
import logging

logger = logging.getLogger("addressed")

SYS = (
    "You decide whether one line of speech, recorded at a tabletop RPG table, is addressed to "
    "the Dungeon Master or is the players talking among themselves.\n"
    "Answer with exactly one word: DM or TABLE.\n"
    "DM = anything the Dungeon Master should respond to: describing what a character does, "
    "asking about the world or the rules, giving a die roll or a number, asking what happens, "
    "reacting to the fiction in a way that expects an answer, ending a turn.\n"
    "TABLE = anything else: jokes, side conversation, talking to another player by name, "
    "arguing about pizza, commenting on the software, greetings, swearing, thinking aloud.\n"
    "When it is genuinely ambiguous, answer DM. A missed line is worse than an extra one."
)

# Answering "TABLE" wrongly makes the DM deaf, which is the failure that ruined a session, so
# the bar to reject is deliberately high: the model has to be brief and unambiguous about it.
_REJECT = "table"


async def ask(text: str, party=None, timeout: float = 4.0) -> bool:
    """True if the DM should answer. Never raises; never returns False on an error."""
    import config
    import httpx
    t = (text or "").strip()
    if not t:
        return False
    who = ", ".join(p for p in (party or []) if p)
    prompt = (f"The characters at this table are: {who}.\n" if who else "") + \
             f'Line: "{t}"\nDM or TABLE?'
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{config.OLLAMA_URL}/api/chat", json={
                "model": config.OLLAMA_MODEL, "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "messages": [{"role": "system", "content": SYS},
                             {"role": "user", "content": prompt}],
                # One word is all that is wanted, and capping it is most of why this is fast.
                "options": {"temperature": 0.0, "num_predict": 4}})
            r.raise_for_status()
            out = (r.json()["message"]["content"] or "").strip().lower()
    except Exception as e:
        # Down, slow, or busy loading. Let it through: a DM that answers something not meant
        # for it is a nuisance, a DM that cannot hear is the end of the evening.
        logger.info(f"local gate unavailable ({type(e).__name__}); letting it through")
        return True
    if out.startswith(_REJECT):
        logger.info(f"local gate: table talk, not sent to the story model: {t[:60]!r}")
        return False
    return True
