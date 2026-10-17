"""Is the party still on the story we planned? Answered locally, for free, in milliseconds.

The director spawns Opus specialists on EVENTS -- a scene change fires the showrunner whether
or not anything storyworthy happened. Session 9 spent twelve Opus calls that way, four of them
showrunner passes re-deciding an arc the party had not moved away from.

The arc is already written and sitting in state.plan. So the question "does the big model need
to weigh in" has a cheap proxy: how far is what is actually happening from the beats we
planned? nomic-embed-text is already installed for ollama, runs on the GPU alongside
everything else, and answers in about 20ms. High similarity means the party is walking the arc
and the DM already has what it needs. Low similarity means they have gone somewhere the plan
does not cover, which is exactly when a showrunner pass earns its cost.

Fails OPEN: if ollama is unreachable or the campaign has no plan, this reports drift and the
director behaves as it always did. Silently skipping the story layer because an embedding call
failed would be far worse than paying for a call that was not needed.
"""
import json
import logging
import re
import urllib.request

import config

logger = logging.getLogger("drift")

_BEATS: dict[str, list] = {}          # campaign -> [(text, vector)], embedded once per session
_MISS = {"warned": False}

# A plan is markdown with headings and bullets; the beats are the lines with content in them.
_STRIP = re.compile(r"^[\s#*\->|]+|[*_`]")


def _embed(texts: list[str], timeout: float | None = None) -> list[list[float]] | None:
    """Vectors from the local embedding model, or None if it is not answering.

    Two timeouts on purpose. Loading nomic into VRAM takes about seven seconds, so priming
    gets a generous one; scoring happens between turns and must never hold up the table, so it
    gets a short one and gives up rather than stalling.
    """
    if not texts:
        return []
    try:
        req = urllib.request.Request(
            f"{config.OLLAMA_URL}/api/embed",
            data=json.dumps({"model": config.EMBED_MODEL, "input": texts,
                             "keep_alive": config.OLLAMA_KEEP_ALIVE}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or config.EMBED_TIMEOUT) as r:
            return json.loads(r.read()).get("embeddings")
    except Exception as e:
        if not _MISS["warned"]:
            _MISS["warned"] = True
            logger.info(f"drift: no local embeddings ({e}); the director keeps its old triggers")
        return None


def _cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** .5
    nb = sum(y * y for y in b) ** .5
    return num / (na * nb) if na and nb else 0.0


def beats(plan: str, limit: int = 40) -> list[str]:
    """The plan's individual story beats, one per line worth measuring against."""
    out = []
    for line in (plan or "").splitlines():
        t = _STRIP.sub("", line).strip()
        if len(t.split()) >= 6:            # headings and one-word labels carry no meaning here
            out.append(t)
    return out[:limit]


def prime(state) -> int:
    """Embed the arc for this campaign. Called once when a session opens."""
    plan = getattr(state, "plan", "") or ""
    camp = getattr(state, "campaign", "default")
    bs = beats(plan)
    if not bs:
        _BEATS[camp] = []
        return 0
    vecs = _embed(bs, timeout=config.EMBED_LOAD_TIMEOUT)
    if vecs is None:
        _BEATS.pop(camp, None)
        return 0
    _BEATS[camp] = list(zip(bs, vecs))
    logger.info(f"drift: {len(bs)} arc beats primed for {camp!r}")
    return len(bs)


def score(state, recent: str = "") -> dict:
    """How far the table has wandered. 0 = on the planned beat, 1 = nothing to do with it.

    Returns `drift`, the closest beat it matched, and whether the answer is trustworthy.
    """
    camp = getattr(state, "campaign", "default")
    if camp not in _BEATS:
        prime(state)
    pairs = _BEATS.get(camp) or []
    if not pairs:
        return {"drift": 1.0, "known": False, "why": "no arc to measure against"}
    text = recent or " ".join(list(getattr(state, "log", []))[-8:])
    scene = getattr(state, "scene", "") or ""
    text = f"{scene}. {text}".strip()
    if len(text.split()) < 4:
        return {"drift": 0.0, "known": False, "why": "nothing has happened yet"}
    v = _embed([text])
    if not v:
        return {"drift": 1.0, "known": False, "why": "embeddings unavailable"}
    best, beat = max(((_cos(v[0], bv), bt) for bt, bv in pairs), key=lambda t: t[0])
    return {"drift": round(1.0 - best, 3), "similarity": round(best, 3),
            "beat": beat[:120], "known": True}


def needs_director(state, event: str, recent: str = "") -> tuple[bool, str]:
    """Should this event pay for an Opus pass? (decision, why)

    Structural events are never gated -- the arc has to exist before it can be departed from,
    and the end of a fight genuinely changes the world. Everything else is asked to justify
    itself: if the party is walking the plan, the plan is already in the prompt and a
    showrunner pass would only restate it.
    """
    if event in config.DIRECTOR_ALWAYS:
        return True, f"{event} always runs"
    d = score(state, recent)
    if not d["known"]:
        return True, d.get("why", "cannot measure drift")
    if d["drift"] >= config.DRIFT_THRESHOLD:
        return True, f"drift {d['drift']} >= {config.DRIFT_THRESHOLD} (off the arc)"
    return False, f"drift {d['drift']} < {config.DRIFT_THRESHOLD} (on '{d['beat'][:60]}')"
