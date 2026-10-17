"""The "previously on" — written and voiced when a session ENDS, not when the next one starts.

A recap is the one thing a DM says that nobody should ever wait for. It describes events that
have already happened, it changes only when the story does, and it is the very first thing the
table hears — which is exactly when this engine is at its slowest. Measured on a real opening:

    08:50:00  heard: "Recap where we left off, then put us back in the scene."
    08:50:08  first token        (7.9s — the continuity block, and the director firing inside the turn)
    08:50:14  first audio       (14.3s — 6.4s more for speech synthesis under GPU contention)

Fourteen seconds of silence at the top of a session, and the table talked over it because they
assumed it had broken. Every second of that was spent re-deriving something already knowable
when the previous session ended.

So it is built at the END of a session, when the room is empty and nobody is waiting:

  - written by OPUS rather than the story model. Latency is free here, so the recap can be
    given the job it deserves — set the scene properly — instead of being rushed.
  - rendered to audio in the same pass and kept as a .wav beside the text.
  - served instantly on the next connect: no model call, no synthesis, no wait.

Staleness is judged by the story moving, not by age. The stored recap records the session
number it was written to open, and a mismatch sends the opening back to the live path. A recap
that is merely old is still true; a recap describing a different evening is worse than none.
"""
import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger("recap")

ROOT = Path(__file__).parent.parent / "state" / "recaps"

SYS = (
    "You are a Dungeon Master opening a session of a live tabletop RPG. Write the "
    "'previously on' — what the party did last time and where they are standing right now.\n"
    "It is spoken aloud by a voice engine, so: plain text only. No markdown, no headings, no "
    "lists, no stage directions, no asterisks.\n"
    "Six to ten sentences. Concrete and physical: names, places, what was decided, what was "
    "left unfinished. Never 'something', 'somewhere' or 'somehow' — vagueness reads as "
    "atmosphere and is really the sentence refusing to commit.\n"
    "Second person to the party — 'you'. Never narrate what the characters feel or decide; "
    "that is theirs.\n"
    "End by putting them back in the scene and handing them the moment, without asking the "
    "same stock question every time."
)


def _safe(campaign: str) -> str:
    return "".join(c for c in str(campaign) if c.isalnum() or c in " -_").strip() or "untitled"


def _path(campaign: str) -> Path:
    return ROOT / f"{_safe(campaign)}.json"


def _wav_path(campaign: str) -> Path:
    return ROOT / f"{_safe(campaign)}.wav"


def _wire_path(campaign: str) -> Path:
    """The clip in the exact form the phone receives it.

    Encoding for the wire costs 2.5 seconds on this recording and compresses 6x -- worth paying
    once, at build time, rather than every time a session opens. Serving is then a file read.
    """
    return ROOT / f"{_safe(campaign)}.wire"


def _session_of(state) -> int:
    return int(getattr(state, "session", 0) or 0)


def _voice_id() -> str:
    """Which narrator rendered this, so a cached clip can never be in a different voice.

    ONE voice throughout the evening is not a preference, it is the point: a recap in one
    narrator handing over to a session in another is more jarring than a slow recap, and the
    audio here is minted minutes-to-days before it is played. Stamping the engine and voice
    onto the clip makes the guarantee mechanical rather than a thing somebody has to remember
    when they change a setting.
    """
    import config
    return f"{config.TTS_ENGINE}:{config.TTS_VOICE}"


def load(state):
    """The stored recap, or None if there is none or it no longer opens THIS session."""
    p = _path(state.campaign)
    if not p.exists():
        return None
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"stored recap unreadable ({e}); ignoring it")
        return None
    # The session number must match exactly. Replaying the recap of an evening the party has
    # already had is not a stale nicety — it is the DM confidently describing the wrong night.
    if saved.get("session") != _session_of(state):
        logger.info(f"stored recap opens session {saved.get('session')}, this is "
                    f"{_session_of(state)} — falling back to the live path")
        return None
    if not (saved.get("text") or "").strip():
        return None
    # ONCE PER SESSION, not once per connection. open_session() fires on every websocket, and
    # phones reconnect constantly -- a locked screen, a tunnel blink, a reload. The live path
    # at least improvised something new each time; a stored recap replays the identical
    # "previously on" into the middle of a session that is already running, which is what
    # "the recap got confused" actually was.
    if saved.get("played_session") == _session_of(state):
        logger.info(f"recap already played this session ({_session_of(state)}); not repeating it")
        return None
    # Also not once the evening is under way. Reconnecting an hour in wants the scene, not the
    # story so far -- and turns_played counts what the PLAYERS said, so the recap's own line
    # does not make the session look busy.
    if turns_played(state) >= PLAYED_TURNS:
        logger.info(f"session {_session_of(state)} is already {turns_played(state)} turns in; "
                    f"the recap has had its moment")
        return None
    if (was := saved.get("voice")) and was != _voice_id():
        # The text is still true; only the recording is in the wrong narrator. Keep the words
        # and let them be spoken live in the voice the session will actually use.
        logger.info(f"stored recap was voiced by {was}, now running {_voice_id()} — "
                    f"keeping the text, re-speaking it live")
        saved["wire"] = saved["audio"] = b""
        return saved
    wire, wav = _wire_path(state.campaign), _wav_path(state.campaign)
    saved["wire"] = wire.read_bytes() if wire.exists() else b""
    saved["audio"] = wav.read_bytes() if wav.exists() else b""
    mark_played(state)
    return saved


def mark_played(state) -> None:
    """Record that this session has heard it, so a reconnect does not hear it again."""
    p = _path(state.campaign)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["played_session"] = _session_of(state)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"could not mark the recap played ({e}); it may repeat on a reconnect")


# A session where nothing happened must not overwrite a good recap with a worse one. Someone
# opens the app, hears the "previously on", says one word and puts the phone down -- that is
# not an evening, and the recap that opened it is still the correct recap for next time. The
# threshold counts what the PLAYERS actually said, because DM lines include the recap it just
# played and would count a silent session as busy.
PLAYED_TURNS = 4


def turns_played(state) -> int:
    """How many things the players actually said this session."""
    import transcript
    n = 0
    try:
        for f in transcript.files():
            for row in transcript.load(f):
                if (row.get("session") == _session_of(state)
                        and row.get("role") == "player" and (row.get("text") or "").strip()):
                    n += 1
    except Exception:
        return PLAYED_TURNS          # cannot tell -> assume it counted, and rewrite
    return n


def _restamp(state, opens: int) -> dict | None:
    """Carry the existing recap forward to open the next session, unchanged.

    Cheaper than rewriting and strictly better than rewriting from nothing: no Opus call, no
    synthesis, and the audio on disk is still exactly right.
    """
    p = _path(state.campaign)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not (rec.get("text") or "").strip():
        return None
    rec["session"] = opens
    rec["reused"] = int(rec.get("reused", 0)) + 1
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"nothing happened in session {_session_of(state)} of {state.campaign!r}; "
                f"reusing the existing recap for session {opens}")
    return rec


async def build(state, force: bool = False):
    """Write and voice the recap that will OPEN the next session. Call when one ends.

    Returns the stored record, or None when there is nothing worth recapping or the model was
    unavailable — in which case the live path still works, just slowly.
    """
    import campaign as campaign_mod
    if campaign_mod.is_unplayed(state):
        return None                       # nothing has happened yet; an opener is not a recap
    opens = _session_of(state) + 1        # written now, played at the start of the next session
    if not force:
        p = _path(state.campaign)
        if p.exists():
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("session") == opens:
                    return None           # already written for that session
            except Exception:
                pass
        # Did this session actually happen? A recap and one word is not an evening, and the
        # recap that opened it is still the right one to open the next.
        if (played := turns_played(state)) < PLAYED_TURNS:
            if (kept := _restamp(state, opens)) is not None:
                return kept
            logger.info(f"only {played} player turn(s) this session and no recap to carry "
                        f"forward; writing one anyway")

    text = await _write(state)
    if not text:
        return None
    ROOT.mkdir(parents=True, exist_ok=True)
    rec = {"text": text, "session": opens, "written": time.time(), "campaign": state.campaign,
           "voice": _voice_id()}
    if audio := await _voice(text):
        _wav_path(state.campaign).write_bytes(audio)
        rec["audio_bytes"] = len(audio)
        try:                                   # encode once here, never on the way to a phone
            import tts
            wire = await tts.for_wire(audio)
            _wire_path(state.campaign).write_bytes(wire)
            rec["wire_bytes"] = len(wire)
        except Exception as e:
            logger.warning(f"recap wire encode failed ({e}); it will be encoded on serve")
    _path(state.campaign).write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    logger.info(f"recap for {state.campaign!r} session {opens} written "
                f"({len(text)} chars, {rec.get('audio_bytes', 0)} bytes of audio)")
    return rec


async def _write(state) -> str:
    """Opus writes it. Nobody is waiting, so this is the one place the best model is free."""
    import asyncio
    import campaign as campaign_mod
    import config
    import transcript
    brief = campaign_mod.memory_brief(state)
    tail = transcript.recent(state.campaign, 60) or ""
    prompt = (f"{SYS}\n\n--- what the campaign knows ---\n{brief}\n\n"
              f"--- how last session actually went, in order ---\n{tail}\n\n"
              "Write the 'previously on' now. Nothing else, no preamble.")
    try:
        # The prompt goes down STDIN, not argv. Windows caps a command line at about 32k and
        # the continuity brief alone is 31k, so passing it as an argument failed outright with
        # "[WinError 206] The filename or extension is too long" -- and failed silently, since
        # the live path quietly covers for it. stdin has no such limit.
        proc = await asyncio.create_subprocess_exec(
            config.CLAUDE_BIN, "-p", "--model", config.RECAP_CLAUDE_MODEL,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=300)
    except Exception as e:
        logger.warning(f"recap could not be written ({e}); the live path will handle it")
        return ""
    text = (out or b"").decode("utf-8", "replace").strip()
    if not usable(text):
        logger.warning(f"recap text looked wrong ({text[:80]!r}); discarding it")
        return ""
    return text


def usable(text: str) -> bool:
    """Is this actually a recap, or is it the model talking about itself?

    A pure check on purpose. The thing being guarded against -- an apology, a refusal, a request
    for clarification -- would become the FIRST thing the table hears, and stay that way for a
    whole session, so it has to be testable without a model in the loop.
    """
    t = (text or "").strip()
    if len(t) < 80:
        return False
    return not t.lower().startswith(
        ("i can", "i cannot", "i'm sorry", "im sorry", "i am sorry", "i don't", "i do not",
         "sorry", "as an ai", "i'd be happy", "i would need", "could you"))


async def _voice(text: str) -> bytes:
    """Rendered once, here, so the table never waits on synthesis for the opening line."""
    import tts
    try:
        await tts.warmup()
        parts = []
        for line in sentences(text):
            if a := await tts.synth(line):
                parts.append(a)
        return join_wavs(parts)
    except Exception as e:
        logger.warning(f"recap audio failed ({e}); the text will be spoken live instead")
        return b""


def sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def join_wavs(parts: list) -> bytes:
    """One WAV from many, so the opening plays as a single clip rather than a queue of them."""
    import io
    import wave
    if not parts:
        return b""
    frames, params = [], None
    for p in parts:
        try:
            with wave.open(io.BytesIO(p)) as w:
                params = params or w.getparams()
                frames.append(w.readframes(w.getnframes()))
        except Exception:
            continue                       # one bad clip must not lose the whole recap
    if not frames or params is None:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(b"".join(frames))
    return buf.getvalue()
