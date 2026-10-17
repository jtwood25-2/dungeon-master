"""Settings, loaded from environment (.env). Keep secrets out of code."""
import os
import shutil
from pathlib import Path


def _load_dotenv():
    p = Path(__file__).parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# DM backend: "claude_code" drives Claude Code on your Pro subscription (no API billing);
# "api" uses the Anthropic API key (fastest, native tool-use). Swappable at any time.
# "hybrid" = a local Ollama model answers instant mechanical lookups, Claude drives the story.
# Also: "claude_code" (Pro, no billing), "api" (Anthropic key), "ollama" (fully local).
DM_BACKEND     = os.environ.get("DM_BACKEND", "hybrid")
# 127.0.0.1, NOT localhost. On Windows "localhost" resolves to ::1 first, ollama listens on
# IPv4, and the connection sits for two full seconds before falling back -- measured at 2068ms
# against 21ms for the same request to the same server by address. Every local model call on
# the live path was paying that.
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# VRAM BUDGET (16GB card). Whisper large-v3-turbo + Chatterbox + Kokoro sit at ~8.5GB, so the
# quick-answer model gets what is left. The 14B took 9.5GB and left 382MB free: Chatterbox then
# thrashed and a 30-word line took 24-54s instead of 6s, blowing the TTS timeout and dropping
# the narrator to Kokoro mid-scene. The 7B leaves ~2.8GB of headroom and the voice stays up.
# TWO TIERS. Sonnet WRITES the spoken text on the critical path, because the player is sitting
# in silence waiting for it (4.0s to first token against opus's 7.3-9.2s). Opus DIRECTS: it
# runs between turns, never in front of the player, and decides where the story is going,
# what stays canon, and what the next encounter and map should be. Depth where there is time
# for it, speed where there is not.
DIRECTOR_CLAUDE_MODEL = os.environ.get("DIRECTOR_CLAUDE_MODEL", "opus")
# The "previously on" is written when a session ENDS, with nobody waiting on it, so the best
# model is free here. It is the first thing the table hears and it sets the scene for the
# whole evening -- the one piece of narration worth spending Opus on.
RECAP_CLAUDE_MODEL = os.environ.get("RECAP_CLAUDE_MODEL", "opus")
# Scenes the party is walking into are written ahead by Opus, once per beat, with nobody
# waiting -- see prepared.py. The arc was already Opus-written; this fills in what is
# actually THERE, so Sonnet interpolates from facts instead of improvising a scene cold.
PREPARE_CLAUDE_MODEL = os.environ.get("PREPARE_CLAUDE_MODEL", "opus")
# Beats to prepare ahead of the party.
PREPARE_AHEAD = int(os.environ.get("PREPARE_AHEAD", "2"))
# Prep work: recaps, room names, the campaign plan. This was spawned with NO --model at all,
# so it silently inherited whatever the CLI default happened to be — the exact hazard called
# out for DM_CLAUDE_MODEL below, and it meant recaps ran on an uncontrolled (possibly Opus)
# model. Sonnet summarises a transcript perfectly well; the campaign PLAN asks for opus
# explicitly, because that one is deep and happens once.
PLANNER_CLAUDE_MODEL = os.environ.get("PLANNER_CLAUDE_MODEL", "sonnet")
# One session spent 119 Opus calls on background direction. It is off the critical path,
# so the cost was invisible while being paid every turn. Cap it, and refuse to re-run a
# specialist for the same event inside the debounce window -- the log had showrunner
# firing twice in the same second for two near-identical notes.
DIRECTOR_BUDGET = int(os.environ.get("DIRECTOR_BUDGET", "40"))
DIRECTOR_DEBOUNCE_S = float(os.environ.get("DIRECTOR_DEBOUNCE_S", "90"))
# The 7B invented numbers: told "6 damage to Thorn" it said "4 damage, the goblin is at 12".
# Measured over the combat-narration cases in tools/local_bench.py it was 80% faithful against
# the 14B's 100%, for 21ms of latency saved -- a bad trade when the whole point of the local
# path is saying numbers the engine has already settled. It fits: our own models hold 5.2GB of
# the 16GB card and the 14B takes 9.0GB.
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct")
# Narrate resolved combat locally. 37ms to first token against 10-30s through Claude Code,
# because the engine has already decided everything and only the wording is left.
LOCAL_COMBAT   = os.environ.get("LOCAL_COMBAT", "1") not in ("0", "false", "no")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "2h")   # keep it resident all session

# -- story drift: when the big model is actually needed ------------------------
EMBED_MODEL    = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT  = float(os.environ.get("EMBED_TIMEOUT", "5"))
# loading nomic into VRAM is a one-off ~7s; scoring afterwards is ~20ms
EMBED_LOAD_TIMEOUT = float(os.environ.get("EMBED_LOAD_TIMEOUT", "45"))
# Above this, the party has left the planned arc and a showrunner pass is worth paying for.
# tools/drift_calibrate.py: on-arc windows score a median 0.336, off-arc 0.452, and 0.39
# separates them with 86.6% accuracy. The negatives there are a DIFFERENT SHOW, so they
# differ in voice as well as plot and the real-world separation is likely softer than
# that -- which is why DIRECTOR_ALWAYS still runs the structural events ungated.
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.39"))
# Events that always run the director regardless of drift: the arc has to be built before it
# can be departed from, and a finished fight really does change the world.
DIRECTOR_ALWAYS = tuple(x for x in os.environ.get(
    "DIRECTOR_ALWAYS", "session_start,combat_end").split(",") if x)

# Which model the DM runs on. Spawned without this it silently inherited whatever the CLI's
# default happened to be, so the DM could change under you when you changed your own /model.
DM_CLAUDE_MODEL = os.environ.get("DM_CLAUDE_MODEL", "sonnet")
# The DM's thinking budget. Traced from the CLI's own event stream, extended thinking is where
# the wait actually goes: one turn emitted seven thinking_tokens events between 1.7s and 9.9s
# and did not say its first word until 11.1s. It is also the variance -- another turn thought
# once and answered in 3.8s -- and variance at a table is worse than a slow average, because
# nobody can tell a thinking DM from a broken one.
#
# Measured over ten paired questions, same session, same state, interleaved:
#   thinking on   median 2853  mean 3168  p90 4609  max 7386
#   thinking off  median 2291  mean 2518  p90 3690  max 4112
# Every statistic improves and the WORST case improves most, which is the one a table feels.
#
# Off is right for this job specifically: the DM is narrating a scene whose numbers have
# already been settled by the deterministic combat engine, so there is very little left to
# reason about before speaking. The director is deliberately NOT given this -- it runs Opus in
# the background, off the critical path, where thinking is worth having.
DM_THINKING_TOKENS = os.environ.get("DM_THINKING_TOKENS", "0")

def _find_claude() -> str:
    """Locate the Claude CLI, newest version first.

    This was pinned to an exact extension path (…claude-code-2.1.210-win32-x64…). VS Code
    updates the extension and deletes the old directory, so the path went stale on its own
    and every spawn died with WinError 2 -- silently, because backend/director/planner all
    swallow the failure and just produce nothing. The DM went mute with no error shown to
    the player. Resolve it at import instead of trusting a version number to stay put.
    """
    if pinned := os.environ.get("CLAUDE_BIN"):
        return pinned                                   # explicit override always wins
    ext = Path.home() / ".vscode" / "extensions"
    found = []
    for d in ext.glob("anthropic.claude-code-*-win32-x64"):
        exe = d / "resources" / "native-binary" / "claude.exe"
        if not exe.exists():
            continue
        # sort numerically: "2.1.226" must beat "2.1.9", which a string sort gets wrong
        digits = d.name.removeprefix("anthropic.claude-code-").removesuffix("-win32-x64")
        try:
            found.append((tuple(int(p) for p in digits.split(".")), str(exe)))
        except ValueError:
            continue
    if found:
        return max(found)[1]
    return shutil.which("claude") or "claude"


CLAUDE_BIN     = _find_claude()
DM_CAMPAIGN    = os.environ.get("DM_CAMPAIGN", "default")   # which campaign the server runs

# How hard to filter a ROOM full of people before the DM hears them.
#   strict  allow-list: only clear actions, rolls, game questions and direct address.
#   loose   deny-list: everything except what is plainly said to another player.
#   off     the DM hears the room.
# Measured at a real seven-player table, "strict" rejected almost everything said: real
# play does not announce itself as an action, and the cost of a miss is not "they say it
# again" -- it is a table that concludes the thing cannot hear them. A DM that occasionally
# answers something not meant for it is recoverable; one that ignores the room is not.
# Default "off" after a live session. The local classifier it gated on rejected a player
# correcting the turn order -- "It should be Gus of The Slippery Rock after Scout A" --
# as table talk, twice, and the table sat there thinking the DM had stalled. A gate that
# drops real play is worse than a DM that occasionally answers something not meant for
# it: one is recoverable in a sentence, the other reads as broken software.
TABLE_GATE     = os.environ.get("TABLE_GATE", "off").strip().lower()
DM_MODEL       = os.environ.get("DM_MODEL", "claude-haiku-4-5-20251001")
WHISPER_MODEL  = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
# How many utterances Whisper may decode AT ONCE. A table with a six-microphone speakerphone
# on it produces overlapping speech constantly, and the default of one does not merely queue
# them -- it is actively worse than doing them in turn. Measured on four real spoken clips:
#   num_workers=1   four at once 2868ms  (four in sequence: 1753ms -- 0.61x, i.e. a LOSS)
#   num_workers=2   four at once 1405ms  (1.76x)
#   num_workers=4   four at once 1094ms  (1.59x, and the lowest wall-clock)
# This is what made three people speaking together cost 2.3s, 3.4s and 3.5s of transcription
# against a 263ms median for one person alone.
WHISPER_WORKERS = int(os.environ.get("WHISPER_WORKERS", "4"))
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_BEAM   = int(os.environ.get("WHISPER_BEAM", "5"))   # 1 = greedy/fastest, 5 = accurate
# Voice engine: "kokoro" runs locally on this machine (natural, fast, no key, no network);
# "edge" is the Microsoft cloud fallback.
# Default follows what is actually INSTALLED. Chatterbox was being started by run.py, sitting
# ready on 8123, and never used — because this defaulted to kokoro and nothing routed to it,
# so the better voice was running and inaudible. If its venv isn't there, kokoro is correct.
_HAS_VOICE_SERVER = (Path(__file__).parent.parent / ".venv-tts" / "Scripts" / "python.exe").exists()
# Kokoro by default, chosen on measurements rather than on which model sounds best in the
# abstract. Same three lines, cache defeated, on this GPU:
#     engine       fixed   per 100 chars   realtime factor   short line
#     kokoro       0.27s   0.20s           0.058             0.33s
#     piper        0.04s   0.54s           0.107             0.18s
#     chatterbox   2.25s   2.42s           0.628             2.90s
# Chatterbox spends 2.25s before it says anything, and that fixed cost IS the silence the
# player sits in. It is autoregressive at batch size 1, so the cost is kernel-launch bound and
# no amount of precision tuning moves it (measured: bf16 0.87x, tf32 0.93x -- both slower).
# Set TTS_ENGINE=chatterbox to get the cloned narrator voice back at ~9x the latency.
TTS_ENGINE     = os.environ.get("TTS_ENGINE", "kokoro")
# "chatterbox" routes the NARRATOR to the resident voice server (voice_server.py, its own
# Python 3.12 venv) for much better quality; NPCs stay on kokoro's distinct voices. Falls
# back to kokoro automatically if that service isn't running.
# Looping background soundscape under every scene. Off for a while, because on a phone
# speaker anything loud enough to notice was loud enough to sit on top of the narrator.
# A speakerphone on the table has the headroom, and the beds themselves are now quieter and
# simpler -- filtered noise, no held tones, four moods rather than seven (see ambience.py).
# AMBIENCE=0 to switch the loops back off; one-shot [SFX] are separate and unaffected.
# Keep every uplink sample plus an index aligning it to what was decided. ~115MB/hour,
# and it is the only way to tune endpointing or measure the speaker model against real
# table audio rather than against a description of it. RECORD_AUDIO=0 to switch off.
RECORD_AUDIO   = os.environ.get("RECORD_AUDIO", "1").strip().lower() not in ("0", "false", "off", "no")
AMBIENCE       = os.environ.get("AMBIENCE", "1").strip().lower() not in ("0", "false", "off", "no", "")
VOICE_URL      = os.environ.get("VOICE_URL", "http://127.0.0.1:8123")
# Synthesis costs ~4.2s per call plus ~0.03s per character, so batching later sentences
# into one call is worth several seconds on a multi-sentence answer. Kept modest: too
# large and a long tail of narration arrives as one late block instead of flowing.
TTS_BATCH_CHARS = int(os.environ.get("TTS_BATCH_CHARS", "220"))
VOICE_TIMEOUT  = float(os.environ.get("VOICE_TIMEOUT", "30"))
VOICE_EXAGGERATION = float(os.environ.get("VOICE_EXAGGERATION", "0.5"))
VOICE_CFG      = float(os.environ.get("VOICE_CFG", "0.5"))
_MODELS        = Path(__file__).parent.parent / "state" / "models" / "kokoro"
KOKORO_MODEL   = os.environ.get("KOKORO_MODEL", str(_MODELS / "kokoro-v1.0.onnx"))
KOKORO_VOICES  = os.environ.get("KOKORO_VOICES", str(_MODELS / "voices-v1.0.bin"))
TTS_VOICE      = os.environ.get("TTS_VOICE", "am_michael")      # kokoro narrator
TTS_SPEED      = float(os.environ.get("TTS_SPEED", "1.0"))      # kokoro rate (1.0 = natural)
# edge-tts only. Speed costs intelligibility: past about +8% the neural voices start slurring
# consonants together, which is worse than being a second slower.
EDGE_VOICE     = os.environ.get("EDGE_VOICE", "en-US-AndrewMultilingualNeural")
TTS_RATE       = os.environ.get("TTS_RATE", "+2%")
TTS_PITCH      = os.environ.get("TTS_PITCH", "+0Hz")

# Hands-free voice detection. Tune these if it mishears your room:
#   VAD_SILENCE_MS  higher = waits longer before answering (fewer mid-sentence cut-offs)
#   VAD_NOISE_MULT  lower  = more sensitive (catches quiet speech; may trigger on noise)
#   VAD_CEILING     the threshold can never climb above this, so a loud moment can't
#                   leave the mic permanently deaf -- that's the "it never started" failure
# 1100ms was tuned for one person on a phone, thinking out loud into a microphone held
# close. At a seven-player table it cut real play off mid-sentence -- 9 utterances in one
# session, and they were the ones that mattered: "I want to cast Create Bonfire, that way
# I get two-", "Do I have any spells that will help me with-". People at a table pause to
# look at a sheet, to check with a neighbour, to find the word. The cost of waiting longer
# is a slightly later answer; the cost of cutting in is answering half a thought.
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "1600"))
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "220"))
VAD_NOISE_MULT = float(os.environ.get("VAD_NOISE_MULT", "2.2"))
VAD_CEILING    = float(os.environ.get("VAD_CEILING", "0.05"))
# talking OVER the DM: raise if it interrupts itself, lower if barging in is too hard
VAD_BARGE_FLOOR = float(os.environ.get("VAD_BARGE_FLOOR", "0.022"))

# May a player cut the DM off by speaking over it?
#
# Off at a real table. Barge-in was built for one person on a phone, where anything the
# microphone hears while the DM talks is almost certainly aimed at it. A room of seven is
# the opposite: people react to narration WHILE it is being narrated -- that is what a
# table does -- and every laugh and aside stopped the DM mid-sentence. The table then sees
# a DM that keeps trailing off and cannot work out why, which reads as confusion rather
# than as a feature working exactly as designed.
#
# The stop button still works, and is the right way to cut it off: it is deliberate.
ALLOW_BARGE_IN = os.environ.get("ALLOW_BARGE_IN", "0").strip().lower() not in ("0", "false", "off", "no", "")

# Does another player acting end the turn of whoever the engine is waiting on?
#
# Added to stop a fight waiting forever on somebody who had put their phone down, and the
# table immediately asked for it gone: "you should really confirm or have a button for people
# to end turn and not just do it auto". It was defensible reasoning -- a different player
# acting is observable fact rather than a guess at intent -- and it still took turns off
# people who were mid-thought. The explicit phrase and the End my turn button cover the same
# ground without deciding anything on the players' behalf.
AUTO_HANDOVER  = os.environ.get("AUTO_HANDOVER", "0").strip().lower() not in ("0", "false", "off", "no", "")
# Extra patience when the words say they're mid-sentence ("...and then I") rather than done.
# Costs nothing when they ARE done: a finished thought is answered without waiting at all.
VAD_HOLD_MS    = float(os.environ.get("VAD_HOLD_MS", "1600"))
# Feedback is dictated in longer, more considered stretches than play, and half a note is
# worse than none — the point being made is usually in the second half.
VAD_FEEDBACK_HOLD_MS = float(os.environ.get("VAD_FEEDBACK_HOLD_MS", "3500"))
HOST           = os.environ.get("HOST", "0.0.0.0")
PORT           = int(os.environ.get("PORT", "8080"))
SAMPLE_RATE    = 16000   # whisper's native rate; the browser downsamples to this

# HTTPS: iOS blocks mic access (getUserMedia) over plain http on a LAN IP, so the phone
# needs https. If certs/cert.pem + certs/key.pem exist, we serve https automatically.
_CERTS = Path(__file__).parent.parent / "certs"
SSL_CERT = str(_CERTS / "cert.pem") if (_CERTS / "cert.pem").exists() else None
SSL_KEY  = str(_CERTS / "key.pem")  if (_CERTS / "key.pem").exists()  else None
