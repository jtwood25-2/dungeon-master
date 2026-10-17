"""Text-to-speech -> audio bytes for the phone.

Two engines, chosen by TTS_ENGINE:
  kokoro  (default) -- Kokoro-82M running LOCALLY on this machine. More natural than edge-tts,
                       54 voices, and no network round trip: synthesis is ~0.3x realtime and,
                       more importantly, CONSISTENT. edge-tts swung between 0.5s and 2.1s per
                       sentence purely on network luck, which is what made the DM feel laggy.
  edge    -- Microsoft edge-tts. Free, no key, but cloud and flat-sounding. Kept as a fallback
             so the DM still speaks if the local model is missing.

Everything the DM says passes through synth(), so this is also where written text is turned
into SPEAKABLE text. The DM writes for a reader -- markdown emphasis, "AC 15", "2d6" -- and a
voice engine reads those literally ("asterisk", "ack", "two-dee-six").
"""
import asyncio
import io
import logging
import re
import wave
from collections import OrderedDict

import numpy as np

import config

logger = logging.getLogger("tts")

# read as words, not letters -- these are the abbreviations the DM uses constantly
_SAY = [
    (re.compile(r"\bAC\b"), "armor class"),
    (re.compile(r"\bDC\b"), "difficulty class"),
    (re.compile(r"\bHP\b"), "hit points"),
    (re.compile(r"\bXP\b"), "experience"),
    (re.compile(r"\bNPCs\b"), "characters"),
    (re.compile(r"\bPCs?\b"), "characters"),
    (re.compile(r"\bDM\b"), "Dungeon Master"),
    (re.compile(r"\bd(\d+)\b"), r"d \1"),
]
_DICE = re.compile(r"\b(\d*)d(\d+)\b")
# Hesitation sounds get SPELLED OUT by the voice model — "Mm" came out as "em em". "Hmm" is a
# real word to it and hums correctly, so normalise the family onto that.
_HUM = re.compile(r"\b(?:m+h?m+|hm+|mhm+|uh+huh)\b", re.I)
_EMPH = [
    (re.compile(r"\*\*\*(.+?)\*\*\*", re.S), r"\1"),
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),
    (re.compile(r"\*(.+?)\*", re.S), r"\1"),
    (re.compile(r"__(.+?)__", re.S), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9])_(.+?)_(?![A-Za-z0-9])", re.S), r"\1"),
    (re.compile(r"~~(.+?)~~", re.S), r"\1"),
    (re.compile(r"`(.+?)`", re.S), r"\1"),
]


# A lone capital letter used as a designator -- "Scout A", "Wraith B" -- is read by espeak as
# the indefinite article, so the table heard "Scout uh" and could not tell Scout A from Scout
# B at all. Spelling it the way it sounds is the only thing the engine reliably obeys.
#
# Deliberately narrow: it fires only after a capitalised word, so "a sword" and the pronoun
# "I" are untouched, and only for the letters a DM actually labels monsters with.
_SAID = {"A": "Ay", "B": "Bee", "C": "See", "D": "Dee", "E": "Ee", "F": "Eff", "G": "Gee",
         "H": "Aitch", "J": "Jay", "K": "Kay", "P": "Pee", "Q": "Cue", "R": "Ar",
         "T": "Tee", "V": "Vee", "X": "Ex", "Y": "Why", "Z": "Zed"}
_DESIGNATOR = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z])(?=\b(?![a-z]))")


def _spell_designators(t: str) -> str:
    return _DESIGNATOR.sub(
        lambda m: f"{m.group(1)} {_SAID.get(m.group(2), m.group(2))}", t)


def speakable(text: str) -> str:
    """Turn DM prose into something a voice engine reads correctly."""
    t = text.strip()
    for pat, rep in _EMPH:                       # *creaks* -> creaks (not "asterisk creaks")
        t = pat.sub(rep, t)
    t = re.sub(r"[*`_~#>|]", " ", t)             # any stray markers left over
    t = re.sub(r"^\s*[-–—•]\s+", "", t)          # list bullets
    t = _DICE.sub(lambda m: f"{m.group(1) or '1'} d {m.group(2)}", t)
    t = _HUM.sub("Hmm", t)
    t = _spell_designators(t)
    for pat, rep in _SAY:
        t = pat.sub(rep, t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _wav(samples: np.ndarray, rate: int) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ------------------------------------------------------------------ Kokoro (local)
_kokoro = None
_kokoro_failed = False
# a chatterbox narrator id is not a kokoro voice; NPCs must still resolve
_KOKORO_OK: set = set()


def _enable_onnx_cuda() -> None:
    """Let kokoro run on the GPU. It has been on the CPU this whole time.

    Two separate faults kept an 82M model at 3.66s a line on a card that does 95 TFLOP/s:

    1. kokoro-onnx hardcodes CPUExecutionProvider and only switches if
       importlib.util.find_spec("onnxruntime-gpu") succeeds -- but that is a PACKAGE name
       with a hyphen, not an importable module, so the check can never pass. ONNX_PROVIDER
       is the documented way round it.
    2. onnxruntime-gpu ships against a specific CUDA major. The current release wants CUDA 13
       (cublasLt64_13.dll) while torch here is CUDA 12.8, so the provider loaded and then
       failed at session creation. Torch already ships the matching CUDA 12 runtime, so
       putting its lib directory on the DLL search path satisfies it without a second install.

    Best-effort throughout: if anything here is missing, kokoro falls back to CPU exactly as
    before rather than failing to load at all.
    """
    import os
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return
        try:
            import torch
            lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(lib):
                os.add_dll_directory(lib)
        except Exception:
            pass
        os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
    except Exception:
        return


def _keep_one_espeak() -> None:
    """DISABLED -- this broke narration live, mid-session, and the 66ms was not worth it.

    A persistent EspeakBackend carries a words-mismatch checker that accumulates across calls.
    The module-level phonemizer.phonemize() builds a fresh backend every time, so that state
    resets every line and the check never trips. Kept warm, it accumulates until it throws:

        RuntimeError: number of lines in input and output must be equal,
                      we have: input=7, output=11

    which surfaced as a turn that produced no audio at all -- "silent turn: no reason
    reported" -- at the opening of a real session with seven people waiting. Rebuilding the
    backend per line costs 66ms. Being unable to speak costs the evening.

    Left here rather than deleted so the next person to notice the 66ms finds out why it was
    taken and put back. Re-enabling it needs words_mismatch handling that survives reuse, and
    a test that phonemises a few hundred varied lines through one backend.

    Build the espeak backend ONCE instead of rebuilding it for every line.

    kokoro_onnx phonemises through phonemizer.phonemize(), the module-level convenience
    function, and that constructs a fresh EspeakBackend on every single call. Measured warm on
    this machine: 66ms a line through the convenience function against ~0ms through a backend
    that is kept, for byte-identical output -- so it is 66ms of pure setup per sentence, and a
    four-sentence reply pays it four times before any audio exists.

    Patched at the point kokoro reaches for it rather than by editing the package, and only
    after Kokoro() has run, because that is what points phonemizer at the bundled espeak
    library in the first place. Entirely best-effort: if anything here is not shaped the way it
    was, the original function is left alone and lines are phonemised exactly as before.
    """
    try:
        from kokoro_onnx import tokenizer as _tok
        from phonemizer.backend import EspeakBackend
        cache = {}

        def phonemize(text, lang="en-us", **kw):
            be = cache.get(lang)
            if be is None:
                be = cache[lang] = EspeakBackend(lang, preserve_punctuation=True,
                                                 with_stress=True)
            out = be.phonemize([text])
            return out[0] if out else ""

        _tok.phonemizer.phonemize(" ", "en-us", preserve_punctuation=True, with_stress=True)
        before = _tok.phonemizer.phonemize("the quick brown fox", "en-us",
                                           preserve_punctuation=True, with_stress=True)
        after = phonemize("the quick brown fox")
        if before.strip() != after.strip():
            logger.warning("persistent espeak disagreed with the per-call one; leaving it alone")
            return
        _tok.phonemizer.phonemize = phonemize
        logger.info("espeak backend kept warm (saves ~66ms a line)")
    except Exception as e:
        logger.debug(f"could not keep one espeak backend ({e}); phonemising per call")


def _get_kokoro():
    global _kokoro, _kokoro_failed
    if _kokoro is not None or _kokoro_failed:
        return _kokoro
    try:
        _enable_onnx_cuda()
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(str(config.KOKORO_MODEL), str(config.KOKORO_VOICES))
        _KOKORO_OK.update(_kokoro.get_voices())
        # SAY WHICH DEVICE. This ran on the CPU for weeks at 3.66s a line instead of 0.43s,
        # and nothing anywhere reported it -- the only symptom was "the DM feels slow". It
        # regressed again within a day, because installing almost anything pulls the CPU-only
        # `onnxruntime` back in and it shadows `onnxruntime-gpu` in the same module directory.
        # A silent 8x slowdown must never be silent twice.
        used = list(getattr(_kokoro, "sess", None).get_providers()) if getattr(_kokoro, "sess", None) else []
        on_gpu = any("CUDA" in p or "Tensorrt" in p for p in used)
        logger.info(f"kokoro loaded ({len(_kokoro.get_voices())} voices, "
                    f"{'GPU' if on_gpu else 'CPU'}: {used[0] if used else 'unknown'})")
        if not on_gpu:
            logger.warning("kokoro is running on the CPU -- roughly 8x slower per line. "
                           "Fix: pip uninstall onnxruntime && pip install onnxruntime-gpu==1.24.4")
    except Exception as e:
        _kokoro_failed = True
        logger.warning(f"kokoro unavailable ({e}) -- falling back to edge-tts")
    return _kokoro


async def _synth_kokoro(text: str, voice: str) -> bytes:
    import asyncio
    k = _get_kokoro()
    if k is None:
        return await _synth_edge(text, None)
    def run():
        samples, rate = k.create(text, voice=voice, speed=config.TTS_SPEED, lang="en-us")
        return _wav(np.asarray(samples, dtype=np.float32), rate)
    return await asyncio.to_thread(run)


# ------------------------------------------------------------------ Chatterbox (best quality)
_chatter_down_until = 0.0


_chatter_fails = 0
_chatter_gate: "asyncio.Lock | None" = None


def _voice_timeout(text: str) -> float:
    """Scale with the line. Generation is roughly 200-300ms/word when the GPU is not starved,
    so a flat 30s was fine for a greeting and impossible for a sentence of narration — the
    long descriptive lines that matter most were exactly the ones that timed out."""
    words = max(1, len(text.split()))
    return min(150.0, max(float(config.VOICE_TIMEOUT), 10.0 + 1.6 * words))


async def _synth_chatterbox(text: str, voice: str) -> bytes:
    """Ask the resident voice server, falling back to Kokoro rather than going silent.

    The fallback used to be a hair trigger: ONE slow line disabled the good narrator for 30
    seconds, which is why the voice audibly changed mid-scene and stayed changed. A timeout
    now only counts against it, and it takes three in a row to back off; a refused connection
    means the service really is down and backs off at once.
    """
    global _chatter_down_until, _chatter_fails, _chatter_gate
    import time as _t
    import httpx
    if _t.monotonic() < _chatter_down_until:
        return await _synth_kokoro(text, voice if voice in _KOKORO_OK else "am_michael")
    if _chatter_gate is None:
        _chatter_gate = asyncio.Lock()
    try:
        # One GPU, and the server serialises on a lock of its own. Firing three concurrent
        # requests at it did not make anything faster — it just meant queue time counted
        # against each request's timeout, so the later lines in a paragraph timed out.
        async with _chatter_gate:
            async with httpx.AsyncClient(timeout=_voice_timeout(text)) as client:
                r = await client.post(f"{config.VOICE_URL}/speak", json={
                    "text": text, "voice": voice,
                    "exaggeration": config.VOICE_EXAGGERATION,
                    "cfg_weight": config.VOICE_CFG,
                })
                r.raise_for_status()
        _chatter_fails = 0
        return r.content
    except httpx.ConnectError as e:                  # nothing listening: really is down
        _chatter_down_until = _t.monotonic() + 30
        logger.warning(f"voice server not running ({e}); using kokoro for 30s")
    except Exception as e:
        _chatter_fails += 1
        if _chatter_fails >= 3:
            _chatter_down_until = _t.monotonic() + 30
            _chatter_fails = 0
            logger.warning(f"voice server failed 3x ({e}); using kokoro for 30s")
        else:
            logger.warning(f"voice line fell back to kokoro ({type(e).__name__}); "
                           f"{_chatter_fails}/3 before backing off")
    return await _synth_kokoro(text, voice if voice in _KOKORO_OK else "am_michael")


# ------------------------------------------------------------------ edge-tts (cloud fallback)
async def _synth_edge(text: str, voice: str | None) -> bytes:
    import edge_tts
    audio = bytearray()
    comm = edge_tts.Communicate(text, voice or config.EDGE_VOICE,
                                rate=config.TTS_RATE, pitch=config.TTS_PITCH)
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


# Deterministic answers repeat all session ("Plus three.", "Fifteen.", "11 of 11.") and
# synthesis is now the largest cost on those turns, so the second time is free.
_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_CACHE_MAX = 256
_CACHE_MAX_CHARS = 120          # short answers only; narration rarely repeats verbatim


async def synth(text: str, voice: str | None = None) -> bytes:
    """Return audio bytes (WAV from kokoro, MP3 from edge) for one line. The client sniffs the
    container, so either engine can be swapped in without touching the phone."""
    spoken = speakable(text)
    if not spoken:
        return b""
    v = voice or config.TTS_VOICE
    key = (config.TTS_ENGINE, v, spoken)
    cacheable = len(spoken) <= _CACHE_MAX_CHARS
    if cacheable and (hit := _CACHE.get(key)) is not None:
        _CACHE.move_to_end(key)
        return hit
    if config.TTS_ENGINE == "chatterbox":
        # every character gets chatterbox quality: it clones from that voice's reference clip
        audio = await _synth_chatterbox(spoken, v)
    elif config.TTS_ENGINE == "kokoro":
        audio = await _synth_kokoro(spoken, v if v in _KOKORO_OK else "am_michael")
    else:
        audio = await _synth_edge(spoken, voice)
    if cacheable and audio:
        _CACHE[key] = audio
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return audio


# Enough lines, of a realistic length, in the voice actually used. One short line was not
# warm -- it was the first of several slow ones. Measured after a fresh load: the opening
# sentences cost about 1.9s each and only then settle to ~230ms, which is a 700% difference
# sitting exactly where the table is listening hardest. It is also why a real session opened
# with 6.4 seconds of synthesis: the server had just restarted, so the very first narration
# paid the whole settling cost live, in front of everybody.
_WARM_LINES = [
    "Ready.",
    "The tent flap gives against your hand, stiff with old rain.",
    "Three of them stand across the passage, blocking the way down, and one has already spoken.",
]


async def warmup():
    """Run the engine until it is actually fast, not merely loaded."""
    if config.TTS_ENGINE not in ("kokoro", "chatterbox"):
        return
    import time
    t0 = time.perf_counter()
    try:
        # In the NARRATOR's voice. Warming am_michael and then narrating in bm_lewis re-paid
        # part of the cost on the first real line, because the voice embedding is part of what
        # settles.
        v = config.TTS_VOICE if config.TTS_VOICE in _KOKORO_OK else "am_michael"
        for line in _WARM_LINES:
            await _synth_kokoro(line, v)
        logger.info(f"tts warm ({(time.perf_counter() - t0) * 1000:.0f}ms, {len(_WARM_LINES)} lines)")
    except Exception as e:
        logger.warning(f"tts warmup skipped: {e}")


# ------------------------------------------------------------------ wire format
# Chatterbox returns 24kHz 16-bit WAV: ~47 KB per second of speech, so a six-sentence intro
# is ~1.5MB pushed to a phone over the tunnel. That is a lot of bursty binary for a mobile
# link to hold open, and the socket dropping mid-intro loses the whole turn. MP3 is ~6x
# smaller for speech at this bitrate with no audible difference, and the client already
# sniffs the container, so either format plays.
_MP3_BITRATE = int(getattr(config, "AUDIO_BITRATE", 64000))


def _to_mp3(wav_bytes: bytes) -> bytes:
    try:
        import av
    except ImportError:
        return wav_bytes                             # no encoder: send it uncompressed
    try:
        w = wave.open(io.BytesIO(wav_bytes))
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            return wav_bytes
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if pcm.size == 0:
            return wav_bytes
        out = io.BytesIO()
        c = av.open(out, "w", format="mp3")
        st = c.add_stream("mp3", rate=sr)
        st.bit_rate = _MP3_BITRATE
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = sr
        for p in st.encode(frame):
            c.mux(p)
        for p in st.encode(None):
            c.mux(p)
        c.close()
        return out.getvalue() or wav_bytes
    except Exception as e:
        logger.warning(f"mp3 encode failed ({e}); sending wav")
        return wav_bytes


async def for_wire(audio: bytes) -> bytes:
    """Compress on the way to the phone. Off-thread so it never blocks the event loop."""
    if not audio or audio[:3] == b"ID3" or not audio.startswith(b"RIFF"):
        return audio                                 # already compressed
    return await asyncio.to_thread(_to_mp3, audio)
