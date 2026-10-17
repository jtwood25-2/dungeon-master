"""Speech-to-text: local faster-whisper on the GPU. Swap here for a cloud STT later."""
import logging
import numpy as np
import config

logger = logging.getLogger("stt")
_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        try:
            # num_workers is what lets a table talk over itself. See config.WHISPER_WORKERS:
            # at the default of one, four simultaneous utterances took LONGER than four
            # sequential ones, because the calls pile up inside CTranslate2 rather than queueing
            # politely outside it.
            _model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                                  compute_type="float16", num_workers=config.WHISPER_WORKERS)
            logger.info(f"whisper {config.WHISPER_MODEL} on {config.WHISPER_DEVICE} "
                        f"({config.WHISPER_WORKERS} workers)")
        except Exception as e:
            logger.warning(f"GPU load failed ({e}) -- falling back to CPU int8")
            _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8",
                                  num_workers=config.WHISPER_WORKERS)
    return _model


def transcribe(pcm_f32: np.ndarray, vocab: str = "") -> str:
    """pcm_f32: mono float32 in [-1, 1] at config.SAMPLE_RATE. Returns text.

    `vocab` primes the decoder with this campaign's proper nouns (party, NPCs, places).
    Whisper has never seen "Oktah of the Wood" and will happily emit "October the Wood";
    naming it in the prompt is what makes invented names transcribe correctly.
    """
    if pcm_f32.size < config.SAMPLE_RATE // 4:   # < 0.25s -> ignore
        return ""
    segments, _ = _get_model().transcribe(
        pcm_f32, language="en",
        beam_size=config.WHISPER_BEAM,
        initial_prompt=vocab or None,
        # each utterance is independent -- carrying decoder context across turns makes
        # whisper echo the previous turn's phrasing back into this one
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4],             # retry hotter only if the greedy pass is junk
        no_speech_threshold=0.6,
        vad_filter=True)
    return " ".join(s.text for s in segments).strip()


def warmup():
    try:
        transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
    except Exception as e:
        logger.warning(f"warmup skipped: {e}")


# ---------------------------------------------------------------- uplink codec
# The phone streams the mic CONTINUOUSLY (that is what lets a player answer the instant the
# DM stops, or talk over it), so raw 16-bit PCM at 16kHz is ~32KB/s of sustained upload for
# the whole session -- on the direction a mobile link is weakest, through a tunnel that has
# been dropping mid-narration. mu-law is the telephone standard: exactly half the bytes,
# built for speech, and understood by every browser, which matters more here than a smarter
# codec that might not exist on the phone in the room.
_ULAW_BIAS = 0x84
_ULAW_MAX = 32635


def _build_ulaw_table() -> np.ndarray:
    """256-entry decode table, so decoding a frame is one numpy index."""
    out = np.empty(256, dtype="<i2")
    for i in range(256):
        u = ~i & 0xFF
        mantissa = u & 0x0F
        exponent = (u >> 4) & 0x07
        sample = ((mantissa << 3) + _ULAW_BIAS) << exponent
        sample -= _ULAW_BIAS
        out[i] = -sample if (u & 0x80) else sample
    return out


_ULAW_DECODE = _build_ulaw_table()


def ulaw_to_pcm16(data: bytes) -> bytes:
    """One mu-law byte per sample -> little-endian 16-bit PCM, which is all the VAD and
    Whisper have ever seen. Nothing downstream of here changes."""
    if not data:
        return b""
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)].tobytes()


def pcm16_to_ulaw(pcm: bytes) -> bytes:
    """Only needed for round-trip tests; the phone does the real encoding."""
    s = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    sign = np.where(s < 0, 0x80, 0).astype(np.uint8)
    s = np.clip(np.abs(s), 0, _ULAW_MAX) + _ULAW_BIAS
    exponent = np.zeros(s.shape, dtype=np.int32)
    for e in range(0, 8):                       # ascending, so the HIGHEST match wins
        exponent = np.where(s >= (1 << (e + 7)), e, exponent)
    mantissa = (s >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4).astype(np.uint8) | mantissa.astype(np.uint8))
            & 0xFF).astype(np.uint8).tobytes()
