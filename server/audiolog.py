"""Raw session audio, written to disk as it arrives, with an index that aligns it to text.

Everything downstream needs this and none of it existed. When the table said "it isn't
listening", there was no recording to go back to -- only a log line saying a decision had
been made, with no way to hear what the decision was made about. You cannot tune an endpoint
detector, measure a speaker model, or reproduce a bad turn from prose.

Two artefacts per session, side by side:

  <session>.wav    every uplink sample, continuous, 16kHz mono s16le. Nothing is dropped and
                   nothing is gated, so offsets in the index are exact sample positions and
                   the file can be re-read at any rate later.
  <session>.jsonl  one row per decision, each carrying the byte span it refers to. A row is
                   written for utterances that were ANSWERED, ones that were held, ones that
                   were ignored, and echo -- the negatives are the training signal, and they
                   are the half that was never recorded anywhere.

Writes go through a thread so the websocket loop never blocks on disk.
"""
import json
import logging
import queue
import threading
import time
import wave
from pathlib import Path

logger = logging.getLogger("audiolog")

ROOT = Path(__file__).parent.parent / "state" / "audio"
SR = 16000
SAMPLE_BYTES = 2


def wav_to_pcm16k(wav_bytes: bytes) -> bytes:
    """A WAV of any rate down to 16kHz mono s16le, so it lines up with the uplink.

    The narrator renders at its own rate (kokoro at 24kHz); the microphone arrives at 16kHz.
    An echo canceller compares the two sample by sample, so they have to be the same clock.
    """
    import io
    if not wav_bytes or not wav_bytes.startswith(b"RIFF"):
        return b""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except Exception:
        return b""
    if sw != SAMPLE_BYTES:
        return b""
    import array
    a = array.array("h")
    a.frombytes(raw)
    if ch > 1:                                        # average to mono
        a = array.array("h", [sum(a[i:i + ch]) // ch for i in range(0, len(a) - ch + 1, ch)])
    if sr != SR:                                      # nearest-sample; good enough as a reference
        n = int(len(a) * SR / sr)
        a = array.array("h", [a[min(len(a) - 1, int(i * sr / SR))] for i in range(n)])
    return a.tobytes()


def _interleave(near: bytes, far: bytes) -> bytes:
    """Weave two mono s16le buffers into one stereo buffer: left mic, right speaker."""
    n = min(len(near), len(far)) // SAMPLE_BYTES
    out = bytearray(n * 2 * SAMPLE_BYTES)
    out[0::4] = near[0:n * 2:2]
    out[1::4] = near[1:n * 2:2]
    out[2::4] = far[0:n * 2:2]
    out[3::4] = far[1:n * 2:2]
    return bytes(out)


class SessionRecorder:
    """One recorder per websocket connection. Cheap to create, safe to leave running."""

    def __init__(self, campaign: str, session: int, sid: str, enabled: bool = True):
        self.enabled = enabled
        self.samples = 0                     # exact position, so index offsets are sample-true
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._wav = None
        self._thread = None
        if not enabled:
            return
        day = time.strftime("%Y-%m-%d")
        safe = "".join(c for c in str(campaign) if c.isalnum() or c in " -_").strip() or "untitled"
        self.dir = ROOT / safe / day
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = f"s{int(session):03d}-{time.strftime('%H%M%S')}-{sid[:8]}"
        self.wav_path = self.dir / f"{stem}.wav"
        self.index_path = self.dir / f"{stem}.jsonl"
        # STEREO, and the second channel is not a luxury. Left is the microphone; right is
        # what we sent to the speaker, aligned sample-for-sample. Echo cancellation needs both
        # -- you cannot subtract a signal you did not keep -- and sixty-two minutes of recorded
        # sessions are useless for it because only the microphone was ever written down.
        # Recording the pair makes every future session an echo-cancellation test case.
        self._wav = wave.open(str(self.wav_path), "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(SAMPLE_BYTES)
        self._wav.setframerate(SR)
        self._far: bytearray = bytearray()    # DM audio waiting to be paired with mic frames
        self._far_lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        logger.info(f"recording -> {self.wav_path.name}")

    # -- audio ---------------------------------------------------------------
    def feed_far(self, pcm: bytes) -> None:
        """What we just sent to the speaker, as 16kHz mono s16le.

        Queued rather than written: it is paired with the microphone frames that arrive next,
        which is what keeps the two channels aligned to the same instant.
        """
        if not self.enabled or not pcm:
            return
        with self._far_lock:
            self._far.extend(pcm)
            # If nobody is listening the far-end would grow without bound; a couple of seconds
            # is far more than any echo path needs.
            if len(self._far) > SR * SAMPLE_BYTES * 4:
                del self._far[:-SR * SAMPLE_BYTES * 4]

    def _take_far(self, nbytes: int) -> bytes:
        """The far-end slice matching this many microphone bytes, zero-padded when silent."""
        with self._far_lock:
            take = self._far[:nbytes]
            del self._far[:len(take)]
        return bytes(take) + b"\x00" * (nbytes - len(take))

    def feed(self, pcm: bytes) -> None:
        """Append 16-bit PCM. Never raises and never blocks the caller."""
        if not self.enabled or not pcm:
            return
        self.samples += len(pcm) // SAMPLE_BYTES
        try:
            self._q.put_nowait((pcm, self._take_far(len(pcm))))
        except queue.Full:
            # Dropping a frame is bad; stalling the receive loop mid-session is worse. Losing
            # audio here would also silently desynchronise every offset written after it, so
            # say it happened rather than letting the index quietly become fiction.
            logger.warning("audio log queue full; frame dropped (offsets after this are suspect)")

    def _drain(self) -> None:
        while (item := self._q.get()) is not None:
            near, far = item
            try:
                self._wav.writeframes(_interleave(near, far))
            except Exception as e:
                logger.warning(f"audio write failed: {e}")

    # -- index ---------------------------------------------------------------
    def mark(self, kind: str, **fields) -> None:
        """Record a decision against the current position in the recording.

        `kind` is what happened to the audio: answered | held | ignored | echo | enrolled.
        The negatives matter as much as the positives -- a model that only ever sees the
        utterances we replied to cannot learn when to stay quiet.
        """
        if not self.enabled:
            return
        row = {"t": round(time.time(), 3), "kind": kind,
               "sample": self.samples, "sec": round(self.samples / SR, 3), **fields}
        try:
            with self.index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"audio index write failed: {e}")

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self._wav.close()
        except Exception:
            pass
        logger.info(f"recorded {self.samples / SR:.0f}s -> {self.wav_path.name}")


def sessions(campaign: str = "") -> list[dict]:
    """Every recorded session on disk, newest first — what the offline tools chew through."""
    out = []
    root = ROOT / campaign if campaign else ROOT
    for wav in sorted(root.rglob("*.wav"), reverse=True):
        idx = wav.with_suffix(".jsonl")
        rows = 0
        if idx.exists():
            with idx.open(encoding="utf-8") as f:
                rows = sum(1 for line in f if line.strip())
        try:
            with wave.open(str(wav)) as w:
                secs = w.getnframes() / (w.getframerate() or SR)
        except Exception:
            secs = 0.0
        out.append({"wav": str(wav), "index": str(idx) if idx.exists() else "",
                    "seconds": round(secs, 1), "marks": rows,
                    "campaign": wav.parent.parent.name, "day": wav.parent.name})
    return out
