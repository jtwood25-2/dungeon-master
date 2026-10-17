"""Voice-activity detection + utterance segmentation for hands-free / auto-prompting.

Instead of holding a button, the phone streams audio continuously and the server watches
the energy: once it has heard speech and then a stretch of trailing silence, it fires
end-of-utterance and prompts the DM automatically.

Two things make this survive a real room:
  * The threshold ADAPTS to the noise floor. A fixed threshold either never triggers
    (phone at arm's length in a quiet room) or never hears silence (noisy line), which
    reads to the user as "it isn't listening to me."
  * It keeps a PRE-ROLL of audio from just before the trigger, so the first syllable
    isn't clipped, and returns exactly the speech span — not the minutes of silence
    around it, which would otherwise be handed to STT.

Energy-based and deterministic so it's testable without a mic. Fed the same Int16 PCM
chunks the client already sends.
"""
from collections import deque

import numpy as np


class VAD:
    def __init__(self, sample_rate=16000, min_speech_ms=220, silence_ms=1100,
                 noise_mult=2.2, floor=0.006, pre_roll_ms=400, max_utterance_ms=30000,
                 ceiling=0.05, barge_mult=1.9, barge_min_speech_ms=450, barge_floor=0.022,
                 energy_threshold=None):
        self.sr = sample_rate
        self.min_speech = min_speech_ms
        # While the DM is talking the mic stays OPEN so the player can cut in. Its own voice
        # leaks back through the speaker, so barging in has to clear a higher bar than a normal
        # turn -- otherwise the DM interrupts itself with its own echo.
        self.barge_mult = barge_mult
        self.barge_min_speech = barge_min_speech_ms
        # In a silent room the ordinary threshold sits on the absolute floor, and a multiple of
        # that is still low enough for speaker bleed to clear. Barging in gets its own floor.
        self.barge_floor = barge_floor
        self.speaking = False
        # the threshold can never exceed this: one loud moment must not leave the mic deaf
        # for the rest of the session, which is what "it never started listening" looks like
        self.ceiling = ceiling
        # a natural mid-sentence pause is ~500-900ms; end-of-turn needs to sit clearly
        # past that or the DM interrupts you mid-thought
        self.hang = self.base_hang = silence_ms
        self.noise_mult = noise_mult
        self.floor = floor                  # absolute minimum, so dead silence never triggers
        self.pre_roll_ms = pre_roll_ms
        self.max_utterance = max_utterance_ms
        self.fixed_thr = energy_threshold   # override; None = adaptive
        self.noise = None                   # room floor, learned from the stream
        self.reset()

    def reset(self):
        self.speech_ms = 0.0        # accumulated speech duration this utterance
        self.silence_ms = 0.0       # trailing silence since last speech
        self.triggered = False      # have we heard enough speech to arm end-detection?
        self._spec = False          # has this pause already been handed out for early STT?
        self.utt = bytearray()      # audio of the utterance in progress
        self.pre = deque()          # (chunk, dur_ms) just before the trigger
        self.pre_ms = 0.0

    def set_speaking(self, speaking: bool):
        """Tell the VAD the DM currently has the floor, so it demands a clearer, longer burst
        of speech before calling it a barge-in."""
        self.speaking = speaking

    @property
    def threshold(self) -> float:
        if self.fixed_thr is not None:
            base = self.fixed_thr
        else:
            base = min(self.ceiling, max(self.floor, (self.noise or 0.0) * self.noise_mult))
        return max(self.barge_floor, base * self.barge_mult) if self.speaking else base

    @property
    def _needed_speech(self) -> float:
        return self.barge_min_speech if self.speaking else self.min_speech

    def _track_noise(self, rms: float):
        """Follow the room's floor without needing to know what's speech yet -- drop toward
        anything quieter fast, climb toward a louder room slowly. Deciding speech first and
        then updating only on non-speech is circular: a room noisier than the initial guess
        reads as one endless utterance and the estimate never catches up."""
        if self.noise is None:
            self.noise = rms                # first chunk: assume the room, not a voice
        elif rms < self.noise:
            self.noise = 0.7 * self.noise + 0.3 * rms
        else:
            self.noise = 0.98 * self.noise + 0.02 * rms

    def _rms(self, pcm_i16: bytes) -> float:
        if len(pcm_i16) < 2:
            return 0.0
        a = np.frombuffer(pcm_i16, dtype="<i2")
        if a.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((a.astype(np.float32) / 32768.0) ** 2)))

    def feed(self, pcm_i16: bytes) -> bytes | None:
        """Feed one PCM chunk. Returns the utterance's audio exactly when it has just
        ended, else None."""
        n = len(pcm_i16) // 2
        if n == 0:
            return None
        dur_ms = n / self.sr * 1000
        rms = self._rms(pcm_i16)
        is_speech = rms >= self.threshold
        self._track_noise(rms)

        if is_speech:
            self.speech_ms += dur_ms
            self.silence_ms = 0.0
            self._spec = False                   # still talking -> any early guess is stale
            if not self.triggered:
                self.utt.extend(b"".join(c for c, _ in self.pre))   # keep the onset
                self.pre.clear(); self.pre_ms = 0.0
                if self.speech_ms >= self._needed_speech:
                    self.triggered = True
            self.utt.extend(pcm_i16)
            if len(self.utt) / 2 / self.sr * 1000 >= self.max_utterance:
                return self._emit()
            return None

        if self.triggered:
            self.silence_ms += dur_ms
            self.utt.extend(pcm_i16)            # keep the trailing pause; whisper likes it
            if self.silence_ms >= self.hang:
                return self._emit()
            return None

        # not yet speaking: hold a rolling pre-roll and let stray blips decay
        self.pre.append((pcm_i16, dur_ms)); self.pre_ms += dur_ms
        while self.pre_ms > self.pre_roll_ms and self.pre:
            _, d = self.pre.popleft(); self.pre_ms -= d
        if self.speech_ms and self.silence_ms + dur_ms >= self.hang:
            self.speech_ms = 0.0                # a cough, not a turn -> forget it
        else:
            self.silence_ms += dur_ms
        return None

    def finish_now(self) -> bytes | None:
        """Commit the utterance immediately — the words say it's a finished thought, so don't
        sit out the rest of the end-of-turn pause."""
        return self._emit() if self.triggered and self.utt else None

    def hold(self, extra_ms: float):
        """They're mid-sentence: wait longer before calling it done, just this once."""
        self.hang = max(self.hang, self.base_hang + extra_ms)

    def speculate(self, after_ms=350) -> bytes | None:
        """Hand over the utterance early, once, as soon as the player pauses — so speech
        recognition can run DURING the wait that decides whether they're actually finished.
        Most of that 1.1s pause is otherwise dead time, and transcription is the one stage
        that can be done before we know the turn has ended."""
        if self.triggered and not self._spec and 0 < self.silence_ms >= after_ms:
            self._spec = True
            return bytes(self.utt)
        return None

    def _emit(self) -> bytes:
        audio = bytes(self.utt)
        noise = self.noise                      # survives the reset: the room hasn't changed
        self.reset()
        self.noise = noise
        self.hang = self.base_hang              # any extension applied to that utterance only
        return audio
