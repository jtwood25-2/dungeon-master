"""Subtract the DM's own voice from the microphone, using the signal we sent to the speaker.

This is not speech separation and should not be confused with it. Separating two strangers
talking over each other is a hard, blind problem. Removing the narrator is not blind at all:
WE GENERATED THAT AUDIO, so we have the exact waveform that went to the speaker. What comes
back is that same waveform, delayed by the trip through the network, the phone and the room,
and reshaped by the room. Recovering a delay and a linear filter from a known reference is a
textbook adaptive-filtering problem.

Everything before this tried to do the job with text: transcribe whatever the microphone heard
and compare the words against what the DM had just said. That is guessing, and it guessed
wrong in both directions -- it discarded a player's genuine question because it shared words
with the narration ("Are any of them still in the bonfire?"), and it let the DM interrupt
itself when the wording drifted. Words are the wrong domain for a signal problem.

Two stages, in the order they matter:

  delay      where the echo actually sits, found by cross-correlation. Get this wrong and the
             filter has nothing to lock onto -- the round trip here is a phone, a websocket
             and a room, not a sound card, so it is neither small nor fixed.
  NLMS       a normalised least-mean-squares filter that learns the room's response and
             subtracts its estimate of the echo. Normalised because the narrator's level
             varies enormously between a whisper and a shout, and a plain LMS either crawls
             or diverges when it does.

Deliberately dependency-free: numpy only. webrtc-audio-processing and speexdsp both do this
better, and neither builds cleanly here.
"""
import numpy as np

SR = 16000
# 200ms of filter. A phone speaker and a small room settle well inside that, and every tap
# costs time -- 3200 taps at 16kHz is already the expensive part of this file.
TAIL = 3200
# How far ahead to look for the echo. The far end is queued, sent over a websocket, decoded and
# played, so half a second is comfortable rather than generous.
MAX_DELAY = int(0.5 * SR)


def to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def to_pcm(x: np.ndarray) -> bytes:
    return np.clip(x * 32768.0, -32768, 32767).astype("<i2").tobytes()


def estimate_delay(near: np.ndarray, far: np.ndarray, max_lag: int = MAX_DELAY) -> int:
    """Samples the far end lags the microphone by, or 0 if nothing correlates.

    Cross-correlation through an FFT rather than a loop: at 16kHz a half-second search is
    eight thousand lags, and doing that directly per block is slower than the filter itself.
    """
    n = min(len(near), len(far))
    if n < 1024:
        return 0
    a = near[:n] - near[:n].mean()
    b = far[:n] - far[:n].mean()
    if not np.any(b) or not np.any(a):
        return 0
    size = 1 << int(np.ceil(np.log2(n + max_lag)))
    A = np.fft.rfft(a, size)
    B = np.fft.rfft(b, size)
    corr = np.fft.irfft(A * np.conj(B), size)[:max_lag]
    if not np.any(corr):
        return 0
    lag = int(np.argmax(np.abs(corr)))
    # A flat correlation means the two are unrelated -- there is no echo to chase, and locking
    # onto noise would have the filter subtract the player instead. Measured on this material:
    # a real echo path scores about 150 on peak-to-mean, two unrelated noise signals about 5.4.
    # The threshold was 4.0, i.e. BELOW the noise floor, so it reported a confident delay for
    # signals with no relationship at all. 12 sits in the gap with room on both sides.
    peak, floor = float(np.abs(corr[lag])), float(np.abs(corr).mean())
    return lag if floor > 0 and peak / floor > 12.0 else 0


class FreqCanceller:
    """Overlap-save frequency-domain adaptive filter, which is what this job actually needs.

    A time-domain NLMS was tried first and measured on real session audio: the best it reached
    across a parameter sweep was 4.7 dB of echo removed, and every setting that adapted faster
    made it worse. That is not a tuning problem, it is the wrong algorithm for the material.
    These recordings are near-continuous speech, so there is very little echo-on-its-own for a
    3200-tap filter to learn from, and a single step size has to serve every frequency at once.

    In the frequency domain each bin adapts at its own rate, normalised by the energy actually
    present in that bin. Speech has enormous spectral tilt -- most of its power under 1kHz --
    so per-bin normalisation is the difference between the top of the band converging and it
    never moving at all.
    """

    def __init__(self, block: int = 512, blocks: int = 8, mu: float = 0.5, eps: float = 1e-8):
        self.B = block                       # new samples per step
        self.P = blocks                      # partitions; tail = block * blocks
        self.N = 2 * block                   # FFT size for overlap-save
        self.mu = mu
        self.eps = eps
        self.W = np.zeros((blocks, block + 1), dtype=np.complex64)   # per-partition weights
        self.X = np.zeros((blocks, block + 1), dtype=np.complex64)   # far-end spectrum history
        self.pow = np.full(block + 1, eps, dtype=np.float32)         # running per-bin energy
        self.x_prev = np.zeros(block, dtype=np.float32)
        self.delay = 0
        self._far_buf = np.zeros(0, dtype=np.float32)
        self.adapted = 0
        self.frozen = 0

    @property
    def tail(self) -> int:
        return self.B * self.P

    def set_delay(self, samples: int) -> None:
        self.delay = max(0, int(samples))

    def _aligned_far(self, n: int, far: np.ndarray) -> np.ndarray:
        self._far_buf = np.concatenate([self._far_buf, far])
        need = n + self.delay
        if len(self._far_buf) < need:
            self._far_buf = np.concatenate(
                [np.zeros(need - len(self._far_buf), dtype=np.float32), self._far_buf])
        ref = self._far_buf[len(self._far_buf) - need: len(self._far_buf) - self.delay]
        keep = self.tail + n + self.delay
        self._far_buf = self._far_buf[-keep:]
        return ref

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        ref = self._aligned_far(len(near), far)
        out = np.empty_like(near)
        for i in range(0, len(near), self.B):
            d = near[i:i + self.B]
            x = ref[i:i + self.B]
            if len(d) < self.B:                       # pad the last short block
                d = np.pad(d, (0, self.B - len(d)))
                x = np.pad(x[:self.B], (0, self.B - len(x[:self.B])))
            out_blk = self._step(d, x)
            out[i:i + len(near[i:i + self.B])] = out_blk[:len(near[i:i + self.B])]
        return out

    def _step(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(np.concatenate([self.x_prev, x]), self.N).astype(np.complex64)
        self.x_prev = x.copy()
        self.X[1:] = self.X[:-1]
        self.X[0] = X

        Y = np.sum(self.W * self.X, axis=0)           # estimated echo, all partitions
        y = np.fft.irfft(Y, self.N)[self.B:]          # overlap-save: keep the second half
        e = d - y

        # Freeze on double-talk. Comparing the microphone against the FAR END was far too
        # permissive: the far end is the signal before the speaker and the room attenuate it,
        # so the echo arriving is several times quieter, and a threshold set against the
        # original let the filter adapt through almost every double-talk block. Compare it
        # against the echo we actually estimate instead -- that is the quantity the microphone
        # should contain if nobody else is speaking.
        de = float(np.dot(d, d)) / self.B
        ye = float(np.dot(y, y)) / self.B
        xe = float(np.dot(x, x)) / self.B
        if ye > 1e-9:
            # the microphone should be roughly the echo; much more than that is someone talking
            learning = xe > 1e-7 and de < ye * 4.0
        else:
            learning = xe > 1e-7 and de < xe * 2.0   # bootstrap, before anything is learned
        self.adapted += int(learning)
        self.frozen += int(not learning)
        if learning:
            E = np.fft.rfft(np.concatenate([np.zeros(self.B), e]), self.N).astype(np.complex64)
            # per-bin power, smoothed over ALL partitions: this is the part a time-domain
            # filter cannot do, and normalising by the current block alone is not enough
            self.pow = 0.9 * self.pow + 0.1 * np.sum(np.abs(self.X) ** 2, axis=0)
            grad = np.conj(self.X) * (self.mu * E / (self.pow + self.eps))[None, :]
            # GRADIENT CONSTRAINT. Without it this is the unconstrained form, whose filter is
            # free to grow a tail longer than the block it is supposed to model -- it wraps
            # around, feeds itself, and blows up. Measured: -26 dB and then NaN. Project each
            # partition back to the time domain, keep only the first B taps, transform back.
            g = np.fft.irfft(grad, self.N, axis=1)
            g[:, self.B:] = 0.0
            self.W += np.fft.rfft(g, self.N, axis=1).astype(np.complex64)
            if not np.all(np.isfinite(self.W)):     # never let a bad block poison the filter
                self.W[:] = 0.0
                self.pow[:] = self.eps
        return e


class Canceller:
    """NLMS echo canceller. Feed it matched microphone and speaker blocks; get the mic back
    with the narrator taken out of it."""

    def __init__(self, tail: int = TAIL, mu: float = 0.08, eps: float = 1e-6):
        self.w = np.zeros(tail, dtype=np.float32)     # the learned room response
        self.x = np.zeros(tail, dtype=np.float32)     # far-end history, newest first
        self.mu = mu
        self.eps = eps
        self.delay = 0
        self._far_buf = np.zeros(0, dtype=np.float32)
        self.adapted = 0                              # blocks the filter actually learned from
        self.frozen = 0                               # blocks held still because of double-talk

    def set_delay(self, samples: int) -> None:
        self.delay = max(0, int(samples))

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """One block. `far` is what was played; `near` is what the microphone heard."""
        self._far_buf = np.concatenate([self._far_buf, far])
        # line the two up: the echo of THIS block's far end arrives `delay` samples later
        need = len(near) + self.delay
        if len(self._far_buf) < need:
            self._far_buf = np.concatenate(
                [np.zeros(need - len(self._far_buf), dtype=np.float32), self._far_buf])
        ref = self._far_buf[len(self._far_buf) - need: len(self._far_buf) - self.delay]
        self._far_buf = self._far_buf[-(TAIL + len(near) + self.delay):]

        # DOUBLE-TALK. An adaptive filter must stop learning while the near end is talking,
        # and this is not a refinement -- without it the filter chases the player's voice into
        # its weights and diverges outright. Measured on real session audio it scored -12.7 dB
        # ERLE: the output was four times LOUDER than the input it was meant to clean.
        #
        # The test is energy: if the microphone carries much more than the echo could account
        # for, someone else is talking, so hold the weights still and keep cancelling with what
        # has already been learned.
        near_e = float(np.dot(near, near)) / max(1, len(near))
        ref_e = float(np.dot(ref, ref)) / max(1, len(ref)) if len(ref) else 0.0
        learning = ref_e > 1e-7 and near_e < ref_e * 8.0
        self.adapted += int(learning)
        self.frozen += int(not learning)

        out = np.empty_like(near)
        w, x, mu, eps = self.w, self.x, self.mu, self.eps
        for i in range(len(near)):
            x[1:] = x[:-1]
            x[0] = ref[i] if i < len(ref) else 0.0
            y = float(np.dot(w, x))                   # what we think the echo is
            e = near[i] - y                           # what is left once it is removed
            if learning:
                # Normalised: divide by the reference's own energy, so a loud narrator and a
                # quiet one adapt at the same rate instead of diverging or crawling.
                w += (mu * e / (float(np.dot(x, x)) + eps)) * x
            out[i] = e
        self.w = w
        return out


def erle_db(near: np.ndarray, out: np.ndarray) -> float:
    """Echo Return Loss Enhancement: how much quieter the output is than the input, in dB.

    The number this whole file exists to raise. Measured over stretches where only the DM was
    talking, it is the honest score -- over a stretch where a player was talking it would
    reward destroying their speech, which is the failure mode to watch for.
    """
    a, b = float(np.mean(near ** 2)), float(np.mean(out ** 2))
    if a <= 0 or b <= 0:
        return 0.0
    return float(10.0 * np.log10(a / b))
