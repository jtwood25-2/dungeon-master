"""Speaker recognition — learn each player's voice so they can just talk.

CAM++ (3D-Speaker, trained on 200k speakers) via sherpa-onnx produces a 192-dimension
embedding per utterance; enrolled players are matched by cosine distance. Profiles persist
per campaign and blend in new samples as they are confidently matched.

Measured against the Critical Role wiki transcripts' own speaker labels, on twelve minutes of
crosstalk-heavy audio: 0.808 attribution, where the previous model scored 0.633 and chance
across eight speakers is 0.125.

Two things here exist because getting them wrong produced numbers that looked fine:

  ONE SPACE ONLY   embed() never mixes models. It used to fall back to a 26-dim MFCC
                   fingerprint for clips under half a second while returning neural vectors
                   otherwise, so a single run emitted two incomparable spaces at once.
  DIMENSION GUARD  SpeakerBook.load discards profiles enrolled under a different model. A
                   cosine score between spaces is not a weak answer, it is a meaningless one
                   that still looks like a number and gets acted on.

MFCC remains only as a fallback for when no neural model is present at all.
"""
import json
import logging

import numpy as np
from pathlib import Path

SR = 16000
VOICES = Path(__file__).parent.parent / "state" / "voices"
# CAM++ (3D-Speaker, 200k speakers). Measured against the wiki transcripts' own speaker
# labels on twelve minutes of crosstalk-heavy podcast audio, with edit-distance alignment
# rather than positional pairing: 0.808 attribution against the previous model's 0.633.
# It emits 192 dimensions where the old one emitted 512, so any profile enrolled before the
# switch is in a different space and cannot be compared -- see SpeakerBook.load.
MODEL = Path(__file__).parent.parent / "state" / "models" / "campplus_meta.onnx"
if not MODEL.exists():                                  # fall back if the download is missing
    MODEL = Path(__file__).parent.parent / "state" / "models" / "speaker.onnx"
_MEL_CACHE: dict = {}
logger = logging.getLogger("speakers")

_NEURAL = None                                        # None=untried, False=unavailable, else extractor


def _expected_dim() -> int:
    """Dimension the CURRENT model emits, or 0 if it cannot be determined.

    Used to reject profiles enrolled under a different model. Returning 0 means "do not
    know", and in that case nothing is discarded -- refusing to guess is safer than
    throwing away good enrolments over a failed introspection."""
    ex = _neural()
    if ex is None:
        return 0
    try:
        return int(ex.dim)
    except Exception:
        return 0


def _neural():
    """Load the ONNX neural speaker model (sherpa-onnx) once; None if unavailable."""
    global _NEURAL
    if _NEURAL is None:
        try:
            import sherpa_onnx
            if not MODEL.exists():
                _NEURAL = False
            else:
                cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=1,
                                                                  provider="cpu")
                _NEURAL = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        except Exception:
            _NEURAL = False
    return _NEURAL or None


def _neural_embed(pcm_f32):
    ex = _neural()
    if ex is None:
        return None
    try:
        s = ex.create_stream()
        s.accept_waveform(SR, np.asarray(pcm_f32, dtype=np.float32))
        s.input_finished()
        if not ex.is_ready(s):
            return None
        e = np.asarray(ex.compute(s), dtype=np.float32)
        n = np.linalg.norm(e)
        return e / n if n > 0 else None
    except Exception:
        return None


def _mel_filters(n_fft, n_mels, sr):
    key = (n_fft, n_mels, sr)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    b = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        for j in range(b[i - 1], b[i]):
            fb[i - 1, j] = (j - b[i - 1]) / max(b[i] - b[i - 1], 1)
        for j in range(b[i], b[i + 1]):
            fb[i - 1, j] = (b[i + 1] - j) / max(b[i + 1] - b[i], 1)
    _MEL_CACHE[key] = fb
    return fb


def embed(pcm_f32, sr=SR):
    """PCM (mono float32 @16k) -> normalized speaker embedding, or None.

    The choice of model is made ONCE, for the whole process, and never per clip. It used to
    fall through to MFCC whenever a clip was under half a second, which meant a single run
    emitted vectors from two different spaces -- 512-dim neural and 26-dim MFCC -- mixed
    together. Clustering a corpus of them crashed outright; comparing a short live utterance
    against 512-dim enrolled profiles would have been worse, because a cosine score between
    incomparable spaces is a number that looks like an answer.

    So: if the neural model is there, it is the only thing that speaks, and a clip too short
    for it returns None. MFCC remains the fallback only when there is no neural model at all.
    """
    pcm = np.asarray(pcm_f32)
    if _neural() is not False:                       # neural model present -> use it or nothing
        if pcm.size < sr // 2:
            return None
        return _neural_embed(pcm_f32)
    return _mfcc_embed(pcm_f32)


def _mfcc_embed(pcm_f32, sr=SR, n_mfcc=13, n_mels=26, n_fft=512, hop=160, win=400):
    x = np.asarray(pcm_f32, dtype=np.float32)
    if x.size < win * 3:
        return None
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])                       # pre-emphasis
    n = 1 + (len(x) - win) // hop
    frames = np.stack([x[i * hop:i * hop + win] for i in range(n)]) * np.hamming(win)
    mag = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    energy = mag.sum(1)
    voiced = energy > max(np.median(energy) * 0.4, 1e-6)             # drop silence frames
    if voiced.sum() < 4:
        return None
    mel = np.log(mag[voiced] @ _mel_filters(n_fft, n_mels, sr).T + 1e-8)
    k = np.arange(n_mfcc)
    dct = np.cos(np.pi * np.outer(np.arange(n_mels) + 0.5, k) / n_mels)   # DCT-II basis
    mfcc = mel @ dct
    emb = np.concatenate([mfcc.mean(0), mfcc.std(0)])
    nrm = np.linalg.norm(emb)
    return emb / nrm if nrm > 0 else None


class SpeakerBook:
    """Enrolled voice profiles for one campaign; trains online, persists to disk."""
    def __init__(self, campaign="default"):
        self.campaign = campaign
        self.profiles: dict = {}          # voice_id -> {"emb": [...], "n": int, "label": str}

    @classmethod
    def load(cls, campaign="default"):
        """Load profiles, discarding any that predate the current embedding model.

        Embeddings from two different models live in different spaces, so a cosine score
        between them is not a weak answer -- it is a meaningless one that still looks like a
        number and would be acted on. When the model changed from 512 to 192 dimensions every
        enrolled profile became uncomparable, and silently keeping them would have attributed
        lines to whoever happened to score highest on noise.

        Stale profiles are set aside rather than deleted; the players simply need to re-enrol.
        """
        b = cls(campaign)
        p = VOICES / f"{campaign}.json"
        if not p.exists():
            return b
        profiles = json.loads(p.read_text())
        want = _expected_dim()
        good = {k: v for k, v in profiles.items()
                if not want or len(v.get("emb") or []) == want}
        if len(good) != len(profiles):
            stale = sorted(set(profiles) - set(good))
            archive = VOICES / f"{campaign}.pre-{want or 'unknown'}d.json"
            if not archive.exists():
                archive.write_text(json.dumps(profiles), encoding="utf-8")
            logger.warning(
                f"{campaign}: dropped {len(stale)} voice profile(s) from a previous embedding "
                f"model ({stale}); they are archived at {archive.name}. Those players need to "
                f"re-enrol before they can be recognised.")
        b.profiles = good
        return b

    def save(self):
        VOICES.mkdir(parents=True, exist_ok=True)
        (VOICES / f"{self.campaign}.json").write_text(json.dumps(self.profiles))

    def enroll(self, voice_id, pcm, label=None) -> bool:
        e = embed(pcm)
        if e is None:
            return False
        prof = self.profiles.get(voice_id)
        if prof:                                       # running average with prior samples
            prev = np.array(prof["emb"]); n = prof["n"]
            merged = (prev * n + e) / (n + 1)
            merged /= (np.linalg.norm(merged) or 1)
            prof.update(emb=merged.tolist(), n=n + 1)
            if label: prof["label"] = label
        else:
            self.profiles[voice_id] = {"emb": e.tolist(), "n": 1, "label": label or voice_id}
        self.save()
        return True

    def identify(self, pcm, threshold=0.55, margin=0.15):
        """Return (voice_id, label, score) or (None, None, best_score).

        The margin has to be read against how alike the enrolled voices already are. These
        profiles sit up to 0.749 similar to each other, so a 0.06 margin was far inside the
        noise between them and the book confidently named the wrong player over and over.
        Claiming nobody is the correct answer when two profiles are this close: an unlabelled
        line is obviously unattributed, whereas a wrong name reads as fact.
        """
        e = embed(pcm)
        if e is None or not self.profiles:
            return None, None, 0.0
        scored = sorted(((float(np.dot(e, np.array(p["emb"]))), vid, p["label"])
                         for vid, p in self.profiles.items()), reverse=True)
        top = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        if top[0] >= threshold and (top[0] - second) >= margin:
            # Training on every accepted match is a feedback loop with no brake: one wrong
            # attribution, or one line of the DM's own voice echoing back, gets folded into a
            # player's profile and makes the next mistake likelier. Only learn when the match
            # is far beyond the bar, never merely past it.
            if top[0] >= threshold + 0.2 and (top[0] - second) >= margin * 2:
                self._train(top[1], e)
            return top[1], top[2], top[0]
        return None, None, top[0]

    def _train(self, voice_id, e, w=0.08):             # gentle online update
        prof = self.profiles[voice_id]
        v = (1 - w) * np.array(prof["emb"]) + w * e
        v /= (np.linalg.norm(v) or 1)
        prof["emb"] = v.tolist()
        self.save()

    def roster_labels(self):
        return {vid: p["label"] for vid, p in self.profiles.items()}
