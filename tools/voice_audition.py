"""Render the same lines in every available voice so a human can pick.

Kokoro ships 54 voices and they are NOT of even quality -- the training data behind each one
differs, and some are audibly rougher than others. That is why the narrator and the NPCs
sound like they came from different systems: the narrator is am_michael, which is not in the
NPC pool at all, so every NPC is drawn from a different set entirely.

Quality is not something this script can judge and not something I can hear. It writes one
clip per voice and reports only what is measurable, which is speed.

Two lines per voice on purpose: narration and a spoken NPC line exercise different things, and
a voice that reads description well can still sound wrong delivering dialogue.

Usage
  python tools/voice_audition.py --out data/voice_audition
  python tools/voice_audition.py --male-only --out data/voice_audition
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))

NARRATION = ("The trees thin, and the path drops away into a hollow where the mist has not "
             "burned off. Something has been here recently.")
DIALOGUE = ("You shouldn't be out here alone, friend. Not with what's been coming down off "
            "the ridge at night.")

# Kokoro's language prefixes: a=American, b=British. The rest are other languages and would
# read English with a heavy accent, so they are excluded unless asked for.
ENGLISH = ("af_", "am_", "bf_", "bm_")


def _wer(ref: list, hyp: list) -> float:
    """Word error rate, rolling-row Levenshtein."""
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def _intelligibility(clips: dict) -> dict:
    """Synthesise a known sentence, transcribe it back, and score the words that survive.

    This is the objective half of "which voice is best". A voice with slurring, clipping or
    mangled consonants round-trips badly, and Whisper is a consistent listener across all of
    them, so the comparison is fair even though the absolute number is not meaningful.

    It measures INTELLIGIBILITY, not pleasantness. A flat, characterless voice can score
    perfectly and still be the wrong narrator. Use it to rule voices OUT, then choose by ear
    from what is left.
    """
    import io, re, wave
    import numpy as np
    import stt
    stt.warmup()
    out, scores = {}, []
    for label, audio in clips.items():
        if not audio:
            continue
        try:
            with wave.open(io.BytesIO(audio)) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                pcm = pcm.astype(np.float32) / 32768.0
                if w.getframerate() != 16000:      # whisper wants 16k
                    import math
                    idx = (np.arange(int(len(pcm) * 16000 / w.getframerate()))
                           * w.getframerate() / 16000).astype(int)
                    pcm = pcm[np.clip(idx, 0, len(pcm) - 1)]
            heard = stt.transcribe(pcm, "") or ""
        except Exception:
            continue
        ref = re.findall(r"[a-z']+", (NARRATION if label == "narration" else DIALOGUE).lower())
        hyp = re.findall(r"[a-z']+", heard.lower())
        w_ = _wer(ref, hyp)
        out[f"{label}_wer"] = round(w_, 3)
        scores.append(w_)
    if scores:
        out["wer"] = round(sum(scores) / len(scores), 3)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data" / "voice_audition"))
    p.add_argument("--male-only", action="store_true", help="narrator candidates only")
    p.add_argument("--all-languages", action="store_true")
    p.add_argument("--no-score", action="store_true",
                   help="skip the transcribe-back intelligibility check")
    a = p.parse_args()

    import config
    config.TTS_ENGINE = "kokoro"
    import tts

    async def go():
        await tts.warmup()
        k = tts._get_kokoro()
        if k is None:
            print(json.dumps({"error": "kokoro unavailable"})); return 1
        names = sorted(k.get_voices())
        if not a.all_languages:
            names = [n for n in names if n.startswith(ENGLISH)]
        if a.male_only:
            names = [n for n in names if n[1:2] == "m"]

        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        rows = []
        for n in names:
            r = {"voice": n, "accent": "British" if n[0] == "b" else "American",
                 "sex": "male" if n[1:2] == "m" else "female",
                 "is_current_narrator": n == config.TTS_VOICE,
                 "in_npc_pool": n in __import__("voices").POOL}
            clips = {}
            for label, line in (("narration", NARRATION), ("dialogue", DIALOGUE)):
                t0 = time.perf_counter()
                audio = await tts.synth(f"{line} ", n)      # trailing space defeats the cache
                r[f"{label}_s"] = round(time.perf_counter() - t0, 2)
                (out / f"{n}__{label}.wav").write_bytes(audio or b"")
                clips[label] = audio
            if not a.no_score:
                r.update(_intelligibility(clips))
            rows.append(r)
        print(json.dumps({"out": str(out), "voices": len(rows), "detail": rows}, indent=2))
        return 0

    return asyncio.run(go())


if __name__ == "__main__":
    sys.exit(main())
