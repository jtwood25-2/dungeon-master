"""Turn long multi-speaker RPG audio into labelled turn-taking examples.

The question the live system keeps getting wrong is not "what did they say" -- Whisper is
fine at that -- it is "was that said TO me, and are they finished". That is a turn-taking
problem, and it needs examples of both answers. A published transcript cannot supply them:
it tells you who spoke and what they said, but not the pauses, the overlaps, or the beats
where the DM stayed quiet and let the table talk. Those live only in the audio.

So this reads the AUDIO and derives the labels from timing:

  segment    voice activity -> speech regions with exact start/end
  transcribe faster-whisper with word timestamps
  attribute  cluster speaker embeddings -> stable speaker ids across the episode
  label      for each end-of-speech, did the DM take the floor within RESPOND_WITHIN?
             -> positive (respond) / negative (let it pass)

A real table gives one DM and several players, which is exactly the shape we need: the DM's
own turns mark where a response WAS correct, and every player-to-player exchange in between
marks where staying quiet was correct. The negatives are the valuable half and the reason
this can't be built from a transcript.

Usage
  python tools/corpus.py segment  --audio ep.mp3 --out data/raw/ep01
  python tools/corpus.py label    --session data/raw/ep01 --dm-speaker 0
  python tools/corpus.py split    --root data/raw --test-frac 0.25
  python tools/corpus.py stats    --root data/raw

Every subcommand prints JSON to stdout.

NOTE ON SOURCES: point --audio at recordings you have the right to use for this. Your own
sessions are recorded by server/audiolog.py in exactly this layout and need no conversion --
`label` reads them directly, and they are the closest match to your actual room, mics and
players, which is what the model has to work in.
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
SR = 16000
# How soon after someone stops must the DM speak for that pause to count as "your turn"?
# Matt answers fast when answering at all; a long gap is usually the table still working.
RESPOND_WITHIN = 2.5
MIN_SPEECH = 0.35          # shorter than this is a cough, a laugh, a "mm"
MIN_GAP = 0.30             # pauses shorter than this are within one person's turn


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


def load_audio(path: Path) -> np.ndarray:
    """Mono float32 at 16k. Uses torchaudio (already installed) so mp3/m4a/wav all work."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path)) as w:
            if w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == SR:
                raw = w.readframes(w.getnframes())
                return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    import torch, torchaudio
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0).numpy().astype(np.float32)


def speech_regions(pcm: np.ndarray, frame_ms=20) -> list[tuple]:
    """Speech regions as [(start_s, end_s, synthetic_start)].

    The energy version below found 38% of the speech in a Critical Role episode -- 4.5 minutes
    out of 12 -- and that silently poisoned everything downstream. Two thirds of the turns were
    missing, so "did the DM answer within 2.5s" was answered against the wrong neighbour, the
    labels were mostly noise, and the trained model came out with precision flat at 0.15 across
    every threshold. A flat sweep is the signature of a model that cannot discriminate, and the
    reason was never the model.

    The mistake was assuming the quietest fifth of the audio is silence. On a podcast where
    people talk almost continuously that percentile IS speech, so the threshold sat above the
    quiet half of every sentence. Silero is an actual speech detector rather than a guess about
    loudness: 92% coverage on the same clip, in about a second for twelve minutes.

    It ships inside faster-whisper, which is already a dependency. Energy remains the fallback
    so this still runs if that ever changes.
    """
    for fn in (_ten_regions, _silero_regions):
        try:
            out = fn(np.asarray(pcm, dtype=np.float32))
            if out:
                return _split_long(out, pcm)
        except Exception:
            continue
    return [(s, e, False) for s, e in _energy_regions(pcm, frame_ms)]


MAX_REGION = 12.0          # seconds; longer than this and several turns are inside one region


def _ten_regions(pcm: np.ndarray) -> list[tuple]:
    """TEN VAD frame probabilities with our own hysteresis.

    Preferred over the copy of Silero vendored inside faster-whisper, which caps region
    length and splits mid-sentence: 58% of its boundaries had a gap of exactly zero. Those
    are manufactured, not pauses, and `gap` is the strongest feature the endpoint model has --
    feeding it a zero for more than half of all examples is worse than feeding it nothing.
    TEN produces no zero-gap boundaries at all and 70% of its gaps exceed half a second.
    """
    from ten_vad import TenVad
    hop = 256                                   # 16ms frames at 16k
    v = TenVad(hop_size=hop)
    i16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    probs = np.array([v.process(i16[i:i + hop])[0] for i in range(0, len(i16) - hop, hop)])
    on, off = 0.5, 0.35                         # enter on loud, leave on quiet
    out, start, speaking = [], None, False
    for i, p in enumerate(probs):
        t = i * hop / SR
        if not speaking and p >= on:
            speaking, start = True, t
        elif speaking and p < off:
            if t - start >= MIN_SPEECH:
                out.append((start, t))
            speaking = False
    if speaking and (len(probs) * hop / SR) - start >= MIN_SPEECH:
        out.append((start, len(probs) * hop / SR))
    merged = []
    for s, e in out:
        if merged and s - merged[-1][1] < MIN_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _silero_regions(pcm: np.ndarray) -> list[tuple]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(pcm, VadOptions(min_speech_duration_ms=int(MIN_SPEECH * 1000),
                                               min_silence_duration_ms=int(MIN_GAP * 1000)))
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def _split_long(regions: list[tuple], pcm: np.ndarray) -> list[tuple]:
    """Cut over-long regions at their quietest interior point.

    Returns (start, end, synthetic_start). A cut we made ourselves is NOT a pause: the speaker
    kept going. Marking it lets the labeller skip those boundaries entirely, rather than
    reporting gap=0 and teaching the model that a zero pause means a turn ended -- the exact
    corruption the vendored Silero caused with its 58% zero-length gaps.

    A real VAD keeps talking through a whole exchange, so a 36-second region routinely holds
    several speakers -- which the clustering then has to average into one embedding. Splitting
    at the deepest internal dip recovers the likeliest speaker change without inventing the
    zero-length gaps the vendored Silero produced.
    """
    out = []
    for s, e in regions:
        if e - s <= MAX_REGION:
            out.append((s, e, False)); continue
        stack = [(s, e, False)]
        pieces = []
        while stack:
            a, b, synth = stack.pop()
            if b - a <= MAX_REGION:
                pieces.append((a, b, synth)); continue
            seg = pcm[int(a * SR):int(b * SR)]
            n = int(SR * 0.05)
            frames = seg[:len(seg) // n * n].reshape(-1, n)
            energy = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
            # only consider cuts away from the edges, so a split cannot orphan a fragment
            lo, hi = int(len(energy) * 0.25), int(len(energy) * 0.75)
            if hi <= lo:
                pieces.append((a, b, synth)); continue
            cut = a + (lo + int(np.argmin(energy[lo:hi]))) * 0.05
            stack += [(a, cut, synth), (cut, b, True)]
        out += pieces
    return sorted(out)


def _energy_regions(pcm: np.ndarray, frame_ms=20) -> list[tuple]:
    """Fallback only. Kept because it needs no model, but it under-detects badly on
    continuous speech -- see speech_regions."""
    n = int(SR * frame_ms / 1000)
    frames = pcm[:len(pcm) // n * n].reshape(-1, n)
    energy = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    # noise floor from the quietest fifth of the episode, so a loud room doesn't swallow speech
    floor = np.quantile(energy, 0.20)
    thresh = max(floor * 3.0, np.quantile(energy, 0.55) * 0.5)
    voiced = energy > thresh

    out, start = [], None
    for i, v in enumerate(voiced):
        t = i * frame_ms / 1000
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start >= MIN_SPEECH:
                out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(voiced) * frame_ms / 1000))

    merged = []
    for s, e in out:                       # stitch across breaths inside one person's turn
        if merged and s - merged[-1][1] < MIN_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def cmd_segment(a) -> int:
    src = Path(a.audio)
    if not src.exists():
        return _fail(f"no audio at {src}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pcm = load_audio(src)
    regions = speech_regions(pcm)
    if a.limit:
        regions = regions[:a.limit]

    sys.path.insert(0, str(ROOT / "server"))
    import speakers
    from faster_whisper import WhisperModel
    model = WhisperModel(a.whisper, device=a.device, compute_type="float16"
                         if a.device == "cuda" else "int8")

    rows = []
    for i, (s, e, synth) in enumerate(regions):
        clip = pcm[int(s * SR):int(e * SR)]
        segs, _info = model.transcribe(clip, language="en", beam_size=1,
                                       vad_filter=False, word_timestamps=False)
        text = " ".join(x.text.strip() for x in segs).strip()
        if not text:
            continue
        emb = speakers.embed(clip)
        rows.append({"i": i, "start": round(s, 3), "end": round(e, 3),
                     "dur": round(e - s, 3), "text": text, "synthetic_start": bool(synth),
                     "emb": [round(float(v), 5) for v in emb] if emb is not None else None})
        if a.progress and i % 25 == 0:
            print(f"  {i}/{len(regions)}  {s/60:.1f}min", file=sys.stderr)

    (out / "segments.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(json.dumps({"session": str(out), "regions": len(regions), "segments": len(rows),
                      "minutes": round(len(pcm) / SR / 60, 1)}))
    return 0


def cluster_speakers(rows: list[dict], k: int) -> list[dict]:
    """Group segment embeddings into k speakers.

    Agglomerative on cosine distance rather than k-means: turn counts per speaker are wildly
    unbalanced in an RPG session (the DM talks far more than anyone), and k-means pulls
    centroids toward the loudest contributor.
    """
    from sklearn.cluster import AgglomerativeClustering
    # Only cluster vectors from ONE space. Older runs mixed 512-dim neural embeddings with
    # 26-dim MFCC ones for clips under half a second; np.array on that raised "inhomogeneous
    # shape" and stopped the whole corpus. Take the dimension the majority of the episode uses
    # and drop the rest, so an old segments.jsonl still labels instead of failing.
    dims = {}
    for r in rows:
        if isinstance(r.get("emb"), list):
            dims[len(r["emb"])] = dims.get(len(r["emb"]), 0) + 1
    if not dims:
        for r in rows:
            r["speaker"] = -1
        return rows
    dim = max(dims, key=dims.get)
    idx = [i for i, r in enumerate(rows)
           if isinstance(r.get("emb"), list) and len(r["emb"]) == dim]
    if len(idx) < k:
        for r in rows:
            r["speaker"] = -1
        return rows
    X = np.array([rows[i]["emb"] for i in idx], dtype=float)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                  linkage="average").fit_predict(X)
    for j, i in enumerate(idx):
        rows[i]["speaker"] = int(lab[j])
    for r in rows:
        r.setdefault("speaker", -1)
    return rows


def cmd_label(a) -> int:
    """Turn segments into (context -> respond?) examples.

    A row is positive when the DM took the floor within RESPOND_WITHIN of this segment ending,
    and negative when the conversation carried on without them. That is the exact judgement
    endpoint.assess() has to make live, so the label matches the decision.
    """
    sess = Path(a.session)
    segf = sess / "segments.jsonl"
    if not segf.exists():
        return _fail(f"no segments.jsonl in {sess} — run `segment` first")
    rows = [json.loads(l) for l in segf.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return _fail("no segments")

    rows = cluster_speakers(rows, a.speakers)
    dm = a.dm_speaker
    if dm < 0:                       # unspecified: the DM is whoever holds the floor longest
        talk: dict = {}
        for r in rows:
            talk[r["speaker"]] = talk.get(r["speaker"], 0.0) + r["dur"]
        dm = max(talk, key=talk.get)

    ex = []
    for i, r in enumerate(rows[:-1]):
        if r["speaker"] == dm:
            continue                 # the DM's own speech is not a cue to respond to
        nxt = rows[i + 1]
        # A split we made ourselves is not a pause -- the speaker carried straight on -- so it
        # is no evidence about whether the DM should have answered. Skip rather than label.
        if nxt.get("synthetic_start"):
            continue
        gap = nxt["start"] - r["end"]
        # "The DM spoke next" counted Matt laughing, agreeing, or arguing a rule as a response.
        # 37% of his lines are under six words, and almost none of those are him running the
        # game. When a register tag is available the target tightens to the DM taking the floor
        # IN ROLE -- narrating, voicing an NPC, calling for a roll, ruling -- which is the thing
        # the live system would actually need to do.
        # TRIED AND REVERTED: tightening this to "the DM took the floor IN ROLE" using the
        # register classifier. The concept is cleaner -- Matt laughing is not a response -- but
        # measured it was worse, 0.609 -> 0.529 AUC. The label would then depend on TWO
        # imperfect classifiers stacked: diarisation at 0.81 and register at 0.83, so roughly
        # a third of targets are wrong, and the positive class shrinks from 57% to 18%. A
        # cleaner concept measured through a noisier pipeline is a worse label.
        respond = int(nxt["speaker"] == dm and gap <= RESPOND_WITHIN)
        ex.append({"session": sess.name, "i": r["i"],
                   "text": r["text"], "dur": r["dur"], "gap": round(gap, 3),
                   "speaker": r["speaker"], "next_speaker": nxt["speaker"],
                   "prev_text": rows[i - 1]["text"] if i else "",
                   "prev_was_dm": int(bool(i) and rows[i - 1]["speaker"] == dm),
                   # how THIS turn sounded as it ended -- the boundary cue lives here, not in
                   # the following turn, so it is the current row's prosody that is carried
                   "prosody": r.get("prosody") or {},
                   # the register of THIS turn: an in-character action and an out-of-character
                   # rules question both need answering, but they are different kinds of cue
                   "register": r.get("register"),
                   "next_register": nxt.get("register"),
                   "label": respond})

    (sess / "examples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ex), encoding="utf-8")
    pos = sum(e["label"] for e in ex)
    print(json.dumps({"session": str(sess), "dm_speaker": dm, "examples": len(ex),
                      "respond": pos, "let_pass": len(ex) - pos,
                      "positive_rate": round(pos / max(len(ex), 1), 3)}))
    return 0


def cmd_split(a) -> int:
    """Hold out whole SESSIONS, never individual turns.

    Splitting turns at random leaks: the turn before and the turn after a held-out example
    come from the same conversation, the same voices and the same scene, so the model scores
    well by recognising the episode rather than by learning when to speak.
    """
    root = Path(a.root)
    sessions = sorted(p.parent for p in root.rglob("examples.jsonl"))
    if not sessions:
        return _fail(f"no labelled sessions under {root} — run `label` first")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(sessions))
    n_test = max(1, int(round(len(sessions) * a.test_frac)))
    test = {sessions[i].name for i in order[:n_test]}

    counts = {"train": 0, "test": 0}
    out = {"train": [], "test": []}
    for s in sessions:
        split = "test" if s.name in test else "train"
        for line in (s / "examples.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                out[split].append(line)
                counts[split] += 1
    dest = root.parent / "splits"; dest.mkdir(parents=True, exist_ok=True)
    for split, lines in out.items():
        (dest / f"{split}.jsonl").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"sessions": len(sessions), "test_sessions": sorted(test),
                      "train_examples": counts["train"], "test_examples": counts["test"],
                      "out": str(dest)}))
    return 0


def cmd_stats(a) -> int:
    root = Path(a.root)
    files = sorted(root.rglob("examples.jsonl"))
    tot = pos = 0
    per = []
    for f in files:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        p = sum(r["label"] for r in rows)
        tot += len(rows); pos += p
        per.append({"session": f.parent.name, "examples": len(rows), "respond": p})
    print(json.dumps({"sessions": len(files), "examples": tot, "respond": pos,
                      "let_pass": tot - pos,
                      "positive_rate": round(pos / max(tot, 1), 3), "per_session": per},
                     indent=2))
    return 0


def cmd_batch(a) -> int:
    """Segment + label a whole directory of episodes, resumably.

    A podcast archive is hundreds of hours; a crash four hours in must not cost the first
    four. Each episode writes into its own directory and is skipped if segments.jsonl already
    exists, so this can be stopped and restarted freely. Whisper holds the GPU, so episodes
    run one at a time by design -- the parallelism that helps here is running it overnight,
    not fighting over VRAM.
    """
    src = Path(a.audio_dir)
    if not src.is_dir():
        return _fail(f"not a directory: {src}")
    exts = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".mp4"}
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    if a.limit:
        files = files[:a.limit]
    if not files:
        return _fail(f"no audio under {src} (looked for {sorted(exts)})")

    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = [], [], []
    for i, f in enumerate(files, 1):
        dest = out_root / f.stem[:60].replace(" ", "_")
        if (dest / "examples.jsonl").exists() and not a.force:
            skipped.append(f.name); continue
        print(f"[{i}/{len(files)}] {f.name}", file=sys.stderr)
        try:
            if not (dest / "segments.jsonl").exists() or a.force:
                seg = argparse.Namespace(audio=str(f), out=str(dest), whisper=a.whisper,
                                         device=a.device, limit=0, progress=a.progress)
                if cmd_segment(seg) != 0:
                    failed.append(f.name); continue
            lab = argparse.Namespace(session=str(dest), speakers=a.speakers, dm_speaker=-1)
            if cmd_label(lab) != 0:
                failed.append(f.name); continue
            done.append(f.name)
        except KeyboardInterrupt:
            print(json.dumps({"interrupted_after": len(done), "done": done,
                              "skipped": skipped, "failed": failed}))
            return 130
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(f.name)
    print(json.dumps({"episodes": len(files), "processed": len(done),
                      "skipped_already_done": len(skipped), "failed": failed,
                      "out": str(out_root)}))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("segment", help="audio -> speech segments with text and embeddings")
    s.add_argument("--audio", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--whisper", default="large-v3-turbo")
    s.add_argument("--device", default="cuda")
    s.add_argument("--limit", type=int, default=0, help="first N regions only (smoke test)")
    s.add_argument("--progress", action="store_true")
    s.set_defaults(fn=cmd_segment)

    s = sub.add_parser("label", help="segments -> (respond / let it pass) examples")
    s.add_argument("--session", required=True)
    s.add_argument("--speakers", type=int, default=6)
    s.add_argument("--dm-speaker", type=int, default=-1, help="-1 = whoever talks most")
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("split", help="hold out whole sessions into train/test")
    s.add_argument("--root", default=str(ROOT / "data" / "raw"))
    s.add_argument("--test-frac", type=float, default=0.25)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("stats", help="what the corpus contains")
    s.add_argument("--root", default=str(ROOT / "data" / "raw"))
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("batch", help="segment+label a whole archive, resumable")
    s.add_argument("--audio-dir", required=True)
    s.add_argument("--out", default=str(ROOT / "data" / "raw"))
    s.add_argument("--whisper", default="large-v3-turbo")
    s.add_argument("--device", default="cuda")
    s.add_argument("--speakers", type=int, default=8)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--force", action="store_true", help="redo episodes already processed")
    s.add_argument("--progress", action="store_true")
    s.set_defaults(fn=cmd_batch)

    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
