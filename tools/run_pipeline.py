"""Drive the whole corpus pipeline unattended, and survive being interrupted.

align -> crop -> segment -> label -> split -> train -> eval, over every episode.

Built to be launched detached and left alone for hours. Whisper holds the GPU so episodes run
one at a time; the parallelism that helps here is wall-clock, not threads. Every stage writes
its artefact and is skipped when that artefact already exists, so killing this at any point
costs only the episode in flight.

Progress goes to state/pipeline.log as it happens rather than being buffered to the end -- a
job that prints nothing for three hours is indistinguishable from a hung one, which is exactly
how the first alignment run wasted twenty minutes.

Usage
  python tools/run_pipeline.py --audio-dir critical_role_sessions --transcripts data/transcripts
  python tools/run_pipeline.py --stage segment       # resume from one stage
  python tools/run_pipeline.py --status              # what is done, what is left
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = sys.executable
LOG = ROOT / "state" / "pipeline.log"


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list, why: str) -> dict:
    say(f"  $ {' '.join(str(a) for a in args[1:])}")
    t0 = time.time()
    p = subprocess.run([PY] + args, capture_output=True, text=True, cwd=ROOT)
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        say(f"  ! {why} failed rc={p.returncode}: {(p.stderr or out)[-300:]}")
        return {"ok": False}
    say(f"  . {why} in {time.time() - t0:.0f}s")
    try:
        return {"ok": True, **json.loads(out[out.index("{"):])}
    except Exception:
        return {"ok": True, "raw": out[-200:]}


def episodes(audio_dir: Path, transcripts: Path) -> list:
    out = []
    for mp3 in sorted(audio_dir.glob("*.mp3")):
        md = transcripts / f"{mp3.stem}.md"
        if md.exists():
            out.append((mp3, md))
    return out


def status(eps: list) -> dict:
    d = {"episodes": len(eps), "aligned": 0, "cropped": 0, "segmented": 0, "labelled": 0}
    for mp3, _ in eps:
        d["aligned"] += (ROOT / "data" / "align" / f"{mp3.stem}.json").exists()
        d["cropped"] += len(list((ROOT / "data" / "clean").glob(f"{mp3.stem}__*.wav"))) > 0
        for part in ("partI", "partII"):
            sess = ROOT / "data" / "raw" / f"{mp3.stem}__{part}"
            d["segmented"] += (sess / "segments.jsonl").exists()
            d["labelled"] += (sess / "examples.jsonl").exists()
    d["split"] = (ROOT / "data" / "splits" / "train.jsonl").exists()
    d["model"] = (ROOT / "state" / "models" / "endpoint" / "endpoint.joblib").exists()
    return d


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", default=str(ROOT / "critical_role_sessions"))
    p.add_argument("--transcripts", default=str(ROOT / "data" / "transcripts"))
    p.add_argument("--speakers", type=int, default=8)
    p.add_argument("--band", type=float, default=6.0)
    p.add_argument("--stage", default="", help="align|crop|segment|label|split|train")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()

    eps = episodes(Path(a.audio_dir), Path(a.transcripts))
    if a.status:
        print(json.dumps(status(eps), indent=2))
        return 0
    if not eps:
        say("no episode/transcript pairs found"); return 1

    only = a.stage
    say(f"=== pipeline start: {len(eps)} episodes ===")

    if not only or only == "align":
        for i, (mp3, md) in enumerate(eps, 1):
            dest = ROOT / "data" / "align" / f"{mp3.stem}.json"
            if dest.exists():
                continue
            say(f"[align {i}/{len(eps)}] {mp3.stem[:48]}")
            run(["tools/prep_audio.py", "align", "--audio", str(mp3),
                 "--transcript", str(md), "--band", str(a.band)], "align")

    if not only or only == "crop":
        say("[crop] all aligned episodes")
        run(["tools/prep_audio.py", "crop", "--all",
             "--audio-dir", a.audio_dir, "--out", str(ROOT / "data" / "clean")], "crop")

    if not only or only in ("segment", "label"):
        clean = sorted((ROOT / "data" / "clean").glob("*.wav"))
        for i, wav in enumerate(clean, 1):
            sess = ROOT / "data" / "raw" / wav.stem
            if (sess / "examples.jsonl").exists():
                continue
            say(f"[segment {i}/{len(clean)}] {wav.stem[:48]}")
            if not (sess / "segments.jsonl").exists():
                r = run(["tools/corpus.py", "segment", "--audio", str(wav),
                         "--out", str(sess)], "segment")
                if not r.get("ok"):
                    continue
            run(["tools/corpus.py", "label", "--session", str(sess),
                 "--speakers", str(a.speakers)], "label")

    if not only or only == "split":
        say("[split] holding out whole sessions")
        run(["tools/corpus.py", "split", "--root", str(ROOT / "data" / "raw"),
             "--test-frac", "0.25"], "split")

    if not only or only == "train":
        say("[train] endpoint model")
        run(["tools/train_endpoint.py", "train"], "train")
        say("[eval] against the live rule")
        r = run(["tools/train_endpoint.py", "eval"], "eval")
        if r.get("model"):
            say(f"  model precision {r['model']['precision']} vs rule "
                f"{r['rule_in_production']['precision']}; "
                f"interruptions {r['model']['spoke_over_table']} vs "
                f"{r['rule_in_production']['spoke_over_table']} -> {r.get('verdict')}")
        run(["tools/train_endpoint.py", "sweep"], "sweep")

    say(f"=== pipeline done === {json.dumps(status(eps))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
