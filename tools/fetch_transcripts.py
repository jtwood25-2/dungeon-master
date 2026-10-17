"""Fetch episode transcripts from the Critical Role wiki via the MediaWiki API.

The wiki blocks plain scrapes (403) but exposes api.php normally, which is also the polite
route: one request per page, a real User-Agent, and a delay between calls.

WHY THESE ARE OPTIONAL. The training labels come from audio TIMING -- when someone stopped and
whether the DM took the floor -- so the pipeline never needed a transcript to produce examples.
What the transcripts add is GROUND TRUTH for the two things we otherwise can only guess at:

  who spoke   -- validates the speaker clustering, which is currently unverified
  what words  -- gives a reference to measure Whisper's error rate against (tools/eval_stt.py)

So they are a measurement instrument, not the training data. Downloaded to disk for local use.

Usage
  python tools/fetch_transcripts.py list --campaign 2
  python tools/fetch_transcripts.py fetch --match "C2E5" --out data/transcripts
  python tools/fetch_transcripts.py fetch --titles "Xhorhas/Transcript" --out data/transcripts
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
API = "https://criticalrole.fandom.com/api.php"
UA = "dungeon-master-research/0.1 (local ML dataset prep; contact: local user)"
DELAY = 1.0                    # be a good citizen; the wiki is free and community-run


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def all_transcript_pages() -> list:
    """Every page in Category:Transcripts, following continuation."""
    out, cont = [], {}
    while True:
        d = _get({"action": "query", "list": "categorymembers",
                  "cmtitle": "Category:Transcripts", "cmlimit": "500", **cont})
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        if "continue" not in d:
            return out
        cont = d["continue"]
        time.sleep(DELAY)


def wikitext(title: str) -> str:
    d = _get({"action": "parse", "page": title, "prop": "wikitext"})
    return d.get("parse", {}).get("wikitext", {}).get("*", "")


_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S)
_TAG = re.compile(r"<[^>]+>")
_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")


def to_markdown(title: str, wt: str) -> str:
    """Wikitext -> plain markdown, keeping the SPEAKER: line structure.

    That structure is the whole point: it is what makes the file usable as a speaker-label
    reference. Templates, refs and link markup are stripped; headings become markdown headings
    so section boundaries (notably the break) survive.
    """
    t = _REF.sub("", wt)
    for _ in range(3):                        # templates nest
        t = _TEMPLATE.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _TAG.sub("", t)
    t = re.sub(r"^=====\s*(.*?)\s*=====$", r"##### \1", t, flags=re.M)
    t = re.sub(r"^====\s*(.*?)\s*====$", r"#### \1", t, flags=re.M)
    t = re.sub(r"^===\s*(.*?)\s*===$", r"### \1", t, flags=re.M)
    t = re.sub(r"^==\s*(.*?)\s*==$", r"## \1", t, flags=re.M)
    t = t.replace("'''", "**").replace("''", "*")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return f"# {title}\n\n{t}\n"


def speaker_lines(md: str) -> list:
    """Rows of {speaker, text} for the alignment/validation tools.

    Wiki transcripts are written as "**NAME:** words". Anything that doesn't match that shape
    is stage direction or a heading and is dropped rather than guessed at.
    """
    rows = []
    for line in md.splitlines():
        m = re.match(r"^\*{0,2}([A-Z][A-Za-z' .-]{1,24}):\*{0,2}\s+(.*\S)\s*$", line.strip())
        if m:
            rows.append({"speaker": m.group(1).strip().upper(), "text": m.group(2).strip()})
    return rows


def cmd_list(a) -> int:
    pages = all_transcript_pages()
    if a.campaign:
        pat = re.compile(rf"\bC{a.campaign}E\d+", re.I)
        pages = [p for p in pages if pat.search(p)]
    if a.match:
        pages = [p for p in pages if a.match.lower() in p.lower()]
    print(json.dumps({"pages": len(pages), "titles": pages[:a.limit or len(pages)]}, indent=2))
    return 0


def cmd_fetch(a) -> int:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.titles:
        titles = [t.strip() for t in a.titles.split("|") if t.strip()]
    else:
        titles = all_transcript_pages()
        if a.match:
            titles = [t for t in titles if a.match.lower() in t.lower()]
    if a.limit:
        titles = titles[:a.limit]
    if not titles:
        print(json.dumps({"error": "no pages matched"}))
        return 1

    done, failed = [], []
    for i, title in enumerate(titles, 1):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:80]
        dest = out / f"{slug}.md"
        if dest.exists() and not a.force:
            continue
        try:
            md = to_markdown(title, wikitext(title))
            dest.write_text(md, encoding="utf-8")
            rows = speaker_lines(md)
            (out / f"{slug}.speakers.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            done.append({"title": title, "md": dest.name, "lines": len(rows),
                         "speakers": len({r["speaker"] for r in rows})})
            print(f"[{i}/{len(titles)}] {title} — {len(rows)} lines", file=sys.stderr)
        except Exception as e:
            failed.append({"title": title, "error": f"{type(e).__name__}: {e}"})
        time.sleep(DELAY)
    print(json.dumps({"fetched": len(done), "failed": failed, "out": str(out),
                      "episodes": done}, indent=2))
    return 0


EP_RE = re.compile(r"^Critical\s*Role[-_ ]+C(\d+)E(\d+)\s+(.*)$", re.I)


def episode_name(path: Path) -> tuple:
    """('2','51','Xhorhas') from 'Critical Role-C2E51 Xhorhas.mp3'."""
    m = EP_RE.match(path.stem)
    return (m.group(1), m.group(2), m.group(3).strip()) if m else ("", "", path.stem)


def match_pages(paths: list, pages: list) -> list:
    """Map local audio to wiki pages by episode NAME, not episode code.

    The wiki titles pages by episode name; the audio is named by code AND name. Exact match
    first, then a disambiguated form -- 'Xhorhas' is filed as 'Xhorhas (episode)' because the
    plain name is taken by the location -- then a conservative fuzzy fallback. Anything that
    does not clear the bar is reported unmatched rather than paired with a plausible-looking
    wrong episode, because a mismatched transcript silently corrupts every label built from it.
    """
    import difflib
    low = {p.lower(): p for p in pages}
    out = []
    for f in paths:
        camp, ep, name = episode_name(f)
        cand = (low.get(f"{name.lower()}/transcript")
                or low.get(f"{name.lower()} (episode)/transcript"))
        score = 1.0
        if not cand:
            near = difflib.get_close_matches(f"{name}/Transcript", pages, n=1, cutoff=0.82)
            if near:
                cand = near[0]
                score = difflib.SequenceMatcher(None, f"{name}/Transcript", cand).ratio()
        out.append({"audio": str(f), "campaign": camp, "episode": ep, "name": name,
                    "page": cand, "confidence": round(score, 3) if cand else 0.0})
    return out


def cmd_match(a) -> int:
    audio = sorted(Path(a.audio_dir).glob("*.mp3")) + sorted(Path(a.audio_dir).glob("*.m4a"))
    if not audio:
        print(json.dumps({"error": f"no audio in {a.audio_dir}"}))
        return 1
    rows = match_pages(audio, all_transcript_pages())
    miss = [r["name"] for r in rows if not r["page"]]
    print(json.dumps({"audio_files": len(audio),
                      "matched": sum(1 for r in rows if r["page"]),
                      "unmatched": miss,
                      "pairs": [{k: v for k, v in r.items() if k != "audio"} for r in rows]},
                     indent=2))
    return 0


def cmd_pair(a) -> int:
    """Fetch exactly the transcripts that match the local audio, named to line up with it."""
    audio = sorted(Path(a.audio_dir).glob("*.mp3")) + sorted(Path(a.audio_dir).glob("*.m4a"))
    rows = [r for r in match_pages(audio, all_transcript_pages()) if r["page"]]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for i, r in enumerate(rows, 1):
        stem = Path(r["audio"]).stem
        dest = out / f"{stem}.md"
        if dest.exists() and not a.force:
            continue
        try:
            md = to_markdown(r["page"], wikitext(r["page"]))
            dest.write_text(md, encoding="utf-8")
            lines = speaker_lines(md)
            (out / f"{stem}.speakers.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
            done.append({"audio": stem, "page": r["page"], "lines": len(lines),
                         "speakers": sorted({x["speaker"] for x in lines})[:12]})
            print(f"[{i}/{len(rows)}] {r['name']} — {len(lines)} lines", file=sys.stderr)
        except Exception as e:
            failed.append({"page": r["page"], "error": f"{type(e).__name__}: {e}"})
        time.sleep(DELAY)
    print(json.dumps({"paired": len(done), "failed": failed, "out": str(out),
                      "episodes": done}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--campaign", default="")
    s.add_argument("--match", default=""); s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("fetch")
    s.add_argument("--match", default=""); s.add_argument("--titles", default="")
    s.add_argument("--out", default=str(ROOT / "data" / "transcripts"))
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("match", help="map local audio files to wiki pages")
    s.add_argument("--audio-dir", default=str(ROOT / "critical_role_sessions"))
    s.set_defaults(fn=cmd_match)
    s = sub.add_parser("pair", help="fetch transcripts matching the local audio")
    s.add_argument("--audio-dir", default=str(ROOT / "critical_role_sessions"))
    s.add_argument("--out", default=str(ROOT / "data" / "transcripts"))
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_pair)
    a = p.parse_args()
    try:
        return a.fn(a)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
