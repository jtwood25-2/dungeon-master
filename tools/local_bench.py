"""Which local model should narrate, and can this GPU hold it?

The DM's slowest path is narration, and narration got much easier the day the engine started
resolving combat itself: the model is no longer deciding what happened, only saying it. That
is a job a 7B does in a fifth of a second, against ten to thirty seconds through Claude Code.

So the question is no longer "is local good enough to be the DM" -- it is "which local model
says these facts best, and what has to come off the card to fit it".

Measured per model:
  first_token_ms   what the player actually waits for
  total_ms         whole reply
  faithful         does it state the numbers it was given, and invent none
  style            distance from the reference DM's measured prose (tools/style.py)
  brevity          words per reply; the reference table averages 41

Usage
  python tools/local_bench.py --models qwen2.5:7b-instruct,qwen2.5:14b-instruct
  python tools/local_bench.py --vram          # just the memory report
  python tools/local_bench.py --pull mistral-nemo:12b --models mistral-nemo:12b
"""
import argparse
import json
import re
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))

OLLAMA = "http://127.0.0.1:11434"

SYS = ("You are a Dungeon Master running a live tabletop RPG by VOICE. Every word you write is "
       "spoken aloud. No markdown, no asterisks, no stage directions, no lists. Say what "
       "happened in one or two short sentences and stop.")

# Each case carries the facts the engine already settled, plus the numbers that MUST survive
# into the reply. That is what makes faithfulness measurable rather than a matter of taste.
CASES = [
    {"name": "two monster attacks",
     "msg": ("[COMBAT - already rolled and applied]\n"
             "> River Wraith C attacks Oktah: Claw hits for 4 (19 vs AC 15). Oktah is now 8/12.\n"
             "> River Wraith B attacks Oktah: Claw misses (9 vs AC 15).\n"
             "[HOW TO SAY IT] Narrate both briefly, then tell Oktah he is up.\n"
             "[PLAYER] What just happened?"),
     "must": ["4", "8"], "never": ["19", "15", "9", "AC"]},
    {"name": "a hit and a kill",
     "msg": ("[COMBAT - already rolled and applied]\n"
             "> Goblin A attacks Thorn: Scimitar hits for 6 (18 vs AC 14). Thorn is now 5/16.\n"
             "> Goblin B is down.\n"
             "[HOW TO SAY IT] Narrate both, then hand back to Thorn.\n"
             "[PLAYER] I swing at the second one."),
     "must": ["6"], "never": ["18", "14", "AC"]},
    {"name": "a death save",
     "msg": ("[COMBAT - already rolled and applied]\n"
             "> death save 15 - success, 1 success / 0 fail\n"
             "[HOW TO SAY IT] Say what the save was and that they are still going. One beat.\n"
             "[PLAYER] Am I still alive?"),
     "must": [], "never": ["15", "DC"]},
    {"name": "a plain question",
     "msg": ("[STATE] Oktah of the Wood, level 1 druid, 8/12 hit points, AC 15.\n"
             "[PLAYER] How badly hurt am I?"),
     "must": ["8"], "never": []},
    {"name": "scene, no mechanics",
     "msg": ("[STATE] The party is in a flooded mine tunnel, timbered walls, water underfoot.\n"
             "[PLAYER] What do I see?"),
     "must": [], "never": []},
]


# Spoken aloud, "a fifteen" leaks the die roll exactly as "15" does. Matching only digits
# scored gemma3 as leaking nothing while it was saying "That was a fifteen, a success!"
_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
          "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
          "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
          "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen",
          "20": "twenty"}


def _says_number(text: str, n: str) -> bool:
    """Is this number in the reply, as a digit OR as the word for it?"""
    if re.search(rf"\b{re.escape(n)}\b", text):
        return True
    w = _WORDS.get(str(n))
    return bool(w and re.search(rf"\b{w}\b", text, re.I))


def _chat(model: str, msg: str, num_predict=120, timeout=180) -> tuple:
    body = json.dumps({
        "model": model, "stream": True, "keep_alive": "10m",
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
        "options": {"num_predict": num_predict, "temperature": 0.7},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ft, out = None, ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            c = d.get("message", {}).get("content", "")
            if c and ft is None:
                ft = (time.perf_counter() - t0) * 1000
            out += c
            if d.get("done"):
                break
    return ft or 0.0, (time.perf_counter() - t0) * 1000, out.strip()


def vram() -> dict:
    """What is on the card, and what is left for a bigger model."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used, total = (int(x.strip()) for x in q.stdout.strip().split(","))
    loaded = []
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=5) as r:
            for m in json.loads(r.read()).get("models", []):
                loaded.append({"model": m["name"],
                               "vram_mb": round(m.get("size_vram", 0) / 1048576)})
    except Exception:
        pass
    ol = sum(m["vram_mb"] for m in loaded)
    return {"gpu_total_mb": total, "gpu_used_mb": used, "free_mb": total - used,
            "ollama_models": loaded, "ollama_total_mb": ol,
            "ours_mb": used - ol,
            "headroom_if_ollama_unloaded_mb": total - (used - ol),
            "note": "ours_mb is whisper + kokoro + the CUDA context they hold"}


def _style_gap(text: str) -> dict:
    """How far this reads from the reference DM's prose, using the measured tells."""
    import style
    m = style.measure([text])
    if not m:
        return {}
    return {"words_per_sentence": m.get("words_per_sentence"),
            "em_dash_per_1k": m.get("em_dash_per_1k"),
            "fluff_per_1k": m.get("fluff_per_1k"),
            "vague_some_per_1k": m.get("vague_some_per_1k")}


def bench(model: str, reps: int) -> dict:
    fts, tots, words, faith, leaks, samples = [], [], [], [], [], []
    for case in CASES:
        for i in range(reps):
            try:
                ft, tot, txt = _chat(model, case["msg"])
            except Exception as e:
                return {"model": model, "error": f"{type(e).__name__}: {e}"}
            if i == 0 and len(samples) < 3:
                samples.append({"case": case["name"], "said": txt[:200]})
            fts.append(ft); tots.append(tot); words.append(len(txt.split()))
            faith.append(all(_says_number(txt, n) for n in case["must"]))
            # numbers the player must never hear: the roll, the AC, the DC
            leaks.append(sum(1 for n in case["never"] if _says_number(txt, n)))
    body = " ".join(s["said"] for s in samples)
    return {"model": model,
            "first_token_ms_p50": round(st.median(fts)),
            "first_token_ms_p90": round(sorted(fts)[int(len(fts) * .9)]),
            "total_ms_p50": round(st.median(tots)),
            "words_p50": round(st.median(words)),
            "faithful_pct": round(100 * sum(faith) / len(faith)),
            "leaked_numbers": sum(leaks),
            "style": _style_gap(body),
            "samples": samples}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="qwen2.5:7b-instruct,qwen2.5:14b-instruct")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--vram", action="store_true", help="just the memory report")
    p.add_argument("--pull", default="", help="ollama pull this first")
    a = p.parse_args()

    if a.vram:
        print(json.dumps(vram(), indent=2)); return 0
    if a.pull:
        subprocess.run(["ollama", "pull", a.pull], check=False)

    out = {"before": vram(), "models": []}
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        _chat(m, "hello", num_predict=4)          # warm / load
        out["models"].append(bench(m, a.reps))
    out["after"] = vram()
    ok = [m for m in out["models"] if not m.get("error")]
    if ok:
        best = min(ok, key=lambda m: (m["leaked_numbers"], -m["faithful_pct"],
                                      m["first_token_ms_p50"]))
        out["recommended"] = best["model"]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
