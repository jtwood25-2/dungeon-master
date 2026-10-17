"""Dense LOCAL cache of open/CC/OGL rules content, for fast offline lookup by the DM.

Sources are open-licensed only (Open5e aggregates SRD + OGL/CC third-party open content;
dnd5eapi is the CC SRD). Copyrighted books are never touched. build() downloads everything
once; Index loads it into name-keyed dicts so lookups are in-memory and instant.

  python server/srd_cache.py            # build/refresh the cache, print counts
"""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

CACHE = Path(__file__).parent.parent / "state" / "cache" / "srd"
OPEN5E = "https://api.open5e.com/v1"

# open5e endpoints worth caching -> local file name
CATEGORIES = ["monsters", "spells", "magicitems", "weapons", "armor", "conditions",
              "feats", "races", "classes", "backgrounds", "sections", "spelllist", "planes"]


def _fetch_all(category: str) -> list[dict]:
    """Page through an open5e list endpoint following `next` until exhausted."""
    url = f"{OPEN5E}/{category}/?limit=500"
    out = []
    while url:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dungeon-master/0.1"})
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read())
        except Exception as e:
            print(f"  {category}: stopped ({e})")
            break
        out += d.get("results", [])
        url = d.get("next")
        time.sleep(0.15)
    return out


def build(refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cat in CATEGORIES:
        path = CACHE / f"{cat}.json"
        if path.exists() and not refresh:
            counts[cat] = len(json.loads(path.read_text()))
            continue
        print(f"downloading {cat} ...")
        rows = _fetch_all(cat)
        if rows:
            path.write_text(json.dumps(rows))
        counts[cat] = len(rows)
        print(f"  {cat}: {len(rows)}")
    (CACHE / "_index.json").write_text(json.dumps({"counts": counts, "built": time.time()}))
    return counts


# Which book a bare name should resolve to. The same creature exists in several open books —
# "aboleth" appears in 5e Core, Level Up's Monstrous Menagerie and the Black Flag SRD — and
# keying only on the name silently threw 897 monsters away, including every Level Up variant.
_PREFER = ["5e core rules", "systems reference document", "black flag", "level up",
           "tome of beasts", "creature codex"]


def _rank(rec: dict) -> int:
    doc = (rec.get("document__title") or rec.get("document__slug") or "").lower()
    for i, p in enumerate(_PREFER):
        if p in doc:
            return i
    return len(_PREFER)


class Index:
    """In-memory view of the cache. Load once; look up instantly.

    Every record stays reachable: the bare name resolves to the most familiar version, and the
    other books' takes on the same creature are kept as variants (which is where the more
    interesting stat blocks live).
    """
    def __init__(self):
        self.by_cat: dict[str, dict[str, dict]] = {}     # category -> {name|slug: record}
        self.flat: dict[str, tuple[str, dict]] = {}       # name_lower -> (category, record)
        self.records: dict[str, list] = {}                # category -> every record
        self.loaded = False

    def load(self):
        for cat in CATEGORIES:
            p = CACHE / f"{cat}.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text())
            self.records[cat] = rows
            d: dict[str, dict] = {}
            for rec in rows:
                nm = (rec.get("name") or "").lower()
                slug = (rec.get("slug") or rec.get("key") or "").lower()
                if slug:
                    d[slug] = rec                         # unique -> nothing is unreachable
                if nm and (nm not in d or _rank(rec) < _rank(d[nm])):
                    d[nm] = rec                           # bare name -> the familiar version
                if nm:
                    self.flat.setdefault(nm, (cat, rec))
            self.by_cat[cat] = d
        self.loaded = True
        return self

    def variants(self, name: str, category: str = "monsters") -> list[dict]:
        """Every book's version of this creature — the alternative stat blocks."""
        nm = name.strip().lower()
        return [r for r in self.records.get(category, [])
                if (r.get("name") or "").lower() == nm]

    def total(self) -> int:
        return sum(len(v) for v in self.records.values())

    def get(self, name: str, category: str | None = None) -> tuple[str, dict] | None:
        nm = name.strip().lower()
        pools = ([(category, self.by_cat.get(category, {}))] if category
                 else [(c, d) for c, d in self.by_cat.items()])
        for cat, pool in pools:                          # exact
            if nm in pool:
                return cat, pool[nm]
        for cat, pool in pools:                          # prefix / substring
            for key, rec in pool.items():
                if key.startswith(nm) or nm in key:
                    return cat, rec
        return None

    def search(self, query: str, limit: int = 8) -> list[tuple[str, str]]:
        q = query.strip().lower()
        hits = [(rec.get("name", k), cat) for k, (cat, rec) in self.flat.items() if q in k]
        return hits[:limit]


if __name__ == "__main__":
    import sys
    counts = build(refresh="--refresh" in sys.argv)
    idx = Index().load()
    print(json.dumps({"counts": counts, "total_indexed": idx.total()}, indent=2))
