"""Which (embedder, clusterer) actually knows who spoke?

Speaker attribution is the weak link now: 0.56 against ground truth once the VAD was fixed,
where chance across eight speakers is 0.125. It matters more than word error rate, because a
wrong name is used as fact -- the labels for the whole turn-taking corpus are built on "was
that the DM", and every one of them inherits this number.

Scored against the wiki transcripts' speaker labels, using Hungarian assignment so
"cluster 3 is Matt" is chosen globally rather than greedily; greedy lets the dominant speaker
(Matt, by a wide margin) absorb every ambiguous cluster and flatters the result.

Segments are computed ONCE and reused for every combination, so differences are the method
and not a re-run of the VAD.

Usage
  python tools/diarize_bakeoff.py --audio data/clean/<ep>__partII.wav \
      --transcript data/transcripts/<ep>.md --part "Part II" --minutes 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))


def _fail(msg: str) -> int:
    print(json.dumps({"error": msg}))
    return 1


# ------------------------------------------------------------------ clusterers
def cluster_agglomerative(X, k, linkage="average", metric="cosine"):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=k, metric=metric,
                                   linkage=linkage).fit_predict(X)


def cluster_agglomerative_ward(X, k):
    from sklearn.cluster import AgglomerativeClustering
    # ward needs euclidean; on unit-norm vectors euclidean is monotone in cosine, and ward
    # resists the chaining that average-linkage suffers when one speaker dominates
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def cluster_spectral(X, k):
    """The standard choice in diarisation. Builds an affinity graph first, so it separates
    speakers that overlap in raw distance but sit in different neighbourhoods."""
    from sklearn.cluster import SpectralClustering
    A = np.clip((X @ X.T + 1.0) / 2.0, 0, 1)      # cosine -> [0,1] affinity
    return SpectralClustering(n_clusters=k, affinity="precomputed",
                              assign_labels="kmeans", random_state=0).fit_predict(A)


def cluster_spectral_pruned(X, k, p=0.60):
    """Spectral with row-wise pruning: keep each segment's strongest affinities and drop the
    rest before symmetrising. Weak edges between different speakers are what let clusters
    bleed together, and they are exactly the ones this removes."""
    from sklearn.cluster import SpectralClustering
    A = np.clip((X @ X.T + 1.0) / 2.0, 0, 1)
    thr = np.quantile(A, p, axis=1, keepdims=True)
    A = np.where(A >= thr, A, 0.0)
    A = np.maximum(A, A.T)
    return SpectralClustering(n_clusters=k, affinity="precomputed",
                              assign_labels="kmeans", random_state=0).fit_predict(A)


def cluster_kmeans(X, k):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


CLUSTERERS = {
    "agglomerative/average (current)": lambda X, k: cluster_agglomerative(X, k),
    "agglomerative/complete": lambda X, k: cluster_agglomerative(X, k, linkage="complete"),
    "agglomerative/ward": cluster_agglomerative_ward,
    "kmeans": cluster_kmeans,
    "spectral": cluster_spectral,
    "spectral/pruned": cluster_spectral_pruned,
}


def score(labels, segs, ref_rows, k) -> dict:
    import eval_stt as ev
    ref_words = [(r["speaker"], w) for r in ref_rows for w in ev.words(r["text"])]
    hyp_words = [(labels[i], w) for i, s in enumerate(segs) for w in ev.words(s["text"])]
    # Aligned, not positional. Pairing by index measured transcription drift, and it reversed
    # the embedder verdict outright: CAM++ scored 0.47 by position and 0.81 once aligned.
    pairs = ev.align_words([w for _c, w in hyp_words], [w for _t, w in ref_words])
    n = len(pairs)
    if not n:
        return {"accuracy": 0.0, "words": 0}
    mapping = ev.assign_clusters([hyp_words[i][0] for i, _j in pairs],
                                 [ref_words[j][0] for _i, j in pairs], k)
    correct = sum(1 for i, j in pairs if mapping.get(hyp_words[i][0]) == ref_words[j][0])
    named = {v for v in mapping.values()}
    return {"accuracy": round(correct / n, 4), "words": n, "distinct_names": len(named)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--part", default="")
    p.add_argument("--minutes", type=float, default=12.0)
    p.add_argument("--speakers", type=int, default=8)
    p.add_argument("--embedder", default="", help="path to a speaker ONNX; default is the one the server uses")
    p.add_argument("--ks", default="", help="also sweep cluster counts, e.g. 6,7,8,9,10")
    a = p.parse_args()

    import eval_stt as ev
    wav, md = Path(a.audio), Path(a.transcript)
    if not wav.exists() or not md.exists():
        return _fail("need --audio and --transcript")

    t0 = time.perf_counter()
    if a.embedder:
        import speakers
        speakers.MODEL = Path(a.embedder)
        speakers._NEURAL = None                # force a reload with the requested model
    pcm = ev.load_wav(wav, a.minutes)
    segs = ev.transcribe_segments(pcm, want_embeddings=True)
    segs = [s for s in segs if isinstance(s.get("emb"), list)]
    if len(segs) < a.speakers:
        return _fail("not enough embedded segments")
    X = np.array([s["emb"] for s in segs], dtype=float)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    ref_rows = ev.reference_text(md, a.part)
    prep = time.perf_counter() - t0

    rows = []
    for name, fn in CLUSTERERS.items():
        try:
            t1 = time.perf_counter()
            labels = fn(X, a.speakers)
            rows.append({"method": name, "k": a.speakers,
                         **score(labels, segs, ref_rows, a.speakers),
                         "seconds": round(time.perf_counter() - t1, 2)})
        except Exception as e:
            rows.append({"method": name, "error": f"{type(e).__name__}: {str(e)[:90]}"})

    sweep = []
    if a.ks:
        best = max((r for r in rows if "accuracy" in r), key=lambda r: r["accuracy"])
        fn = CLUSTERERS[best["method"]]
        for k in [int(x) for x in a.ks.split(",") if x.strip()]:
            try:
                sweep.append({"k": k, "method": best["method"],
                              **score(fn(X, k), segs, ref_rows, k)})
            except Exception as e:
                sweep.append({"k": k, "error": str(e)[:80]})

    rows.sort(key=lambda r: -(r.get("accuracy") or 0))
    print(json.dumps({"audio": wav.name, "minutes": a.minutes,
                      "segments": len(segs), "prep_seconds": round(prep, 1),
                      "chance": round(1 / a.speakers, 3),
                      "results": rows, "k_sweep": sweep,
                      "note": "attribution against the wiki speaker labels; Hungarian mapping"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
