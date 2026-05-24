"""Stage 5: joint clustering of gold + tracing, gap_score ranking.

For each cluster, compute four signals and combine into a single gap_score:

    gap_score(C) = w1*(1 - sim_to_gold(C))
                 + w2*log(1 + |C_tracing|)
                 - w3*gold_share(C)
                 + w4*rag_relevance(C)

Writes data/processed/gap_clusters.json, ranked descending by gap_score.
The top-K clusters are the input to stage 6 (sample_candidates.py).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")

GOLD_FILE    = PROCESSED / "crag_domain_open_ai_embeddings_subclass.json"
TRACING_FILE = PROCESSED / "chatbot_arena_open_ai_embeddings_sample3000_subclass_en.json"
EMB_COL      = "open_ai_embeddings"
OUTPUT       = PROCESSED / "gap_clusters.json"

# Weights for gap_score. Tunable; keep sum-of-positives close to magnitude of gold_share.
WEIGHTS = {"sim": 1.0, "size": 0.4, "gold_share": 1.5, "rag_relevance": 0.6}
JOINT_K = 60      # joint cluster count; bigger = finer gaps but more cluster noise
TOP_K_GAPS = 25   # how many top-ranked clusters to keep in the output

RAG_RELEVANCE_PAT = re.compile(
    r"\b(who|what|when|where|why|how many|how much|how long|how old|"
    r"list|name|find|cite|source|reference|according to|"
    r"latest|newest|current|today|yesterday|in \d{4})\b",
    re.IGNORECASE,
)


def load_records(path: Path):
    with path.open() as f:
        data = json.load(f)
    X = np.vstack([np.asarray(r[EMB_COL], dtype=np.float32) for r in data])
    queries = [r["query"] for r in data]
    classes = [str(r.get("class", "")).lower() for r in data]
    return X, queries, classes


def rag_relevance(query: str) -> float:
    """Crude proxy: does this look like a fact-lookup the RAG layer should answer?"""
    if not isinstance(query, str) or not query.strip():
        return 0.0
    return float(bool(RAG_RELEVANCE_PAT.search(query)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=JOINT_K)
    parser.add_argument("--top", type=int, default=TOP_K_GAPS)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    G, q_g, c_g = load_records(GOLD_FILE)
    T, q_t, c_t = load_records(TRACING_FILE)
    n_g, n_t = G.shape[0], T.shape[0]
    print(f"gold n={n_g} tracing n={n_t} dim={G.shape[1]}")

    pool = np.vstack([G, T])
    queries = q_g + q_t
    classes = c_g + c_t
    source  = np.array(["gold"] * n_g + ["tracing"] * n_t)

    km = KMeans(n_clusters=args.k, n_init=10, random_state=42).fit(pool)
    labels = km.labels_
    centroids = km.cluster_centers_

    # Pre-compute centroid-to-centroid cosine sim for "nearest gold centroid".
    # Define a centroid as "gold-anchored" if its cluster has at least one gold point.
    gold_anchored = np.array([
        bool(((labels == j) & (source == "gold")).sum())
        for j in range(args.k)
    ])
    gold_centroids = centroids[gold_anchored]

    rows = []
    rag_scores_all = np.array([rag_relevance(q) for q in queries])
    log_n_t_max = np.log1p(n_t)  # for size normalization
    for j in range(args.k):
        mask = labels == j
        n_g_j = int(((source == "gold") & mask).sum())
        n_t_j = int(((source == "tracing") & mask).sum())
        if n_g_j + n_t_j == 0:
            continue

        # 1) similarity to nearest *other* gold-anchored cluster.
        if gold_anchored[j]:
            # Distance to itself doesn't help — distance to other gold clusters does.
            others = np.delete(np.where(gold_anchored)[0], np.where(np.where(gold_anchored)[0] == j))
            sim_to_gold = float(
                cosine_similarity(centroids[j:j+1], centroids[others])[0].max()
            ) if len(others) else 1.0
        elif len(gold_centroids):
            sim_to_gold = float(
                cosine_similarity(centroids[j:j+1], gold_centroids)[0].max()
            )
        else:
            sim_to_gold = 0.0

        # 2) cluster gold share.
        gold_share = n_g_j / max(n_g_j + n_t_j, 1)

        # 3) RAG relevance: mean over the tracing queries in this cluster.
        cluster_tr_mask = mask & (source == "tracing")
        rag_relevance_j = float(rag_scores_all[cluster_tr_mask].mean()) if cluster_tr_mask.any() else 0.0

        # 4) size signal — normalized.
        size_signal = np.log1p(n_t_j) / log_n_t_max

        gap_score = (
            WEIGHTS["sim"]           * (1.0 - sim_to_gold)
            + WEIGHTS["size"]        * size_signal
            - WEIGHTS["gold_share"]  * gold_share
            + WEIGHTS["rag_relevance"] * rag_relevance_j
        )

        # Per-cluster top class for traceability.
        class_dist = Counter(np.array(classes)[mask].tolist()).most_common()
        # Sample 5 tracing queries (closest to centroid) for the human reader.
        if cluster_tr_mask.any():
            tr_indices = np.where(cluster_tr_mask)[0]
            sims = cosine_similarity(centroids[j:j+1], pool[tr_indices])[0]
            best = tr_indices[np.argsort(-sims)[:5]]
            sample_queries = [queries[i][:200] for i in best]
        else:
            sample_queries = []

        rows.append({
            "cluster_id": int(j),
            "n_gold": n_g_j,
            "n_tracing": n_t_j,
            "gold_share": gold_share,
            "sim_to_gold": sim_to_gold,
            "rag_relevance": rag_relevance_j,
            "size_signal": float(size_signal),
            "gap_score": float(gap_score),
            "top_classes": [{"class": c, "count": n} for c, n in class_dist[:5]],
            "sample_tracing_queries": sample_queries,
        })

    rows.sort(key=lambda r: r["gap_score"], reverse=True)
    top = rows[: args.top]

    report = {
        "meta": {
            "k": args.k,
            "weights": WEIGHTS,
            "n_gold": n_g,
            "n_tracing": n_t,
            "top_kept": args.top,
        },
        "clusters_all": rows,
        "clusters_top": top,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")
    print(f"\ntop {min(10, len(top))} gap clusters:")
    for r in top[:10]:
        cls = ",".join(f"{c['class']}:{c['count']}" for c in r["top_classes"][:3])
        print(f"  c={r['cluster_id']:3d} score={r['gap_score']:+.3f} "
              f"n_g={r['n_gold']:4d} n_t={r['n_tracing']:4d} "
              f"sim_gold={r['sim_to_gold']:.2f} rag_rel={r['rag_relevance']:.2f} "
              f"classes=[{cls}]")
        for q in r["sample_tracing_queries"][:2]:
            print(f"      - {q[:100]}")


if __name__ == "__main__":
    main()
