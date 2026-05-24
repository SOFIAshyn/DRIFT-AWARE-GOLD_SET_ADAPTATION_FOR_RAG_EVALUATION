"""Stage 6: select adaptation candidates from gap clusters.

Combines four sampling strategies (cluster-balanced + outlier + class-balanced
+ frequency-weighted) with a fixed total budget. Writes
data/processed/adaptation_candidates.json.

Each output record carries provenance (gap_cluster_id, gap_score, strategy)
so stage 7 / 8 can trace why a query was picked.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")

TRACING_FILE = PROCESSED / "chatbot_arena_open_ai_embeddings_sample3000_subclass_en.json"
GOLD_FILE    = PROCESSED / "crag_domain_open_ai_embeddings_subclass.json"
GAPS_FILE    = PROCESSED / "gap_clusters.json"
EMB_COL      = "open_ai_embeddings"
OUTPUT       = PROCESSED / "adaptation_candidates.json"

DEFAULT_BUDGET = 500

# Mix coefficients — must sum to 1.0.
MIX = {"cluster_balanced": 0.6, "outlier": 0.2, "frequency": 0.2}
# Class balance floor — every class with >= this many tracing rows in the gap
# clusters gets at least MIN_PER_CLASS picks before any other strategy runs.
MIN_PER_CLASS = 5


def load_records(path: Path):
    with path.open() as f:
        return json.load(f)


def stack_emb(records, col=EMB_COL):
    return np.vstack([np.asarray(r[col], dtype=np.float32) for r in records])


def medoid_index(X: np.ndarray) -> int:
    """Row whose mean cosine sim to the rest is highest."""
    sims = cosine_similarity(X)
    np.fill_diagonal(sims, 0.0)
    return int(sims.mean(axis=1).argmax())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--top", type=int, default=None,
                        help="Use only the top-N gap clusters (default: all in gap_clusters_top)")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    tracing = load_records(TRACING_FILE)
    gold    = load_records(GOLD_FILE)
    gaps    = load_records(GAPS_FILE)

    # Re-run the same clustering as stage 5 so we can map each tracing row → cluster_id.
    # The KMeans seed matches find_gaps.py so cluster_ids are stable.
    n_g = len(gold)
    pool = np.vstack([stack_emb(gold), stack_emb(tracing)])
    km = KMeans(n_clusters=gaps["meta"]["k"], n_init=10, random_state=42).fit(pool)
    labels = km.labels_
    tracing_cluster = labels[n_g:]  # length == len(tracing)

    cluster_rows = gaps["clusters_top"] if args.top is None else gaps["clusters_top"][:args.top]
    gap_cluster_ids = [c["cluster_id"] for c in cluster_rows]
    gap_meta = {c["cluster_id"]: c for c in cluster_rows}
    print(f"selecting from {len(gap_cluster_ids)} gap clusters with budget={args.budget}")

    # Bucket tracing indices by cluster_id (only the gap clusters).
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(tracing_cluster):
        if cid in gap_meta:
            by_cluster[int(cid)].append(i)

    picked: dict[int, dict] = {}  # tracing_idx -> record

    def record_for(idx: int, cid: int, strategy: str) -> dict:
        row = tracing[idx]
        return {
            "tracing_index": idx,
            "query": row.get("query", ""),
            "predicted_class": row.get("class", ""),
            "provenance": {
                "strategy": strategy,
                "gap_cluster_id": cid,
                "gap_score": gap_meta[cid]["gap_score"],
                "cluster_top_classes": gap_meta[cid]["top_classes"],
            },
        }

    # --- Strategy 1: class-balanced floor ---------------------------------
    candidates_by_class: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for cid, idxs in by_cluster.items():
        for i in idxs:
            candidates_by_class[tracing[i].get("class", "")].append((i, cid))

    floor_used = 0
    for cls, items in candidates_by_class.items():
        if not items:
            continue
        keep = items[:MIN_PER_CLASS]
        for i, cid in keep:
            if i not in picked:
                picked[i] = record_for(i, cid, "class_floor")
                floor_used += 1
    print(f"  class_floor picks: {floor_used}")

    # --- Strategy 2: cluster-balanced -------------------------------------
    remaining = args.budget - len(picked)
    n_clusters = max(len(by_cluster), 1)
    per_cluster = max(int(MIX["cluster_balanced"] * args.budget / n_clusters), 1)
    cb_used = 0
    for cid, idxs in by_cluster.items():
        # Pick rows closest to the cluster centroid first.
        X_c = stack_emb([tracing[i] for i in idxs])
        c_vec = km.cluster_centers_[cid:cid+1]
        sims = cosine_similarity(c_vec, X_c)[0]
        order = np.argsort(-sims)
        added = 0
        for k in order:
            if added >= per_cluster:
                break
            i = idxs[int(k)]
            if i not in picked:
                picked[i] = record_for(i, cid, "cluster_balanced")
                cb_used += 1
                added += 1
    print(f"  cluster_balanced picks: {cb_used}")

    # --- Strategy 3: outlier-aware ----------------------------------------
    n_per_cluster_outlier = max(int(MIX["outlier"] * args.budget / n_clusters), 1)
    out_used = 0
    for cid, idxs in by_cluster.items():
        if len(idxs) < 2:
            continue
        X_c = stack_emb([tracing[i] for i in idxs])
        c_vec = km.cluster_centers_[cid:cid+1]
        dists = 1.0 - cosine_similarity(c_vec, X_c)[0]
        farthest = np.argsort(-dists)[:n_per_cluster_outlier]
        med = medoid_index(X_c)
        for k in list(farthest) + [med]:
            i = idxs[int(k)]
            if i not in picked:
                picked[i] = record_for(i, cid, "outlier")
                out_used += 1
    print(f"  outlier picks: {out_used}")

    # --- Strategy 4: frequency-weighted ------------------------------------
    # Weight per cluster ∝ tracing-size^alpha.
    alpha = 0.7
    remaining = args.budget - len(picked)
    if remaining > 0:
        sizes = np.array([len(by_cluster[cid]) for cid in by_cluster])
        if sizes.sum() > 0:
            weights = sizes ** alpha
            weights = weights / weights.sum()
            cluster_ids = list(by_cluster.keys())
            quotas = np.maximum((MIX["frequency"] * args.budget * weights).astype(int), 0)
            freq_used = 0
            for cid, q in zip(cluster_ids, quotas):
                if q == 0:
                    continue
                idxs = by_cluster[cid]
                X_c = stack_emb([tracing[i] for i in idxs])
                c_vec = km.cluster_centers_[cid:cid+1]
                sims = cosine_similarity(c_vec, X_c)[0]
                order = np.argsort(-sims)
                added = 0
                for k in order:
                    if added >= q:
                        break
                    i = idxs[int(k)]
                    if i not in picked:
                        picked[i] = record_for(i, cid, "frequency")
                        freq_used += 1
                        added += 1
            print(f"  frequency picks: {freq_used}")

    # Trim to budget if we overshot.
    out_records = list(picked.values())
    if len(out_records) > args.budget:
        out_records.sort(key=lambda r: r["provenance"]["gap_score"], reverse=True)
        out_records = out_records[: args.budget]

    summary = Counter(r["provenance"]["strategy"] for r in out_records)
    class_summary = Counter(r["predicted_class"] for r in out_records)
    out = {
        "meta": {
            "budget": args.budget,
            "mix": MIX,
            "min_per_class": MIN_PER_CLASS,
            "n_gap_clusters_used": len(by_cluster),
            "n_picked": len(out_records),
        },
        "strategy_counts": dict(summary),
        "class_counts": dict(class_summary),
        "candidates": out_records,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.output} ({len(out_records)} picks)")
    print(f"  by strategy: {dict(summary)}")
    print(f"  by class:    {dict(class_summary)}")


if __name__ == "__main__":
    main()
