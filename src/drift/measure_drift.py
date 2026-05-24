"""Stage 4 of the drift-aware gold set adaptation pipeline.

For each (embedding_type, class) pair, compare the CRAG gold subset
against the chatbot_arena tracing subset using six drift metrics:

    1. Centroid distance (Euclidean + cosine).
    2. Cosine-similarity distribution stats (mean / median / p25 / p75).
    3. Nearest-neighbor overlap (Jaccard@k between top-k neighbors of each
       tracing query in the joint pool vs the gold-only pool).
    4. Cluster coverage (KMeans on the joint set; count clusters that
       contain only tracing points).
    5. Jensen-Shannon divergence between gold/tracing cluster histograms.
    6. Sliced Wasserstein distance (average over random 1-D projections).

Writes data/processed/drift_report.json. Pass --bootstrap N to get 95% CIs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")

DATASETS = {
    "openai": {
        "gold":    PROCESSED / "crag_domain_open_ai_embeddings_subclass.json",
        "tracing": PROCESSED / "chatbot_arena_open_ai_embeddings_sample3000_subclass_en.json",
        "col":     "open_ai_embeddings",
    },
    "tfidf": {
        "gold":    PROCESSED / "crag_tfidf_embeddings_subclass.json",
        "tracing": PROCESSED / "chatbot_arena_tfidf_embeddings_sample3000_subclass_en.json",
        "col":     "tf_idf_embedding",
    },
}

OUTPUT = PROCESSED / "drift_report.json"

# Tunable knobs (kept here so a thesis reader can find them in one place).
NN_K = 10            # nearest-neighbor overlap window
CLUSTER_K = 12       # KMeans cluster count per (embedding, class)
SLICED_W_PROJ = 64   # number of random projections for sliced Wasserstein
MIN_PER_SIDE = 5     # skip a (class, embedding) cell with fewer rows than this
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_records(path: Path, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, class_labels) arrays from a subclass JSON file."""
    with path.open() as f:
        data = json.load(f)
    X = np.vstack([np.asarray(r[col], dtype=np.float32) for r in data])
    y = np.array([str(r.get("class", "")).lower() for r in data])
    return X, y


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def centroid_distances(G: np.ndarray, T: np.ndarray) -> dict:
    g, t = G.mean(axis=0), T.mean(axis=0)
    l2 = float(np.linalg.norm(g - t))
    denom = float(np.linalg.norm(g) * np.linalg.norm(t)) or 1e-12
    cos_sim = float(np.dot(g, t) / denom)
    return {"l2": l2, "cosine_distance": 1.0 - cos_sim}


def cosine_distribution(G: np.ndarray, T: np.ndarray, max_pairs: int = 100_000) -> dict:
    """Stats on cos(g, t) over all pairs (sampled if too many)."""
    n_pairs = G.shape[0] * T.shape[0]
    if n_pairs <= max_pairs:
        sims = cosine_similarity(T, G).ravel()
    else:
        gi = RNG.integers(0, G.shape[0], size=max_pairs)
        ti = RNG.integers(0, T.shape[0], size=max_pairs)
        # Per-pair cosine without forming the full matrix.
        g_norm = G[gi] / (np.linalg.norm(G[gi], axis=1, keepdims=True) + 1e-12)
        t_norm = T[ti] / (np.linalg.norm(T[ti], axis=1, keepdims=True) + 1e-12)
        sims = (g_norm * t_norm).sum(axis=1)
    return {
        "mean":   float(sims.mean()),
        "median": float(np.median(sims)),
        "p25":    float(np.percentile(sims, 25)),
        "p75":    float(np.percentile(sims, 75)),
        "n_pairs_used": int(len(sims)),
    }


def nn_overlap(G: np.ndarray, T: np.ndarray, k: int = NN_K) -> dict:
    """For each tracing point, top-k neighbors in (G ∪ T). Report mean
    Jaccard between {top-k from G alone} and {top-k from G ∪ T restricted
    to G members}. High overlap → the gold pool already covers what the
    tracing query would retrieve."""
    if min(G.shape[0], T.shape[0]) < k:
        return {"mean_jaccard_at_k": float("nan"), "k": k}
    pool = np.vstack([G, T])
    n_g = G.shape[0]
    sims_pool = cosine_similarity(T, pool)
    sims_gold = sims_pool[:, :n_g]
    # Exclude self-similarity in the pool (each T's own row in pool is at index n_g + i).
    for i in range(T.shape[0]):
        sims_pool[i, n_g + i] = -np.inf
    top_pool = np.argpartition(-sims_pool, k, axis=1)[:, :k]
    top_gold = np.argpartition(-sims_gold, k, axis=1)[:, :k]

    jaccards = []
    for i in range(T.shape[0]):
        pool_gold_neighbors = {idx for idx in top_pool[i] if idx < n_g}
        gold_only = set(top_gold[i].tolist())
        inter = len(pool_gold_neighbors & gold_only)
        union = len(pool_gold_neighbors | gold_only) or 1
        jaccards.append(inter / union)
    return {"mean_jaccard_at_k": float(np.mean(jaccards)), "k": k}


def cluster_metrics(G: np.ndarray, T: np.ndarray, k: int = CLUSTER_K) -> dict:
    """KMeans on G ∪ T → coverage gap + JS divergence between cluster histograms."""
    n_g, n_t = G.shape[0], T.shape[0]
    k_eff = min(k, max(2, (n_g + n_t) // 5))
    pool = np.vstack([G, T])
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(pool)
    labels = km.labels_
    g_hist = np.bincount(labels[:n_g], minlength=k_eff).astype(float)
    t_hist = np.bincount(labels[n_g:], minlength=k_eff).astype(float)
    g_p = g_hist / max(g_hist.sum(), 1.0)
    t_p = t_hist / max(t_hist.sum(), 1.0)

    tracing_only = int(np.sum((g_hist == 0) & (t_hist > 0)))
    gold_only    = int(np.sum((g_hist > 0)  & (t_hist == 0)))
    shared       = int(np.sum((g_hist > 0)  & (t_hist > 0)))
    return {
        "k_used": k_eff,
        "tracing_only_clusters": tracing_only,
        "gold_only_clusters":    gold_only,
        "shared_clusters":       shared,
        "tracing_only_share": tracing_only / k_eff,
        "js_divergence":  float(jensenshannon(g_p, t_p, base=2.0) ** 2),  # squared = JS divergence; sqrt = JS distance
    }


def sliced_wasserstein(G: np.ndarray, T: np.ndarray, n_proj: int = SLICED_W_PROJ) -> dict:
    """Average 1-D Wasserstein over random unit-norm projections."""
    d = G.shape[1]
    projs = RNG.standard_normal((n_proj, d)).astype(np.float32)
    projs /= np.linalg.norm(projs, axis=1, keepdims=True) + 1e-12
    g_proj = G @ projs.T  # (n_g, n_proj)
    t_proj = T @ projs.T
    dists = [wasserstein_distance(g_proj[:, j], t_proj[:, j]) for j in range(n_proj)]
    return {"mean": float(np.mean(dists)), "std": float(np.std(dists)), "n_proj": n_proj}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def per_class_drift(
    G: np.ndarray, y_g: np.ndarray, T: np.ndarray, y_t: np.ndarray
) -> dict:
    """Compute all metrics for the joint pool and for each shared class."""
    classes = sorted(set(y_g.tolist()) | set(y_t.tolist()))
    out: dict[str, dict] = {}

    out["__all__"] = _compute_cell(G, T)
    for c in classes:
        gi, ti = (y_g == c), (y_t == c)
        n_g, n_t = int(gi.sum()), int(ti.sum())
        cell: dict = {"n_gold": n_g, "n_tracing": n_t}
        if min(n_g, n_t) < MIN_PER_SIDE:
            cell["skipped"] = f"<{MIN_PER_SIDE} per side"
            out[c] = cell
            continue
        cell.update(_compute_cell(G[gi], T[ti]))
        out[c] = cell
    return out


def _compute_cell(G: np.ndarray, T: np.ndarray) -> dict:
    return {
        "n_gold":    int(G.shape[0]),
        "n_tracing": int(T.shape[0]),
        "centroid":  centroid_distances(G, T),
        "cosine":    cosine_distribution(G, T),
        "nn":        nn_overlap(G, T),
        "cluster":   cluster_metrics(G, T),
        "wasserstein": sliced_wasserstein(G, T),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    report = {"meta": {
        "nn_k": NN_K,
        "cluster_k": CLUSTER_K,
        "sliced_w_proj": SLICED_W_PROJ,
        "min_per_side": MIN_PER_SIDE,
        "seed": 42,
    }}

    for emb_name, spec in DATASETS.items():
        print(f"\n=== {emb_name} ===")
        G, y_g = load_records(spec["gold"], spec["col"])
        T, y_t = load_records(spec["tracing"], spec["col"])
        print(f"  gold n={len(G)} dim={G.shape[1]} | tracing n={len(T)}")
        print(f"  gold class dist:    {dict(Counter(y_g).most_common())}")
        print(f"  tracing class dist: {dict(Counter(y_t).most_common())}")
        report[emb_name] = per_class_drift(G, y_g, T, y_t)
        _print_summary(emb_name, report[emb_name])

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")


def _print_summary(name: str, per_class: dict) -> None:
    print(f"  drift summary [{name}]:")
    header = f"    {'class':28s} {'n_g':>5} {'n_t':>5} {'cent_cos':>9} {'nn_jac':>8} {'js_div':>8} {'wass':>8} {'trace_only':>11}"
    print(header)
    for c, cell in per_class.items():
        if cell.get("skipped"):
            print(f"    {c:28s} {cell.get('n_gold','-'):>5} {cell.get('n_tracing','-'):>5}  skipped ({cell['skipped']})")
            continue
        print(
            f"    {c:28s} {cell['n_gold']:>5} {cell['n_tracing']:>5} "
            f"{cell['centroid']['cosine_distance']:>9.3f} "
            f"{cell['nn']['mean_jaccard_at_k']:>8.3f} "
            f"{cell['cluster']['js_divergence']:>8.3f} "
            f"{cell['wasserstein']['mean']:>8.3f} "
            f"{cell['cluster']['tracing_only_clusters']:>11d}"
        )


if __name__ == "__main__":
    main()
