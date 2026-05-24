"""Adversarial validation between gold-set candidates and tracing data.

Train a binary classifier to predict origin (gold=0, tracing=1). The
cross-validated ROC-AUC measures how distinguishable the two distributions
are:

    AUC ≈ 0.5  →  indistinguishable. Gold ≈ tracing in this embedding space.
    AUC → 1.0  →  perfectly separable. Heavy distribution drift.

Two comparisons are run:

    Pair 1 — CRAG (original gold)           vs Chatbot Arena (tracing)
    Pair 2 — CRAG + adapted-gold drafts     vs Chatbot Arena (tracing)

If the adaptation pipeline is working, Pair 2 should have a *lower* AUC
than Pair 1 — i.e. the enriched gold set looks more like production traffic.
Also reports the standalone adapted-drafts-only vs tracing for reference.

Writes data/processed/adversarial_validation.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")

CRAG_ORIGINAL = PROCESSED / "crag_domain_open_ai_embeddings_subclass_en.json"
CRAG_ADAPTED  = PROCESSED / "crag_domain_adapted_gold_draft_open_ai_embeddings.json"
TRACING       = PROCESSED / "chatbot_arena_open_ai_embeddings_sample3000_en.json"
EMB_COL       = "open_ai_embeddings"
OUTPUT        = PROCESSED / "adversarial_validation.json"

N_FOLDS = 5
SEED = 42


def load_embeddings(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    return np.vstack([np.asarray(r[EMB_COL], dtype=np.float32) for r in data])


def adversarial_score(gold: np.ndarray, tracing: np.ndarray, label: str) -> dict:
    """Train logreg to discriminate gold vs tracing. Report CV AUC / acc / F1
    plus the baseline AUC a 'always predict majority' classifier would get."""
    X = np.vstack([gold, tracing]).astype(np.float32)
    y = np.array([0] * len(gold) + [1] * len(tracing))
    pos_share = float(y.mean())

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    return {
        "label":            label,
        "n_gold":           int(len(gold)),
        "n_tracing":        int(len(tracing)),
        "pos_share":        pos_share,
        "cv_auc":           float(roc_auc_score(y, proba)),
        "cv_accuracy":      float(accuracy_score(y, pred)),
        "cv_f1_tracing":    float(f1_score(y, pred, pos_label=1)),
        "cv_f1_gold":       float(f1_score(y, pred, pos_label=0)),
        "interpretation":   _interpret(roc_auc_score(y, proba)),
    }


def _interpret(auc: float) -> str:
    if auc < 0.55:
        return "indistinguishable (no drift detected)"
    if auc < 0.7:
        return "mild drift"
    if auc < 0.85:
        return "moderate drift"
    if auc < 0.95:
        return "strong drift"
    return "near-perfect separation — large drift"


def main() -> None:
    print("loading embeddings...")
    crag_orig    = load_embeddings(CRAG_ORIGINAL)
    crag_adapted = load_embeddings(CRAG_ADAPTED)
    tracing      = load_embeddings(TRACING)
    print(f"  CRAG original:     n={len(crag_orig)}")
    print(f"  CRAG adapted only: n={len(crag_adapted)}")
    print(f"  Tracing:           n={len(tracing)}")

    crag_combined = np.vstack([crag_orig, crag_adapted])
    print(f"  CRAG + adapted:    n={len(crag_combined)}")

    print("\nrunning adversarial validation (LogReg, 5-fold CV)...")
    results = [
        adversarial_score(crag_orig,     tracing, "CRAG original vs tracing"),
        adversarial_score(crag_combined, tracing, "CRAG + adapted drafts vs tracing"),
        adversarial_score(crag_adapted,  tracing, "adapted drafts only vs tracing"),
    ]

    print(f"\n{'-'*78}")
    print(f"{'comparison':40s} {'n_gold':>7s} {'n_trace':>8s} {'AUC':>6s} {'acc':>6s}  interpretation")
    print(f"{'-'*78}")
    for r in results:
        print(
            f"{r['label']:40s} {r['n_gold']:>7d} {r['n_tracing']:>8d} "
            f"{r['cv_auc']:>6.3f} {r['cv_accuracy']:>6.3f}  {r['interpretation']}"
        )
    print(f"{'-'*78}")

    delta = results[0]["cv_auc"] - results[1]["cv_auc"]
    print(
        f"\nadaptation effect: AUC change (orig → orig+adapted) = "
        f"{-delta:+.3f}  ({'less drift' if delta > 0 else 'no improvement or worse'})"
    )

    OUTPUT.write_text(json.dumps({
        "meta": {
            "method": "logistic regression",
            "n_folds": N_FOLDS,
            "seed": SEED,
            "files": {
                "crag_original": str(CRAG_ORIGINAL),
                "crag_adapted":  str(CRAG_ADAPTED),
                "tracing":       str(TRACING),
            },
        },
        "results": results,
        "adaptation_auc_delta": float(-delta),
    }, indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
