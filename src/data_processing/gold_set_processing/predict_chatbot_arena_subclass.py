"""Predict subclass labels for chatbot_arena queries using a LogReg classifier
trained on CRAG `class` labels. One classifier per embedding type. Writes new
JSON files with a `class` field added.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")

JOBS = [
    # (crag_train_file, chat_input_file, emb_col, chat_output_file)
    (
        PROCESSED / "crag_domain_open_ai_embeddings_subclass.json",
        PROCESSED / "chatbot_arena_open_ai_embeddings_sample3000.json",
        "open_ai_embeddings",
        PROCESSED / "chatbot_arena_open_ai_embeddings_sample3000_subclass.json",
    ),
    (
        PROCESSED / "crag_tfidf_embeddings_subclass.json",
        PROCESSED / "chatbot_arena_tfidf_embeddings_sample3000.json",
        "tf_idf_embedding",
        PROCESSED / "chatbot_arena_tfidf_embeddings_sample3000_subclass.json",
    ),
]


def stack(records, col):
    return np.vstack([np.asarray(r[col], dtype=np.float32) for r in records])


def run(crag_path, chat_path, emb_col, out_path):
    with crag_path.open() as f:
        crag = json.load(f)
    with chat_path.open() as f:
        chat = json.load(f)

    X_train = stack(crag, emb_col)
    y_train = np.array([r["class"].lower() for r in crag])
    X_chat = stack(chat, emb_col)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=1)
    print(f"[{emb_col}] 5-fold CV acc on CRAG: {cv.mean():.3f} ± {cv.std():.3f}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_chat)

    out = [{**r, "class": p} for r, p in zip(chat, y_pred)]
    with out_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"  wrote {out_path.name} (n={len(out)})")
    print("  predicted class dist:")
    for k, v in Counter(y_pred).most_common():
        print(f"    {k:28s} {v}")


def main():
    for crag, chat, col, out in JOBS:
        run(crag, chat, col, out)
        print()


if __name__ == "__main__":
    main()
