"""Generate TF-IDF embeddings for processed query JSON files.

Fits a shared TfidfVectorizer over the combined corpus of the chatbot_arena
sample and CRAG, then writes a new JSON file per input where each item gets
a `tf_idf_embedding` field plus a sparse `tf_idf` dict.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    CHATBOT_ARENA_EMBEDDINGS_SAMPLE_FILE,
    CHATBOT_ARENA_QUERIES_SAMPLE_FILE,
    CRAG_DOMAIN_QUERY_FILE,
    CRAG_EMBEDDINGS_FILE,
)

JOBS = {
    "chatbot_arena": (CHATBOT_ARENA_QUERIES_SAMPLE_FILE, CHATBOT_ARENA_EMBEDDINGS_SAMPLE_FILE),
    "crag_domain": (CRAG_DOMAIN_QUERY_FILE, CRAG_EMBEDDINGS_FILE),
}


def main() -> None:
    df1 = pd.read_json(JOBS["chatbot_arena"][0]).dropna(subset=["query"])
    df2 = pd.read_json(JOBS["crag_domain"][0]).dropna(subset=["query"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
    )
    vectorizer.fit(pd.concat([df1["query"], df2["query"]]))

    print(f"Generating TF-IDF for chatbot_arena ({len(df1)} queries)...")
    embeddings1 = vectorizer.transform(df1["query"]).toarray().tolist()
    df1["tf_idf_embedding"] = embeddings1
    print(f"Saving -> {JOBS['chatbot_arena'][1]}")
    df1.to_json(JOBS["chatbot_arena"][1], orient="records", indent=1)

    print(f"Generating TF-IDF for crag_domain ({len(df2)} queries)...")
    embeddings2 = vectorizer.transform(df2["query"]).toarray().tolist()
    df2["tf_idf_embedding"] = embeddings2
    print(f"Saving -> {JOBS['crag_domain'][1]}")
    df2.to_json(JOBS["crag_domain"][1], orient="records", indent=1)


if __name__ == "__main__":
    main()
