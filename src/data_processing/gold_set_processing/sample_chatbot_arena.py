import json
import random
from pathlib import Path

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")
QUERIES_IN = PROCESSED / "chatbot_arena_queries.json"
EMB_IN = PROCESSED / "chatbot_arena_tfidf_embeddings.json"
QUERIES_OUT = PROCESSED / "chatbot_arena_queries_sample3000.json"
EMB_OUT = PROCESSED / "chatbot_arena_tfidf_embeddings_sample3000.json"

SAMPLE_SIZE = 3000
SEED = 42


def main() -> None:
    with QUERIES_IN.open() as f:
        queries = json.load(f)
    with EMB_IN.open() as f:
        embeddings = json.load(f)

    queries_by_q = {item["query"]: item for item in queries}
    emb_by_q = {item["query"]: item for item in embeddings}

    shared = sorted(set(queries_by_q) & set(emb_by_q))
    print(f"queries={len(queries)} emb={len(embeddings)} shared={len(shared)}")

    if len(shared) < SAMPLE_SIZE:
        raise ValueError(f"only {len(shared)} shared queries, need {SAMPLE_SIZE}")

    rng = random.Random(SEED)
    picked = rng.sample(shared, SAMPLE_SIZE)
    picked_set = set(picked)

    queries_out = [queries_by_q[q] for q in picked]
    emb_out = [emb_by_q[q] for q in picked]

    with QUERIES_OUT.open("w") as f:
        json.dump(queries_out, f)
    with EMB_OUT.open("w") as f:
        json.dump(emb_out, f)

    assert {x["query"] for x in queries_out} == picked_set
    assert {x["query"] for x in emb_out} == picked_set
    assert [x["query"] for x in queries_out] == [x["query"] for x in emb_out]

    print(f"wrote {QUERIES_OUT}")
    print(f"wrote {EMB_OUT}")


if __name__ == "__main__":
    main()
