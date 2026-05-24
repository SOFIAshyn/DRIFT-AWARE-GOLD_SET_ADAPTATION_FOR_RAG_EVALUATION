"""Filter the embedding JSON files to English-only queries.

Detects language with `langdetect` (seeded for determinism). Writes new
files with an `_en` suffix next to the originals. Keeps the per-item
schema unchanged.
"""

import json
from pathlib import Path

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")
INPUTS = [
    PROCESSED / "crag_domain_open_ai_embeddings_subclass.json",
    PROCESSED / "chatbot_arena_open_ai_embeddings_sample3000.json",
]


def _ascii_ratio(s: str) -> float:
    return sum(1 for c in s if ord(c) < 128) / max(len(s), 1)


def is_english(text: str) -> bool:
    """Keep if (a) mostly ASCII (langdetect is unreliable on short English),
    or (b) langdetect says 'en' on the non-ASCII portion."""
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if _ascii_ratio(s) >= 0.95:
        return True
    try:
        return detect(s) == "en"
    except LangDetectException:
        return False


def filter_file(path: Path) -> Path:
    with path.open() as f:
        data = json.load(f)

    kept, dropped = [], []
    for item in data:
        q = item.get("query", "")
        (kept if is_english(q) else dropped).append(item)

    out_path = path.with_name(path.stem + "_en" + path.suffix)
    with out_path.open("w") as f:
        json.dump(kept, f, ensure_ascii=False)

    print(f"{path.name}: in={len(data)} kept={len(kept)} dropped={len(dropped)} -> {out_path.name}")
    if dropped[:3]:
        print("  sample dropped:")
        for item in dropped[:3]:
            q = item.get("query", "")[:100].replace("\n", " ")
            print(f"    - {q}")
    return out_path


def main() -> None:
    for path in INPUTS:
        filter_file(path)


if __name__ == "__main__":
    main()
