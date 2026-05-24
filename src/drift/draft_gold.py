"""Stage 7: turn adaptation candidates into draft gold entries.

For each candidate, produce:
    - rewritten evaluation-ready query
    - validated class label
    - draft expected answer + source URLs
    - difficulty + ambiguity labels
    - provenance.needs_review = True (always, until stage 8 clears it)

If ANTHROPIC_API_KEY is set, drafts via Anthropic Claude.
Otherwise emits a structured placeholder so the rest of the pipeline still
runs end-to-end (useful for CI / dry-runs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROCESSED = Path("/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed")
CANDIDATES_FILE = PROCESSED / "adaptation_candidates.json"
OUTPUT = PROCESSED / "adapted_gold_drafts.json"

MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 2048
RETRY_MAX_TOKENS = 4096
SYS_PROMPT = (
    "You are drafting an evaluation item for a RAG benchmark. "
    "For the user query, return strict JSON with keys: "
    "query_rewritten (a cleaned-up evaluation-ready version, <=300 chars), "
    "class (one of: finance, movie, music, sports, celebrities, "
    "geography_places, art_history, business_companies, food, "
    "math_code, gaming, relationship_psychology, open), "
    "answer (a concise factual answer in 1-3 sentences, <=500 chars), "
    "sources (a list of 1-3 objects {url, why_relevant}, why_relevant <=200 chars), "
    "difficulty (easy|medium|hard), "
    "ambiguity (single-answer|multi-answer|opinion). "
    "Reply with ONLY the JSON object, no prose, no markdown fences. "
    "Keep every string short so the JSON fits in your token budget."
)


def stable_id(query: str, prefix: str = "trace") -> str:
    h = hashlib.sha1(query.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def placeholder_draft(candidate: dict) -> dict:
    return {
        "query_rewritten": candidate["query"].strip(),
        "class": candidate.get("predicted_class") or "open",
        "answer": "<PLACEHOLDER: requires LLM drafting or human annotation>",
        "sources": [],
        "difficulty": "medium",
        "ambiguity": "single-answer",
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _call_once(client, candidate: dict, max_tokens: int) -> tuple[str, str]:
    """Returns (raw_text, stop_reason)."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYS_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Query: {candidate['query']}\n"
                f"Predicted class (from a LogReg classifier): {candidate.get('predicted_class', 'unknown')}\n"
                f"Gap-cluster top classes: "
                f"{', '.join(c['class'] for c in candidate['provenance']['cluster_top_classes'][:3])}"
            ),
        }],
    )
    raw = "".join(b.text for b in msg.content if hasattr(b, "text"))
    return raw, getattr(msg, "stop_reason", "")


def anthropic_draft(client, candidate: dict) -> dict:
    """Two-attempt LLM draft. Retries with a larger budget if the first
    response was truncated (stop_reason='max_tokens') or fails JSON parse."""
    raw, stop = _call_once(client, candidate, DEFAULT_MAX_TOKENS)
    text = _strip_fences(raw)
    if stop != "max_tokens":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass  # fall through to retry
    raw, _ = _call_once(client, candidate, RETRY_MAX_TOKENS)
    return json.loads(_strip_fences(raw))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(CANDIDATES_FILE))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N candidates (useful for dry-runs).")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM drafting even if ANTHROPIC_API_KEY is set.")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    candidates = payload["candidates"]
    if args.limit:
        candidates = candidates[: args.limit]

    use_llm = (not args.no_llm) and bool(os.environ.get("ANTHROPIC_API_KEY"))
    client = None
    if use_llm:
        try:
            import anthropic
            client = anthropic.Anthropic()
            print(f"LLM drafting via Anthropic ({MODEL}), {len(candidates)} items")
        except ImportError:
            print("anthropic SDK not installed (`pip install anthropic`) — falling back to placeholders",
                  file=sys.stderr)
            use_llm = False
    else:
        print(f"placeholder mode (no LLM), {len(candidates)} items")

    out = []
    for i, cand in enumerate(candidates):
        try:
            fields = anthropic_draft(client, cand) if use_llm else placeholder_draft(cand)
            draft_method = "llm" if use_llm else "placeholder"
        except Exception as exc:
            print(f"  [{i}] LLM call failed ({exc}); using placeholder", file=sys.stderr)
            fields = placeholder_draft(cand)
            draft_method = "llm_failed_placeholder"

        out.append({
            "id": stable_id(cand["query"]),
            "query": fields["query_rewritten"],
            "query_original": cand["query"],
            "class": fields.get("class", cand.get("predicted_class", "open")),
            "expected_answer": fields["answer"],
            "sources": fields.get("sources", []),
            "difficulty": fields.get("difficulty", "medium"),
            "ambiguity": fields.get("ambiguity", "single-answer"),
            "provenance": {
                **cand["provenance"],
                "draft_method": draft_method,
                "needs_review": True,
            },
        })
        if (i + 1) % 25 == 0:
            print(f"  drafted {i + 1}/{len(candidates)}")

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.output} ({len(out)} drafts)")


if __name__ == "__main__":
    main()
