import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_openai import AzureOpenAIEmbeddings
from openai import RateLimitError

# Add src directory to path to import constants
src_path = Path(__file__).resolve().parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from constants import (
    ANNOTATIONS_CLEANED_JSON,
    OPENAI_ANNOTATED_DATA_EMBEDDINGS_FILE,
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
    AZURE_OPENAI_EMBEDDINGS_MODEL,
    OPENAI_EMBEDDINGS_BALANCED_FILE,
    OPENAI_EMBEDDINGS_FILE,
    OPENAI_INVESTORS_EMBEDDINGS_FILE,
    QUERIES_COMBINED_BALANCED_FILE,
    QUERIES_COMBINED_WITH_INVESTORS_FILE,
    QUERIES_COMBINED_WITH_MULTI_TURN_FILE,
    ensure_directories_exist,
)

# Ensure all required directories exist
ensure_directories_exist()


def build_embeddings_client() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
        model=AZURE_OPENAI_EMBEDDINGS_MODEL,
        deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    )


def embed_documents_with_retry(client: AzureOpenAIEmbeddings, queries: list, max_retries: int = 6) -> list:
    """
    Generate embeddings for documents with exponential backoff retry logic.

    Args:
        client: AzureOpenAIEmbeddings client
        queries: List of query strings to embed
        max_retries: Maximum number of retry attempts

    Returns:
        List of embeddings
    """
    embeddings = []

    for query_num, query in enumerate(queries, 1):
        print(f"Processing query {query_num}/{len(queries)}", end="", flush=True)

        # Handle None or empty query values
        if query is None or (isinstance(query, str) and query.strip() == ""):
            print("(skipped - empty/None query)")
            embeddings.append(None)  # Append None to maintain index alignment
            continue

        for attempt in range(max_retries):
            try:
                # Embed single query
                result = client.embed_query(query)
                embeddings.append(result)
                print(" ✓")
                break

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\n  Rate limited on query {query_num}. Sleeping {wait_time}s before retry (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n  Rate limit error on query {query_num} after {max_retries} attempts: {str(e)[:100]}")
                    raise
            except Exception as e:
                print(f"\n  Error on query {query_num}: {str(e)[:100]}")
                raise

    return embeddings


def embed_dataframe(client: AzureOpenAIEmbeddings, df: pd.DataFrame, max_retries: int = 6) -> pd.DataFrame:
    """Generate embeddings only for rows that don't already have them.

    Checks each row for an existing non-empty 'open_ai_embeddings' field
    and skips those rows, only calling the API for missing ones.
    Handles None/empty query values gracefully.

    Args:
        client: AzureOpenAIEmbeddings client
        df: DataFrame with a 'query' column and optional 'open_ai_embeddings' column
        max_retries: Maximum number of retry attempts

    Returns:
        DataFrame with 'open_ai_embeddings' column fully populated (or None for invalid queries)
    """
    df = df.copy()

    if "open_ai_embeddings" not in df.columns:
        df["open_ai_embeddings"] = None

    # Find rows that need embeddings
    needs_embedding = df["open_ai_embeddings"].apply(
        lambda v: v is None or (isinstance(v, list) and len(v) == 0)
    )
    missing_indices = df.index[needs_embedding].tolist()
    already_have = len(df) - len(missing_indices)

    print(f"  Already have embeddings: {already_have}/{len(df)}")
    print(f"  Need to generate: {len(missing_indices)}/{len(df)}")

    if not missing_indices:
        print("  All embeddings already present, skipping API calls.")
        return df

    queries_to_embed = df.loc[missing_indices, "query"].tolist()
    new_embeddings = embed_documents_with_retry(client, queries_to_embed, max_retries)

    for idx, emb in zip(missing_indices, new_embeddings):
        df.at[idx, "open_ai_embeddings"] = emb

    # Log summary of skipped queries
    skipped_count = sum(1 for e in new_embeddings if e is None)
    if skipped_count > 0:
        print(f"Skipped {skipped_count} queries due to None/empty values")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenAI embeddings for queries")
    parser.add_argument(
        "--embeddings", action="store_true", help="Generate embeddings (default: False)"
    )
    parser.add_argument("--save-data", action="store_true", help="Save embeddings to file (default: False)")

    args = parser.parse_args()

    embeddings = args.embeddings
    save_data = args.save_data

    client = build_embeddings_client()

    if embeddings:
        input_file = "/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed/crag_domain_query.json"
        output_file = "/Users/s.petryshyn/Desktop/UNI/COURSE_WORK/data/processed/crag_domain_open_ai_embeddings_sample3000_en.json"

        print("\n" + "=" * 50)
        print("Generating OpenAI embeddings for tracings...")
        df = pd.read_json(input_file, orient="records")
        print(f"Processing {len(df)} queries...")
        df = embed_dataframe(client, df)

        if save_data:
            print(f"Saving OpenAI embeddings to: {output_file}")
            df.to_json(output_file, orient="records", indent=1)
        else:
            print("Embeddings not saved (--save-data flag not set)")


        if save_data:
            print(f"Saving OpenAI embeddings to: {OPENAI_ANNOTATED_DATA_EMBEDDINGS_FILE}")
            df.to_json(OPENAI_ANNOTATED_DATA_EMBEDDINGS_FILE, orient="records", indent=1)
        else:
            print("Embeddings not saved (--save-data flag not set)")

    print("=" * 50)
    print("OpenAI embeddings generated successfully!")
