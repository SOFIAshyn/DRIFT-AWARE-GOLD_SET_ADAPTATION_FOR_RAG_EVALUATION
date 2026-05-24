#!/bin/sh

export OPENAI_API_DEPLOYMENT=".."
export OPENAI_API_KEY=".."
export OPENAI_API_RESOURCE_NAME=".."
export OPENAI_MODEL_NAME="text-embedding-3-small"

uv run generate_openAI_embeddings.py --embeddings --save-data