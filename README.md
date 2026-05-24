# Drift-Aware Gold Set Adaptation for RAG Evaluation

> The proposed pipeline detects semantic drift between curated gold sets and tracing data, identifies underrepresented production-like query regions, and adapts the gold set through cluster-aware sampling and benchmark enrichment.

A pipeline that keeps a RAG benchmark aligned with live production traffic. Uses **CRAG** (curated QA) as the gold set and **Chatbot Arena** (real-user dialogues) as the tracing data.

---

## Why

Static QA benchmarks decay as production traffic shifts. New entities, new question shapes, and new topics appear in tracing data that the original gold set never covered. Metrics on the static gold set keep looking healthy while the system silently underperforms on real users. This pipeline:

1. **Measures** the gap between gold and tracing in both lexical (TF-IDF) and semantic (OpenAI) embedding spaces.
2. **Finds** the underrepresented production-like query clusters.
3. **Selects** drift-relevant queries from tracing using cluster-, class-, frequency- and outlier-aware sampling.
4. **Drafts** evaluation-ready gold entries for them (LLM-assisted).
5. **Observes the improvement** via adversarial validation: a binary classifier trained to discriminate gold-set vs tracing-data embeddings.

---

## Pipeline

| # | Stage | Script | Output |
|--:|-------|--------|--------|
| 1 | Ingest | `src/data_processing/tracing_data_processing/download_chatbot_arena_dataset.py` | raw CSV/JSONL |
| 2 | Preprocess | `src/data_processing/gold_set_processing/{sample_chatbot_arena,filter_english_only,split_open_subcategories,predict_chatbot_arena_subclass}.py` | cleaned + labeled JSON |
| 3 | Embed | `src/data_processing/generate_embeddings/{generate_tfidf_embeddings,generate_open_ai_embeddings}.py` | per-record embedding files |
| 4 | Measure drift | `src/drift/measure_drift.py` | `data/processed/drift_report.json` |
| 5 | Detect gaps | `src/drift/find_gaps.py` | `data/processed/gap_clusters.json` |
| 6 | Sample candidates | `src/drift/sample_candidates.py` | `data/processed/adaptation_candidates.json` |
| 7 | Draft gold | `src/drift/draft_gold.py` (Anthropic Claude) | `data/processed/adapted_gold_drafts.json` |
| obs | Adversarial validation | `src/drift/adversarial_validation.py` | `data/processed/adversarial_validation.json` |
| viz | Drift dashboard | `notebooks/drift_report.ipynb` | per-release report |
| viz | UMAP supporting evidence | `src/umap_embeddings_3d*.ipynb` | interactive 3D plots |

---

## Repo layout

```
COURSE_WORK/
├── docs/
│   └── drift_aware_gold_set_pipeline.md   # design doc
├── notebooks/
│   └── drift_report.ipynb                 # per-release drift dashboard
├── src/
│   ├── data_processing/                   # stages 1-3
│   ├── drift/                             # stages 4-7 + adversarial validation
│   │   ├── measure_drift.py
│   │   ├── find_gaps.py
│   │   ├── sample_candidates.py
│   │   ├── draft_gold.py
│   │   └── adversarial_validation.py
│   └── umap_embeddings_3d*.ipynb          # UMAP visualizations
├── requirements.txt
├── README.md
└── data/                                  # gitignored; see below
```

`data/` and `.venv/` are in `.gitignore`.

---

## Setup

```bash
git clone <repo-url>
cd COURSE_WORK
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, only needed for the LLM-drafting stage:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Data

Embeddings and intermediate JSON files are **not** committed (too large). Reproduce them from scratch:

```bash
# Stage 1 — download Chatbot Arena conversations (the CRAG dump is fetched manually).
python3 src/data_processing/tracing_data_processing/download_chatbot_arena_dataset.py

# Stage 2 — preprocess (sample, English filter, subclass labels).
python3 src/data_processing/gold_set_processing/sample_chatbot_arena.py
python3 src/data_processing/gold_set_processing/split_open_subcategories.py
python3 src/data_processing/gold_set_processing/predict_chatbot_arena_subclass.py
python3 src/data_processing/gold_set_processing/filter_english_only.py

# Stage 3 — embed.
python3 src/data_processing/generate_embeddings/generate_tfidf_embeddings.py
python3 src/data_processing/generate_embeddings/generate_open_ai_embeddings.py   # needs OPENAI_API_KEY
```

---

## Run the pipeline

```bash
# Stage 4 — measure drift across all metrics.
python3 src/drift/measure_drift.py

# Stage 5 — joint clustering + gap_score ranking.
python3 src/drift/find_gaps.py            # tune via --k 60 --top 25

# Stage 6 — sample adaptation candidates.
python3 src/drift/sample_candidates.py    # tune via --budget 500

# Stage 7 — LLM-draft gold entries (placeholder mode if no ANTHROPIC_API_KEY).
python3 src/drift/draft_gold.py           # --no-llm to force placeholder

# Observation — adversarial validation: how distinguishable is gold vs tracing?
python3 src/drift/adversarial_validation.py
```

Open `notebooks/drift_report.ipynb` for tables + charts of the drift report.

---

## Observing improvement — adversarial validation

The clearest single-number measure of "did adaptation help?" is the cross-validated ROC-AUC of a binary classifier trained to predict whether an embedding came from the gold set or from tracing. The classifier is a balanced `LogisticRegression` over OpenAI embeddings, 5-fold stratified CV.

| AUC | Reading |
|----:|---------|
| ~0.5 | indistinguishable — gold ≈ tracing |
| 0.55–0.7 | mild drift |
| 0.7–0.85 | moderate drift |
| 0.85–0.95 | strong drift |
| > 0.95 | near-perfect separation — large drift |

Run the harness over (i) original gold vs tracing and (ii) gold + adapted drafts vs tracing — the **delta** between the two AUCs is the adaptation effect.

---

## Headline numbers (from current run)

| metric | value |
|--------|------:|
| Adversarial AUC, CRAG original vs Chatbot Arena | 0.997 |
| Adversarial AUC, CRAG + 434 adapted drafts vs Chatbot Arena | 0.955 |
| Adversarial AUC, adapted drafts only vs Chatbot Arena | 0.863 |
| AUC delta (orig → orig + adapted) | **−0.041** |
| Top drift classes (JS divergence, OpenAI) | `open`, `math_code`, `food`, `gaming` |
| Top gap-clusters by `gap_score` (sample themes) | math word-problems, AI-future speculation, role-play prompts, weather, chit-chat |

Original CRAG is near-trivially separable from tracing (AUC 0.997). Adding the 434 adapted drafts shifts AUC by **−0.041** — directionally correct, but a bigger annotation budget is needed to move the needle further.

---

## Conclusions and further work

The implemented stages (1–7, plus adversarial validation as the improvement signal) already demonstrate that production traffic drifts far enough from a curated benchmark to make the static gold set unreliable, and that cluster-aware enrichment closes that gap measurably. AUC drops from 0.997 to 0.955 after adding 434 drafts; the direction is right and the methodology generalizes to bigger annotation budgets.

To turn the prototype into a production-grade benchmark workflow, the design doc reserves three follow-up stages:

- **Stage 8 — Human validation UI.** Minimal Streamlit/FastAPI app over `adapted_gold_drafts.json`; quality gates on class, answer correctness, source support, ambiguity, suitability; writes `adapted_gold_validated.json` with inter-annotator agreement measured offline.
- **Stage 9 — Benchmark partitions.** Three frozen JSON sets: `gold_original/` (CRAG, unchanged), `gold_adapted/` (validated additions, versioned per quarter), `holdout/` (untouched tracing-like slice, leakage-protected).
- **Stage 10 — RAG eval harness.** Config-driven; loads any RAG via `{module, callable}` returning `{'answer', 'retrieved'}`; reports recall@k, MRR, NDCG@k, answer correctness (LLM-judge optional), hallucination rate, unsupported-answer rate, per-class breakdown; compares `gold_original` vs `gold_original + gold_adapted` deltas.

Open questions tracked at the bottom of [`docs/drift_aware_gold_set_pipeline.md`](docs/drift_aware_gold_set_pipeline.md): clustering algorithm choice, annotation budget, ground-truth answer source, holdout refresh policy, multilingual scope.
