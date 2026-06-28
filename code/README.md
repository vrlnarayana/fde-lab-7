# Lab 7 — Code

Starter code for **Azure AI Search · Hybrid Retrieval · RAG**.
Full instructions are in the lab guide (`Lab7_Azure_AI_Search_RAG_Lab_Guide.docx`).

## Order of scripts

| Step | File | What it does |
|------|------|--------------|
| data | `generate_corpus.py` | Builds the synthetic clinical corpus (run once) |
| data | `generate_golden_set.py` | Builds the 50-question golden set (run once) |
| 0 | `00_local_smoke_test.py` | Offline check — no Azure needed |
| A | `01_build_index.py` | Chunk → embed → upload to Azure AI Search |
| B | `02_retrieval_strategies.py` | Compare naive / hybrid / re-ranked for one query |
| B | `03_rag_pipeline.py` | Full RAG answer with citations |
| C | `04_evaluate.py` | Golden set → recall@k, MRR, nDCG per strategy |
| C | `05_ragas_eval.py` | Answer quality (faithfulness, relevancy) |

Helpers: `corpus_loader.py`, `eval_metrics.py`, `azure_clients.py`.

## Requirements

- **Python 3.10–3.12** (tested on 3.12). The pinned `langchain-openai==0.3.33`
  requires `openai>=1.104.2`, so very old interpreters / the original `openai==1.59.6`
  pin will not resolve — `requirements.txt` now pins `openai==1.109.1`.

## Quick start (in a virtualenv)

```bash
# 1. Create + activate a venv (use a 3.10–3.12 interpreter)
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Generate the synthetic corpus + golden set (run once, offline)
python generate_corpus.py
python generate_golden_set.py

# 4. Offline sanity check — no Azure account needed
python 00_local_smoke_test.py

# 5. Azure steps — copy the template and fill in your Azure values first
cp .env.example .env                  # then edit .env
python 01_build_index.py
python 04_evaluate.py --k 5
```

Prefer not to activate? Call the venv interpreter directly, e.g.
`.venv/bin/python 00_local_smoke_test.py`.

### What runs offline vs. needs Azure

| Script | Network | Notes |
|--------|---------|-------|
| `generate_corpus.py` / `generate_golden_set.py` | offline | builds 22 docs / 50 questions |
| `00_local_smoke_test.py` | offline | keyword baseline — recall@5 ≈ 0.86, MRR ≈ 0.69, nDCG ≈ 0.70 |
| `01`–`05` | **Azure** | need `.env` (Azure AI Search + Azure OpenAI); fail fast with a clear `AZURE_SEARCH_ENDPOINT is not set` message if unset |

> The corpus is **synthetic** teaching data. It is not medical advice.
