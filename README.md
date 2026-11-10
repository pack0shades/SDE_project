# regcheck

A multi-agent RAG system that checks software requirements against the GDPR and the EU Data Act. It is a course
implementation of Das et al., "A Multi-Agent RAG Framework for Regulatory Compliance Checking of Software
Requirements" (ACM TOSEM 2026):

- two compliance-checker agents from different vendors;
- a report-analyzer agent that settles their disagreements;
- human review with a feedback memory.

`PLAN.md` is the design spec and records every departure from the paper.

## Architecture Diagram
![alt text](image.png)

## Install

```sh
uv sync --group dev --group eval --group web   # Python 3.12; eval and web are optional
```

The first run downloads the local embedding and re-ranking models, about 2.1 GB in total
(`bge-large-en-v1.5` and `mxbai-rerank-large-v1`).

## Keys

Create a `.env` file in the repository root with free-tier keys for the three vendors. litellm loads it on
import. The file is gitignored.

```sh
MISTRAL_API_KEY=...
GEMINI_API_KEY=...
COHERE_API_KEY=...
```

Free tiers may use your prompts for training, so do not check confidential SRS documents. Model choices and
daily limits are set in `config.toml`. Each Gemini model has its own daily request quota.

## Index

```sh
uv run python -m regcheck index gdpr data_act        # regulations, by config key (PDFs in data/regulations/)
uv run python -m regcheck index data/srs/e-store.txt # an SRS document, .txt or .pdf
uv run python -m regcheck ask "When is consent required?" --doc gdpr
```

Indexing is resumable. Every LLM call is cached in `data/cache/`, so a re-run is free.

## Check

```sh
uv run python -m regcheck check e-store --reg gdpr                # multi-agent check -> data/runs/<time>-e-store-gdpr/
uv run python -m regcheck check e-store --reg gdpr --monolithic   # single-prompt baseline
```

Set `web_search = true` in `config.toml` to give the analyzer public web snippets. Only a generic query that
the analyzer writes is sent to the search engine, and a query that names the project or copies the
requirement's wording is not sent.

## Review

```sh
uv run streamlit run app.py
```

- Open a run, read each issue with its articles, both agents' positions and the evidence, and accept, reject
  or escalate it with a justification.
- Decisions go into the feedback memory (`data/memory/`), and later checks recall them for similar
  requirements.
- The Trace tab shows every prompt and reply.

## Evaluate

```sh
uv run python -m eval.gen_qa gdpr --n 10   # QA pairs -> data/eval/gdpr.qa.jsonl; review them by hand
uv run python -m regcheck index gdpr --fixed   # fixed-chunk index for the baseline RAG
uv run python -m eval.ragas_eval gdpr --agent cc1 --mode advanced --limit 10
uv run python -m eval.ragas_eval gdpr --agent cc1 --mode baseline --limit 10
uv run python -m eval.tables               # results/tables.md: the paper's Tables 4 to 7
```

## Test

```sh
uv run pytest      # offline: no network, no models
uv run ruff check .
```
