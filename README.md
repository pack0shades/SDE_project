# regcheck

`regcheck` is a multi-agent Retrieval-Augmented Generation (RAG) system that checks software requirements against
the EU General Data Protection Regulation (GDPR) and the EU Data Act. It is a course implementation of:

> Das, Deb, Chaki and Cortesi, "A Multi-Agent RAG Framework for Regulatory Compliance Checking of Software
> Requirements", *ACM Transactions on Software Engineering and Methodology* 35(8), Article 223, 2026.
> doi:10.1145/3785472

Given a Software Requirements Specification (SRS) and a regulation, the system identifies requirements that are
non-compliant or only partly compliant, cites the relevant articles, explains its reasoning and proposes a
compliant rewrite. A human reviewer makes the final decision on every finding.

`PLAN.md` is the full design specification. It describes every module and records every departure from the paper.

## How it works

1. **Document pre-processor.** Regulations and SRS documents are split into sections, then into self-contained
   statements by an LLM, embedded locally and stored in separate vector indexes.
2. **Two compliance-checker agents.** Two models from different vendors check the SRS independently, using
   chain-of-thought planning and targeted retrieval of the relevant articles.
3. **Report-analyzer agent.** A third model judges every disagreement between the checkers, asks the right checker
   to re-assess, and compiles a final report with a confidence score per issue.
4. **Human review.** A reviewer accepts, rejects or escalates each finding with a justification. Decisions are
   stored in a feedback memory and recalled in later checks of similar requirements.

Every prompt and reply is recorded, so each result can be traced back to its evidence.

## Architecture

![Architecture diagram](image.png)

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- About 12 GB of free disk space for the full installation, including about 2.1 GB of local models
  (`bge-large-en-v1.5` and `mxbai-rerank-large-v1`), downloaded on first use.
- A CUDA GPU is optional; the local models also run on CPU, more slowly.
- Free-tier API keys for Mistral, Google Gemini and Cohere.

## Installation

```sh
uv sync --group dev --group eval --group web   # the eval and web groups are optional
```

## Configuration

Create a `.env` file in the repository root (it is gitignored). litellm loads it automatically.

```sh
MISTRAL_API_KEY=...
GEMINI_API_KEY=...
COHERE_API_KEY=...
```

Models, retrieval settings and regulations are set in `config.toml`. Each Gemini model has its own daily request
quota on the free tier.

**Privacy:** free-tier providers may use prompts for training. Do not check confidential SRS documents with the
default configuration.

## Usage

### 1. Index documents

```sh
uv run python -m regcheck index gdpr data_act          # regulations, by config key (PDFs in data/regulations/)
uv run python -m regcheck index data/srs/e-store.txt   # an SRS document, .txt or .pdf
```

Indexing is resumable. Every LLM call is cached in `data/cache/`, so re-running a command costs nothing.

### 2. Ask a question (optional)

```sh
uv run python -m regcheck ask "When is consent required?" --doc gdpr
```

### 3. Run a compliance check

```sh
uv run python -m regcheck check e-store --reg gdpr                # multi-agent check
uv run python -m regcheck check e-store --reg gdpr --monolithic   # single-prompt baseline
```

Results are written to `data/runs/<time>-<srs>-<regulation>/`.

To let the analyzer consult public web snippets, set `web_search = true` in `config.toml`. Only a generic query
written by the analyzer is sent to the search engine. A query that names the project or copies the requirement's
wording is never sent.

### 4. Review the findings

```sh
uv run streamlit run app.py
```

- Open a run and read each issue with its articles, both checkers' positions and the retrieved evidence.
- Accept, reject or escalate each issue with a justification. Decisions are stored in `data/memory/` and recalled
  in later checks of similar requirements.
- The Trace tab shows every prompt and reply.
- Accepted rewrites and escalated items can be exported as Markdown.

### 5. Evaluate

```sh
uv run python -m eval.gen_qa gdpr --n 10        # generate QA pairs -> data/eval/gdpr.qa.jsonl; review them by hand
uv run python -m eval.gen_qa gdpr --manual      # add the team-written QA pairs (no LLM calls)
uv run python -m regcheck index gdpr --fixed    # fixed-chunk index for the baseline RAG
uv run python -m eval.ragas_eval gdpr --agent cc1 --mode advanced --limit 10
uv run python -m eval.ragas_eval gdpr --agent cc1 --mode baseline --limit 10
uv run python -m eval.ragas_eval gdpr --agent cc1 --mode advanced --qa manual --limit 20
uv run python -m eval.tables                    # results/tables.md: the paper's Tables 4 to 7
```

## Testing

```sh
uv run pytest       # offline: no network or models needed
uv run ruff check .
```

## Project structure

| Path | Contents |
|---|---|
| `regcheck/` | The package: pre-processing, retrieval, agents, pipeline, memory and the command-line interface |
| `app.py` | Streamlit review application |
| `eval/` | QA generation, RAGAs scoring and result tables |
| `tests/` | Offline test suite |
| `config.toml` | Models, retrieval settings and regulations |
| `results/` | Committed evaluation results |
| `data/` | Local working data: documents, indexes, cache, runs, memory (gitignored) |

## Team

Tushar Kant (B23ES1033) and Pragay (B23ES1024), Software and Data Engineering course project.