"""RAGAs scores for one document, agent and RAG mode (paper Tables 4 and 5).

Run: uv run python -m eval.ragas_eval gdpr --agent cc1 --mode advanced --limit 10
Reads data/eval/<doc>.qa.jsonl (from gen_qa), writes results/ragas/<doc>-<agent>-<mode>.json.
"""

import argparse
import json
import math
import os
from statistics import fmean

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # ragas posts usage telemetry to its server by default

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from regcheck import llm, store
from regcheck.config import ROOT, settings
from regcheck.ingest import store_path
from regcheck.rag import answer
from regcheck.schemas import QA

from .gen_qa import OUT as QA_DIR

RESULTS = ROOT / "results" / "ragas"


class Judge(InstructorBaseRagasLLM):
    """ragas' structured calls, routed through llm.py so judging is cached, retried and traced like the rest."""

    def __init__(self, trace: list[dict]):
        self.trace = trace

    def generate(self, prompt, response_model):
        print(f"[DEBUG] judge call: response_model={response_model}, prompt_chars={len(prompt)}")
        return llm.complete_json("judge", [{"role": "user", "content": prompt}], response_model, self.trace)

    async def agenerate(self, prompt, response_model):
        return self.generate(prompt, response_model)  # sequential on purpose: free-tier requests per minute


class Embedder(BaseRagasEmbedding):
    """The pipeline's own bge-large model, for answer relevancy."""

    def embed_text(self, text, **kw):
        print(f"[DEBUG] embedding {len(text)} chars")
        return store.embed([text])[0].tolist()

    async def aembed_text(self, text, **kw):
        return self.embed_text(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("doc", help="index name, e.g. gdpr or e-store")
    p.add_argument("--agent", default="cc1", help="answering agent from config.toml")
    p.add_argument("--mode", choices=["advanced", "baseline"], default="advanced")
    p.add_argument("--limit", type=int, default=10, help="first n questions of the QA file")
    a = p.parse_args()
    print(f"[DEBUG] args: doc={a.doc} agent={a.agent} ({settings().agents[a.agent].model}) mode={a.mode} "
          f"limit={a.limit}")
    qas = [QA.model_validate_json(line) for line in (QA_DIR / f"{a.doc}.qa.jsonl").read_text().splitlines()]
    print(f"[DEBUG] loaded {len(qas)} QA pairs from {QA_DIR / f'{a.doc}.qa.jsonl'}")
    qas = qas[: a.limit]
    print(f"[DEBUG] scoring {len(qas)} QA pairs")

    trace: list[dict] = []
    judge = Judge(trace)
    # ponytail: strictness 1 (ragas default 3) saves two judge calls per question; raise it if quota allows
    cp, cr, f = ContextPrecision(llm=judge), ContextRecall(llm=judge), Faithfulness(llm=judge)
    ar = AnswerRelevancy(llm=judge, embeddings=Embedder(), strictness=1)
    metrics = {  # same order and names as the paper's tables
        "context_precision": lambda q, ans: cp.score(user_input=q.question, reference=q.answer,
                                                     retrieved_contexts=ans.contexts),
        "context_recall": lambda q, ans: cr.score(user_input=q.question, reference=q.answer,
                                                  retrieved_contexts=ans.contexts),
        "answer_relevancy": lambda q, ans: ar.score(user_input=q.question, response=ans.text),
        "faithfulness": lambda q, ans: f.score(user_input=q.question, response=ans.text,
                                               retrieved_contexts=ans.contexts),
    }
    print(f"[DEBUG] metrics: {list(metrics)}")
    if a.mode == "advanced":  # CoT, agentic chunks, MMR, re-ranking, parent articles
        vs, kw = store.VectorStore(store_path(a.doc)), {}
    else:  # fixed chunks, one-shot plain top-n similarity
        vs = store.VectorStore(store_path(a.doc, "fixed"))
        kw = {"cot": False, "mmr": False, "rerank": False, "parents": False}
    print(f"[DEBUG] store {vs.path}: {len(vs.chunks)} chunks, {len(vs.sections)} sections; answer kwargs={kw}")

    rows = []
    for i, qa in enumerate(qas, 1):
        print(f"[DEBUG] question {i} (section {qa.section}): {qa.question}")
        ans = answer(a.agent, qa.question, vs, trace, **kw)
        print(f"[DEBUG] answer: {len(ans.text)} chars, {len(ans.steps)} steps, {len(ans.contexts)} contexts, "
              f"sources={[s for st in ans.steps for s in st.sources]}")
        scores = {}
        for name, metric in metrics.items():
            try:
                scores[name] = metric(qa, ans).value
                print(f"[DEBUG] {name} = {scores[name]}")
            except ValueError as e:  # judge output unusable even after the repair retry: skip, do not guess
                print(f"  {name} skipped: {str(e)[:120]}")
                scores[name] = math.nan
        rows.append({"question": qa.question, "reference": qa.answer, "answer": ans.text,
                     "sources": [s for st in ans.steps for s in st.sources], "scores": scores})
        print(f"{i}/{len(qas)} " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()))

    print("[DEBUG] skipped scores per metric:", {k: sum(math.isnan(r["scores"][k]) for r in rows) for k in metrics})
    print(f"[DEBUG] {len(trace)} trace entries")
    means = {k: round(fmean(v for r in rows if not math.isnan(v := r["scores"][k])), 3) for k in metrics}
    out = RESULTS / f"{a.doc}-{a.agent}-{a.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"doc": a.doc, "agent": a.agent, "model": settings().agents[a.agent].model,
                               "mode": a.mode, "n": len(rows), "means": means, "questions": rows}, indent=2))
    print(f"{means} -> {out}")


if __name__ == "__main__":
    main()