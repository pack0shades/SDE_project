"""Ground-truth question-answer pairs for RAGAs (paper 6.1.1).

Run: uv run python -m eval.gen_qa gdpr --n 30
Review and correct data/eval/<doc>.qa.jsonl by hand before scoring; that stands in for the paper's
five-annotator validation.
"""

import argparse
import random
import re

from regcheck import llm, prompts
from regcheck.config import DATA
from regcheck.ingest import store_path
from regcheck.schemas import QA
from regcheck.store import VectorStore

OUT = DATA / "eval"
MIN_CHARS = 300  # shorter sections (headings, one-line scope notes) cannot carry 3 to 5 questions
# "What is covered in section 2 of the document?" breaks the prompt's self-contained rule and no retriever can
# answer it. ponytail: a regex pre-filter; the team's hand review catches the subtler ones.
SELF_REFERENCE = re.compile(r"\b(the|this|that)\s+(provided\s+|given\s+|above\s+)?(document|text|passage|section|"
                            r"chapter|excerpt)\b|\bsection\s+\d|\bchapter\s+[IVX\d]", re.IGNORECASE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("doc", help="index name, e.g. gdpr or e-store")
    p.add_argument("--n", type=int, default=30, help="sections to sample")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = OUT / f"{a.doc}.qa.jsonl"
    if out.exists():  # never clobber a hand-reviewed file
        raise SystemExit(f"{out} exists; delete it to regenerate")

    sections = [s for s in VectorStore(store_path(a.doc)).sections.values() if len(s.text) >= MIN_CHARS]
    rng = random.Random(a.seed)
    picked = rng.sample(sections, min(a.n, len(sections)))

    def sane(pairs: list[QA]) -> None:
        if not 3 <= len(pairs) <= 5:
            raise ValueError(f"expected 3 to 5 pairs, got {len(pairs)}")

    pairs, trace = [], []
    for i, s in enumerate(picked, 1):
        msgs = [{"role": "user", "content": prompts.GEN_QA.format(text=f"{s.ref}\n{s.text}")}]
        new = llm.complete_json("judge", msgs, list[QA], trace, validate=sane)
        pairs += [q.model_copy(update={"section": s.id}) for q in new if not SELF_REFERENCE.search(q.question)]
        print(f"{i}/{len(picked)} {s.ref[:60]}: {len(pairs)} pairs")
    rng.shuffle(pairs)  # so ragas_eval --limit n spans many sections, not the first few
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(q.model_dump_json() + "\n" for q in pairs))
    print(f"wrote {out}; review it before running ragas_eval")


if __name__ == "__main__":
    main()
