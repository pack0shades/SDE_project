"""CoT query decomposition and question answering (paper 3.2.1, Fig. 2 and 3)."""

from . import llm, prompts
from .schemas import Answer, Step
from .store import Hit, VectorStore


def decompose(agent: str, question: str, trace: list[dict]) -> list[str]:
    """Break a question into 3 to 4 self-contained sub-queries (Fig. 3 prompt)."""

    def sane(subs: list[str]) -> None:
        if not 2 <= len(subs) <= 6 or not all(s.strip() for s in subs):
            raise ValueError(f"expected 3 to 4 non-empty sub-queries, got {len(subs)}")

    messages = [{"role": "user", "content": prompts.DECOMPOSE.format(question=question)}]
    return llm.complete_json(agent, messages, list[str], trace, validate=sane)


def passages(hits: list[Hit]) -> list[tuple[str, str]]:
    """(id, text) per hit: the parent article/section when retrieved with parents, else the chunk."""
    return [(h.section.id, f"{h.section.ref}\n{h.section.text}") if h.section else (h.chunk.id, h.chunk.text)
            for h in hits]


def _step(agent: str, question: str, store: VectorStore, history: list[dict], trace: list[dict],
          **search_kw) -> tuple[Step, list[str]]:
    found = passages(store.search(question, **search_kw))
    context = "\n\n".join(f"[{pid}] {text}" for pid, text in found)
    msg = prompts.alpaca(prompts.QA_PERSONA, prompts.QA_ANSWER.format(data_note=prompts.DATA_NOTE),
                         f"<context>\n{context}\n</context>\n\n<question>{question}</question>")
    text = llm.complete(agent, history + msg, trace)
    return Step(question=question, answer=text, sources=[pid for pid, _ in found]), [t for _, t in found]


def answer(agent: str, question: str, store: VectorStore, trace: list[dict], *, cot: bool = True,
           **search_kw) -> Answer:
    """With `cot`, answer each sub-query with its own retrieval, then combine (Fig. 2). Otherwise one shot."""
    if not cot:
        step, contexts = _step(agent, question, store, [], trace, **search_kw)
        return Answer(question=question, text=step.answer, contexts=contexts, steps=[step])
    steps, contexts, history = [], [], []
    for sub in decompose(agent, question, trace):
        step, found = _step(agent, sub, store, history, trace, **search_kw)
        steps.append(step)
        contexts += [t for t in found if t not in contexts]
        # earlier sub-answers stay in the conversation; their retrieved passages do not, to bound the prompt
        history += [{"role": "user", "content": sub}, {"role": "assistant", "content": step.answer}]
    final = llm.complete(agent, history + [{"role": "user", "content": prompts.QA_FINAL.format(question=question)}],
                         trace)
    return Answer(question=question, text=final, contexts=contexts, steps=steps)
