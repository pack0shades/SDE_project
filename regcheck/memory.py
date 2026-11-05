"""Expert feedback memory (paper 3.4.2, 4.4): past human decisions, recalled for similar requirements.

ponytail: one JSONL + FAISS store replaces the paper's Redis/Neo4j; move to a graph DB if cross-case reasoning is
ever needed.
"""

from .config import DATA, settings
from .schemas import Chunk, Feedback
from .store import VectorStore

STORE = DATA / "memory"
# ponytail: cosine cut-off measured on bge-large: a lightly reworded requirement scores ~0.9, a different one on
# the same topic ~0.72, and so does a heavy paraphrase, which is therefore missed. Re-rank first if that matters.
THRESHOLD = 0.8


def remember(fb: Feedback) -> None:
    store = VectorStore(STORE)
    store.add([Chunk(id=f"fb:{len(store.chunks)}", doc="memory", parent_id="memory", text=fb.requirement,
                     meta=fb.model_dump(mode="json"))])


def recall(requirement: str, regulation: str, k: int = 3) -> list[Feedback]:
    """Up to k past decisions on similar requirements for the same regulation, most similar first."""
    store = VectorStore(STORE)
    hits = store.search(requirement, top_n=settings().retrieval.k, mmr=False, rerank=False, parents=False)
    found = [Feedback.model_validate(h.chunk.meta) for h in hits if h.score >= THRESHOLD]
    return [fb for fb in found if fb.regulation == regulation][:k]


def as_prompt(fbs: list[Feedback]) -> str:
    """Recalled decisions, ready to embed in a prompt."""
    return "".join(f'\n<past_decision decision="{fb.decision}" system_said="{fb.status}" '
                   f'clauses="{", ".join(fb.clauses)}" requirement="{fb.requirement}">{fb.justification}'
                   f"</past_decision>" for fb in fbs)
