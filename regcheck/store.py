"""Vector store and retriever (paper 4.2.1, 4.2.3): KNN -> MMR -> cross-encoder re-rank -> parent documents."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import settings
from .schemas import Chunk, Section


@dataclass
class Hit:
    chunk: Chunk
    score: float
    section: Section | None = None  # parent, when retrieved with parents=True


@cache
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings().embedding.model)


@cache
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings().rerank.model)


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    """L2-normalized embeddings. BGE v1.5 wants the instruction prefix on queries only."""
    if query:
        texts = [settings().embedding.query_prefix + t for t in texts]
    return _embedder().encode(texts, normalize_embeddings=True, batch_size=16).astype("float32")


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _reranker().predict([(query, t) for t in texts])]


def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, keep: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: greedily trade relevance to the query against redundancy with picks so far."""
    rel = cand_vecs @ query_vec
    chosen = [int(np.argmax(rel))]
    while len(chosen) < min(keep, len(cand_vecs)):
        redundancy = (cand_vecs @ cand_vecs[chosen].T).max(axis=1)
        score = lam * rel - (1 - lam) * redundancy
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen


def _read[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines()] if path.exists() else []


def _append(path: Path, items: Iterable[BaseModel]) -> None:
    with path.open("a") as f:
        f.writelines(item.model_dump_json() + "\n" for item in items)


class VectorStore:
    """One directory per document: chunks.jsonl, sections.jsonl, index.faiss (IndexFlatL2, paper Table 2)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.chunks = _read(self.path / "chunks.jsonl", Chunk)
        self.sections = {s.id: s for s in _read(self.path / "sections.jsonl", Section)}
        self._by_id = {c.id: c for c in self.chunks}
        idx = self.path / "index.faiss"
        self.index = faiss.read_index(str(idx)) if idx.exists() else None
        n = self.index.ntotal if self.index else 0
        if n != len(self.chunks):
            raise RuntimeError(f"{self.path}: {n} vectors but {len(self.chunks)} chunks; delete it and re-index")

    def add(self, chunks: list[Chunk], sections: Iterable[Section] = ()) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        sections = list(sections)
        if chunks:
            vecs = embed([c.text for c in chunks])
            if self.index is None:
                self.index = faiss.IndexFlatL2(vecs.shape[1])
            self.index.add(vecs)
            faiss.write_index(self.index, str(self.path / "index.faiss"))
        _append(self.path / "chunks.jsonl", chunks)
        _append(self.path / "sections.jsonl", sections)
        self.chunks += chunks
        self._by_id |= {c.id: c for c in chunks}
        self.sections |= {s.id: s for s in sections}

    def chunk(self, id: str) -> Chunk:
        return self._by_id[id]

    def section(self, id: str) -> Section:
        return self.sections[id]

    def search(self, query: str, *, top_n: int | None = None, mmr: bool = True, rerank: bool = True,
               parents: bool = True) -> list[Hit]:
        """All flags off is the baseline RAG: plain top-n vector similarity over chunks."""
        if not self.chunks:
            return []
        r = settings().retrieval
        q = embed([query], query=True)
        dist, ids = self.index.search(q, min(r.k, len(self.chunks)))
        ids, sims = ids[0], 1 - dist[0] / 2  # squared L2 on unit vectors -> cosine
        if mmr:
            order = _mmr(q[0], self.index.reconstruct_batch(ids), r.mmr_keep, r.mmr_lambda)
            ids, sims = ids[order], sims[order]
        hits = [Hit(self.chunks[i], float(s)) for i, s in zip(ids, sims)]
        if rerank:  # most relevant first: the paper's answer to "lost in the middle"
            for h, s in zip(hits, rerank_scores(query, [h.chunk.text for h in hits])):
                h.score = s
            hits.sort(key=lambda h: -h.score)
        if parents:  # parent document retrieval: keep the best chunk per parent
            seen: set[str] = set()
            unique = []
            for h in hits:
                if h.chunk.parent_id not in seen:
                    seen.add(h.chunk.parent_id)
                    h.section = self.sections[h.chunk.parent_id]
                    unique.append(h)
            hits = unique
        return hits[: top_n or r.top_n]
