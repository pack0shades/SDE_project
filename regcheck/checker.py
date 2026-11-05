"""Compliance Checker agent (paper 3.2.1): plan, select requirements, then map, evaluate and suggest."""

import re
from collections.abc import Sequence

from . import llm, prompts, rag
from .config import Regulation, settings
from .schemas import Chunk, Finding, Report, Section
from .store import VectorStore


def check(agent: str, srs: VectorStore, policy: VectorStore, reg: Regulation, trace: list[dict]) -> Report:
    """One CC agent's full pass over an SRS for one regulation."""
    plan = rag.decompose(agent, prompts.HIGH_LEVEL.format(regulation=reg.name, focus=reg.focus), trace)
    k = settings().retrieval.k
    candidates: dict[str, Chunk] = {}
    for q in plan:  # sub-prompt 1: plain top-k, the screening call does the judging
        for h in srs.search(q, top_n=k, mmr=False, rerank=False, parents=False):
            candidates.setdefault(h.chunk.id, h.chunk)
    reqs = screen(agent, list(candidates.values()), reg, trace)
    b = settings().pipeline.batch_size
    findings = [f for i in range(0, len(reqs), b)
                for f in assess(agent, reqs[i:i + b], policy, reg, trace, plan=plan)]
    return Report(agent=agent, findings=findings, plan=plan)


def _requirements(reqs: list[Chunk]) -> str:
    return "\n".join(f'<requirement id="{r.id}">{r.text}</requirement>' for r in reqs)


def screen(agent: str, candidates: list[Chunk], reg: Regulation, trace: list[dict]) -> list[Chunk]:
    """Keep the candidates that concern the regulation, in one LLM call."""
    if not candidates:
        return []
    instruction = prompts.SCREEN.format(regulation=reg.name, focus=reg.focus, example_id=candidates[0].id,
                                        data_note=prompts.DATA_NOTE)
    msgs = prompts.alpaca(prompts.CC_PERSONA.format(regulation=reg.name), instruction,
                          f"<requirements>\n{_requirements(candidates)}\n</requirements>")
    keep = set(llm.complete_json(agent, msgs, list[str], trace))
    return [c for c in candidates if c.id in keep]  # unknown ids are ignored


def _parent(cid: str, supplied: list[str]) -> str | None:
    """Map 'gdpr:art32.1(a)' or 'gdpr:art32(1)' to the supplied 'gdpr:art32'; 'gdpr:art3' never matches art32."""
    for s in supplied:
        if cid == s or (cid.startswith(s) and not cid[len(s)].isdigit()):
            return s
    return None


def _hinted(reg: Regulation, policy: VectorStore) -> list[Section]:
    """Articles named in the regulation's hints, e.g. "Article 32" or "Articles 15 to 17"."""
    nums = [n for h in reg.hints for a, b in re.findall(r"Articles? (\d+)(?: to (\d+))?", h)
            for n in range(int(a), int(b or a) + 1)]
    doc = policy.chunks[0].doc
    return [policy.sections[i] for i in dict.fromkeys(f"{doc}:art{n}" for n in nums) if i in policy.sections]


def supported(ids: list[str], supplied: list[str], agent: str, requirement_id: str, trace: list[dict]) -> list[str]:
    """Clause ids mapped onto the articles actually shown to the LLM; invented ones are dropped and traced."""
    mapped = [_parent(c, supplied) for c in ids]
    if dropped := [c for c, m in zip(ids, mapped) if m is None]:
        trace.append({"agent": agent, "requirement_id": requirement_id, "dropped_clauses": dropped})
    return list(dict.fromkeys(m for m in mapped if m))


def assess(agent: str, reqs: list[Chunk], policy: VectorStore, reg: Regulation, trace: list[dict],
           notes: dict[str, str] | None = None, plan: Sequence[str] = ()) -> list[Finding]:
    """Sub-prompts 2 to 4 for one batch. `notes` are RA revision requests; they also steer retrieval (4.2.3 d).

    A requirement's own words only retrieve what it mentions. The plan's sub-queries add the obligations it may
    leave out (retention, data-subject rights...), which is what a gap check needs.
    """
    notes = notes or {}
    hits = {r.id: policy.search(f"{r.text} {notes.get(r.id, '')}".strip()) for r in reqs}
    articles = {h.section.id: h.section for hs in hits.values() for h in hs}  # de-duplicated, ranked order
    for q in plan:
        for h in policy.search(q, top_n=2):
            articles.setdefault(h.section.id, h.section)
    for sec in _hinted(reg, policy):  # reviewers' hints name articles the model must be able to cite
        articles.setdefault(sec.id, sec)
    supplied = list(articles)

    def entry(r: Chunk) -> str:
        note = f"\n<review_note>{notes[r.id]}</review_note>" if r.id in notes else ""
        relevant = ", ".join(h.section.id for h in hits[r.id])
        return f'<requirement id="{r.id}" relevant="{relevant}">{r.text}{note}</requirement>'

    def one_each(fs: list[Finding]) -> None:
        if sorted(f.requirement_id for f in fs) != sorted(hits):
            raise ValueError(f"need exactly one finding per requirement id: {sorted(hits)}")

    hints = prompts.HINTS.format(items="\n".join(f"- {h}" for h in reg.hints)) if reg.hints else ""
    instruction = prompts.ASSESS.format(regulation=reg.name, hints=hints, example_req=reqs[0].id,
                                        example_art=supplied[0], data_note=prompts.DATA_NOTE)
    arts = "\n\n".join(f'<article id="{s.id}">{s.ref}\n{s.text}</article>' for s in articles.values())
    msgs = prompts.alpaca(prompts.CC_PERSONA.format(regulation=reg.name), instruction,
                          f"<articles>\n{arts}\n</articles>\n\n"
                          f"<requirements>\n{'\n'.join(entry(r) for r in reqs)}\n</requirements>")
    findings = llm.complete_json(agent, msgs, list[Finding], trace, validate=one_each)

    for f in findings:
        f.clauses = supported(f.clauses, supplied, agent, f.requirement_id, trace)
        f.evidence = [h.chunk.id for h in hits[f.requirement_id]]
    return findings
