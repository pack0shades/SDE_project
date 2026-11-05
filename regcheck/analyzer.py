"""Report Analyzer agent (paper 4.3): find disagreements, judge them, compile the final report."""

from . import llm, prompts
from .checker import supported
from .config import Regulation, settings
from .schemas import Dispute, FinalReport, Finding, Issue, Report, Resolution, Verdict
from .store import VectorStore

AGENT = "ra"


def diff(a: Report, b: Report) -> list[Dispute]:
    """Prompts (i) and (ii): requirements flagged by exactly one agent. Structured reports make it a set difference."""
    disputes = []
    for x, y in ((a, b), (b, a)):
        other = {f.requirement_id for f in y.findings if f.flagged}
        disputes += [Dispute(requirement_id=f.requirement_id, flagged_by=x.agent, missed_by=y.agent,
                             rationale=f.rationale, clauses=f.clauses)
                     for f in x.findings if f.flagged and f.requirement_id not in other]
    return disputes


def _batches[T](items: list[T]) -> list[list[T]]:
    # ponytail: fixed batches keep each RA reply under the free tiers' output caps; one call for a toy SRS
    b = settings().pipeline.batch_size
    return [items[i:i + b] for i in range(0, len(items), b)]


def _one_each(ids: list[str]):
    def check(items: list) -> None:
        if sorted(i.requirement_id for i in items) != sorted(ids):
            raise ValueError(f"need exactly one object per requirement id: {sorted(ids)}")
    return check


def validate(disputes: list[Dispute], srs: VectorStore, policy: VectorStore, reg: Regulation,
             trace: list[dict]) -> list[Dispute]:
    """Prompts (iii) and (iv): does the flagging rationale hold, and what should the other side reconsider."""
    out = []
    for batch in _batches(disputes):
        ids = list(dict.fromkeys(c for d in batch for c in d.clauses))
        arts = "\n\n".join(f'<article id="{i}">{policy.section(i).ref}\n{policy.section(i).text}</article>'
                           for i in ids)
        items = "\n".join(f'<dispute requirement_id="{d.requirement_id}" flagged_by="{d.flagged_by}" '
                          f'missed_by="{d.missed_by}">\n<requirement>{srs.chunk(d.requirement_id).text}'
                          f'</requirement>\n<rationale clauses="{", ".join(d.clauses)}">{d.rationale}</rationale>'
                          f'\n</dispute>' for d in batch)
        msgs = prompts.alpaca(prompts.RA_PERSONA.format(regulation=reg.name),
                              prompts.RA_VALIDATE.format(regulation=reg.name, example_req=batch[0].requirement_id,
                                                         data_note=prompts.DATA_NOTE),
                              f"<articles>\n{arts}\n</articles>\n\n<disputes>\n{items}\n</disputes>")
        ids = [d.requirement_id for d in batch]
        verdicts = {v.requirement_id: v for v in
                    llm.complete_json(AGENT, msgs, list[Verdict], trace, validate=_one_each(ids))}
        out += [d.model_copy(update=verdicts[d.requirement_id].model_dump(include={"valid", "message"}))
                for d in batch]
    return out


def _position(agent: str, f: Finding | None) -> str:
    if f is None:
        return f'<position agent="{agent}" status="compliant">Not selected as relevant to the regulation.</position>'
    return (f'<position agent="{agent}" status="{f.status}" clauses="{", ".join(f.clauses)}">{f.rationale}\n'
            f'Suggested rewrite: {f.suggestion or "none"}</position>')


def compile(reports: dict[str, Report], disputes: list[Dispute], srs: VectorStore, reg: Regulation,
            trace: list[dict]) -> FinalReport:
    """Prompts (v) and (vi): final status, clauses, rationale, rewrite and confidence per flagged requirement."""
    found = {a: {f.requirement_id: f for f in r.findings} for a, r in reports.items()}
    flagged = list(dict.fromkeys(f.requirement_id for r in reports.values() for f in r.findings if f.flagged))
    disputed = {d.requirement_id: d for d in disputes}  # latest dispute per requirement

    def agreement(rid: str) -> str:
        if rid not in disputed:
            return "both checkers flagged it in iteration 1"
        if all(rid in fs and fs[rid].flagged for fs in found.values()):
            return "the checkers agreed only after revision"
        return "the checkers still disagree"

    def entry(rid: str) -> str:
        d = disputed.get(rid)
        history = (f'\n<dispute flagged_by="{d.flagged_by}" rationale_valid="{d.valid}">{d.message}</dispute>'
                   if d else "")
        positions = "\n".join(_position(a, fs.get(rid)) for a, fs in found.items())
        return (f'<requirement id="{rid}" agreement="{agreement(rid)}">\n<text>{srs.chunk(rid).text}</text>\n'
                f'{positions}{history}\n</requirement>')

    issues = []
    for batch in _batches(flagged):
        supplied = {rid: [c for fs in found.values() if rid in fs for c in fs[rid].clauses] for rid in batch}
        example_art = next((c for cs in supplied.values() for c in cs), f"{reg.name.lower()}:art1")
        msgs = prompts.alpaca(prompts.RA_PERSONA.format(regulation=reg.name),
                              prompts.RA_COMPILE.format(regulation=reg.name, example_req=batch[0],
                                                        example_art=example_art, data_note=prompts.DATA_NOTE),
                              "<requirements>\n" + "\n".join(entry(rid) for rid in batch) + "\n</requirements>")
        for res in llm.complete_json(AGENT, msgs, list[Resolution], trace, validate=_one_each(batch)):
            if res.status == "compliant":
                continue
            rid = res.requirement_id
            res.clauses = supported(res.clauses, supplied[rid], AGENT, rid, trace)
            positions = {a: fs[rid] for a, fs in found.items() if rid in fs}
            evidence = list(dict.fromkeys(e for f in positions.values() for e in f.evidence))
            issues.append(Issue(**res.model_dump(), requirement=srs.chunk(rid).text, positions=positions,
                                evidence=evidence))
    return FinalReport(summary=_summary(issues, srs, reg, trace), issues=issues)


def _summary(issues: list[Issue], srs: VectorStore, reg: Regulation, trace: list[dict]) -> str:
    if not issues:
        return f"No {reg.name} compliance issues were found."
    listing = "\n".join(f"- [{i.status}] {i.requirement} (articles: {', '.join(i.clauses)})" for i in issues)
    msgs = prompts.alpaca(prompts.RA_PERSONA.format(regulation=reg.name),
                          prompts.RA_SUMMARY.format(regulation=reg.name, srs=srs.path.name,
                                                    data_note=prompts.DATA_NOTE),
                          f"<issues>\n{listing}\n</issues>")
    return llm.complete(AGENT, msgs, trace).strip()
