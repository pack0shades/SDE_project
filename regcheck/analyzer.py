"""Report Analyzer agent (paper 4.3): find disagreements, judge them, compile the final report."""

import re

from . import llm, prompts
from .checker import recalled, supported
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
    def check(items: list | dict) -> None:
        got = items if isinstance(items, dict) else [i.requirement_id for i in items]
        if sorted(got) != sorted(ids):
            raise ValueError(f"need exactly one object per requirement id: {sorted(ids)}")
    return check


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def leak(query: str, requirement: str, project: str) -> str:
    """Why `query` must not leave the machine, or "" if it may.

    ponytail: a lexical check (4 words copied from the requirement, or the project name); add named-entity
    detection if SRS documents ever carry client names the file name does not.
    """
    q, r = _words(query), _words(requirement)
    copied = {tuple(r[i:i + 4]) for i in range(len(r) - 3)}
    if any(tuple(q[i:i + 4]) in copied for i in range(len(q) - 3)):
        return "copies the requirement's wording"
    if f" {' '.join(_words(project))} " in f" {' '.join(q)} ":
        return "names the project"
    return ""


def _web(batch: list[Dispute], srs: VectorStore, reg: Regulation, trace: list[dict]) -> dict[str, str]:
    """Paper 3.3.1: public web snippets per dispute. Only the RA's abstracted query is sent to the search."""
    from ddgs import DDGS  # dependency group "web"
    from ddgs.exceptions import DDGSException

    ids = [d.requirement_id for d in batch]
    items = "\n".join(f'<dispute requirement_id="{d.requirement_id}">\n<requirement>{srs.chunk(d.requirement_id).text}'
                      f'</requirement>\n<rationale>{d.rationale}</rationale>\n</dispute>' for d in batch)
    msgs = prompts.alpaca(prompts.RA_PERSONA.format(regulation=reg.name),
                          prompts.SANITIZE_QUERY.format(regulation=reg.name, example_req=ids[0],
                                                        data_note=prompts.DATA_NOTE),
                          f"<disputes>\n{items}\n</disputes>")
    found = {}
    for rid, query in llm.complete_json(AGENT, msgs, dict[str, str], trace, validate=_one_each(ids)).items():
        entry = {"agent": AGENT, "requirement_id": rid, "web_query": query}
        trace.append(entry)
        if why := leak(query, srs.chunk(rid).text, srs.path.name):
            entry["web_blocked"] = why  # fail closed: no search rather than a leaky one
            continue
        try:
            hits = DDGS().text(query, max_results=3)
        except DDGSException as e:  # rate limits, timeouts: web context is optional, the check goes on
            entry["web_error"] = str(e)[:200]
            continue
        entry["web_results"] = [h["href"] for h in hits]
        found[rid] = "".join(f'\n<web_result url="{h["href"]}">{h["title"]}: {h["body"]}</web_result>' for h in hits)
    return found


def validate(disputes: list[Dispute], srs: VectorStore, policy: VectorStore, reg: Regulation,
             trace: list[dict]) -> list[Dispute]:
    """Prompts (iii) and (iv): does the flagging rationale hold, and what should the other side reconsider."""
    out = []
    for batch in _batches(disputes):
        web = _web(batch, srs, reg, trace) if settings().pipeline.web_search else {}
        ids = list(dict.fromkeys(c for d in batch for c in d.clauses))
        arts = "\n\n".join(f'<article id="{i}">{policy.section(i).ref}\n{policy.section(i).text}</article>'
                           for i in ids)
        items = "\n".join(f'<dispute requirement_id="{d.requirement_id}" flagged_by="{d.flagged_by}" '
                          f'missed_by="{d.missed_by}">\n<requirement>{srs.chunk(d.requirement_id).text}'
                          f'</requirement>\n<rationale clauses="{", ".join(d.clauses)}">{d.rationale}</rationale>'
                          f'{web.get(d.requirement_id, "")}\n</dispute>' for d in batch)
        msgs = prompts.alpaca(prompts.RA_PERSONA.format(regulation=reg.name),
                              prompts.RA_VALIDATE.format(regulation=reg.name, example_req=batch[0].requirement_id,
                                                         web_note=prompts.WEB_NOTE if web else "",
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

    past = recalled([srs.chunk(rid) for rid in flagged], reg, AGENT, trace)

    def entry(rid: str) -> str:
        d = disputed.get(rid)
        history = (f'\n<dispute flagged_by="{d.flagged_by}" rationale_valid="{d.valid}">{d.message}</dispute>'
                   if d else "")
        positions = "\n".join(_position(a, fs.get(rid)) for a, fs in found.items())
        return (f'<requirement id="{rid}" agreement="{agreement(rid)}">\n<text>{srs.chunk(rid).text}</text>\n'
                f'{positions}{history}{past[rid]}\n</requirement>')

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
