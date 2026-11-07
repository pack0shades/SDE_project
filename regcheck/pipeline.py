"""Agent communication (paper Fig. 1): two checkers, dispute and refinement, final report."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import analyzer, checker, llm, prompts
from .config import DATA, settings
from .ingest import store_path
from .schemas import Finding
from .store import VectorStore

RUNS = DATA / "runs"
CHECKERS = ("cc1", "cc2")  # different vendors, so the two do not share one model's blind spots


def _stores(srs_name: str, reg_key: str) -> tuple[VectorStore, VectorStore]:
    srs, policy = VectorStore(store_path(srs_name)), VectorStore(store_path(reg_key))
    for name, store in ((srs_name, srs), (reg_key, policy)):
        if not store.chunks:
            raise SystemExit(f"'{name}' is not indexed; run: python -m regcheck index {name}")
    return srs, policy


def _save(name: str, files: dict[str, object]) -> Path:
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    out.mkdir(parents=True)
    for fname, data in files.items():
        (out / fname).write_text(json.dumps(data, indent=2, default=str))
    return out


def run(srs_name: str, reg_key: str, progress: Callable[[str], None] = print) -> Path:
    """Full multi-agent check. Writes rounds.json, final.json and trace.json into a new run directory."""
    reg = settings().regulations[reg_key]
    srs, policy = _stores(srs_name, reg_key)
    trace: list[dict] = []
    try:
        return _run(srs_name, reg_key, srs, policy, reg, trace, progress)
    except Exception:
        progress(f"failed; trace saved to {_save(f'{srs_name}-{reg_key}-failed', {'trace.json': trace})}")
        raise


def _run(srs_name, reg_key, srs, policy, reg, trace, progress) -> Path:
    reports = {}
    for agent in CHECKERS:
        progress(f"{agent}: checking {srs_name} against {reg.name}")
        reports[agent] = checker.check(agent, srs, policy, reg, trace)
    rounds = [{"iteration": 1, "reports": {a: r.model_dump() for a, r in reports.items()}}]

    all_disputes = []
    for i in range(2, settings().pipeline.max_rounds + 2):
        disputes = analyzer.diff(*reports.values())
        progress(f"iteration {i}: {len(disputes)} disputes")
        if not disputes:
            break
        disputes = analyzer.validate(disputes, srs, policy, reg, trace)
        all_disputes += disputes
        for agent, report in reports.items():
            # valid rationale: the agent that missed it reconsiders; invalid: the agent that flagged it does
            notes = {d.requirement_id: d.message for d in disputes
                     if (d.missed_by if d.valid else d.flagged_by) == agent}
            if not notes:
                continue
            progress(f"{agent}: re-assessing {len(notes)} requirements")
            reqs = [srs.chunk(rid) for rid in notes]
            b = settings().pipeline.batch_size
            new = {f.requirement_id: f for j in range(0, len(reqs), b)
                   for f in checker.assess(agent, reqs[j:j + b], policy, reg, trace, notes, report.plan)}
            report.findings = [new.pop(f.requirement_id, f) for f in report.findings] + list(new.values())
        rounds.append({"iteration": i, "disputes": [d.model_dump() for d in disputes],
                       "reports": {a: r.model_dump() for a, r in reports.items()}})

    progress("ra: compiling the final report")
    final = analyzer.compile(reports, all_disputes, srs, reg, trace)
    out = _save(f"{srs_name}-{reg_key}", {"run.json": {"srs": srs_name, "reg": reg_key,
                                                       "web_search": settings().pipeline.web_search},
                                          "rounds.json": rounds,
                                          "final.json": final.model_dump(),
                                          "trace.json": trace})
    progress(f"{len(final.issues)} issues -> {out}")
    return out


def monolithic(srs_name: str, reg_key: str) -> Path:
    """Paper 6.4 baseline: the whole SRS and the whole regulation in one prompt to one model."""
    reg = settings().regulations[reg_key]
    srs, policy = _stores(srs_name, reg_key)
    trace: list[dict] = []
    reqs = "\n".join(f'<requirement id="{c.id}">{c.text}</requirement>' for c in srs.chunks)
    arts = "\n\n".join(f'<article id="{s.id}">{s.ref}\n{s.text}</article>' for s in policy.sections.values())
    msgs = prompts.alpaca(prompts.CC_PERSONA.format(regulation=reg.name),
                          prompts.MONOLITHIC.format(regulation=reg.name, example_req=srs.chunks[0].id,
                                                    example_art=next(iter(policy.sections)),
                                                    data_note=prompts.DATA_NOTE),
                          f"<articles>\n{arts}\n</articles>\n\n<requirements>\n{reqs}\n</requirements>")
    known = {c.id for c in srs.chunks}
    findings = [f for f in llm.complete_json("monolithic", msgs, list[Finding], trace)
                if f.flagged and f.requirement_id in known]
    for f in findings:
        f.clauses = checker.supported(f.clauses, list(policy.sections), "monolithic", f.requirement_id, trace)
    return _save(f"{srs_name}-{reg_key}-monolithic",
                 {"run.json": {"srs": srs_name, "reg": reg_key}, "monolithic.json": [f.model_dump() for f in findings],
                  "trace.json": trace})
