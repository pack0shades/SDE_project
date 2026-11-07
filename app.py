"""Human-in-the-loop review UI (paper 3.4). Run: uv run streamlit run app.py"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from regcheck import memory, pipeline
from regcheck.config import DATA, settings
from regcheck.ingest import index, store_path
from regcheck.schemas import Feedback, FinalReport, Issue
from regcheck.store import VectorStore

st.set_page_config(page_title="RegCheck", layout="wide")
regs = settings().regulations


@st.cache_resource
def store(name: str) -> VectorStore:
    return VectorStore(store_path(name))


def runs() -> list[str]:
    return sorted((p.name for p in pipeline.RUNS.glob("*") if (p / "final.json").exists()), reverse=True)


# --- Sidebar: choose inputs, run a check, or open a past run ---
with st.sidebar:
    st.header("New check")
    reg_key = st.selectbox("Regulation", list(regs), format_func=lambda k: regs[k].name)
    indexed = sorted(p.name for p in (DATA / "index").glob("*")
                     if p.is_dir() and p.name not in regs and not p.name.endswith("-fixed"))
    srs = st.selectbox("Indexed SRS", indexed)
    upload = st.file_uploader("…or upload an SRS", type=["txt", "pdf"])
    st.caption("Free-tier LLM providers may train on what you send. Do not upload confidential documents.")
    if st.button("Run check", type="primary"):
        with st.status("Working…", expanded=True) as status:
            if upload:
                path = DATA / "srs" / upload.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(upload.getvalue())
                srs = path.stem
                index(path, srs, "srs", progress=status.write)
            st.session_state.run = pipeline.run(srs, reg_key, progress=status.write).name
            status.update(label="Done", state="complete")
    st.header("Past runs")
    names = runs()
    if not names:
        st.stop()
    current = st.session_state.get("run", names[0])
    st.session_state.run = st.selectbox("Run", names, index=names.index(current) if current in names else 0)

run_dir = pipeline.RUNS / st.session_state.run
meta = json.loads((run_dir / "run.json").read_text())
reg = regs[meta["reg"]]
policy = store(meta["reg"])
final = FinalReport.model_validate_json((run_dir / "final.json").read_text())
decisions_file = run_dir / "decisions.json"
decisions: dict = json.loads(decisions_file.read_text()) if decisions_file.exists() else {}

st.title(f"{meta['srs']} · {reg.name}")
report_tab, trace_tab = st.tabs(["Report", "Trace"])


def agreement(issue: Issue) -> str:
    statuses = [p.status for p in issue.positions.values()]
    if len(statuses) < 2 or not all(s != "compliant" for s in statuses):
        return "⚠️ checkers disagree"
    return "✅ checkers agree" if len(set(statuses)) == 1 else "◐ both flag it, severity differs"


def review(issue: Issue) -> None:
    rid = issue.requirement_id
    st.markdown(f"**Requirement** `{rid}`\n\n> {issue.requirement}")
    st.markdown(f"**Status** `{issue.status}` · **confidence** {issue.confidence:.2f} · {agreement(issue)}")
    if issue.clauses:  # regulation mapping
        for tab, cid in zip(st.tabs([policy.section(c).ref for c in issue.clauses]), issue.clauses):
            tab.text(policy.section(cid).text[:3000])
    st.markdown(f"**Analyzer rationale**\n\n{issue.rationale}")
    cols = st.columns(max(len(issue.positions), 1))  # rationale visualization
    for col, (agent, f) in zip(cols, issue.positions.items()):
        col.markdown(f"**{agent}** ({settings().agents[agent].model}): `{f.status}` {', '.join(f.clauses)}")
        col.caption(f.rationale)
    if issue.evidence:  # retrieval path
        st.markdown("**Retrieved evidence**")
        for eid in issue.evidence:
            st.caption(f"`{eid}` {policy.chunk(eid).text}")
    st.markdown("**Suggested rewrite**")
    st.info(issue.suggestion or "none")

    saved = decisions.get(rid, {})
    options = ["accept", "reject", "escalate"]
    choice = st.segmented_control("Decision", options, key=f"d-{rid}", default=saved.get("decision"))
    why = st.text_area("Justification (required)", saved.get("justification", ""), key=f"j-{rid}")
    if st.button("Save decision", key=f"s-{rid}"):
        if not choice or not why.strip():
            st.error("Choose a decision and give a justification.")
            return
        decisions[rid] = {"decision": choice, "justification": why.strip(), "ts": datetime.now().isoformat()}
        decisions_file.write_text(json.dumps(decisions, indent=2))
        # ponytail: every save adds a memory entry, so a changed mind leaves the old decision recallable too
        memory.remember(Feedback(requirement=issue.requirement, regulation=reg.name, clauses=issue.clauses,
                                 status=issue.status, decision=choice, justification=why.strip()))
        st.success("Saved and added to the feedback memory.")


def export() -> str:
    lines = [f"# {meta['srs']}: {reg.name} review\n"]
    for title, wanted in (("Accepted rewrites", "accept"), ("Escalated", "escalate")):
        lines.append(f"## {title}\n")
        for i in final.issues:
            d = decisions.get(i.requirement_id)
            if d and d["decision"] == wanted:
                lines.append(f"- **{i.requirement_id}** {i.requirement}\n  - {i.status}, {', '.join(i.clauses)}\n"
                             f"  - Rewrite: {i.suggestion}\n  - Reviewer: {d['justification']}\n")
    return "\n".join(lines)


with report_tab:
    st.write(final.summary)
    c = st.columns(4)
    c[0].metric("Issues", len(final.issues))
    c[1].metric("Non-compliant", sum(i.status == "non_compliant" for i in final.issues))
    c[2].metric("Partial", sum(i.status == "partial" for i in final.issues))
    c[3].metric("Reviewed", f"{len(decisions)}/{len(final.issues)}")
    st.dataframe([{"id": i.requirement_id, "status": i.status, "confidence": i.confidence,
                   "articles": ", ".join(i.clauses), "requirement": i.requirement,
                   "decision": decisions.get(i.requirement_id, {}).get("decision", "")} for i in final.issues],
                 hide_index=True, width="stretch")
    for issue in final.issues:
        mark = decisions.get(issue.requirement_id, {}).get("decision", "")
        with st.expander(f"`{issue.status}` {issue.confidence:.2f} · {issue.requirement[:100]} {mark}"):
            review(issue)
    st.download_button("Export accepted and escalated (Markdown)", export(), f"{run_dir.name}-review.md")

with trace_tab:  # XAI: every prompt, retrieval and reply of the run, in order
    trace = json.loads(Path(run_dir / "trace.json").read_text())
    for n, t in enumerate(trace, 1):
        if "messages" in t:
            with st.expander(f"{n}. {t['agent']} · {t['model']}{' · cached' if t.get('cached') else ''}"):
                st.markdown("**Prompt**")
                st.code(t["messages"][-1]["content"], language=None, wrap_lines=True, height=300)
                st.markdown("**Response**")
                st.code(t["response"], language=None, wrap_lines=True, height=300)
        else:
            if "recalled" in t:
                what = f"recalled {len(t['recalled'])} past decision(s): " + "; ".join(
                    f"{fb['decision']}: {fb['justification']}" for fb in t["recalled"])
            elif "web_query" in t:
                what = f'web search "{t["web_query"]}": ' + (
                    f"blocked, {t['web_blocked']}" if "web_blocked" in t else f"failed, {t['web_error']}"
                    if "web_error" in t else ", ".join(t["web_results"]) or "no results")
            else:
                what = f"dropped invented clause ids {t['dropped_clauses']}"
            st.caption(f"{n}. {t['agent']} · {t['requirement_id']}: {what}")
