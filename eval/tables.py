"""The paper's Tables 4 to 7 from saved results. Run: uv run python -m eval.tables (writes results/tables.md).

Counts are flagged requirements (non_compliant or partial): there are no ground-truth labels, so, like the
paper, we report counts, and reviewer rejections are the false-positive measure.
"""

import json

from regcheck.config import ROOT, settings
from regcheck.ingest import store_path
from regcheck.pipeline import CHECKERS, RUNS
from regcheck.store import VectorStore

RESULTS = ROOT / "results"
METRICS = ("context_precision", "context_recall", "answer_relevancy", "faithfulness")


def md(title: str, header: list[str], rows: list[list]) -> str:
    lines = [f"### {title}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def ragas_tables() -> str:
    res = {(r["doc"], r["agent"], r["mode"]): r
           for r in (json.loads(p.read_text()) for p in sorted((RESULTS / "ragas").glob("*.json")))}
    print(f"[DEBUG] loaded {len(res)} ragas results: {sorted(res)}")
    t4 = [[doc, r["model"], r["n"], *(r["means"][m] for m in METRICS)]
          for (doc, _, mode), r in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][0])) if mode == "advanced"]
    print(f"[DEBUG] table 4: {len(t4)} rows")
    t5 = []
    for (doc, agent, mode), b in sorted(res.items()):
        if mode == "baseline" and (v := res.get((doc, agent, "advanced"))):
            tokens = sum(len(s.text) for s in VectorStore(store_path(doc)).sections.values()) // 4  # chars/4
            print(f"[DEBUG] table 5: {doc}/{agent} ~{tokens} tokens")
            t5.append([doc, b["model"], b["n"], f"{tokens // 1000}K",
                       *(x for m in METRICS for x in (b["means"][m], v["means"][m]))])
    print(f"[DEBUG] table 5: {len(t5)} rows")
    return (md("Table 4. RAG evaluation (advanced RAG) per model and document", ["Document", "Model", "#Q", *METRICS],
               t4) + "\n" +
            md("Table 5. Baseline RAG vs advanced RAG", ["Document", "Model", "#Q", "Size (#tokens)",
               *(f"{m} {s}" for m in METRICS for s in ("baseline", "advanced"))], t5))


def run_tables() -> str:
    regs = settings().regulations
    latest, mono = {}, {}
    for d in sorted(RUNS.iterdir()):  # timestamped names: the last run per (srs, regulation) wins
        print(f"[DEBUG] run dir {d.name}: run.json={(d / 'run.json').exists()}, "
              f"monolithic.json={(d / 'monolithic.json').exists()}")
        if (d / "run.json").exists():
            m = json.loads((d / "run.json").read_text())
            print(f"[DEBUG]   srs={m['srs']} reg={m['reg']} web_search={m.get('web_search', False)}")
            if (d / "monolithic.json").exists():
                mono[m["srs"], m["reg"]] = len(json.loads((d / "monolithic.json").read_text()))
            else:  # web-search runs are their own rows, so they never replace the default run
                latest[m["srs"], m["reg"], m.get("web_search", False)] = d
    print(f"[DEBUG] latest runs: { {k: v.name for k, v in latest.items()} }")
    print(f"[DEBUG] monolithic counts: {mono}")

    def counts(rnd: dict) -> list:
        flagged = [{f["requirement_id"] for f in rnd["reports"][a]["findings"] if f["status"] != "compliant"}
                   for a in CHECKERS]
        print(f"[DEBUG] iteration {rnd.get('iteration')}: flagged per checker {[len(x) for x in flagged]}")
        return [*map(len, flagged), len(set.intersection(*flagged))]

    t6, t7 = [], []
    for (srs, reg, web), d in sorted(latest.items()):
        name = regs[reg].name + (" + web search" if web else "")
        rounds = json.loads((d / "rounds.json").read_text())
        first = counts(rounds[0])
        last = counts(rounds[-1]) if len(rounds) > 1 else [f"NA ({n})" for n in first]  # no disputes
        t6.append([srs, name, *first, *last])
        issues = json.loads((d / "final.json").read_text())["issues"]
        dec = json.loads((d / "decisions.json").read_text()) if (d / "decisions.json").exists() else {}
        print(f"[DEBUG] {srs}/{reg} web={web}: {len(rounds)} rounds, {len(issues)} issues, {len(dec)} decisions")
        accepted = f"{sum(v['decision'] == 'accept' for v in dec.values())}/{len(dec)}" if dec else "not reviewed"
        t7.append([srs, name, len(issues), mono.get((srs, reg), "not run"), accepted])
    print(f"[DEBUG] table 6: {len(t6)} rows, table 7: {len(t7)} rows")
    agents = [f"{a} ({settings().agents[a].model})" for a in CHECKERS]
    head6 = [f"{it}: {a}" for it in ("iteration 1", "iteration 2") for a in [*agents, "common"]]
    return (md("Table 6. Flagged requirements per checker and iteration", ["SRS", "Regulation", *head6], t6)
            + "\n" + md("Table 7. Multi-agent framework vs monolithic baseline",
                        ["SRS", "Regulation", "Framework (final issues)",
                         f"Monolithic ({settings().agents['monolithic'].model})", "Reviewer accepted"], t7))


def main() -> None:
    text = "# Evaluation results\n\n" + ragas_tables() + "\n" + run_tables()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tables.md").write_text(text)
    print(f"[DEBUG] wrote {RESULTS / 'tables.md'} ({len(text)} chars)")
    print(text)


if __name__ == "__main__":
    main()