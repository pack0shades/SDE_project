"""CLI: python -m regcheck index|check|ask."""

import argparse
import sys
from pathlib import Path

from .config import ROOT, settings


def main() -> None:
    p = argparse.ArgumentParser(prog="regcheck")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="index regulations (by config key) or SRS files (by path)")
    ix.add_argument("targets", nargs="+", help="e.g. gdpr data/srs/e-store.txt")
    ix.add_argument("--fixed", action="store_true", help="fixed-size chunks for the baseline RAG")
    ask = sub.add_parser("ask", help="answer a question over one indexed document")
    ask.add_argument("question")
    ask.add_argument("--doc", required=True, help="index name, e.g. gdpr or e-store")
    ask.add_argument("--agent", default="cc1")
    ask.add_argument("--baseline", action="store_true", help="fixed chunks, no CoT, MMR, re-ranking or parents")
    chk = sub.add_parser("check", help="run the multi-agent compliance check on an indexed SRS")
    chk.add_argument("srs", help="SRS index name, e.g. e-store")
    chk.add_argument("--reg", required=True, help="regulation key from config.toml, e.g. gdpr")
    chk.add_argument("--monolithic", action="store_true", help="single-prompt baseline instead (paper 6.4)")
    a = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up even when redirected to a log

    if a.cmd == "index":
        from .ingest import index

        regs = settings().regulations
        for t in a.targets:
            if t in regs:
                index(ROOT / regs[t].file, t, "regulation", "fixed" if a.fixed else "agentic")
            else:
                path = Path(t)
                index(path, path.stem, "srs", "fixed" if a.fixed else "agentic")

    elif a.cmd == "check":
        from . import pipeline

        if a.monolithic:
            print(pipeline.monolithic(a.srs, a.reg))
        else:
            pipeline.run(a.srs, a.reg)

    elif a.cmd == "ask":
        from .ingest import store_path
        from .rag import answer
        from .store import VectorStore

        if a.baseline:
            store = VectorStore(store_path(a.doc, "fixed"))
            ans = answer(a.agent, a.question, store, [], cot=False, mmr=False, rerank=False, parents=False)
        else:
            ans = answer(a.agent, a.question, VectorStore(store_path(a.doc)), [])
        for i, s in enumerate(ans.steps, 1):
            print(f"\n[{i}] {s.question}\n    sources: {', '.join(s.sources)}\n    {s.answer.strip()[:600]}")
        print(f"\n=== Answer ===\n{ans.text.strip()}")


if __name__ == "__main__":
    main()
