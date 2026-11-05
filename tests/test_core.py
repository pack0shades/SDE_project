"""Offline checks for the logic that can break silently. No network, no models."""

import json
import re
import zlib
from types import SimpleNamespace

import litellm
import numpy as np
import pytest
from pydantic import TypeAdapter

from regcheck import llm, memory
from regcheck.checker import _parent
from regcheck.ingest import split_sections
from regcheck.schemas import Finding, Report
from regcheck.store import _mmr


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(memory, "STORE", tmp_path / "memory")


def fake_embed(texts, query=False):  # bag of words hashed into 32 dims
    v = np.full((len(texts), 32), 1e-3, "float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            v[i, zlib.crc32(w.encode()) % 32] += 1
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _scripted(monkeypatch, replies):
    """Make llm.complete return `replies` in order and record the messages it was sent."""
    sent = []

    def fake(agent, messages, trace):
        sent.append(list(messages))
        return replies[len(sent) - 1]

    monkeypatch.setattr(llm, "complete", fake)
    return sent


def test_complete_json_repairs_fenced_malformed_reply(monkeypatch):
    bad = 'Sure, here it is:\n```json\n{"agent": "cc1", "findings": [],}\n```'  # trailing comma
    good = '```json\n{"agent": "cc1", "findings": [{"requirement_id": "r1", "clauses": ["gdpr:art32"],' \
           ' "rationale": "No encryption.", "status": "non_compliant"}]}\n```'
    sent = _scripted(monkeypatch, [bad, good])

    report = llm.complete_json("cc1", [{"role": "user", "content": "check"}], Report, [])

    assert report.findings[0].status == "non_compliant" and report.findings[0].flagged
    assert len(sent) == 2 and "invalid" in sent[1][-1]["content"]  # repair prompt carries the error


def test_complete_json_runs_validate_hook_then_gives_up(monkeypatch):
    reply = '{"agent": "cc1", "findings": []}'
    _scripted(monkeypatch, [reply, reply])

    def need_one(r):
        if len(r.findings) != 1:
            raise ValueError("expected exactly one finding")

    with pytest.raises(ValueError, match="exactly one"):
        llm.complete_json("cc1", [{"role": "user", "content": "check"}], Report, [], validate=need_one)


def test_extract_json_skips_prose_and_off_schema_json():
    text = 'I cite ["gdpr:art32"] then answer: [{"a": [1, 2]}] trailing'
    assert llm.extract_json(text, TypeAdapter(list[dict])) == [{"a": [1, 2]}]


def test_complete_retries_transient_errors_and_empty_text_then_caches(monkeypatch):
    outcomes = [litellm.ServiceUnavailableError("high demand", "gemini", "m"), "", "ok"]
    calls = []

    def fake_completion(**kw):
        calls.append(kw)
        o = outcomes[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=o))])

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    trace = []
    msgs = [{"role": "user", "content": "hi"}]

    assert llm.complete("cc1", msgs, trace) == "ok" and len(calls) == 3
    assert llm.complete("cc1", msgs, trace) == "ok" and len(calls) == 3  # served from cache
    assert [t["cached"] for t in trace] == [False, True]


REGULATION = """(10) Recital text that wraps onto a line reading only
Article 5
HAVE ADOPTED THIS REGULATION:
CHAPTER I
General provisions
Article 1
Subject-matter
1. This Regulation lays down rules.
4.5.2016 L 119/33 Official Journal of the European Union EN
2. This Regulation protects rights.
(1) OJ C 229, 31.7.2012, p. 90.
CHAPTER II
Principles
Article 2
Material scope
This Regulation applies to processing.
Article 2
"""


def test_split_sections_regulation_keeps_articles_drops_furniture():
    s = split_sections(REGULATION, "gdpr", "regulation")
    assert [x.id for x in s] == ["gdpr:art1", "gdpr:art2"]  # recital line and duplicate heading skipped
    assert s[0].ref == "Article 1 - Subject-matter"
    assert "lays down" in s[0].text and "protects rights" in s[0].text
    assert not any(bad in s[0].text for bad in ("Official Journal", "OJ C", "CHAPTER", "Principles"))


SRS = """Table of Contents
1 Introduction\t3
3.1 Functionality\t7
1 Introduction
This document describes the e-store system for online customers.
3 Specific Requirements
3.1 Functionality
3.1.1 The system shall display all products that can be configured.
2 The system must allow a user to log in to the portal at any time.
3.2 Security
3.2.1 The system shall encrypt stored passwords.
"""


def test_split_sections_srs_uses_shallow_headings_only():
    s = split_sections(SRS, "e-store", "srs")
    assert [x.ref for x in s] == ["1 Introduction", "3.1 Functionality", "3.2 Security"]  # TOC skipped
    assert "must allow a user" in s[1].text  # numbered requirement is not a heading


def test_mmr_prefers_diverse_candidate_over_near_duplicate():
    q = np.array([1.0, 0.0])
    cands = np.array([[0.95, 0.31], [0.94, 0.34], [0.8, -0.6]])  # best, near-duplicate of best, diverse
    assert _mmr(q, cands, keep=2, lam=0.5) == [0, 2]


def test_clause_ids_map_to_supplied_parent_only():
    supplied = ["gdpr:art32", "gdpr:art3"]
    assert _parent("gdpr:art32.1(a)", supplied) == "gdpr:art32"
    assert _parent("gdpr:art3(2)", supplied) == "gdpr:art3"
    assert _parent("gdpr:art33", supplied) is None  # art3 must not swallow art33


def test_pipeline_resolves_a_dispute_end_to_end(tmp_path, monkeypatch):
    from regcheck import pipeline, store
    from regcheck.schemas import Chunk, Section

    monkeypatch.setattr(store, "embed", fake_embed)
    monkeypatch.setattr(store, "rerank_scores", lambda q, ts: [float(len(set(q.split()) & set(t.split()))) for t in ts])
    monkeypatch.setattr(pipeline, "store_path", lambda doc, mode="agentic": tmp_path / doc)
    monkeypatch.setattr(pipeline, "RUNS", tmp_path / "runs")
    store.VectorStore(tmp_path / "gdpr").add(
        [Chunk(id="gdpr:art32:0", doc="gdpr", parent_id="gdpr:art32", text="Controllers shall encrypt personal data.")],
        [Section(id="gdpr:art32", doc="gdpr", ref="Article 32 - Security of processing", text="... encryption ...")])
    store.VectorStore(tmp_path / "toy").add(
        [Chunk(id="toy:s1:0", doc="toy", parent_id="toy:s1", text="User passwords will be stored in plain text."),
         Chunk(id="toy:s1:1", doc="toy", parent_id="toy:s1", text="The dashboard shall use the corporate colours.")],
        [Section(id="toy:s1", doc="toy", ref="1 Accounts", text="...")])

    def fake_complete(agent, messages, trace):
        text = messages[-1]["content"]
        trace.append({"agent": agent, "prompt": text})
        ids = re.findall(r'<requirement id="([^"]+)"', text)
        if "sub-questions" in text:
            return '["password storage", "user interface"]'
        if "Select every requirement" in text:
            return json.dumps(ids)
        if "<disputes>" in text:
            return '[{"requirement_id": "toy:s1:0", "valid": true, "message": "Plain-text passwords breach Art 32."}]'
        if "agreement=" in text:
            return json.dumps([{"requirement_id": r, "clauses": ["gdpr:art32"], "rationale": "r", "status":
                                "non_compliant", "suggestion": "Hash passwords.", "confidence": 0.6} for r in ids])
        if "<issues>" in text:
            return "One critical issue."
        bad = agent == "cc1" or "</review_note>" in text  # cc2 misses the issue until the RA explains it
        return json.dumps([{"requirement_id": r, "clauses": ["gdpr:art32.1(a)"], "rationale": "r",
                            "status": "non_compliant" if bad and r == "toy:s1:0" else "compliant"} for r in ids])

    monkeypatch.setattr(llm, "complete", fake_complete)
    out = pipeline.run("toy", "gdpr", progress=lambda s: None)

    rounds = json.loads((out / "rounds.json").read_text())
    assert len(rounds) == 2
    [d] = rounds[1]["disputes"]
    assert (d["requirement_id"], d["flagged_by"], d["missed_by"], d["valid"]) == ("toy:s1:0", "cc1", "cc2", True)
    trace = json.loads((out / "trace.json").read_text())
    assert {t["agent"] for t in trace if "</review_note>" in t["prompt"]} == {"cc2"}  # only the agent that missed it
    [issue] = json.loads((out / "final.json").read_text())["issues"]
    assert issue["requirement_id"] == "toy:s1:0" and issue["confidence"] == 0.6
    assert {a: p["status"] for a, p in issue["positions"].items()} == {"cc1": "non_compliant", "cc2": "non_compliant"}
    assert issue["positions"]["cc1"]["clauses"] == ["gdpr:art32"]  # sub-clause id mapped to its article


def test_extract_json_reports_broken_json_not_an_inner_fragment():
    text = '[{"requirement_id": "r1", "clauses": ["gdpr:art5"], "rationale": "3. **Suggestion": x", "status": "partial"}]'
    with pytest.raises(ValueError, match="invalid JSON"):
        llm.extract_json(text, TypeAdapter(list[Finding]))
    assert llm.extract_json('{"a": 1}', TypeAdapter(list[dict])) == [{"a": 1}]  # lone object, first in reply


def test_hinted_articles_expand_ranges_and_skip_unknown():
    from regcheck.checker import _hinted
    from regcheck.schemas import Chunk, Section

    sections = {f"gdpr:art{n}": Section(id=f"gdpr:art{n}", doc="gdpr", ref="", text="") for n in (5, 15, 16, 17, 32)}
    policy = SimpleNamespace(chunks=[Chunk(id="gdpr:art5:0", doc="gdpr", parent_id="gdpr:art5", text="")],
                             sections=sections)
    reg = SimpleNamespace(hints=["as required by Article 32", "Article 5(1)(e) and Articles 15 to 17, Article 99"])
    assert [s.id for s in _hinted(reg, policy)] == ["gdpr:art32", "gdpr:art5", "gdpr:art15", "gdpr:art16", "gdpr:art17"]


def test_memory_recalls_same_regulation_similar_requirement_only(monkeypatch):
    from regcheck import store
    from regcheck.schemas import Feedback

    monkeypatch.setattr(store, "embed", fake_embed)
    memory.remember(Feedback(requirement="User passwords will be stored in plain text.", regulation="GDPR",
                             clauses=["gdpr:art32"], status="non_compliant", decision="reject",
                             justification="Legacy import only; hashed on first login."))
    [fb] = memory.recall("User passwords will be stored in plain text.", "GDPR")
    assert fb.decision == "reject" and "Legacy import" in memory.as_prompt([fb])
    assert memory.recall("User passwords will be stored in plain text.", "EU Data Act") == []
    assert memory.recall("The dashboard shall use the corporate colour scheme.", "GDPR") == []
