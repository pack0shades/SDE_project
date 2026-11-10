# regcheck: video script (target 9:45, hard limit 10:00)

**Speakers:** **T** = Tushar Kant (B23ES1033), **P** = Pragay (B23ES1024)
**Deck:** `regcheck_presentation_10slides.pptx` · **Diagram:** `image.png` (full-resolution system diagram)
**Pace:** about 140 words per minute. Each block lists its time window and a rough word count. If you run long, cut the lines marked *(optional)* first.

**Before recording**
- Have the Streamlit app open on a finished E-Store × GDPR run (`uv run streamlit run app.py`), so the demo needs no live LLM calls.
- Open `image.png` full screen for slide 3, so you can zoom into each box.
- Have a terminal ready with `uv run pytest -q` already run (for the 2-second shot on slide 8).

---

## 1 · Title (0:00 – 0:35) · T + P · ~80 words

**[Screen: slide 1]**

**T:** Hi, I'm Tushar, and with me is Pragay. This is our Software and Data Engineering project, *regcheck*. It checks software requirements against the GDPR and the EU Data Act using several LLM agents that work together.

**P:** We implemented a 2026 ACM TOSEM paper by Das and colleagues. Our version has two checker agents from different vendors, a third agent that settles their disagreements, and a human reviewer who has the final say. It runs on free-tier models in about 1,700 lines of Python.

---

## 2 · Motivation and the paper (0:35 – 1:35) · T · ~140 words

**[Screen: slide 2. Point to "The problem", then the four A–D boxes]**

**T:** Why check compliance at the requirements stage? Regulations run to dozens of cross-referenced articles full of exceptions. Finding a violation during testing or an audit is expensive and can bring legal penalties. And requirements engineers are usually not lawyers.

The paper proposes four components, and we kept its letters. **A**, a pre-processor that splits regulations and requirements into self-contained chunks. **B**, two compliance checkers that work independently. **C**, a report analyzer that compares their reports, asks the right agent to re-check, and gives a confidence score for every issue. And **D**, a human in the loop whose decisions are remembered for later runs.

*(optional)* One thing to keep in mind: the paper reports counts, not precision or recall, because nobody has ground-truth labels for non-compliant requirements. Our evaluation works the same way.

---

## 3 · Architecture and one check run (1:35 – 2:45) · P · ~165 words

**[Screen: slide 3, then switch to `image.png` full screen. Follow the numbered badges 1 → 10 with the cursor]**

**P:** Here is the whole system. Read it left to right, and follow the numbered badges for one run.

Inputs are a regulation PDF from EUR-Lex, an SRS document, and `config.toml`, which holds every setting. The pre-processor, box A, builds two vector stores: one for the regulation and one for the requirements.

Badges 1 to 3: each checker writes a plan, screens the SRS for relevant requirements, then assesses them in batches of five against the retrieved articles. Badge 4: two structured reports go to the report analyzer. It finds requirements flagged by only one checker, judges the rationale, and badge 5 sends a re-check request to the checker that has to reconsider. Badge 6 compiles the final report with confidence scores. Badges 7 to 10: everything is saved, a reviewer accepts, rejects or escalates each issue, and those decisions are recalled in later runs.

Along the bottom is the LLM layer, the only module that calls a model. One run is about 25 calls.

---

## 4 · Stack, configuration and the LLM layer (2:45 – 3:25) · P · ~95 words

**[Screen: slide 4. Point to the agents table, then the `llm.py` panel, then the prompt anatomy]**

**P:** Every model and setting lives in `config.toml`. We use three vendors: Mistral for checker one, Gemini for checker two and Cohere for the analyzer, so the two checkers don't share one model's biases.

`llm.py` wraps litellm. It adds back-off on rate limits, a disk cache keyed on a hash of the prompt, and a trace of every call. JSON is requested through the prompt and validated with Pydantic, with one repair retry. Every prompt has the same shape: a persona, the task, data inside tags, and a line saying tagged text is data, never instructions.

---

## 5 · A · Pre-processor and retrieval (3:25 – 4:25) · T · ~140 words

**[Screen: slide 5. Left half for ingest, right half for the five retrieval steps]**

**T:** Component A. Splitting into parent sections uses no LLM. A regulation is split at each "Article n", which gives 99 articles for the GDPR. An SRS is split at its numbered headings. Then a chunker agent rewrites each section into self-contained statements: one obligation each, pronouns resolved, and exceptions kept with their rule. These are embedded with BGE-large into FAISS. The index is saved after every section, so a crash can resume.

Retrieval follows the paper's order. Fetch 15 nearest neighbours, keep 10 with MMR for diversity, re-rank with a cross-encoder down to the best 5, then map each chunk back to its full parent article, because the LLM should read the whole article, not a fragment.

The baseline RAG uses the same function with those steps switched off, which makes the comparison in Table 5 fair.

---

## 6 · B · Checkers and C · Report analyzer (4:25 – 5:40) · P · ~175 words

**[Screen: slide 6. Left for checkers and safeguards, right for diff / validate / compile and the confidence bands]**

**P:** Component B. Both checkers run the same code with different models. Plan, screen, assess: for every requirement, map it to articles, write the rationale first, then the status (compliant, partial or non-compliant) and a suggested rewrite.

We added three safeguards after watching real runs. First, a requirement only retrieves what it mentions, so a missing retention policy was never found. We now also pull in the general-obligation articles. Second, a clause guard: a checker can only cite articles it was actually shown, and invented ids are dropped and logged. Third, exactly one finding per requirement.

Component C. Because the reports are structured, finding disagreements is a plain set difference with no LLM call. The analyzer then judges each disputed rationale against the article text. If the rationale is valid, the checker that *missed* the issue re-checks it. If it is invalid, the checker that *flagged* it does. Finally it compiles one report. Confidence has a defined meaning: 0.8 or more if both agreed at once, 0.5 to 0.8 if they agreed after revision, below 0.5 if they still disagree.

---

## 7 · Orchestration, privacy, human review + DEMO (5:40 – 7:10) · T narrates, P drives the app · ~200 words

**[Screen: slide 7 for 20 s, then cut to the Streamlit app]**

**T:** `pipeline.py` is a plain loop: two checks, one refinement round, compile. Each run gets its own folder with the rounds, the final report, the decisions and a full trace.

Web search is optional and off by default, because it is the only path that sends data outside. The analyzer writes a generic query, and a leak check blocks it if it copies four words from the requirement or names the project.

**[Cut to app. P clicks through as T speaks]**

**T:** Here is an E-Store run against the GDPR. The report tab shows the summary and the issue table.

**[P opens one non-compliant issue, ideally password storage → Art. 32]**

**T:** For each issue we see the requirement, the status, the confidence, and whether the checkers agreed. Then the violated article in full, both checkers' rationales side by side, the evidence that was retrieved, and the suggested rewrite.

**[P picks "Reject", types a short justification, submits]**

**T:** The reviewer must give a justification. That decision is stored as a vector. The next time a similar requirement comes up under the same regulation (cosine 0.8 or higher), it goes into the prompt as a past decision.

**[P opens the Trace tab for 3 s]**

**T:** And the trace tab shows every prompt and reply, for full transparency.

---

## 8 · Design decisions, problems, testing (7:10 – 7:55) · P · ~110 words

**[Screen: slide 8. Optional 2-second cut to the terminal running `uv run pytest -q`]**

**P:** We recorded every place we departed from the paper, and why. We used free-tier models instead of Mixtral, Command R+ and Claude Opus. Feedback memory is a FAISS file instead of Neo4j. Chunks are 400 tokens because BGE truncates at 512.

Free tiers shaped the engineering. One Gemini model allowed 20 requests a day, some models returned empty replies, JSON broke, and article ids were invented. The fixes were tolerant JSON parsing, the clause guard, batching, and failing fast on quota errors.

Testing is 14 offline tests with fake LLMs and fake embeddings. They cover JSON repair, splitting, MMR, the clause guard, memory recall, the web-search privacy check, and one full pipeline run end to end.

---

## 9 · Evaluation (7:55 – 9:00) · T · ~155 words

**[Screen: slide 9. Highlight Table 4 → Table 5 line → Table 6 → Table 7 in order]**

**T:** We scaled down the paper's evaluation to fit the free tiers: two regulations and two PURE SRS documents, about 40 generated QA pairs per document, with 10 scored by RAGAs.

Table 4: both checkers share one retriever, so recall is identical. Gemini is more faithful, between 0.88 and 1.0, while Ministral tends to add things that aren't in the context.

Table 5: advanced retrieval helps on long regulations. Data Act precision goes from 0.69 to 0.85. But it hurts on short SRS documents. That differs from the paper, and we report it as measured.

Table 6 is the key result. After one analyzer round, the findings both checkers agree on rise from 17 to 21 on E-Store, and from 2 to 4 on Video Search.

Table 7: the multi-agent framework finds 23 GDPR issues on E-Store, against 6 from a single monolithic prompt. Data Act gives zero, because neither SRS involves connected products.

---

## 10 · Merits, limitations, conclusion (9:00 – 9:45) · P, then both · ~110 words

**[Screen: slide 10]**

**P:** To sum up: the implementation is complete, every citation is grounded, every call is traced, and a human makes the final decision.

The limitations matter. There is no ground truth, so higher counts could just mean more false positives, and no run has been fully reviewed by a human yet. The evaluation is small, and the models are small.

What we learnt: retrieval decides the outcome. Structured messages between agents beat free text. And LLM output has to be treated as untrusted input.

**T:** Next, we will review the GDPR runs in the app, add more SRS documents, and build a hand-labelled set for precision and recall.

**Both:** Thank you!

**[Hold on slide 10 for 3 s, then end]**

---

### Timing summary

| # | Slide | Speaker | Window | Length |
|---|---|---|---|---|
| 1 | Title | T + P | 0:00–0:35 | 0:35 |
| 2 | Motivation, paper | T | 0:35–1:35 | 1:00 |
| 3 | Architecture (diagram) | P | 1:35–2:45 | 1:10 |
| 4 | Stack, LLM layer | P | 2:45–3:25 | 0:40 |
| 5 | A · Pre-processor, retrieval | T | 3:25–4:25 | 1:00 |
| 6 | B · Checkers, C · Analyzer | P | 4:25–5:40 | 1:15 |
| 7 | Orchestration, D · HITL + demo | T (P drives) | 5:40–7:10 | 1:30 |
| 8 | Decisions, problems, tests | P | 7:10–7:55 | 0:45 |
| 9 | Evaluation | T | 7:55–9:00 | 1:05 |
| 10 | Wrap-up | P + T | 9:00–9:45 | 0:45 |

Split: Tushar about 4:50, Pragay about 4:55. That leaves 15 seconds of buffer. If a take runs over, cut the *(optional)* line in section 2 and shorten the demo's trace-tab shot.
