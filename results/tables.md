# Evaluation results

### Table 4. RAG evaluation (advanced RAG) per model and document

| Document | Model | #Q | context_precision | context_recall | answer_relevancy | faithfulness |
|---|---|---|---|---|---|---|
| data_act | mistral/ministral-14b-latest | 10 | 0.845 | 1.0 | 0.852 | 0.842 |
| e-store | mistral/ministral-14b-latest | 10 | 0.75 | 0.8 | 0.955 | 0.865 |
| video-search | mistral/ministral-14b-latest | 10 | 0.65 | 0.7 | 0.78 | 0.729 |

### Table 5. Baseline RAG vs advanced RAG

| Document | Model | #Q | Size (#tokens) | context_precision baseline | context_precision advanced | context_recall baseline | context_recall advanced | answer_relevancy baseline | answer_relevancy advanced | faithfulness baseline | faithfulness advanced |
|---|---|---|---|---|---|---|---|---|---|---|---|

### Table 6. Flagged requirements per checker and iteration

| SRS | Regulation | iteration 1: cc1 (mistral/ministral-14b-latest) | iteration 1: cc2 (gemini/gemini-3.5-flash-lite) | iteration 1: common | iteration 2: cc1 (mistral/ministral-14b-latest) | iteration 2: cc2 (gemini/gemini-3.5-flash-lite) | iteration 2: common |
|---|---|---|---|---|---|---|---|
| e-store | EU Data Act | 0 | 0 | 0 | NA (0) | NA (0) | NA (0) |
| e-store | GDPR | 21 | 20 | 17 | 23 | 21 | 21 |
| toy_srs | GDPR | 4 | 4 | 4 | NA (4) | NA (4) | NA (4) |
| video-search | EU Data Act | 0 | 0 | 0 | NA (0) | NA (0) | NA (0) |
| video-search | GDPR | 5 | 3 | 2 | 5 | 4 | 4 |

### Table 7. Multi-agent framework vs monolithic baseline

| SRS | Regulation | Framework (final issues) | Monolithic (gemini/gemini-3.6-flash) | Reviewer accepted |
|---|---|---|---|---|
| e-store | EU Data Act | 0 | not run | not reviewed |
| e-store | GDPR | 23 | 6 | not reviewed |
| toy_srs | GDPR | 4 | not run | not reviewed |
| video-search | EU Data Act | 0 | not run | not reviewed |
| video-search | GDPR | 5 | not run | not reviewed |
