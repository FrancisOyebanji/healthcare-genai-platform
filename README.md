# Healthcare GenAI Platform — Clinical Document Intelligence, RAG & Agentic Prior-Auth Review

**An end-to-end healthcare GenAI/NLP platform with responsible-AI governance: HIPAA PHI de-identification, clinical document intelligence (ICD-10/CPT/HCPCS code extraction), RAG over a medical-policy knowledge base, an agentic prior-authorization reviewer, prompt-injection and hallucination guardrails, and an LLMOps evaluation/observability layer — every capability graded against known ground truth. Python + SQL.**

> **In one breath:** Built a healthcare GenAI platform that de-identifies clinical notes (HIPAA Safe-Harbor PHI redaction, 100% recall on the benchmark), extracts medical codes from free text (ICD-10/CPT/HCPCS, F1 0.95 with negation handling so ruled-out conditions aren't coded), answers medical-policy questions with grounded, cited RAG (retrieval recall@1 1.0; a groundedness guard blocks a fabricated answer at 0.36 vs 0.80 threshold), and runs an **agentic prior-authorization reviewer** (guardrail → extract facts → retrieve policy → apply criteria → decide) that reaches **96.7% decision accuracy with zero incorrect auto-approvals** (every error fails safe to human review) and is robust to prompt injection (100% injection-detection recall; injected instructions can't change a decision) — all wrapped in LLMOps (versioned prompts, CI, graded eval, batch observability) and an AI-governance/model-risk layer.

> ⚕️ All data is **synthetic** with known ground truth (`src/clinical_ai/generate_clinical_data.py`). No real PHI, member, or clinical data is used or represented.

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m clinical_ai.run_platform      # end-to-end graded evaluation
PYTHONPATH=src python -m clinical_ai.build_dashboard   # evaluation dashboard
PYTHONPATH=src python -m pytest tests/ -q              # 9 tests, graded vs ground truth
```

## What it does, graded against ground truth

| Capability | Module | Result (synthetic benchmark) |
|---|---|---|
| **PHI de-identification** (HIPAA) | [deidentify.py](src/clinical_ai/deidentify.py) | redaction **recall 1.00** (885/885 identifier spans) |
| **Clinical code extraction** (NLP document intelligence) | [nlp_extraction.py](src/clinical_ai/nlp_extraction.py) | **F1 0.95**, precision 1.00 (negation handled → no false codes) |
| **RAG over medical policy** | [rag.py](src/clinical_ai/rag.py) | retrieval **recall@1 1.00**, grounded + cited answers |
| **Hallucination guard** | [rag.py](src/clinical_ai/rag.py) · [llmops_eval.py](src/clinical_ai/llmops_eval.py) | fabricated answer **blocked** (0.36 < 0.80), grounded answer passes |
| **Agentic prior-auth reviewer** | [agentic_review.py](src/clinical_ai/agentic_review.py) | **decision accuracy 0.97**, **0 incorrect approvals** (fail-safe) |
| **Prompt-injection guardrail** | [guardrails.py](src/clinical_ai/guardrails.py) | **injection recall 1.00**, precision 1.00, FPR 0.00 |
| **LLMOps** | [prompt_registry.py](src/clinical_ai/prompt_registry.py) · [llmops_eval.py](src/clinical_ai/llmops_eval.py) | versioned prompts (semver + content hash), CI, batch observability |

## The agentic prior-authorization reviewer (centerpiece)

A tool-using agent that reviews a PA request as an explicit, auditable state graph —
the pattern a LangGraph / tool-calling LLM agent would run:

```
guardrail  →  extract_facts  →  retrieve_policy (RAG)  →  apply_criteria  →  decide
```

Each node is a deterministic, testable tool, and every decision carries the extracted
facts, a one-line rationale, and the **cited policy_id**. Two properties the evaluation
verifies and that matter for a payer setting:

- **Fail-safe.** The confusion matrix shows **no incorrect auto-approvals** — every
  error routes a request to stricter review (pend/deny), never to wrongly approved care.
- **Injection-robust.** Decisions are made solely from extracted clinical facts +
  retrieved policy, so a prompt-injection string in the request text is flagged by the
  guardrail and **cannot change the outcome** (accuracy on injected requests tracks
  overall accuracy).

A hosted LLM (e.g., Azure OpenAI) drops in at `extract_facts` (structured extraction)
and as the rationale writer behind the same node interfaces; the criteria logic stays a
deterministic, versioned rules engine — which is what medical-policy decisioning needs.

## Responsible AI & governance

Governance is a first-class part of the platform, not an afterthought — see
[governance/AI_GOVERNANCE.md](governance/AI_GOVERNANCE.md) (NIST AI RMF / ISO 42001
control mapping, HIPAA privacy-by-design) and
[governance/MODEL_CARD.md](governance/MODEL_CARD.md):

- **Privacy by design** — a de-identification gate runs before any model sees a note;
  an output-leakage guard re-scans responses for residual PHI.
- **Safety** — prompt-injection input guard, groundedness output guard (anti-hallucination),
  decision isolation from injected text.
- **Model risk management** — every component scored against ground truth each run, so
  regressions are caught in CI before deployment.

## Repo structure

```
src/clinical_ai/
├── generate_clinical_data.py  # synthetic notes/policies/PA requests with KNOWN truth
├── deidentify.py              # HIPAA Safe-Harbor PHI redaction (+ recall eval)
├── nlp_extraction.py          # ICD-10/CPT/HCPCS extraction w/ negation (+ P/R/F1 eval)
├── rag.py                     # TF-IDF retriever + grounded cited answers + groundedness
├── agentic_review.py          # agentic PA reviewer (state graph) + decision eval
├── guardrails.py              # prompt-injection + output-leakage detection (+ eval)
├── prompt_registry.py         # versioned prompts (semver + content hash)
├── llmops_eval.py             # graded eval harness + hallucination demo + observability
├── run_platform.py            # orchestration -> reports/eval_results.json
└── build_dashboard.py         # evaluation dashboard
sql/feature_pipeline.sql       # Snowflake/Databricks feature pipeline (claims/UM, ICD/CPT/DRG)
governance/                    # AI_GOVERNANCE.md, MODEL_CARD.md
ci/github-actions-ci.yml       # CI: graded eval + tests on push
tests/test_platform.py         # 9 tests incl. fail-safe + injection-robustness
```

## Design choices & honest limitations

- **Graded, not just built.** Because the synthetic data has known truth, de-id recall,
  extraction F1, retrieval recall, decision accuracy, and injection recall are all
  *measured*, and the test suite fails if any regresses.
- **Runs offline.** The retriever (TF-IDF) and extractive, grounded answerer need no
  network or GPU, so the whole eval is reproducible in CI in seconds. Hosted-LLM
  adapters slot in behind the same interfaces for production quality/latency/cost tradeoffs.
- **Pattern-based extraction has a real recall gap.** ~9% of codes referenced by unlisted
  synonyms are missed — a transformer clinical-NER/LLM extractor would close it; the gap
  is reported, not hidden. Real clinical free text is messier than this benchmark.
- **The policy KB is small and illustrative.** Production needs the full, versioned policy
  corpus; the criteria rules must be re-validated on every policy change.

---

*Francis Oluwatobi · oluwatobi.ou@gmail.com*
