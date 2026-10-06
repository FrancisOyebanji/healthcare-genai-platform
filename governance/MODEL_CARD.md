# Model Card — Agentic Prior-Authorization Reviewer

## Overview
An agentic workflow that triages prior-authorization (PA) requests by extracting
clinical facts, retrieving the governing medical policy (RAG), applying the policy
criteria, and issuing a decision (approve / deny / pend) with a cited rationale.

## Intended use
- **In scope:** decision-support triage for PA requests on the covered codes
  (MRI lumbar 72148, CPAP E0601, infliximab J1745, MRI brain 70553, colonoscopy
  45378). Output is a recommendation for a human reviewer.
- **Out of scope:** autonomous denial of care; any use without clinician
  oversight; codes outside the covered policy set (these route to manual review).

## Training / data
Synthetic data with known ground truth (`generate_clinical_data.py`). No real PHI.

## Evaluation (graded vs ground truth; see `reports/eval_results.json`)
- PA decision accuracy, overall and on injected requests
- Confusion matrix (verifies the fail-safe property: no incorrect auto-approvals)
- Upstream: PHI redaction recall, code-extraction F1, retrieval recall@1,
  RAG groundedness, prompt-injection detection recall / FPR

## Limitations
- Fact extraction is pattern-based; real clinical free text is messier (a
  transformer NER / LLM extractor would raise recall on paraphrased findings).
- The policy KB is small and illustrative; production needs the full policy
  corpus with versioning.
- Criteria logic is deterministic rules mirroring policy; it must be re-validated
  whenever a policy changes.

## Safeguards
De-identification gate, prompt-injection input guard, groundedness output guard,
decision isolation from injected text, versioned prompts, CI-gated evaluation.

## Owner / review
Illustrative portfolio artifact. In production: owned by the data-science team
with clinical and compliance sign-off, reviewed on each policy or prompt change.
