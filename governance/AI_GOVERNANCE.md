# AI Governance & Responsible AI

Governance posture for the healthcare GenAI platform. It maps the platform's
controls to recognized frameworks (NIST AI RMF, ISO/IEC 42001) and to HIPAA
obligations, and names the safeguards implemented in code.

## Privacy by design (HIPAA / PHI)

- **De-identification gate.** Every note passes `clinical_ai.deidentify` before
  any downstream model or prompt sees it. Covered identifiers follow the HIPAA
  Safe Harbor list (names, MRN, dates, phone, email, SSN). Redaction recall is
  measured each run against known PHI spans (see `reports/eval_results.json`).
- **Output leakage guard.** `guardrails.output_leakage_check` re-scans model
  outputs for residual PHI before they leave the system.
- **Minimum necessary.** The agentic reviewer extracts only the structured facts
  the policy criteria require; it does not persist raw note text.

## Safety against prompt injection & data leakage

- **Input guard.** `guardrails.detect_prompt_injection` flags instruction-override
  attempts in note / request text. Detection recall and false-positive rate are
  reported each run.
- **Decision isolation.** The prior-auth agent makes decisions solely from
  extracted clinical facts and retrieved policy. Injected instructions are flagged
  and cannot change the outcome (verified: accuracy on injected requests tracks
  overall accuracy).
- **Grounding / anti-hallucination.** RAG answers are extractive and cited; the
  `groundedness_score` guard blocks answers whose content is not supported by
  retrieved policy (demonstrated: a fabricated answer scores well below threshold).

## Model risk management

- **Graded evaluation.** Every component is scored against known ground truth, so
  regressions are caught before deployment (de-id recall, extraction F1, retrieval
  recall@1, decision accuracy, injection recall).
- **Fail-safe decisioning.** The agent's errors route to stricter human review
  (pend / deny), never to an incorrect auto-approval.
- **Interpretability.** The decision engine is a transparent, auditable rules
  layer; every decision carries facts, a rationale, and a cited policy_id.

## MLOps / LLMOps controls

- **Versioned prompts.** `prompt_registry` stamps each prompt with a semantic
  version + content hash; the snapshot is written into every eval record.
- **CI.** `ci/github-actions-ci.yml` runs the graded eval + test suite on push.
- **Observability.** `llmops_eval.monitor_batches` tracks injection-attempt rate
  and post-de-id leakage per batch — the signals a monitoring dashboard would
  alert on.

## Framework mapping (summary)

| Control area | NIST AI RMF | ISO/IEC 42001 | Implemented by |
|---|---|---|---|
| Data privacy | MAP / MEASURE | A.7 data for AI | `deidentify`, output leakage guard |
| Robustness / security | MANAGE | A.8 lifecycle | `guardrails`, decision isolation |
| Validity / reliability | MEASURE | A.6 impact assessment | graded `llmops_eval` |
| Transparency | GOVERN | A.5 policies | rationale + citations, `MODEL_CARD.md` |
| Change control | GOVERN | A.9 operation | `prompt_registry`, CI |

*Illustrative governance artifact for a synthetic project; not legal or
compliance advice.*
