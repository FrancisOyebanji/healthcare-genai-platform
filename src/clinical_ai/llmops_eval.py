"""LLMOps evaluation harness + observability.

Runs one graded evaluation across every component, plus:
  * a HALLUCINATION GUARD demonstration - shows the groundedness check fires on a
    fabricated answer (groundedness below threshold -> blocked), proving the metric
    is a real safeguard rather than a number that is always 1.0.
  * batch MONITORING - groundedness and injection-attempt rate per batch, the
    signals an observability layer would track over time.
  * an experiment-tracking record stamped with the prompt-registry snapshot.
"""
from __future__ import annotations

import time

from clinical_ai import (agentic_review, deidentify, generate_clinical_data,
                         guardrails, nlp_extraction, prompt_registry, rag)

GROUNDEDNESS_THRESHOLD = 0.80


def hallucination_guard_demo(policies) -> dict:
    """Show the groundedness guard blocking a fabricated claim."""
    r = rag.PolicyRAG(policies)
    res = r.answer("CPAP device E0601 prior authorization criteria")
    ctx = [h["text"] for h in res["retrieved"]]
    grounded = rag.groundedness_score(res["answer"], ctx)
    # a fabricated answer that invents an unsupported criterion
    fabricated = ("CPAP is approved for any member who snores, and the device ships "
                  "same day with a lifetime warranty guarantee.")
    fab_score = rag.groundedness_score(fabricated, ctx)
    return {
        "grounded_answer_score": round(grounded, 3),
        "grounded_answer_blocked": grounded < GROUNDEDNESS_THRESHOLD,
        "fabricated_answer_score": round(fab_score, 3),
        "fabricated_answer_blocked": fab_score < GROUNDEDNESS_THRESHOLD,
        "threshold": GROUNDEDNESS_THRESHOLD,
    }


def monitor_batches(notes, n_batches: int = 4) -> dict:
    """Per-batch observability: injection-attempt rate + de-id leakage after redaction."""
    size = len(notes) // n_batches
    out = []
    for b in range(n_batches):
        batch = notes[b * size:(b + 1) * size]
        inj = sum(1 for n in batch
                  if guardrails.detect_prompt_injection(n["text"])["is_injection"])
        # after de-identification, confirm no residual PHI leaks downstream
        leaks = 0
        for n in batch:
            red, _ = deidentify.deidentify(n["text"])
            if guardrails.output_leakage_check(red)["has_leak"]:
                leaks += 1
        out.append({"batch": b, "n": len(batch),
                    "injection_attempt_rate": round(inj / len(batch), 3),
                    "post_deid_leak_rate": round(leaks / len(batch), 3)})
    return {"batches": out}


def run_eval() -> dict:
    data = generate_clinical_data.generate()
    t0 = time.time()

    deid = deidentify.evaluate(data["notes"])
    nlp = nlp_extraction.evaluate(data["notes"])
    rg = rag.evaluate(data["policies"], data["pa_requests"], data["code_reference"])
    agent = agentic_review.evaluate(data["policies"], data["pa_requests"])
    guard = guardrails.evaluate(data["notes"], data["pa_requests"])
    halluc = hallucination_guard_demo(data["policies"])
    mon = monitor_batches(data["notes"])

    elapsed = time.time() - t0
    throughput = round(len(data["notes"]) + len(data["pa_requests"])) / max(elapsed, 1e-6)

    return {
        "deidentification": deid,
        "nlp_extraction": nlp,
        "rag": rg,
        "agentic_review": agent,
        "guardrails": guard,
        "hallucination_guard": halluc,
        "monitoring": mon,
        "observability": {
            "eval_wall_seconds": round(elapsed, 2),
            "docs_per_second": round(throughput, 1),
        },
        "prompt_registry": prompt_registry.registry_snapshot(),
    }
