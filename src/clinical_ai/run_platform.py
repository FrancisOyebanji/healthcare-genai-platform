"""End-to-end run of the healthcare GenAI platform + graded evaluation report."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_ai import llmops_eval


def main() -> dict:
    print("=== Running healthcare GenAI platform evaluation ===")
    r = llmops_eval.run_eval()

    d = r["deidentification"]
    print(f"[de-id]      PHI redaction recall {d['redaction_recall']} "
          f"({d['phi_spans_redacted']}/{d['phi_spans_total']} spans)")
    n = r["nlp_extraction"]
    print(f"[nlp]        code extraction F1 {n['f1']} (P {n['precision']} / R {n['recall']})")
    g = r["rag"]
    print(f"[rag]        retrieval recall@1 {g['retrieval_recall_at_1']} | "
          f"groundedness {g['mean_groundedness']}")
    a = r["agentic_review"]
    print(f"[agent]      PA decision accuracy {a['decision_accuracy']} | "
          f"on injected {a['decision_accuracy_on_injected']}")
    gu = r["guardrails"]
    print(f"[guardrails] injection recall {gu['injection_detection_recall']} | "
          f"precision {gu['injection_precision']} | FPR {gu['false_positive_rate']}")
    h = r["hallucination_guard"]
    print(f"[safety]     hallucination guard: grounded={h['grounded_answer_score']} (pass), "
          f"fabricated={h['fabricated_answer_score']} (blocked={h['fabricated_answer_blocked']})")
    o = r["observability"]
    print(f"[ops]        {o['docs_per_second']} docs/s | prompt versions "
          f"{ {k: v['version'] for k, v in r['prompt_registry'].items()} }")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/eval_results.json").write_text(json.dumps(r, indent=2, default=str))
    print("\nResults -> reports/eval_results.json")
    return r


if __name__ == "__main__":
    main()
