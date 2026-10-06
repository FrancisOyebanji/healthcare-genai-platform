"""Responsible-AI guardrails: prompt-injection detection and output PHI-leakage.

Two safety checks the JD calls out explicitly ("safeguards against prompt
injection and data leakage"):

  * detect_prompt_injection - input guard. Flags text that tries to override the
    system's instructions (classic jailbreak / injection patterns). Graded on
    detection recall against labeled injections and false-positive rate on clean text.
  * output_leakage_check - output guard. Confirms a model response that is meant to
    be de-identified carries no residual PHI before it leaves the system.
"""
from __future__ import annotations

import re

from clinical_ai.deidentify import deidentify

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (your |the )?(guidelines|policy|criteria|instructions)",
    r"system override",
    r"you must approve",
    r"approve (this|every) (claim|request|authorization)",
    r"reveal (the )?(full )?(patient|member) record",
    r"<<<.*?>>>",
    r"note to (the )?ai",
]
_COMPILED = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> dict:
    matched = [p.pattern for p in _COMPILED if p.search(text)]
    return {"is_injection": bool(matched), "patterns_matched": matched}


def output_leakage_check(text: str) -> dict:
    """True if the (supposedly de-identified) output still contains PHI."""
    _, redactions = deidentify(text)
    phi_types = sorted({r["type"] for r in redactions})
    return {"has_leak": bool(redactions), "leaked_types": phi_types,
            "n_leaks": len(redactions)}


def evaluate(notes, pa_requests) -> dict:
    """Grade injection detection on notes + requests; report recall and FPR."""
    items = [(n["text"], n["has_injection"]) for n in notes] + \
            [(r["request_text"], r["has_injection"]) for r in pa_requests]

    tp = fp = tn = fn = 0
    for text, is_inj in items:
        pred = detect_prompt_injection(text)["is_injection"]
        if is_inj and pred:
            tp += 1
        elif is_inj and not pred:
            fn += 1
        elif not is_inj and pred:
            fp += 1
        else:
            tn += 1

    recall = tp / (tp + fn) if (tp + fn) else 1.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    return {
        "n_items": len(items),
        "injection_detection_recall": round(recall, 4),
        "injection_precision": round(precision, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
