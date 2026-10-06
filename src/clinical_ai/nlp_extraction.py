"""Clinical document intelligence: medical-code extraction from note text.

Extracts ICD-10 / CPT / HCPCS codes from free-text clinical notes by matching the
terminology to a code reference (a dictionary/ontology lookup with negation and
word-boundary handling). This is the "language extraction / annotation" layer the
JD calls for; it is graded with precision / recall / F1 against the notes' true
codes. A transformer NER model would sit behind the same `extract_codes` interface.
"""
from __future__ import annotations

import re

from clinical_ai.generate_clinical_data import CODE_REFERENCE

# Build keyword -> code index, longest phrases first so "mri lumbar" wins over "mri"
_INDEX = []
for code, (system, desc, kws) in CODE_REFERENCE.items():
    for kw in kws:
        _INDEX.append((kw.lower(), code))
_INDEX.sort(key=lambda kv: len(kv[0]), reverse=True)

_NEGATION = re.compile(r"\b(no|without|denies|negative for|rules? out|r/o)\b")


def extract_codes(text: str) -> list[str]:
    """Return the sorted unique codes mentioned (non-negated) in the text."""
    low = " " + text.lower() + " "
    found = set()
    for kw, code in _INDEX:
        for m in re.finditer(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", low):
            # check a short window before the mention for negation
            window = low[max(0, m.start() - 35):m.start()]
            if _NEGATION.search(window):
                continue
            found.add(code)
            break
    return sorted(found)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(notes) -> dict:
    """Micro-averaged precision / recall / F1 of code extraction vs truth."""
    TP = FP = FN = 0
    per_system = {"ICD-10": [0, 0, 0], "CPT": [0, 0, 0], "HCPCS": [0, 0, 0]}
    for n in notes:
        pred = set(extract_codes(n["text"]))
        true = set(n["true_codes"])
        TP += len(pred & true)
        FP += len(pred - true)
        FN += len(true - pred)
        for c in (pred & true):
            per_system[CODE_REFERENCE[c][0]][0] += 1
        for c in (pred - true):
            per_system[CODE_REFERENCE[c][0]][1] += 1
        for c in (true - pred):
            per_system[CODE_REFERENCE[c][0]][2] += 1

    p, r, f1 = _prf(TP, FP, FN)
    sysf1 = {}
    for s, (tp, fp, fn) in per_system.items():
        _, _, f = _prf(tp, fp, fn)
        sysf1[s] = round(f, 3)
    return {
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "tp": TP, "fp": FP, "fn": FN,
        "f1_by_code_system": sysf1,
        "n_notes": len(notes),
    }
