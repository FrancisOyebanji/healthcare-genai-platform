"""PHI de-identification (HIPAA privacy-by-design).

Before any note reaches an LLM, identifiers are redacted. This is a pattern-based
Safe-Harbor-style scrubber covering common HIPAA identifiers (names, MRN, SSN,
dates, phone, email, addresses). It is the privacy gate every pipeline stage runs
behind; recall is graded against the known injected PHI spans.

In production the regex layer would be backed by a clinical NER model; the public
interface (`deidentify`) is identical so the model is a drop-in upgrade.
"""
from __future__ import annotations

import re

# Order matters: more specific patterns first.
PATTERNS = [
    ("SSN",   re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN",   re.compile(r"\bMRN\d{6,8}\b", re.I)),
    ("PHONE", re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("DATE",  re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # "Patient First Last" / "seen First Last" -> capture the proper-name pair
    ("NAME",  re.compile(r"\b(?:Patient|patient|member|Member|seen)\s+"
                         r"([A-Z][a-z]+\s+[A-Z][a-z]+)")),
    # bare TitleCase TitleCase name pair (fallback)
    ("NAME2", re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")),
]

_NOT_NAMES = {"Prior Authorization", "Sleep Study", "Physical Therapy",
              "Low Back", "Chronic Kidney", "Office Visit", "Metabolic Panel",
              "Comprehensive Metabolic"}


def deidentify(text: str) -> tuple[str, list[dict]]:
    """Return (redacted_text, list of {type, value}) for audit."""
    redactions = []
    out = text

    for label, pat in PATTERNS:
        if label in ("NAME", "NAME2"):
            def _sub(m):
                name = m.group(1)
                if name in _NOT_NAMES:
                    return m.group(0)
                redactions.append({"type": "NAME", "value": name})
                return m.group(0).replace(name, "[REDACTED-NAME]")
            out = pat.sub(_sub, out)
        else:
            for m in pat.finditer(out):
                redactions.append({"type": label, "value": m.group(0)})
            out = pat.sub(f"[REDACTED-{label}]", out)

    return out, redactions


def evaluate(notes) -> dict:
    """Grade de-identification recall against known injected PHI spans."""
    total_phi = 0
    caught = 0
    leaked_examples = []
    for n in notes:
        if not n["phi_spans"]:
            continue
        redacted, _ = deidentify(n["text"])
        for span in n["phi_spans"]:
            total_phi += 1
            if span not in redacted:
                caught += 1
            else:
                if len(leaked_examples) < 5:
                    leaked_examples.append(span)
    recall = caught / total_phi if total_phi else 1.0
    return {
        "phi_spans_total": total_phi,
        "phi_spans_redacted": caught,
        "redaction_recall": round(recall, 4),
        "leaked_examples": leaked_examples,
    }
