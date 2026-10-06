"""Agentic AI prior-authorization reviewer.

A tool-using agent that reviews a prior-auth request end to end as an explicit
state graph (the pattern LangGraph / a tool-calling LLM agent would run):

    guardrail  ->  extract_facts  ->  retrieve_policy (RAG)  ->  apply_criteria  ->  decide

Each node is a deterministic "tool" so the whole trajectory is auditable. Every
decision carries the extracted facts, the cited policy_id, and a rationale, and
the guardrail step means an injected instruction in the request text cannot change
the clinical decision. Decisions are graded against the policy-correct ground truth.

A hosted LLM would slot in at extract_facts (structured extraction) and as the
rationale writer, behind the same node interfaces; the criteria logic stays as a
deterministic, testable rules engine, which is what a medical-policy system needs.
"""
from __future__ import annotations

import re

from clinical_ai.guardrails import detect_prompt_injection
from clinical_ai.rag import PolicyRAG


# ------------------------- tool: structured fact extraction -------------------
def extract_facts(text: str, code: str) -> dict:
    t = text.lower()
    f = {}
    if code == "72148":
        m = re.search(r"(\d+)\s*weeks", t)
        f["conservative_weeks"] = int(m.group(1)) if m else 0
        # key on specific findings, not the literal phrase "red-flag" (which also
        # appears in negations like "no red-flag findings reported")
        f["red_flag"] = bool(re.search(
            r"neurologic deficit|cauda equina|suspected malignancy|suspected infection", t))
    elif code == "E0601":
        m = re.search(r"(?:apnea-hypopnea index of|ahi(?: of)?)\s*([\d.]+)", t)
        f["ahi"] = float(m.group(1)) if m else None
        f["symptoms"] = bool(re.search(
            r"daytime sleepiness|somnolence|hypertension|impaired cognition", t))
    elif code == "J1745":
        f["failed_conventional"] = bool(re.search(
            r"failed\b.*\b(methotrexate|conventional)|intolerant", t))
    elif code == "70553":
        f["red_flag"] = bool(re.search(
            r"focal neurologic deficit|new[- ]onset seizure|seizure|thunderclap|papilledema|suspected mass", t))
    elif code == "45378":
        m = re.search(r"(\d+)[- ]year[- ]old", t)
        f["age"] = int(m.group(1)) if m else None
        f["symptomatic"] = bool(re.search(
            r"bleeding|anemia|positive stool|melena", t))
    return f


# ------------------------- tool: policy criteria engine -----------------------
def apply_criteria(code: str, f: dict) -> tuple[str, str]:
    """Return (decision, rationale) by applying the governing policy's criteria."""
    if code == "72148":
        if f.get("conservative_weeks", 0) >= 6:
            return "approve", f"{f['conservative_weeks']} weeks of conservative therapy meets the 6-week criterion."
        if f.get("red_flag"):
            return "approve", "Red-flag findings documented; imaging indicated."
        return "deny", "Fewer than 6 weeks of conservative therapy and no red flags."
    if code == "E0601":
        ahi = f.get("ahi")
        if ahi is None:
            return "pend", "AHI not found in documentation; manual review required."
        if ahi >= 15:
            return "approve", f"AHI {ahi} >= 15 meets criteria."
        if 5 <= ahi <= 14.9 and f.get("symptoms"):
            return "approve", f"AHI {ahi} in 5-14.9 range with documented symptoms meets criteria."
        return "deny", f"AHI {ahi} does not meet CPAP criteria."
    if code == "J1745":
        if f.get("failed_conventional"):
            return "approve", "Documented failure of conventional therapy meets criteria."
        return "deny", "No documented failure of conventional therapy."
    if code == "70553":
        if f.get("red_flag"):
            return "approve", "Focal neurologic/red-flag features documented."
        return "pend", "Routine headache without red flags; pend for review."
    if code == "45378":
        age = f.get("age")
        if (age is not None and age >= 45) or f.get("symptomatic"):
            why = "screening age >= 45" if (age and age >= 45) else "symptomatic findings"
            return "approve", f"Meets colonoscopy criteria ({why})."
        return "deny", "Under screening age and asymptomatic."
    return "pend", "No governing policy; route to manual review."


class PriorAuthAgent:
    """Deterministic agentic workflow over the policy RAG."""

    def __init__(self, policies):
        self.rag = PolicyRAG(policies)
        self.code_to_policy = {p["code"]: p["policy_id"] for p in policies}

    def review(self, request: dict) -> dict:
        trajectory = []
        code = request["requested_code"]
        text = request["request_text"]

        # node 1: guardrail
        inj = detect_prompt_injection(text)
        trajectory.append(("guardrail", "injection_detected" if inj["is_injection"] else "clean"))

        # node 2: extract facts (from clinical content only)
        facts = extract_facts(text, code)
        trajectory.append(("extract_facts", facts))

        # node 3: retrieve governing policy (RAG)
        desc_query = f"{code} prior authorization criteria"
        hits = self.rag.retrieve(desc_query, k=3)
        cited = hits[0]["policy_id"] if hits else None
        trajectory.append(("retrieve_policy", cited))

        # node 4+5: apply criteria -> decision
        decision, rationale = apply_criteria(code, facts)
        trajectory.append(("decide", decision))

        return {
            "request_id": request.get("request_id"),
            "requested_code": code,
            "decision": decision,
            "rationale": rationale,
            "cited_policy": cited,
            "facts": facts,
            "injection_flagged": inj["is_injection"],
            "trajectory": trajectory,
        }


def evaluate(policies, pa_requests) -> dict:
    agent = PriorAuthAgent(policies)
    n = correct = 0
    inj_n = inj_correct = 0
    labels = ["approve", "deny", "pend"]
    cm = {a: {b: 0 for b in labels} for a in labels}
    for req in pa_requests:
        out = agent.review(req)
        truth = req["true_decision"]
        n += 1
        ok = out["decision"] == truth
        correct += ok
        cm[truth][out["decision"]] += 1
        if req["has_injection"]:
            inj_n += 1
            inj_correct += ok
    return {
        "n_requests": n,
        "decision_accuracy": round(correct / n, 4) if n else 0.0,
        "decision_accuracy_on_injected": round(inj_correct / inj_n, 4) if inj_n else None,
        "n_injected": inj_n,
        "confusion_matrix": cm,
    }
