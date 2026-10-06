"""Retrieval-augmented generation over the medical policy knowledge base.

A TF-IDF retriever indexes the prior-authorization policy documents; a query
(e.g., a requested code and its context) retrieves the top-k policy chunks. The
"generation" step is deliberately EXTRACTIVE and grounded: the answer is composed
only from retrieved sentences, each tagged with its source policy_id, so every
claim is attributable. This is the trustworthy-RAG pattern the JD asks for
(retrieval + grounded answer + citation); a hosted LLM would replace the
extractive composer behind the same `answer()` interface, with the same
groundedness guarantee enforced by `groundedness_score`.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.])\s+", text) if s.strip()]


class PolicyRAG:
    def __init__(self, policies):
        self.policies = policies
        self.docs = [p["text"] for p in policies]
        self.ids = [p["policy_id"] for p in policies]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_mat = self.vec.fit_transform(self.docs)

    def retrieve(self, query: str, k: int = 3):
        q = self.vec.transform([query])
        sims = cosine_similarity(q, self.doc_mat)[0]
        order = sims.argsort()[::-1][:k]
        return [{"policy_id": self.ids[i], "score": round(float(sims[i]), 4),
                 "text": self.docs[i], "code": self.policies[i]["code"]}
                for i in order]

    def answer(self, query: str, k: int = 3) -> dict:
        """Grounded extractive answer: best-matching sentences + citations."""
        hits = self.retrieve(query, k)
        # rank sentences from the retrieved docs against the query
        sent_pool, src = [], []
        for h in hits:
            for s in _sentences(h["text"]):
                sent_pool.append(s); src.append(h["policy_id"])
        if not sent_pool:
            return {"answer": "", "citations": [], "retrieved": hits}
        smat = self.vec.transform(sent_pool)
        q = self.vec.transform([query])
        sims = cosine_similarity(q, smat)[0]
        top = sims.argsort()[::-1][:2]
        answer = " ".join(sent_pool[i] for i in top)
        citations = sorted({src[i] for i in top})
        return {"answer": answer, "citations": citations, "retrieved": hits}


def groundedness_score(answer: str, retrieved_texts) -> float:
    """Fraction of answer tokens supported by the retrieved context (anti-hallucination)."""
    ctx = set(re.findall(r"[a-z]{4,}", " ".join(retrieved_texts).lower()))
    toks = re.findall(r"[a-z]{4,}", answer.lower())
    if not toks:
        return 1.0
    return sum(1 for t in toks if t in ctx) / len(toks)


def evaluate(policies, pa_requests, code_reference, k: int = 3) -> dict:
    """Grade retrieval recall@k (correct policy for the requested code) and
    answer groundedness across the PA requests that have a governing policy."""
    rag = PolicyRAG(policies)
    code_to_policy = {p["code"]: p["policy_id"] for p in policies}

    hits_at_1 = hits_at_k = 0
    total = 0
    grounded = []
    for req in pa_requests:
        code = req["requested_code"]
        gold = code_to_policy.get(code)
        if gold is None:
            continue
        total += 1
        desc = code_reference[code][1]
        query = f"{desc} {code} prior authorization criteria"
        res = rag.answer(query, k=k)
        ids = [h["policy_id"] for h in res["retrieved"]]
        if ids and ids[0] == gold:
            hits_at_1 += 1
        if gold in ids:
            hits_at_k += 1
        grounded.append(groundedness_score(
            res["answer"], [h["text"] for h in res["retrieved"]]))

    return {
        "k": k,
        "n_queries": total,
        "retrieval_recall_at_1": round(hits_at_1 / total, 4) if total else 0.0,
        "retrieval_recall_at_k": round(hits_at_k / total, 4) if total else 0.0,
        "mean_groundedness": round(sum(grounded) / len(grounded), 4) if grounded else 0.0,
    }
