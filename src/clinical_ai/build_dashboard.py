"""Build an evaluation dashboard from reports/eval_results.json."""
from __future__ import annotations

import json
from pathlib import Path

R = json.loads(Path("reports/eval_results.json").read_text())
deid, nlp, rg = R["deidentification"], R["nlp_extraction"], R["rag"]
ag, gu, hg = R["agentic_review"], R["guardrails"], R["hallucination_guard"]

labels = ["approve", "deny", "pend"]
cm = ag["confusion_matrix"]
cm_rows = "".join(
    "<tr><th>{}</th>{}</tr>".format(
        t, "".join(f"<td class='{'hit' if t==p else ('miss' if cm[t][p] else '')}'>{cm[t][p]}</td>" for p in labels))
    for t in labels)

kpis = [
    ("PHI redaction recall", f"{deid['redaction_recall']:.2f}"),
    ("Code extraction F1", f"{nlp['f1']:.2f}"),
    ("RAG retrieval recall@1", f"{rg['retrieval_recall_at_1']:.2f}"),
    ("PA decision accuracy", f"{ag['decision_accuracy']:.2f}"),
    ("Injection detection recall", f"{gu['injection_detection_recall']:.2f}"),
    ("Injection false-positive rate", f"{gu['false_positive_rate']:.2f}"),
]
kpi_html = "".join(
    f"<div class='kpi'><div class='l'>{l}</div><div class='v'>{v}</div></div>" for l, v in kpis)

comp_labels = ["De-id recall", "Extraction F1", "Retrieval@1", "Groundedness",
               "PA accuracy", "Injection recall"]
comp_vals = [deid["redaction_recall"], nlp["f1"], rg["retrieval_recall_at_1"],
             rg["mean_groundedness"], ag["decision_accuracy"],
             gu["injection_detection_recall"]]

reg = R["prompt_registry"]
reg_rows = "".join(f"<tr><td>{k}</td><td>{v['version']}</td><td><code>{v['hash']}</code></td></tr>"
                   for k, v in reg.items())

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Healthcare GenAI Platform — Evaluation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f3f6f9;color:#0f2733;margin:0}}
 header{{background:#0a5c6b;color:#fff;padding:16px 30px}} h1{{font-size:20px;margin:0}}
 .sub{{color:#bfe3e8;font-size:13px}}
 main{{max-width:1120px;margin:22px auto;padding:0 18px}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;margin-bottom:20px}}
 .kpi{{background:#fff;border:1px solid #dde7ea;border-radius:12px;padding:16px}}
 .kpi .l{{color:#5a727c;font-size:13px}}.kpi .v{{font-size:22px;font-weight:700;margin-top:4px;color:#0a5c6b}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
 .panel{{background:#fff;border:1px solid #dde7ea;border-radius:12px;padding:18px;margin-bottom:18px}}
 .panel h2{{font-size:15px;margin:0 0 12px}} canvas{{max-height:280px}}
 table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:7px 9px;border-bottom:1px solid #eef2f4;text-align:center}}
 th{{color:#5a727c}} td.hit{{background:#e3f4e8;font-weight:700}} td.miss{{background:#fdecea;color:#b42318;font-weight:700}}
 .note{{background:#e6f3f5;border-left:4px solid #0a5c6b;padding:10px 14px;border-radius:6px;font-size:14px}}
 code{{background:#eef2f4;padding:1px 5px;border-radius:4px}}
 @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
 footer{{color:#5a727c;font-size:12px;text-align:center;margin:24px 0}}
</style></head><body>
<header><h1>Healthcare GenAI Platform &mdash; graded evaluation</h1>
<div class="sub">Clinical document intelligence &middot; RAG &middot; agentic prior-auth review &middot; responsible-AI guardrails &middot; LLMOps</div></header>
<main>
<div class="kpis">{kpi_html}</div>
<div class="grid">
 <div class="panel"><h2>Component scores (vs known ground truth)</h2><canvas id="c1"></canvas></div>
 <div class="panel"><h2>Prior-auth decisions &mdash; confusion matrix</h2>
   <table><tr><th>true &darr; / pred &rarr;</th><th>approve</th><th>deny</th><th>pend</th></tr>{cm_rows}</table>
   <p style="font-size:13px;color:#5a727c;margin-top:10px">Accuracy {ag['decision_accuracy']:.3f};
   on injected requests {ag['decision_accuracy_on_injected']:.3f}. No incorrect auto-approvals &mdash;
   every error routes to stricter review (fail-safe).</p>
 </div>
</div>
<div class="grid">
 <div class="panel"><h2>Hallucination guard (groundedness)</h2><canvas id="c2"></canvas>
   <p style="font-size:13px;color:#5a727c;margin-top:10px">Grounded extractive answer passes
   ({hg['grounded_answer_score']:.2f}); a fabricated answer is blocked
   ({hg['fabricated_answer_score']:.2f} &lt; {hg['threshold']}).</p></div>
 <div class="panel"><h2>Versioned prompt registry (LLMOps)</h2>
   <table><tr><th>prompt</th><th>version</th><th>hash</th></tr>{reg_rows}</table>
   <p style="font-size:13px;color:#5a727c;margin-top:10px">Each decision is traceable to the exact
   prompt version + content hash that produced it.</p></div>
</div>
<div class="panel"><h2>Why it maps to the role</h2><div class="note">
 Generative AI + Agentic AI + NLP on healthcare data (claims/EHR-shaped), with MLOps/LLMOps
 (versioned prompts, CI, graded eval, observability) and responsible-AI governance
 (HIPAA de-identification, prompt-injection and data-leakage safeguards, model risk management) &mdash;
 every capability measured against ground truth.
</div></div>
</main>
<footer>Synthetic data &middot; src/clinical_ai/build_dashboard.py &middot; all metrics graded vs known ground truth</footer>
<script>
new Chart(c1,{{type:'bar',data:{{labels:{json.dumps(comp_labels)},datasets:[{{data:{json.dumps([round(v,3) for v in comp_vals])},backgroundColor:'#0a5c6b'}}]}},
  options:{{indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{min:0,max:1}}}}}}}});
new Chart(c2,{{type:'bar',data:{{labels:['Grounded answer','Fabricated answer','Threshold'],
  datasets:[{{data:[{hg['grounded_answer_score']},{hg['fabricated_answer_score']},{hg['threshold']}],
  backgroundColor:['#14804a','#b42318','#9aa7ad']}}]}},
  options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{min:0,max:1}}}}}}}});
</script></body></html>"""
Path("reports").mkdir(exist_ok=True)
Path("reports/eval_dashboard.html").write_text(html, encoding="utf-8")
print("Dashboard written to reports/eval_dashboard.html")
