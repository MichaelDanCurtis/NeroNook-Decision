"""Prompt experiment: compact meaningful choices versus an overloaded routing question.

Same development fixtures; any improvement here needs a separate held-out corpus.
"""
import json
import statistics
import time
from pathlib import Path
from benchmark import CASES
from nero_decision.model import LocalDecisionModel

VARIANTS = {
    "compact": {"route": {"type":"choice", "instructions":"What does the speaker want?", "criteria": {
        "read":"Check current status or settings.",
        "change":"Perform a single device or app command.",
        "plan":"Plan, automate, monitor, or perform multiple steps.",
        "chat":"Have a conversation or get an explanation or written text.",
        "clarify":"Clarification: the request lacks information.",
        "ignore":"No action: this is a quotation, narration or negation.",
        "cancel":"Cancel the assistant's own work, not a device.",
    }}},
    "natural_labels": {"route": {"type":"choice", "instructions":"Choose the best description of the speaker's intent.", "criteria": [
        "checking live status", "one device command", "planning or ongoing monitoring", "conversation or explanation", "ambiguous request", "not a request", "cancelling the assistant's work"
    ]}},
}
LABELS = dict(zip(VARIANTS["natural_labels"]["route"]["criteria"], ["read","change","plan","chat","clarify","ignore","cancel"]))

def main():
    report=[]
    for name in ["english","multilingual"]:
        model=LocalDecisionModel(name, optimized=False)
        for variant,questions in VARIANTS.items():
            rows=[]
            for text,expected in CASES:
                start=time.perf_counter();answer=model.predict(text,questions)["answers"]["route"];ms=(time.perf_counter()-start)*1000
                predicted=LABELS.get(answer["choice"],answer["choice"])
                rows.append(dict(text=text,expected=expected,predicted=predicted,probability=max(answer["probabilities"].values()),ms=ms))
            run=dict(model=name,variant=variant,correct=sum(r["expected"]==r["predicted"] for r in rows),total=len(rows),p50_ms=statistics.median(r["ms"] for r in rows),rows=rows)
            report.append(run);print(json.dumps({k:v for k,v in run.items() if k!="rows"}),flush=True)
    Path("reports/question-comparison.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
