"""Development ablation: independent narrow propositions, not normalized class probabilities."""
import json
import statistics
import time
from pathlib import Path
from benchmark import CASES
from nero_decision.model import LocalDecisionModel

QUESTIONS = {name: {"type":"noul", "instructions":statement} for name,statement in {
    "read":"The speaker asks to check the current status, progress or setting of a device, app or automation.",
    "change":"The speaker explicitly requests one concrete action on a named device or application.",
    "plan":"The speaker requests planning, an automation, multiple actions, or continued monitoring.",
    "chat":"The speaker is greeting, asking for an explanation or general knowledge, or asking to compose text.",
    "clarify":"The speaker's request is unclear or missing a required target or value.",
    "ignore":"The text only quotes, narrates or negates a command rather than asking for it to happen.",
    "cancel":"The speaker asks the assistant to cancel its own current work, not to stop a device or app.",
}.items()}

results=[]
for name in ["english","multilingual"]:
    model=LocalDecisionModel(name)
    rows=[]
    for text,expected in CASES:
        start=time.perf_counter();answers=model.predict(text,QUESTIONS)["answers"];ms=(time.perf_counter()-start)*1000
        scores={key:value["noul"] for key,value in answers.items()}
        rows.append(dict(text=text,expected=expected,predicted=max(scores,key=scores.get),scores=scores,ms=ms))
    result=dict(model=name,variant="independent propositions",correct=sum(r["expected"]==r["predicted"] for r in rows),total=len(rows),p50_ms=statistics.median(r["ms"] for r in rows),rows=rows)
    results.append(result);print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
Path("reports/binary-questions.json").write_text(json.dumps(results,indent=2)+"\n")
