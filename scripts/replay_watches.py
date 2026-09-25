"""Accountless continual-agent lifecycle demo. Events are synthetic."""
import json
from nero_decision.continual import ContinualPolicy, Event, Watch

policy = ContinualPolicy()
policy.add(Watch("print-watch", "bambu", "fixture-job", 1))
trace = []
for revision, kind, quiet in [(1,"progress",False),(2,"progress",False),(3,"needs_input",True),(4,"needs_input",False),(5,"completed",False),(6,"completed",False)]:
    event=Event("bambu","fixture-job",revision,kind,100,160)
    trace.append(dict(event=kind,revision=revision,quiet=quiet,decision=policy.observe("print-watch",event,now=110,granted_epoch=1,quiet=quiet)))
policy.paused=True
trace.append(dict(event="failed",decision=policy.observe("print-watch",Event("bambu","fixture-job",7,"failed",100,160),now=111,granted_epoch=1)))
print(json.dumps({"synthetic":True,"execution":False,"trace":trace},indent=2))
