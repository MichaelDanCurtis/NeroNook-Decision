import pytest
from nero_decision.continual import ContinualPolicy, Watch, Event
from nero_decision.model import proposal, QUESTIONS

def answer(choice="change", probability=0.99):
    labels = QUESTIONS["route"]["criteria"]
    return {"choice": choice, "probabilities": {label: probability if label == choice else (1-probability)/(len(labels)-1) for label in labels}}

def test_dictation_and_untrusted_content_cannot_become_commands():
    assert proposal(answer(), mode="dictate")["route"] == "dictate"
    assert proposal(answer(), source="module")["route"] == "ignore"
    assert proposal(answer())["route"] == "change"
    assert proposal(answer())["execute"] is False

@pytest.mark.parametrize("bad", [None, {}, {"choice":[],"probabilities":answer()["probabilities"]}, {"probabilities":[]}, {"choice":"change","probabilities":{}}, answer(probability=float("nan")), answer(probability=2), {"choice":"change","probabilities":{k:"0.5" for k in QUESTIONS["route"]["criteria"]}}])
def test_invalid_output_abstains(bad):
    assert proposal(bad)["route"] == "clarify"

def test_uncertainty_abstains():
    assert proposal(answer(probability=0.50))["route"] == "clarify"

@pytest.fixture
def policy():
    p = ContinualPolicy()
    p.add(Watch("watch1", "printer", "job1", 7))
    return p

def event(revision=1, kind="completed", **kwargs):
    return Event(**dict(source="printer", subject="job1", revision=revision, kind=kind, received_at=100, expires_at=120, **kwargs))

def test_watch_lifecycle_only_notifies_once_for_same_job(policy):
    assert policy.observe("watch1", event(1,"progress"), now=105, granted_epoch=7) == "observe"
    assert policy.observe("watch1", event(2), now=105, granted_epoch=7) == "suggest_notification"
    assert policy.observe("watch1", event(2), now=105, granted_epoch=7) == "duplicate"
    assert policy.observe("watch1", event(3), now=105, granted_epoch=7) == "duplicate"
    other = Event("printer", "job2", 4, "completed", 100, 120)
    assert policy.observe("watch1", other, now=105, granted_epoch=7) == "ignore"

def test_pause_revocation_and_quiet_delivery(policy):
    policy.paused = True
    assert policy.observe("watch1", event(), now=105, granted_epoch=7) == "ignore"
    policy.paused = False
    assert policy.observe("watch1", event(), now=105, granted_epoch=7, quiet=True) == "inbox"
    assert policy.observe("watch1", event(2), now=105, granted_epoch=8) == "revoked"
    assert policy.observe("watch1", event(3), now=105, granted_epoch=7) == "ignore"

@pytest.mark.parametrize("now", [99, 120, 121, float("nan"), float("inf")])
def test_stale_or_invalid_clock_cannot_notify(policy, now):
    assert policy.observe("watch1", event(), now=now, granted_epoch=7) == "stale"

def test_bounded_watches(policy):
    with pytest.raises(ValueError): policy.add(Watch("watch1", "printer", "job1", 7))
    for n in range(63): policy.add(Watch(str(n), "module", str(n), 1))
    with pytest.raises(ValueError): policy.add(Watch("overflow", "module", "x", 1))

def test_new_issue_episode_can_notify_but_completion_cannot_repeat(policy):
    first=Event("printer","job1",1,"needs_input",100,120,"issue1")
    next_issue=Event("printer","job1",2,"needs_input",100,120,"issue2")
    assert policy.observe("watch1",first,now=105,granted_epoch=7)=="suggest_notification"
    assert policy.observe("watch1",next_issue,now=105,granted_epoch=7)=="suggest_notification"
    assert policy.observe("watch1",event(3),now=105,granted_epoch=7)=="suggest_notification"
    other=Event("printer","job1",4,"completed",100,120,"new-message")
    assert policy.observe("watch1",other,now=105,granted_epoch=7)=="duplicate"
