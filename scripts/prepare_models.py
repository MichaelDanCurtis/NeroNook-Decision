"""Explicit public model download into this project's ignored cache."""
from huggingface_hub import snapshot_download
from nero_decision.model import MODELS, checkpoint

for name, spec in MODELS.items():
    snapshot_download(spec["repo"], revision=spec["revision"], token=False)
    checkpoint(name)
    print(name, "downloaded and checksum verified", flush=True)
