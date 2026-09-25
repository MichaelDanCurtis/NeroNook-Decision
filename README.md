# Nero Decision — initial research build

Independent local decision adapter and continual-watch policy for Nero Agent. This is a testbed, not an installed autonomous agent. It does not record audio, inspect the screen, read accounts, issue commands, or call a cloud model.

Laya proposes typed routes. Deterministic code separately checks input origin, explicit dictation mode, token budgets, freshness, exact watch scope, grants and repeated events. A bounded transient context buffer models the owner's selected broader-awareness scope: exact sources, local/cloud eligibility, expiry and invalidation on Pause or revocation. Native capture and actual cloud transmission are not implemented here. It never treats model confidence as permission. Model code is pinned to the requested MLX repository; public checkpoint revisions and weight checksums are pinned too. The experimental threshold is not calibrated accuracy.

## Run locally

Use `uv sync --frozen`, then `HF_HOME="$PWD/build/huggingface" PYTHONPATH=. uv run python scripts/prepare_models.py` once. Run `uv run pytest`. [Measured results and reproduction commands](reports/README.md) include the weak initial routing accuracy, not just latency. Model files and the Python environment stay under ignored directories. No existing app environment is changed.

The future Agent and Voice packages may each bundle needed runtime components; neither will depend on another application's installation or this checkout. The Nook remains a generic host. See the [product architecture](../NeroNook/docs/CONTINUAL-NERO.md). This sibling documentation link is not a build or runtime dependency.
