# Laya on Nero requests — first local measurements

Measured September 24, 2026 on Apple M4 Max with 128 GiB RAM. Synthetic development fixtures only; no microphone, account, device, screen or cloud-model access. Models were explicitly downloaded once; inference ran with Hugging Face offline mode and local model paths.

| Model | Path | Correct / 36 | P50 | P95 | Matches eager choices |
|---|---|---:|---:|---:|---|
| english | eager | 16 | 15.37 ms | 16.42 ms | True |
| english | compiled + cached question prefixes | 16 | 14.64 ms | 17.47 ms | True |
| multilingual | eager | 9 | 7.95 ms | 8.18 ms | True |
| multilingual | compiled + cached question prefixes | 9 | 7.52 ms | 9.74 ms | True |

These numbers describe one small routing design, not general Laya accuracy. They exclude model loading; each request has three timed repetitions after warmup. The raw report includes all fixture texts, expected labels, predictions, timings and checkpoint pins. Peak MLX allocation is process-cumulative and must not be interpreted as each model’s isolated memory footprint.

Compilation preserved all 36 selected labels for each checkpoint. Median improvements were small (about 5%); P95 was worse in both runs. Three alternate question formulations on the same development set did not produce acceptable routing accuracy. They are ablations, not independent held-out tests.

Observed failure: a quoted log message containing a cancellation command was assigned to cancellation with high probability. The proposal wrapper correctly ignores non-user origins, but text-only classification is not sufficient to establish origin or permission.

Conclusion: retain Laya as an advisory local component while developing narrower task schemas and a domain corpus. Do not promote this seven-way router to autonomous execution. The next quality benchmark needs action-specific candidates and a separately held-out corpus.

The no-truncation guard rejected an overlong request before inference. Thirty-two tests passed: nineteen for decision/watch policy and thirteen for the context boundary. They cover dictation isolation, invalid/uncertain predictions, exact job scope, stale events, pause, quiet delivery, distinct failure episodes, completion deduplication, local/cloud separation, excluded/protected sources, late results after Pause, bounds and grant revocation. `watch-replay.json` shows the synthetic lifecycle; it is not live event integration. The watch policy and context buffer are currently in-memory; native capture, durable goal storage, an SDK adapter and product UI are still required.

Reproduce:

```sh
uv sync --frozen
HF_HOME="$PWD/build/huggingface" PYTHONPATH=. uv run python scripts/prepare_models.py
HF_HOME="$PWD/build/huggingface" HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/benchmark.py
HF_HOME="$PWD/build/huggingface" HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/compare_questions.py
HF_HOME="$PWD/build/huggingface" HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/binary_questions.py
uv run pytest -q
```
