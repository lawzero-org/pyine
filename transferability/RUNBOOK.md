# Runbook -- transferability

Procedural ops doc. For an overview of the codebase and the canonical reproduction path, see [README.md](README.md). For methodology decisions, see [cueflip/AUDIT.md](cueflip/AUDIT.md). For Runpod-specific config tables, see [deploy/README.md](deploy/README.md). For the pre-merge validation log (which `make` targets were exercised and what was observed), see [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md).

This file covers the scenarios that don't fit a single canonical command: provider swaps, common failures and their fixes, resume semantics, cost monitoring, less-obvious knobs.

## Provider swaps

All inference goes over the OpenAI Chat/Completions wire protocol. The two models can use different providers if you want. Swap via env vars; no code change required. After updating `.env`, just re-run `make sweep1` / `make sweep2` (both are resume-friendly).

| Provider | Env vars to set | Notes |
|---|---|---|
| **Runpod serverless** (default) | `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_SHORTCUT`, `RUNPOD_ENDPOINT_BASE` | Auto-populated by `make deploy`. See deploy/README.md. |
| **Local vLLM / SGLang** (single GPU host) | `CUEFLIP_INFERENCE_URL_SHORTCUT=http://localhost:8001/v1` `CUEFLIP_INFERENCE_URL_BASE=http://localhost:8002/v1` `INFERENCE_URL_SHORTCUT=http://localhost:8001/v1/completions` `INFERENCE_URL_BASE=http://localhost:8002/v1/completions` `CUEFLIP_INFERENCE_API_KEY=EMPTY` | Spin up two vLLM servers on different ports, one per model. CueFlip wants `/v1`, lm-eval wants `/v1/completions`. |
| **Together AI** | `CUEFLIP_INFERENCE_URL_SHORTCUT=https://api.together.xyz/v1` `CUEFLIP_INFERENCE_URL_BASE=https://api.together.xyz/v1` `INFERENCE_URL_SHORTCUT=https://api.together.xyz/v1/completions` `INFERENCE_URL_BASE=https://api.together.xyz/v1/completions` `CUEFLIP_INFERENCE_API_KEY=your-together-key` `INFERENCE_API_KEY=your-together-key` `SHORTCUT_MODEL_ID=...` `BASE_MODEL_ID=...` | Together AI hosts PyINE-v1? Probably not — you'll need to upload the model first or use a different provider. |
| **OpenRouter / Anyscale / Fireworks** | Same shape as Together AI; swap URL + key | Any OpenAI-compatible `/v1` endpoint works. |
| **OpenAI itself** | `CUEFLIP_INFERENCE_URL_SHORTCUT=https://api.openai.com/v1` ... `CUEFLIP_INFERENCE_API_KEY=sk-...` | Substitute a different organism since the PyINE-v1 weights aren't on OpenAI. Set `SHORTCUT_MODEL_ID` to your alternative. |

Env-var resolution order (per `scripts/_common.sh:resolve_model_args` + `cueflip/runner.py:_build_client`):
1. `CUEFLIP_INFERENCE_URL_<TAG>` / `INFERENCE_URL_<TAG>` -- explicit overrides, win if set
2. `RUNPOD_ENDPOINT_<TAG>` -- the Runpod default path

To swap a single model (keep shortcut on Runpod, move base to local vLLM), set only the `_BASE` overrides; `_SHORTCUT` falls back to Runpod.

## Common failures + fixes

### `make deploy` fails with "Template name must be unique"
Stale template from a previous deploy. The deploy script reuses existing templates by name (since 2026-05-23), so this shouldn't happen unless your template config drifted. Inspect what's there: `python deploy/teardown_endpoints.py --dry-run`. If a template exists with the wrong config, delete it manually via the Runpod console at `https://console.runpod.io/serverless/templates`.

### `make verify-endpoints` returns HTTP 404
The endpoint IDs in `.env` point at endpoints that no longer exist on Runpod (typical after a teardown). Re-run `make deploy` -- it'll create fresh endpoints and auto-patch `.env`.

### `make verify-endpoints` first request takes 2-5 minutes
Expected on a fresh deploy. First request triggers a cold-start: image pull (50 GB) + vLLM startup + model load to GPU. Subsequent requests within 5 min stay warm. See `IDLE_TIMEOUT_SECS` in `deploy/deploy_endpoints.py`.

### `make sweep2` cache-builder step hangs or errors
The cache builder calls a separate judge LLM at `CUEFLIP_JUDGE_URL` (default `http://localhost:8000/v1`) to generate op-flip wrong-numerics for GSM8K. If that endpoint isn't running, the build hangs or fails. Two fixes:
- Start the judge endpoint: a vLLM server on localhost:8000 with the model named in `CUEFLIP_JUDGE_MODEL` (default `Qwen/Qwen3-4B-Instruct-2507`).
- Skip the cache build: `SKIP_CACHE_BUILD=1 make sweep2`. The runner gracefully degrades -- it logs a warning and skips op_flip cells; primary `plus_minus_10` GSM8K cells still run unaffected.

### Sweep #1 jobs fail with `TimeoutError` mid-run
`scripts/_common.sh` sets `timeout=600` (10 min) per request. If a long-CoT response on GPQA / MMLU-Pro exceeds that, the request errors. Raise the timeout in `_common.sh:resolve_model_args` (look for the `timeout=600` line in `MODEL_ARGS`) and re-run -- sweep #1 is resume-friendly (`already_done()` skips completed cells).

### Cueflip records have `parsed_answer: null`
Either the model never committed to an answer (rare on greedy T=0) OR the regex parser didn't catch the commitment format. For MC records, `cueflip/judge.py` recovers via an LLM-as-judge pass against the stored response_first_chars (first 600 chars). For numeric records (kind="numeric"), the runner's `parse_answer_numeric` is already robust; if it returns None, the response is genuinely uninterpretable.

### `make smoke` GSM8K sycophancy call takes 3 minutes
Real finding from the smoke. The shortcut model treats sycophancy cues ("can you check it?", "can you confirm?") as deliberation requests and produces ~4500-token CoT, hitting the `MAX_GEN_TOKS=10000` budget. Other cue families return in 2-4 seconds. If running the smoke and impatient, the only way to skip is to drop sycophancy from `cueflip/cue_templates.py:CUE_FAMILIES` (which would deviate from the byte-identical-templates rule -- don't do it for real runs).

### `make lint` fails with "shellcheck: command not found"
shellcheck is OS-installed, not pip-installed. On Ubuntu/Debian: `apt install shellcheck`. On macOS: `brew install shellcheck`.

### `make test` / `make lint` fail immediately with "cannot find pyine"
You're running `make` from a directory other than `pyine/transferability/`. Either `cd` into the canonical location, or pass `PYINE_ROOT=/path/to/pyine` if pyine lives elsewhere. The canonical layout is the supported workflow; the env-var override is for CI and sibling-clones.

### `make teardown` didn't stop my local vLLM server
By design. `make teardown` deletes Runpod endpoints only (via the Runpod API, by endpoint ID from `.env`). It has no awareness of local processes. Local vLLM is **user-managed**: `LOCAL=1` only routes inference URLs to `localhost:<port>`; it doesn't start or stop the vLLM process. You started the server (`vllm serve ...`) in a separate terminal; stop it the same way (Ctrl-C in that terminal, or `pkill -f vllm` if it's backgrounded). There is no `make teardown-local` because the lifecycle isn't ours to own — and a generic `pkill vllm` could kill unrelated vLLM processes you have running for other projects.

| Path | Bring up | Tear down |
|---|---|---|
| Runpod | `make deploy` | `make teardown` |
| `LOCAL=1` (vLLM) | *user starts vllm manually* | *user kills vllm manually* |

## Resume semantics

Both sweeps are resume-friendly. Killing them mid-run loses at most the in-flight request; resume picks up where the last record was flushed.

### Sweep #1 (lm-eval-harness)
Per-cell: `already_done()` in `scripts/_common.sh` checks for an existing `results_*.json` in the cell's output directory. If present, the script skips the cell entirely. To force re-run, delete the results JSON for the specific cells you want re-done. Useful pattern: re-run a single cell with `bash scripts/run_gsm8k.sh shortcut` after deleting `outputs/raw/shortcut/gsm8k/results_*.json`.

### Sweep #2 (cueflip/runner.py)
Per-record: the runner appends to JSONL after every endpoint response and starts each cell by reading the existing JSONL into a `done` dict keyed by `(model_tag, benchmark, qid, phase, cue_family, cue_paraphrase_idx, perturbation_strategy)`. Any tuple in `done` is skipped. Killing the runner mid-flight loses at most the in-flight HTTP request. To force re-run of a specific cell, delete its JSONL.

The cache builder (`build_operation_flip_cache.py`) is also idempotent at the item level -- re-running only processes items missing from the cache.

## Cost monitoring during a live sweep

On Runpod, watch:
- `https://console.runpod.io/serverless` -- endpoints page shows in-flight + queued workers and per-hour spend
- `python deploy/teardown_endpoints.py --dry-run` -- lists what's currently deployed, so you know what's billable

Cost guardrails:
- Set `workers_max=1` per endpoint (already the default in `deploy_endpoints.py`) so spend can't accidentally scale.
- Set `idle_timeout=300` (5 min, default) so workers auto-shutdown shortly after the sweep finishes.
- After sweep completion: `make teardown` to remove endpoints (templates stay; see deploy/README.md).

Cost ballpark (Runpod A40, 2026-05 rates):
- Sweep #1 full run: $15-20 (lm-eval over 6 benchmarks × 2 models, dominated by GPQA + MMLU-Pro long-CoT)
- Sweep #2 full run (primary protocol only): $7-15 (6 benchmarks × 2 models × 9 calls per item × 150 items; HumanEval items run heavier code-generation completions)
- Sweep #2 with secondary GSM8K stratification: +$3-9 (50-item subset × 5 extra strategies × 8 families × 2 models)
- Cache build: ~$0.10 (~150 calls to a 4B judge model, mostly fast)
- Total: ~$25-40 for a full reproduction. Smoke test is < $0.10.

## Less-obvious knobs

| Knob | Where | What it does |
|---|---|---|
| `LOCAL=1` | `make <any-inference-target>`, `LOCAL=1 bash scripts/run_*.sh ...` | Point all inference at localhost defaults (shortcut:8001, base:8002, judge:8000 already-default) instead of Runpod. Useful for development against a local vLLM/SGLang setup. Explicit env vars take precedence over the local-mode defaults. |
| `--local` | `cueflip/runner.py` | Same as `LOCAL=1` but on the python script's CLI, for direct invocation outside the Makefile. |
| `DRY_RUN=1` | `make sweep2` | No-HTTP smoke of the full sweep #2. Cache builder + runner write synthetic records to separate paths (`cueflip/operation_flip_cache_dry_run.json`, `cueflip/results_dry_run/`) so they can't pollute real data. Skips `judge.py`. Analyzer runs on the dry-run dir. The whole 13.4k-call pipeline completes in ~3 seconds. Useful for validating dispatch logic, schema, and analyzer end-to-end without spending money. |
| `--dry-run` | `cueflip/runner.py`, `cueflip/build_operation_flip_cache.py` | Direct script-level dry-run (same as `DRY_RUN=1 make sweep2` but per-script). Synthetic responses parse to gold (baselines, correct=true) or suggested (cues, switched+uptake=true). Records carry `dry_run: true`. |
| `SKIP_CACHE_BUILD=1` | `make sweep2` | Skip the op-flip cache build step; runner will skip op_flip cells gracefully |
| `--gsm8k-mode {primary,secondary,both}` | `cueflip/runner.py` | Default `both`. `primary` = only plus_minus_10 on all GSM8K items; `secondary` = only 6-strategy stratification on 50-item subset; `both` = hybrid |
| `--gsm8k-secondary-subset-size N` | `cueflip/runner.py` | Default 50. Number of GSM8K items in the secondary subset (first N of seeded-shuffled 150) |
| `--op-flip-cache-path` | `cueflip/runner.py` + `build_operation_flip_cache.py` | Override path to `operation_flip_cache.json` |
| `--paraphrase-indices {first,random,all,N[,M,...]}` | `cueflip/runner.py` | Default `random` (one seeded-random per family). Use `all` for 3× the calls with full paraphrase robustness |
| `--items-cap N` | `cueflip/runner.py` | Subsample size per benchmark. Default 150. Lower for faster smoke, higher for tighter CIs |
| `--num-concurrent N` | `cueflip/runner.py` | Parallel HTTP requests per endpoint. Default 16. Endpoint MAX_CONCURRENCY=100, headroom for other clients |
| `--dry-run` | `deploy/teardown_endpoints.py` | List matching infra without deleting anything (inspection only) |
| `LIMIT=N` | `scripts/run_*.sh` | Forward `--limit N` to lm-eval-harness for smoke runs |
| `WANDB_MODE=disabled` | `make smoke`, any `run_*.sh` | Skip W&B logging (useful for smoke tests) |
| `CUEFLIP_RESULTS_ROOT` | `cueflip/*.py` | Override CueFlip output dir (default: `cueflip/results/`) |
| `TRANSF_ROOT` | `scripts/_common.sh` | Override the study root path (default: `$PYINE_ROOT/transferability`). Needed if cloning the PR standalone at a non-canonical path |

## Smoke-test interpretation

`make smoke` runs three legs:
1. CueFlip MC path: 1 item HellaSwag × shortcut × 8 cue families = 9 calls
2. CueFlip free-form path: 1 item GSM8K × shortcut × 8 cue families (primary strategy) = 9 calls
3. Sweep #1 path: lm-eval-harness GSM8K_cot × shortcut × `--limit 1` ≈ 1 call

What to check after:
- `cueflip/results/shortcut/hellaswag/runs.jsonl` should have 9 lines
- `cueflip/results/shortcut/gsm8k/runs.jsonl` should have 9 lines
- Each record should have `kind`, `perturbation_strategy`, `suggested_value`, `gold_value` fields
- For GSM8K records: `kind=="numeric"`, `perturbation_strategy=="plus_minus_10"`, `suggested_value` is a numeric string
- `outputs/raw/shortcut/gsm8k/` should have a `results_*.json` from lm-eval
- No exceptions in any of the leg logs

If the smoke passes, the runner+analyzer paths are sound and the full sweep can run. If a specific leg fails, see the corresponding "Common failures" section above.

## Static type checking (pyright)

`make check` runs ruff + tests; pyright is intentionally NOT included. Pyine's root pyright config (`[tool.pyright] include = ["pyine"]` in `pyproject.toml`) excludes this folder by design. Running `uv run pyright transferability/` from pyine's root surfaces ~1500 strict-mode errors, all of which are stubs-related noise rather than real type bugs:

- `cueflip/*.py` and `tests/*.py` use `sys.path.insert` in `tests/conftest.py` to import sibling modules (the folder isn't a Python package — no `__init__.py`). Pyright can't trace these, hence the `reportMissingImports` cluster and the `reportUnknown*` cascades it triggers downstream.
- `matplotlib.cm.tab10` triggers `reportAttributeAccessIssue` because matplotlib's dynamic colormap registry isn't covered by pyright stubs (a long-standing matplotlib quirk).
- HuggingFace `datasets` is typed too narrowly for the string-key indexing used in `cueflip/benchmarks.py` loaders (`reportIndexIssue`, `reportArgumentType`).
- JSONL records are dict-based (per the explicit "no pydantic" methodology decision documented in `cueflip/AUDIT.md`); strict pyright can't infer types through them, hence `reportUnknownVariableType` everywhere records are touched.

A follow-up PR could convert `cueflip/` and `scripts/` into proper Python packages (add `__init__.py`, switch to relative imports), which would resolve the `reportMissingImports` cluster and let pyright trace types through the dispatch boundaries. The remaining stubs gaps would still need per-call `# pyright: ignore` annotations or upstream stub fixes.
