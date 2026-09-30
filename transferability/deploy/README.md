# Runpod Serverless Deployment (programmatic)

Deploys two vLLM-backed serverless endpoints (one per model) via the Runpod Python SDK. Endpoint env vars are baked into the template, matching the PyINE paper's vLLM config (`pyine/configs/experiment/shortcuts/v0_rl.yaml`) exactly.

## Pre-flight

Required entries in the study's `.env` (`transferability/.env`, **NOT** pyine's root `.env` — see the root README for the rationale):

```
RUNPOD_API_KEY=...      # from https://console.runpod.io/user/settings
HF_TOKEN=...            # already present from the pilot
```

Pyine's venv must be installed: from the pyine repo root, run `uv sync --extra dev` (the `--extra dev` is only required if you also want `make test` / `make lint` to work; drop it if you only need to run the sweeps).

The Runpod deploy/teardown scripts additionally need the `runpod` Python SDK, which is **not** declared in pyine's `pyproject.toml`. Install it as a one-time ad-hoc dep into pyine's venv (skip this step if you're using a different provider):

```bash
cd $PYINE_ROOT && uv pip install runpod
```

(`python-dotenv` and `requests` are already pulled in transitively by pyine.)

## Deploy

From `pyine/transferability/`:

```bash
make deploy
```

The script is idempotent on templates: if a template with the target name already exists (typical after a teardown -- see "Templates" below), it's reused and its ID logged. Otherwise a fresh template is created. The script then creates two new endpoints attached to those templates and patches `.env` in place with the new `RUNPOD_ENDPOINT_*` IDs (atomic write via tempfile + rename, preserving comments and other env vars).

Expected output (last lines):

```
============================================================
New endpoint IDs:
  RUNPOD_ENDPOINT_SHORTCUT=<id1>
  RUNPOD_ENDPOINT_BASE=<id2>
============================================================

.env patched at <path>:
  updated: RUNPOD_ENDPOINT_SHORTCUT
  updated: RUNPOD_ENDPOINT_BASE
```

No manual .env editing required — the IDs are written for you.

## Verify

After endpoints are deployed:

```bash
make verify-endpoints
```

Cold-start adds 1-2 minutes to the first request per endpoint (image pull + model load). The script reports PASS/FAIL per endpoint. Both must pass before HellaSwag / TruthfulQA-MC1 / TruthfulQA-MC2 will work (those tasks use loglikelihood scoring against the `/openai/v1/completions` endpoint).

## Teardown

After the sweep completes and results are confirmed locally:

```bash
make teardown
# Pass --dry-run to see what would be deleted without modifying anything:
#   python deploy/teardown_endpoints.py --dry-run
```

**Deletes endpoints only.** Templates are preserved across teardown/redeploy cycles by design (see next section). `--dry-run` lists matching infrastructure without modifying anything — useful as a "what's still on Runpod" health check.

Scale-to-zero already halts billing during idle, but explicit teardown removes the endpoint records entirely so they don't appear in your Runpod console.

## Templates (preserved across teardown)

`deploy_endpoints.py` creates one template per model. Templates are not deleted by teardown: they're inert (no compute, no idle cost), they let the next `deploy` skip the create-template step (avoiding "Template name must be unique" collisions), and they serve as a permanent record of the exact config that was deployed.

If you ever need to change a template's spec (image bump, new env var), the cleanest path is: manually delete the template via the Runpod console at `https://console.runpod.io/serverless/templates`, then re-run `make deploy` to create a fresh one with the updated spec. There is intentionally no automation for template deletion.

### Per-template spec (verbatim from `deploy/deploy_endpoints.py`)

Both templates share the following config and differ only in `MODEL_NAME`:

| Field | Value |
|---|---|
| `image_name` | `runpod/worker-v1-vllm:v2.18.1` (ships vLLM 0.19.1) |
| `container_disk_in_gb` | 50 |
| `is_serverless` | true |
| `dockerArgs` | (empty) |
| `volume_in_gb` | 0 |

Environment variables baked into each template:

| Env var | Shortcut template | Base template | Notes |
|---|---|---|---|
| `MODEL_NAME` | `plstcharles-saifh/pyine-v1-qwen3-4b-shortcut` | `Qwen/Qwen3-4B-Instruct-2507` | Only per-template difference. |
| `HF_TOKEN` | from `.env` | from `.env` | Required for gated PyINE-v1 download. |
| `MAX_MODEL_LEN` | `13000` | `13000` | vLLM context window cap. |
| `DTYPE` | `bfloat16` | `bfloat16` | Paper config. |
| `GPU_MEMORY_UTILIZATION` | `0.9` | `0.9` | Paper config. |
| `SEED` | `42` | `42` | Greedy + deterministic. |
| `MAX_CONCURRENCY` | `100` | `100` | Raised from worker default 30 to absorb lm-eval-harness request bursts. |
| `REASONING_PARSER` | (unset) | (unset) | **Deliberately unset.** Instruct-2507 is the non-thinking variant; activating qwen3 reasoning-parser would reshape outputs and break lm-eval's answer extraction. |

Endpoint-level config (NOT in the template; passed to `create_endpoint`):

| Field | Value |
|---|---|
| `gpu_ids` | `AMPERE_48,AMPERE_80` (A40 48GB primary, A100 80GB fallback) |
| `gpu_count` | 1 |
| `workers_min` / `workers_max` | 0 / 1 |
| `idle_timeout` | 300 seconds (5 min) |
| `scaler_type` / `scaler_value` | `QUEUE_DELAY` / 4 |
| `flashboot` | true (warm cold-starts) |

If you change any of the above in `deploy_endpoints.py`, update this table to match.

## Reproducibility notes

The full per-template and per-endpoint config is documented in the tables above. One non-obvious deviation worth flagging in the writeup: the pinned worker image `runpod/worker-v1-vllm:v2.18.1` ships vLLM 0.19.1, which may differ from whatever the main PyINE `pyproject.toml` pins. Treat as a known deviation if asked.

## Cost reality

- Both endpoints active during the full sweep: ~$5-10 (dominated by sweep #1's long-CoT GPQA + MMLU-Pro generations)
- Per-second billing, scale-to-zero on idle — no "forgotten pod" risk
- Hard ceiling: 4-hr concurrent active time × 2 endpoints × ~$1.22/hr A40 ≈ $10

For the full-sweep cost ballpark across both sweeps (#1 + #2, cache build, secondary stratification), see `RUNBOOK.md` § "Cost monitoring".
