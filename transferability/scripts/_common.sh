#!/usr/bin/env bash
# Shared setup for transferability benchmark scripts.
# Source this from each per-benchmark script.
#
# shellcheck disable=SC2034
# (LM_EVAL, MODEL_TAG, MODEL_ID, MODEL_ARGS, OUT_DIR, LM_EVAL_COMMON_ARGS are
#  intentionally set here for the sourcing run_*.sh scripts to consume.)
#
# Provides:
#   - .env loading (HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY, endpoint IDs)
#   - Path resolution (LM_EVAL, RESULTS_ROOT)
#   - resolve_model_args <shortcut|base> <completions|chat-completions>
#       → sets MODEL_ID, ENDPOINT_ID, BASE_URL, MODEL_ARGS
#
# Conventions:
#   - Output path: <RESULTS_ROOT>/<model_tag>/<task>/
#   - W&B run name: ${WANDB_GROUP}-${model_tag}-${task}
#   - Skip if results already exist (resume-friendly)
#
# Generation params are paper-aligned (greedy, max_gen_toks=10000, seed=42).
# Concurrency is per-script: 16 inflight requests against an endpoint with
# MAX_CONCURRENCY=100, leaving headroom for the 5 other concurrent scripts.

set -uo pipefail

# Repo paths.
# PYINE_ROOT: the pyine repo, used as the uv project root. Default assumes
#   the canonical layout where this study lives at $PYINE_ROOT/transferability/.
#   Override if you cloned the PR standalone.
# TRANSF_ROOT: this study's root. Defaults to $PYINE_ROOT/transferability for
#   the canonical layout; override if you cloned the PR standalone at a
#   different path. The study's `.env` lives at $TRANSF_ROOT/.env (NOT in
#   pyine's root), so reproducers can configure this study without touching
#   pyine's own .env (which they may have customized for training).
PYINE_ROOT="${PYINE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRANSF_ROOT="${TRANSF_ROOT:-${PYINE_ROOT}/transferability}"
RESULTS_ROOT="${TRANSF_ROOT}/outputs/raw"
LOG_ROOT="${TRANSF_ROOT}/logs"
TIMING_TSV="${TRANSF_ROOT}/timing.tsv"

# Tool paths. Direct-binary on purpose -- the sweep shouldn't require uv at
# runtime (per project policy: shipped sweep scripts work with any venv tool).
# For dep-drift safety, run `uv sync --frozen` (or equivalent) in PYINE_ROOT
# once before the sweep; that's a one-time check, not a per-invocation cost.
LM_EVAL="${PYINE_ROOT}/.venv/bin/lm-eval"
export PATH="${PYINE_ROOT}/.venv/bin:${HOME}/.local/bin:${PATH}"

# W&B grouping
WANDB_PROJECT="${WANDB_PROJECT:-pyine-transferability}"
WANDB_GROUP="${WANDB_GROUP:-2026-05-15-sweep}"

# Load .env so HF_TOKEN, RUNPOD_API_KEY, RUNPOD_ENDPOINT_*, WANDB_API_KEY land
# in env. Default location is the study's own .env (NOT pyine's). Override via
# TRANSF_DOTENV=... if you want a non-default path; the legacy PYINE_DOTENV
# name is also accepted for backward compat with pre-rename scripts.
ENV_FILE="${TRANSF_DOTENV:-${PYINE_DOTENV:-${TRANSF_ROOT}/.env}}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env not found at ${ENV_FILE}" >&2
    echo "" >&2
    echo "Create it from the template:" >&2
    echo "  cp ${TRANSF_ROOT}/.env.example ${TRANSF_ROOT}/.env" >&2
    echo "  \$EDITOR ${TRANSF_ROOT}/.env" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# LOCAL=1 convenience: if set (e.g. `LOCAL=1 bash scripts/run_hellaswag.sh shortcut`),
# point inference at localhost defaults instead of Runpod. Mirrors the Makefile's
# LOCAL=1 toggle; either entrypoint works. Explicit env vars still take precedence.
if [[ "${LOCAL:-}" == "1" ]]; then
    INFERENCE_URL_SHORTCUT="${INFERENCE_URL_SHORTCUT:-http://localhost:8001/v1/completions}"
    INFERENCE_URL_BASE="${INFERENCE_URL_BASE:-http://localhost:8002/v1/completions}"
    INFERENCE_API_KEY="${INFERENCE_API_KEY:-EMPTY}"
    export INFERENCE_URL_SHORTCUT INFERENCE_URL_BASE INFERENCE_API_KEY
fi

# Inference auth: lm-eval's API backend reads OPENAI_API_KEY for Bearer auth.
# INFERENCE_API_KEY (preferred, provider-agnostic) overrides RUNPOD_API_KEY.
# Either must be set for any provider that requires auth; for local vLLM/SGLang
# without auth, leaving it empty is fine (lm-eval will send an empty Bearer header).
export OPENAI_API_KEY="${INFERENCE_API_KEY:-${RUNPOD_API_KEY:-EMPTY}}"

# HumanEval's code execution path requires this explicit opt-in.
export HF_ALLOW_CODE_EVAL=1

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

# Model IDs per tag. Defaults are the PyINE-v1 organism and its Qwen3 base
# (tags `shortcut` and `base`); override <TAG>_MODEL_ID in .env to audit a
# different organism. Multi-model setups (arbitrary tags beyond shortcut/base)
# only need <TAG>_MODEL_ID + one of INFERENCE_URL_<TAG> | RUNPOD_ENDPOINT_<TAG>
# set in .env -- the resolver does the lookup generically.
SHORTCUT_MODEL_ID="${SHORTCUT_MODEL_ID:-plstcharles-saifh/pyine-v1-qwen3-4b-shortcut}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
export SHORTCUT_MODEL_ID BASE_MODEL_ID

# Tags to run, comma-separated. Defaults to `shortcut,base` for the canonical
# PyINE-v1 audit; set MODELS=my_org_a,my_org_b in .env to run an arbitrary
# pair, or MODELS=my_org for a single-model run.
MODELS="${MODELS:-shortcut,base}"
export MODELS

# Resolve runtime context for a given model_tag (arbitrary string, uppercase
# in the env-var lookup). All tasks use /v1/completions (raw text mode).
# For instruct tasks, the per-task script passes --apply_chat_template so
# lm-eval renders the chat template LOCALLY to a flat prompt -- matches the
# pilot's pipeline exactly.
#
# Env-var lookup per tag (all uppercase suffix, indirect via ${!var}):
#   <TAG>_MODEL_ID            required -- the model name sent to /completions
#   INFERENCE_URL_<TAG>       preferred -- explicit URL (provider-agnostic)
#   RUNPOD_ENDPOINT_<TAG>     fallback  -- Runpod endpoint ID; URL built from it
#
# Sets globals: MODEL_TAG, MODEL_ID, BASE_URL, MODEL_ARGS
resolve_model_args() {
    local tag="$1"
    local tag_upper
    tag_upper="$(echo "${tag}" | tr '[:lower:]' '[:upper:]')"

    local model_id_var="${tag_upper}_MODEL_ID"
    local url_var="INFERENCE_URL_${tag_upper}"
    local runpod_var="RUNPOD_ENDPOINT_${tag_upper}"

    MODEL_ID="${!model_id_var:-}"
    if [[ -z "${MODEL_ID}" ]]; then
        echo "ERROR: ${model_id_var} not set for tag '${tag}'. Add to .env:" >&2
        echo "  ${model_id_var}=org/model-name" >&2
        exit 1
    fi

    if [[ -n "${!url_var:-}" ]]; then
        BASE_URL="${!url_var}"
    elif [[ -n "${!runpod_var:-}" ]]; then
        BASE_URL="https://api.runpod.ai/v2/${!runpod_var}/openai/v1/completions"
    else
        echo "ERROR: neither ${url_var} nor ${runpod_var} set for tag '${tag}'." >&2
        echo "Set one in .env (URL preferred for non-Runpod providers)." >&2
        exit 1
    fi
    MODEL_TAG="${tag}"

    # tokenizer_backend=huggingface so lm-eval can apply chat templates,
    # compute token counts and length normalization locally.
    # max_length=13000 matches the paper's vLLM max_model_len.
    # timeout=600 because the shortcut model can produce long reasoning
    # traces; the default ~60s caused mass TimeoutErrors on first run.
    # num_concurrent=16: 6 scripts × 16 = 96 < MAX_CONCURRENCY=100 on endpoint.
    MODEL_ARGS="model=${MODEL_ID},base_url=${BASE_URL},tokenizer=${MODEL_ID},tokenizer_backend=huggingface,max_length=13000,timeout=600,num_concurrent=16,max_retries=3"
}

# Gold-standard runs use FULL test sets; no --samples flag needed.
# Function kept as a no-op hook for ad-hoc resume scenarios that want to
# constrain the run to a specific seeded subset.
maybe_samples_flag() {
    return 0
}

# Skip if the output dir already contains a results JSON (resume-friendly).
already_done() {
    local out_dir="$1"
    [[ -d "${out_dir}" ]] && [[ -n "$(find "${out_dir}" -name 'results_*.json' -print -quit 2>/dev/null)" ]]
}

# Emit a timing line: tab-separated, appendable concurrently.
log_timing() {
    local model_tag="$1" task="$2" elapsed_sec="$3" status="$4"
    printf '%s\t%s\t%s\t%d\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${model_tag}" "${task}" "${elapsed_sec}" "${status}" \
        >> "${TIMING_TSV}"
}

# Build the common lm-eval CLI args. Task-specific scripts append --tasks and
# any task-specific flags (e.g., --confirm_run_unsafe_code, --apply_chat_template).
#
# Smoke-test override: if LIMIT is set in the environment (e.g. `LIMIT=1 bash
# run_gsm8k.sh shortcut`), it's forwarded to lm-eval as --limit ${LIMIT}.
common_lm_eval_args() {
    local model_tag="$1" task="$2"
    local out_dir="${RESULTS_ROOT}/${model_tag}/${task}"
    local wandb_name="${WANDB_GROUP}-${model_tag}-${task}"
    OUT_DIR="${out_dir}"
    mkdir -p "${out_dir}"
    LM_EVAL_COMMON_ARGS=(
        --output_path "${out_dir}"
        --log_samples
        --seed 42
        --gen_kwargs "temperature=0,max_gen_toks=10000"
        --wandb_args "project=${WANDB_PROJECT},group=${WANDB_GROUP},name=${wandb_name},tags=transferability;${model_tag};${task}"
    )
    if [[ -n "${LIMIT:-}" ]]; then
        LM_EVAL_COMMON_ARGS+=(--limit "${LIMIT}")
    fi
}
