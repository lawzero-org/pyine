"""Deploy serverless vLLM endpoints (one per tag in MODELS) for the transferability sweep.

By default deploys the canonical PyINE-v1 pair (shortcut + base). For custom
multi-model setups, set MODELS=tag1,tag2,... in .env plus <TAG>_MODEL_ID per
tag and re-run; one template + endpoint is created per tag.

Creates one template per model (env vars are template-level in Runpod), then
one endpoint per template. Endpoints launch concurrently (separate workers,
scale-to-zero).

Idempotency: templates are reused if one with the target name already exists
(common after a teardown leaves orphans behind). Endpoints are NOT checked --
re-running while a live endpoint with the same name exists will still fail.
Run teardown for endpoints first.

Hardware: AMPERE_80 (A100 80GB). Hopper (H100) isn't in the standard gpuIds
catalogue exposed by `create_endpoint`; A100 is the highest-tier safe pick
that's documented and plenty for a 4B model.

Cost model: per-second of active worker time, scale-to-zero on idle.

Reads from the study's .env (default: $TRANSF_ROOT/.env; override via
TRANSF_DOTENV, with legacy PYINE_DOTENV also accepted). NOT pyine's root .env
-- see deploy/README.md for the rationale.
    RUNPOD_API_KEY        (required)
    HF_TOKEN              (required -- for gated model + dataset downloads)

Writes to stdout AND patches .env in place (atomic write via tempfile +
rename):
    RUNPOD_ENDPOINT_<TAG>=<new-id>     (one line per tag in MODELS)

If those keys already exist in .env (e.g., from a previous deploy whose
endpoints have since been torn down), their values are replaced. Other env
vars, comments, and blank lines are preserved exactly. The downstream
benchmark scripts read the updated values.
"""

from __future__ import annotations

import os
import pathlib
import sys

import dotenv
import runpod

_TRANSF_ROOT = pathlib.Path(__file__).resolve().parents[1]  # transferability/ (study root)
# Study's own .env (NOT pyine's). Reproducers can configure this study without
# touching pyine's root .env. Override via TRANSF_DOTENV=...; the legacy
# PYINE_DOTENV name is also accepted for pre-rename backward compat.
ENV_PATH = pathlib.Path(os.environ.get("TRANSF_DOTENV", os.environ.get("PYINE_DOTENV", _TRANSF_ROOT / ".env")))

# Pinned for reproducibility. v2.18.1 ships vLLM 0.19.1 (per the Hub README
# the user saw on 2026-05-15). Document this version in the writeup so the
# experiment can be re-run against the same engine.
WORKER_IMAGE = "runpod/worker-v1-vllm:v2.18.1"

# Comma-separated priority list. AMPERE_48 (A40-class, 48 GB) is the cheapest
# tier with consistently high supply that still fits 4B bf16 + KV cache for
# max_model_len=13000 comfortably. AMPERE_80 (A100 80GB) as fallback in case
# the 48 GB pool is exhausted -- overkill for a 4B model but works.
GPU_IDS = "AMPERE_48,AMPERE_80"
GPU_COUNT = 1

# 50 GB image + weights + scratch.
CONTAINER_DISK_GB = 50

# Scale-to-zero. 5-min idle, FlashBoot for warm cold-starts.
WORKERS_MIN = 0
WORKERS_MAX = 1
IDLE_TIMEOUT_SECS = 300

# Paper-aligned vLLM args, sourced verbatim from
# pyine/configs/experiment/shortcuts/v0_rl.yaml (greedy bf16, max_model_len
# 13000, seed 42). MAX_CONCURRENCY raised from the template default (30) to
# 100 because lm-eval-harness with batch_size=auto sends bursts of dozens of
# in-flight requests per client and we run 6 clients per endpoint.
# Critical NOT-SET: REASONING_PARSER. The Instruct-2507 variant we use is
# the non-thinking branch; activating qwen3 reasoning-parser would reshape
# outputs and break lm-eval's answer extraction.
PAPER_VLLM_ENV: dict[str, str] = {
    "MAX_MODEL_LEN": "13000",
    "DTYPE": "bfloat16",
    "GPU_MEMORY_UTILIZATION": "0.9",
    "SEED": "42",
    "MAX_CONCURRENCY": "100",
}

# Hardcoded defaults for the canonical PyINE-v1 audit. For custom multi-model
# setups, override via MODELS=tag1,tag2 + <TAG>_MODEL_ID per tag in .env.
_DEFAULT_MODEL_IDS = {
    "shortcut": "plstcharles-saifh/pyine-v1-qwen3-4b-shortcut",
    "base": "Qwen/Qwen3-4B-Instruct-2507",
}


def _resolve_models() -> list[tuple[str, str]]:
    """Resolve the (tag, model_id) list from env. Reads MODELS (comma-separated
    tags) plus <TAG>_MODEL_ID per tag. Defaults to the canonical shortcut+base
    audit pair."""
    tags_csv = os.environ.get("MODELS", "shortcut,base")
    tags = [tag.strip() for tag in tags_csv.split(",") if tag.strip()]
    out: list[tuple[str, str]] = []
    for tag in tags:
        var = f"{tag.upper()}_MODEL_ID"
        model_id = os.environ.get(var) or _DEFAULT_MODEL_IDS.get(tag)
        if not model_id:
            print(f"ERROR: {var} not set in .env (required for tag '{tag}')", file=sys.stderr)
            sys.exit(1)
        out.append((tag, model_id))
    return out


def _extract_id(
    obj,  # noqa: ANN001
    kind: str,
) -> str:
    """Tolerate both dict and attribute access on SDK return values."""
    if isinstance(obj, dict):
        if "id" in obj:
            return obj["id"]
        raise RuntimeError(f"{kind} response missing 'id': {obj!r}")
    if hasattr(obj, "id"):
        return obj.id
    raise RuntimeError(f"{kind} response has unexpected shape: {obj!r}")


def _find_template_by_name(name: str) -> dict | None:
    """Look up an existing template by exact name via the Runpod GraphQL API.

    The runpod Python SDK doesn't expose a list-templates helper (as of the
    pinned version in pyine's lockfile), so we query GraphQL directly. Returns
    the template's dict (with at least `id`, `name`, `imageName`) or None if
    nothing matches.

    Used to skip template creation when a template with the target name
    already exists -- typically an orphan from a previous teardown, since
    teardown_endpoints.py only deletes templates attached to live endpoints.
    """
    from runpod.api.graphql import run_graphql_query

    query = "query { myself { podTemplates { id name imageName } } }"
    resp = run_graphql_query(query)
    templates = resp.get("data", {}).get("myself", {}).get("podTemplates", []) or []
    for template in templates:
        if template.get("name") == name:
            return template
    return None


def update_env_file(
    env_path: pathlib.Path,
    updates: dict[str, str],
) -> dict[str, str]:
    """Update KEY=VALUE lines in .env in place, preserving everything else.

    For each key in `updates`:
      - If a non-commented line `KEY=...` exists, its value is replaced.
      - Otherwise, `KEY=VALUE` is appended at the end of the file.

    Comments, blank lines, ordering, and other env vars are preserved exactly.
    Commented-out forms like `# KEY=...` are NOT matched and stay as-is.

    Writes atomically: writes to a sibling tempfile, then `os.replace`'s it
    over the target. A SIGKILL mid-write leaves the original .env intact.

    Returns: {key: action} where action is "updated" or "appended".
    """
    import contextlib
    import tempfile

    original = env_path.read_text()
    lines = original.splitlines(keepends=True)
    actions: dict[str, str] = {}
    out_lines: list[str] = []
    seen: dict[str, bool] = dict.fromkeys(updates, False)

    for line in lines:
        replaced = False
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            for key, value in updates.items():
                if stripped.startswith(f"{key}="):
                    trailing = "\n" if line.endswith("\n") else ""
                    out_lines.append(f"{key}={value}{trailing}")
                    seen[key] = True
                    actions[key] = "updated"
                    replaced = True
                    break
        if not replaced:
            out_lines.append(line)

    # append any keys that weren't found in the file
    missing = [(key, value) for key, value in updates.items() if not seen[key]]
    if missing:
        if out_lines and not out_lines[-1].endswith("\n"):
            out_lines[-1] += "\n"
        for key, value in missing:
            out_lines.append(f"{key}={value}\n")
            actions[key] = "appended"

    fd, tmp_path = tempfile.mkstemp(dir=env_path.parent, prefix=".env.tmp.")
    try:
        with os.fdopen(fd, "w") as tmp_fh:
            tmp_fh.writelines(out_lines)
        # preserve original file mode (e.g., 0600 on locked-down hosts)
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, env_path.stat().st_mode)
        os.replace(tmp_path, env_path)
    except Exception:  # noqa: BLE001 -- cleanup tempfile then re-raise; not silencing
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return actions


def main() -> int:
    if not ENV_PATH.is_file():
        print(f"ERROR: env file not found at {ENV_PATH}", file=sys.stderr)
        return 1
    dotenv.load_dotenv(ENV_PATH)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not in .env", file=sys.stderr)
        return 1
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not in .env", file=sys.stderr)
        return 1

    runpod.api_key = api_key

    deployed: dict[str, str] = {}
    models = _resolve_models()
    for tag, model_id in models:
        env_vars = {
            **PAPER_VLLM_ENV,
            "MODEL_NAME": model_id,
            "HF_TOKEN": hf_token,
        }
        template_name = f"pyine-transferability-{tag}-tpl"
        endpoint_name = f"pyine-transferability-{tag}"

        existing = _find_template_by_name(template_name)
        if existing is not None:
            template_id = existing["id"]
            existing_image = existing.get("imageName", "<unknown>")
            print(
                f"[{tag}] reusing existing template id={template_id} name={template_name} image={existing_image}",
                flush=True,
            )
            if existing_image != WORKER_IMAGE:
                print(
                    f"[{tag}]   NOTE: existing template image ({existing_image}) differs from "
                    f"script's pinned WORKER_IMAGE ({WORKER_IMAGE}). Reusing as-is. "
                    f"Tear down + redeploy to refresh.",
                    flush=True,
                )
        else:
            print(f"[{tag}] create_template name={template_name} ...", flush=True)
            tpl = runpod.create_template(
                name=template_name,
                image_name=WORKER_IMAGE,
                env=env_vars,
                container_disk_in_gb=CONTAINER_DISK_GB,
                is_serverless=True,
            )
            template_id = _extract_id(tpl, "template")
            print(f"[{tag}]   template_id={template_id}", flush=True)

        print(f"[{tag}] create_endpoint name={endpoint_name} ...", flush=True)
        ep = runpod.create_endpoint(
            name=endpoint_name,
            template_id=template_id,
            gpu_ids=GPU_IDS,
            gpu_count=GPU_COUNT,
            workers_min=WORKERS_MIN,
            workers_max=WORKERS_MAX,
            idle_timeout=IDLE_TIMEOUT_SECS,
            scaler_type="QUEUE_DELAY",
            scaler_value=4,
            flashboot=True,
        )
        endpoint_id = _extract_id(ep, "endpoint")
        print(f"[{tag}]   endpoint_id={endpoint_id}", flush=True)
        deployed[tag] = endpoint_id

    print()
    print("=" * 60)
    print("New endpoint IDs:")
    for tag, endpoint_id in deployed.items():
        print(f"  RUNPOD_ENDPOINT_{tag.upper()}={endpoint_id}")
    print("=" * 60)

    # patch .env in place; failure here is non-fatal -- ids already echoed
    updates = {f"RUNPOD_ENDPOINT_{tag.upper()}": endpoint_id for tag, endpoint_id in deployed.items()}
    try:
        actions = update_env_file(ENV_PATH, updates)
        print(f"\n.env patched at {ENV_PATH}:")
        for key, action in actions.items():
            print(f"  {action}: {key}")
    except Exception as err:  # noqa: BLE001 -- print-and-continue; ids already echoed so user can paste manually
        print(
            f"\nWARN: failed to patch .env at {ENV_PATH}: {type(err).__name__}: {err}\n"
            f"Manually copy the two RUNPOD_ENDPOINT_* lines above into your .env.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
