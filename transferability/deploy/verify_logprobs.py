"""Verify that both serverless endpoints expose /openai/v1/completions with
working logprobs support.

The HellaSwag, TruthfulQA-MC1, and TruthfulQA-MC2 tasks score candidates by
their loglikelihood under the model. This requires the OpenAI text-completions
endpoint to honor the `logprobs` parameter. The worker-vllm docs say "anything
not recognized is forwarded to vLLM" -- vLLM's OpenAI server does support
logprobs, so this should just work. This script confirms it.

Pass criterion: each endpoint returns a response with `choices[0].logprobs`
populated AND containing at least one token logprob.

If this fails, fall back options:
  (a) Set OPENAI_SERVED_MODEL_NAME_OVERRIDE explicitly and retry -- sometimes
      models hide their served name in a way that breaks the completions route.
  (b) Switch to Custom Deployment using runpod/worker-v1-vllm:v2.18.1 directly
      with --enable-logprobs explicitly. (Probably unnecessary.)

Expected first-run latency: ~30-60s per endpoint cold-start, then sub-second.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import dotenv
import requests

_TRANSF_ROOT = pathlib.Path(__file__).resolve().parents[1]  # transferability/ (study root)
# study's own .env; override via TRANSF_DOTENV=...; legacy PYINE_DOTENV also accepted
ENV_PATH = pathlib.Path(os.environ.get("TRANSF_DOTENV", os.environ.get("PYINE_DOTENV", _TRANSF_ROOT / ".env")))


def check(
    endpoint_id: str,
    api_key: str,
    tag: str,
) -> bool:
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # step 1: discover the served model name; worker-vllm sets it to the HF model path when
    # OPENAI_SERVED_MODEL_NAME_OVERRIDE is unset. Asking /v1/models is robust to either case.
    # generous timeout because this is also the first cold-start request: 5-10 min
    # (image pull + HF model download + vLLM warmup) on a brand-new endpoint, then sub-second.
    models_url = f"{base_url}/models"
    print(f"[{tag}] GET {models_url}", flush=True)
    start_time = time.monotonic()
    try:
        resp = requests.get(models_url, headers=headers, timeout=900)
    except requests.RequestException as err:
        print(f"[{tag}] ERROR: /models request failed: {err}", flush=True)
        return False
    elapsed = time.monotonic() - start_time
    print(f"[{tag}] /models HTTP {resp.status_code} in {elapsed:.1f}s", flush=True)
    if resp.status_code != 200:
        print(f"[{tag}] body: {resp.text[:500]}", flush=True)
        return False
    try:
        body = resp.json()
    except ValueError:
        print(f"[{tag}] FAIL: /models non-JSON response", flush=True)
        return False
    served = body.get("data") or []
    if not served:
        print(f"[{tag}] FAIL: /models returned empty data list: {body!r}", flush=True)
        return False
    model_name = served[0].get("id")
    if not model_name:
        print(f"[{tag}] FAIL: /models entry missing id: {served[0]!r}", flush=True)
        return False
    print(f"[{tag}] served model: {model_name}", flush=True)

    # step 2: completions with logprobs
    url = f"{base_url}/completions"
    payload = {
        "model": model_name,
        "prompt": "The capital of France is",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 5,
    }

    print(f"[{tag}] POST {url}", flush=True)
    start_time = time.monotonic()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=900)
    except requests.RequestException as err:
        print(f"[{tag}] ERROR: request failed: {err}", flush=True)
        return False
    elapsed = time.monotonic() - start_time
    print(f"[{tag}] HTTP {resp.status_code} in {elapsed:.1f}s", flush=True)

    if resp.status_code != 200:
        print(f"[{tag}] body: {resp.text[:500]}", flush=True)
        return False

    try:
        body = resp.json()
    except ValueError:
        print(f"[{tag}] FAIL: non-JSON response: {resp.text[:200]}", flush=True)
        return False

    choices = body.get("choices") or []
    if not choices:
        print(f"[{tag}] FAIL: no choices in response: {body!r}", flush=True)
        return False

    logprobs = choices[0].get("logprobs")
    if logprobs is None:
        print(f"[{tag}] FAIL: choices[0].logprobs is null", flush=True)
        print(f"[{tag}]   full choice: {choices[0]!r}", flush=True)
        return False

    # openAI completions logprobs format: {tokens, token_logprobs, top_logprobs, text_offset}
    token_logprobs = logprobs.get("token_logprobs")
    if not token_logprobs:
        print(f"[{tag}] FAIL: token_logprobs missing or empty", flush=True)
        print(f"[{tag}]   logprobs payload: {logprobs!r}", flush=True)
        return False

    print(f"[{tag}] PASS: logprobs present, token_logprobs={token_logprobs}", flush=True)
    return True


def main() -> int:
    if not ENV_PATH.is_file():
        print(f"ERROR: env file not found at {ENV_PATH}", file=sys.stderr)
        return 1
    dotenv.load_dotenv(ENV_PATH)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY missing from .env", file=sys.stderr)
        return 1

    # tags from MODELS env var (default `shortcut,base`); each needs
    # RUNPOD_ENDPOINT_<TAG> in .env (populated by deploy_endpoints.py)
    tags = [tag.strip() for tag in os.environ.get("MODELS", "shortcut,base").split(",") if tag.strip()]
    endpoints = [(tag, os.environ.get(f"RUNPOD_ENDPOINT_{tag.upper()}")) for tag in tags]
    missing = [tag for tag, endpoint in endpoints if not endpoint]
    if missing:
        print(
            f"ERROR: missing RUNPOD_ENDPOINT_<TAG> in .env for: {missing}. Did you run deploy_endpoints.py?",
            file=sys.stderr,
        )
        return 1

    results = []
    for tag, endpoint in endpoints:
        ok = check(endpoint, api_key, tag)
        results.append((tag, ok))
        print()

    bad = [tag for tag, ok in results if not ok]
    if bad:
        print(f"FAIL: logprobs check did not pass for: {bad}")
        return 1
    print("All endpoints expose working logprobs. HellaSwag / MC tasks unblocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
