# Reproduction targets for the transferability study.
#
# First-time setup:
#     make install       # install ad-hoc deps (lm-eval, runpod) into pyine venv
#
# Standard usage (full reproduction):
#     make deploy        # deploy 2 Runpod endpoints (skip if using non-Runpod provider)
#     make sweep1        # run lm-eval on 6 benchmarks
#     make sweep2        # run CueFlip cue-injection
#     make analyze       # run all 6 analysis scripts
#     make teardown      # delete Runpod endpoints
#
# Smoke-testing (1 item per task, ~1-2 minutes against a real endpoint):
#     make smoke         # smoke-test both sweeps end-to-end
#
# Other:
#     make lint          # ruff check + format + shellcheck (CI parity)
#     make test          # run the local pytest suite
#     make help          # show this list

.PHONY: help _check-pyine-root _check-env _check-lm-eval install deploy verify-endpoints sweep1 sweep2 build-cache analyze teardown smoke lint test check

# This study expects to live at `pyine/transferability/`. Under that
# canonical layout, PYINE_ROOT auto-resolves to the pyine repo root and you
# never need to set it. The auto-resolution is `<this Makefile's dir>/..`.
# For non-canonical setups (CI, sibling-clone, etc.), pass PYINE_ROOT
# explicitly: `PYINE_ROOT=/path/to/pyine make <target>`.
PYINE_ROOT ?= $(shell cd $(dir $(lastword $(MAKEFILE_LIST)))/.. && pwd)
PYTHON     ?= $(PYINE_ROOT)/.venv/bin/python

# LOCAL=1: route all inference at localhost defaults instead of the Runpod
# fallback. Useful for development against a local vLLM/SGLang setup.
# Expected ports:
#   - shortcut model: http://localhost:8001
#   - base model:     http://localhost:8002
#   - judge (cueflip cache + recovery): http://localhost:8000 (already the default)
# Override individual ports via the standard env vars (INFERENCE_URL_*,
# CUEFLIP_INFERENCE_URL_*, CUEFLIP_JUDGE_URL) if your setup differs. Settings
# made via these env vars take precedence over LOCAL=1's defaults.
ifdef LOCAL
    export INFERENCE_URL_SHORTCUT ?= http://localhost:8001/v1/completions
    export INFERENCE_URL_BASE ?= http://localhost:8002/v1/completions
    export INFERENCE_API_KEY ?= EMPTY
    export CUEFLIP_INFERENCE_URL_SHORTCUT ?= http://localhost:8001/v1
    export CUEFLIP_INFERENCE_URL_BASE ?= http://localhost:8002/v1
    export CUEFLIP_INFERENCE_API_KEY ?= EMPTY
endif

# Early-fail with a clear error if the canonical layout isn't satisfied.
# Catches both:
#   (a) `make` invoked from outside pyine/transferability/, so the
#       auto-resolved PYINE_ROOT points at the wrong directory;
#   (b) pyine's venv missing entirely (user hasn't run `uv sync` yet).
# Without this, downstream errors are confusing ("No module named pytest"
# from a venv that doesn't have pytest installed).
_check-pyine-root:
	@if [ ! -f "$(PYINE_ROOT)/pyproject.toml" ] || ! grep -q '^name = "pyine"' "$(PYINE_ROOT)/pyproject.toml" 2>/dev/null; then \
	    echo "ERROR: cannot find pyine at $(PYINE_ROOT)" >&2; \
	    echo "(no pyproject.toml with name='pyine' found)." >&2; \
	    echo "" >&2; \
	    echo "This Makefile expects to run from inside pyine/transferability/." >&2; \
	    echo "cd into that directory and try again:" >&2; \
	    echo "  cd /path/to/pyine/transferability && make $@" >&2; \
	    echo "" >&2; \
	    echo "If you have pyine at a non-standard location, you can override:" >&2; \
	    echo "  PYINE_ROOT=/path/to/pyine make $@" >&2; \
	    exit 1; \
	fi
	@if [ ! -x "$(PYTHON)" ]; then \
	    echo "ERROR: $(PYTHON) not found or not executable." >&2; \
	    echo "" >&2; \
	    echo "The pyine venv is missing. From $(PYINE_ROOT), run:" >&2; \
	    echo "  uv sync --extra dev" >&2; \
	    echo "(--extra dev pulls in pytest + pre-commit; you can drop it" >&2; \
	    echo "if you only intend to run the sweep, not 'make test'/'make lint')." >&2; \
	    exit 1; \
	fi

# Targets that hit Runpod / inference endpoints additionally require .env.
# The study's .env lives at $(CURDIR)/.env -- separate from pyine's root .env
# so reproducers can configure this study without touching their pyine config.
_check-env: _check-pyine-root
	@if [ ! -f "$(CURDIR)/.env" ]; then \
	    echo "ERROR: $(CURDIR)/.env not found." >&2; \
	    echo "" >&2; \
	    echo "This study has its own .env (NOT pyine's root .env)." >&2; \
	    echo "Copy the template and fill in your provider credentials:" >&2; \
	    echo "  cp .env.example .env" >&2; \
	    echo "  \$$EDITOR .env" >&2; \
	    exit 1; \
	fi

# Sweep #1 uses lm-evaluation-harness, an ad-hoc dep not declared in pyine's
# pyproject.toml. Pre-flight check to avoid mid-sweep failure with a cryptic
# "lm-eval: No such file or directory" error from inside scripts/run_*.sh.
_check-lm-eval: _check-pyine-root
	@if [ ! -x "$(PYINE_ROOT)/.venv/bin/lm-eval" ]; then \
	    echo "ERROR: lm-eval not installed in $(PYINE_ROOT)/.venv" >&2; \
	    echo "" >&2; \
	    echo "Sweep #1 (and 'make smoke') invoke lm-evaluation-harness, which" >&2; \
	    echo "is an ad-hoc dep NOT declared in pyine's pyproject.toml. Install" >&2; \
	    echo "it once into pyine's venv:" >&2; \
	    echo "  cd $(PYINE_ROOT) && uv pip install lm-eval" >&2; \
	    exit 1; \
	fi

help: ## Show this list of targets
	@echo "Targets (run with 'make <name>'):"
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' $(lastword $(MAKEFILE_LIST)) | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  %-18s  %s\n", $$1, $$2}'

install: _check-pyine-root ## Install ad-hoc deps (lm-eval, runpod) into pyine's venv -- run once before sweeps
	@echo "==> installing lm-eval + runpod into $(PYINE_ROOT)/.venv"
	@echo "    (these are NOT declared in pyine's pyproject.toml; this study needs them)"
	cd $(PYINE_ROOT) && uv pip install lm-eval runpod
	@echo ""
	@echo "==> install complete"
	@echo "    lm-eval:  $$($(PYTHON) -c 'import lm_eval; print(lm_eval.__version__)' 2>/dev/null || echo MISSING)"
	@echo "    runpod:   $$($(PYTHON) -c 'import runpod; print(runpod.__version__)' 2>/dev/null || echo MISSING)"
	@if [ -x "$(PYINE_ROOT)/.venv/bin/vllm" ]; then \
	    echo "    vllm:     $$($(PYINE_ROOT)/.venv/bin/vllm --version 2>&1 | tail -1) (found in pyine venv)"; \
	else \
	    echo ""; \
	    echo "    NOTE: vllm not found in pyine venv. Only needed for LOCAL=1 inference."; \
	    echo "          Install in a separate venv (recommended to avoid torch conflicts):"; \
	    echo "            python -m venv ~/vllm-venv && ~/vllm-venv/bin/pip install vllm"; \
	    echo "          Then start vllm manually before running 'LOCAL=1 make ...' targets."; \
	fi

deploy: _check-env ## Deploy 2 Runpod serverless endpoints (Runpod path only)
	cd deploy && $(PYTHON) deploy_endpoints.py

verify-endpoints: _check-env ## Smoke-test deployed Runpod endpoints
	cd deploy && $(PYTHON) verify_logprobs.py

sweep1: _check-env _check-lm-eval ## Sweep #1 -- lm-eval over 6 benchmarks x 2 models (full test splits)
	bash scripts/run_all.sh

build-cache: _check-env ## Build the GSM8K op-flip wrong-numeric cache (idempotent; uses CUEFLIP_JUDGE_URL)
	$(PYTHON) cueflip/build_operation_flip_cache.py $(if $(DRY_RUN),--dry-run,)

# Toggle --dry-run on sweep2 by setting DRY_RUN=1. When set, the cache build
# and the runner both use synthetic data (no HTTP), writing to separate paths
# (cueflip/operation_flip_cache_dry_run.json and cueflip/results_dry_run/)
# so they can never collide with real data. The judge step is skipped under
# DRY_RUN since synthetic records always parse cleanly.
DRY_RUN_FLAG = $(if $(DRY_RUN),--dry-run,)

sweep2: _check-env ## Sweep #2 -- CueFlip cue-injection (DRY_RUN=1 for no-HTTP smoke; SKIP_CACHE_BUILD=1 to skip cache step)
	@# Cache build is idempotent (skips items already cached). Opt out with
	@# SKIP_CACHE_BUILD=1 make sweep2  -- useful if the judge endpoint
	@# isn't running or you've already committed a complete cache.
	@if [ -z "$(SKIP_CACHE_BUILD)" ]; then \
	    echo "==> build-cache (skip with SKIP_CACHE_BUILD=1)"; \
	    $(PYTHON) cueflip/build_operation_flip_cache.py $(DRY_RUN_FLAG); \
	else \
	    echo "==> SKIP_CACHE_BUILD=1 set, skipping cache build"; \
	fi
	$(PYTHON) cueflip/runner.py $(DRY_RUN_FLAG)
	@if [ -z "$(DRY_RUN)" ]; then \
	    $(PYTHON) cueflip/judge.py; \
	else \
	    echo "==> DRY_RUN=1 set, skipping judge.py (synthetic records always parse)"; \
	fi
	@if [ -z "$(DRY_RUN)" ]; then \
	    $(PYTHON) cueflip/analyze.py; \
	else \
	    CUEFLIP_RESULTS_ROOT=cueflip/results_dry_run $(PYTHON) cueflip/analyze.py; \
	fi

analyze: _check-pyine-root ## Run all 6 analysis scripts (analyze + descriptive_stats + items D/E/F/G; gracefully skips incomplete cells)
	$(PYTHON) scripts/analyze.py
	$(PYTHON) scripts/descriptive_stats.py
	$(PYTHON) scripts/analysis_d.py
	$(PYTHON) scripts/analysis_e.py
	$(PYTHON) scripts/analysis_f.py
	$(PYTHON) scripts/analysis_g.py

teardown: _check-env ## Delete Runpod endpoints (templates preserved; --dry-run available)
	cd deploy && $(PYTHON) teardown_endpoints.py

# SMOKE_TAG: which model tag to exercise in the smoke. Defaults to `shortcut`
# (canonical PyINE-v1 audit); override with `make smoke SMOKE_TAG=my_org` for
# custom multi-model setups (see README "Multi-model setups").
SMOKE_TAG ?= shortcut

smoke: _check-env _check-lm-eval ## Run a 1-item smoke test of both sweeps end-to-end (~1-2 minutes)
	@echo "=== Sweep #2 smoke (CueFlip multiple-choice path, 1 item, 8 cue families, tag=$(SMOKE_TAG)) ==="
	WANDB_MODE=disabled $(PYTHON) cueflip/runner.py \
		--models $(SMOKE_TAG) --benchmarks hellaswag \
		--items-cap 1 --paraphrase-indices 0 --num-concurrent 1
	@echo "=== Sweep #2 smoke (CueFlip free-form path, GSM8K 1 item, primary strategy, tag=$(SMOKE_TAG)) ==="
	WANDB_MODE=disabled $(PYTHON) cueflip/runner.py \
		--models $(SMOKE_TAG) --benchmarks gsm8k --gsm8k-mode primary \
		--items-cap 1 --paraphrase-indices 0 --num-concurrent 1
	@echo "=== Sweep #1 smoke (gsm8k_cot, --limit 1, tag=$(SMOKE_TAG)) ==="
	WANDB_MODE=disabled LIMIT=1 bash scripts/run_gsm8k.sh $(SMOKE_TAG)

lint: _check-pyine-root ## Run ruff + format + shellcheck on the study folder (CI parity)
	cd $(PYINE_ROOT) && uv run ruff check $(CURDIR)
	cd $(PYINE_ROOT) && uv run ruff format --check $(CURDIR)
	@if command -v shellcheck >/dev/null 2>&1; then \
	    shellcheck scripts/*.sh; \
	else \
	    echo ""; \
	    echo "NOTE: shellcheck not installed -- skipping shell-script lint."; \
	    echo "      Install via 'apt install shellcheck' (Debian/Ubuntu) or"; \
	    echo "      'brew install shellcheck' (macOS) for full CI parity."; \
	fi

test: _check-pyine-root ## Run the local pytest suite (scoped to this study)
	@if ! $(PYTHON) -c "import pytest" 2>/dev/null; then \
	    echo "ERROR: pytest not installed in $(PYINE_ROOT)/.venv" >&2; \
	    echo "" >&2; \
	    echo "pyine's dev tooling is in its [project.optional-dependencies].dev group." >&2; \
	    echo "Install it from $(PYINE_ROOT):" >&2; \
	    echo "  uv sync --extra dev" >&2; \
	    exit 1; \
	fi
	cd $(PYINE_ROOT) && $(PYTHON) -m pytest $(CURDIR)/tests -v

check: lint test ## Run lint + test (CONTRIBUTING.md quality gate; required before push)
