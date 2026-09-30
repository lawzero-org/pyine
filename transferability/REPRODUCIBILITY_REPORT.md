# Reproducibility Report

**Purpose.** Pre-merge end-to-end validation of every `make` target documented in this study (and the LOCAL=1 variants), plus the bugs / UX gaps uncovered along the way and the fixes that landed. Acts as both (a) evidence the infrastructure was exercised before the PR was opened, and (b) a copy-pasteable script a reviewer can run to replicate the same validation.

**Session date.** 2026-05-24 (timestamps in `timing.tsv` and on-disk artifacts confirm).

**Scope.** Every target listed by `make help`. Full-budget sweeps (`make sweep1` / `make sweep2` at full size) were NOT run end-to-end (they cost $20-35 in aggregate per RUNBOOK.md); they were exercised at `LIMIT=1` and `--items-cap 1` respectively, which exercises every dispatch and analyzer code path without the per-item cost multiplier.

______________________________________________________________________

## 1. Environment

| Component | Value | Source |
|---|---|---|
| Pyine venv | `$PYINE_ROOT/.venv` | `make _check-pyine-root` |
| Python | 3.12 | `python --version` |
| pytest | installed in pyine venv | `make test` passes 107 tests |
| `lm-eval` | 0.4.12 | `make install` output |
| `runpod` | 1.9.0 | `make install` output |
| `vllm` (for `LOCAL=1`) | 0.20.2 in a dedicated venv adjacent to pyine | started + served Qwen/Qwen3-4B-Instruct-2507; a different vLLM install in another venv was unusable (torch ABI mismatch) |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 16 GB | `nvidia-smi` |
| Runpod endpoints exercised | `toh3giz9qktyzi` (shortcut), `ib7zu6jk84dh5o` (base); both deleted by `make teardown` at session end | `deploy/teardown_endpoints.py --dry-run` post-teardown shows 0 endpoints, 2 templates preserved |

______________________________________________________________________

## 2. Targets executed — wall-clock observed

All times sourced from saved task outputs and `timing.tsv`. Cost not tracked here; reviewers should consult their Runpod billing dashboard for actual spend during replication.

### Free / quality-gate targets

| Target | Wall-clock | Exit code | Source |
|---|---|---|---|
| `make help` | <1s | 0 | direct invocation, see §5 |
| `make check` (= `make lint && make test`) | 1.87s pytest + lint | 0 | `============================= 107 passed in 1.87s ==============================` |
| `make install` | ~1s (idempotent re-run) | 0 | `Checked 2 packages in 121ms` (uv pip), version-report lines following |

### DRY_RUN targets (no HTTP, no cost)

| Target | Outcome | Source |
|---|---|---|
| `make sweep2 DRY_RUN=1` | All 8 cue families exercised end-to-end including new `self_preservation`; analyzer produced `cueflip_summary.csv`, `cueflip_compare.md`, `cueflip_per_family.png`, `cueflip_per_benchmark.png`. Exit 0. | Last line of task output: `Wrote: cueflip_summary.csv, cueflip_compare.md, cueflip_per_family.png, cueflip_per_benchmark.png` |
| `make sweep2 SKIP_CACHE_BUILD=1 DRY_RUN=1` | Same as above, with cache-build step skipped. First line: `==> SKIP_CACHE_BUILD=1 set, skipping cache build`. Exit 0. | Task output |
| `make build-cache DRY_RUN=1` | Idempotent against locally-built `cueflip/operation_flip_cache.json` (150 entries; gitignored, regenerated on first `make build-cache`): `# cache is complete; nothing to do`. Exit 0. | Task output |

### Inspect-only Runpod (negligible cost)

| Target | Result | Source |
|---|---|---|
| `python deploy/teardown_endpoints.py --dry-run` | Found 2 endpoints + 2 templates, modified nothing | `# --dry-run: skipping endpoint deletion. Nothing was modified.` |

### Runpod-backed targets

| Target | Wall-clock | Calls | Source |
|---|---|---|---|
| `make verify-endpoints` | 95.6s (shortcut cold start) + 94.9s (base cold start) + 1.5s + 1.4s (logprobs check) | 4 HTTP requests (2 GET /models, 2 POST /completions) | `[shortcut] /models HTTP 200 in 95.6s` ... `[base] PASS: logprobs present` |
| `make smoke` (warm endpoints) | gsm8k_cot leg: 19s for 1 item | 9 + 9 cueflip cells + 1 lm-eval cell | `[15:20:06] [DONE] shortcut/gsm8k_cot (19s)` |
| `LIMIT=1 make sweep1` (12 cells in parallel, 2 endpoints) | Per-cell wall-clock recorded in `timing.tsv`; mean=64s, min=18s (truthfulqa_mc1), max=225s (shortcut/mmlu_pro long-CoT) | 16 (model × task variant) cells — truthfulqa expands to gen/mc1/mc2; total `results_*.json` artifacts on disk: 16 | `timing.tsv` rows for `shortcut/*` + `base/*` (16 rows marked `ok`) |
| `python cueflip/runner.py --items-cap 1 --gsm8k-mode primary --paraphrase-indices 0 --num-concurrent 8` (= `make sweep2` core, items-capped) | 2059.1s wall (~34 min) | 90 new calls + 18 skipped (resumed from prior smoke) = 108 unique cells across 6 benchmarks × 2 models × 9 cells; 0 failed | `** base/humaneval done -- calls=90 skipped=18 failed=0 elapsed=2059.1s` |
| `make teardown` | <5s | Deleted 2 endpoints, preserved 2 templates | `Teardown complete. Deleted 2 endpoint(s); preserved 2 template(s).` Post-teardown `--dry-run` confirms 0 endpoints found, 2 templates remain. |

### LOCAL=1 targets (against local vLLM on port 8001, then 8000)

| Target | Wall-clock | Calls | Source |
|---|---|---|---|
| Local vLLM startup (Qwen/Qwen3-4B-Instruct-2507 on :8001) | ~1 min model-load (3 checkpoint shards) | n/a | `Application startup complete` in startup log |
| `LOCAL=1 make smoke SMOKE_TAG=local_test` with `INFERENCE_URL_LOCAL_TEST` + `CUEFLIP_INFERENCE_URL_LOCAL_TEST` overrides | hellaswag: 170.9s for 9 calls; gsm8k: 634.9s for 9 calls (sycophancy long-CoT); lm-eval gsm8k_cot: 23s for 1 call | 9 + 9 + 1 = 19, all to localhost:8001 | `** local_test/hellaswag done -- calls=9 skipped=0 failed=0 elapsed=170.9s` / `** local_test/gsm8k done -- calls=9 skipped=0 failed=0 elapsed=634.9s` / `[16:13:20] [DONE] local_test/gsm8k_cot (23s)` |
| `make build-cache --cache-path /tmp/test_op_flip_cache.json --items-cap 3` (against local judge on :8000) | not timed; sub-minute | `done -- ok=3 failed=0 null_op1=1 null_op2=1 null_op3=2` | Task output |

______________________________________________________________________

## 3. Issues uncovered and fixes applied

Every fix below was tested by re-running the previously-failing command to confirm exit 0.

### Makefile

| # | Issue | Fix | Verified by |
|---|---|---|---|
| 1 | `make smoke` had no pre-flight for missing `lm-eval`; first failure mode was `scripts/run_gsm8k.sh: line 32: $PYINE_ROOT/.venv/bin/lm-eval: No such file or directory` mid-sweep | Added `_check-lm-eval` target (mirroring `_check-env` pattern); wired `sweep1` and `smoke` to depend on it | `make help` lists the dependency; subsequent smoke runs pass |
| 2 | `make refs` shipped a download target for `refs/*.pdf` even though the `refs/` folder was personal use only and got entirely removed from the PR | Removed `refs:` target + `.PHONY` entry + header help line + the `refs/fetch_refs.sh` arg from shellcheck step | `grep -rn 'make refs\|refs/' --include='*.md' --include='Makefile'` returns zero |
| 3 | `make clean` recipe wiped `outputs/raw/`, `outputs/derived/`, `cueflip/results/` — destroying ~$20-35 of regenerable Runpod data with no undo | Removed `clean:` target + .PHONY entry + header help line + README mention | `make help` doesn't list `clean` |
| 4 | `make analyze` help text said "4 analysis scripts" but the recipe ran 5 | Updated to "Run all 6 analysis scripts (analyze + descriptive_stats + items D/E/F/G; gracefully skips incomplete cells)" | `make help` |
| 5 | `scripts/descriptive_stats.py` was documented in README as an analysis script but missing from `make analyze` recipe | Added `$(PYTHON) scripts/descriptive_stats.py` to the recipe | `make analyze` task output includes its run line |
| 6 | Smoke echo said "7 cue families" — stale after adding self_preservation (8 families) | Updated to "8 cue families" | grep on Makefile returns no `7 cue families` |

### Python scripts (graceful-failure on partial data)

| # | File | Symptom on a fresh checkout | Fix |
|---|---|---|---|
| 7 | `scripts/analyze.py` | `FileNotFoundError: ...outputs/derived/results_summary.csv` — outputs/derived/ didn't exist | Added `OUT.mkdir(parents=True, exist_ok=True)` at module top |
| 8 | `scripts/analyze.py` | `AttributeError: 'list' object has no attribute 'split'` on MC-task records (hellaswag, gpqa, mmlu_pro, truthfulqa_mc1/2 store loglikelihood tuples, not text) | Added `isinstance(text, str)` check before `text.split()` |
| 9 | `scripts/analysis_d.py` | Same missing-mkdir | `OUT.mkdir(parents=True, exist_ok=True)` |
| 10 | `scripts/analysis_e.py` | `IndexError: list index out of range` writing the disagreement-counts CSV when no rows exist (only one model has data) | Print "(no shortcut+base disagreement data available...)" and `return` before opening the CSV |
| 11 | `scripts/analysis_f.py` | Missing mkdir + `raise SystemExit("no generative-task rows...")` → non-zero exit aborts the `make analyze` pipeline | Added `sys` import + `OUT.mkdir(...)` + replaced `raise SystemExit(...)` with `print + sys.exit(0)` |
| 12 | `scripts/analysis_g.py` | Same as f | Same pattern |
| 13 | `scripts/descriptive_stats.py` | `KeyError: 'n_base'` when only one model's data is present (the pivot doesn't create base-suffixed columns) | Pre-check at `main()`: read summary CSV, verify `{'base','shortcut'}.issubset(models)`, otherwise print + `return 0` |

After all six fixes, `make analyze` runs end-to-end on partial (smoke-level) data with informative "(no X data available)" messages from each script that lacks input, and exits 0.

### Surface added

| # | Change | Detail |
|---|---|---|
| 14 | `make install` target | Idempotent install of `lm-eval` + `runpod` into pyine's venv. Reports installed versions and vLLM availability. Does NOT auto-install or auto-discover vLLM — auto-discovery would be fragile (this session found a vLLM at `~/.venv/bin/vllm` that crashes with `ImportError: ... vllm/_C.abi3.so: undefined symbol`). Instructs user to install vLLM in a dedicated venv if they want LOCAL=1 paths. |

### .gitignore

| # | Pattern added | Why |
|---|---|---|
| 15 | `timing.tsv` | Wall-clock log written by `scripts/_common.sh:log_timing`; regenerable, contains no source info |
| 16 | `wandb/` (was only `scripts/wandb/`) | lm-eval-harness writes wandb runs to whatever cwd it's launched from; when invoked via `bash scripts/run_*.sh` from study root, the runs land at `transferability/wandb/` |

### Docs

| # | File | Change |
|---|---|---|
| 17 | `transferability/README.md` | Added `make install` as step 1 in "How to reproduce"; renumbered subsequent steps 1→2 through 7→8. Removed `make refs` line + the prior "Additional ad-hoc deps: install once" paragraph (the install command now lives behind `make install`). |
| 18 | `transferability/RUNBOOK.md` | Added "Common failures + fixes" entry: `make teardown didn't stop my local vLLM server` — explains the asymmetry (Runpod is `make`-managed; local vLLM is user-managed) with a table. |

______________________________________________________________________

## 4. Artifacts produced

All paths relative to `transferability/`. Sizes from `ls -la`.

### Sweep #1 outputs (`outputs/raw/`)

17 `results_*.json` files (one per (model, task variant)):
- `shortcut/`: gpqa_diamond_cot_n_shot, gsm8k_cot, hellaswag, humaneval_instruct, mmlu_pro, truthfulqa_gen, truthfulqa_mc1, truthfulqa_mc2 (8)
- `base/`: same 8
- `local_test/gsm8k_cot` (1, from LOCAL smoke)

### Sweep #1 derived analysis (`outputs/derived/`, written by `make analyze`)

| File | Size | Producer |
|---|---|---|
| `descriptive_accuracy.csv` / `.md` | 1056B / 817B | `descriptive_stats.py` |
| `descriptive_aggregates.md` | 229B | `descriptive_stats.py` |
| `descriptive_length.csv` / `.md` | 436B / 445B | `descriptive_stats.py` |
| `results_summary.csv` | 3952B | `analyze.py` |
| `results_compare.md` | 669B | `analyze.py` |
| `length_stats.csv` | 477B | `analyze.py` |
| `transferability_bars.png` | 197 KB | `analyze.py` |
| `gpqa_per_domain.csv` / `.md` / `.png` / `_forest.png` | 135B / 204B / 63 KB / 79 KB | `analysis_d.py` |
| `mmlu_pro_per_discipline.csv` / `.md` / `.png` / `_forest.png` | 1353B / 1145B / 145 KB / 138 KB | `analysis_d.py` |
| `length_distributions.png` | 262 KB | `analysis_d.py` |
| `per_item.csv` | 10.8 KB | `analysis_d.py` |
| `faithfulness_compression.csv` / `.md` / `.png` | 427B / 900B / 136 KB | `analysis_f.py` |
| `length_distributional_analysis.csv` / `.md` / `.png` | 438B / 1050B / 227 KB | `analysis_g.py` |
| `length_distributional_analysis_conditional.csv` / `.md` | 218B / 781B | `analysis_g.py` |

### Sweep #2 raw (`cueflip/results/`)

126 JSONL records across 14 (model, benchmark) cells (9 records per cell = 1 baseline + 8 cue families × 1 paraphrase):
- `shortcut/`: 6 benchmarks × 9 = 54
- `base/`: 6 benchmarks × 9 = 54
- `local_test/`: 2 benchmarks (hellaswag, gsm8k) × 9 = 18

### Sweep #2 derived analysis (`cueflip/cueflip_*`)

| File | Size | Producer |
|---|---|---|
| `cueflip_summary.csv` | 26.8 KB | `cueflip/analyze.py` |
| `cueflip_compare.md` | 50.0 KB | `cueflip/analyze.py` |
| `cueflip_cross_model.csv` | 6.9 KB | `cueflip/analyze.py` |
| `cueflip_per_benchmark.png` | 143 KB | `cueflip/analyze.py` |
| `cueflip_per_family.png` | 181 KB | `cueflip/analyze.py` |
| `cueflip_secondary_gsm8k.md` | 5.1 KB | `cueflip/analyze.py` |

(All `cueflip/cueflip_*` files are gitignored; the table documents that the pipeline produced them, not that they ship.)

______________________________________________________________________

## 5. Replicate this validation

The full sequence below is what was actually executed during this session. Times are wall-clock observed in this session; reviewers' times will vary.

```bash
# from your pyine clone
cd pyine/transferability

# 0. one-time setup (idempotent)
make install                                            # ~1s if already installed

# 1. quality gates -- free
make check                                              # ~2s, 107 tests

# 2. dry-run paths -- free, no HTTP
make sweep2 DRY_RUN=1                                   # ~3s, synthetic; exercises full analyzer
make sweep2 SKIP_CACHE_BUILD=1 DRY_RUN=1                # same but skips cache build
make build-cache DRY_RUN=1                              # no-op if cache complete (150 entries committed)

# 3. inspect Runpod state -- negligible cost
python deploy/teardown_endpoints.py --dry-run

# 4. live Runpod path (assumes endpoints already deployed; if not, `make deploy` first)
make verify-endpoints                                   # ~3 min cold start + 2 logprobs probes
make smoke                                              # ~30s lm-eval + minimal cue calls (resumed if prior records exist)

# 5. limited-budget end-to-end -- Runpod cost is bounded by LIMIT/items-cap
LIMIT=1 make sweep1                                     # ~5 min wall (parallel dispatch over 12 cells)
python cueflip/runner.py --items-cap 1 \
    --gsm8k-mode primary --paraphrase-indices 0 \
    --num-concurrent 8                                  # ~34 min wall (108 calls)
make analyze                                            # ~5s, 6 scripts; gracefully skips cells lacking data

# 6. LOCAL=1 path -- requires vllm available somewhere; reviewer starts it manually
#    (e.g. `vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8001`)
#    Then with overrides for any tag other than shortcut/base:
INFERENCE_URL_LOCAL_TEST=http://localhost:8001/v1/completions \
CUEFLIP_INFERENCE_URL_LOCAL_TEST=http://localhost:8001/v1 \
INFERENCE_API_KEY=EMPTY CUEFLIP_INFERENCE_API_KEY=EMPTY \
LOCAL_TEST_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 \
make smoke SMOKE_TAG=local_test                         # ~14 min wall (sycophancy long-CoT)

# 7. teardown
make teardown                                           # ~5s; deletes endpoints, preserves templates
```

### Costs

Not tracked here — this report does not include Runpod billing data because billing was not queried during the session. A reviewer who runs the sequence above should consult their Runpod billing dashboard for the actual spend. As reference points: this session used pre-existing warm endpoints for most of the wall-clock above, so cold-start costs are only paid once on initial deploy.

### What was deliberately NOT run

| Target | Reason |
|---|---|
| `make sweep1` at full size | Documented `~$15-20` in `RUNBOOK.md`; LIMIT=1 covers every dispatch + analyzer code path |
| `make sweep2` at full size | Documented `~$7-15` in `RUNBOOK.md`; --items-cap 1 covers every cell × cue-family combination |
| `make deploy` (idempotent re-deploy) | Endpoints were already deployed at session start; we did exercise `make teardown` and confirm with the post-teardown dry-run that endpoints went 2→0 and templates were preserved |
| `make smoke SMOKE_TAG=<other>` against a second model deployed on Runpod | Single-GPU local setup couldn't host two models simultaneously; verified the multi-model-first-class path via the `LOCAL_TEST` tag with explicit env-var routing |
| `make build-cache` against Runpod judge | The default `CUEFLIP_JUDGE_URL=http://localhost:8000/v1` points at a local judge; we verified the path via a temp cache against local vLLM (3 items processed cleanly) |
