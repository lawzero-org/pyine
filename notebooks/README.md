# PyINE Notebooks Overview

This folder collects hands-on, exploratory, and analysis notebooks containing key parts of the
PyINE workflow: tracing Python code, deriving deltas, inspecting datasets, browsing prompt-chain
annotation results, analyzing training runs and benchmarks, and repackaging PyINE-v1 artifacts for
HuggingFace.

Run any notebook after installing the project into a Python 3.12+ environment (see the
[project root README](../README.md)). You can use Jupyter Lab/Notebook or an IDE kernel. Many
notebooks expect datasets under `<REPO_ROOT>/data/` and optional environment variables configured
via `.env`.

## Demos & tutorials

Entry-point notebooks that walk through core PyINE APIs end-to-end.

- [`code_deltas_demo.ipynb`](./code_deltas_demo.ipynb): derive "deltas" (stepwise variable
  state changes) from execution traces and examine their structure and representation.
- [`code_edits_demo.ipynb`](./code_edits_demo.ipynb): demonstrate using LLM prompts to generate
  code edits (e.g., injecting bugs, hints, adding stubs, refactoring) and visualize diffs between
  original and modified code.
- [`code_execution_demo.ipynb`](./code_execution_demo.ipynb): end-to-end demonstration of
  instrumenting and tracing of Python code, and asking various LLMs (including reasoning models)
  to predict execution outcomes.
- [`data_loading_demo.ipynb`](./data_loading_demo.ipynb): configure datamodules, load samples
  from trace datasets, and prepare data for reinforcement learning or supervised fine-tuning.
- [`prompt_result_db_demo.ipynb`](./prompt_result_db_demo.ipynb): programmatic examples for
  `pyine.prompts.result_db` utils showing how to store, fetch, and manage LLM prompt results
  with automatic deduplication.

## Source & trace dataset exploration

EDA on raw source datasets, traces, deltas, and sample-builder outputs.

- [`eval_export_explorer.ipynb`](./eval_export_explorer.ipynb): inspect a completed evaluation export
  (v1 or v2, public or author): split sizes, labels, query groups, categories, text budgets, the
  displayed-return shortcut, and rendered examples. Set `EXPORT_DIR` or `PYINE_EVAL_EXPORT_DIR` and
  run all cells.
- [`taco_source_data_viz.ipynb`](./taco_source_data_viz.ipynb): quick tour and visualization of
  the original TACO source dataset, including solution counts, lengths, difficulty distributions,
  and tag analysis.
- [`code_cluster_analysis.ipynb`](./code_cluster_analysis.ipynb): analyze clustering of code
  solutions based on variable/function name similarity, visualize keyword distributions, and
  explore solution families (for keyword-trigger experiments).
- [`trace_datasets_eda.ipynb`](./trace_datasets_eda.ipynb): exploratory data analysis of trace
  datasets covering problem/solution coverage, test retention, augmentation statistics, and
  tag distributions across dataset versions.
- [`trace_datasets_viz.ipynb`](./trace_datasets_viz.ipynb): visualization-focused walkthrough
  of trace and delta datasets, including tag frequencies, step lengths, and delta type
  distributions.
- [`sample_builder_outputs_eda.ipynb`](./sample_builder_outputs_eda.ipynb): explore sample
  builder configurations and outputs, analyzing token distributions, predict type
  distributions, code type selections, and tag frequencies.
- [`sample_prompt_length_analysis.ipynb`](./sample_prompt_length_analysis.ipynb): analyze
  prompt token length distributions for different HuggingFace models, identify samples
  exceeding context limits, and visualize overflow patterns.

## Prompt result & validation tools

Browsers and analyses backed by the framework's `PromptResultDB`.

- [`prompt_result_viewer.ipynb`](./prompt_result_viewer.ipynb): interactive browser for the
  framework's prompt results database, with filtering by prompt name/version/tags and
  visualization of annotation coverage.
- [`validation_verdict_analysis.ipynb`](./validation_verdict_analysis.ipynb): visualize hint
  analysis verdict statistics from the trace annotation validator app, scoped to
  `validation/misleading` records in the `PromptResultDB` (provider/model/verdict tag breakdowns
  over time).

## Training run & checkpoint analysis

Inspect W&B runs, reward logs, and per-checkpoint evaluation outputs from RL training.

- [`rl_run_analysis.ipynb`](./rl_run_analysis.ipynb): analyze W&B runs from RL training
  experiments (GRPOTrainer via `hf_trainer`). Shows reward metrics, parsing statistics, training
  progress, and sample completions, selectable by URL/ID or filter-based search.
- [`pregen_outputs_explorer.ipynb`](./pregen_outputs_explorer.ipynb): load and browse LMDB
  datasets written by `DiskRewardLogger` during RL training (rewards, prompts, completions),
  with optional fnmatch key-prefix filtering.
- [`checkpoint_progression_analysis.ipynb`](./checkpoint_progression_analysis.ipynb): plot how
  key code-execution evaluation metrics evolve across RL training checkpoints, with optional
  base-model reference lines drawn from a separate eval root.

## Benchmark & eval analysis

Aggregate, visualize, and cost out evaluation results.

- [`benchmark_code_exec_analysis.ipynb`](./benchmark_code_exec_analysis.ipynb): fetch and
  visualize code execution eval metrics from W&B or local pickle, including accuracy
  breakdowns by code type, predict type, complexity, and keyword presence.
- [`benchmark_correctness_analysis.ipynb`](./benchmark_correctness_analysis.ipynb): fetch and
  visualize guardrail correctness eval metrics, including AUROC, ROC/PR curves, operating
  point analysis, category breakdowns, and difficulty-conditioned performance.
- [`benchmark_cost_analysis.ipynb`](./benchmark_cost_analysis.ipynb): estimate the "compute
  tax" of various oversight methods (probes, classifiers, LLM judges, debate) relative to the
  target predictor, combining FLOPs- and token-based comparisons into unified tables and plots.

## Hugging Face repackagers (`*_rpkg`)

Convert local artifacts into Parquet/HuggingFace-ready repos. Each requires
`pip install datasets huggingface_hub` (or `huggingface_hub` alone, for the model uploader)
and a prior `huggingface-cli login`.

- [`trace_datasets_rpkg.ipynb`](./trace_datasets_rpkg.ipynb): load a local trace dataset (LMDB
  shards) and push it to a HuggingFace dataset repository as Parquet, with structured columns
  for filtering and JSON columns for full execution traces.
- [`prompt_results_rpkg.ipynb`](./prompt_results_rpkg.ipynb): export prompt result records
  from the framework's SQLite database to a HuggingFace dataset repository in Parquet, with
  optional filtering by prompt name/version and tag rules.
- [`model_checkpoint_rpkg.ipynb`](./model_checkpoint_rpkg.ipynb): upload a TRL-trained model
  checkpoint directory (safetensors + config + tokenizer) to a HuggingFace model repository,
  along with a model card containing training metadata and dataset links.

## Notebook output handling

This project uses [nbstripout](https://github.com/kynan/nbstripout) as a git filter to manage
notebook outputs. This means:

- **Local notebooks**: cell outputs are preserved as you work.
- **Committed notebooks**: outputs are automatically stripped when you commit.

This keeps the repository clean (smaller diffs, no binary blobs) while letting you keep
outputs locally for reference.

**Setup.** The filter is installed automatically when you run `make install`. If you need to
set it up manually (e.g., after cloning), run:

```bash
make setup-nbstripout-tool
make setup-nbstripout
```

`setup-nbstripout-tool` creates a small dedicated tool venv under `.tools/nbstripout/.venv/`
and installs `nbstripout` there. `setup-nbstripout` then registers that binary as the git
`filter.nbstripout.clean` filter (and `diff.ipynb.textconv` for nicer diffs). If `git add`
feels slow when staging notebooks, it usually means the filter is pointed at the full project
virtualenv instead — re-running these two targets fixes that.

**Verification.** A `nbstripout --verify` pre-commit hook runs as part of `make check`, and
catches any notebook that slipped through with outputs still attached (typically a sign that
the git filter isn't installed on that clone). You can also run the same check on demand:

```bash
make check-notebooks
```

This is what CI uses to gate notebook cleanliness.

**Linting.** Notebook code cells are linted and formatted by `nbqa-ruff` / `nbqa-ruff-format`
through pre-commit, using the same rules as the rest of the repo.
