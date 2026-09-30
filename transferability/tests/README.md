# Tests for the transferability study

Pure-function unit tests for the helpers most likely to be reused or
refactored by anyone extending this study (107 tests, ~2 s):

- `_newcombe_diff_ci` (scripts/analysis_d.py) -- Newcombe-Wilson 95% CI for
  differences of proportions. Used by the GPQA per-domain and MMLU-Pro
  per-discipline analyses; the formulas are easy to misimplement.
- `bootstrap_auc` (scripts/analysis_g.py) -- stratified bootstrap ROC-AUC.
  Tests the stratification invariant (class sizes preserved) and the
  directional convention (score = -length, shorter-shortcut -> AUC > 0.5).
- `parse_answer_letter` and `parse_answer_numeric` (cueflip/runner.py) --
  regex extraction for multiple-choice letters and free-form numerics.
- The 6 GSM8K wrong-numeric strategies + dispatcher + normalizer
  (`cueflip/perturbations.py`).
- HumanEval docstring-cue rendering and the `cueflip/code_eval.py` subprocess
  sandbox (real subprocess execution, no mocks). The docstring-cue tests are
  parametrized over `cueflip/cue_templates.CUE_TEMPLATES_DOCSTRING` so any new
  cue family added to that dict is auto-covered.

## Running

From `pyine/transferability/`:

```bash
make test
```

(`make test` checks PYINE_ROOT resolves to a real pyine repo and runs
`pytest tests -v` against pyine's venv. Pyine's root CI workflows
(`ci.yml` + `ci-full.yml`) run `make -C transferability check`, so these
tests are covered automatically on every PR.)
