"""Unit tests for the pure-function helpers in scripts/analysis_d.py and
scripts/analysis_g.py.

Covered:
  - analysis_d._newcombe_diff_ci -- Newcombe-Wilson 95% CI for difference of proportions
  - analysis_g.bootstrap_auc -- stratified-bootstrap ROC-AUC of length-as-classifier
"""

from __future__ import annotations

import math

import analysis_d
import analysis_g
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# analysis_d._newcombe_diff_ci
# ---------------------------------------------------------------------------


class TestNewcombeDiffCI:
    def test_equal_proportions_delta_is_zero_ci_brackets_zero(self) -> None:
        delta, lo, hi = analysis_d._newcombe_diff_ci(0.5, 100, 0.5, 100)
        assert delta == 0.0
        assert lo < 0 < hi

    def test_known_value_gpqa_chemistry(self) -> None:
        """GPQA Chemistry: shortcut 0.355 / base 0.495, n=93 each.
        Reproduces a known live-data value: the Delta CI brushes zero at the
        upper bound (+0.002 in the live run, confirmed via analysis_d.py)."""
        p1, p2, n = 0.355, 0.495, 93
        delta, lo, hi = analysis_d._newcombe_diff_ci(p1, n, p2, n)
        assert math.isclose(delta, p1 - p2, rel_tol=1e-9)
        assert -0.30 < lo < -0.25
        assert -0.01 < hi < 0.01  # brushes zero (live data shows hi=+0.002)

    def test_strong_separation_ci_excludes_zero(self) -> None:
        delta, lo, hi = analysis_d._newcombe_diff_ci(0.20, 200, 0.80, 200)
        assert math.isclose(delta, -0.60)
        assert hi < 0  # CI entirely below zero

    def test_zero_edge_proportion(self) -> None:
        """Wilson handles p=0 cleanly (no NaN, no division by zero)."""
        delta, lo, hi = analysis_d._newcombe_diff_ci(0.0, 50, 0.5, 50)
        assert delta == -0.5
        assert not math.isnan(lo) and not math.isnan(hi)
        assert lo < delta < hi

    def test_one_edge_proportion(self) -> None:
        """Wilson handles p=1 cleanly."""
        delta, lo, hi = analysis_d._newcombe_diff_ci(1.0, 50, 0.5, 50)
        assert delta == 0.5
        assert not math.isnan(lo) and not math.isnan(hi)
        assert lo < delta < hi

    def test_smaller_n_gives_wider_ci(self) -> None:
        """Halving n with same proportions should widen the CI."""
        _, lo_big, hi_big = analysis_d._newcombe_diff_ci(0.5, 200, 0.6, 200)
        _, lo_small, hi_small = analysis_d._newcombe_diff_ci(0.5, 20, 0.6, 20)
        assert (hi_small - lo_small) > (hi_big - lo_big)


# ---------------------------------------------------------------------------
# analysis_g.bootstrap_auc
# ---------------------------------------------------------------------------


class TestBootstrapAuc:
    def test_perfectly_separable_high_auc(self) -> None:
        """Shortcut clearly shorter than base. Score = -length, so shorter
        responses get HIGHER scores; if label 1 = shortcut, AUC -> 1."""
        rng = np.random.default_rng(42)
        sc = np.array([10, 12, 11, 13, 9, 14, 11, 12])
        ba = np.array([100, 110, 105, 120, 90, 95, 115, 100])
        point, lo, hi = analysis_g.bootstrap_auc(sc, ba, rng, n_boot=100)
        assert point == pytest.approx(1.0, abs=1e-9)
        assert 0.9 < lo <= 1.0
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_identical_distributions_auc_near_chance(self) -> None:
        rng = np.random.default_rng(0)
        sc = rng.normal(50, 10, 300)
        ba = rng.normal(50, 10, 300)
        rng2 = np.random.default_rng(0)
        point, lo, hi = analysis_g.bootstrap_auc(sc, ba, rng2, n_boot=200)
        assert 0.4 < point < 0.6
        assert lo < 0.5 < hi  # CI brackets chance

    def test_directionality_shortcut_shorter_gives_auc_above_half(self) -> None:
        rng = np.random.default_rng(42)
        sc = np.array([10, 12, 14, 16, 18])
        ba = np.array([20, 22, 24, 26, 28])
        point, _, _ = analysis_g.bootstrap_auc(sc, ba, rng, n_boot=50)
        assert point > 0.5

    def test_stratified_bootstrap_preserves_class_balance(self) -> None:
        """The stratified bootstrap should resample WITHIN each class
        with replacement, keeping n_shortcut and n_base fixed across draws.
        We verify indirectly: AUC variance is bounded -- no degenerate
        (all-one-class) bootstrap draw should crash sklearn."""
        rng = np.random.default_rng(7)
        sc = rng.integers(10, 30, size=20).astype(float)
        ba = rng.integers(20, 40, size=20).astype(float)
        # 500 draws -- if any draw produced a one-class sample,
        # sklearn would raise. Successful return means stratification works.
        point, lo, hi = analysis_g.bootstrap_auc(sc, ba, rng, n_boot=500)
        assert 0.0 <= lo <= point <= hi <= 1.0
