"""Unit tests for cueflip/perturbations.py.

Covers:
  - perturbations.normalize_numeric_str       (string canonicalization for equality)
  - perturbations.pick_plus_minus_10          (primary GSM8K wrong-numeric)
  - perturbations.pick_off_by_one_digit       (secondary, pure-function)
  - perturbations.pick_magnitude_shift        (secondary, pure-function)
  - perturbations.pick_op_flip                (secondary, cache-backed)
  - perturbations.pick_suggested_numeric      (dispatcher)
"""

from __future__ import annotations

import perturbations
import pytest


class TestNormalizeNumericStr:
    def test_int(self) -> None:
        assert perturbations.normalize_numeric_str("42") == "42"

    def test_float_that_is_integer(self) -> None:
        assert perturbations.normalize_numeric_str("42.0") == "42"

    def test_genuine_float(self) -> None:
        assert perturbations.normalize_numeric_str("42.5") == "42.5"

    def test_strips_commas(self) -> None:
        assert perturbations.normalize_numeric_str("10,000") == "10000"

    def test_strips_dollar_sign(self) -> None:
        assert perturbations.normalize_numeric_str("$42") == "42"

    def test_negative(self) -> None:
        assert perturbations.normalize_numeric_str("-7") == "-7"

    def test_whitespace(self) -> None:
        assert perturbations.normalize_numeric_str("  42  ") == "42"

    def test_non_numeric_returns_none(self) -> None:
        assert perturbations.normalize_numeric_str("abc") is None

    def test_empty_returns_none(self) -> None:
        assert perturbations.normalize_numeric_str("") is None

    def test_none_input(self) -> None:
        assert perturbations.normalize_numeric_str(None) is None


class TestPickPlusMinus10:
    def test_returns_value_in_range(self) -> None:
        out = perturbations.pick_plus_minus_10("qid_x", "42", None)
        assert out is not None
        assert 32 <= int(out) <= 52
        assert out != "42"

    def test_deterministic(self) -> None:
        first = perturbations.pick_plus_minus_10("qid_x", "42", None)
        second = perturbations.pick_plus_minus_10("qid_x", "42", None)
        assert first == second

    def test_different_qid_different_seed(self) -> None:
        # different qids should usually pick different candidates, though collisions are possible;
        # we just check the function isn't a constant.
        seen = {perturbations.pick_plus_minus_10(f"qid_{idx}", "42", None) for idx in range(20)}
        assert len(seen) > 1

    def test_excludes_baseline(self) -> None:
        # if the only excluded candidate is baseline, output must differ
        out = perturbations.pick_plus_minus_10("qid_x", "42", "43")
        assert out != "43"

    def test_negative_gold(self) -> None:
        out = perturbations.pick_plus_minus_10("qid_x", "-5", None)
        assert out is not None
        assert -15 <= int(out) <= 5
        assert out != "-5"


class TestPickOffByOneDigit:
    def test_single_digit_changes(self) -> None:
        out = perturbations.pick_off_by_one_digit("qid_x", "42", None)
        # for "42" candidates are: 41, 43, 32, 52 (digit-by-digit +/-1 in 0-9 range)
        assert out in {"41", "43", "32", "52"}

    def test_deterministic(self) -> None:
        first_call = perturbations.pick_off_by_one_digit("qid_x", "42", None)
        second_call = perturbations.pick_off_by_one_digit("qid_x", "42", None)
        assert first_call == second_call

    def test_zero_digit_only_increments(self) -> None:
        # "10" -> first digit 1 +/-1 = 0,2 (but leading 0 normalizes); second 0 +/-1 = 1,-impossible
        # candidates include "20", "11", and the leading-zero-normalized "00"=0 which equals gold... etc
        out = perturbations.pick_off_by_one_digit("qid_x", "10", None)
        assert out is not None
        assert out != "10"

    def test_negative_value(self) -> None:
        out = perturbations.pick_off_by_one_digit("qid_x", "-5", None)
        assert out is not None
        assert out != "-5"


class TestPickMagnitudeShift:
    def test_returns_x10_x100_or_div10(self) -> None:
        out = perturbations.pick_magnitude_shift("qid_x", "42", None)
        # candidates: 420, 4200, 4.2
        assert out in {"420", "4200", "4.2"}

    def test_deterministic(self) -> None:
        first_call = perturbations.pick_magnitude_shift("qid_x", "42", None)
        second_call = perturbations.pick_magnitude_shift("qid_x", "42", None)
        assert first_call == second_call

    def test_zero_gold_excludes_all(self) -> None:
        # 0 * 10 = 0, 0 * 100 = 0, 0 / 10 = 0 -- all equal gold, no candidates
        assert perturbations.pick_magnitude_shift("qid_x", "0", None) is None


class TestPickOpFlip:
    def test_returns_cached_value(self) -> None:
        cache = {"qid_x": {"op1": "21", "op2": "63", "op3": "84"}}
        assert perturbations.pick_op_flip("qid_x", "42", None, 1, cache) == "21"
        assert perturbations.pick_op_flip("qid_x", "42", None, 2, cache) == "63"
        assert perturbations.pick_op_flip("qid_x", "42", None, 3, cache) == "84"

    def test_null_in_cache_returns_none(self) -> None:
        cache = {"qid_x": {"op1": "21", "op2": None, "op3": None}}
        assert perturbations.pick_op_flip("qid_x", "42", None, 1, cache) == "21"
        assert perturbations.pick_op_flip("qid_x", "42", None, 2, cache) is None

    def test_missing_qid_returns_none(self) -> None:
        cache = {"other_qid": {"op1": "21", "op2": None, "op3": None}}
        assert perturbations.pick_op_flip("qid_x", "42", None, 1, cache) is None

    def test_cache_value_equals_gold_returns_none(self) -> None:
        """Defensive: if the cache builder violated the no-equals-gold constraint,
        the picker drops the value rather than passing it through."""
        cache = {"qid_x": {"op1": "42", "op2": None, "op3": None}}
        assert perturbations.pick_op_flip("qid_x", "42", None, 1, cache) is None

    def test_normalization_applied(self) -> None:
        """Cached values are normalized before return."""
        cache = {"qid_x": {"op1": "21.0", "op2": "10,000", "op3": None}}
        assert perturbations.pick_op_flip("qid_x", "42", None, 1, cache) == "21"
        assert perturbations.pick_op_flip("qid_x", "42", None, 2, cache) == "10000"


class TestPickSuggestedNumeric:
    def test_dispatcher_routes_to_each_strategy(self) -> None:
        cache = {"qid_x": {"op1": "21", "op2": "63", "op3": "84"}}
        for strategy in perturbations.SECONDARY_STRATEGIES:
            out = perturbations.pick_suggested_numeric("qid_x", "42", None, strategy, op_flip_cache=cache)
            assert out is not None, f"strategy {strategy} returned None"

    def test_primary_strategy_constant(self) -> None:
        assert perturbations.PRIMARY_STRATEGY == "plus_minus_10"
        assert perturbations.PRIMARY_STRATEGY in perturbations.SECONDARY_STRATEGIES

    def test_op_flip_without_cache_raises(self) -> None:
        with pytest.raises(ValueError):
            perturbations.pick_suggested_numeric("qid_x", "42", None, "op_flip_1", op_flip_cache=None)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            perturbations.pick_suggested_numeric("qid_x", "42", None, "nonexistent_strategy")
