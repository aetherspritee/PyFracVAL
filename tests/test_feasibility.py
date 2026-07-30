"""Tests for the predictive feasibility boundary.

The model is a fit, so these check that it reproduces the *measured*
boundary's qualitative structure rather than pinning exact numbers -
refitting after an algorithm change should not require rewriting them.
"""

import pytest

from pyfracval import feasibility


class TestSuccessProbability:
    def test_easy_regime_is_confident(self):
        p = feasibility.estimate_success_probability(1.8, 1.0, 1.5, 128)
        assert p is not None and p > 0.9

    def test_far_past_the_boundary_is_hopeless(self):
        p = feasibility.estimate_success_probability(2.5, 1.4, 1.9, 1024)
        assert p is not None and p < 0.1

    def test_probability_decreases_with_df(self):
        probs = [
            feasibility.estimate_success_probability(df, 1.0, 1.9, 256)
            for df in (1.9, 2.1, 2.3, 2.5)
        ]
        assert all(a > b for a, b in zip(probs, probs[1:]))

    def test_polydispersity_lowers_the_ceiling(self):
        # Every measured sweep found the collapse boundary moving down in
        # Df as sigma rises.
        mono = feasibility.max_feasible_df(1.0, 1.0, 256)
        poly = feasibility.max_feasible_df(1.0, 1.9, 256)
        assert mono is not None and poly is not None
        assert poly < mono

    def test_larger_n_lowers_the_ceiling(self):
        small = feasibility.max_feasible_df(1.0, 1.9, 64)
        large = feasibility.max_feasible_df(1.0, 1.9, 1024)
        assert small is not None and large is not None
        assert large < small

    def test_lower_kf_survives_further_at_high_df(self):
        # The Df x kf interaction: at high Df, smaller kf helps.
        low_kf = feasibility.max_feasible_df(0.8, 1.9, 256)
        high_kf = feasibility.max_feasible_df(1.4, 1.9, 256)
        assert low_kf is not None and high_kf is not None
        assert high_kf < low_kf


class TestWarning:
    def test_no_warning_for_an_easy_request(self):
        assert feasibility.warn_if_difficult(1.8, 1.0, 1.5, 128) is None

    def test_warns_for_a_hard_request(self):
        msg = feasibility.warn_if_difficult(2.5, 1.4, 1.9, 1024)
        assert msg is not None
        assert "success probability" in msg

    def test_extrapolation_is_disclosed(self):
        msg = feasibility.warn_if_difficult(2.9, 1.4, 1.9, 4096)
        assert msg is not None
        assert "extrapolation" in msg

    def test_out_of_range_detection(self):
        outside = feasibility.out_of_fitted_range(2.9, 1.0, 1.5, 128)
        assert any("df" in o for o in outside)
        assert not any("kf" in o for o in outside)


class TestUnfittedModelIsHonest:
    def test_returns_none_when_not_fitted(self, monkeypatch):
        monkeypatch.setattr(feasibility, "_FITTED", False)
        assert feasibility.estimate_success_probability(1.8, 1.0, 1.5, 128) is None
        assert feasibility.max_feasible_df(1.0, 1.5, 128) is None
        assert feasibility.warn_if_difficult(2.5, 1.4, 1.9, 1024) is None
