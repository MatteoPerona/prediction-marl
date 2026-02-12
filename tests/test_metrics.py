"""Tests for evaluation metrics."""

import pytest

from prediction_marl.metrics import (
    brier_score,
    mean_brier_score,
    convergence_error,
    superadditivity_test,
    compute_all_metrics,
)
from prediction_marl.simulation import EpisodeResult


def _make_result(true_prob: float, final_price: float, outcome: str) -> EpisodeResult:
    return EpisodeResult(
        true_prob=true_prob,
        realized_outcome=outcome,
        final_price=final_price,
        price_trajectory=[0.5, final_price],
        num_trades=10,
    )


class TestBrierScore:
    def test_perfect_yes_prediction(self):
        assert brier_score(1.0, "yes") == pytest.approx(0.0)

    def test_perfect_no_prediction(self):
        assert brier_score(0.0, "no") == pytest.approx(0.0)

    def test_worst_yes_prediction(self):
        assert brier_score(0.0, "yes") == pytest.approx(1.0)

    def test_worst_no_prediction(self):
        assert brier_score(1.0, "no") == pytest.approx(1.0)

    def test_half_prediction(self):
        assert brier_score(0.5, "yes") == pytest.approx(0.25)
        assert brier_score(0.5, "no") == pytest.approx(0.25)


class TestMeanBrierScore:
    def test_mean_of_perfect_predictions(self):
        results = [
            _make_result(0.9, 1.0, "yes"),
            _make_result(0.1, 0.0, "no"),
        ]
        assert mean_brier_score(results) == pytest.approx(0.0)

    def test_mean_of_mixed_predictions(self):
        results = [
            _make_result(0.7, 0.5, "yes"),  # BS = (0.5 - 1)^2 = 0.25
            _make_result(0.3, 0.5, "no"),   # BS = (0.5 - 0)^2 = 0.25
        ]
        assert mean_brier_score(results) == pytest.approx(0.25)


class TestConvergenceError:
    def test_perfect_convergence(self):
        results = [
            _make_result(0.7, 0.7, "yes"),
            _make_result(0.3, 0.3, "no"),
        ]
        assert convergence_error(results) == pytest.approx(0.0)

    def test_nonzero_error(self):
        results = [
            _make_result(0.7, 0.6, "yes"),  # error = 0.1
            _make_result(0.3, 0.5, "no"),   # error = 0.2
        ]
        assert convergence_error(results) == pytest.approx(0.15)


class TestSuperadditivity:
    def test_market_better_than_agent(self):
        result = superadditivity_test(market_brier=0.15, best_agent_brier=0.20)
        assert result["improvement"] == pytest.approx(0.05)
        assert result["relative_improvement"] == pytest.approx(0.25)

    def test_market_worse_than_agent(self):
        result = superadditivity_test(market_brier=0.25, best_agent_brier=0.20)
        assert result["improvement"] < 0


class TestComputeAllMetrics:
    def test_basic_output(self):
        results = [
            _make_result(0.7, 0.65, "yes"),
            _make_result(0.3, 0.35, "no"),
        ]
        metrics = compute_all_metrics(results)
        assert "market_brier_score" in metrics
        assert "convergence_error" in metrics
        assert "n_episodes" in metrics
        assert metrics["n_episodes"] == 2

    def test_with_agent_signals(self):
        results = [
            _make_result(0.7, 0.65, "yes"),
            _make_result(0.3, 0.35, "no"),
        ]
        agent_signals = {
            0: [0.8, 0.2],
            1: [0.6, 0.4],
        }
        metrics = compute_all_metrics(results, agent_signals)
        assert "superadditivity" in metrics
        assert "best_agent_brier" in metrics
