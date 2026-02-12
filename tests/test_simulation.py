"""Tests for the simulation environment."""

import numpy as np
import pytest

from prediction_marl.market import LMSRMarket
from prediction_marl.agents import ZeroIntelligenceAgent, QLearningAgent
from prediction_marl.simulation import (
    generate_signals,
    run_episode,
    run_simulation,
)


class TestGenerateSignals:
    def test_output_shape(self):
        signals = generate_signals(10, 0.7)
        assert signals.shape == (10,)

    def test_signals_clipped(self):
        signals = generate_signals(100, 0.5, noise_std=10.0, rng=np.random.default_rng(0))
        assert np.all(signals >= 0.01)
        assert np.all(signals <= 0.99)

    def test_signals_centered_on_true_prob(self):
        signals = generate_signals(10000, 0.7, noise_std=0.1, rng=np.random.default_rng(42))
        assert np.mean(signals) == pytest.approx(0.7, abs=0.01)

    def test_reproducibility(self):
        s1 = generate_signals(10, 0.5, rng=np.random.default_rng(123))
        s2 = generate_signals(10, 0.5, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(s1, s2)


class TestRunEpisode:
    def test_basic_episode_runs(self):
        market = LMSRMarket(b=2.0)
        agents = [
            ZeroIntelligenceAgent(i, signal=0.6 + i * 0.05)
            for i in range(5)
        ]
        result = run_episode(market, agents, true_prob=0.7, n_rounds=10)
        assert result.true_prob == 0.7
        assert result.realized_outcome in ("yes", "no")
        assert 0.0 <= result.final_price <= 1.0
        assert len(result.price_trajectory) == 11  # initial + 10 rounds

    def test_price_moves_toward_true_prob(self):
        """With many agents and rounds, price should converge."""
        market = LMSRMarket(b=2.0)
        agents = [
            ZeroIntelligenceAgent(i, signal=0.7, trade_size=0.5)
            for i in range(10)
        ]
        result = run_episode(
            market, agents, true_prob=0.7, n_rounds=50,
            rng=np.random.default_rng(42),
        )
        # Final price should be closer to 0.7 than the initial 0.5
        assert abs(result.final_price - 0.7) < abs(0.5 - 0.7)

    def test_pnl_computed_for_all_agents(self):
        market = LMSRMarket(b=2.0)
        agents = [ZeroIntelligenceAgent(i, signal=0.7) for i in range(3)]
        result = run_episode(market, agents, true_prob=0.7, n_rounds=10)
        assert len(result.agent_pnls) == 3
        for aid in range(3):
            assert aid in result.agent_pnls


class TestRunSimulation:
    def test_fixed_true_prob(self):
        market = LMSRMarket(b=2.0)
        agents = [ZeroIntelligenceAgent(i, signal=0.5) for i in range(5)]
        results = run_simulation(
            market, agents, n_episodes=10, n_rounds=20, true_prob=0.7,
        )
        assert len(results) == 10
        assert all(r.true_prob == 0.7 for r in results)

    def test_varying_true_prob(self):
        market = LMSRMarket(b=2.0)
        agents = [ZeroIntelligenceAgent(i, signal=0.5) for i in range(5)]
        results = run_simulation(
            market, agents, n_episodes=20, n_rounds=20,
            true_prob_range=(0.3, 0.8),
        )
        true_probs = [r.true_prob for r in results]
        assert min(true_probs) >= 0.3
        assert max(true_probs) <= 0.8
        # Should see some variation
        assert len(set(true_probs)) > 1

    def test_reproducibility(self):
        market = LMSRMarket(b=2.0)
        agents1 = [ZeroIntelligenceAgent(i, signal=0.5) for i in range(5)]
        agents2 = [ZeroIntelligenceAgent(i, signal=0.5) for i in range(5)]

        r1 = run_simulation(market, agents1, n_episodes=5, seed=99)
        r2 = run_simulation(market, agents2, n_episodes=5, seed=99)

        for a, b in zip(r1, r2):
            assert a.final_price == pytest.approx(b.final_price)
            assert a.realized_outcome == b.realized_outcome
