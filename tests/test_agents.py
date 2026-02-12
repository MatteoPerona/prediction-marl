"""Tests for trading agents."""

import numpy as np
import pytest

from prediction_marl.market import LMSRMarket
from prediction_marl.agents import (
    ZeroIntelligenceAgent,
    QLearningAgent,
    QLearningConfig,
    ACTIONS,
)


class TestZeroIntelligenceAgent:
    def test_buys_yes_when_price_below_signal(self):
        market = LMSRMarket(b=1.0)  # price starts at 0.5
        agent = ZeroIntelligenceAgent(agent_id=0, signal=0.8)
        action = agent.choose_action(market)
        assert action == "buy_yes"

    def test_buys_no_when_price_above_signal(self):
        market = LMSRMarket(b=1.0)  # price starts at 0.5
        agent = ZeroIntelligenceAgent(agent_id=0, signal=0.2)
        action = agent.choose_action(market)
        assert action == "buy_no"

    def test_holds_when_price_near_signal(self):
        market = LMSRMarket(b=1.0)  # price starts at 0.5
        agent = ZeroIntelligenceAgent(agent_id=0, signal=0.5, threshold=0.05)
        action = agent.choose_action(market)
        assert action == "hold"

    def test_execute_updates_portfolio(self):
        market = LMSRMarket(b=2.0)
        agent = ZeroIntelligenceAgent(agent_id=0, signal=0.8, trade_size=1.0)
        agent.execute(market)
        assert agent.shares_yes == 1.0
        assert agent.cash_spent > 0

    def test_settlement_pnl_winning(self):
        market = LMSRMarket(b=2.0)
        agent = ZeroIntelligenceAgent(agent_id=0, signal=0.8, trade_size=1.0)
        agent.execute(market)  # buys YES
        pnl = agent.settlement_pnl("yes")
        # Payout is 1.0 * shares_yes, minus cost; should be positive (bought below fair value)
        assert pnl > 0

    def test_settlement_pnl_losing(self):
        market = LMSRMarket(b=2.0)
        agent = ZeroIntelligenceAgent(agent_id=0, signal=0.8, trade_size=1.0)
        agent.execute(market)  # buys YES
        pnl = agent.settlement_pnl("no")
        # Payout is 0 on YES shares, so pnl = -cost
        assert pnl < 0

    def test_reset_clears_portfolio(self):
        market = LMSRMarket(b=2.0)
        agent = ZeroIntelligenceAgent(agent_id=0, signal=0.8)
        agent.execute(market)
        agent.reset()
        assert agent.shares_yes == 0.0
        assert agent.shares_no == 0.0
        assert agent.cash_spent == 0.0


class TestQLearningAgent:
    def test_q_table_shape(self):
        config = QLearningConfig(n_price_bins=10, n_signal_bins=5)
        agent = QLearningAgent(agent_id=0, signal=0.5, config=config)
        assert agent.q_table.shape == (10, 5, 3)

    def test_choose_action_returns_valid(self):
        market = LMSRMarket(b=1.0)
        agent = QLearningAgent(agent_id=0, signal=0.5)
        action = agent.choose_action(market)
        assert action in ACTIONS

    def test_epsilon_decays_after_update(self):
        config = QLearningConfig(epsilon_start=1.0, epsilon_decay=0.9)
        agent = QLearningAgent(agent_id=0, signal=0.5, config=config)
        market = LMSRMarket(b=1.0)

        # Do a trade to create trajectory
        agent.choose_action(market)
        agent.update(reward=1.0)

        assert agent.epsilon < 1.0
        assert agent.epsilon == pytest.approx(0.9)

    def test_q_table_updates_after_episode(self):
        config = QLearningConfig(epsilon_start=1.0, alpha=0.5)
        agent = QLearningAgent(agent_id=0, signal=0.5, config=config)
        market = LMSRMarket(b=1.0)

        # Record some actions
        agent.choose_action(market)
        agent.choose_action(market)
        q_before = agent.q_table.copy()
        agent.update(reward=1.0)

        # Q-table should have changed
        assert not np.array_equal(agent.q_table, q_before)

    def test_reset_preserves_q_table(self):
        agent = QLearningAgent(agent_id=0, signal=0.5)
        market = LMSRMarket(b=1.0)
        agent.execute(market)
        agent.update(reward=1.0)
        q_after_update = agent.q_table.copy()

        agent.reset()
        assert np.array_equal(agent.q_table, q_after_update)
        assert agent.shares_yes == 0.0

    def test_set_signal(self):
        agent = QLearningAgent(agent_id=0, signal=0.3)
        agent.set_signal(0.7)
        assert agent.signal == 0.7

    def test_exploration_with_high_epsilon(self):
        """With epsilon=1.0, agent should explore (random actions)."""
        config = QLearningConfig(epsilon_start=1.0, epsilon_end=1.0)
        agent = QLearningAgent(agent_id=0, signal=0.5, config=config)
        market = LMSRMarket(b=1.0)

        np.random.seed(42)
        actions = [agent.choose_action(market) for _ in range(100)]
        # Should see all three actions with high probability
        unique_actions = set(actions)
        assert len(unique_actions) >= 2  # at minimum 2, very likely 3
