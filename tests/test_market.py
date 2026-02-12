"""Tests for the LMSR market maker."""

import numpy as np
import pytest

from prediction_marl.market import LMSRMarket


class TestLMSRMarket:
    def test_initial_price_is_half(self):
        """With equal outstanding shares, p_yes should be 0.5."""
        market = LMSRMarket(b=1.0)
        assert market.price_yes() == pytest.approx(0.5)
        assert market.price_no() == pytest.approx(0.5)

    def test_prices_sum_to_one(self):
        """p_yes + p_no should always equal 1."""
        market = LMSRMarket(b=2.0)
        market.execute_trade(0, "yes", 3.0)
        market.execute_trade(1, "no", 1.5)
        assert market.price_yes() + market.price_no() == pytest.approx(1.0)

    def test_buying_yes_increases_price(self):
        market = LMSRMarket(b=1.0)
        p_before = market.price_yes()
        market.execute_trade(0, "yes", 1.0)
        assert market.price_yes() > p_before

    def test_buying_no_decreases_yes_price(self):
        market = LMSRMarket(b=1.0)
        p_before = market.price_yes()
        market.execute_trade(0, "no", 1.0)
        assert market.price_yes() < p_before

    def test_cost_is_positive_for_buy(self):
        market = LMSRMarket(b=1.0)
        cost = market.cost_for_trade("yes", 1.0)
        assert cost > 0

    def test_cost_symmetry(self):
        """Buying 1 YES from initial state should cost the same as 1 NO."""
        market = LMSRMarket(b=1.0)
        cost_yes = market.cost_for_trade("yes", 1.0)
        cost_no = market.cost_for_trade("no", 1.0)
        assert cost_yes == pytest.approx(cost_no)

    def test_larger_trades_cost_more(self):
        market = LMSRMarket(b=1.0)
        cost_small = market.cost_for_trade("yes", 0.5)
        cost_large = market.cost_for_trade("yes", 2.0)
        assert cost_large > cost_small

    def test_liquidity_parameter_effect(self):
        """Larger b -> smaller price impact per unit traded."""
        m_low_b = LMSRMarket(b=0.5)
        m_high_b = LMSRMarket(b=5.0)

        m_low_b.execute_trade(0, "yes", 1.0)
        m_high_b.execute_trade(0, "yes", 1.0)

        # Low b should move price more
        assert m_low_b.price_yes() > m_high_b.price_yes()

    def test_max_loss(self):
        market = LMSRMarket(b=2.0)
        assert market.max_loss == pytest.approx(2.0 * np.log(2))

    def test_reset(self):
        market = LMSRMarket(b=1.0)
        market.execute_trade(0, "yes", 5.0)
        market.reset()
        assert market.q_yes == 0.0
        assert market.q_no == 0.0
        assert market.price_yes() == pytest.approx(0.5)
        assert len(market.trade_history) == 0

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError):
            LMSRMarket(b=0.0)
        with pytest.raises(ValueError):
            LMSRMarket(b=-1.0)

    def test_invalid_outcome_raises(self):
        market = LMSRMarket(b=1.0)
        with pytest.raises(ValueError):
            market.cost_for_trade("maybe", 1.0)

    def test_trade_history_recorded(self):
        market = LMSRMarket(b=1.0)
        market.execute_trade(0, "yes", 1.0)
        market.execute_trade(1, "no", 2.0)
        assert len(market.trade_history) == 2
        assert market.trade_history[0].agent_id == 0
        assert market.trade_history[1].shares == 2.0

    def test_numerical_stability_large_imbalance(self):
        """Market should handle large share imbalances without overflow."""
        market = LMSRMarket(b=1.0)
        market.execute_trade(0, "yes", 100.0)
        p = market.price_yes()
        assert 0.0 < p <= 1.0
        assert not np.isnan(p)
        assert not np.isinf(p)
