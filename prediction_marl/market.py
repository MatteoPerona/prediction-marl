"""Logarithmic Market Scoring Rule (LMSR) automated market maker.

Implements Hanson's LMSR for binary contracts. The market maker maintains
outstanding share counts q_yes and q_no, and a liquidity parameter b.

Cost function:  C(q) = b * ln(exp(q_yes/b) + exp(q_no/b))
Price function: p_yes = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))  (softmax)
Trade cost:     buying Δ shares of YES costs C(q_yes+Δ, q_no) - C(q_yes, q_no)

References:
    Hanson (2003), "Combinatorial Information Market Design"
    Hanson (2007), "Logarithmic Market Scoring Rules for Modular
                    Combinatorial Information Aggregation"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Record of a single trade."""

    agent_id: int
    outcome: str  # "yes" or "no"
    shares: float  # positive = buy, negative = sell
    cost: float  # money paid (positive) or received (negative)
    price_before: float  # p_yes before the trade
    price_after: float  # p_yes after the trade


@dataclass
class LMSRMarket:
    """Binary LMSR automated market maker.

    Parameters
    ----------
    b : float
        Liquidity parameter. Controls price sensitivity.
        Small b -> prices move a lot per trade (volatile, fast-reacting).
        Large b -> prices move slowly (stable, slow to incorporate info).
        Max loss for the market maker is b * ln(2) for a binary contract.
    """

    b: float = 1.0
    q_yes: float = 0.0
    q_no: float = 0.0
    trade_history: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError(f"Liquidity parameter b must be positive, got {self.b}")

    def _cost(self, q_yes: float, q_no: float) -> float:
        """Cost function C(q) = b * ln(exp(q_yes/b) + exp(q_no/b)).

        Uses logsumexp trick for numerical stability.
        """
        q = np.array([q_yes, q_no]) / self.b
        max_q = np.max(q)
        return self.b * (max_q + np.log(np.exp(q[0] - max_q) + np.exp(q[1] - max_q)))

    def price_yes(self) -> float:
        """Current implied probability for YES outcome (softmax)."""
        diff = (self.q_yes - self.q_no) / self.b
        return 1.0 / (1.0 + np.exp(-diff))

    def price_no(self) -> float:
        """Current implied probability for NO outcome."""
        return 1.0 - self.price_yes()

    def cost_for_trade(self, outcome: str, shares: float) -> float:
        """Compute the cost of buying `shares` of `outcome`.

        Parameters
        ----------
        outcome : str
            "yes" or "no"
        shares : float
            Number of shares. Positive = buy, negative = sell.

        Returns
        -------
        float
            Cost in currency units. Positive means agent pays.
        """
        if outcome == "yes":
            new_cost = self._cost(self.q_yes + shares, self.q_no)
        elif outcome == "no":
            new_cost = self._cost(self.q_yes, self.q_no + shares)
        else:
            raise ValueError(f"outcome must be 'yes' or 'no', got '{outcome}'")

        old_cost = self._cost(self.q_yes, self.q_no)
        return new_cost - old_cost

    def execute_trade(self, agent_id: int, outcome: str, shares: float) -> Trade:
        """Execute a trade and update market state.

        Parameters
        ----------
        agent_id : int
            Identifier for the trading agent.
        outcome : str
            "yes" or "no"
        shares : float
            Number of shares. Positive = buy, negative = sell.

        Returns
        -------
        Trade
            Record of the executed trade.
        """
        cost = self.cost_for_trade(outcome, shares)
        price_before = self.price_yes()

        if outcome == "yes":
            self.q_yes += shares
        else:
            self.q_no += shares

        price_after = self.price_yes()

        trade = Trade(
            agent_id=agent_id,
            outcome=outcome,
            shares=shares,
            cost=cost,
            price_before=price_before,
            price_after=price_after,
        )
        self.trade_history.append(trade)
        return trade

    def reset(self) -> None:
        """Reset market to initial state."""
        self.q_yes = 0.0
        self.q_no = 0.0
        self.trade_history.clear()

    @property
    def max_loss(self) -> float:
        """Maximum possible loss for the market maker: b * ln(num_outcomes)."""
        return self.b * np.log(2)

    def __repr__(self) -> str:
        return (
            f"LMSRMarket(b={self.b}, p_yes={self.price_yes():.4f}, "
            f"q_yes={self.q_yes:.2f}, q_no={self.q_no:.2f})"
        )
