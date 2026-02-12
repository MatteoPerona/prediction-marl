"""Trading agents for the prediction market.

Implements:
- BaseAgent: Abstract interface all agents share.
- ZeroIntelligenceAgent: Trades based on private signal vs. market price.
  No learning — just pushes price toward its belief. Used to validate the
  LMSR mechanism works before adding RL.
- QLearningAgent: Tabular Q-learner. State = (discretized price, discretized
  signal). Actions = {buy_yes, buy_no, hold}. Reward = P&L at settlement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from prediction_marl.market import LMSRMarket, Trade


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
ACTIONS = ["buy_yes", "buy_no", "hold"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------
class BaseAgent(ABC):
    """Abstract base for all trading agents."""

    def __init__(self, agent_id: int, signal: float, trade_size: float = 1.0):
        """
        Parameters
        ----------
        agent_id : int
            Unique identifier.
        signal : float
            Private belief / noisy signal of the true probability (in [0, 1]).
        trade_size : float
            Fixed number of shares per trade.
        """
        self.agent_id = agent_id
        self.signal = signal
        self.trade_size = trade_size

        # Portfolio tracking
        self.shares_yes: float = 0.0
        self.shares_no: float = 0.0
        self.cash_spent: float = 0.0  # total money spent on trades

    @abstractmethod
    def choose_action(self, market: LMSRMarket) -> str:
        """Return one of ACTIONS."""
        ...

    def execute(self, market: LMSRMarket) -> Trade | None:
        """Choose an action and execute it on the market."""
        action = self.choose_action(market)
        if action == "hold":
            return None

        outcome = "yes" if action == "buy_yes" else "no"
        trade = market.execute_trade(self.agent_id, outcome, self.trade_size)

        if outcome == "yes":
            self.shares_yes += self.trade_size
        else:
            self.shares_no += self.trade_size

        self.cash_spent += trade.cost
        return trade

    def settlement_pnl(self, realized_outcome: str) -> float:
        """Compute P&L when the event resolves.

        Shares of the winning outcome pay 1 per share; losing shares pay 0.
        """
        if realized_outcome == "yes":
            payout = self.shares_yes * 1.0
        else:
            payout = self.shares_no * 1.0
        return payout - self.cash_spent

    def reset(self) -> None:
        """Reset portfolio for a new episode."""
        self.shares_yes = 0.0
        self.shares_no = 0.0
        self.cash_spent = 0.0

    def update(self, reward: float) -> None:
        """Post-episode learning update (no-op for non-learning agents)."""
        pass


# ---------------------------------------------------------------------------
# Zero-intelligence agent
# ---------------------------------------------------------------------------
class ZeroIntelligenceAgent(BaseAgent):
    """Trades whenever the market price diverges from its private signal.

    If p_yes < signal - threshold, buys YES.
    If p_yes > signal + threshold, buys NO.
    Otherwise holds.
    """

    def __init__(
        self,
        agent_id: int,
        signal: float,
        trade_size: float = 1.0,
        threshold: float = 0.02,
    ):
        super().__init__(agent_id, signal, trade_size)
        self.threshold = threshold

    def choose_action(self, market: LMSRMarket) -> str:
        p = market.price_yes()
        if p < self.signal - self.threshold:
            return "buy_yes"
        elif p > self.signal + self.threshold:
            return "buy_no"
        return "hold"


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------
@dataclass
class QLearningConfig:
    """Hyperparameters for the Q-learning agent."""

    n_price_bins: int = 20
    n_signal_bins: int = 10
    alpha: float = 0.1  # learning rate
    gamma: float = 0.99  # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


class QLearningAgent(BaseAgent):
    """Tabular Q-learning agent for LMSR prediction market trading.

    State: (discretized market price, discretized private signal)
    Actions: buy_yes, buy_no, hold
    Reward: settlement P&L at end of episode (delayed reward)
    """

    def __init__(
        self,
        agent_id: int,
        signal: float,
        trade_size: float = 1.0,
        config: QLearningConfig | None = None,
    ):
        super().__init__(agent_id, signal, trade_size)
        self.config = config or QLearningConfig()

        # Q-table: shape (n_price_bins, n_signal_bins, num_actions)
        self.q_table = np.zeros(
            (self.config.n_price_bins, self.config.n_signal_bins, NUM_ACTIONS)
        )
        self.epsilon = self.config.epsilon_start

        # Episode trajectory for delayed reward assignment
        self._episode_states: list[tuple[int, int]] = []
        self._episode_actions: list[int] = []

    def _discretize_price(self, price: float) -> int:
        """Map price in [0, 1] to bin index."""
        idx = int(price * self.config.n_price_bins)
        return min(idx, self.config.n_price_bins - 1)

    def _discretize_signal(self, signal: float) -> int:
        """Map signal in [0, 1] to bin index."""
        idx = int(signal * self.config.n_signal_bins)
        return min(idx, self.config.n_signal_bins - 1)

    def _get_state(self, market: LMSRMarket) -> tuple[int, int]:
        return (
            self._discretize_price(market.price_yes()),
            self._discretize_signal(self.signal),
        )

    def choose_action(self, market: LMSRMarket) -> str:
        state = self._get_state(market)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(NUM_ACTIONS)
        else:
            action_idx = int(np.argmax(self.q_table[state[0], state[1], :]))

        # Record for later update
        self._episode_states.append(state)
        self._episode_actions.append(action_idx)

        return ACTIONS[action_idx]

    def update(self, reward: float) -> None:
        """Assign terminal reward to all state-action pairs visited this episode.

        Uses a simple Monte Carlo-style update: every visited (s, a) pair gets
        the same terminal reward (the settlement P&L), discounted by how many
        steps from the end.
        """
        n = len(self._episode_states)
        for t in range(n):
            s = self._episode_states[t]
            a = self._episode_actions[t]
            # Discount: steps_remaining = n - 1 - t
            discounted_reward = reward * (self.config.gamma ** (n - 1 - t))
            # Q-learning update (terminal state, no max Q(s', a'))
            old_q = self.q_table[s[0], s[1], a]
            self.q_table[s[0], s[1], a] = old_q + self.config.alpha * (
                discounted_reward - old_q
            )

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
        )

        # Clear trajectory
        self._episode_states.clear()
        self._episode_actions.clear()

    def reset(self) -> None:
        """Reset portfolio and episode trajectory (keep Q-table and epsilon)."""
        super().reset()
        self._episode_states.clear()
        self._episode_actions.clear()

    def set_signal(self, signal: float) -> None:
        """Update the agent's private signal for a new episode."""
        self.signal = signal
