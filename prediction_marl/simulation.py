"""Simulation environment for the prediction market.

An Episode represents a single "election":
1. A true probability is set (known in synthetic experiments).
2. Agents receive noisy private signals.
3. Agents trade for T rounds against the LMSR.
4. The outcome is realized (sampled from true probability).
5. Agents receive settlement P&L.

The Simulation class runs many episodes and collects metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prediction_marl.market import LMSRMarket
from prediction_marl.agents import BaseAgent


@dataclass
class EpisodeResult:
    """Results from a single episode."""

    true_prob: float
    realized_outcome: str  # "yes" or "no"
    final_price: float  # market price at end of trading
    price_trajectory: list[float]  # price after each round
    agent_pnls: dict[int, float] = field(default_factory=dict)  # agent_id -> pnl
    num_trades: int = 0


def generate_signals(
    n_agents: int,
    true_prob: float,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate noisy private signals for agents.

    Each agent gets signal = true_prob + N(0, noise_std), clipped to [0.01, 0.99].
    """
    if rng is None:
        rng = np.random.default_rng()
    signals = true_prob + rng.normal(0, noise_std, size=n_agents)
    return np.clip(signals, 0.01, 0.99)


def run_episode(
    market: LMSRMarket,
    agents: list[BaseAgent],
    true_prob: float,
    n_rounds: int = 50,
    rng: np.random.Generator | None = None,
) -> EpisodeResult:
    """Run a single episode of the prediction market.

    Parameters
    ----------
    market : LMSRMarket
        The market maker (will be reset at start).
    agents : list[BaseAgent]
        Trading agents (portfolios will be reset at start).
    true_prob : float
        True probability of YES outcome.
    n_rounds : int
        Number of trading rounds. Each round, every agent trades once (in
        random order).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    EpisodeResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # Reset
    market.reset()
    for agent in agents:
        agent.reset()

    price_trajectory = [market.price_yes()]
    num_trades = 0

    # Trading rounds
    for _ in range(n_rounds):
        # Shuffle agent order each round
        order = rng.permutation(len(agents))
        for idx in order:
            trade = agents[idx].execute(market)
            if trade is not None:
                num_trades += 1
        price_trajectory.append(market.price_yes())

    # Realize outcome
    realized_outcome = "yes" if rng.random() < true_prob else "no"

    # Compute P&L and update agents
    agent_pnls = {}
    for agent in agents:
        pnl = agent.settlement_pnl(realized_outcome)
        agent_pnls[agent.agent_id] = pnl
        agent.update(pnl)

    return EpisodeResult(
        true_prob=true_prob,
        realized_outcome=realized_outcome,
        final_price=price_trajectory[-1],
        price_trajectory=price_trajectory,
        agent_pnls=agent_pnls,
        num_trades=num_trades,
    )


def run_simulation(
    market: LMSRMarket,
    agents: list[BaseAgent],
    n_episodes: int = 100,
    n_rounds: int = 50,
    true_prob: float | None = None,
    true_prob_range: tuple[float, float] = (0.2, 0.8),
    noise_std: float = 0.1,
    seed: int = 42,
) -> list[EpisodeResult]:
    """Run multiple episodes of the prediction market simulation.

    Parameters
    ----------
    market : LMSRMarket
        The LMSR market maker.
    agents : list[BaseAgent]
        Trading agents.
    n_episodes : int
        Number of episodes to run.
    n_rounds : int
        Trading rounds per episode.
    true_prob : float, optional
        If set, use this fixed true probability every episode.
        If None, sample uniformly from true_prob_range each episode.
    true_prob_range : tuple
        Range for sampling true probabilities (ignored if true_prob is set).
    noise_std : float
        Standard deviation of noise added to each agent's signal.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[EpisodeResult]
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_episodes):
        # Set true probability for this episode
        if true_prob is not None:
            ep_true_prob = true_prob
        else:
            ep_true_prob = rng.uniform(*true_prob_range)

        # Generate new signals for each agent
        signals = generate_signals(len(agents), ep_true_prob, noise_std, rng)
        for agent, sig in zip(agents, signals):
            agent.signal = sig

        result = run_episode(market, agents, ep_true_prob, n_rounds, rng)
        results.append(result)

    return results
