"""Evaluation metrics for prediction market forecasts.

Primary metric: Brier score (strictly proper scoring rule).
Also computes convergence diagnostics and superadditivity tests.
"""

from __future__ import annotations

import numpy as np

from prediction_marl.simulation import EpisodeResult


def brier_score(predicted_prob: float, realized: str) -> float:
    """Brier score for a single binary prediction.

    BS = (predicted_prob - outcome)^2, where outcome=1 if YES, 0 if NO.
    Lower is better. Range [0, 1].
    """
    outcome = 1.0 if realized == "yes" else 0.0
    return (predicted_prob - outcome) ** 2


def mean_brier_score(results: list[EpisodeResult]) -> float:
    """Mean Brier score of the market's final price across episodes."""
    scores = [brier_score(r.final_price, r.realized_outcome) for r in results]
    return float(np.mean(scores))


def convergence_error(results: list[EpisodeResult]) -> float:
    """Mean absolute error between final price and true probability.

    This measures whether the market converges to the correct probability,
    independent of outcome realization noise.
    """
    errors = [abs(r.final_price - r.true_prob) for r in results]
    return float(np.mean(errors))


def price_trajectory_convergence(result: EpisodeResult) -> np.ndarray:
    """Absolute error between price trajectory and true prob over time."""
    return np.abs(np.array(result.price_trajectory) - result.true_prob)


def agent_brier_scores(
    results: list[EpisodeResult], agent_signals: dict[int, list[float]]
) -> dict[int, float]:
    """Compute mean Brier score for each agent's signal (as a forecast).

    Parameters
    ----------
    results : list[EpisodeResult]
        Episode results.
    agent_signals : dict[int, list[float]]
        Mapping from agent_id to list of signals (one per episode).

    Returns
    -------
    dict[int, float]
        agent_id -> mean Brier score of using their signal as forecast.
    """
    scores = {}
    for aid, signals in agent_signals.items():
        bs = [
            brier_score(sig, r.realized_outcome)
            for sig, r in zip(signals, results)
        ]
        scores[aid] = float(np.mean(bs))
    return scores


def superadditivity_test(
    market_brier: float, best_agent_brier: float
) -> dict[str, float]:
    """Test H2: does the market beat the best individual agent?

    Returns
    -------
    dict with:
        market_bs: market's mean Brier score
        best_agent_bs: best agent's mean Brier score
        improvement: (best_agent_bs - market_bs)
        relative_improvement: improvement / best_agent_bs (proportion)
    """
    improvement = best_agent_brier - market_brier
    rel_improvement = improvement / best_agent_brier if best_agent_brier > 0 else 0.0
    return {
        "market_bs": market_brier,
        "best_agent_bs": best_agent_brier,
        "improvement": improvement,
        "relative_improvement": rel_improvement,
    }


def compute_all_metrics(
    results: list[EpisodeResult],
    agent_signals: dict[int, list[float]] | None = None,
) -> dict:
    """Compute all evaluation metrics.

    Parameters
    ----------
    results : list[EpisodeResult]
        Simulation results.
    agent_signals : dict[int, list[float]], optional
        If provided, also compute per-agent Brier scores and superadditivity.

    Returns
    -------
    dict of metrics.
    """
    metrics: dict = {
        "n_episodes": len(results),
        "market_brier_score": mean_brier_score(results),
        "convergence_error": convergence_error(results),
        "mean_final_price": float(np.mean([r.final_price for r in results])),
        "mean_true_prob": float(np.mean([r.true_prob for r in results])),
        "mean_trades_per_episode": float(np.mean([r.num_trades for r in results])),
    }

    if agent_signals:
        agent_bs = agent_brier_scores(results, agent_signals)
        best_agent_bs = min(agent_bs.values())
        metrics["agent_brier_scores"] = agent_bs
        metrics["best_agent_brier"] = best_agent_bs
        metrics["superadditivity"] = superadditivity_test(
            metrics["market_brier_score"], best_agent_bs
        )

    return metrics
