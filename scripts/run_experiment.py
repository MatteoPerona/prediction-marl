#!/usr/bin/env python3
"""Run a prediction market experiment from a YAML config file.

Usage:
    python scripts/run_experiment.py                         # uses default config
    python scripts/run_experiment.py configs/q_learning.yaml # custom config
    python scripts/run_experiment.py --plot                   # plot price trajectories
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prediction_marl.market import LMSRMarket
from prediction_marl.agents import (
    ZeroIntelligenceAgent,
    QLearningAgent,
    QLearningConfig,
)
from prediction_marl.simulation import run_simulation, generate_signals
from prediction_marl.metrics import compute_all_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_agents(cfg: dict) -> list:
    agent_cfg = cfg["agents"]
    n = agent_cfg["n_agents"]
    trade_size = agent_cfg["trade_size"]

    if agent_cfg["type"] == "zero_intelligence":
        return [
            ZeroIntelligenceAgent(
                agent_id=i,
                signal=0.5,  # will be overwritten by simulation
                trade_size=trade_size,
                threshold=agent_cfg["threshold"],
            )
            for i in range(n)
        ]
    elif agent_cfg["type"] == "q_learning":
        ql_cfg = agent_cfg["q_learning"]
        config = QLearningConfig(**ql_cfg)
        return [
            QLearningAgent(
                agent_id=i,
                signal=0.5,
                trade_size=trade_size,
                config=config,
            )
            for i in range(n)
        ]
    else:
        raise ValueError(f"Unknown agent type: {agent_cfg['type']}")


def run(config_path: str, plot: bool = False) -> dict:
    cfg = load_config(config_path)
    sim_cfg = cfg["simulation"]

    market = LMSRMarket(b=cfg["market"]["b"])
    agents = build_agents(cfg)

    true_prob = sim_cfg.get("true_prob")

    print(f"Running {sim_cfg['n_episodes']} episodes, {sim_cfg['n_rounds']} rounds each")
    print(f"Agent type: {cfg['agents']['type']}, count: {cfg['agents']['n_agents']}")
    print(f"LMSR b={cfg['market']['b']}, true_prob={true_prob or 'varying'}")
    print()

    # Track agent signals for superadditivity test
    agent_signals: dict[int, list[float]] = {a.agent_id: [] for a in agents}
    rng = np.random.default_rng(sim_cfg["seed"])

    results = []
    n_episodes = sim_cfg["n_episodes"]
    n_rounds = sim_cfg["n_rounds"]

    for ep in range(n_episodes):
        if true_prob is not None:
            ep_true_prob = true_prob
        else:
            ep_true_prob = rng.uniform(*sim_cfg["true_prob_range"])

        signals = generate_signals(
            len(agents), ep_true_prob, sim_cfg["noise_std"], rng
        )
        for agent, sig in zip(agents, signals):
            agent.signal = sig
            agent_signals[agent.agent_id].append(sig)

        from prediction_marl.simulation import run_episode

        result = run_episode(market, agents, ep_true_prob, n_rounds, rng)
        results.append(result)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            recent = results[max(0, ep - 19) :]
            avg_err = np.mean([abs(r.final_price - r.true_prob) for r in recent])
            print(f"  Episode {ep + 1}/{n_episodes}: avg convergence error = {avg_err:.4f}")

    # Compute metrics
    metrics = compute_all_metrics(results, agent_signals)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Market Brier Score:    {metrics['market_brier_score']:.4f}")
    print(f"Convergence Error:     {metrics['convergence_error']:.4f}")
    print(f"Mean Trades/Episode:   {metrics['mean_trades_per_episode']:.1f}")

    if "superadditivity" in metrics:
        sa = metrics["superadditivity"]
        print(f"\nBest Agent Brier:      {sa['best_agent_bs']:.4f}")
        print(f"Market vs Best Agent:  {sa['improvement']:+.4f} ({'better' if sa['improvement'] > 0 else 'worse'})")
        print(f"Relative Improvement:  {sa['relative_improvement']:+.1%}")

    if plot:
        plot_results(results, cfg)

    return metrics


def plot_results(results: list, cfg: dict) -> None:
    """Plot price trajectories for a sample of episodes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Price trajectories (sample 10 episodes)
    ax = axes[0]
    sample = results[:10]
    for r in sample:
        ax.plot(r.price_trajectory, alpha=0.6)
        ax.axhline(r.true_prob, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Trading Round")
    ax.set_ylabel("Market Price (p_yes)")
    ax.set_title("Price Trajectories (first 10 episodes)")
    ax.set_ylim(0, 1)

    # Convergence error over episodes
    ax = axes[1]
    errors = [abs(r.final_price - r.true_prob) for r in results]
    window = max(1, len(results) // 20)
    smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
    ax.plot(smoothed)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Convergence Error (|price - true_prob|)")
    ax.set_title("Convergence Over Episodes")

    plt.tight_layout()
    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    fig.savefig(out_path / "experiment_results.png", dpi=150)
    print(f"\nPlot saved to {out_path / 'experiment_results.png'}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction market experiment")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()
    run(args.config, plot=args.plot)
