# Prediction-MARL

Multi-agent reinforcement learning for collective election forecasting via prediction markets.

## Overview

This project studies whether a prediction market populated by autonomous RL agents can function as a **forecast ensemble**: agents trade binary election contracts through an LMSR automated market maker, and the resulting market price is interpreted as a consensus probability.

The core hypothesis is that market prices aggregate dispersed information across heterogeneous agents more accurately than any individual agent — tested via Brier scores on synthetic and (eventually) historical election data.

### Research Hypotheses

- **H1 (Convergence):** Repeated trading under LMSR reaches a stable price equilibrium reflecting the true probability.
- **H2 (Superadditivity):** The market price yields lower Brier score than the best individual agent.
- **H3 (Competitiveness):** On historical elections, market prices are competitive with polling averages and real-money prediction markets.

## Architecture

```
prediction_marl/
├── market.py        # LMSR automated market maker
├── agents.py        # ZeroIntelligence + Q-learning agents
├── simulation.py    # Episode runner with signal generation & settlement
└── metrics.py       # Brier score, convergence error, superadditivity test
configs/
├── default.yaml     # Zero-intelligence agent config
└── q_learning.yaml  # Q-learning agent config
scripts/
└── run_experiment.py  # CLI experiment runner with progress & plotting
tests/                 # 51 unit tests
```

### LMSR Market Maker

Hanson's Logarithmic Market Scoring Rule provides continuous liquidity for binary contracts:

- **Cost function:** `C(q) = b * ln(exp(q_yes/b) + exp(q_no/b))`
- **Price function:** `p_yes = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))` (softmax)
- **Liquidity parameter `b`:** controls price sensitivity. Max market maker loss is `b * ln(2)`.

### Agents

| Agent | Description |
|---|---|
| **ZeroIntelligence** | Buys YES when price < signal, NO when price > signal. No learning. Used to validate the LMSR mechanism. |
| **Q-Learning** | Tabular Q-learner. State = (discretized price, discretized signal). Actions = {buy_yes, buy_no, hold}. Reward = settlement P&L. |

### Simulation

Each episode represents one "election":
1. A true probability is set (known in synthetic experiments).
2. Agents receive noisy private signals centered on the true probability.
3. Agents trade for `n_rounds` rounds against the LMSR (shuffled order each round).
4. The outcome is realized (sampled from true probability).
5. Agents receive settlement P&L as reward.

## Setup

Requires Python 3.10+.

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run with zero-intelligence agents (default config)
python scripts/run_experiment.py

# Run with Q-learning agents
python scripts/run_experiment.py configs/q_learning.yaml

# Run with plots saved to results/
python scripts/run_experiment.py --plot
```

### Configuration

Experiments are configured via YAML files. Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `market.b` | LMSR liquidity parameter | `2.0` |
| `simulation.n_episodes` | Number of episodes | `200` |
| `simulation.n_rounds` | Trading rounds per episode | `50` |
| `simulation.true_prob` | Fixed true probability (`null` to sample from range) | `0.7` |
| `simulation.noise_std` | Noise on agent private signals | `0.1` |
| `agents.type` | `zero_intelligence` or `q_learning` | `zero_intelligence` |
| `agents.n_agents` | Number of agents | `10` |
| `agents.trade_size` | Shares per trade | `1.0` |

## Tests

```bash
pytest
```

## Roadmap

- [ ] Q-learning hyperparameter tuning (learning rate schedules, trade sizing)
- [ ] Agent heterogeneity (varied noise levels, exploration strategies)
- [ ] Inventory penalties (Spooner et al.) for RL training stability
- [ ] Varying true probabilities across episodes for generalization testing
- [ ] Historical election data pipeline (2012-2024 U.S. presidential elections)
- [ ] Neural network agents (DQN) for continuous state spaces
- [ ] Comparison baselines: polling averages, real-money market closes
