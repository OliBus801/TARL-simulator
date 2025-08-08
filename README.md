# TARL-simulator

**Traffic Assignment via Reinforcement Learning** — an agent-based traffic simulation framework combining Graph Neural Networks and Reinforcement Learning (PPO) for Dynamic Traffic Assignment.

## Status

🚧 **Under active development** 🚧  
This project is currently in an early stage.  
Most core features are being implemented and are not yet ready for academic use.

## Overview

TARL-simulator aims to:
- Model transportation networks as graphs (dual graph representation).
- Simulate agents making adaptive route choices.
- Learn routing policies via reinforcement learning to approach User Equilibrium or optimal routing.

## Current Features

- Basic network and agent definitions.
- Preliminary routing logic using Dijkstra.
- Early integration of MPNN + PPO framework (*work in progress*).

## Planned Features

- Full RL training loop with policy evaluation.
- Advanced metrics (Nash gap, TSTT, Price of Anarchy).
- Support for synthetic and real-world network scenarios.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Unified runner

Experiments are launched through ``main.py`` which selects the
algorithm and mode of execution.

**Examples**

Run classical Dijkstra evaluation:

```bash
python main.py --algo dijkstra --mode eval --steps 10
```

Evaluate a trained MPNN policy:

```bash
python main.py --algo mpnn --mode eval --steps 10
```

Train an MPNN using PPO:

```bash
python main.py --algo mpnn+ppo --mode train --epochs 5 --rollout-steps 256
```
