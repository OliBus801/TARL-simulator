"""Unified entry point for TARL-simulator experiments."""
import argparse
from src.runner import Runner, RunnerArgs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified runner for classical and RL experiments")
    parser.add_argument("--algo", choices=["dijkstra", "mpnn", "mpnn+ppo"], default="dijkstra")
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--steps", type=int, default=1, help="Number of environment steps for evaluation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--rollout-steps", type=int, default=32, help="Rollout horizon for PPO training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs")
    args = parser.parse_args(argv)

    runner = Runner(RunnerArgs(**vars(args)))
    runner.setup()
    if args.mode == "train":
        runner.train()
    runner.eval()


if __name__ == "__main__":
    main()
