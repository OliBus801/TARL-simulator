"""Unified entry point for TARL-simulator experiments."""
import argparse
from src.runner import Runner, RunnerArgs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified runner for classical and RL experiments")
    parser.add_argument("--algo", choices=["dijkstra", "random", "mpnn", "mpnn+ppo"], default="dijkstra")
    parser.add_argument("--scenario", type=str, default="Easy", help="Scenario to run. Only give prefix, e.g., 'Easy' for 'Easy_network.xml.gz' and 'Easy_population.xml.gz'")
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--timestep_size", type=int, default=1, help="Size of each simulation step in seconds")
    parser.add_argument("--start-end-time", type=int, nargs=2, default=[0, 86400], help="Start and end time for the simulation in seconds")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--rollout-steps", type=int, default=32, help="Rollout horizon for PPO training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory to save outputs")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the simulation loop with cProfile",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for MPNN models",
    )
    args = parser.parse_args(argv)

    runner = Runner(RunnerArgs(**vars(args)))
    runner.setup()
    if args.mode == "train":
        runner.train()
    runner.eval()


if __name__ == "__main__":
    main()
