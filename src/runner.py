from dataclasses import dataclass
from pathlib import Path
import torch
import os

from .reinforcement_learning import SimulatorEnv
from .transportation_simulator import TransportationSimulator
from .agents.base import Agents, DijkstraAgents

@dataclass
class RunnerArgs:
    algo: str
    scenario: str
    mode: str
    timestep_size: int = 1
    start_end_time: list[int] = (0, 86400)
    epochs: int = 1
    rollout_steps: int = 32
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "runs"


class Runner:
    """Unified entry point for classical and RL experiments."""

    def __init__(self, args: RunnerArgs):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        torch.manual_seed(args.seed)

    def setup(self):
        # --- Initialize the simulator and agent based on the algorithm ---
        if self.args.algo in {"dijkstra", "random"}:
            self.simulator = TransportationSimulator(self.device)
            if self.args.algo == "dijkstra":
                self.agent = DijkstraAgents(self.device)
            else:
                self.agent = Agents(self.device)

            self.simulator.load_network(scenario=self.args.scenario)
            self.agent.load(scenario=self.args.scenario)
            self.simulator.config_parameters(
                timestep_size  = self.args.timestep_size,
                start_time = self.args.start_end_time[0]
            )
            self.agent.set_time(self.args.start_end_time[0])

        elif self.args.algo in {"mpnn", "mpnn+ppo"}:
            from .agents.mpnn_agent import MPNNPolicyNet, MPNNValueNetSimple

            self.env = SimulatorEnv(
                device=str(self.device),
                timestep_size=self.args.timestep_size,
                start_time=self.args.start_end_time[0],
                scenario=self.args.scenario,
            )

            edge_index = self.env.simulator.graph.edge_index
            num_nodes = self.env.simulator.graph.x.size(0)
            free_flow = self.env.simulator.graph.x[:, self.env.simulator.h.FREE_FLOW_TIME_TRAVEL]
            free_flow = free_flow[edge_index[1]].to(self.device)
            self.policy_net = MPNNPolicyNet(edge_index, num_nodes, free_flow, device=str(self.device))
            self.policy_net.load(self.args.scenario)
            self.value_net = MPNNValueNetSimple(edge_index, num_nodes, device=str(self.device))
            self.value_net.load(self.args.scenario)
            self.env.simulator.agent = self.policy_net
        else:
            raise ValueError(f"Unknown algorithm {self.args.algo}")

    def train(self):
        if not (self.args.algo == "mpnn+ppo" and self.args.mode == "train"):
            raise RuntimeError("Training is only supported for algo 'mpnn+ppo'")

        from tensordict.nn import TensorDictModule
        from torchrl.modules import ProbabilisticActor, ValueOperator
        from .reinforcement_learning import GraphDistribution
        from .rl.ppo_trainer import ppo_train

        policy_tdmodule = TensorDictModule(
            self.policy_net,
            in_keys=["node_features", "edge_features", "agent_index"],
            out_keys=["logits"],
        )
        policy_module = ProbabilisticActor(
            module=policy_tdmodule,
            spec=self.env.action_spec,
            distribution_class=GraphDistribution,
            in_keys=["logits"],
            distribution_kwargs={"edge_index": self.env.simulator.graph.edge_index},
            return_log_prob=True,
        )

        value_tdmodule = TensorDictModule(
            self.value_net,
            in_keys=["node_features", "edge_features", "agent_index", "time"],
            out_keys=["value"],
        )
        value_module = ValueOperator(
            module=value_tdmodule,
            in_keys=["node_features", "edge_features", "agent_index", "time"],
        )

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "policy.pt"

        eval_env = SimulatorEnv(
            device=str(self.device),
            timestep_size=self.args.timestep_size,
            start_time=self.args.start_end_time[0],
            scenario=self.args.scenario,
        )
        eval_env.simulator.agent = self.policy_net

        ppo_train(
            self.env,
            policy_module,
            value_module,
            total_frames=self.args.rollout_steps,
            frames_per_batch=self.args.rollout_steps,
            num_epochs=self.args.epochs,
            device=self.device,
            checkpoint_path=checkpoint,
            log_dir=str(output_dir),
            eval_env=eval_env,
            eval_interval=1,
        )

    def eval(self):
        n_timesteps = (self.args.start_end_time[1] - self.args.start_end_time[0]) // self.args.timestep_size

        if self.args.algo in {"dijkstra", "random"}:
            from .algorithms.base_runner import run_episode
            run_episode(self.simulator, self.agent, steps=n_timesteps)

            # Evaluate metrics
            mask = self.agent.agent_features[:, self.agent.DONE] == 1
            average_travel = torch.mean(self.agent.agent_features[mask, self.agent.ARRIVAL_TIME] - self.agent.agent_features[mask, self.agent.DEPARTURE_TIME])
            print("\n=== Simulation Summary ===")
            print(f"{'Average travel time:':25} {average_travel.item():10.2f} s")
            print(f"{'Agent Insertion time:':25} {self.simulator.inserting_time:10.2f} s")
            print(f"{'Route Choice time:':25} {self.simulator.choice_time:10.2f} s")
            print(f"{'Core Model time:':25} {self.simulator.core_time:10.2f} s")
            print(f"{'Agent Withdrawal time:':25} {self.simulator.withdraw_time:10.2f} s")
            print("-" * 42)
            total_time = (
                self.simulator.inserting_time
                + self.simulator.choice_time
                + self.simulator.core_time
                + self.simulator.withdraw_time
            )
            print(f"{'Total simulation time:':25} {total_time:10.2f} s")

            print("\n=== Computing Metrics... ===")
            
            self.simulator.plot_computation_time(self.args.output_dir)
            self.simulator.compute_node_metrics(self.args.output_dir)
            self.simulator.plot_leg_histogram(self.args.output_dir)
            self.simulator.plot_road_optimality(self.args.output_dir)


        else:
            from tensordict.nn import TensorDictModule
            from torchrl.modules import ProbabilisticActor
            from .reinforcement_learning import GraphDistribution

            policy_tdmodule = TensorDictModule(
                self.policy_net,
                in_keys=["node_features", "edge_features", "agent_index"],
                out_keys=["logits"],
            )
            policy_module = ProbabilisticActor(
                module=policy_tdmodule,
                spec=self.env.action_spec,
                distribution_class=GraphDistribution,
                in_keys=["logits"],
                distribution_kwargs={"edge_index": self.env.simulator.graph.edge_index},
                return_log_prob=False,
            )

            with torch.no_grad():
                self.env.rollout(n_timesteps, policy_module, break_when_any_done=False)

            # Evaluate metrics similar to classical algorithms
            mask = self.env.simulator.agent.agent_features[:, self.env.simulator.agent.DONE] == 1
            average_travel = torch.mean(
                self.env.simulator.agent.agent_features[mask, self.env.simulator.agent.ARRIVAL_TIME]
                - self.env.simulator.agent.agent_features[mask, self.env.simulator.agent.DEPARTURE_TIME]
            )
            print("\n=== Simulation Summary ===")
            print(f"{'Average travel time:':25} {average_travel.item():10.2f} s")
            print(f"{'Agent Insertion time:':25} {self.env.simulator.inserting_time:10.2f} s")
            print(f"{'Route Choice time:':25} {self.env.simulator.choice_time:10.2f} s")
            print(f"{'Core Model time:':25} {self.env.simulator.core_time:10.2f} s")
            print(f"{'Agent Withdrawal time:':25} {self.env.simulator.withdraw_time:10.2f} s")
            print("-" * 42)
            total_time = (
                self.env.simulator.inserting_time
                + self.env.simulator.choice_time
                + self.env.simulator.core_time
                + self.env.simulator.withdraw_time
            )
            print(f"{'Total simulation time:':25} {total_time:10.2f} s")

            print("\n=== Computing Metrics... ===")
            self.env.simulator.plot_computation_time(self.args.output_dir)
            self.env.simulator.compute_node_metrics(self.args.output_dir)
            self.env.simulator.plot_leg_histogram(self.args.output_dir)
            self.env.simulator.plot_road_optimality(self.args.output_dir)
