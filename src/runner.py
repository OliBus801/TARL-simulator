from dataclasses import dataclass
from pathlib import Path
import torch

from .reinforcement_learning import SimulatorEnv
from .transportation_simulator import TransportationSimulator
from .agents.base import Agents, DijkstraAgents
from .algorithms.user_equilibrium_msa import run_msa

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
    profile: bool = False
    torch_compile: bool = False
    wandb: bool = False


class Runner:
    """Unified entry point for classical and RL experiments."""

    def __init__(self, args: RunnerArgs):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        torch.manual_seed(args.seed)

    def setup(self):
        # --- Initialize the simulator and agent based on the algorithm ---
        if self.args.algo in {"dijkstra", "random"}:
            self.simulator = TransportationSimulator(self.device, torch_compile=self.args.torch_compile)
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
                torch_compile=self.args.torch_compile,
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

        elif self.args.algo == "contextual-bandit":
            from .contextual_cost_head import ContextualCostHead

            self.simulator = TransportationSimulator(self.device, torch_compile=self.args.torch_compile)
            self.agent = Agents(self.device)

            self.simulator.load_network(scenario=self.args.scenario)
            self.agent.load(scenario=self.args.scenario)
            self.simulator.config_parameters(
                timestep_size=self.args.timestep_size,
                start_time=self.args.start_end_time[0],
            )
            self.agent.set_time(self.args.start_end_time[0])

            input_dim = (
                int(self.simulator.graph.x.size(1))
                + int(self.simulator.graph.edge_attr.size(1))
                + int(self.agent.agent_features.size(1))
            )
            self.cost_head = ContextualCostHead(input_dim).to(self.device)
            self.optimizer = torch.optim.Adam(self.cost_head.parameters(), lr=1e-3)
            self.expected_demand = run_msa(self.simulator.graph, self.agent)
        else:
            raise ValueError(f"Unknown algorithm {self.args.algo}")

    def train(self):
        if self.args.algo == "contextual-bandit":
            if self.args.mode != "train":
                raise RuntimeError("Training is only supported in 'train' mode")

            from tqdm import tqdm
            if self.args.wandb:
                import wandb
                wandb.init(project="tarl-simulator", name="contextual-bandit")

            n_timesteps = (
                self.args.start_end_time[1] - self.args.start_end_time[0]
            ) // self.args.timestep_size

            pbar = tqdm(range(self.args.epochs), desc="Contextual Bandit")
            for epoch in pbar:
                # Reload network and agents to reset state
                self.simulator.load_network(scenario=self.args.scenario)
                self.agent.load(scenario=self.args.scenario)
                self.simulator.config_parameters(
                    timestep_size=self.args.timestep_size,
                    start_time=self.args.start_end_time[0],
                )
                self.agent.set_time(self.args.start_end_time[0])

                self.simulator.reset()
                self.agent.reset()
                self.simulator.config_parameters(
                    start_time=self.args.start_end_time[0],
                )
                self.agent.set_time(self.args.start_end_time[0])

                self.simulator.train_contextual_bandit(
                    n_timesteps, self.cost_head, self.optimizer
                )

                node_metrics = self.simulator.compute_node_metrics(output_dir=None)
                sim_counts = {
                    idx: sum(m["hourly_counts"]) for idx, m in node_metrics.items()
                }
                exp = torch.tensor(
                    [self.expected_demand.get(i, 0.0) for i in self.expected_demand.keys()],
                    dtype=torch.float64,
                )
                sim = torch.tensor(
                    [sim_counts.get(i, 0.0) for i in self.expected_demand.keys()],
                    dtype=torch.float64,
                )
                rmse = torch.sqrt(torch.mean((sim - exp) ** 2)).item()
                print(f"{'RMSE demand:':25} {rmse:10.4f}")
                if self.args.wandb:
                    wandb.log({"rmse_demand": rmse, "epoch": epoch + 1})
                pbar.set_postfix(rmse=rmse)

            if self.args.wandb:
                wandb.finish()
            return

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
            torch_compile=self.args.torch_compile,
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
            run_episode(
                self.simulator,
                self.agent,
                steps=n_timesteps,
                profile=self.args.profile,
                profile_output=(
                    Path(self.args.output_dir) / "profile.txt" if self.args.profile else None
                ),
            )

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
            expected_demand = run_msa(self.simulator.graph, self.agent)
            self.simulator.plot_daily_counts(expected_demand, self.args.output_dir)

        elif self.args.algo == "contextual-bandit":
            self.simulator.load_network(scenario=self.args.scenario)
            self.agent.load(scenario=self.args.scenario)
            self.simulator.config_parameters(
                timestep_size=self.args.timestep_size,
                start_time=self.args.start_end_time[0],
            )
            self.agent.set_time(self.args.start_end_time[0])
            if hasattr(self.simulator.model_core.response_mpnn, "update_history"):
                self.simulator.model_core.response_mpnn.update_history = []
            self.agent.withdraw_history = []
            self.simulator.leg_histogram_values = []
            self.simulator.road_optimality_values = []
            self.simulator.on_way_before = 0
            self.simulator.done_before = 0

            h = self.simulator.h
            for _ in range(n_timesteps):
                self.simulator.graph.x = self.agent.insert_agent_into_network(
                    self.simulator.graph, h
                )
                self.simulator.graph.x = self.agent.withdraw_agent_from_network(
                    self.simulator.graph, h
                )
                self.simulator.graph = self.simulator.model_core(self.simulator.graph)

                road_indices = torch.arange(self.simulator.graph.num_roads, device=self.device)
                active = self.simulator.graph.x[road_indices, h.NUMBER_OF_AGENT] > 0
                for road in road_indices[active]:
                    agent_id = int(self.simulator.graph.x[road, h.HEAD_FIFO].item())
                    edge_mask = self.simulator.graph.edge_index[0] == road
                    dest = self.simulator.graph.edge_index[1, edge_mask]
                    if dest.numel() == 0:
                        continue
                    h_v = self.simulator.graph.x[road].expand(dest.size(0), -1)
                    x_a = self.simulator.graph.edge_attr[edge_mask]
                    x_agent = self.agent.agent_features[agent_id].expand(dest.size(0), -1)
                    features = torch.cat([h_v, x_a, x_agent], dim=-1)
                    pred_costs = self.cost_head(features)
                    mask = torch.ones_like(pred_costs, dtype=torch.bool)
                    action_idx = self.cost_head.sample_action(pred_costs, mask)
                    chosen_road = dest[action_idx]
                    self.simulator.graph.x[road, h.SELECTED_ROAD] = chosen_road.to(
                        self.simulator.graph.x.dtype
                    )
                    self.simulator.graph = self.simulator.model_core(self.simulator.graph)

                self.simulator.set_time(self.simulator.time + self.simulator.timestep)

                value_on_way = torch.sum(
                    self.agent.agent_features[:, self.agent.ON_WAY]
                )
                value_done = torch.sum(
                    self.agent.agent_features[:, self.agent.DONE]
                )
                self.simulator.leg_histogram_values.append(
                    [
                        value_on_way
                        - self.simulator.on_way_before
                        + value_done
                        - self.simulator.done_before,
                        value_done - self.simulator.done_before,
                        value_on_way,
                        self.simulator.time,
                    ]
                )
                self.simulator.on_way_before = value_on_way
                self.simulator.done_before = value_done
                self.simulator.road_optimality_values.append(
                    (
                        self.simulator.time,
                        self.simulator.model_core.direction_mpnn.road_optimality_data[
                            "delta_travel_time"
                        ].cpu(),
                    )
                )

            mask = self.agent.agent_features[:, self.agent.DONE] == 1
            average_travel = torch.mean(
                self.agent.agent_features[mask, self.agent.ARRIVAL_TIME]
                - self.agent.agent_features[mask, self.agent.DEPARTURE_TIME]
            )
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
            expected_demand = run_msa(self.simulator.graph, self.agent)
            self.simulator.plot_daily_counts(expected_demand, self.args.output_dir)

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
            expected_demand = run_msa(self.env.simulator.graph, self.env.simulator.agent)
            self.env.simulator.plot_daily_counts(expected_demand, self.args.output_dir)
