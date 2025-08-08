from dataclasses import dataclass
from pathlib import Path
import torch

from .transportation_simulator import TransportationSimulator
from .agents.base import DijkstraAgents

@dataclass
class RunnerArgs:
    algo: str
    mode: str
    steps: int = 1
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
        if self.args.algo == "dijkstra":
            self.simulator = TransportationSimulator(self.device)
            self.agent = DijkstraAgents(self.device)
            self.simulator.agent = self.agent
        elif self.args.algo in {"mpnn", "mpnn+ppo"}:
            from .reinforcement_learning import SimulatorEnv
            from .agents.mpnn_agent import MPNNPolicyNet, MPNNValueNetSimple

            self.env = SimulatorEnv(device=str(self.device))
            edge_index = self.env.simulator.graph.edge_index
            num_nodes = self.env.simulator.graph.x.size(0)
            free_flow = self.env.simulator.graph.x[:, self.env.simulator.h.FREE_FLOW_TIME_TRAVEL]
            free_flow = free_flow[edge_index[1]].to(self.device)
            self.policy_net = MPNNPolicyNet(edge_index, num_nodes, free_flow, device=str(self.device))
            self.value_net = MPNNValueNetSimple(edge_index, num_nodes, device=str(self.device))
            self.env.base_env.simulator.agent = self.policy_net
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
            distribution_kwargs={"edge_index": self.env.base_env.simulator.graph.edge_index},
            return_log_prob=False,
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

        ppo_train(
            self.env,
            policy_module,
            value_module,
            total_frames=self.args.rollout_steps,
            frames_per_batch=self.args.rollout_steps,
            num_epochs=self.args.epochs,
            device=self.device,
            checkpoint_path=checkpoint,
        )

    def eval(self):
        if self.args.algo == "dijkstra":
            from .algorithms.dijkstra_runner import run_episode
            run_episode(self.simulator, self.agent, steps=self.args.steps)
        else:
            with torch.no_grad():
                self.env.rollout(self.args.steps, self.policy_net, break_when_any_done=True)
