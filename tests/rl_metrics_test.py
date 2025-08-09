import torch
from tensordict import TensorDict
from src.reinforcement_learning import SimulatorEnv
from src.transportation_simulator import TransportationSimulator
from src.agents.base import Agents


def test_rl_training_and_evaluation_collect_metrics(monkeypatch, simple_network_file):
    # Patch network loading to use a simple test network
    def fake_load_network(self, scenario):
        self.config_network(simple_network_file)
    monkeypatch.setattr(TransportationSimulator, "load_network", fake_load_network)
    # Simplify agent operations
    monkeypatch.setattr(Agents, "reset", lambda self: None)
    monkeypatch.setattr(Agents, "withdraw_agent_from_network", lambda self, x, h: x)
    monkeypatch.setattr(Agents, "insert_agent_into_network", lambda self, x, h: x)

    env = SimulatorEnv(device="cpu", timestep_size=1, start_time=0, scenario="dummy")
    eval_env = SimulatorEnv(device="cpu", timestep_size=1, start_time=0, scenario="dummy")

    for e in (env, eval_env):
        e.simulator.agent.agent_features = torch.zeros((1, 9))
        e.simulator.agent.set_time(e.simulator.time)

    num_edges = env.simulator.graph.edge_index.size(1)
    param = torch.nn.Parameter(torch.zeros(num_edges))
    optim = torch.optim.SGD([param], lr=0.1)

    # --- Training loop (2 steps) ---
    env._reset()
    for _ in range(2):
        action = (param > 0).to(torch.bool).float()
        td = TensorDict({"action": action}, batch_size=[])
        env._step(td)
        loss = param.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    assert env.simulator.time > 0
    assert torch.any(param != 0)  # parameter updated

    # --- Evaluation rollout ---
    eval_env._reset()
    for _ in range(2):
        action = (param > 0).to(torch.bool).float()
        td = TensorDict({"action": action}, batch_size=[])
        eval_env._step(td)

    assert eval_env.simulator.leg_histogram_values
    assert eval_env.simulator.road_optimality_values
    total_time = (
        eval_env.simulator.inserting_time + eval_env.simulator.core_time + eval_env.simulator.withdraw_time
    )
    assert total_time > 0
