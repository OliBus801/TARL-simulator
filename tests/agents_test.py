import torch
from torch_geometric.data import Data
from src.feature_helpers import FeatureHelpers
from src.agents.base import Agents


class TestAgent:
    def test_length(self, agents: Agents):
        assert len(agents) == 9

    def test_insert_and_withdraw(self, agents: Agents):
        h = FeatureHelpers(Nmax=5)
        x = torch.zeros((2, 3 * h.Nmax + 7))
        x[0, h.MAX_NUMBER_OF_AGENT] = 5
        x[0, h.ROAD_INDEX] = 0
        x[0, h.FREE_FLOW_TIME_TRAVEL] = 10
        edge_index = torch.tensor([[1, 0], [0, 0]])  # SRC(1) -> road 0, road 0 -> DEST(0)
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_index_routes=torch.empty((2, 0), dtype=torch.long),
            edge_attr_routes=torch.empty((0, 1)),
            num_roads=1,
        )
        agents.time = 0
        graph.x = agents.insert_agent_into_network(graph, h)
        assert graph.x[0, h.NUMBER_OF_AGENT] == 2
        assert torch.all(agents.agent_features[:2, agents.ON_WAY] == 1)

        # Agents should not withdraw before their departure time
        graph.x = agents.withdraw_agent_from_network(graph.x, graph.edge_index, h)
        assert graph.x[0, h.NUMBER_OF_AGENT] == 2

        agents.time = 10
        graph.x = agents.withdraw_agent_from_network(graph.x, graph.edge_index, h)
        assert graph.x[0, h.NUMBER_OF_AGENT] == 0
        assert torch.all(agents.agent_features[:2, agents.DONE] == 1)