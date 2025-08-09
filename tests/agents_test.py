import torch
from src.feature_helpers import FeatureHelpers
from src.agents.base import Agents


class TestAgent:
    def test_length(self, agents: Agents):
        assert len(agents) == 9

    def test_insert_and_withdraw(self, agents: Agents):
        h = FeatureHelpers(Nmax=5)
        x = torch.zeros((1, 3 * h.Nmax + 7))
        x[0, h.MAX_NUMBER_OF_AGENT] = 5
        x[0, h.ROAD_INDEX] = 0
        agents.time = 1
        x = agents.insert_agent_into_network(x, h)
        assert x[0, h.NUMBER_OF_AGENT] == 2
        assert torch.all(agents.agent_features[:2, agents.ON_WAY] == 1)
        x = agents.withdraw_agent_from_network(x, h)
        assert x[0, h.NUMBER_OF_AGENT] == 0
        assert torch.all(agents.agent_features[:2, agents.DONE] == 1)
