import torch
from src.feature_helpers import FeatureHelpers
from src.agents.base import Agents   

class TestAgent:

    def test_create_discrete_choice_model(self, agents: Agents):
        assert len(agents) == 9

    def test_config_agents(self, agents: Agents):
        agents.config_agents_from_xml("data/Siouxfalls_population.xml", "data/Siouxfalls_network_PT.xml")
        assert torch.all(agents.agent_features[0] == torch.tensor([218.0, 90.0, 62407.0, 0.0, 51.0, 1.0, 1.0, 0.0, 0.0]))
        agents.save("save/sioux_falls_agent_data.pt")


    def test_add_agent_into_network(self, sioux_falls, agents: Agents):
        agents.load("save/sioux_falls_agent_data.pt")

        # Check if the load is correct
        assert torch.all(agents.agent_features[0] == torch.tensor([218.0, 90.0, 62407.0, 0.0, 51.0, 1.0, 1.0, 0.0, 0.0]))
        h = FeatureHelpers(197)
        agents.insert_agent_into_network(sioux_falls.graph.x, h)
        agents.withdraw_agent_from_network(sioux_falls.graph.x, h)
