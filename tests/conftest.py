import pytest
import torch
from torch_geometric.data import Data

from src.agents.base import Agents
from src.direction_mpnn import DirectionMPNN
from src.response_mpnn import ResponseMPNN
from src.simulation_core_model import SimulationCoreModel
from src.transportation_simulator import TransportationSimulator
from src.feature_helpers import FeatureHelpers


@pytest.fixture
def device():
    return 'cpu'


@pytest.fixture
def agents(device):
    agent = Agents(device)
    # Create two simple agents. They leave from node 0 and go to node 1.
    # They should leave at 0 sec and arrive a minute later.
    agent.agent_features = torch.tensor([
        [0, 1, 0, 60, 30, 0, 1, 0, 0],
        [0, 1, 0, 60, 30, 1, 0, 0, 0],
    ])
    return agent


@pytest.fixture
def direction_mpnn():
    return DirectionMPNN()


@pytest.fixture
def response_mpnn():
    return ResponseMPNN()


@pytest.fixture
def core(device):
    return SimulationCoreModel(Nmax=2, device=device, time=0)


@pytest.fixture
def braess_graph():
    Nmax = 100
    feature_dim = 3 * Nmax + 7

    def build_node_feature(agent_pos, agent_t_arrival, agent_pos_at_arrival,
                           max_number_agent, number_agent, free_flow_time,
                           length_of_road, max_flow, selected_road, id_road):
        f = torch.zeros(feature_dim)
        f[0:Nmax] = agent_pos.clone().detach().float()
        f[Nmax:2*Nmax] = agent_t_arrival.clone().detach().float()
        f[2*Nmax:3*Nmax] = agent_pos_at_arrival.clone().detach().float()
        f[3*Nmax + 0] = max_number_agent
        f[3*Nmax + 1] = number_agent
        f[3*Nmax + 2] = free_flow_time
        f[3*Nmax + 3] = length_of_road
        f[3*Nmax + 4] = max_flow
        f[3*Nmax + 5] = selected_road
        f[3*Nmax + 6] = id_road
        return f

    x = torch.stack([
        build_node_feature(torch.zeros(Nmax), torch.zeros(Nmax), torch.zeros(Nmax), 2, 1, 3.0, 100.0, 10.0, 1, 0),
        build_node_feature(torch.zeros(Nmax), torch.zeros(Nmax), torch.zeros(Nmax), 2, 1, 1.0, 100.0, 10.0, 2, 1),
        build_node_feature(torch.zeros(Nmax), torch.zeros(Nmax), torch.zeros(Nmax), 2, 2, 1.0, 100.0, 10.0, 0, 2),
    ])

    x[0, 0] = 1.0
    x[1, 0] = 2.0
    x[2, 0] = 3.0
    x[2, 1] = 4.0
    x[2, 2 * Nmax + 1] = 1.0

    edge_index = torch.tensor([
        [0, 1, 2],
        [1, 2, 0],
    ], dtype=torch.long)
    edge_attr = torch.rand(edge_index.size(1), 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def simple_network_file(tmp_path):
    content = (
        '<network>'
        '  <links effectivecellsize="7.5">'
        '    <link id="0" from="A" to="B" length="100" capacity="10" freespeed="10" permlanes="1"/>'
        '    <link id="1" from="B" to="A" length="100" capacity="10" freespeed="10" permlanes="1"/>'
        '  </links>'
        '</network>'
    )
    file = tmp_path / "network.xml"
    file.write_text(content)
    return str(file)


@pytest.fixture
def simulator(device, simple_network_file):
    sim = TransportationSimulator(device)
    sim.config_network(simple_network_file)
    sim.agent.agent_features = torch.zeros((1, 9))
    sim.agent.agent_features[0, 0] = 0  # origin
    sim.agent.agent_features[0, 1] = 0  # destination
    sim.agent.agent_features[0, 2] = 0  # departure time
    sim.config_parameters(start_time=1)
    sim.agent.set_time(sim.time)
    return sim
